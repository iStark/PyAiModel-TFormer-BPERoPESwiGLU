# PyAiModel-TFormer-BPERoPESwiGLU — generator.py
# Training a GPT-style decoder (BPE, RoPE, SwiGLU) on GPU/CPU with export to a single GGUF file.
# Includes Flask-based UI with SSE progress streaming, AMP (Tensor Cores) support, and monitoring.
#
# Developed by: Artur Strazewicz — concept, architecture, model training, GGUF export, UI.
# Year: 2025. License: MIT.
#
# Features:
#   • BPE (byte-level) tokenizer, fully embedded into GGUF (tokens + merges, special IDs).
#   • Architecture: pre-LN Transformer, Rotary Positional Embeddings (RoPE), SwiGLU MLP, tied output head.
#   • CUDA/CPU support, device auto-detection, AMP training, progress metrics (loss, s/it, ETA).
#   • Single self-contained .gguf model file (no external JSON artifacts).
#
# Links:
#   GitHub:      https://github.com/iStark/PyAiModel-TFormer-BPERoPESwiGLU
#   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
#   TruthSocial: https://truthsocial.com/@strazewicz
#   X (Twitter): https://x.com/strazewicz


import argparse, json, math, os, time, struct, tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# BPE tokenizer (HF tokenizers) — no external JSON persisted
# ============================================================
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    HAVE_TOKENIZERS = True
except Exception:
    HAVE_TOKENIZERS = False

class BPETokenizerWrapper:
    def __init__(self, tok: "Tokenizer"):
        self.tok = tok
    @property
    def vocab_size(self): return self.tok.get_vocab_size()
    def encode(self, s: str): return [1] + self.tok.encode(s).ids + [2]   # <bos> ... <eos>
    def decode(self, ids):
        core = [i for i in ids if i not in (0,1,2)]
        return self.tok.decode(core)

def build_bpe_tokenizer_and_extract_tables(corpus_path: Path, vocab_size: int) -> Tuple[BPETokenizerWrapper, List[str], List[str]]:
    """Train BPE once in-memory; extract tokens/merges by saving to a temp JSON (not persisted)."""
    if not HAVE_TOKENIZERS:
        raise RuntimeError("Install `tokenizers` package to use BPE: pip install tokenizers")
    text = corpus_path.read_text(encoding='utf-8')
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel()
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>","<bos>","<eos>","<unk>"])
    tok.train_from_iterator([text], trainer)

    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False, encoding='utf-8') as tf:
        tmp_path = tf.name
    try:
        tok.save(tmp_path)
        obj = json.loads(Path(tmp_path).read_text(encoding='utf-8'))
        vocab = obj.get("model", {}).get("vocab", {})
        inv = {i:t for t,i in vocab.items()}
        max_id = max(inv) if inv else -1
        tokens = [inv.get(i, "") for i in range(max_id+1)]
        merges = obj.get("model", {}).get("merges", [])
        merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges]
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

    # normalize tables (pure strings)
    def _norm(x):
        if isinstance(x, str): return x
        if isinstance(x, (list, tuple)): return " ".join(str(t) for t in x)
        return str(x)
    tokens = ["" if t is None else (t if isinstance(t,str) else str(t)) for t in tokens]
    merges = [_norm(m) for m in merges]
    return BPETokenizerWrapper(tok), tokens, merges

# ============================================================
# Dataset
# ============================================================
class TextDataset(Dataset):
    def __init__(self, text_ids, block_size):
        self.ids = text_ids
        self.block = block_size
    def __len__(self): return max(0, len(self.ids) - self.block - 1)
    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i+self.block], dtype=torch.long)
        y = torch.tensor(self.ids[i+1:i+self.block+1], dtype=torch.long)
        return x, y

# ============================================================
# Model (RoPE + SwiGLU, pre-LN, tied head)
# ============================================================
@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 512
    dropout: float = 0.1
    bias: bool = False
    dtype: str = "float16"
    use_rope: bool = True

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)  # up
        self.proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.proj(F.silu(self.w1(x)) * self.w2(x)))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(4 * cfg.n_embd * 2 / 3)  # 2/3 rule for SwiGLU sizing
        self.swiglu = SwiGLU(cfg.n_embd, hidden, bias=cfg.bias, dropout=cfg.dropout)
    def forward(self, x): return self.swiglu(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def get_cos_sin(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin
    def apply_rotary(self, x, cos, sin):
        x1 = x[..., ::2]; x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (x_rot * sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.register_buffer("bias", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1,1,cfg.block_size,cfg.block_size))
        self.dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.rope = RotaryEmbedding(self.head_dim) if getattr(cfg, 'use_rope', False) else None
    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        if self.rope is not None:
            cos, sin = self.rope.get_cos_sin(T, x.device)
            q = self.rope.apply_rotary(q, cos, sin)
            k = self.rope.apply_rotary(k, cos, sin)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # tied weights
        self.lm_head.weight = self.tok_emb.weight
    def forward(self, idx, targets=None):
        B,T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx)
        if not getattr(self.cfg, 'use_rope', False):
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ============================================================
# GGUF v3 writer (weights + tokenizer kv)
# ============================================================
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 2
GGUF_TYPE_FLOAT32 = 5
GGUF_TYPE_BOOL   = 8
GGUF_TYPE_STRING = 9
GGUF_TYPE_ARRAY  = 10
GGUF_ARRAY_UINT32  = 2
GGUF_ARRAY_FLOAT32 = 5
GGUF_ARRAY_STRING  = 9
T_DTYPE_MAP = { 'float32':0, 'float16':1, 'bfloat16':2, 'int8':3 }

def _align_to(f, multiple=32):
    pos = f.tell()
    pad = (-pos) % multiple
    if pad: f.write(b'\x00'*pad)

class GGUFWriter:
    def __init__(self, path: str):
        self.path = path
        self.kv = []      # list[(key, (type, payload))]
        self.tensors = [] # list[(name, np_array)]
    # KV helpers
    def add_u32(self, key, v:int):      self.kv.append((key, (GGUF_TYPE_UINT32, v)))
    def add_f32(self, key, v:float):    self.kv.append((key, (GGUF_TYPE_FLOAT32, v)))
    def add_bool(self, key, v:bool):    self.kv.append((key, (GGUF_TYPE_BOOL, bool(v))))
    def add_str(self, key, s:str):      self.kv.append((key, (GGUF_TYPE_STRING, s)))
    def add_arr_str(self, key, arr: list):
        norm = []
        for x in arr:
            if isinstance(x, str): norm.append(x)
            elif isinstance(x, (list, tuple)): norm.append(" ".join(str(t) for t in x))
            else: norm.append(str(x))
        self.kv.append((key, (GGUF_TYPE_ARRAY, (GGUF_ARRAY_STRING, norm))))
    # tensors
    def add_tensor(self, name: str, np_arr: np.ndarray):
        assert np_arr.flags['C_CONTIGUOUS']
        self.tensors.append((name, np_arr))
    # writers
    def _w_u32(self, f, v): f.write(struct.pack('<I', v))
    def _w_u64(self, f, v): f.write(struct.pack('<Q', v))
    def _w_f32(self, f, v): f.write(struct.pack('<f', v))
    def _w_bool(self, f, v): f.write(struct.pack('<?', bool(v)))
    def _w_str(self, f, s:str):
        b = s.encode('utf-8'); self._w_u64(f, len(b)); f.write(b)
    def _write_kv(self, f):
        self._w_u64(f, len(self.kv))
        for key, (tp, payload) in self.kv:
            self._w_str(f, key)
            self._w_u32(f, tp)
            if tp == GGUF_TYPE_UINT32:
                self._w_u32(f, int(payload))
            elif tp == GGUF_TYPE_FLOAT32:
                self._w_f32(f, float(payload))
            elif tp == GGUF_TYPE_BOOL:
                self._w_bool(f, bool(payload))
            elif tp == GGUF_TYPE_STRING:
                self._w_str(f, payload)
            elif tp == GGUF_TYPE_ARRAY:
                subtype, arr = payload
                self._w_u32(f, subtype)
                self._w_u64(f, len(arr))
                if subtype == GGUF_ARRAY_STRING:
                    for s in arr: self._w_str(f, s)
                elif subtype == GGUF_ARRAY_FLOAT32:
                    for x in arr: self._w_f32(f, float(x))
                elif subtype == GGUF_ARRAY_UINT32:
                    for x in arr: self._w_u32(f, int(x))
            else:
                raise ValueError("Unsupported KV type")
    def _write_tensors_index(self, f):
        self._w_u64(f, len(self.tensors))
        offset = 0
        headers = []
        for name, arr in self.tensors:
            n_dims = arr.ndim
            dims = list(arr.shape)[::-1]  # reverse order in GGUF
            dtype_code = T_DTYPE_MAP[str(arr.dtype)]
            nbytes = arr.nbytes
            headers.append((name, n_dims, dims, dtype_code, offset, nbytes))
            offset += nbytes + ((-nbytes) % 32)
        for name, n_dims, dims, dtype_code, off, nbytes in headers:
            self._w_str(f, name)
            self._w_u32(f, n_dims)
            for d in dims: self._w_u64(f, d)
            self._w_u32(f, dtype_code)
            self._w_u64(f, off)
            self._w_u64(f, nbytes)
        return headers
    def write(self):
        with open(self.path, 'wb') as f:
            f.write(GGUF_MAGIC)
            self._w_u32(f, GGUF_VERSION)
            self._write_kv(f)
            headers = self._write_tensors_index(f)
            _align_to(f, 32)
            for (_, arr), (_,_,_,_,_, nbytes) in zip(self.tensors, headers):
                f.write(arr.tobytes(order='C'))
                pad = (-nbytes) % 32
                if pad: f.write(b'\x00'*pad)
        return self.path

# ============================================================
# Training
# ============================================================
def _to_np(t: torch.Tensor, dtype: str):
    if dtype == 'float16':
        return t.detach().half().contiguous().cpu().numpy()
    elif dtype == 'bfloat16':
        return t.detach().to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy()
    elif dtype == 'int8':
        return t.detach().to(torch.int8).contiguous().cpu().numpy()
    else:
        return t.detach().float().contiguous().cpu().numpy()

def train_once(run_name:str, data_path:str, models_dir:str,
               vocab_size:int, block_size:int,
               n_layer:int, n_head:int, n_embd:int,
               lr:float, batch_size:int, max_epochs:int,
               weight_dtype:str='float16', amp_enabled:bool=True,
               progress_cb=None) -> str:
    # --- device + env diag ---
    cuda_ok = torch.cuda.is_available()
    device = 'cuda' if cuda_ok else 'cpu'
    if progress_cb:
        progress_cb(json.dumps({
            "device": device,
            "cuda_available": cuda_ok,
            "cuda_device_count": torch.cuda.device_count() if cuda_ok else 0,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
        }))

    os.makedirs(models_dir, exist_ok=True)
    corpus = Path(data_path)
    assert corpus.exists(), f"Missing {data_path}"

    # tokenizer
    tokenizer, tokens_tbl, merges_tbl = build_bpe_tokenizer_and_extract_tables(corpus, vocab_size)

    # ids
    text = corpus.read_text(encoding='utf-8')
    ids = tokenizer.encode(text)
    ids = torch.tensor(ids, dtype=torch.long)

    # split
    val_split = 0.01
    n = len(ids)
    train_len = max(0, int(n*(1.0-val_split)))
    train_ids, val_ids = ids[:train_len], ids[train_len:] if n - train_len > block_size+1 else ids[:]
    train_ds = TextDataset(train_ids.tolist(), block_size)
    val_ds   = TextDataset(val_ids.tolist(), block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # model
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=n_layer, n_head=n_head,
                    n_embd=n_embd, block_size=block_size, dropout=0.1,
                    dtype=weight_dtype, use_rope=True)
    model = GPT(cfg).to(device)

    # param count
    param_count = sum(p.numel() for p in model.parameters())
    if progress_cb:
        progress_cb(f"PARAMS {param_count}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda' and amp_enabled))

    total_steps = max_epochs * max(1, len(train_loader))
    started = time.time()
    if progress_cb:
        progress_cb(f"Training started (device={device}, AMP={scaler.is_enabled()})")

    global_step = 0
    best_val = float('inf')
    model.train()
    for epoch in range(max_epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{max_epochs}")
        step_in_epoch = 0
        for xb, yb in pbar:
            t0 = time.time()
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            step_time = time.time() - t0
            global_step += 1
            step_in_epoch += 1
            pct = 100.0 * global_step / max(1, total_steps)
            elapsed_m = (time.time() - started) / 60.0
            eta_m = (elapsed_m / max(1e-9, pct/100.0)) - elapsed_m if pct > 0 else 0.0
            line = (f"Progress:  {pct:.2f}% | epoch {epoch+1}/{max_epochs} | "
                    f"step {step_in_epoch}/{len(train_loader)} | loss {loss.item():.4f} | "
                    f"s_it {step_time:.2f}s/it | elapsed {elapsed_m:.2f}m | ETA {eta_m:.2f}m")
            if progress_cb: progress_cb(line)

        # validation
        model.eval()
        with torch.no_grad():
            vlosses = []
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                _, vloss = model(xb, yb)
                vlosses.append(vloss.item())
        model.train()
        if vlosses:
            v = float(np.mean(vlosses))
            if progress_cb: progress_cb(f"Validation loss: {v:.4f}")
            if v < best_val: best_val = v

    # --- write single GGUF (weights + tokenizer)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_path = Path(models_dir) / f"{run_name}-{stamp}.gguf"
    writer = GGUFWriter(str(out_path))

    # general/model KV
    writer.add_str ("general.architecture", "gpt-neox")
    writer.add_u32 ("vocab_size", cfg.vocab_size)
    writer.add_u32 ("context_length", cfg.block_size)
    writer.add_u32 ("embedding_length", cfg.n_embd)
    writer.add_u32 ("block_count", cfg.n_layer)
    writer.add_u32 ("attention.head_count", cfg.n_head)
    writer.add_bool("rope.enabled", bool(cfg.use_rope))
    # parameter count (string + possible u32)
    writer.add_str("general.parameter_count", str(param_count))
    try:
        writer.add_u32("general.parameter_count_u32", int(param_count))
    except Exception:
        pass

    # tokenizer KV (embedded, no external JSON)
    writer.add_str("tokenizer.ggml.model", "bpe")
    writer.add_arr_str("tokenizer.ggml.tokens", tokens_tbl)
    writer.add_arr_str("tokenizer.ggml.merges", merges_tbl)
    writer.add_u32("vocab.special_pad_id", 0)
    writer.add_u32("vocab.special_bos_id", 1)
    writer.add_u32("vocab.special_eos_id", 2)

    # tensors
    for name, tensor in model.state_dict().items():
        arr = _to_np(tensor, cfg.dtype)
        writer.add_tensor(name, arr)

    saved = writer.write()
    if progress_cb:
        progress_cb(f"Saved weights: {saved}")
        progress_cb("DONE")
    return str(saved)

# ============================================================
# CLI (no web UI)
# ============================================================
def cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, default=None, help='Run name for CLI training (no web UI)')
    ap.add_argument('--data', type=str, default='Datasets/dataset.txt')
    ap.add_argument('--models', type=str, default='Models')
    ap.add_argument('--vocab', type=int, default=32000)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--embd', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--dtype', type=str, default='float16', choices=['float32','float16','bfloat16','int8'])
    args = ap.parse_args()

    if args.run:
        def _print(msg): print(msg, flush=True)
        train_once(run_name=args.run, data_path=args.data, models_dir=args.models,
                   vocab_size=args.vocab, block_size=args.block,
                   n_layer=args.layers, n_head=args.heads, n_embd=args.embd,
                   lr=args.lr, batch_size=args.batch, max_epochs=args.epochs,
                   weight_dtype=args.dtype, amp_enabled=True,
                   progress_cb=_print)
    else:
        print("Nothing to do. Use --run to train via CLI or start the web UI from generator_frontend.py")

if __name__ == "__main__":
    cli_main()
