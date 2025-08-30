#!/usr/bin/env python3
# generator.py — Train a GPT-style Transformer (BPE, RoPE, SwiGLU) on GPU and export a single GGUF file.
# - Tokenizer is embedded into GGUF (tokens + merges), no external JSON artifacts.
# - Flask UI with SSE streaming (queue+thread), progress, s/it, env diagnostics, single-run lock.
# Author: Artur Strazewicz — 2025 — MIT

import argparse, json, math, os, time, struct, mmap, io, tempfile, hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from threading import Thread, Lock
from typing import Optional

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

def dataset_hash(path: Path, vocab_size: int) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    h.update(str(vocab_size).encode())
    return h.hexdigest()[:12]

def build_bpe_tokenizer_and_extract_tables(corpus_path: Path, vocab_size: int):
    """Train BPE once in-memory; extract tokens/merges by saving to a temp JSON (not persisted)."""
    text = corpus_path.read_text(encoding='utf-8')
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel()
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>","<bos>","<eos>","<unk>"])
    tok.train_from_iterator([text], trainer)

    # Save to a temp JSON to extract tokens/merges tables (we delete it immediately)
    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False, encoding='utf-8') as tf:
        tmp_path = tf.name
    try:
        tok.save(tmp_path)
        obj = json.loads(Path(tmp_path).read_text(encoding='utf-8'))
        # Expected HF tokenizers layout:
        # obj["model"]["vocab"] : dict token -> id
        # obj["model"]["merges"]: list of "A B" strings
        vocab = obj.get("model", {}).get("vocab", {})
        inv = {i:t for t,i in vocab.items()}
        max_id = max(inv) if inv else -1
        tokens = [inv.get(i, "") for i in range(max_id+1)]
        merges = obj.get("model", {}).get("merges", [])
        merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges]
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        # --- normalize tables so GGUF gets pure strings ---

    def _str_or_join(x):
        if isinstance(x, str): return x
        if isinstance(x, (list, tuple)):
            return " ".join(str(t) for t in x)
        return str(x)

    tokens = ["" if t is None else (t if isinstance(t, str) else str(t)) for t in tokens]
    merges = [_str_or_join(m) for m in merges]

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

# KV types (minimal set)
GGUF_TYPE_UINT32 = 2
GGUF_TYPE_FLOAT32 = 5
GGUF_TYPE_BOOL   = 8
GGUF_TYPE_STRING = 9
GGUF_TYPE_ARRAY  = 10

# Array element types
GGUF_ARRAY_UINT32  = 2
GGUF_ARRAY_FLOAT32 = 5
GGUF_ARRAY_STRING  = 9

# Tensor dtype codes (simplified)
T_DTYPE_MAP = { 'float32':0, 'float16':1, 'bfloat16':2, 'int8':3 }

def align_to(f, multiple=32):
    pos = f.tell()
    pad = (-pos) % multiple
    if pad: f.write(b'\x00'*pad)

class GGUFWriter:
    def __init__(self, path: str):
        self.path = path
        self.kv = []      # list of (key, (type, payload))
        self.tensors = [] # list of (name, np_array)
    # --- KV helpers ---
    def add_u32(self, key, v:int):      self.kv.append((key, (GGUF_TYPE_UINT32, v)))
    def add_f32(self, key, v:float):    self.kv.append((key, (GGUF_TYPE_FLOAT32, v)))
    def add_bool(self, key, v:bool):    self.kv.append((key, (GGUF_TYPE_BOOL, bool(v))))
    def add_str(self, key, s:str):      self.kv.append((key, (GGUF_TYPE_STRING, s)))

    def add_arr_str(self, key, arr: list):
        norm = []
        for x in arr:
            if isinstance(x, str):
                norm.append(x)
            elif isinstance(x, (list, tuple)):
                norm.append(" ".join(str(t) for t in x))
            else:
                norm.append(str(x))
        self.kv.append((key, (GGUF_TYPE_ARRAY, (GGUF_ARRAY_STRING, norm))))
    def add_arr_f32(self, key, arr:list[float]):
        self.kv.append((key, (GGUF_TYPE_ARRAY, (GGUF_ARRAY_FLOAT32, arr))))
    def add_arr_u32(self, key, arr:list[int]):
        self.kv.append((key, (GGUF_TYPE_ARRAY, (GGUF_ARRAY_UINT32, arr))))
    # --- tensors ---
    def add_tensor(self, name: str, np_arr: np.ndarray):
        assert np_arr.flags['C_CONTIGUOUS']
        self.tensors.append((name, np_arr))

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
                    raise ValueError("Unsupported array subtype")
            else:
                raise ValueError("Unsupported KV type")

    def _write_tensors_index(self, f):
        self._w_u64(f, len(self.tensors))
        # Precompute offsets (each tensor data aligned to 32)
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
            align_to(f, 32)
            # write data
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
               progress_cb=None):
    # --- device + env diag ---
    cuda_ok = torch.cuda.is_available()
    device = 'cuda' if cuda_ok else 'cpu'
    if progress_cb:
        diag = {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": cuda_ok,
            "cuda_device_count": torch.cuda.device_count() if cuda_ok else 0,
            "cudnn": getattr(torch.backends, 'cudnn', None) and torch.backends.cudnn.is_available(),
            "device": device,
        }
        if cuda_ok:
            try: diag["cuda_name0"] = torch.cuda.get_device_name(0)
            except Exception: pass
        progress_cb("ENV " + json.dumps(diag))

    os.makedirs(models_dir, exist_ok=True)
    corpus = Path(data_path)
    assert corpus.exists(), f"Missing {data_path}"

    # --- tokenizer (train once in-memory, no external JSON) ---
    if not HAVE_TOKENIZERS:
        raise RuntimeError("Install `tokenizers` package to use BPE: pip install tokenizers")
    tokenizer, tokens_tbl, merges_tbl = build_bpe_tokenizer_and_extract_tables(corpus, vocab_size)

    # --- make ids ---
    text = corpus.read_text(encoding='utf-8')
    ids = tokenizer.encode(text)
    ids = torch.tensor(ids, dtype=torch.long)

    # train/val split
    val_split = 0.01
    n = len(ids)
    train_len = max(0, int(n*(1.0-val_split)))
    train_ids, val_ids = ids[:train_len], ids[train_len:] if n - train_len > block_size+1 else ids[:]
    train_ds = TextDataset(train_ids.tolist(), block_size)
    val_ds   = TextDataset(val_ids.tolist(), block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # --- model ---
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=n_layer, n_head=n_head,
                    n_embd=n_embd, block_size=block_size, dropout=0.1, dtype=weight_dtype, use_rope=True)
    model = GPT(cfg).to(device)

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
                logits, vloss = model(xb, yb)
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

    # tokenizer KV (embedded, no external JSON)
    writer.add_str("tokenizer.ggml.model", "bpe")
    writer.add_arr_str("tokenizer.ggml.tokens", tokens_tbl)
    writer.add_arr_str("tokenizer.ggml.merges", merges_tbl)
    # special ids (align with our wrapper: <pad>=0, <bos>=1, <eos>=2)
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
# Flask UI (SSE via queue+thread, single-run lock)
# ============================================================
from flask import Flask, request, Response, render_template_string, stream_with_context

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>BPE Transformer Trainer</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:960px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:inline-block;margin:6px 12px 6px 0}
input,select{padding:8px;border:1px solid #ddd;border-radius:8px}
button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;cursor:pointer}
button[disabled]{opacity:.5;cursor:not-allowed}
.bar{height:8px;background:#ececff;border-radius:999px;overflow:hidden}
.bar i{display:block;height:100%;width:0%;background:#4f46e5}
.grid{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:8px;margin-top:8px}
.kv{background:#fafafe;border:1px solid #eee;border-radius:8px;padding:8px}
.kv b{display:block;font-size:12px;color:#666;margin-bottom:4px}
.kv span{font-variant-numeric:tabular-nums}
pre{white-space:pre-wrap}
.small{color:#666;font-size:12px;margin-top:6px}
footer{margin:0 auto;margin-bottom:2rem;font-size:.9em;color:#666}
footer div a{color:inherit}
.links a{margin-right:.75rem}
</style></head>
<body><div class="wrap">
  <h1>BPE Transformer Trainer (CUDA)</h1>
  <div class="card">
    <form id="f" onsubmit="start();return false;">
      <label>Dataset
        <select id="ds">
          {% for f in datasets %}<option value="{{f}}">{{f}}</option>{% endfor %}
        </select>
      </label>
      <label>D_model <input id="dmodel" type="number" value="256" min="64" max="1024"></label>
      <label>Heads <input id="heads" type="number" value="4" min="1" max="16"></label>
      <label>Layers <input id="layers" type="number" value="4" min="1" max="48"></label>
      <label>Seq <input id="seq" type="number" value="256" min="32" max="4096"></label>
      <label>Batch <input id="bs" type="number" value="64" min="1" max="512"></label>
      <label>Epochs <input id="ep" type="number" value="2" min="1" max="100"></label>
      <label>LR <input id="lr" step="0.0001" type="number" value="0.001"></label>
      <label><input id="amp" type="checkbox" checked> Use AMP (Tensor Cores)</label>
      <button id="btn">Start training</button>
    </form>
  </div>

  <div class="card">
    <h3>Прогресс</h3>
    <div class="bar"><i id="p"></i></div>
    <div class="grid">
      <div class="kv"><b>%</b><span id="s_pct">0.00%</span></div>
      <div class="kv"><b>Прошло</b><span id="s_elapsed">0.00m</span></div>
      <div class="kv"><b>ETA</b><span id="s_eta">—</span></div>
      <div class="kv"><b>Эпоха</b><span id="s_epoch">0 / 0</span></div>
      <div class="kv"><b>Итерации</b><span id="s_step">0 / 0</span></div>
      <div class="kv"><b>Loss</b><span id="s_loss">—</span></div>
      <div class="kv"><b>s/it</b><span id="s_spit">—</span></div>
    </div>
    <div class="small" id="s_device">device: —</div>
    <pre id="log"></pre>
  </div>
  <footer>
    <div><strong>PyAiModel TFormer</strong> — BPE, RoPE, SwiGLU, CUDA AMP, single-file GGUF export.</div>
    <div>© <span id="year">2025</span>. MIT.</div>
  </footer>
</div>
<script>
(function(){
  const $ = (id)=>document.getElementById(id);
  function val(id){ return $(id).value; }
  function parseProgress(line){
    const pct = (line.match(/Progress:\\s+([\\d.]+)%/) || [,''])[1];
    const epoch = (line.match(/epoch\\s+(\\d+)\\/(\\d+)/) || [,,])[1];
    const epochT = (line.match(/epoch\\s+(\\d+)\\/(\\d+)/) || [,,,''])[2];
    const step  = (line.match(/step\\s+(\\d+)\\/(\\d+)/) || [,,])[1];
    const stepT = (line.match(/step\\s+(\\d+)\\/(\\d+)/) || [,,,''])[2];
    const loss  = (line.match(/loss\\s+([\\d.]+)/) || [,''])[1];
    const spit  = (line.match(/s_it\\s+([\\d.]+s\\/it)/) || [,''])[1];
    const elapsed = (line.match(/elapsed\\s+([\\d.]+m)/) || [,''])[1];
    const eta     = (line.match(/ETA\\s+([\\d.]+m)/) || [,''])[1];
    if (pct) { $('p').style.width = pct + '%'; $('s_pct').textContent = pct + '%'; }
    if (epoch && epochT) $('s_epoch').textContent = epoch + ' / ' + epochT;
    if (step && stepT) $('s_step').textContent = step + ' / ' + stepT;
    if (loss) $('s_loss').textContent = loss;
    if (spit) $('s_spit').textContent = spit;
    if (elapsed) $('s_elapsed').textContent = elapsed;
    if (eta) $('s_eta').textContent = eta || '—';
  }
  let es = null;
  function setBusy(b){ $('btn').disabled = b; }
  window.start = function(){
    const sel = $('ds');
    if (!sel || !sel.value || sel.value.indexOf('.txt') === -1) {
      alert('Выбери dataset (.txt) в списке.');
      return;
    }
    setBusy(true);
    const params = new URLSearchParams({
      ds: sel.value,
      dmodel: val('dmodel'),
      heads:  val('heads'),
      layers: val('layers'),
      seq:    val('seq'),
      bs:     val('bs'),
      ep:     val('ep'),
      lr:     val('lr'),
      amp:    $('amp').checked ? '1':'0'
    });
    es  = new EventSource('/train?' + params.toString());
    const log = $('log');
    es.onmessage = function(e){
      const line = e.data || '';
      if (line.startsWith('ENV ')) {
        try {
          const env = JSON.parse(line.slice(4));
          const dev = env.device + (env.cuda_name0 ? ` (${env.cuda_name0})` : '');
          $('s_device').textContent = `device: ${dev} | AMP: ${env.cuda_available ? 'True' : 'False'}`;
        } catch (_){}
        return;
      }
      if (line.startsWith('Progress:')) parseProgress(line);
      if (line === 'DONE'){ es.close(); setBusy(false); return; }
      if (line.startsWith('ERR:busy')){ alert('Тренировка уже идёт'); es.close(); setBusy(false); return; }
      log.textContent += line + "\\n";
      log.scrollTop = log.scrollHeight;
    };
    es.onerror = function(){
      if (es){ es.close(); }
      setBusy(false);
      alert('Поток прерван. Смотри логи в терминале.');
    };
  };
})();
</script>
</body></html>"""

app = Flask(__name__)
TRAIN_LOCK = Lock()

@app.route("/")
def index():
    ds_dir = Path("Datasets")
    ds_dir.mkdir(exist_ok=True)
    datasets = sorted([p.name for p in ds_dir.glob("*.txt")])
    return render_template_string(HTML, datasets=datasets)

def _sse(line:str): return f"data: {line}\n\n"

@app.route("/train")
def train_route():
    # params
    ds = request.args.get("ds", "dataset.txt")
    dmodel = int(request.args.get("dmodel", 256))
    heads  = int(request.args.get("heads", 4))
    layers = int(request.args.get("layers", 4))
    seq    = int(request.args.get("seq", 256))
    bs     = int(request.args.get("bs", 64))
    ep     = int(request.args.get("ep", 2))
    lr     = float(request.args.get("lr", 0.001))
    amp    = request.args.get("amp", "1") == "1"

    data_path = str(Path("Datasets") / ds)
    models_dir = "Models"
    vocab_size = 32000

    # auto run_name (без внешних токенайзеров)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = Path(ds).stem
    run_name = f"{base}-d{dmodel}h{heads}l{layers}-seq{seq}-bs{bs}-{stamp}"

    # single-run lock
    if not TRAIN_LOCK.acquire(blocking=False):
        return Response(_sse("ERR:busy") + _sse("DONE"), mimetype="text/event-stream")

    q: Queue[str] = Queue()

    def progress_cb(msg: str): q.put(msg)

    def worker():
        try:
            saved_path = train_once(
                run_name=run_name,
                data_path=data_path,
                models_dir=models_dir,
                vocab_size=vocab_size,
                block_size=seq,
                n_layer=layers,
                n_head=heads,
                n_embd=dmodel,
                lr=lr,
                batch_size=bs,
                max_epochs=ep,
                weight_dtype='float16',
                amp_enabled=amp,
                progress_cb=progress_cb
            )
            q.put(f"Saved weights: {saved_path}")
        except Exception as e:
            q.put(f"ERROR: {type(e).__name__}: {e}")
        finally:
            q.put("DONE")
            TRAIN_LOCK.release()

    Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                msg = q.get(timeout=1.0)
                yield _sse(msg)
                if msg == "DONE": break
            except Empty:
                pass

    return Response(stream_with_context(stream()), mimetype="text/event-stream")

# ============================================================
# CLI
# ============================================================
def cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, default=None, help='If set, run a CLI training (no Flask UI)')
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
    ap.add_argument('--port', type=int, default=5000, help='Flask port (if UI)')
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
        os.makedirs("Models", exist_ok=True)
        os.makedirs("Datasets", exist_ok=True)
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    cli_main()
