#!/usr/bin/env python3
# generator.py — Train a GPT-style Transformer (BPE, RoPE, SwiGLU) on GPU and export GGUFx.
# + Flask UI with streaming progress (SSE via queue+thread).
# Author: Artur Strazewicz — 2025 — MIT

import argparse, json, math, os, time, struct, mmap, io
from pathlib import Path
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------------------
# Optional tokenizer deps (BPE)
# ----------------------------
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    HAVE_TOKENIZERS = True
except Exception:
    HAVE_TOKENIZERS = False

# ----------------------------
# GGUF-like writer/reader (GGUFx)
# ----------------------------
DTYPE_MAP = { 'float32':0, 'float16':1, 'bfloat16':2, 'int8':3 }
INV_DTYPE_MAP = {v:k for k,v in DTYPE_MAP.items()}
NP_DTYPE = {
    'float32': np.float32,
    'float16': np.float16,
    'bfloat16': np.uint16,  # store raw BF16 as u16
    'int8': np.int8,
}
def align64(x): return (x + 63) & ~63

class GGUFxWriter:
    def __init__(self, path:str, meta:dict):
        self.path = path
        self.meta = meta
        self.entries = []
    def add_tensor(self, name:str, arr:np.ndarray):
        assert arr.flags['C_CONTIGUOUS']
        dtype = str(arr.dtype)
        if dtype == 'uint16':  # our BF16 packing
            dtype = 'bfloat16'
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.entries.append({
            'name': name,
            'dtype': dtype,
            'shape': list(arr.shape),
            'nbytes': arr.nbytes,
            'data': arr,
        })
    def write(self):
        meta_bytes = json.dumps(self.meta, ensure_ascii=False).encode('utf-8')
        with open(self.path, 'wb') as f:
            f.write(b'GGUFx\x00')
            f.write(struct.pack('<I', 1))  # version
            f.write(struct.pack('<I', len(meta_bytes)))
            f.write(meta_bytes)
            f.write(struct.pack('<I', len(self.entries)))
            # index
            index = io.BytesIO()
            data_off = 0
            for e in self.entries:
                name_b = e['name'].encode('utf-8')
                index.write(struct.pack('<H', len(name_b)))
                index.write(name_b)
                index.write(struct.pack('<B', DTYPE_MAP[e['dtype']]))
                index.write(struct.pack('<B', len(e['shape'])))
                for d in e['shape']:
                    index.write(struct.pack('<I', d))
                index.write(struct.pack('<Q', data_off))
                data_off += e['nbytes']
            f.write(index.getvalue())
            pad = align64(f.tell()) - f.tell()
            if pad: f.write(b'\x00'*pad)
            # data
            for e in self.entries:
                f.write(e['data'].tobytes(order='C'))
        return self.path

class GGUFxReader:
    def __init__(self, path:str):
        self.path = path
        self.meta = {}
        self.tensors = {}
        self._load()
    def _load(self):
        with open(self.path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            assert mm.read(6) == b'GGUFx\x00'
            ver = struct.unpack('<I', mm.read(4))[0]; assert ver == 1
            meta_len = struct.unpack('<I', mm.read(4))[0]
            self.meta = json.loads(mm.read(meta_len).decode('utf-8'))
            n = struct.unpack('<I', mm.read(4))[0]
            entries = []
            for _ in range(n):
                name_len = struct.unpack('<H', mm.read(2))[0]
                name = mm.read(name_len).decode('utf-8')
                dt = INV_DTYPE_MAP[mm.read(1)[0]]
                ndim = mm.read(1)[0]
                shape = [struct.unpack('<I', mm.read(4))[0] for __ in range(ndim)]
                off = struct.unpack('<Q', mm.read(8))[0]
                entries.append((name, dt, shape, off))
            data_start = align64(mm.tell())
            for name, dt, shape, off in entries:
                npdt = NP_DTYPE[dt]
                byte_offset = data_start + off
                arr = np.memmap(self.path, dtype=npdt, mode='r', shape=tuple(shape), offset=byte_offset)
                self.tensors[name] = arr
            mm.close()

# ----------------------------
# Tokenizers
# ----------------------------
class SimpleCharTokenizer:
    def __init__(self):
        self.stoi = {"<pad>":0, "<bos>":1, "<eos>":2}
        self.itos = ["<pad>","<bos>","<eos>"]
    def train_from_text(self, text:str):
        for ch in sorted(set(text)):
            if ch not in self.stoi:
                self.stoi[ch] = len(self.itos)
                self.itos.append(ch)
    @property
    def vocab_size(self): return len(self.itos)
    def encode(self, s:str):
        ids = [self.stoi.get(ch,0) for ch in s]
        return [1] + ids + [2]
    def decode(self, ids):
        toks = []
        for i in ids:
            if i < len(self.itos):
                t = self.itos[i]
                if t not in ("<pad>","<bos>","<eos>"):
                    toks.append(t)
        return "".join(toks)
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type":"char","itos":self.itos}, f)
    @staticmethod
    def load(path):
        obj = json.load(open(path,"r",encoding="utf-8"))
        t = SimpleCharTokenizer()
        t.itos = obj["itos"]
        t.stoi = {s:i for i,s in enumerate(t.itos)}
        return t

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tok = tok
    @property
    def vocab_size(self): return self.tok.get_vocab_size()
    def encode(self, s): return [1] + self.tok.encode(s).ids + [2]
    def decode(self, ids):
        core = [i for i in ids if i not in (0,1,2)]
        return self.tok.decode(core)
    def save(self, path): self.tok.save(path)
    @staticmethod
    def load(path): return BPETokenizerWrapper(Tokenizer.from_file(path))

def build_tokenizer(corpus_path:Path, vocab_size:int, out_path:Path):
    text = corpus_path.read_text(encoding='utf-8')
    if HAVE_TOKENIZERS:
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>","<bos>","<eos>","<unk>"])
        tok.train_from_iterator([text], trainer)
        wrapper = BPETokenizerWrapper(tok)
        wrapper.save(str(out_path))
        return wrapper
    else:
        t = SimpleCharTokenizer()
        t.train_from_text(text)
        t.save(str(out_path))
        return t

# ----------------------------
# Dataset
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, text_ids, block_size):
        self.ids = text_ids
        self.block = block_size
    def __len__(self): return max(0, len(self.ids) - self.block - 1)
    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i+self.block], dtype=torch.long)
        y = torch.tensor(self.ids[i+1:i+self.block+1], dtype=torch.long)
        return x, y

# ----------------------------
# Model (RoPE + SwiGLU)
# ----------------------------
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
    def forward(self, x):
        return self.swiglu(x)

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

# ----------------------------
# Training
# ----------------------------
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
    cuda_ok = torch.cuda.is_available()
    device = 'cuda' if cuda_ok else (
        'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu')
    if device == 'mps':  # на всякий случай
        device = 'cuda' if cuda_ok else 'mps'
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
            diag["cuda_name0"] = torch.cuda.get_device_name(0)
        progress_cb("ENV " + json.dumps(diag))
    os.makedirs(models_dir, exist_ok=True)

    corpus = Path(data_path)
    assert corpus.exists(), f"Missing {data_path}"

    tok_path = Path(models_dir) / f"{run_name}.tokenizer.json"
    tokenizer = build_tokenizer(corpus, vocab_size, tok_path)

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

            step_time = time.time() - t0  # seconds per iteration

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

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_path = Path(models_dir) / f"{run_name}-{stamp}.ggufx"

    meta = {
        "format": "GGUFx",
        "version": 1,
        "model": {"arch":"gpt-decoder", "cfg": asdict(cfg)},
        "tokenizer_file": str(Path(tok_path).name),
        "created": stamp,
    }

    writer = GGUFxWriter(str(out_path), meta)
    for name, tensor in model.state_dict().items():
        arr = _to_np(tensor, cfg.dtype)
        writer.add_tensor(name, arr)
    saved = writer.write()

    if progress_cb:
        progress_cb(f"Saved weights: {saved}")
        progress_cb(f"Saved tokenizer: {tok_path}")
        progress_cb("DONE")

    return str(saved)

# ----------------------------
# Flask UI (SSE via queue+thread)
# ----------------------------
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
      <label>FF <input id="dff" type="number" value="1024" min="128" max="8192"></label>
      <label>Seq <input id="seq" type="number" value="256" min="32" max="4096"></label>
      <label>Batch <input id="bs" type="number" value="64" min="1" max="512"></label>
      <label>Epochs <input id="ep" type="number" value="2" min="1" max="100"></label>
      <label>LR <input id="lr" step="0.0001" type="number" value="0.001"></label>
      <label><input id="amp" type="checkbox" checked> Use AMP (Tensor Cores)</label>
      <button>Start training</button>
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
    <div><strong>PyAiModel TFormer</strong> — BPE, RoPE, SwiGLU, CUDA AMP, GGUFx export.</div>
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
  window.start = function(){
    const sel = $('ds');
    if (!sel || !sel.value || sel.value.indexOf('.txt') === -1) {
      alert('Выбери dataset (.txt) в списке.');
      return;
    }
    const params = new URLSearchParams({
      ds: sel.value,
      dmodel: val('dmodel'),
      heads:  val('heads'),
      layers: val('layers'),
      dff:    val('dff'),
      seq:    val('seq'),
      bs:     val('bs'),
      ep:     val('ep'),
      lr:     val('lr'),
      amp:    $('amp').checked ? '1':'0'
    });
    const es  = new EventSource('/train?' + params.toString());
    const log = $('log');
    es.onmessage = function(e){
      const line = e.data || '';
      if (line.startsWith('PCT:')){
        $('p').style.width = line.slice(4) + '%';
        $('s_pct').textContent = line.slice(4) + '%';
      } else {
        if (line.startsWith('Training started')){
          const mdev = line.match(/device=([^,\\)]+)/);
          const mamp = line.match(/AMP=(True|False)/);
          $('s_device').textContent = 'device: ' + (mdev ? mdev[1] : '—') + ' | AMP: ' + (mamp ? mamp[1] : '—');
        }
        if (line.startsWith('ENV ')) {
  try {
    const env = JSON.parse(line.slice(4));
    const dev = env.device + (env.cuda_name0 ? ` (${env.cuda_name0})` : '');
    $('s_device').textContent = `device: ${dev} | AMP: ${('cuda_available' in env) ? (env.cuda_available ? 'True' : 'False') : '—'}`;
  } catch (e) {}
}
        if (line.startsWith('Progress:')) parseProgress(line);
        log.textContent += line + "\\n";
        log.scrollTop = log.scrollHeight;
      }
      if (line === 'DONE') es.close();
    };
    es.onerror = function(){
      es.close();
      alert('Поток прерван. Смотри Network/Console и логи в терминале Flask.');
    };
  };
})();
</script>
</body></html>
"""

from flask import Flask, request, Response, render_template_string, stream_with_context
app = Flask(__name__)

@app.route("/")
def index():
    ds_dir = Path("Datasets")
    ds_dir.mkdir(exist_ok=True)
    datasets = sorted([p.name for p in ds_dir.glob("*.txt")])
    return render_template_string(HTML, datasets=datasets)

def _sse_yield(line:str):  # one SSE message
    return f"data: {line}\n\n"

@app.route("/train")
def train_route():
    # params
    ds = request.args.get("ds", "dataset.txt")
    dmodel = int(request.args.get("dmodel", 256))
    heads  = int(request.args.get("heads", 4))
    layers = int(request.args.get("layers", 4))
    dff    = int(request.args.get("dff", 1024))  # not used directly with SwiGLU sizing; kept for UI symmetry
    seq    = int(request.args.get("seq", 256))
    bs     = int(request.args.get("bs", 64))
    ep     = int(request.args.get("ep", 2))
    lr     = float(request.args.get("lr", 0.001))
    amp    = request.args.get("amp", "1") == "1"

    data_path = str(Path("Datasets") / ds)
    models_dir = "Models"
    vocab_size = 32000

    # auto run_name (без поля Out name)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = Path(ds).stem
    run_name = f"{base}-d{dmodel}h{heads}l{layers}-seq{seq}-bs{bs}-{stamp}"

    q: Queue[str] = Queue()

    def progress_cb(msg: str):
        q.put(msg)

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
            q.put(f"Model saved: {saved_path}")
        except Exception as e:
            q.put(f"ERROR: {type(e).__name__}: {e}")
        finally:
            q.put("DONE")

    Thread(target=worker, daemon=True).start()

    def stream():
        #yield _sse_yield(f"Training started (device={'cuda' if torch.cuda.is_available() else 'cpu'}, AMP={amp})")
        while True:
            try:
                msg = q.get(timeout=1.0)
                yield _sse_yield(msg)
                if msg == "DONE":
                    break
            except Empty:
                pass

    return Response(stream_with_context(stream()), mimetype="text/event-stream")

# ----------------------------
# CLI entry
# ----------------------------
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
