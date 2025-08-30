#!/usr/bin/env python3
# chat.py — Flask UI for interactive generation with our GPT (BPE, RoPE, SwiGLU) and GGUFx weights.
# Author: Artur Strazewicz — 2025 — MIT

import os, io, json, math, struct, mmap, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, Response, render_template_string, stream_with_context

# ===========================
# GGUFx reader (mmap)
# ===========================
DTYPE_MAP = { 'float32':0, 'float16':1, 'bfloat16':2, 'int8':3 }
INV_DTYPE_MAP = {v:k for k,v in DTYPE_MAP.items()}
NP_DTYPE = {
    'float32': np.float32,
    'float16': np.float16,
    'bfloat16': np.uint16,  # packed raw bf16
    'int8': np.int8,
}
def align64(x): return (x + 63) & ~63

class GGUFxReader:
    def __init__(self, path:str):
        self.path = path
        self.meta = {}
        self.entries = []  # (name, dtype, shape, off)
        self.data_start = 0
        self._load()
    def _load(self):
        with open(self.path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            assert mm.read(6) == b'GGUFx\x00'
            ver = struct.unpack('<I', mm.read(4))[0]; assert ver == 1
            meta_len = struct.unpack('<I', mm.read(4))[0]
            self.meta = json.loads(mm.read(meta_len).decode('utf-8'))
            n = struct.unpack('<I', mm.read(4))[0]
            entries=[]
            for _ in range(n):
                name_len = struct.unpack('<H', mm.read(2))[0]
                name = mm.read(name_len).decode('utf-8')
                dt = INV_DTYPE_MAP[mm.read(1)[0]]
                ndim = mm.read(1)[0]
                shape = [struct.unpack('<I', mm.read(4))[0] for __ in range(ndim)]
                off = struct.unpack('<Q', mm.read(8))[0]
                entries.append((name, dt, shape, off))
            self.data_start = align64(mm.tell())
            self.entries = entries
            mm.close()
    def load_tensor(self, name, dtype, shape, off):
        return np.memmap(self.path, dtype=NP_DTYPE[dtype], mode='r', shape=tuple(shape),
                         offset=self.data_start + off)

# ===========================
# Tokenizer wrapper
# ===========================
try:
    from tokenizers import Tokenizer
    HAVE_TOKENIZERS = True
except Exception:
    HAVE_TOKENIZERS = False

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tok = tok
    @property
    def vocab_size(self): return self.tok.get_vocab_size()
    def encode(self, s): return [1] + self.tok.encode(s).ids + [2]   # <bos> ... <eos>
    def decode(self, ids):
        core = [i for i in ids if i not in (0,1,2)]
        return self.tok.decode(core)
    @staticmethod
    def load(path): return BPETokenizerWrapper(Tokenizer.from_file(path))

# ===========================
# Model (same arch as in generator)
# ===========================
@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float = 0.0
    bias: bool = False
    dtype: str = "float16"
    use_rope: bool = True

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False, dropout=0.0):
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
        hidden = int(4 * cfg.n_embd * 2 / 3)
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
        # tied weights
        self.lm_head.weight = self.tok_emb.weight
    @torch.no_grad()
    def forward(self, idx):
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
        return logits

# ===========================
# Weight loading
# ===========================
def _from_np(arr: np.ndarray, want_dtype: torch.dtype, on_cuda: bool):
    # handle bf16 packing
    if arr.dtype == np.uint16:  # our bf16 stored as u16
        t = torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    else:
        # map numpy dtype → torch dtype
        if arr.dtype == np.float16: t = torch.from_numpy(arr.astype(np.float16, copy=False)).to(torch.float16)
        elif arr.dtype == np.float32: t = torch.from_numpy(arr.astype(np.float32, copy=False)).to(torch.float32)
        elif arr.dtype == np.int8: t = torch.from_numpy(arr.astype(np.int8, copy=False)).to(torch.int8)
        else: t = torch.from_numpy(np.array(arr, copy=False))
    t = t.to(want_dtype if want_dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32)
    if on_cuda: t = t.cuda(non_blocking=True)
    return t

def load_model_and_tokenizer(model_path: str, prefer_cuda: bool=True):
    rdr = GGUFxReader(model_path)
    meta = rdr.meta
    cfgd = meta["model"]["cfg"]
    cfg = GPTConfig(
        vocab_size = cfgd["vocab_size"],
        n_layer = cfgd["n_layer"],
        n_head  = cfgd["n_head"],
        n_embd  = cfgd["n_embd"],
        block_size = cfgd["block_size"],
        dropout = 0.0,
        bias = cfgd.get("bias", False),
        dtype = cfgd.get("dtype", "float16"),
        use_rope = cfgd.get("use_rope", True),
    )

    device = 'cuda' if (prefer_cuda and torch.cuda.is_available()) else 'cpu'
    want_dtype = torch.float16 if (device=='cuda' and cfg.dtype=='float16') else (torch.bfloat16 if cfg.dtype=='bfloat16' else torch.float32)

    model = GPT(cfg).to(device)
    model.eval()

    # build index
    name_to_entry = { name:(dtype,shape,off) for (name,dtype,shape,off) in rdr.entries }
    sd = model.state_dict()
    loaded = {}
    for name, param in sd.items():
        if name not in name_to_entry:
            # allow missing (e.g. pos_emb when RoPE)
            if name == "pos_emb.weight" and cfg.use_rope:
                continue
            raise KeyError(f"Tensor '{name}' not found in GGUFx file")
        dtype, shape, off = name_to_entry[name]
        arr = rdr.load_tensor(name, dtype, shape, off)
        t = _from_np(arr, want_dtype=torch.float32 if param.dtype==torch.float32 else want_dtype, on_cuda=(device=='cuda'))
        if t.shape != param.shape:
            raise ValueError(f"Shape mismatch for {name}: file {tuple(t.shape)} vs model {tuple(param.shape)}")
        loaded[name] = t

    # load into model
    with torch.no_grad():
        for name, t in loaded.items():
            sd[name].copy_(t, non_blocking=True)

    # tokenizer path (relative to model dir)
    model_dir = Path(model_path).parent
    tok_file = meta.get("tokenizer_file")
    if not tok_file:
        # fallback: try any tokenizer nearby
        toks = list(model_dir.glob("*.tokenizer.json"))
        if not toks:
            raise FileNotFoundError("Tokenizer json not found next to model")
        tok_path = toks[0]
    else:
        tok_path = model_dir / tok_file
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer file missing: {tok_path}")

    if not HAVE_TOKENIZERS:
        raise RuntimeError("The 'tokenizers' package is required for chat. Install: pip install tokenizers")

    tokenizer = BPETokenizerWrapper.load(str(tok_path))
    return model, tokenizer, device

# ===========================
# Sampling utils
# ===========================
def top_k_logits(logits: torch.Tensor, k: int):
    if k <= 0:
        return logits
    v, ix = torch.topk(logits, k)
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(1, ix, v)
    return out

@torch.no_grad()
def generate_stream(model: GPT, tokenizer: BPETokenizerWrapper, device: str,
                    prompt: str, max_new_tokens: int = 200, temperature: float = 0.9, top_k: int = 40):
    # encode
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]  # (1,T)

    sent_text = ""  # last sent to client

    # greedily stream token by token
    for _ in range(max_new_tokens):
        # crop to block size
        x_cond = x if x.size(1) <= model.cfg.block_size else x[:, -model.cfg.block_size:]
        logits = model(x_cond)  # (1, T, vocab)
        logits = logits[:, -1, :]  # (1, vocab)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id], dim=1)

        nid = int(next_id.item())
        if nid == 2:  # <eos>
            break

        # decode full and send only delta (simple but effective)
        full = tokenizer.decode(x[0].tolist())
        delta = full[len(sent_text):]
        if delta:
            sent_text = full
            yield delta

# ===========================
# Flask app / UI
# ===========================
HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Transformer Chat</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#f7f7fb;margin:0}
.wrap{max-width:960px;margin:0 auto;padding:16px;display:flex;flex-direction:column;height:100vh}
.card{background:#fff;border:1px solid #eee;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
label{display:inline-block;margin:6px 12px 6px 0}
input,select{padding:8px;border:1px solid #ddd;border-radius:8px}
button{padding:10px 14px;border-radius:10px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;cursor:pointer}
.row{display:flex;gap:8px;align-items:center}
.grow{flex:1}
.chat{flex:1;overflow-y:auto;background:#fff;border:1px solid #eee;border-radius:12px;padding:12px}
.msg{background:#fafafe;border:1px solid #eee;border-radius:10px;padding:10px 12px;margin:8px 0}
.msg.ai{border-left:4px solid #4f46e5}
.msg.user{border-left:4px solid #9ca3af}
.small{color:#666;font-size:12px;margin-left:4px}
:root {--mut:#556;}
        footer{
    margin: 0 auto;
    margin-bottom: 2rem;
    font-size: .9em;font-size:.9em;color:var(--mut)}
        footer div a{color:inherit}
        .links a{margin-right:.75rem}
</style></head>
<body><div class="wrap">
  <div class="card">
    <div class="row">
      <label>Model
        <select id="model">
          {% for m in models %}<option value="{{m}}">{{m}}</option>{% endfor %}
        </select>
      </label>
      <label>Max new <input id="max_new" type="number" value="200" min="1" max="2048"></label>
      <label>Temp <input id="temp" type="number" step="0.1" value="0.9" min="0.1" max="2.0"></label>
      <label>Top-k <input id="topk" type="number" value="40" min="0" max="500"></label>
      <span class="small" id="dev">device: —</span>
    </div>
  </div>

  <div class="chat" id="chat"></div>

  <div class="card">
    <div class="row">
      <textarea id="ta" class="grow" placeholder="Введите сообщение..." rows="4"></textarea>
      <button id="send">Send</button>
    </div>
  </div>
  <footer>

        <div><strong>PyAiModel-TFormer-BPERoPESwiGLU</strong> —BPE, RoPE, SwiGLU, CUDA AMP, GGUFx export.</div>
        <div>© <span id="year">2025</span>. Developed by <strong>Artur Strazewicz</strong> — concept, architecture, PyAiModel-TFormer-BPERoPESwiGLU, UI,  <strong>MIT license</strong>.</div>
        <div class="links">
            <a href="https://github.com/iStark/PyAiModel-TFormer-BPERoPESwiGLU" target="_blank" rel="noopener">GitHub</a>
            <a href="https://www.linkedin.com/in/arthur-stark/" target="_blank" rel="noopener">LinkedIn</a>
            <a href="https://truthsocial.com/@strazewicz" target="_blank" rel="noopener">Truth Social</a>
            <a href="https://x.com/strazewicz" target="_blank" rel="noopener">X (Twitter)</a>
        </div>
    </footer>
</div>

<script>
(function(){
  const $ = (id)=>document.getElementById(id);
  const chat = $('chat');

  function add(role, text){
    const el = document.createElement('div');
    el.className = 'msg ' + (role==='ai'?'ai':'user');
    el.textContent = text;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
  }

  async function startGen(prompt){
    const params = new URLSearchParams({
      model: $('model').value,
      max_new: $('max_new').value,
      temperature: $('temp').value,
      top_k: $('topk').value,
      prompt: prompt
    });
    const es = new EventSource('/gen?'+params.toString());
    let aiEl = add('ai', '');
    es.onmessage = (e)=>{
      const s = e.data || '';
      if (s.startsWith('DEV:')){
        $('dev').textContent = 'device: ' + s.slice(4);
        return;
      }
      if (s.startsWith('TOK:')){
        aiEl.textContent += s.slice(4);
        chat.scrollTop = chat.scrollHeight;
        return;
      }
      if (s === 'DONE'){ es.close(); return; }
      if (s.startsWith('ERR:')){ aiEl.textContent += '\\n['+s.slice(4)+']'; es.close(); }
    };
    es.onerror = ()=>{ es.close(); };
  }

  $('send').onclick = ()=>{
    const t = $('ta').value.trim();
    if(!t) return;
    add('user', t);
    $('ta').value = '';
    startGen(t);
  };
  $('ta').addEventListener('keydown', (e)=>{
    if(e.key==='Enter' && (e.ctrlKey||e.metaKey)){ $('send').click(); }
  });
})();
</script>

</body></html>"""

app = Flask(__name__)

# cache model per filename to avoid reloading each time
_MODEL_CACHE = {}

@app.route("/")
def index():
    models_dir = Path("Models")
    models_dir.mkdir(exist_ok=True)
    models = [p.name for p in sorted(models_dir.glob("*.ggufx"))] + [p.name for p in sorted(models_dir.glob("*.gguf"))]
    return render_template_string(HTML, models=models)

def _sse(line:str): return f"data: {line}\n\n"

@app.route("/gen")
def gen_route():
    try:
        model_file = request.args.get("model", "")
        prompt = request.args.get("prompt", "")
        max_new = int(request.args.get("max_new", 200))
        temperature = float(request.args.get("temperature", 0.9))
        top_k = int(request.args.get("top_k", 40))

        if not model_file:
            return Response(_sse("ERR:select model") + _sse("DONE"), mimetype="text/event-stream")

        model_path = str(Path("Models") / model_file)

        def stream():
            # load (cache)
            if model_path not in _MODEL_CACHE:
                model, tokenizer, device = load_model_and_tokenizer(model_path, prefer_cuda=True)
                _MODEL_CACHE[model_path] = (model, tokenizer, device)
            else:
                model, tokenizer, device = _MODEL_CACHE[model_path]

            devline = device
            if device == 'cuda':
                try:
                    devline = f"cuda ({torch.cuda.get_device_name(0)})"
                except Exception:
                    devline = "cuda"
            yield _sse("DEV:" + devline)

            # generate and stream tokens
            try:
                for piece in generate_stream(model, tokenizer, device, prompt, max_new_tokens=max_new,
                                             temperature=temperature, top_k=top_k):
                    yield _sse("TOK:" + piece)
            except Exception as e:
                yield _sse("ERR:" + f"{type(e).__name__}: {e}")
            yield _sse("DONE")

        return Response(stream_with_context(stream()), mimetype="text/event-stream")
    except Exception as e:
        return Response(_sse("ERR:" + f"{type(e).__name__}: {e}") + _sse("DONE"), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
