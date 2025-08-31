# PyAiModel-TFormer-BPERoPESwiGLU — generator_frontend.py
# Web frontend (Flask template / static UI) for launching training and monitoring progress. 
# Provides a parameter form, dataset list, and live progress via SSE.
#
# Developed by: Artur Strazewicz — concept, architecture, UX design, backend integration.
# Year: 2025. License: MIT.
#
# Features:
#   • Training form: Dataset, D_model, Heads, Layers, Seq, Batch, Epochs, LR, AMP.
#   • Status panel: progress %, epochs/steps, loss, s/it, elapsed time, ETA, device info.
#   • Safe handling of concurrent runs, error reporting, and connection recovery.
#   • Clean, lightweight interface with plain HTML/CSS/JS (no heavy dependencies).
#
# Links:
#   GitHub:      https://github.com/iStark/PyAiModel-TFormer-BPERoPESwiGLU
#   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
#   TruthSocial: https://truthsocial.com/@strazewicz
#   X (Twitter): https://x.com/strazewicz

import time, os
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Lock

from flask import Flask, request, Response, render_template_string, stream_with_context

from generator import train_once  # <<< логика импортируется отсюда

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

def _sse(line:str) -> str:
    return f"data: {line}\n\n"

@app.route("/")
def index():
    ds_dir = Path("Datasets")
    ds_dir.mkdir(exist_ok=True)
    datasets = sorted([p.name for p in ds_dir.glob("*.txt")])
    return render_template_string(HTML, datasets=datasets)

@app.route("/train")
def train_route():
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

    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = Path(ds).stem
    run_name = f"{base}-d{dmodel}h{heads}l{layers}-seq{seq}-bs{bs}-{stamp}"

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

if __name__ == "__main__":
    os.makedirs("Models", exist_ok=True)
    os.makedirs("Datasets", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
