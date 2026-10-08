"""Generate Colab notebook v2: three follow-up experiments."""
import json

NB = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"}},
      "cells": []}

def md(src):
    NB["cells"].append({"cell_type": "markdown", "metadata": {},
                        "source": src.splitlines(keepends=True)})

def code(src):
    NB["cells"].append({"cell_type": "code", "metadata": {},
                        "source": src.splitlines(keepends=True),
                        "outputs": [], "execution_count": None})

md("""# Linear Attention Replacement — Experiments 1, 2, 3

Follow-up to the first run (linear running state beat attention on val
loss 1.608 vs 1.634 with fewer params, but the naive Python loop was
~10x slower per batch: 109.7ms vs 11.2ms).

- **Exp 1 — the quadratic killer:** scale context T = 128 → 256 → 512 → 1024.
  Attention is O(T²), the linear state is O(T). Watch the gap explode.
- **Exp 2 — parallel scan speed:** loop vs parallel-scan vs attention,
  milliseconds per batch on GPU. The scan should erase the 10x slowdown.
- **Exp 3 — scale up:** ~6M-param models (dim 256, 8 layers). Does the
  linear compression hold up on harder linguistic structure?
""")

md("## 0. Setup")
code("""import torch
print("torch", torch.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
""")

md("## 1. Data (same 2.47 MB corpus)")
code("""import os, urllib.request
os.makedirs("data", exist_ok=True)
URLS = {
    "tinyshakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "alice.txt":           "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "frankenstein.txt":    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "pride.txt":           "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
}
parts = []
for name, url in URLS.items():
    p = f"data/{name}"
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    with open(p, encoding="utf-8", errors="ignore") as f:
        parts.append(f.read())
text = "\\n".join(parts)
chars = sorted(set(text)); vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.95 * len(data)); train_data = data[:n]
print(f"corpus {len(text)/1e6:.2f} MB, vocab {vocab}")

def get_batch(seq_len, batch):
    i = torch.randint(0, len(train_data) - seq_len - 1, (batch,))
    x = torch.stack([train_data[j:j+seq_len] for j in i]).to(device)
    y = torch.stack([train_data[j+1:j+seq_len+1] for j in i]).to(device)
    return x, y
""")

md("""## 2. Models — attention, linear-loop, linear-scan (all inlined, no imports)""")
code(open("/home/hatch/workspace/linear-attention-lab/models.py").read().replace(
    '"""Linear-recurrent replacement for Transformer self-attention.',
    '"""(inlined) Linear-recurrent replacement for Transformer self-attention.').replace(
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import math\nimport torch.nn as nn\nimport torch.nn.functional as F\n"))

code("""# extend the builder: mode="loop" or "scan" for the linear model
def build_all(vocab, dim=128, n_layers=4, n_heads=4, seq_len=128, dropout=0.0):
    attn = TinyLM(vocab, dim, n_layers, n_heads, seq_len,
                  mixer_fn=lambda d, h: CausalSelfAttention(d, h, dropout))
    lin_loop = TinyLM(vocab, dim, n_layers, n_heads, seq_len,
                  mixer_fn=lambda d, h: LinearRunningState(d, None, dropout, mode="loop"))
    lin_scan = TinyLM(vocab, dim, n_layers, n_heads, seq_len,
                  mixer_fn=lambda d, h: LinearRunningState(d, None, dropout, mode="scan"))
    return {"attention": attn, "linear-loop": lin_loop, "linear-scan": lin_scan}

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# correctness gate: scan must equal loop before we trust any timing
torch.manual_seed(0)
_chk = build_all(vocab, dim=64, n_layers=1, n_heads=2, seq_len=64)
_chk = {k: v.to(device) for k, v in _chk.items()}
_chk["linear-loop"].load_state_dict(_chk["linear-scan"].state_dict())
_chk["linear-loop"].eval(); _chk["linear-scan"].eval()
_xb, _yb = get_batch(64, 4)
with torch.no_grad():
    _a = _chk["linear-loop"](_xb)[0]; _b = _chk["linear-scan"](_xb)[0]
print("scan-vs-loop max diff:", float((_a - _b).abs().max()))
del _chk
""")

md("""## Experiment 1 — Context scaling: the quadratic killer

T = 128 → 256 → 512 → 1024. For each: wall time per batch (default
backend, what people actually run) and peak CUDA memory with the math
backend forced (exposes attention's true O(T²) footprint — production
flash attention saves memory but NOT the quadratic FLOPs).
""")
code("""import time

def bench_time(model, seq_len, batch, iters=20):
    model.train()
    xb, yb = get_batch(seq_len, batch)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(iters):
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize() if device == "cuda" else None
    return (time.time() - t0) / iters * 1000

def bench_mem(model, seq_len, batch):
    if device != "cuda":
        return float("nan")
    model.train()
    xb, yb = get_batch(seq_len, batch)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False,
                                            enable_mem_efficient=False,
                                            enable_math=True):
            _, loss = model(xb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.cuda.max_memory_allocated() / 1e6
    except RuntimeError as e:
        return float("nan") if "out of memory" in str(e).lower() else (_ for _ in ()).throw(e)
    finally:
        torch.cuda.empty_cache()

print(f"{'T':>6} | {'model':<12} | {'ms/batch':>9} | {'peak MB (math bwd)':>18}")
print("-" * 55)
exp1 = {}
for T in [128, 256, 512, 1024]:
    models = build_all(vocab, dim=128, n_layers=2, n_heads=4, seq_len=T)
    exp1[T] = {}
    for name, m in models.items():
        if name == "linear-loop" and T >= 512:
            print(f"{T:>6} | {name:<12} | {'skipped (too slow)':>9} |")
            continue
        m = m.to(device)
        ms = bench_time(m, T, batch=8)
        mb = bench_mem(m, T, batch=8)
        exp1[T][name] = (ms, mb)
        print(f"{T:>6} | {name:<12} | {ms:>9.1f} | {mb:>18.0f}")
    del models
print("EXP1 DONE")
""")

md("""## Experiment 2 — Loop vs parallel scan vs attention (pure speed)

Milliseconds per train batch (forward + backward) on the GPU.
If the scan works, linear-scan should beat attention outright —
erasing the 109.7ms vs 11.2ms deficit from run 1.
""")
code("""print(f"{'T':>6} | {'model':<12} | {'ms/batch':>9} |")
print("-" * 34)
for T, bs in [(128, 32), (512, 8)]:
    models = build_all(vocab, dim=128, n_layers=4, n_heads=4, seq_len=T)
    for name, m in models.items():
        m = m.to(device)
        ms = bench_time(m, T, batch=bs, iters=30 if T == 128 else 15)
        print(f"{T:>6} | {name:<12} | {ms:>9.1f} |")
    del models
print("EXP2 DONE")
""")

md("""## Experiment 3 — Scale up to ~6M params

dim=256, 8 layers, 8 heads → ~6.3M params each. Train attention vs
linear-**scan** for 1500 steps and compare final val loss: does the
fixed-size compression hold up on harder structure?
""")
code("""import math

DIM, LAYERS, HEADS, SEQ, BS, STEPS, EVAL = 256, 8, 8, 128, 32, 1500, 250
models = {
    "attention": build_all(vocab, DIM, LAYERS, HEADS, SEQ)["attention"].to(device),
    "linear-scan": build_all(vocab, DIM, LAYERS, HEADS, SEQ)["linear-scan"].to(device),
}
for name, m in models.items():
    print(f"{name}: {count_params(m)/1e6:.2f}M params")
opt = {n: torch.optim.AdamW(m.parameters(), lr=3e-4) for n, m in models.items()}

def val_loss(m):
    m.eval()
    with torch.no_grad():
        return sum(float(m(*get_batch(SEQ, BS))[1]) for _ in range(10)) / 10

for step in range(1, STEPS + 1):
    for name, m in models.items():
        m.train()
        xb, yb = get_batch(SEQ, BS)
        _, loss = m(xb, yb)
        opt[name].zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt[name].step()
    if step % EVAL == 0 or step == 1:
        msg = f"step {step:5d}"
        for name, m in models.items():
            vl = val_loss(m)
            msg += f" | {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})"
        print(msg, flush=True)
print("EXP3 DONE")
for name, m in models.items():
    vl = val_loss(m)
    print(f"FINAL {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
""")

md("""## How to read the three experiments

- **Exp 1:** attention's ms/batch and peak MB should grow ~4x per doubling
  of T (quadratic); linear-scan should grow ~2x (linear). At T=1024 the
  ratio is the whole argument.
- **Exp 2:** linear-scan ms/batch vs attention ms/batch — the moment the
  10x deficit flips into a win.
- **Exp 3:** final val losses at ~6M params — does the linear state still
  match/beat attention when the modeling gets harder?
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v2_colab.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("v2 notebook written")
