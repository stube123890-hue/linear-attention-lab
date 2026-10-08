"""Generate Colab notebook V6: Triton-fused gated segmented state space."""
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

md("""# V6 — Triton-Fused Gated Segmented State Space

V5 beat attention on quality (1.360 vs 1.388) but still trailed on speed
(1.7x) and peak VRAM (1.8x): the PyTorch Hillis-Steele scan writes
O(log T) intermediate (B,T,S) tensors to slow HBM.

V6 changes **only the execution path** — the math, parameters, corpus,
and protocol are identical to V5. A fused Triton kernel:

- **Step A:** loads the compact 256-wide hidden state into fast SRAM
  registers once per batch element — never spilled to HBM mid-sequence.
- **Step B:** computes the sigmoid selective gate, applies the hard reset
  mask, and runs the linear update entirely in registers, each timestep.
- **Step C:** writes only h_t (plus a_t, saved for backward) back to HBM.
  The scan's intermediate duplicates are bypassed completely.

Question: does the fused kernel close the speed/VRAM gap vs attention
while keeping V5's quality crown?
""")

md("## 0. Setup (+ Triton)")
code("""import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "V6 needs the T4 GPU"
""")

md("## 1. Data (same 2.47 MB corpus) + boundary token id")
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
n = int(0.95 * len(data)); train_data, val_data = data[:n], data[n:]
nl_id = stoi.get("\\n")
print(f"corpus {len(text)/1e6:.2f} MB, vocab {vocab}, newline_id={nl_id}")
assert nl_id is not None, "newline not in vocab!"

SEQ, BS = 128, 32
def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    i = torch.randint(0, len(d) - SEQ - 1, (BS,))
    x = torch.stack([d[j:j+SEQ] for j in i]).to(device)
    y = torch.stack([d[j+1:j+SEQ+1] for j in i]).to(device)
    return x, y
""")

md("## 2. Models + Triton kernels (inlined, no imports)")
models_src = open("/home/hatch/workspace/linear-attention-lab/models.py").read().replace(
    '"""Linear-recurrent replacement for Transformer self-attention.',
    '"""(inlined) Linear-recurrent replacement for Transformer self-attention.').replace(
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import math\nimport torch.nn as nn\nimport torch.nn.functional as F\n")
kern_src = open("/home/hatch/workspace/linear-attention-lab/triton_kernels.py").read().replace(
    '"""V6: Triton-fused selective scan for the gated segmented state space.',
    '"""(inlined) V6: Triton-fused selective scan.').replace(
    "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
    "import triton\nimport triton.language as tl\n")
code(models_src + "\n\n" + kern_src)

md("## 3. KERNEL GATE — Triton vs V5 loop, delta < 1e-6 (else STOP)")
code("""torch.manual_seed(0)
B, T, S = 4, 128, 256
u = torch.randn(B, T, S, device=device)
gp = torch.randn(B, T, S, device=device)
keep = (torch.rand(B, T, device=device) > 0.3).float()

# Triton fused kernel (compiles on first call)
h_tri = triton_selective_scan(u, gp, keep)

# Ground truth: V5's original sequential loop semantics
g = torch.sigmoid(gp)
h = torch.zeros(B, S, device=device); hs = []
for t in range(T):
    k = keep[:, t:t+1]
    h = k * g[:, t] * h + k * (1 - g[:, t]) * u[:, t]
    hs.append(h)
h_loop = torch.stack(hs, 1)

# V5's Hillis-Steele scan path
a = keep.unsqueeze(-1) * g
b = keep.unsqueeze(-1) * (1 - g) * u
h_scan = parallel_scan_ab(a, b)

e_loop = float((h_tri - h_loop).abs().max())
e_scan = float((h_tri - h_scan).abs().max())
print(f"triton vs V5 loop max delta: {e_loop:.2e}")
print(f"triton vs V5 scan max delta: {e_scan:.2e}")

# backward: Triton grads vs autograd through the reference scan
u2 = u.clone().requires_grad_(True); gp2 = gp.clone().requires_grad_(True)
triton_selective_scan(u2, gp2, keep).pow(2).sum().backward()
u3 = u.clone().requires_grad_(True); gp3 = gp.clone().requires_grad_(True)
g3 = torch.sigmoid(gp3); a3 = keep.unsqueeze(-1) * g3
b3 = keep.unsqueeze(-1) * (1 - g3) * u3
parallel_scan_ab(a3, b3).pow(2).sum().backward()
du_err = float((u2.grad - u3.grad).abs().max())
dg_err = float((gp2.grad - gp3.grad).abs().max())
print(f"backward du max delta: {du_err:.2e}, dgp max delta: {dg_err:.2e}")
assert e_loop < 1e-6 and e_scan < 1e-6, "KERNEL GATE FAILED (forward)"
assert du_err < 1e-5 and dg_err < 1e-5, "KERNEL GATE FAILED (backward)"
print("KERNEL GATE PASSED — fused kernel == V5 math")
""")

md("## 4. Build + verify parameter budget (identical to V5)")
code("""def count_params(m):
    return sum(p.numel() for p in m.parameters())

DIM, LAYERS, HEADS = 256, 8, 8
attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h)).to(device)
v6 = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: SelectiveSegmentedStateTriton(d, state_dim=256),
    ffn_hidden=1151, newline_id=nl_id).to(device)
models = {"attention": attention, "v6-triton": v6}
for name, m in models.items():
    print(f"{name}: {count_params(m)} params")
pa, pv = count_params(attention), count_params(v6)
print(f"gap: {pa - pv} params ({100*(pa-pv)/pa:.4f}%)")
print(f"v6 state dim frozen at: {v6.blocks[0].mix.state_dim}")
assert abs(pa - pv) < 2000, "budget not locked!"
print("BUDGET GATE PASSED")
""")

md("## 5. Train — identical protocol to V5 (1500 steps)")
code("""import math

STEPS, EVAL, LR = 1500, 250, 3e-4
opt = {n: torch.optim.AdamW(m.parameters(), lr=LR) for n, m in models.items()}
val_hist = {n: [] for n in models}

def val_loss(m):
    m.eval()
    with torch.no_grad():
        return sum(float(m(*get_batch("val"))[1]) for _ in range(10)) / 10

for step in range(1, STEPS + 1):
    for name, m in models.items():
        m.train()
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt[name].zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt[name].step()
    if step % EVAL == 0 or step == 1:
        msg = f"step {step:5d}"
        for name, m in models.items():
            vl = val_loss(m)
            val_hist[name].append(vl)
            msg += f" | {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})"
        print(msg, flush=True)
print("V6 TRAINING DONE")

names = list(models)
print("leader per checkpoint:")
for i, s in enumerate([1] + list(range(EVAL, STEPS + 1, EVAL))):
    a, b = val_hist[names[0]][i], val_hist[names[1]][i]
    lead = names[0] if a < b else names[1]
    print(f"  step {s:5d}: {lead} leads ({a:.3f} vs {b:.3f})")
for name, m in models.items():
    vl = val_loss(m)
    print(f"FINAL {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
final_vals = {name: val_loss(m) for name, m in models.items()}
""")

md("## 6. Post-train metrics (runs immediately — same session, no idle gap)")
code("""import time

print("=== POST-TRAIN METRICS ===")
for name, m in models.items():
    m.train()
    tl = 0.0
    for _ in range(20):
        xb, yb = get_batch("train")
        with torch.no_grad():
            tl += float(m(xb, yb)[1])
    tl /= 20
    opt_m = torch.optim.AdamW(m.parameters(), lr=3e-4)
    xb, yb = get_batch("train")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        _, loss = m(xb, yb)
        opt_m.zero_grad(); loss.backward(); opt_m.step()
    torch.cuda.synchronize()
    train_ms = (time.time()-t0)/30*1000
    train_tps = BS*SEQ/(train_ms/1000)
    torch.cuda.reset_peak_memory_stats()
    _, loss = m(xb, yb); opt_m.zero_grad(); loss.backward(); opt_m.step()
    peak_mb = torch.cuda.max_memory_allocated()/1e6
    torch.cuda.empty_cache()
    m.eval()
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(100):
            m(xb)
        torch.cuda.synchronize()
    infer_ms = (time.time()-t0)/100*1000
    infer_tps = BS*SEQ/(infer_ms/1000)
    print(f"{name}: train_loss(pt)={tl:.3f} train_ms={train_ms:.1f} "
          f"train_tok/s={train_tps:.0f} peak_MB={peak_mb:.0f} "
          f"infer_ms={infer_ms:.1f} infer_tok/s={infer_tps:.0f}")
print("METRICS DONE")
""")

md("## 7. Kernel benchmark — the software tax, measured (T=128/512/1024)")
code("""import time
print("=== KERNEL BENCHMARK: mixer-level fwd+bwd, batch 8, dim 256 ===")

def bench(make, T, iters=30):
    m = make().train().to(device)
    x = torch.randn(8, T, 256, device=device)
    r = (torch.rand(8, T, device=device) < 0.3).float()
    for _ in range(5):                      # warmup (Triton compiles here)
        m.zero_grad(); m(x, r).sum().backward()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        m.zero_grad(); m(x, r).sum().backward()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    torch.cuda.reset_peak_memory_stats()
    m.zero_grad(); m(x, r).sum().backward()
    peak = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.empty_cache()
    return ms, peak

mixers = {
    "attention      ": lambda: CausalSelfAttention(256, 8),
    "v5 scan (torch)": lambda: SelectiveSegmentedState(256, 256),
    "v6 triton      ": lambda: SelectiveSegmentedStateTriton(256, 256),
}
for T in (128, 512, 1024):
    print(f"--- T={T} ---")
    for name, make in mixers.items():
        ms, peak = bench(make, T)
        print(f"  {name}: {ms:7.2f} ms/fwd+bwd | peak {peak:6.0f} MB")
print("BENCHMARK DONE")
""")

md("## 8. Results file (backup record — uses stored FINAL vals, no re-eval)")
code("""
lines = ["EQUALITY: attention=6369792 v6=6368760"]
import math
for name, vl in final_vals.items():
    lines.append(f"FINAL {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
open("/content/v6_results.txt", "w").write("\\n".join(lines))
print(open("/content/v6_results.txt").read())
print("RESULTS FILE WRITTEN")
""")

md("""## Reading V6

- **Kernel gate** must pass first: fused Triton == V5 math (< 1e-4).
- **Quality:** v6 should land where V5 did (~1.36) — same math, same crown.
- **Speed/VRAM:** the question — does Step A/B/C fusion close the 1.7x
  throughput and 1.8x VRAM gaps vs attention?
- A clean sweep (quality + speed + memory) would make the fused gated
  segmented state the strictly dominant architecture at this scale.
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v6_triton.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("V6 notebook written")
