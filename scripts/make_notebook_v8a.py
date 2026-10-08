"""Generate Colab notebook V8-A: diagonal gated delta-rule SSM."""
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

md("""# V8-A — Diagonal Gated Delta-Rule SSM

V7 (output gate + graded forget) took the crown at 1.339 with no quality
regression vs V6. V8-A tests the literature's highest-capacity-per-param
graft: the **Gated DeltaNet delta rule** (erase-before-write), restricted
to diagonal per-channel 1x1 systems so the 256-vector state and the fused
Triton scan survive unchanged.

Per channel:
- `alpha_t = sigmoid(W_a x + b_a + floor_l)` — reuse V7 gate_proj + floor
- `beta_t  = sigmoid(W_b x + b_b)`, `b_b = 0` (NEW write_proj)
- `k_t     = sigmoid(W_k x + b_k)`, `b_k = +2.0` (NEW key_proj)
- `v_t     = SiLU(B x)` — reuse in_proj, add SiLU
- `a_t = alpha_t * (1 - beta_t * k_t^2)`, `b_t = beta_t * k_t * v_t`
- `h_t = (1 - r_t) * (a_t .* h_{t-1} + b_t)`

Init starts near V7's convex-blend regime (k~=0.88, beta~=0.5); sigmoid
bounds keep every step strictly contractive — no explosion by construction.
V7's output gate + RMSNorm stay (the delta rule needs its stabilizers:
GDN's ablation shows naive delta integration is +3.52 ppl worse).

Budget: +2 projections/layer; FFN 766 keeps the diff <= 3000 (relaxed gate,
his explicit call). Challenger vs V7's 1.339.
""")

md("## 0. Setup (+ Triton)")
code("""import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "V8-A needs the T4 GPU"
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
    '"""(inlined) V6/V7/V8-A: Triton-fused scans.').replace(
    "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
    "import triton\nimport triton.language as tl\n")
code(models_src + "\n\n" + kern_src)

md("## 3. KERNEL GATE — Triton delta kernel vs reference, delta < 1e-6 (else STOP)")
code("""torch.manual_seed(0)
B, T, S = 4, 128, 256
u = torch.randn(B, T, S, device=device)
ap = torch.randn(B, T, S, device=device)
bp = torch.randn(B, T, S, device=device)
kp = torch.randn(B, T, S, device=device)
keep = (torch.rand(B, T, device=device) > 0.3).float()

# Triton fused delta kernel (compiles on first call)
h_tri = triton_delta_scan(u, ap, bp, kp, keep)

# Reference: delta (a,b) pair + Hillis-Steele scan (differentiable)
kr = keep.unsqueeze(-1)
alpha = torch.sigmoid(ap); beta = torch.sigmoid(bp); kk = torch.sigmoid(kp)
v = u * torch.sigmoid(u)
a = kr * alpha * (1 - beta * kk * kk)
b = kr * beta * kk * v
h_scan = parallel_scan_ab(a, b)

# Ground truth: direct sequential loop of the delta recurrence
h = torch.zeros(B, S, device=device); hs = []
for t in range(T):
    at = kr[:, t] * alpha[:, t] * (1 - beta[:, t] * kk[:, t] ** 2)
    bt = kr[:, t] * beta[:, t] * kk[:, t] * v[:, t]
    h = at * h + bt
    hs.append(h)
h_loop = torch.stack(hs, 1)

e_scan = float((h_tri - h_scan).abs().max())
e_loop = float((h_tri - h_loop).abs().max())
print(f"triton vs delta scan max delta: {e_scan:.2e}")
print(f"triton vs delta loop max delta: {e_loop:.2e}")

# backward: Triton grads vs autograd through the reference scan
u2 = u.clone().requires_grad_(True); ap2 = ap.clone().requires_grad_(True)
bp2 = bp.clone().requires_grad_(True); kp2 = kp.clone().requires_grad_(True)
triton_delta_scan(u2, ap2, bp2, kp2, keep).pow(2).sum().backward()
u3 = u.clone().requires_grad_(True); ap3 = ap.clone().requires_grad_(True)
bp3 = bp.clone().requires_grad_(True); kp3 = kp.clone().requires_grad_(True)
al3 = torch.sigmoid(ap3); be3 = torch.sigmoid(bp3); kk3 = torch.sigmoid(kp3)
v3 = u3 * torch.sigmoid(u3)
a3 = kr * al3 * (1 - be3 * kk3 * kk3)
b3 = kr * be3 * kk3 * v3
parallel_scan_ab(a3, b3).pow(2).sum().backward()
errs = {}
for name, g2, g3 in [("du", u2.grad, u3.grad), ("dap", ap2.grad, ap3.grad),
                     ("dbp", bp2.grad, bp3.grad), ("dkp", kp2.grad, kp3.grad)]:
    errs[name] = float((g2 - g3).abs().max())
print("backward max deltas:", {k: f"{v:.2e}" for k, v in errs.items()})
assert e_scan < 1e-6 and e_loop < 1e-6, "KERNEL GATE FAILED (forward)"
assert all(v < 1e-5 for v in errs.values()), "KERNEL GATE FAILED (backward)"
print("KERNEL GATE PASSED — fused delta kernel == reference math")
""")

md("## 4. Build + verify parameter budget (relaxed gate: diff <= 3000)")
code("""def count_params(m):
    return sum(p.numel() for p in m.parameters())

DIM, LAYERS, HEADS = 256, 8, 8
attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h)).to(device)
v8a = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
    ffn_hidden=766, newline_id=nl_id).to(device)
import math as _math
_floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
for _i, _blk in enumerate(v8a.blocks):
    _blk.mix.forget_floor.data.fill_(_floors[_i])
print("forget floors:", [f"{f:.2f}" for f in _floors.tolist()])
print("key_proj bias (expect 2.0):", float(v8a.blocks[0].mix.key_proj.bias[0]))
print("write_proj bias (expect 0.0):", float(v8a.blocks[0].mix.write_proj.bias[0]))
models = {"attention": attention, "v8a-delta": v8a}
for name, m in models.items():
    print(f"{name}: {count_params(m)} params")
pa, pv = count_params(attention), count_params(v8a)
print(f"gap: {pa - pv} params ({100*(pa-pv)/pa:.4f}%)")
print(f"v8a state dim frozen at: {v8a.blocks[0].mix.state_dim}")
assert abs(pa - pv) <= 3000, "budget not locked!"
print("BUDGET GATE PASSED (relaxed <= 3000)")
""")

md("## 5. Train — identical protocol to V7 (1500 steps)")
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
print("V8-A TRAINING DONE")

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

md("## 7. Kernel benchmark — mixer-level fwd+bwd (T=128/512/1024)")
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
    "v8a delta      ": lambda: SelectiveSegmentedStateV8A(256, 256),
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
lines = [f"EQUALITY: attention={pa} v8a={pv} diff={pa-pv}"]
import math
for name, vl in final_vals.items():
    lines.append(f"FINAL {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
open("/content/v8a_results.txt", "w").write("\\n".join(lines))
print(open("/content/v8a_results.txt").read())
print("RESULTS FILE WRITTEN")
""")

md("""## Reading V8-A

- **Kernel gate** must pass first: fused Triton delta kernel == reference
  math (< 1e-6 fwd, < 1e-5 bwd) — else STOP, do not train.
- **Budget gate:** diff <= 3000 (relaxed from 2000 by explicit call).
- **Quality:** challenger vs V7's 1.339. The delta rule's erase-before-write
  should beat convex blending on this newline-segmented char corpus —
  GDN's analysis says delta helps memorization, gating helps filtering.
- **Watch:** early training is more forgetful than the blend; if step-250
  val trails V7's trajectory badly, suspect the missing short conv
  (known -1.6 ppl gap), not the rule itself.
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v8a_delta.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("V8-A notebook written")
