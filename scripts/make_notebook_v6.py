"""Generate Colab notebook V5: gated segmented state-space model."""
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

md("""# V5 — Gated Segmented State Space Model

Breaking the capacity ceiling found in V4 (a stable ~2% perplexity tax
for the fixed-size linear state at 6.37M params).

**Architecture (state dim strictly frozen at 256 — VRAM edge kept):**

1. **Dynamic Selective Gate** — the model's "will":
   `g_t = sigmoid(W_g x_t + b_g)`, per token and per channel.
   Low-information tokens are filtered on the fly (g→0: forgotten fast),
   salient tokens are kept (g→1). The static decay is gone.

2. **Hard Memory Reset Mask** — segment boundaries:
   when the input token is a structural boundary (newline → paragraph/
   topic break), the state is multiplied by zero: `h_t = 0`.
   The past memory buffer is completely destroyed; the new segment
   starts fresh. (The residual stream still carries x_t forward.)

The recurrence stays affine (`h_t = a_t h_{t-1} + b_t`), so the parallel
associative scan still applies — O(T) work, log2(T) parallel passes.

**Budget:** attention 6,369,792 vs V5 6,368,760 (1,032 apart, 0.016%).
**Protocol:** identical to V4 — same 2.47MB corpus, 1500 steps, batch 32,
seq 128, LR 3e-4. The only change is the mixer.
""")

md("## 0. Setup")
code("""import torch
print("torch", torch.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
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
nl_frac = float((data == nl_id).sum()) / len(data)
print(f"boundary-token fraction: {nl_frac*100:.2f}% of tokens trigger reset")

SEQ, BS = 128, 32
def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    i = torch.randint(0, len(d) - SEQ - 1, (BS,))
    x = torch.stack([d[j:j+SEQ] for j in i]).to(device)
    y = torch.stack([d[j+1:j+SEQ+1] for j in i]).to(device)
    return x, y
""")

md("## 2. Models (inlined, no imports)")
code(open("/home/hatch/workspace/linear-attention-lab/models.py").read().replace(
    '"""Linear-recurrent replacement for Transformer self-attention.',
    '"""(inlined) Linear-recurrent replacement for Transformer self-attention.').replace(
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import math\nimport torch.nn as nn\nimport torch.nn.functional as F\n"))

md("## 3. Build + verify parameter budget")
code("""def count_params(m):
    return sum(p.numel() for p in m.parameters())

DIM, LAYERS, HEADS = 256, 8, 8
attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h)).to(device)
v5 = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: SelectiveSegmentedState(d, state_dim=256),
    ffn_hidden=1151, newline_id=nl_id).to(device)
models = {"attention": attention, "v5-segmented": v5}
for name, m in models.items():
    print(f"{name}: {count_params(m)} params")
pa, pv = count_params(attention), count_params(v5)
print(f"gap: {pa - pv} params ({100*(pa-pv)/pa:.4f}%)")
print(f"v5 state dim frozen at: {v5.blocks[0].mix.state_dim}")
assert abs(pa - pv) < 2000, "budget not locked!"
print("BUDGET GATE PASSED")
""")

md("## 4. Train — identical protocol to V4 (1500 steps)")
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
print("V5 TRAINING DONE")

# crossing analysis: who leads at each checkpoint?
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

md("## 5. Post-train metrics (runs immediately — same session, no idle gap)")
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

md("## 6. Results file (backup record — uses stored FINAL vals, no re-eval)")
code("""
lines = ["EQUALITY: attention=6369792 v5=6368760"]
import math
for name, vl in final_vals.items():
    lines.append(f"FINAL {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
open("/content/v5_results.txt", "w").write("\\n".join(lines))
print(open("/content/v5_results.txt").read())
print("RESULTS FILE WRITTEN")
""")

md("""## Reading V5

- **2% tax destroyed?** FINAL v5 val ≤ attention val → the selective gate
  + reset mask broke the ceiling.
- **Tax persists?** Compare the gap to V4's ~0.03: shrunk = progress,
  same = the bottleneck is deeper than gating.
- **Crossings:** does v5 still lead early? does attention still overtake?
- **Cost:** peak VRAM and tok/s vs the V4 scan (state still 256, but the
  gate adds per-step compute).
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v5_gated_segmented.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("V5 notebook written")
