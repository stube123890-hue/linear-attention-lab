"""Generate Colab notebook v3: parameter-equalized rematch."""
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

md("""# V4 RERUN — FFN-Only Equalization (fresh notebook, metrics baked in)

The full rematch equalized parameters by widening BOTH the state
(256→384) and the FFN — and the gap persisted (1.378 vs 1.415).
But that leaves a sharper question unanswered: is the bottleneck
the fixed-size STATE specifically, or just total capacity?

**This experiment:** the compressed state stays FROZEN at 256
(untouched), and the entire 1.05M parameter gap is closed by
aggressively widening ONLY the FFN: 1024 → **1279**.

New counts: attention **6,369,792** vs linear-scan **6,369,784**
(8 params apart — the closest mathematically possible).

**Reading it:**
- Gap persists → the fixed-size state is the bottleneck. Params
  parked in the FFN cannot compensate for state compression.
- Gap closes → it was total capacity all along; the state is fine.
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
n = int(0.95 * len(data)); train_data, val_data = data[:n], data[n:]
print(f"corpus {len(text)/1e6:.2f} MB, vocab {vocab}")

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

md("## 3. Build + verify parameter equality")
code("""def count_params(m):
    return sum(p.numel() for p in m.parameters())

DIM, LAYERS, HEADS = 256, 8, 8
attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h)).to(device)
linear = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: LinearRunningState(d, state_dim=256, mode="scan"),
    ffn_hidden=1279).to(device)
models = {"attention": attention, "linear-scan": linear}
for name, m in models.items():
    print(f"{name}: {count_params(m)} params")
pa, pl = count_params(attention), count_params(linear)
print(f"gap: {pa - pl} params ({100*(pa-pl)/pa:.4f}%)")
assert abs(pa - pl) < 100, "parameter counts are not equalized!"
print("EQUALITY GATE PASSED")
""")

md("## 4. The rematch — identical protocol to Exp 3")
code("""import math

STEPS, EVAL, LR = 1500, 250, 3e-4
opt = {n: torch.optim.AdamW(m.parameters(), lr=LR) for n, m in models.items()}

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
            msg += f" | {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})"
        print(msg, flush=True)
print("V4 DONE")
for name, m in models.items():
    vl = val_loss(m)
    print(f"FINAL {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
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

md("""## Reading the rematch

- **Gap persists (~0.03+)** → the STATE is the bottleneck, proven:
  a million extra FFN params cannot buy back what the fixed-size
  compression loses.
- **Gap closes to ~0** → total capacity was the story; the state
  dimension was never the problem.
""")

# final cell: dump everything to a text file as well as stdout
md("## 6. Results file (backup record)")
code("""
lines = []
lines.append("EQUALITY: attention=6369792 linear-scan=6369784")
import math
for name, m in models.items():
    vl = val_loss(m)
    lines.append(f"FINAL-VERIFY {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
open("/content/v4_rerun_results.txt", "w").write("\\n".join(lines))
print(open("/content/v4_rerun_results.txt").read())
print("RESULTS FILE WRITTEN")
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v4_rerun.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("v3 notebook written")
