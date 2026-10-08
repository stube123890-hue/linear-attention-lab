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

md("""# Parameter-Equalized Rematch

Exp 3 showed attention winning narrowly at ~6M params (1.450 vs 1.484)
— but the linear model had 16% fewer parameters (5.32M vs 6.37M).
Was that a capacity gap or a parameter-count gap?

**The equalization:** the linear-scan model gets
- state_dim 256 → **384** (1.5x wider running state)
- FFN hidden 1024 → **1151**

New counts: attention **6,369,792** vs linear-scan **6,369,784**
(8 params apart — the closest mathematically possible).

**Protocol:** identical to Exp 3 in every other respect — same corpus,
same 1500 steps, same batches, same LR. The *only* change is the
linear model's dimensions. If it ties or wins now, the capacity
argument is decoupled from the parameter-count argument.
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
    mixer_fn=lambda d, h: LinearRunningState(d, state_dim=384, mode="scan"),
    ffn_hidden=1151).to(device)
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
print("REMATCH DONE")
for name, m in models.items():
    vl = val_loss(m)
    print(f"FINAL {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})")
""")

md("""## Reading the rematch

- **Linear ties or wins** → the Exp-3 gap was a parameter-count artifact.
  Capacity argument decoupled. The architecture is validated at 6.37M.
- **Attention still wins by a similar margin** → the gap is architectural:
  the fixed-size state genuinely bottlenecks representation at this
  scale, and the next question is how the gap scales (wider state?
  hybrid attention+state layers?).
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v3_rematch.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("v3 notebook written")
