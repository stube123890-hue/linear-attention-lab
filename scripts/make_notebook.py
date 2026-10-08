"""Generate the Colab notebook: linear-recurrent attention replacement experiment."""
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
                        "source": src.splitlines(keepends=True) if isinstance(src, str) else src,
                        "outputs": [], "execution_count": None})

# ---------------------------------------------------------------- 1. intro
md("""# Replacing Transformer Attention with a Linear Running State

**Question:** what happens if we take a standard decoder-only Transformer,
delete the self-attention layer entirely, and replace it with a *moving
linear equation* that compresses all past tokens into a **fixed-size
running state**?

```
h_t = d * h_{t-1} + (1 - d) * (B x_t)      <- linear recurrence
y_t = C h_t                                 <- readout
```

- `d` in (0,1): learned per-channel decay (how fast the past fades)
- `B`, `C`: learned projections in/out of the state
- `h`: **fixed size** — it never grows with context length

Complexity per layer: attention is **O(T²)** time / **O(T)** memory,
the running state is **O(T)** time / **O(1)** memory.

**Experiment plan:** train two *identical* tiny LMs on 2–10 MB of text —
one with causal self-attention (baseline), one with the linear running
state — then compare loss, perplexity, speed, and generated samples.
""")

# ---------------------------------------------------------------- 2. setup
md("## 1. Setup — GPU check")
code("""import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"
""")

# ---------------------------------------------------------------- 3. data
md("""## 2. Data — build a ~2–10 MB text corpus

TinyShakespeare (~1 MB) plus three public-domain Gutenberg books
(Alice in Wonderland, Frankenstein, Pride & Prejudice) lands us
comfortably inside the 2–10 MB window.
""")
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
        print("downloading", name)
        urllib.request.urlretrieve(url, p)
    with open(p, encoding="utf-8", errors="ignore") as f:
        parts.append(f.read())
text = "\\n".join(parts)
print(f"corpus size: {len(text)/1e6:.2f} MB  ({len(text)} chars)")
""")

# ---------------------------------------------------------------- 4. tok
md("## 3. Tokenizer + batches (character-level, keeps it simple)")
code("""import torch

chars = sorted(set(text))
vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.95 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"vocab size: {vocab}, train chars: {len(train_data)}, val chars: {len(val_data)}")

SEQ_LEN, BATCH = 128, 32
def get_batch(split):
    d = train_data if split == "train" else val_data
    i = torch.randint(0, len(d) - SEQ_LEN - 1, (BATCH,))
    x = torch.stack([d[j:j+SEQ_LEN] for j in i]).to(device)
    y = torch.stack([d[j+1:j+SEQ_LEN+1] for j in i]).to(device)
    return x, y
""")

# ---------------------------------------------------------------- 5. models
md("""## 4. The two models — identical except the mixing layer

`CausalSelfAttention` = the standard Transformer layer (baseline).
`LinearRunningState` = the experiment: attention deleted, replaced by the
fixed-size linear recurrence.
""")
code(open("/home/hatch/workspace/linear-attention-lab/models.py").read())

# ---------------------------------------------------------------- 6. build
md("## 5. Build both + parameter counts")
code("""from models import build_models, count_params

DIM, LAYERS, HEADS = 128, 4, 4
attn, lin = build_models(vocab, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                         seq_len=SEQ_LEN)
models = {"attention (baseline)": attn.to(device),
          "linear running state": lin.to(device)}
for name, m in models.items():
    print(f"{name}: {count_params(m)/1e6:.3f}M params")

# sanity: shapes + one forward pass each
xb, yb = get_batch("train")
for name, m in models.items():
    logits, loss = m(xb, yb)
    print(name, "logits", tuple(logits.shape), "loss", round(float(loss), 3))
""")

# ---------------------------------------------------------------- 7. train
md("""## 6. Train both side-by-side

Same optimizer, same LR, same batches — the *only* difference is the
mixing layer. (~5–10 min on a Colab T4 for 2000 steps.)
""")
code("""import math, time

STEPS, LR, EVAL_EVERY = 2000, 3e-4, 200
opt = {n: torch.optim.AdamW(m.parameters(), lr=LR) for n, m in models.items()}
hist = {n: {"train": [], "val": [], "ms": []} for n in models}

for step in range(1, STEPS + 1):
    for name, m in models.items():
        m.train()
        xb, yb = get_batch("train")
        t0 = time.time()
        _, loss = m(xb, yb)
        opt[name].zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt[name].step()
        hist[name]["train"].append(float(loss))
        hist[name]["ms"].append((time.time() - t0) * 1000)
    if step % EVAL_EVERY == 0 or step == 1:
        msg = f"step {step:5d}"
        for name, m in models.items():
            m.eval()
            with torch.no_grad():
                vl = sum(float(m(*get_batch("val"))[1]) for _ in range(10)) / 10
            hist[name]["val"].append(vl)
            msg += (f" | {name}: train {hist[name]['train'][-1]:.3f} "
                    f"val {vl:.3f} (ppl {math.exp(vl):.1f})")
        print(msg, flush=True)

print("done.")
for name in models:
    ms = sum(hist[name]['ms']) / len(hist[name]['ms'])
    print(f"{name}: avg {ms:.1f} ms/batch")
""")

# ---------------------------------------------------------------- 8. plot
md("## 7. Loss curves")
code("""import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
for name in models:
    tr = hist[name]["train"]
    xs = range(1, len(tr) + 1)
    plt.plot(xs, tr, alpha=0.35, label=name + " train")
    ev = hist[name]["val"]
    ex = [i * EVAL_EVERY for i in range(1, len(ev) + 1)]
    plt.plot([1] + ex, [ev[0]] + ev, marker="o", label=name + " val")
plt.xlabel("step"); plt.ylabel("cross-entropy loss")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

# ---------------------------------------------------------------- 9. sample
md("## 8. Generate samples — can you tell which is which?")
code("""for name, m in models.items():
    m.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = m.generate(ctx, 300, temp=0.8, top_k=40)[0].tolist()
    print(f"----- {name} -----")
    print("".join(itos[i] for i in out))
    print()
""")

# ---------------------------------------------------------------- 10. results
md("""## 9. Reading the results

**What to look for:**

| Signal | Meaning |
|---|---|
| Val loss / perplexity gap | How much modeling power the fixed state loses vs attention |
| ms/batch | The speed win of O(T) vs O(T²) — grows with `SEQ_LEN` |
| Sample quality | Whether the linear model keeps coherence over long spans |

**Expected outcome (hypothesis):** the linear state trains faster and
uses constant memory, but scores worse on perplexity — especially on
long-range dependencies (e.g. closing a quote opened 100 chars ago),
because a fixed-size vector is a lossy compression of the past while
attention keeps *every* past token around.

**Try next:**
1. Bump `SEQ_LEN` to 512 — the speed gap should widen, the quality gap too.
2. Increase the state size (`state_dim=256`) — does a bigger fixed memory close the gap?
3. Stack test: give the linear model *more layers* for equal wall-clock time — can depth compensate for the missing attention?
4. Inspect learned decays: `lin.blocks[0].mix.logit_decay.sigmoid()` — which channels learned long vs short memory?
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_replacement_colab.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("notebook written:",
      "/home/hatch/workspace/linear-attention-lab/linear_attention_replacement_colab.ipynb")
