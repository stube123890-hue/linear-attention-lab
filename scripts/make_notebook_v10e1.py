"""Generate Colab notebook V10-E1: micro-batch 4x8 + grad accumulation.

E0's ablation map (accepted by him): mixer ~408 MB, FFN ~277 MB live-at-peak,
batch slope 20.8 MB/sample. E1 tests the exact-dynamics lever: micro-batch 8
with 4 accumulation steps instead of batch 32. Same data, same order, loss/4
per micro-step -> mathematically identical update; fp32 summation order is the
only divergence source (V9-B proved 2x16 at 2.98e-08).

Gates (from v10_study_brief.md): grad delta <= 1e-6 (E1.1); peak <= -40%
(E1.2); val@500 within +/-0.02 (E1.3). [E]: ~420-630 MB saved, 5-15% step cost.
"""
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


# ----------------------------------------------------------------------------
# Model + Triton source: byte-identical build to V10-E0 (same files, same edits)
# ----------------------------------------------------------------------------
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
cell2_src = models_src + "\n\n" + kern_src

# ----------------------------------------------------------------------------
# Notebook cells
# ----------------------------------------------------------------------------
md("""# V10-E1 — Micro-batch 4×8 + gradient accumulation

E0's accepted map: mixer ~408 MB, FFN ~277 MB live-at-peak, batch slope
~20.8 MB/sample. E1 tests the exact-dynamics lever: micro-batch 8 with 4
accumulation steps instead of batch 32. Same data in the same order, loss
scaled by 1/4 per micro-step — mathematically the identical update; only
fp32 summation order can diverge (V9-B proved 2×16 equivalence at 2.98e-08).

**Gates:** grad delta ≤ 1e-6 (E1.1) · peak ≤ −40% vs batch-32 (E1.2) ·
val@500 within ±0.02 of the batch-32 reference (E1.3).
Expectation: ~420–630 MB saved at 5–15% step-time cost.
""")

code('''import os, urllib.request, time
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "needs GPU"
device = "cuda"

def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed(0)
print("torch", torch.__version__, "| cuda:", torch.cuda.get_device_name(0))
''')

md("## 1. Data — campaign corpus A only (vocab 104, same as V8-A Run 1 / E0)")
code('''def dl(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, path)
    return path

URLS = {
    "tinyshakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "alice.txt":           "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "frankenstein.txt":    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "pride.txt":           "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
}
parts = []
for name, url in URLS.items():
    p = dl(url, f"data/{name}")
    with open(p, encoding="utf-8", errors="ignore") as f:
        parts.append(f.read())
text_a = "\\n".join(parts)
chars_a = sorted(set(text_a)); vocab_a = len(chars_a)
stoi_a = {c: i for i, c in enumerate(chars_a)}
data_a = torch.tensor([stoi_a[c] for c in text_a], dtype=torch.long)
nl_a = stoi_a.get("\\n")
assert vocab_a == 104, f"vocab drift: {vocab_a}"
print(f"vocab {vocab_a} | tokens {len(data_a)/1e6:.1f}M")
''')

md("## 2. Models + Triton kernels (inlined, byte-identical to E0)")
code(cell2_src)

md("""## 3. Model builder + deterministic lockstep data

Both arms see the SAME 500 batches in the SAME order (pre-generated on CPU
with a seeded generator). The accumulation arm splits each batch-32 into
4 micro-batches of 8. Val set is fixed (20 batches).
""")
code('''DIM, LAYERS, HEADS, SEQ, FFN_H = 256, 8, 8, 128, 766
N_STEPS = 500

def build_v8a(seed):
    set_seed(seed)
    m = TinyLM(vocab_a, DIM, LAYERS, HEADS, SEQ,
               mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
               ffn_hidden=FFN_H, newline_id=nl_a).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        _blk.mix.forget_floor.data.fill_(_floors[_i])
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    return m, o

def pregen_batches(d, seq_len, batch_size, n_steps, seed, split="train"):
    dd = d[:int(0.95*len(d))] if split == "train" else d[int(0.95*len(d)):]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_steps):
        i = torch.randint(0, len(dd)-seq_len-1, (batch_size,), generator=g)
        x = torch.stack([dd[j:j+seq_len] for j in i])
        y = torch.stack([dd[j+1:j+seq_len+1] for j in i])
        out.append((x, y))
    return out

train_batches = pregen_batches(data_a, SEQ, 32, N_STEPS, seed=1234, split="train")
val_batches = pregen_batches(data_a, SEQ, 32, 20, seed=999, split="val")
print(f"train batches: {len(train_batches)} | val batches: {len(val_batches)}")

def val_loss(m):
    m.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_batches:
            _, loss = m(xb.to(device), yb.to(device))
            tot += float(loss); n += 1
    m.train()
    return tot / n

def ref_step(m, o, xb, yb):
    o.zero_grad(set_to_none=True)
    _, loss = m(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    o.step()
    return float(loss)

def accum_step(m, o, xb, yb, micro=8):
    o.zero_grad(set_to_none=True)
    n = xb.size(0) // micro
    for k in range(n):
        xs, ys = xb[k*micro:(k+1)*micro], yb[k*micro:(k+1)*micro]
        _, loss = m(xs, ys)
        (loss / n).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    o.step()
''')

md("""## E1.1 — Micro-batch equivalence probe (gate: grad delta ≤ 1e-6)

One batch-32 step vs four micro-batch-8 accumulation steps on identical data
and identically-initialized models. Max absolute grad delta across all params.
""")
code('''m_ref, o_ref = build_v8a(0)
m_acc, o_acc = build_v8a(0)
# warmup (Triton compile) on throwaway batch, then re-init for a clean probe
xb0, yb0 = train_batches[0]
xb0d, yb0d = xb0.to(device), yb0.to(device)
with torch.no_grad():
    m_ref(xb0d); m_acc(xb0d)
m_ref, o_ref = build_v8a(0)
m_acc, o_acc = build_v8a(0)

ref_step(m_ref, o_ref, xb0d, yb0d)
accum_step(m_acc, o_acc, xb0d, yb0d, micro=8)
max_delta = 0.0
for pr, pa in zip(m_ref.parameters(), m_acc.parameters()):
    d = (pr.grad - pa.grad).abs().max().item()
    max_delta = max(max_delta, d)
print(f"E1.1 max grad delta: {max_delta:.3e}  (gate <= 1e-6)")
''')

md("""## E1.2 — Peak VRAM: batch-32 vs 4×8 accumulation (gate: peak ≤ −40%)""")
code('''m_ref, o_ref = build_v8a(0)
m_acc, o_acc = build_v8a(0)
xb0d, yb0d = train_batches[0][0].to(device), train_batches[0][1].to(device)
with torch.no_grad():
    m_ref(xb0d); m_acc(xb0d)  # warmup outside measurement

torch.cuda.reset_peak_memory_stats()
for s in range(3):
    xb, yb = train_batches[s][0].to(device), train_batches[s][1].to(device)
    ref_step(m_ref, o_ref, xb, yb)
peak_ref = torch.cuda.max_memory_allocated() / 1e6

torch.cuda.reset_peak_memory_stats()
for s in range(3):
    xb, yb = train_batches[s][0].to(device), train_batches[s][1].to(device)
    accum_step(m_acc, o_acc, xb, yb, micro=8)
peak_acc = torch.cuda.max_memory_allocated() / 1e6

reduction = 100 * (peak_ref - peak_acc) / peak_ref
print(f"E1.2 peak batch-32 : {peak_ref:.0f} MB")
print(f"E1.2 peak 4x8 accum: {peak_acc:.0f} MB")
print(f"E1.2 reduction: {reduction:.1f}%  (gate >= 40%)")
''')

md("""## E1.3 — 500-step lockstep trajectories (gate: val@500 within ±0.02)

Fresh identically-initialized models, same 500 batches in the same order.
Val checked at 0/100/.../500 on the fixed val set. Step times recorded for
the throughput-cost estimate.
""")
code('''results = {}
for tag, step_fn in (("ref", ref_step), ("acc", accum_step)):
    m, o = build_v8a(0)
    xb0d, yb0d = train_batches[0][0].to(device), train_batches[0][1].to(device)
    with torch.no_grad():
        m(xb0d)  # warmup (Triton compile) outside timing
    vals = [val_loss(m)]
    t_start = time.perf_counter()
    for s in range(N_STEPS):
        xb, yb = train_batches[s][0].to(device), train_batches[s][1].to(device)
        step_fn(m, o, xb, yb)
        if (s + 1) % 100 == 0:
            vals.append(val_loss(m))
            print(f"[{tag}] step {s+1:4d}  val {vals[-1]:.4f}", flush=True)
    dt = time.perf_counter() - t_start
    results[tag] = {"vals": vals, "secs": dt}
    print(f"[{tag}] done: {dt:.1f}s  ({1000*dt/N_STEPS:.1f} ms/step)", flush=True)

vr, va = results["ref"]["vals"], results["acc"]["vals"]
val_gap_500 = abs(va[-1] - vr[-1])
max_val_gap = max(abs(a - b) for a, b in zip(va, vr))
tok_s_ref = N_STEPS * 32 * SEQ / results["ref"]["secs"]
tok_s_acc = N_STEPS * 32 * SEQ / results["acc"]["secs"]
overhead = 100 * (results["acc"]["secs"] - results["ref"]["secs"]) / results["ref"]["secs"]
print(f"E1.3 val@500 ref {vr[-1]:.4f} vs acc {va[-1]:.4f}  gap {val_gap_500:.4f} (gate <= 0.02)")
print(f"E1.3 max val gap over run: {max_val_gap:.4f}")
print(f"E1.3 tok/s ref {tok_s_ref:.0f} vs acc {tok_s_acc:.0f}  step overhead {overhead:+.1f}% (expect +5..15%)")
''')

md("## E1.4 — Verdict")
code('''g1 = max_delta <= 1e-6
g2 = reduction >= 40
g3 = val_gap_500 <= 0.02
print(f"GATE grad delta <= 1e-6 : {'PASS' if g1 else 'FAIL'} ({max_delta:.3e})")
print(f"GATE peak reduction >=40%: {'PASS' if g2 else 'FAIL'} ({reduction:.1f}%)")
print(f"GATE val@500 within 0.02: {'PASS' if g3 else 'FAIL'} ({val_gap_500:.4f})")
if overhead > 15:
    print(f"NOTE: step overhead {overhead:.1f}% exceeds the 5-15% estimate")
if g1 and g2 and g3:
    print("E1 VERDICT: PASS — micro-batch 4x8 is an exact-dynamics VRAM lever")
else:
    print("E1 VERDICT: FAIL — see gate(s) above")
''')

md("## 8. Save")
code('''lines = []
lines.append(f"e11_grad_delta={max_delta:.3e}")
lines.append(f"e12_peak_ref_MB={peak_ref:.0f}")
lines.append(f"e12_peak_acc_MB={peak_acc:.0f}")
lines.append(f"e12_reduction_pct={reduction:.1f}")
lines.append(f"e13_val500_ref={vr[-1]:.4f}")
lines.append(f"e13_val500_acc={va[-1]:.4f}")
lines.append(f"e13_val500_gap={val_gap_500:.4f}")
lines.append(f"e13_max_val_gap={max_val_gap:.4f}")
lines.append(f"e13_toks_ref={tok_s_ref:.0f}")
lines.append(f"e13_toks_acc={tok_s_acc:.0f}")
lines.append(f"e13_overhead_pct={overhead:.1f}")
lines.append(f"gates_pass={bool(g1 and g2 and g3)}")
open("/content/e1_results.txt", "w").write("\\n".join(lines))
print(open("/content/e1_results.txt").read())
from google.colab import files
files.download("/content/e1_results.txt")
print("SAVE DONE")
''')

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v10e1_microbatch.ipynb", "w") as f:
    json.dump(NB, f)
print("notebook written:",
      "/home/hatch/workspace/linear-attention-lab/linear_attention_v10e1_microbatch.ipynb",
      f"({len(NB['cells'])} cells)")
