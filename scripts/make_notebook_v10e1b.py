"""Generate Colab notebook V10-E1b: clean single-model peak comparison.

E1.2 measured 1066 -> 668 MB (-37.3%, gate >=40% FAIL) but kept TWO models
resident during both measurements, inflating both numbers. E1b re-measures
with ONE model at a time: batch-32 vs 4x8 accumulation. ~5 min on T4.
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

md("""# V10-E1b — Clean single-model peak: batch-32 vs 4×8 accumulation

E1.2's peak comparison kept two models resident, inflating both readings
(1066 → 668 MB, −37.3%, just under the 40% gate). This re-measures with one
model at a time — the production configuration.
""")

code('''import os, urllib.request
import torch
assert torch.cuda.is_available(), "needs GPU"
device = "cuda"
def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(0)
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
''')

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
assert vocab_a == 104, f"vocab drift: {vocab_a}"
print(f"vocab {voc_a} ok" if False else f"vocab {vocab_a} | tokens {len(data_a)/1e6:.1f}M")
''')

code(cell2_src)

code('''DIM, LAYERS, HEADS, SEQ, FFN_H = 256, 8, 8, 128, 766
def build_v8a(seed):
    set_seed(seed)
    m = TinyLM(vocab_a, DIM, LAYERS, HEADS, SEQ,
               mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
               ffn_hidden=FFN_H, newline_id=stoi_a.get("\\n")).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        _blk.mix.forget_floor.data.fill_(_floors[_i])
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    return m, o

def ref_step(m, o, xb, yb):
    o.zero_grad(set_to_none=True)
    _, loss = m(xb, yb); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()

def accum_step(m, o, xb, yb, micro=8):
    o.zero_grad(set_to_none=True)
    n = xb.size(0) // micro
    for k in range(n):
        _, loss = m(xb[k*micro:(k+1)*micro], yb[k*micro:(k+1)*micro])
        (loss / n).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()

# deterministic batch-32 inputs (CPU)
g = torch.Generator().manual_seed(1234)
dd = data_a[:int(0.95*len(data_a))]
batches = []
for _ in range(5):
    i = torch.randint(0, len(dd)-SEQ-1, (32,), generator=g)
    batches.append((torch.stack([dd[j:j+SEQ] for j in i]),
                    torch.stack([dd[j+1:j+SEQ+1] for j in i])))
print("batches ready:", len(batches))
''')

md("## E1.2b — one model at a time")
code('''def measure(step_fn, tag):
    m, o = build_v8a(0)
    xb0, yb0 = batches[0][0].to(device), batches[0][1].to(device)
    with torch.no_grad():
        m(xb0)  # warmup (Triton compile) outside measurement
    torch.cuda.reset_peak_memory_stats()
    for s in range(3):
        xb, yb = batches[s][0].to(device), batches[s][1].to(device)
        step_fn(m, o, xb, yb)
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"{tag}: peak {peak:.0f} MB", flush=True)
    del m, o
    torch.cuda.empty_cache()
    return peak

peak_ref = measure(ref_step, "batch-32")
peak_acc = measure(accum_step, "4x8 accum")
reduction = 100 * (peak_ref - peak_acc) / peak_ref
print(f"E1.2b reduction: {reduction:.1f}%  (gate >= 40%)")
print("E1.2b VERDICT:", "PASS" if reduction >= 40 else "FAIL")
open("/content/e1b_results.txt", "w").write(
    f"peak_ref_MB={peak_ref:.0f}\\npeak_acc_MB={peak_acc:.0f}\\nreduction_pct={reduction:.1f}\\n")
print(open("/content/e1b_results.txt").read())
from google.colab import files
files.download("/content/e1b_results.txt")
print("SAVE DONE")
''')

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v10e1b_peak.ipynb", "w") as f:
    json.dump(NB, f)
print("notebook written (6 cells)")
