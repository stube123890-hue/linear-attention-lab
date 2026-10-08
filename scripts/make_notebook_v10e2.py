"""Generate Colab notebook pair V10-E2: expandable_segments allocator probe.

E2: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lets the caching
allocator split large segments -> attacks the 61 MB frag gap from E0
(761 MB alloc / 822 MB reserved). Almost-free experiment per his ladder.

Two notebooks, identical except cell 1 (env var must precede CUDA init,
so each gets a fresh runtime): v10e2a = baseline, v10e2b = expandable.
Each measures peak alloc + peak reserved over 3 train steps + ms/step.

Gate: take it if reserved drops >= 30 MB with no throughput regression.
"""
import json

WS = "/home/hatch/workspace/linear-attention-lab"
models_src = open(f"{WS}/models.py").read().replace(
    '"""Linear-recurrent replacement for Transformer self-attention.',
    '"""(inlined) Linear-recurrent replacement for Transformer self-attention.').replace(
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import math\nimport torch.nn as nn\nimport torch.nn.functional as F\n")
kern_src = open(f"{WS}/triton_kernels.py").read().replace(
    '"""V6: Triton-fused selective scan for the gated segmented state space.',
    '"""(inlined) V6/V7/V8-A: Triton-fused scans.').replace(
    "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
    "import triton\nimport triton.language as tl\n")
cell2_src = models_src + "\n\n" + kern_src


def build(tag, expandable):
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

    md(f"""# V10-E2{tag} — expandable_segments allocator probe ({'ON' if expandable else 'OFF'})

E0 map: 761 MB allocated / 822 MB reserved → 61 MB fragmentation gap.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lets the caching
allocator split large segments instead of stranding them. Gate: reserved
drops ≥ 30 MB with no step-time regression → take it (free win).
""")

    if expandable:
        code('''import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("alloc conf:", os.environ["PYTORCH_CUDA_ALLOC_CONF"])
import torch
assert torch.cuda.is_available(), "needs GPU"
device = "cuda"
tag = "b"
''')
    else:
        code('''import os
print("alloc conf:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"))
import torch
assert torch.cuda.is_available(), "needs GPU"
device = "cuda"
tag = "a"
''')

    code('''def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(0)
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
''')

    md("## Data — campaign corpus A (vocab 104)")
    code('''import urllib.request
def dl(url, path):
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
print(f"vocab {vocab_a} | tokens {len(data_a)/1e6:.1f}M")
''')

    md("## Models + Triton kernels (byte-identical to E0/E1)")
    code(cell2_src)

    md("## Builder + peak measurement (alloc AND reserved)")
    code('''import time
import torch.nn as nn
DIM, LAYERS, HEADS, SEQ, FFN_H = 256, 8, 8, 128, 766

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

def pregen(d, seq_len, bs, n, seed):
    dd = d[:int(0.95*len(d))]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        i = torch.randint(0, len(dd)-seq_len-1, (bs,), generator=g)
        out.append((torch.stack([dd[j:j+seq_len] for j in i]),
                    torch.stack([dd[j+1:j+seq_len+1] for j in i])))
    return out

batches = pregen(data_a, SEQ, 32, 5, 1234)
print(f"batches ready: {len(batches)}")

m, o = build_v8a(0)
xb0, yb0 = batches[0][0].to(device), batches[0][1].to(device)
with torch.no_grad():
    m(xb0)  # warmup (Triton compile) outside measurement

def train_step(m, o, xb, yb):
    o.zero_grad(set_to_none=True)
    _, loss = m(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    o.step()
    return float(loss)

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for s in range(3):
    xb, yb = batches[s][0].to(device), batches[s][1].to(device)
    train_step(m, o, xb, yb)
dt = time.perf_counter() - t0
peak_alloc = torch.cuda.max_memory_allocated() / 1e6
peak_res = torch.cuda.max_memory_reserved() / 1e6
print(f"E2{tag} peak alloc: {peak_alloc:.0f} MB | peak reserved: {peak_res:.0f} MB "
      f"| frag gap: {peak_res-peak_alloc:.0f} MB | {1000*dt/3:.1f} ms/step")
open(f"/content/e2{tag}_results.txt", "w").write(
    f"peak_alloc_MB={peak_alloc:.0f}\\npeak_reserved_MB={peak_res:.0f}\\n"
    f"frag_gap_MB={peak_res-peak_alloc:.0f}\\nms_per_step={1000*dt/3:.1f}\\n")
from google.colab import files
files.download(f"/content/e2{tag}_results.txt")
print("SAVE DONE")
''')

    path = f"{WS}/linear_attention_v10e2{tag}.ipynb"
    with open(path, "w") as f:
        json.dump(NB, f)
    print("written:", path, f"({len(NB['cells'])} cells)")
    return path


build("a", expandable=False)
build("b", expandable=True)
