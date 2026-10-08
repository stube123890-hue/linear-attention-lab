"""Generate Colab notebook V10-E4: optimizer-state CPU offload probe.

E4: AdamW exp_avg/exp_avg_sq (~51 MB [M]) live on CPU, prefetched to GPU only
around o.step(). Single notebook, two phases in one session:
  Phase A = states on GPU (baseline), Phase B = states offloaded.
Gates: peak ~= -51 MB exactly; step cost < 2%; final params bit-identical
(math is exact — any deviation is a bug).
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


md("""# V10-E4 — optimizer-state CPU offload

AdamW's exp_avg/exp_avg_sq (~51 MB) live on CPU; prefetched to GPU only
around `o.step()`. Math is bit-identical — any trajectory deviation is a bug.
Gates: peak ≈ −51 MB; step cost < 2%; final params bit-identical.
""")

code('''import os
print("alloc conf:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"))
import torch
assert torch.cuda.is_available(), "needs GPU"
device = "cuda"
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

md("## Models + Triton kernels (byte-identical to E0–E3)")
code(cell2_src)

md("## Builder, offload helpers, step fns")
code('''import time, gc
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

OPT_KEYS = ("exp_avg", "exp_avg_sq")
def offload_opt(o):
    for s in o.state.values():
        for k in OPT_KEYS:
            if k in s:
                s[k] = s[k].to("cpu")
def prefetch_opt(o):
    for s in o.state.values():
        for k in OPT_KEYS:
            if k in s:
                s[k] = s[k].to(device)
def opt_state_bytes(o):
    return sum(s[k].numel() * s[k].element_size()
               for s in o.state.values() for k in OPT_KEYS if k in s)

def fwd_bwd(m, o, xb, yb):
    o.zero_grad(set_to_none=True)
    _, loss = m(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    return float(loss)

def train_step_gpu(m, o, xb, yb):
    loss = fwd_bwd(m, o, xb, yb)
    o.step()
    return loss

def train_step_offloaded(m, o, xb, yb):
    loss = fwd_bwd(m, o, xb, yb)
    prefetch_opt(o)
    o.step()
    offload_opt(o)
    return loss

def param_checksum(m):
    return sum(p.detach().double().sum().item() for p in m.parameters())

def timed_phase(m, o, step_fn, label):
    xb0, yb0 = batches[0][0].to(device), batches[0][1].to(device)
    with torch.no_grad():
        m(xb0)
    step_fn(m, o, xb0, yb0)  # warmup (materializes opt state)
    if label == "offloaded":
        offload_opt(o)
        print(f"opt states on CPU: {opt_state_bytes(o)/1e6:.1f} MB")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for s in range(3):
        xb, yb = batches[s][0].to(device), batches[s][1].to(device)
        step_fn(m, o, xb, yb)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (torch.cuda.max_memory_allocated() / 1e6,
            torch.cuda.max_memory_reserved() / 1e6, 1000 * dt / 3)
''')

md("## Phase A — optimizer states on GPU (baseline)")
code('''mA, oA = build_v8a(0)
aA, rA, msA = timed_phase(mA, oA, train_step_gpu, "gpu")
sumA = param_checksum(mA)
print(f"E4-A peak alloc: {aA:.0f} MB | peak reserved: {rA:.0f} MB | {msA:.1f} ms/step")
del mA, oA
gc.collect(); torch.cuda.empty_cache()
''')

md("## Phase B — optimizer states offloaded to CPU")
code('''mB, oB = build_v8a(0)
aB, rB, msB = timed_phase(mB, oB, train_step_offloaded, "offloaded")
sumB = param_checksum(mB)
print(f"E4-B peak alloc: {aB:.0f} MB | peak reserved: {rB:.0f} MB | {msB:.1f} ms/step")
print(f"param checksum A={sumA:.10f} B={sumB:.10f} | bit-identical: {sumA == sumB}")
print(f"reserved delta: {rB-rA:.0f} MB (gate ~= -51) | step cost: {100*(msB-msA)/msA:+.1f}% (gate < +2%)")
open("/content/e4_results.txt", "w").write(
    f"a_alloc_MB={aA:.0f}\\na_reserved_MB={rA:.0f}\\na_ms={msA:.1f}\\n"
    f"b_alloc_MB={aB:.0f}\\nb_reserved_MB={rB:.0f}\\nb_ms={msB:.1f}\\n"
    f"reserved_delta_MB={rB-rA:.0f}\\nstep_cost_pct={100*(msB-msA)/msA:.1f}\\n"
    f"bit_identical={sumA == sumB}\\n")
from google.colab import files
files.download("/content/e4_results.txt")
print("SAVE DONE")
''')

path = f"{WS}/linear_attention_v10e4.ipynb"
with open(path, "w") as f:
    json.dump(NB, f)
print("written:", path, f"({len(NB['cells'])} cells)")
