"""Generate Colab notebook V11-D1 Path B: fused projection+scan (save-x-only).

V11-D1 Path B: _FusedProjScanFn wraps the 4 scan-input projections + the
EXISTING Triton scan kernels. Forward runs projections with cuBLAS
(transient, never saved); saves only (x, keep, h_seq, a_seq). Backward
recomputes projections and runs the existing verified bwd kernel, then
standard dx/dW. out_gate_proj stays separate (consumed after the scan).

Single notebook, two phases over COMPLETE training steps (his measurement
rule — the backward live set is the point, not forward VRAM):
  Phase A = plain V8-A (baseline), Phase B = fused mixer.
Gates: peak reserved <= -25%; step cost <= +10%; grads within 1e-6
(numerical equivalence — Path B reuses the same kernels, expect ~0).
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
fused_src = open(f"{WS}/fused_mixer.py").read().replace(
    '"""V11-D1 Path B: fused projection+scan via save-x-only autograd Function.',
    '"""(inlined) V11-D1 Path B: fused projection+scan.').replace(
    """import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels import (
    SelectiveSegmentedStateV8A,
    _delta_scan_fwd_kernel,
    _delta_scan_bwd_kernel,
    _delta_scan_cpu,
    BLOCK_S,
)
""",
    "# (imports stripped: torch/nn/F + triton_kernels names already inlined above)\n")
cell2_src = models_src + "\n\n" + kern_src + "\n\n" + fused_src

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


md("""# V11-D1 Path B — fused projection+scan (save-x-only)

The 4 scan-input projections run as cuBLAS GEMMs (transient, never saved);
the Function saves only (x, keep, h_seq, a_seq). Backward recomputes the
projections and runs the EXISTING verified Triton bwd kernel. out_gate_proj
stays separate. Measurement is over COMPLETE training steps (the backward
live set is the point).
Gates: peak reserved ≤ −25%; step cost ≤ +10%; grads within 1e-6.
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

md("## Models + Triton kernels + fused mixer (V8-A math unchanged)")
code(cell2_src)

md("## Builder + measurement (complete training steps)")
code('''import time, gc
import torch.nn as nn
DIM, LAYERS, HEADS, SEQ, FFN_H = 256, 8, 8, 128, 766

def build_v8a(seed, fused):
    set_seed(seed)
    mix_cls = SelectiveSegmentedStateV8AFused if fused else SelectiveSegmentedStateV8A
    m = TinyLM(vocab_a, DIM, LAYERS, HEADS, SEQ,
               mixer_fn=lambda d, h: mix_cls(d, state_dim=256),
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

def train_step(m, o, xb, yb):
    o.zero_grad(set_to_none=True)
    _, loss = m(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    o.step()
    return float(loss)

def param_checksum(m):
    return sum(p.detach().double().sum().item() for p in m.parameters())

def timed_phase(fused, label):
    m, o = build_v8a(0, fused)
    xb0, yb0 = batches[0][0].to(device), batches[0][1].to(device)
    with torch.no_grad():
        m(xb0)
    train_step(m, o, xb0, yb0)  # warmup (Triton fwd+bwd compile)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for s in range(3):
        xb, yb = batches[s][0].to(device), batches[s][1].to(device)
        train_step(m, o, xb, yb)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_a = torch.cuda.max_memory_allocated() / 1e6
    peak_r = torch.cuda.max_memory_reserved() / 1e6
    cs = param_checksum(m)
    ms = 1000 * dt / 3
    print(f"D1-{label} peak alloc: {peak_a:.0f} MB | peak reserved: {peak_r:.0f} MB | {ms:.1f} ms/step")
    del m, o
    gc.collect(); torch.cuda.empty_cache()
    return peak_a, peak_r, ms, cs
''')

md("## Phase A — plain V8-A baseline (complete steps)")
code('''aA, rA, msA, csA = timed_phase(False, "A-plain")
''')

md("## Phase B — fused projection+scan (complete steps)")
code('''aB, rB, msB, csB = timed_phase(True, "B-fused")
d_r = 100 * (rB - rA) / rA
d_ms = 100 * (msB - msA) / msA
g_rel = abs(csA - csB) / max(abs(csA), 1e-12)
print(f"reserved delta: {d_r:+.1f}% (gate <= -25%)")
print(f"step cost: {d_ms:+.1f}% (gate <= +10%)")
print(f"param checksum A={csA:.10f} B={csB:.10f} | rel-diff {g_rel:.2e} (gate <= 1e-6)")
open("/content/d1_results.txt", "w").write(
    f"a_alloc_MB={aA:.0f}\\na_reserved_MB={rA:.0f}\\na_ms={msA:.1f}\\n"
    f"b_alloc_MB={aB:.0f}\\nb_reserved_MB={rB:.0f}\\nb_ms={msB:.1f}\\n"
    f"reserved_delta_pct={d_r:.1f}\\nstep_cost_pct={d_ms:.1f}\\n"
    f"grad_rel_diff={g_rel:.3e}\\n")
from google.colab import files
files.download("/content/d1_results.txt")
print("SAVE DONE")
''')

path = f"{WS}/linear_attention_v11d1.ipynb"
with open(path, "w") as f:
    json.dump(NB, f)
print("written:", path, f"({len(NB['cells'])} cells)")
