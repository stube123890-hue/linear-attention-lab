"""Generate Colab notebook V12 Phase 1: Vision anchor (ImageNette, AR next-patch).

V8-A (frozen V11 stack: fused mixer + FFN checkpoint) vs matched causal
attention. ONLY the mixer differs: same vision encoder (arch+init), same
param budget (V4-style FFN match), same data/order, same optimizer/schedule.

T=256 (128x128 images, 8x8 patches), 1500 steps, batch 32, fp32.
Gates: report val MSE (quality), peak reserved (VRAM), ms/step (throughput).
No quality gate vs text — this is a generalization probe, not a SOTA claim.
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
mm_src = open(f"{WS}/v12_multimodal.py").read().replace(
    '"""V12 multimodal: AR sequence modeling over vision/audio on the frozen V8-A stack.',
    '"""(inlined) V12 multimodal modules.').replace(
    """import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Block  # frozen
""",
    "# (imports stripped: torch/nn/F + Block already inlined above)\n")
cell_models = models_src + "\n\n" + kern_src + "\n\n" + fused_src + "\n\n" + mm_src

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


md("""# V12 Phase 1 — Vision anchor (ImageNette, AR next-patch)

V8-A (frozen V11 stack) vs matched causal attention. Only the mixer differs.
T=256 (128×128, 8×8 patches), 1500 steps, batch 32. Generalization probe —
not a SOTA claim.
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

md("## Data — ImageNette (320px tarball, resized to 128)")
code('''import urllib.request, tarfile
from PIL import Image
import numpy as np

def dl(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, path)
    return path

tgz = dl("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
         "data/imagenette2-320.tgz")
BASE = "data/imagenette2-320"
if not os.path.exists(BASE):
    print("extracting...", flush=True)
    with tarfile.open(tgz) as tf:
        tf.extractall("data")
import glob
def _imgs(sub):
    # .JPEG (uppercase) on Linux glob is case-sensitive — match both
    return sorted(glob.glob(f"{BASE}/{sub}/*/*.JPEG") +
                  glob.glob(f"{BASE}/{sub}/*/*.jpg"))
train_files = _imgs("train")
val_files = _imgs("val")
print(f"train {len(train_files)} | val {len(val_files)}")
assert len(train_files) > 9000 and len(val_files) > 3000, "imagenette extract incomplete"
''')

md("## Models + kernels + V12 modules (frozen V8-A/V11, new encoders)")
code(cell_models)

md("## Builder (mixer-only difference; FFN checkpoint both arms; param-matched)")
code('''import time, gc
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

DIM, LAYERS, HEADS, IMG, PATCH, FFN_V8A = 256, 8, 8, 128, 8, 766
T = (IMG // PATCH) ** 2
assert T == 256, T

_orig_block_forward = Block.forward
def _block_forward_ckpt(self, x, reset=None):
    x = x + self.mix(self.ln1(x), reset)
    x = x + _ckpt.checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
    return x
Block.forward = _block_forward_ckpt

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_batch(files, idx, device):
    imgs = []
    for i in idx:
        im = Image.open(files[i]).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
        t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        imgs.append(t)
    x = torch.stack(imgs)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)

def build(mixer, seed, ffn_hidden):
    set_seed(seed)
    enc = VisionEncoder(DIM, patch=PATCH)
    m = MultimodalAR(enc, DIM, LAYERS, HEADS, T, mixer, ffn_hidden=ffn_hidden).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        if hasattr(_blk.mix, "forget_floor"):
            _blk.mix.forget_floor.data.fill_(_floors[_i])
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    return m, o

# param match (V4-style): attention arm gets bigger FFN
torch.manual_seed(0)
_m0 = MultimodalAR(VisionEncoder(DIM), DIM, 1, HEADS, T,
                   lambda d, h: SelectiveSegmentedStateV8AFused(d, state_dim=256),
                   ffn_hidden=FFN_V8A)
_pv = sum(p.numel() for p in _m0.blocks[0].mix.parameters())
_ma = MultimodalAR(VisionEncoder(DIM), DIM, 1, HEADS, T,
                   lambda d, h: CausalSelfAttention(d, h), ffn_hidden=FFN_V8A)
_pa = sum(p.numel() for p in _ma.blocks[0].mix.parameters())
FFN_ATTN = match_ffn_hidden(FFN_V8A, DIM, _pv, _pa)
del _m0, _ma
print(f"mixer params/layer: v8a {_pv} vs attn {_pa} -> ffn {FFN_V8A} vs {FFN_ATTN}")
''')

md("## Train (1500 steps, batch 32) + measure")
code('''import numpy as np
STEPS, BS = 1500, 32
g = torch.Generator().manual_seed(1234)
train_order = torch.randperm(len(train_files), generator=g).tolist()
val_idx = list(range(0, len(val_files), 8))[:128]  # fixed val subset

def train_arm(mixer_fn, ffn_hidden, label):
    m, o = build(mixer_fn, 0, ffn_hidden)
    xb0 = load_batch(train_files, train_order[:BS], device)
    with torch.no_grad():
        m(xb0)
    o.zero_grad(set_to_none=True); m.ar_loss(xb0).backward(); o.step()  # warmup
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ptr = 0
    for s in range(STEPS):
        idx = train_order[ptr:ptr+BS]; ptr += BS
        if ptr + BS > len(train_order):
            ptr = 0
        xb = load_batch(train_files, idx, device)
        o.zero_grad(set_to_none=True)
        loss = m.ar_loss(xb); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_a = torch.cuda.max_memory_allocated() / 1e6
    peak_r = torch.cuda.max_memory_reserved() / 1e6
    ms = 1000 * dt / STEPS
    # val MSE (fixed subset, no grad)
    m.eval()
    with torch.no_grad():
        vl = 0.0
        for i in range(0, len(val_idx), BS):
            xb = load_batch(val_files, val_idx[i:i+BS], device)
            vl += m.ar_loss(xb).item() * len(val_idx[i:i+BS])
        vl /= len(val_idx)
    m.train()
    print(f"{label}: val MSE {vl:.4f} | peak {peak_a:.0f}/{peak_r:.0f} MB | {ms:.1f} ms/step")
    del m, o
    gc.collect(); torch.cuda.empty_cache()
    return vl, peak_a, peak_r, ms

r_v8a = train_arm(lambda d, h: SelectiveSegmentedStateV8AFused(d, state_dim=256), FFN_V8A, "V8-A")
r_attn = train_arm(lambda d, h: CausalSelfAttention(d, h), FFN_ATTN, "ATTN")
print(f"quality delta (attn - v8a val MSE): {r_attn[0] - r_v8a[0]:+.4f}")
print(f"vram ratio (attn/v8a reserved): {r_attn[2]/r_v8a[2]:.2f}x")
print(f"throughput ratio (attn/v8a ms): {r_attn[3]/r_v8a[3]:.2f}x")
open("/content/v12p1_results.txt", "w").write(
    f"v8a_val={r_v8a[0]:.4f}\\nv8a_alloc={r_v8a[1]:.0f}\\nv8a_res={r_v8a[2]:.0f}\\nv8a_ms={r_v8a[3]:.1f}\\n"
    f"attn_val={r_attn[0]:.4f}\\nattn_alloc={r_attn[1]:.0f}\\nattn_res={r_attn[2]:.0f}\\nattn_ms={r_attn[3]:.1f}\\n")
from google.colab import files
files.download("/content/v12p1_results.txt")
print("SAVE DONE")
''')

path = f"{WS}/linear_attention_v12p1.ipynb"
with open(path, "w") as f:
    json.dump(NB, f)
print("written:", path, f"({len(NB['cells'])} cells)")
