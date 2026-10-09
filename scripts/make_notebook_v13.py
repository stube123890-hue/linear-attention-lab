"""Generate Colab notebook V13: Vision hyperparameter probe.

Question (from V12 brief): Is V8-A's ~22% vision quality gap (0.2578 vs
0.2102 on ImageNette) a hyperparameter artifact of text-tuned settings, or
a fundamental limitation of compressive state for spatial detail?

Design: LR grid {1e-4, 3e-4, 1e-3} x {V8-A, ATTN} = 6 arms, 1500 steps each,
ImageNette T=256, param-matched, mixer-only difference. Both arms get EQUAL
tuning budget. Tensor-level init verification on shared encoder (Phase 4
lesson: identical seeds do not guarantee identical weights).

Stopping rule: if V8-A's best LR closes the gap substantially vs
attention's best -> artifact (revise modality story). If gap persists ->
evidence for architectural limitation.

Frozen: models.py, triton_kernels.py, fused_mixer.py, v12_multimodal.py.
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


md("""# V13 — Vision hyperparameter probe

**Question:** Is V8-A's ImageNette gap (0.2578 vs 0.2102) a hyperparameter
artifact or fundamental?

**Design:** LR grid {1e-4, 3e-4, 1e-3} × {V8-A, ATTN}, 1500 steps each.
Equal tuning budget. Tensor-level init verification (Phase 4 lesson).

**Stopping rule:** V8-A best closes gap substantially → artifact.
Gap persists → architectural evidence.
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

md("## Data — ImageNette (same as V12 Phase 1)")
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
    return sorted(glob.glob(f"{BASE}/{sub}/*/*.JPEG") +
                  glob.glob(f"{BASE}/{sub}/*/*.jpg"))
train_files = _imgs("train")
val_files = _imgs("val")
print(f"train {len(train_files)} | val {len(val_files)}")
assert len(train_files) > 9000 and len(val_files) > 3000
''')

md("## Models + kernels + V12 modules (frozen)")
code(cell_models)

md("## Builder with tensor-level init verification")
code('''import time, gc
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

DIM, LAYERS, HEADS, IMG, PATCH, FFN_V8A = 256, 8, 8, 128, 8, 766
T = (IMG // PATCH) ** 2
assert T == 256, T
LRS = [1e-4, 3e-4, 1e-3]

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

# param match (V4-style)
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

def build_arm(mixer_fn, ffn_hidden, lr, seed=0):
    """Build arm with tensor-verified shared encoder init."""
    # Build reference encoder with fixed seed, save state
    set_seed(seed)
    ref_enc = VisionEncoder(DIM, patch=PATCH)
    ref_state = {k: v.clone() for k, v in ref_enc.state_dict().items()}
    del ref_enc
    # Build full model (MultimodalAR.__init__ calls self.apply(self._init),
    # which re-randomizes the encoder too — so load ref state AFTER.)
    set_seed(seed)
    enc = VisionEncoder(DIM, patch=PATCH)
    m = MultimodalAR(enc, DIM, LAYERS, HEADS, T, mixer_fn,
                     ffn_hidden=ffn_hidden).to(device)
    m.encoder.load_state_dict(ref_state)
    # Verify: encoder weights must match reference exactly
    for k, v in m.encoder.state_dict().items():
        if not torch.equal(v.cpu(), ref_state[k].cpu()):
            raise RuntimeError(f"INIT MISMATCH: encoder.{k}")
    print(f"INIT OK: encoder verified ({len(ref_state)} tensors)")
    # forget floors (V8-A specific, after verification)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        if hasattr(_blk.mix, "forget_floor"):
            _blk.mix.forget_floor.data.fill_(_floors[_i])
    o = torch.optim.AdamW(m.parameters(), lr=lr)
    return m, o
''')

md("## Train grid (6 arms x 1500 steps)")
code('''STEPS, BS = 1500, 32
g = torch.Generator().manual_seed(1234)
train_order = torch.randperm(len(train_files), generator=g).tolist()
val_idx = list(range(0, len(val_files), 8))[:128]

def train_arm(mixer_fn, ffn_hidden, lr, label):
    m, o = build_arm(mixer_fn, ffn_hidden, lr)
    xb0 = load_batch(train_files, train_order[:BS], device)
    with torch.no_grad():
        m(xb0)
    o.zero_grad(set_to_none=True); m.ar_loss(xb0).backward(); o.step()
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
    m.eval()
    with torch.no_grad():
        vl = 0.0
        for i in range(0, len(val_idx), BS):
            xb = load_batch(val_files, val_idx[i:i+BS], device)
            vl += m.ar_loss(xb).item() * len(val_idx[i:i+BS])
        vl /= len(val_idx)
    m.train()
    print(f"{label}: val {vl:.4f} | {peak_a:.0f}/{peak_r:.0f} MB | {ms:.1f} ms/step",
          flush=True)
    del m, o
    gc.collect(); torch.cuda.empty_cache()
    return vl

results = {}
for lr in LRS:
    tag = f"lr{lr:.0e}"
    print(f"\\n=== {tag} ===", flush=True)
    v = train_arm(lambda d, h: SelectiveSegmentedStateV8AFused(d, state_dim=256),
                  FFN_V8A, lr, f"V8-A {tag}")
    a = train_arm(lambda d, h: CausalSelfAttention(d, h),
                  FFN_ATTN, lr, f"ATTN {tag}")
    results[tag] = (v, a)
    print(f"{tag}: gap (attn-v8a) = {a-v:+.4f}", flush=True)

print("\\n=== SUMMARY ===")
best_v = min(v for v, a in results.values())
best_a = min(a for v, a in results.values())
gap = best_v - best_a
print(f"V8-A best: {best_v:.4f} | ATTN best: {best_a:.4f} | gap: {gap:+.4f}")
print("Phase 1 baseline gap was +0.0476 (v8a-attn; positive = v8a worse)")
if abs(gap) < 0.02:
    print("GAP SUBSTANTIALLY CLOSED -> hyperparameter artifact (revise modality story)")
else:
    print("GAP PERSISTS -> tuning-artifact hypothesis not supported under this grid")

with open("/content/v13_results.txt", "w") as f:
    for tag, (v, a) in results.items():
        f.write(f"{tag}: v8a={v:.4f} attn={a:.4f} gap={a-v:+.4f}\\n")
    f.write(f"best_v8a={best_v:.4f}\\nbest_attn={best_a:.4f}\\n")
from google.colab import files
files.download("/content/v13_results.txt")
print("SAVE DONE")
''')

path = f"{WS}/linear_attention_v13_vision_tuning.ipynb"
with open(path, "w") as f:
    json.dump(NB, f)
print("written:", path, f"({len(NB['cells'])} cells)")
