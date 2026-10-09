"""Generate 4 V12 Phase 3 rung notebooks (one T per execution), per his spec:

1. One rung per notebook execution: T=256/529/1024/2025 (T=144 done x3).
2. Primary = compute-only on PRELOADED FIXED tensors (seed 42, identical batch
   for both arms). End-to-end (fresh load + step) as secondary.
3. Warmup (5 steps, excludes Triton compile) then 30 timed steps; report MEDIAN.
4. Both arms, same batch size (32), fp32, same T4, same procedure.
5. Save immediately: print verbatim + write file + download in the same cell.
6. OOM caught -> resource-limit observation. Disconnect (no output) ->
   infrastructure failure (classified from outside).
"""
import json
import sys

WS = "/home/hatch/workspace/linear-attention-lab"


def inline(path, old_doc, new_doc, old_imp, new_imp=""):
    src = open(f"{WS}/{path}").read().replace(old_doc, new_doc)
    assert old_imp in src, path
    return src.replace(old_imp, new_imp)


cell_models = "\n\n".join([
    inline("models.py",
           '"""Linear-recurrent replacement for Transformer self-attention.',
           '"""(inlined) Linear-recurrent replacement for Transformer self-attention.',
           "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
           "import math\nimport torch.nn as nn\nimport torch.nn.functional as F\n"),
    inline("triton_kernels.py",
           '"""V6: Triton-fused selective scan for the gated segmented state space.',
           '"""(inlined) V6/V7/V8-A: Triton-fused scans.',
           "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
           "import triton\nimport triton.language as tl\n"),
    inline("fused_mixer.py",
           '"""V11-D1 Path B: fused projection+scan via save-x-only autograd Function.',
           '"""(inlined) V11-D1 Path B: fused projection+scan.',
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
           "# (imports stripped)\n"),
    inline("v12_multimodal.py",
           '"""V12 multimodal: AR sequence modeling over vision/audio on the frozen V8-A stack.',
           '"""(inlined) V12 multimodal modules.',
           """import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Block  # frozen
""",
           "# (imports stripped)\n"),
])


def build_notebook(img_size, T, tag):
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

    md(f"""# V12 Phase 3 rung — T={T} (vision, {img_size}px)

One rung per execution. Primary: compute-only on preloaded FIXED batch
(seed 42, identical for both arms). Warmup 5 steps, then 30 timed steps,
median reported. End-to-end as secondary. OOM = resource limit; disconnect
(no output) = infrastructure failure.
""")

    code('''import os
import torch
assert torch.cuda.is_available(), "needs GPU"
device = "cuda"
def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(0)
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
''')

    code('''import urllib.request, tarfile, glob
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
def _imgs(sub):
    return sorted(glob.glob(f"{BASE}/{sub}/*/*.JPEG") +
                  glob.glob(f"{BASE}/{sub}/*/*.jpg"))
train_files = _imgs("train")
print(f"train {len(train_files)}")
assert len(train_files) > 9000
''')

    code(cell_models)

    code(f'''import time, gc, random, statistics
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

DIM, LAYERS, HEADS, PATCH, FFN_V8A = 256, 8, 8, 8, 766
IMG, T, BS, TAG = {img_size}, {T}, 32, "{tag}"

_orig_block_forward = Block.forward
def _block_forward_ckpt(self, x, reset=None):
    x = x + self.mix(self.ln1(x), reset)
    x = x + _ckpt.checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
    return x
Block.forward = _block_forward_ckpt

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_batch(files, idx, img_size, device):
    imgs = []
    for i in idx:
        im = Image.open(files[i]).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        imgs.append(t)
    x = torch.stack(imgs)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device)

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
print(f"mixer params/layer: v8a {{_pv}} vs attn {{_pa}} -> ffn {{FFN_V8A}} vs {{FFN_ATTN}}")

# FIXED preloaded batch (seed 42) — identical input for both arms
frng = random.Random(42)
fixed_idx = [frng.randrange(len(train_files)) for _ in range(BS)]
xb_fixed = load_batch(train_files, fixed_idx, IMG, device)
print(f"fixed batch: {{tuple(xb_fixed.shape)}}")

def bench(mixer_fn, ffn_hidden, label):
    """Primary: compute-only on fixed batch. Returns dict or OOM marker."""
    try:
        m, o = build(mixer_fn, 0, ffn_hidden)
        # warmup: 5 steps (excludes Triton compile + init from timing)
        for _ in range(5):
            o.zero_grad(set_to_none=True)
            m.ar_loss(xb_fixed).backward(); o.step()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        times = []
        for _ in range(30):
            o.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            m.ar_loss(xb_fixed).backward(); o.step()
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0))
        med = statistics.median(times)
        peak_a = torch.cuda.max_memory_allocated() / 1e6
        peak_r = torch.cuda.max_memory_reserved() / 1e6
        # secondary: end-to-end (fresh load + one step)
        e_idx = [random.randrange(len(train_files)) for _ in range(BS)]
        t0 = time.perf_counter()
        xb2 = load_batch(train_files, e_idx, IMG, device)
        o.zero_grad(set_to_none=True)
        m.ar_loss(xb2).backward(); o.step()
        torch.cuda.synchronize()
        e2e = 1000 * (time.perf_counter() - t0)
        del m, o; gc.collect(); torch.cuda.empty_cache()
        return {{"med_ms": f"{{med:.1f}}", "alloc": f"{{peak_a:.0f}}",
                 "res": f"{{peak_r:.0f}}", "e2e_ms": f"{{e2e:.0f}}"}}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect(); torch.cuda.empty_cache()
            return {{"OOM": True}}
        raise

results = {{}}
results["v8a"] = bench(lambda d, h: SelectiveSegmentedStateV8AFused(d, state_dim=256),
                       FFN_V8A, "V8-A")
results["attn"] = bench(lambda d, h: CausalSelfAttention(d, h), FFN_ATTN, "ATTN")

def fmt(r):
    return "OOM (resource limit)" if "OOM" in r else (
        f"{{r['med_ms']}} ms (median) alloc {{r['alloc']}} res {{r['res']}} | e2e {{r['e2e_ms']}} ms")
line = f"T={{T}} V8-A: {{fmt(results['v8a'])}} | ATTN: {{fmt(results['attn'])}}"
print(line, flush=True)
with open(f"/content/v12p3_{{TAG}}.txt", "w") as f:
    f.write(line + "\\n")
from google.colab import files
files.download(f"/content/v12p3_{{TAG}}.txt")
print(f"SAVE {{TAG}} DONE")
''')

    path = f"{WS}/linear_attention_v12p3_{tag}.ipynb"
    with open(path, "w") as f:
        json.dump(NB, f)
    return path, len([c for c in NB["cells"] if c["cell_type"] == "code"])


if __name__ == "__main__":
    import subprocess, tempfile, os
    os.environ.setdefault("PYTHONPATH", os.path.expanduser("~/workspace/.pylibs"))
    for img_size, T, tag in [(128, 256, "t256"), (184, 529, "t529"),
                             (256, 1024, "t1024"), (360, 2025, "t2025")]:
        path, n = build_notebook(img_size, T, tag)
        nb = json.load(open(path))
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        for i, c in enumerate(code_cells):
            compile("".join(c["source"]), f"<{tag}#{i}>", "exec")
        src = "\n".join("".join(c["source"]) for c in code_cells)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src); tpath = f.name
        r = subprocess.run(["python3", "-m", "pyflakes", tpath],
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": os.path.expanduser("~/workspace/.pylibs")})
        os.unlink(tpath)
        out = (r.stdout + r.stderr).strip()
        crit = [l for l in out.splitlines()
                if "undefined name" in l or "SyntaxError" in l or "invalid syntax" in l]
        assert not crit, (tag, crit)
        print(f"{tag}: {n} code cells, verified clean -> {path}")
    print("ALL 4 RUNG NOTEBOOKS VERIFIED")
