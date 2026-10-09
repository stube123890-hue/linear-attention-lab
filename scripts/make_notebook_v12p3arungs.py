"""Generate 4 V12 Phase 3 AUDIO rung notebooks (one T per execution).

Same protocol as vision rungs: fixed preloaded batch (seed 42), 5 warmup +
30 timed steps, MEDIAN, immediate save. T = mel frames: 256/529/1024/2025.
Data: LibriSpeech dev-clean mel spectrograms (same as Phase 2 anchor).
"""
import json
import subprocess
import tempfile
import os

WS = "/home/hatch/workspace/linear-attention-lab"
os.environ.setdefault("PYTHONPATH", os.path.expanduser("~/workspace/.pylibs"))


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


def build_notebook(T, tag):
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

    md(f"""# V12 Phase 3 AUDIO rung — T={T} mel frames

One rung per execution. Primary: compute-only on preloaded FIXED mel batch
(seed 42, identical for both arms). Warmup 5 steps, then 30 timed steps,
median reported. OOM = resource limit; disconnect = infrastructure failure.
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

    code('''import torchaudio
from datasets import load_dataset

# LibriSpeech dev-clean mel (same pipeline as Phase 2 anchor)
# NOTE: datasets 4.x / huggingface_hub 1.x require the namespaced Hub ID
ds = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)
mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80,
                                              n_fft=400, hop_length=160)
print("dataset ready")
''')

    code(cell_models)

    code(f'''import time, gc, random, statistics

DIM, LAYERS, HEADS, FFN_V8A = 256, 8, 8, 766
T, BS, TAG, N_MELS = {T}, 32, "{tag}", 80

_orig_block_forward = Block.forward
def _block_forward_ckpt(self, x, reset=None):
    x = x + self.mix(self.ln1(x), reset)
    x = x + _ckpt.checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
    return x
Block.forward = _block_forward_ckpt

def get_mel_batch(n, seed):
    """Fixed mel batch: deterministic (seed). For long T, few utterances are
    long enough, so collect more candidates and pad shorter ones."""
    rng = random.Random(seed)
    cands = []
    for ex in ds:
        if len(cands) >= n * 10:
            break
        wav = torch.tensor(ex["audio"]["array"]).float()
        mel = mel_tf(wav).log1p().transpose(0, 1)  # [frames, 80]
        cands.append(mel)
    rng.shuffle(cands)
    out = []
    for mel in cands:
        if len(out) >= n:
            break
        L = mel.shape[0]
        if L >= T:
            out.append(mel[:T])
        elif L >= 64:
            pad = T - L
            out.append(torch.cat([mel, mel[-1:].repeat(pad, 1)], 0))
    while len(out) < n and cands:
        longest = max(cands, key=lambda m: m.shape[0])
        reps = (T + longest.shape[0] - 1) // longest.shape[0]
        out.append(longest.repeat(reps, 1)[:T])
    assert len(out) == n, f"only {{len(out)}} usable"
    xb = torch.stack(out).to(device)  # [B, T, 80] — what AudioEncoder expects
    # normalize per-utterance
    mu, sd = xb.mean((1, 2), keepdim=True), xb.std((1, 2), keepdim=True) + 1e-5
    return (xb - mu) / sd

def build(mixer, seed, ffn_hidden):
    set_seed(seed)
    enc = AudioEncoder(DIM, N_MELS)
    m = MultimodalAR(enc, DIM, LAYERS, HEADS, T, mixer, ffn_hidden=ffn_hidden).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        if hasattr(_blk.mix, "forget_floor"):
            _blk.mix.forget_floor.data.fill_(_floors[_i])
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    return m, o

import torch.nn as nn
import torch.utils.checkpoint as _ckpt  # noqa (used in patched forward)

torch.manual_seed(0)
_m0 = MultimodalAR(AudioEncoder(DIM, N_MELS), DIM, 1, HEADS, T,
                   lambda d, h: SelectiveSegmentedStateV8AFused(d, state_dim=256),
                   ffn_hidden=FFN_V8A)
_pv = sum(p.numel() for p in _m0.blocks[0].mix.parameters())
_ma = MultimodalAR(AudioEncoder(DIM, N_MELS), DIM, 1, HEADS, T,
                   lambda d, h: CausalSelfAttention(d, h), ffn_hidden=FFN_V8A)
_pa = sum(p.numel() for p in _ma.blocks[0].mix.parameters())
FFN_ATTN = match_ffn_hidden(FFN_V8A, DIM, _pv, _pa)
del _m0, _ma
print(f"mixer params/layer: v8a {{_pv}} vs attn {{_pa}} -> ffn {{FFN_V8A}} vs {{FFN_ATTN}}")

# FIXED preloaded mel batch (seed 42) — identical input for both arms
xb_fixed = get_mel_batch(BS, 42)
print(f"fixed mel batch: {{tuple(xb_fixed.shape)}}")

def bench(mixer_fn, ffn_hidden, label):
    """Control: xb_fixed shape asserted identical for both arms; warmup/
    compile overhead timed separately from the 30-step timed median."""
    assert tuple(xb_fixed.shape) == (BS, T, N_MELS), f"shape drift: {{tuple(xb_fixed.shape)}}"
    try:
        m, o = build(mixer_fn, 0, ffn_hidden)
        # warmup: step 1 includes Triton compile; steps 2-5 steady-state.
        # timed separately so compile/warmup overhead is DISTINGUISHED
        # from the reported median, never silently mixed in.
        warm = []
        for _ in range(5):
            o.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            m.ar_loss(xb_fixed).backward(); o.step()
            torch.cuda.synchronize()
            warm.append(1000 * (time.perf_counter() - t0))
        compile_ms, warmup_med = warm[0], statistics.median(warm[1:])
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
        del m, o; gc.collect(); torch.cuda.empty_cache()
        return {{"med_ms": f"{{med:.1f}}", "alloc": f"{{peak_a:.0f}}",
                 "res": f"{{peak_r:.0f}}",
                 "compile_ms": f"{{compile_ms:.0f}}",
                 "warmup_med_ms": f"{{warmup_med:.1f}}"}}
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
    if "OOM" in r:
        return "OOM (resource limit)"
    return (f"{{r['med_ms']}} ms (median of 30) alloc {{r['alloc']}} res {{r['res']}} "
            f"[compile {{r['compile_ms']}} ms | warmup med {{r['warmup_med_ms']}} ms]")
line = f"T={{T}} V8-A: {{fmt(results['v8a'])}} | ATTN: {{fmt(results['attn'])}}"
print(line, flush=True)
with open(f"/content/v12p3a_{{TAG}}.txt", "w") as f:
    f.write(line + "\\n")
from google.colab import files
files.download(f"/content/v12p3a_{{TAG}}.txt")
print(f"SAVE {{TAG}} DONE")
''')

    path = f"{WS}/linear_attention_v12p3a_{tag}.ipynb"
    with open(path, "w") as f:
        json.dump(NB, f)
    return path, len([c for c in NB["cells"] if c["cell_type"] == "code"])


if __name__ == "__main__":
    for T, tag in [(256, "t256"), (529, "t529"), (1024, "t1024"), (2025, "t2025")]:
        path, n = build_notebook(T, tag)
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
    print("ALL 4 AUDIO RUNG NOTEBOOKS VERIFIED")
