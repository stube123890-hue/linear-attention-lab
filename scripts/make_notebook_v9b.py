"""Generate Colab notebook V9-B: micro-batch + bf16 (VRAM reduction, no quality risk)."""
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

md("""# V9-B — Micro-batch + bf16: slashing VRAM with zero quality risk

V9-A confirmed the crown (seeded 1.312 vs 1.402, no flips, enwik8/text8 hold,
zero-shot flat to T=2048). V9-B attacks the one remaining tax: peak VRAM.

Key reframing (v9_study_brief §2a): the "fixed-size state" is 8 layers x 256 x
4B = **8 KB**. The 943 MB peak is ~840 MB of **activations held for backward**.
So VRAM reduction needs no architectural change:

1. **Micro-batch** (batch 32 -> 2x16 with grad accumulation): identical
   effective batch -> zero quality risk *by construction*; ~-350-400 MB.
2. **bf16 activations + scan state**: halves activation bytes; params stay
   fp32; the Triton kernel already loads with an explicit cast to fp32, so
   compute stays fp32 in SRAM and only the *stored* h/a go to bf16.

Targets: peak **<= 500 MB** (allocated AND reserved reported), final val
within noise of 1.319, train tok/s >= 58,848, O(T) scaling intact.
The 250-step probe gates bf16 on the full 5-clause trajectory-divergence
acceptance gate (not just final loss); on FAIL the run falls back to
micro-batch-only fp32 and still banks the VRAM win.
""")

md("## 0. Setup (+ Triton)")
code("""import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "V9-B needs the T4 GPU"
assert torch.cuda.is_bf16_supported(), "T4 must support bf16"

def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

import math
def check_finite(tag, vals):
    for v in vals:
        if math.isnan(v) or math.isinf(v):
            print(f"NUMERICAL FAULT at {tag}: non-finite value {v} -- STOPPING", flush=True)
            raise SystemExit(f"numerical fault at {tag}")
""")

md("## 1. Data (campaign corpus, vocab 104)")
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
        urllib.request.urlretrieve(url, p)
    with open(p, encoding="utf-8", errors="ignore") as f:
        parts.append(f.read())
text = "\\n".join(parts)
chars = sorted(set(text)); vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.95 * len(data)); train_data, val_data = data[:n], data[n:]
nl_id = stoi.get("\\n")
print(f"corpus {len(text)/1e6:.2f} MB, vocab {vocab}, newline_id={nl_id}")
assert nl_id is not None, "newline not in vocab!"

SEQ_, BS, MB = 128, 32, 16   # effective batch 32 = 2x micro-batch 16
def get_batch(split="train", bs=32):
    d = train_data if split == "train" else val_data
    i = torch.randint(0, len(d) - SEQ_ - 1, (bs,))
    x = torch.stack([d[j:j+SEQ_] for j in i]).to(device)
    y = torch.stack([d[j+1:j+SEQ_+1] for j in i]).to(device)
    return x, y
""")

md("## 2. Models + Triton kernels (inlined) + V9-B bf16 scan path")
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

V9B_EXTRA = '''
# ---- V9-B: bf16 scan path (appended; triton_kernels.py untouched) ----
class _DeltaScanBF16Fn(torch.autograd.Function):
    # bf16 scan: inputs may be bf16 (kernel loads cast to fp32, so compute
    # is fp32 in SRAM); h/a STORED as bf16 (the VRAM win); backward rebuilds
    # fp32 hs/apn from the bf16 stores (loads cast back). Params stay fp32.
    @staticmethod
    def forward(ctx, u, ap, bp, kp, keep):
        assert u.is_cuda, "V9-B bf16 scan needs CUDA"
        assert u.shape[2] == BLOCK_S
        B, T, S = u.shape
        u, ap, bp, kp = (t.contiguous() for t in (u, ap, bp, kp))
        keep = keep.contiguous()
        h32 = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        a32 = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        _delta_scan_fwd_kernel[(B,)](u, ap, bp, kp, keep, h32, a32, T,
                                     BLOCK=BLOCK_S, num_warps=4)
        h16 = h32.to(torch.bfloat16)
        a16 = a32.to(torch.bfloat16)
        ctx.save_for_backward(u, ap, bp, kp, keep, h16, a16)
        return h32

    @staticmethod
    def backward(ctx, dh):
        u, ap, bp, kp, keep, h16, a16 = ctx.saved_tensors
        B, T, S = u.shape
        # fp32 grad buffers: the bwd kernel is then exactly the fp32-gated
        # configuration (no mixed-dtype store semantics to rely on).
        dh = dh.float().contiguous()
        h = h16.float()
        a = a16.float()
        hs = torch.zeros(B, T + 1, S, device=u.device, dtype=torch.float32)
        hs[:, 1:] = h      # hs[t] = h_{t-1}
        apn = torch.zeros(B, T + 1, S, device=u.device, dtype=torch.float32)
        apn[:, :T] = a     # apn[t+1] = a_{t+1}
        du = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        dap = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        dbp = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        dkp = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        _delta_scan_bwd_kernel[(B,)](dh, hs, apn, u.float(), ap.float(),
                                     bp.float(), kp.float(), keep,
                                     du, dap, dbp, dkp, T,
                                     BLOCK=BLOCK_S, num_warps=4)
        return (du.to(u.dtype), dap.to(ap.dtype),
                dbp.to(bp.dtype), dkp.to(kp.dtype), None)


def triton_delta_scan_bf16(u, ap, bp, kp, keep):
    # Fused delta scan with bf16 stores. CUDA -> bf16 path above;
    # CPU -> differentiable fp32 reference (verification only).
    if u.is_cuda:
        return _DeltaScanBF16Fn.apply(u, ap, bp, kp, keep)
    ku, kap, kbp, kkp = (t.float() for t in (u, ap, bp, kp))
    return _delta_scan_cpu(ku, kap, kbp, kkp, keep.float())


class SelectiveSegmentedStateV9B(SelectiveSegmentedStateV8A):
    # V9-B mixer: identical architecture AND parameters to V8-A; only the
    # scan path stores (h, a) as bf16. Run under torch.autocast(bf16):
    # no .float() forcing -- dtypes flow from autocast.
    def forward(self, x, reset=None):
        u = self.in_proj(x).contiguous()
        ap = (self.gate_proj(x) + self.forget_floor).contiguous()
        bp = self.write_proj(x).contiguous()
        kp = self.key_proj(x).contiguous()
        og = self.out_gate_proj(x)
        if reset is not None:
            keep = (1.0 - reset).contiguous()
        else:
            keep = torch.ones(x.shape[0], x.shape[1], device=x.device,
                              dtype=u.dtype)
        h_seq = triton_delta_scan_bf16(u, ap, bp, kp, keep)
        self.last_state = h_seq.detach()  # stability logging only
        h_norm = h_seq / torch.sqrt(
            (h_seq ** 2).mean(dim=-1, keepdim=True) + self.eps)
        y = self.out_proj((torch.sigmoid(og) * h_norm).to(x.dtype))
        return self.dropout(y)
'''
code(models_src + "\n\n" + kern_src + "\n" + V9B_EXTRA)

md("## 3. Param-diff gate — v9b vs attention (diff <= 3000)")
code("""def count_params(m):
    return sum(p.numel() for p in m.parameters())

DIM, LAYERS, HEADS = 256, 8, 8
set_seed(0)
attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h), newline_id=nl_id).to(device)
v9b = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
    mixer_fn=lambda d, h: SelectiveSegmentedStateV9B(d, state_dim=256),
    ffn_hidden=766, newline_id=nl_id).to(device)
_floors = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
for _i, _blk in enumerate(v9b.blocks):
    _blk.mix.forget_floor.data.fill_(_floors[_i])
pa, pv = count_params(attention), count_params(v9b)
print(f"attention={pa} v9b={pv} diff={abs(pa-pv)}")
print("V9-B module params identical to V8-A by construction (same __init__)")
assert abs(pa - pv) <= 3000, "PARAM-DIFF GATE FAILED -- STOPPING RUN"
print("PARAM-DIFF GATE PASSED (diff <= 3000)")
del attention
torch.cuda.empty_cache()
""")

md("## 4. Micro-batch equivalence (fp32) — 2x16 accumulation == batch 32")
code("""print("=== MICRO-BATCH EQUIVALENCE: 2x16 accumulated grads vs batch-32 grads ===")
set_seed(7)
def _build_v8a(seed):
    set_seed(seed)
    m = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
        mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
        ffn_hidden=766, newline_id=nl_id).to(device)
    _fl = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        _blk.mix.forget_floor.data.fill_(_fl[_i])
    return m

m32 = _build_v8a(7)
macc = _build_v8a(0)
macc.load_state_dict(m32.state_dict())   # identical weights
xb32, yb32 = get_batch("train", 32)
# path 1: single batch-32 backward
m32.zero_grad()
_, loss32 = m32(xb32, yb32)
loss32.backward()
g32 = [p.grad.detach().clone() for p in m32.parameters()]
# path 2: 2x16 with loss/2 accumulation
macc.zero_grad()
for k in range(2):
    _, lossk = macc(xb32[k*MB:(k+1)*MB], yb32[k*MB:(k+1)*MB])
    (lossk / 2).backward()
gacc = [p.grad.detach().clone() for p in macc.parameters()]
mdiff = max((a - b).abs().max().item() for a, b in zip(g32, gacc))
print(f"max grad diff (2x16 accum vs batch 32): {mdiff:.2e} (bar 1e-6)")
assert mdiff < 1e-6, "MICRO-BATCH EQUIVALENCE FAILED -- STOPPING RUN"
print("MICRO-BATCH EQUIVALENCE PASSED -- accumulation is exact, zero quality risk by construction")
del m32, macc
torch.cuda.empty_cache()
""")

md("""## 5. bf16 kernel gate — why these bars (read first)

The gate checks **kernel == math**, not fp32 == bf16. The Triton kernel loads
bf16 inputs with an explicit cast to fp32, computes the whole recurrence in
fp32 SRAM registers, and rounds once on store — so the only bf16 effects here
are one input rounding and one output rounding. The manual reference mirrors
exactly that structure (bf16 I/O, fp32 compute, identical op order); residual
differences are fp32 op-ordering noise (~1e-6). The **1e-2 relative bar** is
~5x bf16 eps: loose enough to never false-trip, tight enough to catch any
transcription error in the kernel.

The backward kernel is byte-identical to the fp32-gated one (only the stored
h/a dtypes changed, and loads cast back to fp32) — its check is bf16-path
grads vs fp32-path grads on identical values (pure rounding), bar **5e-2**
for bf16's grad noise floor. The formulas themselves were proven at 1e-5 by
the fp32 gate. bf16's own accumulation epsilon is the *probe's* business,
not the gate's.
""")
code("""def bf16_kernel_fidelity(B=4, T=128, S=256, verbose=True):
    # Triton kernel (bf16 I/O) vs manual bf16-structured recurrence.
    # Returns (rel_fwd, rel_bwd_max).
    u16 = torch.randn(B, T, S, device=device, dtype=torch.bfloat16)
    ap16 = torch.randn(B, T, S, device=device, dtype=torch.bfloat16)
    bp16 = torch.randn(B, T, S, device=device, dtype=torch.bfloat16)
    kp16 = torch.randn(B, T, S, device=device, dtype=torch.bfloat16)
    keep = (torch.rand(B, T, device=device) > 0.3).float()
    # Triton path: bf16 in -> fp32 compute -> bf16 out
    h32 = torch.empty(B, T, S, device=device, dtype=torch.float32)
    a32 = torch.empty(B, T, S, device=device, dtype=torch.float32)
    _delta_scan_fwd_kernel[(B,)](u16, ap16, bp16, kp16, keep, h32, a32, T,
                                 BLOCK=256, num_warps=4)
    h_tri = h32.to(torch.bfloat16)
    # Reference: same math, bf16 I/O rounding, fp32 compute, same op order
    kr = keep.unsqueeze(-1)
    uf, apf, bpf, kpf = u16.float(), ap16.float(), bp16.float(), kp16.float()
    alpha = torch.sigmoid(apf); beta = torch.sigmoid(bpf); kk = torch.sigmoid(kpf)
    su = torch.sigmoid(uf); v = uf * su
    h = torch.zeros(B, S, device=device); hs = []
    for t in range(T):
        at = kr[:, t] * alpha[:, t] * (1.0 - beta[:, t] * kk[:, t] * kk[:, t])
        bt = kr[:, t] * beta[:, t] * kk[:, t] * v[:, t]
        h = at * h + bt
        hs.append(h.to(torch.bfloat16))
    h_ref = torch.stack(hs, 1)
    denom = h_ref.float().abs().max().clamp(min=1e-6)
    rel_fwd = float((h_tri.float() - h_ref.float()).abs().max() / denom)
    # Backward: bf16-path grads vs fp32-path grads, identical values
    def _grads(use_bf16):
        dt = torch.bfloat16 if use_bf16 else torch.float32
        uu = u16.to(dt).detach().requires_grad_(True)
        aa = ap16.to(dt).detach().requires_grad_(True)
        bb = bp16.to(dt).detach().requires_grad_(True)
        kk_ = kp16.to(dt).detach().requires_grad_(True)
        fn = triton_delta_scan_bf16 if use_bf16 else triton_delta_scan
        fn(uu, aa, bb, kk_, keep).pow(2).sum().backward()
        return [t.grad.float().clone() for t in (uu, aa, bb, kk_)]
    g_b = _grads(True); g_f = _grads(False)
    rel_bwd = max(float((a - b).abs().max() / b.abs().max().clamp(min=1e-6))
                  for a, b in zip(g_b, g_f))
    if verbose:
        print(f"bf16 kernel-vs-math: rel fwd err {rel_fwd:.2e} (bar 1e-2), "
              f"rel bwd err {rel_bwd:.2e} (bar 5e-2)")
    return rel_fwd, rel_bwd

rel_fwd, rel_bwd = bf16_kernel_fidelity()
if not rel_fwd < 1e-2:
    raise SystemExit("BF16 KERNEL GATE FAILED (forward) -- STOPPING RUN")
if not rel_bwd < 5e-2:
    raise SystemExit("BF16 KERNEL GATE FAILED (backward) -- STOPPING RUN")
print("BF16 KERNEL GATE PASSED -- bf16 I/O path through the Triton kernel is faithful")
""")

md("## 6. 250-step probe — matched seed, fp32 ref vs bf16, both 2x16")
code("""import time
print("=== 250-STEP PROBE (seed 42; fp32 ref vs bf16; identical batches) ===")

def build_pair(seed):
    set_seed(seed)
    ref = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
        mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
        ffn_hidden=766, newline_id=nl_id).to(device)
    _fl = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(ref.blocks):
        _blk.mix.forget_floor.data.fill_(_fl[_i])
    tst = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
        mixer_fn=lambda d, h: SelectiveSegmentedStateV9B(d, state_dim=256),
        ffn_hidden=766, newline_id=nl_id).to(device)
    tst.load_state_dict(ref.state_dict())   # identical init; only precision differs
    return ref, tst

def accum_train_step(m, opt, xb32, yb32, use_amp):
    m.train(); opt.zero_grad()
    tot = 0.0
    for k in range(2):
        xb, yb = xb32[k*MB:(k+1)*MB], yb32[k*MB:(k+1)*MB]
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = m(xb, yb)
        else:
            _, loss = m(xb, yb)
        (loss / 2).backward()
        tot += float(loss) / 2
    gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    check_finite("probe accum step", [tot, gn])
    opt.step()
    return tot, gn

def state_rms_norms(m):
    norms = []
    for blk in m.blocks:
        h = getattr(getattr(blk, 'mix', None), 'last_state', None)
        if h is not None:
            norms.append(float(torch.sqrt((h.float() ** 2).mean()).item()))
    return norms

def timed_accum(m, use_amp, iters=10):
    # fwd+bwd timing only: no opt step, grads cleaned after
    m.train(); m.zero_grad()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        xb32, yb32 = get_batch("train", 32)
        for k in range(2):
            xb, yb = xb32[k*MB:(k+1)*MB], yb32[k*MB:(k+1)*MB]
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = m(xb, yb)
            else:
                _, loss = m(xb, yb)
            (loss / 2).backward()
        m.zero_grad()
    torch.cuda.synchronize()
    return iters * 32 * SEQ_ / (time.time() - t0)

def peak_mem(m, use_amp):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m.train(); m.zero_grad()
    xb32, yb32 = get_batch("train", 32)
    for k in range(2):
        xb, yb = xb32[k*MB:(k+1)*MB], yb32[k*MB:(k+1)*MB]
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = m(xb, yb)
        else:
            _, loss = m(xb, yb)
        (loss / 2).backward()
    m.zero_grad()
    alloc = torch.cuda.max_memory_allocated() / 1e6
    reserved = torch.cuda.max_memory_reserved() / 1e6
    torch.cuda.empty_cache()
    return alloc, reserved

VAL_BATCHES = [get_batch("val", 32) for _ in range(10)]
TRN_BATCHES = [get_batch("train", 32) for _ in range(20)]

def probe_measure(m, use_amp, light=False):
    # NO optimizer steps (V4 lesson). Returns the 8 probe metrics.
    was_training = m.training; m.eval()
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vl = sum(float(m(xb, yb)[1]) for xb, yb in VAL_BATCHES) / len(VAL_BATCHES)
                tloss_acc = sum(float(m(xb, yb)[1]) for xb, yb in TRN_BATCHES) / len(TRN_BATCHES)
        else:
            vl = sum(float(m(xb, yb)[1]) for xb, yb in VAL_BATCHES) / len(VAL_BATCHES)
            tloss_acc = sum(float(m(xb, yb)[1]) for xb, yb in TRN_BATCHES) / len(TRN_BATCHES)
        sn = float(torch.tensor(state_rms_norms(m)).mean())
    m.train(); m.zero_grad()
    xb0, yb0 = TRN_BATCHES[0]
    for k in range(2):
        xb, yb = xb0[k*MB:(k+1)*MB], yb0[k*MB:(k+1)*MB]
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = m(xb, yb)
        else:
            _, loss = m(xb, yb)
        (loss / 2).backward()
    gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    m.zero_grad()
    out = {"val": vl, "train": tloss_acc, "snorm": sn, "gnorm": gn,
           "kfwd": float("nan"), "kbwd": float("nan"),
           "tps": float("nan"), "mem_alloc": float("nan"), "mem_reserved": float("nan")}
    if not light:
        kfw, kbw = bf16_kernel_fidelity(verbose=False)
        out["kfwd"], out["kbwd"] = kfw, kbw
        out["tps"] = timed_accum(m, use_amp)
        out["mem_alloc"], out["mem_reserved"] = peak_mem(m, use_amp)
    if was_training:
        m.train()
    return out

ref, tst = build_pair(42)
opt_r = torch.optim.AdamW(ref.parameters(), lr=3e-4)
opt_b = torch.optim.AdamW(tst.parameters(), lr=3e-4)
probe = {}
for step in range(1, 251):
    xb32, yb32 = get_batch("train", 32)   # one batch, fed to BOTH models
    accum_train_step(ref, opt_r, xb32, yb32, use_amp=False)
    accum_train_step(tst, opt_b, xb32, yb32, use_amp=True)
    if step in (1, 250):
        mr = probe_measure(ref, False); mb = probe_measure(tst, True)
        probe[step] = (mr, mb)
        print(f"[probe] step {step:3d} dval={mb['val']-mr['val']:+.4f} "
              f"dtrain={mb['train']-mr['train']:+.4f} dsnorm={mb['snorm']-mr['snorm']:+.4f} "
              f"dgnorm={mb['gnorm']-mr['gnorm']:+.4f} kfwd={mb['kfwd']:.2e} kbwd={mb['kbwd']:.2e} "
              f"tps {mr['tps']:.0f}->{mb['tps']:.0f} "
              f"alloc {mr['mem_alloc']:.0f}->{mb['mem_alloc']:.0f}MB "
              f"rsvd {mr['mem_reserved']:.0f}->{mb['mem_reserved']:.0f}MB", flush=True)
print("PROBE TRAINING DONE")
""")

md("## 7. Probe verdict -> full 1500-step run, or micro-batch-only fallback")
code("""def apply_probe_gate(probe):
    # 5-clause trajectory-divergence acceptance gate (his 13:49 correction):
    # trajectory, not just final loss. Probe evaluates at steps {1, 250};
    # clause (b) uses a probe approximation (b') -- the full 6/7 version
    # runs on the 7-checkpoint full run below.
    trips = []
    steps = sorted(probe)
    dval = {s: probe[s][1]["val"] - probe[s][0]["val"] for s in steps}
    if max(abs(v) for v in dval.values()) > 0.02:                       # (a)
        trips.append("(a) val-loss |bf16-fp32| > 0.02")
    s1, sL = steps[0], steps[-1]                                        # (b')
    if dval[s1] != 0 and (dval[s1] > 0) == (dval[sL] > 0) \\
       and abs(dval[sL]) > abs(dval[s1]) and abs(dval[sL]) > 0.005:
        trips.append("(b') consistent growing one-sided drift")
    for s in steps:                                                     # (c)
        mr, mb = probe[s]
        if abs(mb["snorm"] - mr["snorm"]) / max(mr["snorm"], 1e-9) > 0.10:
            trips.append(f"(c) state-norm rel drift >10% @step {s}")
    if max(probe[s][1]["gnorm"] / max(probe[s][0]["gnorm"], 1e-9)      # (d)
           for s in steps) > 1.5:
        trips.append("(d) grad-norm ratio >1.5x")
    for s in steps:                                                     # (e)
        for tag, m_ in (("fp32", probe[s][0]), ("bf16", probe[s][1])):
            for k in ("val", "train", "snorm", "gnorm"):
                v = m_[k]
                if math.isnan(v) or math.isinf(v):
                    trips.append(f"(e) non-finite {k} ({tag}) @step {s}")
    return trips

trips = apply_probe_gate(probe)
PROBE_PASS = not trips
print("PROBE VERDICT:", "PASS -> full 1500-step bf16 run" if PROBE_PASS
      else f"FAIL -- tripped clauses: {trips}")
print("FALLBACK PLAN: micro-batch-only fp32 (still banks the VRAM win)" if not PROBE_PASS else "")

STEPS, EVAL = 1500, 250
CKPTS = [1] + list(range(EVAL, STEPS + 1, EVAL))
full = {}

if PROBE_PASS:
    del ref, tst, opt_r, opt_b
    torch.cuda.empty_cache()
    ref2, tst2 = build_pair(123)
    opt_r2 = torch.optim.AdamW(ref2.parameters(), lr=3e-4)
    opt_b2 = torch.optim.AdamW(tst2.parameters(), lr=3e-4)
    hist = {"fp32": [], "bf16": []}
    gnh = {"fp32": [], "bf16": []}
    snh = {"fp32": [], "bf16": []}
    for step in range(1, STEPS + 1):
        xb32, yb32 = get_batch("train", 32)
        _, gn_r = accum_train_step(ref2, opt_r2, xb32, yb32, use_amp=False)
        _, gn_b = accum_train_step(tst2, opt_b2, xb32, yb32, use_amp=True)
        if step in CKPTS:
            light = step != STEPS
            mr = probe_measure(ref2, False, light=light)
            mb = probe_measure(tst2, True, light=light)
            hist["fp32"].append(mr["val"]); hist["bf16"].append(mb["val"])
            gnh["fp32"].append(mr["gnorm"]); gnh["bf16"].append(mb["gnorm"])
            snh["fp32"].append(mr["snorm"]); snh["bf16"].append(mb["snorm"])
            check_finite(f"full step {step}", [mr["val"], mb["val"], gn_r, gn_b])
            print(f"[full] step {step:5d} fp32 {mr['val']:.3f} | bf16 {mb['val']:.3f} "
                  f"d={mb['val']-mr['val']:+.4f} gnorm {gn_r:.2f}/{gn_b:.2f}", flush=True)
    alloc_b, rsvd_b = mb["mem_alloc"], mb["mem_reserved"]
    tps_b = mb["tps"]
    final_b = hist["bf16"][-1]
    full = {"hist": hist, "gnh": gnh, "snh": snh, "alloc": alloc_b,
            "reserved": rsvd_b, "tps": tps_b, "final": final_b}
    print(f"FULL bf16: final val {final_b:.3f} (bar: within 0.03 of 1.319)")
    print(f"peak alloc {alloc_b:.0f} MB / reserved {rsvd_b:.0f} MB (target: alloc <= 500)")
    print(f"train tok/s {tps_b:.0f} (bar: >= 58848)")
    print("FULL 1500-STEP BF16 RUN DONE")
else:
    print("FALLBACK: micro-batch-only fp32, 1500 steps, seed 123")
    set_seed(123)
    mfb = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
        mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
        ffn_hidden=766, newline_id=nl_id).to(device)
    _fl = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(mfb.blocks):
        _blk.mix.forget_floor.data.fill_(_fl[_i])
    opt_fb = torch.optim.AdamW(mfb.parameters(), lr=3e-4)
    fb_hist = []
    for step in range(1, STEPS + 1):
        xb32, yb32 = get_batch("train", 32)
        _, gn = accum_train_step(mfb, opt_fb, xb32, yb32, use_amp=False)
        if step in CKPTS:
            mr = probe_measure(mfb, False, light=True)
            fb_hist.append(mr["val"])
            check_finite(f"fallback step {step}", [mr["val"], gn])
            print(f"[fb] step {step:5d} val {mr['val']:.3f} gnorm {gn:.2f}", flush=True)
    alloc_f, rsvd_f = peak_mem(mfb, False)
    full = {"fallback_hist": fb_hist, "alloc": alloc_f, "reserved": rsvd_f,
            "final": fb_hist[-1]}
    print(f"FALLBACK DONE: final val {fb_hist[-1]:.3f}, "
          f"peak alloc {alloc_f:.0f} MB / reserved {rsvd_f:.0f} MB")
""")

md("## 8. O(T) spot-check + full-run trajectory gate (7 checkpoints)")
code("""print("=== O(T) SPOT-CHECK: v9b mixer, bf16, batch 8, fwd+bwd ===")
def bench_bf16(make, T, iters=30):
    m = make().train().to(device)
    x = torch.randn(8, T, 256, device=device)
    r = (torch.rand(8, T, device=device) < 0.3).float()
    for _ in range(5):
        m.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            m(x, r).sum().backward()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        m.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            m(x, r).sum().backward()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    torch.cuda.reset_peak_memory_stats(); m.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        m(x, r).sum().backward()
    peak = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.empty_cache()
    return ms, peak

for T in (512, 1024):
    ms, peak = bench_bf16(lambda: SelectiveSegmentedStateV9B(256, 256), T)
    ref_ms = "3.55" if T == 512 else "7.11"
    print(f"  T={T}: {ms:6.2f} ms/fwd+bwd | peak {peak:6.0f} MB  (V8-A fp32: {ref_ms} ms)")
print("SPOT-CHECK DONE")

if PROBE_PASS:
    print("=== FULL-RUN TRAJECTORY GATE (7 checkpoints) ===")
    dvals = [b - a for a, b in zip(full["hist"]["fp32"], full["hist"]["bf16"])]
    trips7 = []
    if max(abs(d) for d in dvals) > 0.02:
        trips7.append("(a) val-loss |bf16-fp32| > 0.02")
    pos = sum(1 for d in dvals if d > 0); neg = sum(1 for d in dvals if d < 0)
    if max(pos, neg) >= 6:
        trips7.append(f"(b) drift: {max(pos,neg)}/7 checkpoints one-sided")
    for i, s in enumerate(CKPTS):
        a, b = full["snh"]["fp32"][i], full["snh"]["bf16"][i]
        if abs(b - a) / max(a, 1e-9) > 0.10:
            trips7.append(f"(c) state-norm drift >10% @step {s}")
    if max(b / max(a, 1e-9) for a, b in
           zip(full["gnh"]["fp32"], full["gnh"]["bf16"])) > 1.5:
        trips7.append("(d) grad-norm ratio >1.5x")
    print("checkpoints:", CKPTS)
    print("dvals:", [f"{d:+.4f}" for d in dvals])
    print("7-CHECKPOINT GATE:", "PASS" if not trips7 else f"TRIPPED {trips7}")
    ok_vram = full["alloc"] <= 500
    ok_val = abs(full["final"] - 1.319) <= 0.03
    ok_tps = full["tps"] >= 58848
    print(f"targets: VRAM<=500MB alloc: {'OK' if ok_vram else 'MISS'} | "
          f"val~1.319: {'OK' if ok_val else 'MISS'} | tok/s>=58848: {'OK' if ok_tps else 'MISS'}")
    print("V9-B VERDICT:", "bf16 ACCEPTED -- all bars hold" if
          (not trips7 and ok_vram and ok_val and ok_tps)
          else "bf16 CONDITIONAL -- see flags above")
else:
    print("V9-B VERDICT: probe failed -- micro-batch-only fallback banked; "
          "bf16 rejected for this run")
""")

md("## 9. Save everything (results file + Drive best-effort + download)")
code("""lines = ["V9-B micro-batch + bf16 -- run record",
         f"probe: {'PASS' if PROBE_PASS else 'FAIL ' + str(trips)}"]
if PROBE_PASS:
    for i, s in enumerate(CKPTS):
        lines.append(f"ckpt {s}: fp32 {full['hist']['fp32'][i]:.3f} "
                     f"bf16 {full['hist']['bf16'][i]:.3f}")
    lines.append(f"final bf16 {full['final']:.3f} | peak alloc {full['alloc']:.0f} MB "
                 f"| reserved {full['reserved']:.0f} MB | tok/s {full['tps']:.0f}")
else:
    for i, s in enumerate(CKPTS):
        lines.append(f"ckpt {s}: fallback fp32 {full['fallback_hist'][i]:.3f}")
    lines.append(f"fallback final {full['final']:.3f} | peak alloc {full['alloc']:.0f} MB "
                 f"| reserved {full['reserved']:.0f} MB")
open("/content/v9b_results.txt", "w").write("\\n".join(lines))
print(open("/content/v9b_results.txt").read())
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    import shutil, datetime
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    shutil.copy("/content/v9b_results.txt",
                f"/content/drive/MyDrive/v9b_results_{stamp}.txt")
    print("Drive copy saved")
except Exception as e:
    print("Drive copy skipped:", e)
try:
    from google.colab import files
    files.download("/content/v9b_results.txt")
    print("download triggered")
except Exception as e:
    print("download skipped:", e)
print("SAVE DONE -- download the .ipynb via File > Download > .ipynb")
""")

md("""## Reading V9-B

- **Micro-batch equivalence** (cell 4) must pass first: it proves the 2x16
  accumulation is bit-exact vs batch 32 *before* bf16 enters the picture.
- **bf16 kernel gate** (cell 5) checks kernel==math at bf16 I/O, not fp32==bf16.
- **250-step probe** (cells 6-7) is the real bf16 test: 8 metrics as
  bf16-fp32 differences, 5-clause trajectory gate. A matching final loss
  with a diverged trajectory FAILS by design.
- On probe FAIL the notebook still delivers value: micro-batch-only fp32
  banks ~-400 MB with zero quality risk.
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v9b_bf16.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("V9-B notebook written")
