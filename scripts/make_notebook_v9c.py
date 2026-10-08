"""Generate Colab notebook V9-C: fused-backward recompute of (a_t)."""
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

md("""# V9-C — Fused-backward recompute: freeing the saved `a` tensor

V9-B attacks activation bytes via precision. V9-C attacks them via *not
storing*: the fused delta-scan backward currently saves the per-step
`a_t` tensor (B,T,256) fp32 for the backward pass. The backward can
recompute `a_{t+1}` elementwise from the already-stored `(ap, bp, kp, keep)`
instead — a handful of exp/sigmoid-equivalent ops per step, memory-bound-cheap.

Honest arithmetic (correcting the brief's ~60 MB): the backward saves `a`
alone — `b_t` was never saved (the bwd kernel rebuilds `v` from `u` inline).
At the training config that is 32x128x256x4B = **4 MB/layer**, x8 layers =
**~32 MB** freed. Target: step-time cost < 2%; falsifies (not worth it) if
recompute exceeds 5% of step time. Zero quality delta by construction —
the math is identical — verified by a 1e-6 grad-equivalence gate and a
500-step same-seed training run.
""")

md("## 0. Setup (+ Triton)")
code("""import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "V9-C needs the T4 GPU"

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

SEQ_, BS = 128, 32
def get_batch(split="train", bs=32):
    d = train_data if split == "train" else val_data
    i = torch.randint(0, len(d) - SEQ_ - 1, (bs,))
    x = torch.stack([d[j:j+SEQ_] for j in i]).to(device)
    y = torch.stack([d[j+1:j+SEQ_+1] for j in i]).to(device)
    return x, y
""")

md("## 2. Models + Triton kernels (inlined) + V9-C recompute variant")
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

V9C_EXTRA = '''
# ---- V9-C: recompute-backward variant (appended; triton_kernels.py untouched) ----
@triton.jit
def _delta_scan_bwd_recompute_kernel(dh_ptr, hs_ptr, u_ptr, ap_ptr, bp_ptr,
                           kp_ptr, keep_ptr, du_ptr, dap_ptr, dbp_ptr,
                           dkp_ptr, T, BLOCK: tl.constexpr):
    # Backward identical to _delta_scan_bwd_kernel, except a_{t+1} is
    # recomputed elementwise from (ap,bp,kp,keep) instead of loaded from a
    # saved (B,T,S) tensor. Frees 4 MB/layer at the training config.
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    bs = T * BLOCK
    bsp = (T + 1) * BLOCK
    dh_b = dh_ptr + b * bs
    hs_b = hs_ptr + b * bsp
    u_b = u_ptr + b * bs
    ap_b = ap_ptr + b * bs
    bp_b = bp_ptr + b * bs
    kp_b = kp_ptr + b * bs
    k_b = keep_ptr + b * T
    du_b = du_ptr + b * bs
    dap_b = dap_ptr + b * bs
    dbp_b = dbp_ptr + b * bs
    dkp_b = dkp_ptr + b * bs
    dnext = tl.zeros([BLOCK], dtype=tl.float32)      # D_{t+1}, SRAM-resident
    for ti in range(T):
        t = T - 1 - ti
        gout = tl.load(dh_b + t * BLOCK + offs).to(tl.float32)
        if t + 1 < T:                                # recompute a_{t+1}
            ap_n = tl.load(ap_b + (t + 1) * BLOCK + offs).to(tl.float32)
            bp_n = tl.load(bp_b + (t + 1) * BLOCK + offs).to(tl.float32)
            kp_n = tl.load(kp_b + (t + 1) * BLOCK + offs).to(tl.float32)
            kr_n = tl.load(k_b + (t + 1)).to(tl.float32)
            al_n = 1.0 / (1.0 + tl.exp(-ap_n))
            be_n = 1.0 / (1.0 + tl.exp(-bp_n))
            kk_n = 1.0 / (1.0 + tl.exp(-kp_n))
            a_next = kr_n * al_n * (1.0 - be_n * kk_n * kk_n)
        else:
            a_next = tl.zeros([BLOCK], dtype=tl.float32)
        D = gout + a_next * dnext
        h_prev = tl.load(hs_b + t * BLOCK + offs).to(tl.float32)
        u_t = tl.load(u_b + t * BLOCK + offs).to(tl.float32)
        ap_t = tl.load(ap_b + t * BLOCK + offs).to(tl.float32)
        bp_t = tl.load(bp_b + t * BLOCK + offs).to(tl.float32)
        kp_t = tl.load(kp_b + t * BLOCK + offs).to(tl.float32)
        kr = tl.load(k_b + t).to(tl.float32)
        alpha = 1.0 / (1.0 + tl.exp(-ap_t))
        beta = 1.0 / (1.0 + tl.exp(-bp_t))
        kk = 1.0 / (1.0 + tl.exp(-kp_t))
        su = 1.0 / (1.0 + tl.exp(-u_t))
        v = u_t * su
        silu_p = su * (1.0 + u_t * (1.0 - su))
        k2 = kk * kk
        omb = 1.0 - beta * k2
        tl.store(du_b + t * BLOCK + offs,
                 D * kr * beta * kk * silu_p)
        tl.store(dap_b + t * BLOCK + offs,
                 D * h_prev * kr * alpha * (1.0 - alpha) * omb)
        tl.store(dbp_b + t * BLOCK + offs,
                 D * kr * beta * (1.0 - beta) * (kk * v - alpha * k2 * h_prev))
        tl.store(dkp_b + t * BLOCK + offs,
                 D * kr * beta * kk * (1.0 - kk) * (v - 2.0 * alpha * kk * h_prev))
        dnext = D


class _DeltaScanRecomputeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, ap, bp, kp, keep):
        assert u.is_cuda and u.dtype == torch.float32, "V9-C: CUDA fp32"
        for t in (u, ap, bp, kp):
            assert t.shape == u.shape and t.dim() == 3
        assert keep.shape == u.shape[:2]
        assert u.shape[2] == BLOCK_S
        B, T, S = u.shape
        u, ap, bp, kp = u.contiguous(), ap.contiguous(), bp.contiguous(), kp.contiguous()
        keep = keep.contiguous()
        h = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        a = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        _delta_scan_fwd_kernel[(B,)](u, ap, bp, kp, keep, h, a, T,
                                     BLOCK=BLOCK_S, num_warps=4)
        # NOTE: `a` is deliberately NOT saved. The backward recomputes a_{t+1}
        # elementwise from (ap, bp, kp, keep). Frees one (B,T,S) fp32 tensor.
        del a
        ctx.save_for_backward(u, ap, bp, kp, keep, h)
        return h

    @staticmethod
    def backward(ctx, dh):
        u, ap, bp, kp, keep, h = ctx.saved_tensors
        B, T, S = u.shape
        dh = dh.contiguous()
        hs = torch.zeros(B, T + 1, S, device=u.device, dtype=torch.float32)
        hs[:, 1:] = h      # hs[t] = h_{t-1}
        du = torch.empty_like(u)
        dap = torch.empty_like(ap)
        dbp = torch.empty_like(bp)
        dkp = torch.empty_like(kp)
        _delta_scan_bwd_recompute_kernel[(B,)](dh, hs, u, ap, bp, kp, keep,
                                     du, dap, dbp, dkp, T,
                                     BLOCK=BLOCK_S, num_warps=4)
        return du, dap, dbp, dkp, None


def triton_delta_scan_recompute(u, ap, bp, kp, keep):
    # Fused delta scan with recompute-backward. CUDA -> new kernels above;
    # CPU -> differentiable reference (verification only; no recompute needed).
    if u.is_cuda:
        return _DeltaScanRecomputeFn.apply(u, ap, bp, kp, keep)
    return _delta_scan_cpu(u, ap, bp, kp, keep)


class SelectiveSegmentedStateV9C(SelectiveSegmentedStateV8A):
    # V9-C mixer: identical architecture AND parameters to V8-A; the scan
    # backward recomputes a_t instead of saving it.
    def forward(self, x, reset=None):
        u = self.in_proj(x).float().contiguous()
        ap = (self.gate_proj(x) + self.forget_floor).float().contiguous()
        bp = self.write_proj(x).float().contiguous()
        kp = self.key_proj(x).float().contiguous()
        og = self.out_gate_proj(x).float()
        if reset is not None:
            keep = (1.0 - reset).float().contiguous()
        else:
            keep = torch.ones(x.shape[0], x.shape[1],
                              device=x.device, dtype=torch.float32)
        h_seq = triton_delta_scan_recompute(u, ap, bp, kp, keep)
        self.last_state = h_seq.detach()  # stability logging only
        h_norm = h_seq / torch.sqrt(
            (h_seq ** 2).mean(dim=-1, keepdim=True) + self.eps)
        y = self.out_proj((torch.sigmoid(og) * h_norm).to(x.dtype))
        return self.dropout(y)
'''
code(models_src + "\n\n" + kern_src + "\n" + V9C_EXTRA)

md("## 3. Numerical gate — recompute-bwd grads == standard-bwd grads (< 1e-6)")
code("""print("=== V9-C NUMERICAL GATE: recompute-backward vs standard-backward ===")
set_seed(0)
B, T, S = 4, 128, 256
u = torch.randn(B, T, S, device=device)
ap = torch.randn(B, T, S, device=device)
bp = torch.randn(B, T, S, device=device)
kp = torch.randn(B, T, S, device=device)
keep = (torch.rand(B, T, device=device) > 0.3).float()

h_std = triton_delta_scan(u, ap, bp, kp, keep)          # compiles std kernels
h_rec = triton_delta_scan_recompute(u, ap, bp, kp, keep)  # compiles recompute bwd
e_fwd = float((h_std - h_rec).abs().max())
print(f"fwd recompute vs standard max delta: {e_fwd:.2e} (same fwd kernel; expect 0)")

def _grads(fn):
    uu = u.clone().requires_grad_(True); aa = ap.clone().requires_grad_(True)
    bb = bp.clone().requires_grad_(True); kk_ = kp.clone().requires_grad_(True)
    fn(uu, aa, bb, kk_, keep).pow(2).sum().backward()
    return [t.grad for t in (uu, aa, bb, kk_)]

g_std = _grads(triton_delta_scan)
g_rec = _grads(triton_delta_scan_recompute)
errs = {}
for name, (a, b) in zip(("du", "dap", "dbp", "dkp"), zip(g_std, g_rec)):
    errs[name] = float((a - b).abs().max())
print("recompute-vs-standard grad max deltas:",
      {k: f"{v:.2e}" for k, v in errs.items()})

# second anchor: recompute grads vs autograd through the differentiable
# reference math (the bar the original fp32 gate used)
u4 = u.clone().requires_grad_(True); ap4 = ap.clone().requires_grad_(True)
bp4 = bp.clone().requires_grad_(True); kp4 = kp.clone().requires_grad_(True)
_delta_scan_cpu(u4, ap4, bp4, kp4, keep).pow(2).sum().backward()
errs2 = {}
for name, (a, b) in zip(("du", "dap", "dbp", "dkp"),
                      zip(g_rec, (u4.grad, ap4.grad, bp4.grad, kp4.grad))):
    errs2[name] = float((a - b).abs().max())
print("recompute-vs-reference-math grad max deltas:",
      {k: f"{v:.2e}" for k, v in errs2.items()})

assert e_fwd < 1e-7, "V9-C GATE FAILED (fwd) -- STOPPING RUN"
assert all(v < 1e-6 for v in errs.values()), \\
    "V9-C GATE FAILED (recompute vs standard) -- STOPPING RUN"
assert all(v < 1e-5 for v in errs2.values()), \\
    "V9-C GATE FAILED (vs reference math) -- STOPPING RUN"
print("V9-C NUMERICAL GATE PASSED -- recompute backward == standard backward")
""")

md("## 4. VRAM + step-time delta (standard vs recompute, same weights)")
code("""import time
print("=== VRAM + STEP-TIME: standard-bwd vs recompute-bwd ===")
DIM, LAYERS, HEADS = 256, 8, 8

def _build(mixer_cls, seed):
    set_seed(seed)
    m = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
        mixer_fn=lambda d, h: mixer_cls(d, state_dim=256),
        ffn_hidden=766, newline_id=nl_id).to(device)
    _fl = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        _blk.mix.forget_floor.data.fill_(_fl[_i])
    return m

m_std = _build(SelectiveSegmentedStateV8A, 11)
m_rec = _build(SelectiveSegmentedStateV9C, 0)
m_rec.load_state_dict(m_std.state_dict())   # identical weights
opt_s = torch.optim.AdamW(m_std.parameters(), lr=3e-4)
opt_r = torch.optim.AdamW(m_rec.parameters(), lr=3e-4)

def train_step_timed(m, opt):
    m.train(); opt.zero_grad()
    xb, yb = get_batch("train")
    torch.cuda.synchronize(); t0 = time.time()
    _, loss = m(xb, yb); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000

for _ in range(5):   # warmup (kernels compile here)
    train_step_timed(m_std, opt_s); train_step_timed(m_rec, opt_r)

def peak_of(m, opt):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    train_step_timed(m, opt)
    a = torch.cuda.max_memory_allocated() / 1e6
    r = torch.cuda.max_memory_reserved() / 1e6
    torch.cuda.empty_cache()
    return a, r

a_s, r_s = peak_of(m_std, opt_s)
a_r, r_r = peak_of(m_rec, opt_r)
ms_s = sum(train_step_timed(m_std, opt_s) for _ in range(50)) / 50
ms_r = sum(train_step_timed(m_rec, opt_r) for _ in range(50)) / 50
print(f"standard : peak alloc {a_s:.0f} MB / reserved {r_s:.0f} MB | step {ms_s:.2f} ms")
print(f"recompute: peak alloc {a_r:.0f} MB / reserved {r_r:.0f} MB | step {ms_r:.2f} ms")
print(f"dVRAM alloc {a_r - a_s:+.0f} MB (expect ~-32: the saved `a` tensor, 4 MB/layer x8)")
dtime = (ms_r / ms_s - 1) * 100
print(f"step-time delta {dtime:+.1f}% (target <+2%; FALSIFIES the lever if >+5%)")
VRAM_SAVED_MB = a_s - a_r
TIME_DELTA_PCT = dtime
""")

md("## 5. Param-diff gate — v9c vs attention (diff <= 3000)")
code("""def count_params(m):
    return sum(p.numel() for p in m.parameters())

set_seed(0)
attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h), newline_id=nl_id).to(device)
v9c = _build(SelectiveSegmentedStateV9C, 0)
pa, pv = count_params(attention), count_params(v9c)
print(f"attention={pa} v9c={pv} diff={abs(pa-pv)}")
print("V9-C params identical to V8-A by construction (same __init__)")
assert abs(pa - pv) <= 3000, "PARAM-DIFF GATE FAILED -- STOPPING RUN"
print("PARAM-DIFF GATE PASSED (diff <= 3000)")
del attention
torch.cuda.empty_cache()
""")

md("## 6. Short training run — 500 steps, seed 0, lockstep v8a-std vs v9c")
code("""print("=== 500-STEP LOCKSTEP: v8a standard-bwd vs v9c recompute-bwd (seed 0) ===")
STEPS, EVAL = 500, 250

def state_rms_norms(m):
    norms = []
    for blk in m.blocks:
        h = getattr(getattr(blk, 'mix', None), 'last_state', None)
        if h is not None:
            norms.append(float(torch.sqrt((h.float() ** 2).mean()).item()))
    return norms

def val_loss(m, n=10):
    m.eval()
    with torch.no_grad():
        return sum(float(m(*get_batch("val"))[1]) for _ in range(n)) / n

ma = _build(SelectiveSegmentedStateV8A, 0)
mc = _build(SelectiveSegmentedStateV9C, 99)
mc.load_state_dict(ma.state_dict())   # identical init; only the backward differs
opt_a = torch.optim.AdamW(ma.parameters(), lr=3e-4)
opt_c = torch.optim.AdamW(mc.parameters(), lr=3e-4)
hist = {"v8a-std": [], "v9c-rec": []}
for step in range(1, STEPS + 1):
    xb, yb = get_batch("train")          # one batch, fed to BOTH models
    for name, m, opt in (("v8a-std", ma, opt_a), ("v9c-rec", mc, opt_c)):
        m.train(); opt.zero_grad()
        _, loss = m(xb, yb); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        check_finite(f"lockstep step {step} {name}", [float(loss), gn])
        opt.step()
    if step % EVAL == 0 or step == 1:
        va, vc = val_loss(ma), val_loss(mc)
        hist["v8a-std"].append(va); hist["v9c-rec"].append(vc)
        sn = sum(state_rms_norms(mc)) / 8
        print(f"step {step:5d} v8a-std {va:.4f} | v9c-rec {vc:.4f} "
              f"d={vc-va:+.2e} stateRMS {sn:.3f}", flush=True)

traj_d = max(abs(a - b) for a, b in zip(hist["v8a-std"], hist["v9c-rec"]))
final_c = hist["v9c-rec"][-1]
print(f"max trajectory deviation (same seed): {traj_d:.2e} (bar 1e-3)")
print(f"v9c final val @500: {final_c:.3f} (V8-A Run 1 reference @500: 1.586)")
assert traj_d <= 1e-3, "QUALITY GATE FAILED: recompute changed the trajectory -- STOPPING"
print("QUALITY GATE PASSED -- recompute backward is trajectory-identical (zero quality delta)")
if abs(final_c - 1.586) > 0.02:
    print(f"NOTE: v9c @500 ({final_c:.3f}) differs from the cross-seed 1.586 anchor "
          f"by >0.02 -- investigate, but the binding check is the same-seed identity above")
""")

md("## 7. Verdict")
code("""print("=== V9-C VERDICT ===")
print(f"numerical gate: recompute-bwd == standard-bwd < 1e-6  (PASSED in cell 3)")
print(f"VRAM saved: {VRAM_SAVED_MB:.0f} MB alloc (expect ~32)")
print(f"step-time delta: {TIME_DELTA_PCT:+.1f}% (target <+2%, falsify >+5%)")
print(f"trajectory: v9c == v8a-std to {traj_d:.2e} over 500 lockstep steps")
ok_vram = VRAM_SAVED_MB >= 20      # expect ~32; 20 is the don't-bother floor
ok_time = TIME_DELTA_PCT <= 5.0
ok_qual = traj_d <= 1e-3
print(f"VRAM bar (>=20 MB saved): {'OK' if ok_vram else 'MISS'}")
print(f"time bar (<=+5%): {'OK' if ok_time else 'MISS -- lever falsified'}")
print(f"quality bar (traj <=1e-3): {'OK' if ok_qual else 'MISS'}")
print("V9-C VERDICT:",
      "recompute ACCEPTED -- ~32 MB freed, no measurable cost"
      if (ok_vram and ok_time and ok_qual) else
      "recompute CONDITIONAL -- see flags above")
""")

md("## 8. Save everything (results file + Drive best-effort + download)")
code("""lines = ["V9-C fused-backward recompute -- run record",
         f"vram_saved_alloc_MB={VRAM_SAVED_MB:.0f}",
         f"step_time_delta_pct={TIME_DELTA_PCT:+.1f}",
         f"traj_dev={traj_d:.2e}",
         f"v9c final@500={final_c:.3f}"]
for i, s in enumerate([1, 250, 500]):
    lines.append(f"step {s}: v8a-std {hist['v8a-std'][i]:.4f} v9c-rec {hist['v9c-rec'][i]:.4f}")
open("/content/v9c_results.txt", "w").write("\\n".join(lines))
print(open("/content/v9c_results.txt").read())
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    import shutil, datetime
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    shutil.copy("/content/v9c_results.txt",
                f"/content/drive/MyDrive/v9c_results_{stamp}.txt")
    print("Drive copy saved")
except Exception as e:
    print("Drive copy skipped:", e)
try:
    from google.colab import files
    files.download("/content/v9c_results.txt")
    print("download triggered")
except Exception as e:
    print("download skipped:", e)
print("SAVE DONE -- download the .ipynb via File > Download > .ipynb")
""")

md("""## Reading V9-C

- **Numerical gate** (cell 3) is the whole experiment in one cell: if the
  recompute backward isn't grad-identical to < 1e-6, nothing downstream
  means anything — it hard-STOPs.
- **~32 MB, not ~60**: the brief estimated (a,b); the code only ever saved
  `a`. The header corrects the record; the measured number rules.
- **Quality** is proven by same-seed trajectory identity (bit-identical
  math), not by beating a cross-seed anchor — the 1.586 line is context.
- If step-time delta lands between +2% and +5%, the lever is *acceptable
  but unimpressive*; above +5% it is falsified per the brief.
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v9c_fused_bwd.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("V9-C notebook written")
