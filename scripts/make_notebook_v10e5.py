"""Generate Colab notebook V10-E5: ActNN-style quantized activation storage.

E1 verdict (his): mathematically valid, memory-effective, REJECTED as a
production lever on throughput (+48.9%). His strategic implication: attack
the large activation tensors WITHOUT multiplying kernel launches. E5 is the
centerpiece: stochastic-rounding int8 storage of linear/GELU activations via
custom autograd Functions (qstore.py, CPU-verified: unbiased, fwd-exact,
grads close, 25.4% / 6.6% storage at 8/2-bit).

Decisions on the 4 open questions (his "do it" = my call, stated for
correction): (1) all-at-once scope (all nn.Linear + nn.GELU);
(2) per-channel scales; (3) torch.rand stochastic rounding (fuse later);
(3) 2-bit conditional on 8-bit passing E5.3.

Probes: E5.0 quantizer correctness (fast fail) -> E5.1 VRAM (gate -25%) ->
E5.2 500-step lockstep (gate dev <= 0.03) -> E5.3 1500-step re-gate
(conditional; final within 0.03 of fp32 ref and 1.319) ->
E5.4 2-bit probe (conditional on E5.3).
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
qstore_src = open(f"{WS}/qstore.py").read()

md("""# V10-E5 — ActNN-style quantized activation storage (8-bit probe)

E1's verdict: exact dynamics, −62.1% VRAM, rejected on throughput (+48.9%).
The strategic implication: cut activation VRAM **without multiplying kernel
launches**. E5 stores linear/GELU activations as int8 (stochastic rounding,
unbiased) via custom autograd Functions — 2 cheap elementwise kernels per
tensor, zero new matmuls, zero recompute passes. Targets: mixer ~408 MB,
FFN ~277 MB (E0 map). Scan kernel interior untouched (V9 closed that).

**Decisions** (made per his "do it", open to correction): all-at-once scope
(all nn.Linear + nn.GELU) · per-channel scales · torch.rand stochastic
rounding (Triton fusion later) · 2-bit only if 8-bit passes E5.3.

**Gates:** E5.0 correctness+unbiasedness (fast fail) · E5.1 peak ≤ −25% vs
761 MB · E5.2 500-step lockstep max val dev ≤ 0.03 · E5.3 1500-step final
within 0.03 of fp32 ref and 1.319 · E5.4 2-bit probe (conditional).
""")

code('''import os, urllib.request, time
import torch
import torch.nn as nn
import torch.nn.functional as F
assert torch.cuda.is_available(), "needs GPU"
device = "cuda"
def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(0)
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
''')

md("## 1. Data — campaign corpus A (vocab 104)")
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
print(f"vocab {vocab_a} | tokens {len(data_a)/1e6:.1f}M")
''')

md("## 2. Models + Triton kernels (byte-identical to E0/E1)")
code(cell2_src)

md("## 3. qstore — quantized autograd wrappers (CPU-verified pre-flight)")
code(qstore_src)

md("""## 4. Builders + lockstep data

`build_v8a(seed, quantized)`: identical init both arms; quantization applied
after build, optimizer constructed after. 1500 pre-generated batch-32s, one
seeded order, shared by both arms. Fixed 20-batch val set.
""")
code('''DIM, LAYERS, HEADS, SEQ, FFN_H = 256, 8, 8, 128, 766
N_STEPS = 1500

def build_v8a(seed, quantized=False, bits=8):
    set_seed(seed)
    m = TinyLM(vocab_a, DIM, LAYERS, HEADS, SEQ,
               mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
               ffn_hidden=FFN_H, newline_id=stoi_a.get("\\n")).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        _blk.mix.forget_floor.data.fill_(_floors[_i])
    if quantized:
        set_qbits(bits)
        nl, ng = quantize_model(m)
        print(f"quantized: {nl} linears + {ng} gelus -> int{bits}")
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    return m, o

def pregen(d, seq_len, bs, n, seed, split="train"):
    dd = d[:int(0.95*len(d))] if split == "train" else d[int(0.95*len(d)):]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        i = torch.randint(0, len(dd)-seq_len-1, (bs,), generator=g)
        out.append((torch.stack([dd[j:j+seq_len] for j in i]),
                    torch.stack([dd[j+1:j+seq_len+1] for j in i])))
    return out

train_batches = pregen(data_a, SEQ, 32, N_STEPS, 1234, "train")
val_batches = pregen(data_a, SEQ, 32, 20, 999, "val")
print(f"batches: train {len(train_batches)}, val {len(val_batches)}")

def val_loss(m):
    tot = 0.0
    with torch.no_grad():
        for xb, yb in val_batches:
            _, loss = m(xb.to(device), yb.to(device))
            tot += float(loss)
    return tot / len(val_batches)

def train_step(m, o, xb, yb):
    o.zero_grad(set_to_none=True)
    _, loss = m(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    o.step()
    return float(loss)
''')

md("""## E5.0 — Quantizer correctness on REAL activations (fast fail)

Roundtrip stats on captured linear inputs (mixer in_proj, FFN up, head) +
GELU input: max/mean abs err, mean signed err (≈ 0 = unbiased), clamp-hit %.
Separate-model one-step grad comparison vs fp32 (expect small, unbiased).
""")
code('''set_qbits(8)
m_probe, _ = build_v8a(0, quantized=False)
captured = {}
hooks = []
def _hook(name):
    def fn(mod, inp, out):
        captured[name] = inp[0].detach()
    return fn
targets = [("mixer_in", m_probe.blocks[0].mix.in_proj),
           ("ffn_up", m_probe.blocks[0].mlp[0]),
           ("head", m_probe.head)]
for name, mod in targets:
    hooks.append(mod.register_forward_hook(_hook(name)))
xb0, yb0 = train_batches[0][0].to(device), train_batches[0][1].to(device)
with torch.no_grad():
    m_probe(xb0)
for h in hooks:
    h.remove()
n_params_before = sum(p.numel() for p in m_probe.parameters())

ok = True
for name, act in captured.items():
    d, s, mt = quantize_stochastic(act, 8)
    r = dequantize(d, s, 8, mt)
    err = (r - act)
    mx, mn, ms = err.abs().max().item(), err.abs().mean().item(), err.mean().item()
    sc = act.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
    rel = (err.abs() / (act.abs() + sc)).max().item()
    print(f"{name:10s} shape {tuple(act.shape)}  max|e|={mx:.3e} mean|e|={mn:.3e} "
          f"signed={ms:.2e} maxrel={rel:.3f}")
    if abs(ms) > 5e-4:
        ok = False; print("  FAIL: biased!")
print("E5.0 unbiasedness:", "PASS" if ok else "FAIL")

# grad closeness: separate identical-init models, one step, same batch
mf, of = build_v8a(0, quantized=False)
mq, oq = build_v8a(0, quantized=True, bits=8)
with torch.no_grad():
    mf(xb0); mq(xb0)  # warmup (Triton compile)
mf, of = build_v8a(0, quantized=False)
mq, oq = build_v8a(0, quantized=True, bits=8)
train_step(mf, of, xb0, yb0)
train_step(mq, oq, xb0, yb0)
num = den = 0.0
for pf, pq in zip(mf.parameters(), mq.parameters()):
    num += (pf.grad - pq.grad).pow(2).sum().item()
    den += pf.grad.pow(2).sum().item()
rel_grad = (num / (den + 1e-12)) ** 0.5
print(f"E5.0 one-step grad rel-delta vs fp32: {rel_grad:.3e} (info; expect small, nonzero)")
''')

md("""## E5.1 — VRAM probe (gate: peak ≤ −25% vs 761 MB baseline)""")
code('''mq, oq = build_v8a(0, quantized=True, bits=8)
xb0, yb0 = train_batches[0][0].to(device), train_batches[0][1].to(device)
with torch.no_grad():
    mq(xb0)  # warmup outside measurement
torch.cuda.reset_peak_memory_stats()
for s in range(3):
    xb, yb = train_batches[s][0].to(device), train_batches[s][1].to(device)
    train_step(mq, oq, xb, yb)
peak_q = torch.cuda.max_memory_allocated() / 1e6
reduction = 100 * (761 - peak_q) / 761
print(f"E5.1 peak quantized: {peak_q:.0f} MB vs 761 MB baseline")
print(f"E5.1 reduction: {reduction:.1f}%  (gate >= 25%)")
e51_pass = reduction >= 25
''')

md("""## E5.2/E5.3 — 1500-step lockstep: quantized vs fp32

Same 1500 batches, same order, val every 100. E5.2 gate at step 500
(max val dev ≤ 0.03); E5.3 gate at 1500 (final within 0.03 of fp32 ref
and within 0.05 of the 1.319 V8-A anchor — looser externally since data
order differs from the original run).
""")
code('''results = {}
for tag, quant in (("fp32", False), ("q8", True)):
    m, o = build_v8a(0, quantized=quant, bits=8)
    xb0, yb0 = train_batches[0][0].to(device), train_batches[0][1].to(device)
    with torch.no_grad():
        m(xb0)  # warmup outside timing
    vals = [val_loss(m)]
    t0 = time.perf_counter()
    for s in range(N_STEPS):
        xb, yb = train_batches[s][0].to(device), train_batches[s][1].to(device)
        train_step(m, o, xb, yb)
        if (s + 1) % 100 == 0:
            vals.append(val_loss(m))
            print(f"[{tag}] step {s+1:4d} val {vals[-1]:.4f}", flush=True)
    dt = time.perf_counter() - t0
    results[tag] = {"vals": vals, "secs": dt}
    print(f"[{tag}] done {dt:.0f}s ({1000*dt/N_STEPS:.1f} ms/step)", flush=True)

vf, vq = results["fp32"]["vals"], results["q8"]["vals"]
dev500 = max(abs(a - b) for a, b in zip(vq[:6], vf[:6]))
dev1500 = abs(vq[-1] - vf[-1])
anchor = abs(vq[-1] - 1.319)
overhead = 100 * (results["q8"]["secs"] - results["fp32"]["secs"]) / results["fp32"]["secs"]
print(f"E5.2 max val dev @500: {dev500:.4f} (gate <= 0.03)")
print(f"E5.3 final q8 {vq[-1]:.4f} vs fp32 {vf[-1]:.4f}  dev {dev1500:.4f} (gate <= 0.03)")
print(f"E5.3 vs 1.319 anchor: {anchor:.4f} (gate <= 0.05)")
print(f"E5 cost: step overhead {overhead:+.1f}% (expect +3..8%; falsify > 15%)")
e52_pass = dev500 <= 0.03
e53_pass = e52_pass and dev1500 <= 0.03 and anchor <= 0.05 and overhead <= 15
''')

md("""## E5.4 — 2-bit probe (conditional on E5.3)

Correctness on real activations + 500-step lockstep probe at 2-bit.
""")
code('''e54 = {}
if e53_pass:
    set_qbits(2)
    m2, o2 = build_v8a(0, quantized=True, bits=2)
    xb0, yb0 = train_batches[0][0].to(device), train_batches[0][1].to(device)
    with torch.no_grad():
        m2(xb0)
    torch.cuda.reset_peak_memory_stats()
    for s in range(3):
        xb, yb = train_batches[s][0].to(device), train_batches[s][1].to(device)
        train_step(m2, o2, xb, yb)
    peak2 = torch.cuda.max_memory_allocated() / 1e6
    vals2, valsf = [], []
    m2, o2 = build_v8a(0, quantized=True, bits=2)
    mf2, of2 = build_v8a(0, quantized=False)
    for mm, oo in ((m2, o2), (mf2, of2)):
        with torch.no_grad():
            mm(xb0)
    for s in range(500):
        xb, yb = train_batches[s][0].to(device), train_batches[s][1].to(device)
        train_step(m2, o2, xb, yb)
        train_step(mf2, of2, xb, yb)
        if (s + 1) % 100 == 0:
            vals2.append(val_loss(m2)); valsf.append(val_loss(mf2))
            print(f"[2bit] step {s+1:4d} val {vals2[-1]:.4f} vs fp32 {valsf[-1]:.4f}", flush=True)
    dev2 = max(abs(a - b) for a, b in zip(vals2, valsf))
    e54 = {"peak2": peak2, "dev2": dev2,
           "pass": dev2 <= 0.05 and peak2 <= 0.5 * 761}
    print(f"E5.4 2-bit: peak {peak2:.0f} MB, max val dev {dev2:.4f} -> "
          f"{'PASS' if e54['pass'] else 'FAIL'}")
else:
    print("E5.4 SKIPPED (E5.3 did not pass)")
''')

md("## E5.5 — Verdict + save")
code('''print(f"GATE E5.0 unbiasedness        : {'PASS' if ok else 'FAIL'}")
print(f"GATE E5.1 peak reduction >=25% : {'PASS' if e51_pass else 'FAIL'} ({reduction:.1f}%)")
print(f"GATE E5.2 500-step dev <=0.03  : {'PASS' if e52_pass else 'FAIL'} ({dev500:.4f})")
print(f"GATE E5.3 1500-step re-gate    : {'PASS' if e53_pass else 'FAIL'}")
if e54:
    print(f"GATE E5.4 2-bit probe          : {'PASS' if e54['pass'] else 'FAIL'}")
e5_pass = ok and e51_pass and e52_pass and e53_pass
print("E5 VERDICT:", "PASS — quantized activation storage is a production lever"
      if e5_pass else "FAIL — see gate(s) above")
lines = [f"e50_unbiased={ok}", f"e50_rel_grad={rel_grad:.3e}",
         f"e51_peak_MB={peak_q:.0f}", f"e51_reduction_pct={reduction:.1f}",
         f"e52_dev500={dev500:.4f}", f"e53_dev1500={dev1500:.4f}",
         f"e53_anchor={anchor:.4f}", f"e53_overhead_pct={overhead:.1f}",
         f"e53_final_q8={vq[-1]:.4f}", f"e53_final_fp32={vf[-1]:.4f}"]
if e54:
    lines += [f"e54_peak2_MB={e54['peak2']:.0f}", f"e54_dev2={e54['dev2']:.4f}",
              f"e54_pass={e54['pass']}"]
lines.append(f"gates_pass={bool(e5_pass)}")
open("/content/e5_results.txt", "w").write("\\n".join(lines))
print(open("/content/e5_results.txt").read())
from google.colab import files
files.download("/content/e5_results.txt")
print("SAVE DONE")
''')

with open(f"{WS}/linear_attention_v10e5_actnn.ipynb", "w") as f:
    json.dump(NB, f)
print("notebook written:", f"{WS}/linear_attention_v10e5_actnn.ipynb",
      f"({len(NB['cells'])} cells)")
