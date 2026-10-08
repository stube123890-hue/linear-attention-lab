"""Generate Colab notebook V9-A: replication & generalization.

V8-A (diagonal gated delta rule) took the crown at 1.319, single seed.
V9-A is the validation the writeup needs before any strong conclusion:
  1. multi-seed replication (5 seeds x V8-A + attention, 1500 steps)
  2. dataset generalization (enwik8 + text8, param match re-derived per dataset)
  3. length generalization (zero-shot T=256/512/1024/2048 + one T=512 training run)
Verdict cell applies the brief's confirm/overturn criteria.
His watch-items are baked in: kernel hard-STOP, state RMS logging,
pre-clip grad-norm logging (clip 1.0), NaN/Inf SystemExit, gate stats,
"param-diff gate" labeling, best-val tracking, no GitHub push.
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

md("""# V9-A — Replication & generalization

V8-A (diagonal gated delta-rule SSM) took the crown at **1.319** vs V7's
1.339 — on a single seed. The 0.020 margin sits near run-to-run noise, so
the crown is provisional until this validation lands. V9-A changes nothing
architectural; it re-runs the campaign's key results:

1. **Multi-seed replication** — 5 seeds x (V8-A + attention), 1500 steps,
   identical config to V8-A Run 1. All campaign gates on every seed.
2. **Dataset generalization** — enwik8 (byte vocab 256) + text8 (27-char
   vocab); the parameter match is re-derived per dataset (embedding rows
   change — the matched-params discipline is re-established, not assumed).
3. **Length generalization** — zero-shot eval of the T=128-trained models at
   T=256/512/1024/2048 (free), plus one token-matched T=512 training run
   (375 steps). An attention OOM where v8a fits is a result, not a failure.

Runtime estimate: ~1.5-2h on a Colab T4 (10 x 1500-step runs at ~5 min each
+ datasets + T=512 + evals).

**Confirm criterion** (brief §1a): mean(v8a) < mean(attn) with margin > 2x
pooled std, AND mean(v8a) < 1.339 - noise. Report mean +/- std for final
val, best val, and the step-250 checkpoint (early-learning claim).

**Overturn criterion**: if the v8a-V7 gap vanishes inside noise, or any seed
flips the v8a>attn ranking -> crown is provisional, V9 quality work is
premature. Dataset ranking flips or faster v8a degradation with T scope the
claims down to the current corpus/length — itself the honest result.
""")

md("## 0. Setup (+ Triton)")
code("""import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "V9-A needs the T4 GPU"

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

md("## 1. Data — three corpora (current + enwik8 + text8)")
code("""import os, urllib.request, zipfile
os.makedirs("data", exist_ok=True)

def dl(url, path):
    if not os.path.exists(path):
        print("downloading", path, flush=True)
        urllib.request.urlretrieve(url, path)
    return path

# --- corpus A: the campaign corpus (TinyShakespeare + Gutenberg pg11/84/1342)
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
nl_a = stoi_a.get("\\n")
assert nl_a is not None, "newline not in vocab A!"

# --- corpus B: enwik8 (byte-level, vocab 256)
p = dl("https://mattmahoney.net/dc/enwik8.zip", "data/enwik8.zip")
with zipfile.ZipFile(p) as z:
    z.extract("enwik8", "data")
raw_b = open("data/enwik8", "rb").read()
data_b = torch.tensor(list(raw_b), dtype=torch.long)  # bytes are the tokens
vocab_b, nl_b = 256, 10  # byte 10 == newline

# --- corpus C: text8 (27-char vocab; no newlines -> word-boundary reset)
p = dl("https://mattmahoney.net/dc/text8.zip", "data/text8.zip")
with zipfile.ZipFile(p) as z:
    z.extract("text8", "data")
text_c = open("data/text8", encoding="utf-8").read()
chars_c = sorted(set(text_c)); vocab_c = len(chars_c)
stoi_c = {c: i for i, c in enumerate(chars_c)}
data_c = torch.tensor([stoi_c[c] for c in text_c], dtype=torch.long)
# text8 has no newlines: use SPACE as the segment boundary so the reset
# mechanism stays active (documented adaptation, not a silent change).
nl_c = stoi_c.get(" ")
print(f"corpus A: {len(text_a)/1e6:.2f} MB vocab {vocab_a} nl_id={nl_a}")
print(f"corpus B (enwik8): {len(raw_b)/1e6:.2f} MB vocab {vocab_b} nl_id={nl_b}")
print(f"corpus C (text8): {len(text_c)/1e6:.2f} MB vocab {vocab_c} chars={''.join(chars_c)!r} boundary(space)_id={nl_c}")
assert vocab_c == 27, f"text8 vocab drift: {vocab_c}"

def split95(d):
    n = int(0.95 * len(d))
    return d[:n], d[n:]

DATA = {
    "A-corpus": (data_a, vocab_a, nl_a),
    "B-enwik8": (data_b, vocab_b, nl_b),
    "C-text8":  (data_c, vocab_c, nl_c),
}

def make_get_batch(d, seq_len, batch_size):
    def get_batch(split="train"):
        dd = d[:int(0.95 * len(d))] if split == "train" else d[int(0.95 * len(d)):]
        i = torch.randint(0, len(dd) - seq_len - 1, (batch_size,))
        x = torch.stack([dd[j:j+seq_len] for j in i]).to(device)
        y = torch.stack([dd[j+1:j+seq_len+1] for j in i]).to(device)
        return x, y
    return get_batch
""")

md("## 2. Models + Triton kernels (inlined, no imports)")
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
code(models_src + "\n\n" + kern_src)

md("## 3. KERNEL GATE — Triton delta kernel vs reference (HARD STOP on fail)")
code("""# The kernel is seed-independent: run the gate ONCE, hard-STOP the run if it fails.
set_seed(0)
B, T, S = 4, 128, 256
u = torch.randn(B, T, S, device=device)
ap = torch.randn(B, T, S, device=device)
bp = torch.randn(B, T, S, device=device)
kp = torch.randn(B, T, S, device=device)
keep = (torch.rand(B, T, device=device) > 0.3).float()

h_tri = triton_delta_scan(u, ap, bp, kp, keep)  # compiles on first call

kr = keep.unsqueeze(-1)
alpha = torch.sigmoid(ap); beta = torch.sigmoid(bp); kk = torch.sigmoid(kp)
v = u * torch.sigmoid(u)
a = kr * alpha * (1 - beta * kk * kk)
b = kr * beta * kk * v
h_scan = parallel_scan_ab(a, b)

h = torch.zeros(B, S, device=device); hs = []
for t in range(T):
    at = kr[:, t] * alpha[:, t] * (1 - beta[:, t] * kk[:, t] ** 2)
    bt = kr[:, t] * beta[:, t] * kk[:, t] * v[:, t]
    h = at * h + bt
    hs.append(h)
h_loop = torch.stack(hs, 1)

e_scan = float((h_tri - h_scan).abs().max())
e_loop = float((h_tri - h_loop).abs().max())
print(f"triton vs delta scan max delta: {e_scan:.2e}")
print(f"triton vs delta loop max delta: {e_loop:.2e}")

u2 = u.clone().requires_grad_(True); ap2 = ap.clone().requires_grad_(True)
bp2 = bp.clone().requires_grad_(True); kp2 = kp.clone().requires_grad_(True)
triton_delta_scan(u2, ap2, bp2, kp2, keep).pow(2).sum().backward()
u3 = u.clone().requires_grad_(True); ap3 = ap.clone().requires_grad_(True)
bp3 = bp.clone().requires_grad_(True); kp3 = kp.clone().requires_grad_(True)
al3 = torch.sigmoid(ap3); be3 = torch.sigmoid(bp3); kk3 = torch.sigmoid(kp3)
v3 = u3 * torch.sigmoid(u3)
a3 = kr * al3 * (1 - be3 * kk3 * kk3)
b3 = kr * be3 * kk3 * v3
parallel_scan_ab(a3, b3).pow(2).sum().backward()
errs = {}
for name, g2, g3 in [("du", u2.grad, u3.grad), ("dap", ap2.grad, ap3.grad),
                     ("dbp", bp2.grad, bp3.grad), ("dkp", kp2.grad, kp3.grad)]:
    errs[name] = float((g2 - g3).abs().max())
print("backward max deltas:", {k: f"{v:.2e}" for k, v in errs.items()})
if not (e_scan < 1e-6 and e_loop < 1e-6):
    raise SystemExit("KERNEL GATE FAILED (forward) -- STOPPING RUN")
if not all(v < 1e-5 for v in errs.values()):
    raise SystemExit("KERNEL GATE FAILED (backward) -- STOPPING RUN")
print("KERNEL GATE PASSED — fused delta kernel == reference math")
""")

md("## 4. Param-diff gate per dataset (HARD STOP if diff > 3000)")
code("""# The matched-params discipline is re-established PER DATASET, not assumed:
# embedding rows change with vocab, so the FFN width is re-solved for each
# corpus and the counted diff is verified by construction.
def count_params(m):
    return sum(p.numel() for p in m.parameters())

DIM, LAYERS, HEADS, SEQ_MAX = 256, 8, 8, 2048
# SEQ_MAX=2048 at construction so the zero-shot length eval (T up to 2048)
# needs no rebuild; training still uses T=128 windows. pos_emb rows are
# identical in both models, so the diff is unaffected.

def solve_ffn(vocab, newline_id):
    # Measure the diff once with both models at FFN 1024, then solve the width
    # analytically: per-block FFN params are exactly 513h+256 (Linear(256->h)
    # + Linear(h->256), both biased), so diff(h) = C + 8*513*h with C measured.
    # The slope is structural from the Block code; the intercept is measured,
    # not assumed.
    set_seed(0)
    a = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
               mixer_fn=lambda d, h: CausalSelfAttention(d, h),
               newline_id=newline_id).to(device)
    v = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
               mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
               ffn_hidden=1024, newline_id=newline_id).to(device)
    d1024 = count_params(a) - count_params(v)
    del a, v
    torch.cuda.empty_cache()
    h_star = int(round(1024 + d1024 / (8 * 513)))
    return h_star

FFN = {}
for dname, (d, vocab, nl_id) in DATA.items():
    h = solve_ffn(vocab, nl_id)
    # verify by construction on real tensors
    set_seed(0)
    attn_m = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
                    mixer_fn=lambda d_, h_: CausalSelfAttention(d_, h_),
                    newline_id=nl_id).to(device)
    v8a_m = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
                   mixer_fn=lambda d_, h_: SelectiveSegmentedStateV8A(d_, state_dim=256),
                   ffn_hidden=h, newline_id=nl_id).to(device)
    pa_c, pv_c = count_params(attn_m), count_params(v8a_m)
    diff = abs(pa_c - pv_c)
    print(f"{dname}: vocab={vocab} ffn_hidden={h} attention={pa_c} v8a={pv_c} diff={diff}")
    if diff > 3000:
        raise SystemExit(f"PARAM-DIFF GATE FAILED on {dname}: diff {diff} > 3000 -- STOPPING RUN")
    FFN[dname] = h
    del attn_m, v8a_m
    torch.cuda.empty_cache()
print("PARAM-DIFF GATE PASSED on all datasets (diff <= 3000)")

def build_models(vocab, newline_id, ffn_hidden, seed):
    set_seed(seed)
    attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
        mixer_fn=lambda d, h: CausalSelfAttention(d, h),
        newline_id=newline_id).to(device)
    v8a = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
        mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
        ffn_hidden=ffn_hidden, newline_id=newline_id).to(device)
    _floors = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(v8a.blocks):
        _blk.mix.forget_floor.data.fill_(_floors[_i])
    return {"attention": attention, "v8a-delta": v8a}
""")

md("## 5. Multi-seed replication (seeds 0-4, V8-A + attention, 1500 steps)")
code("""import time

SEEDS = range(5)
STEPS, EVAL, LR, BS, SEQ = 1500, 250, 3e-4, 32, 128
dA, vocabA, nlA = DATA["A-corpus"]
get_batch = make_get_batch(dA, SEQ, BS)
CKPT_STEPS = [1] + list(range(EVAL, STEPS + 1, EVAL))

def val_loss(m, gb, n=10):
    m.eval()
    with torch.no_grad():
        return sum(float(m(*gb("val"))[1]) for _ in range(n)) / n

def state_rms_norms(m):
    norms = []
    for blk in m.blocks:
        h = getattr(getattr(blk, 'mix', None), 'last_state', None)
        if h is not None:
            norms.append(float(torch.sqrt((h.float() ** 2).mean()).item()))
    return norms

def gate_stats(m, xb):
    # layer-averaged sigmoid gate means; alpha includes the forget floor.
    # attention has no gate_proj -> returns {} and is skipped.
    acc = {"alpha": [], "beta": [], "k": []}
    hooks = []
    for blk in m.blocks:
        mix = blk.mix
        if not hasattr(mix, "gate_proj"):
            continue
        def mk_alpha(mod, inp, out, _mix=mix):
            acc["alpha"].append(float(torch.sigmoid(out.float() + _mix.forget_floor).mean()))
        def mk_beta(mod, inp, out):
            acc["beta"].append(float(torch.sigmoid(out.float()).mean()))
        def mk_k(mod, inp, out):
            acc["k"].append(float(torch.sigmoid(out.float()).mean()))
        hooks += [mix.gate_proj.register_forward_hook(mk_alpha),
                  mix.write_proj.register_forward_hook(mk_beta),
                  mix.key_proj.register_forward_hook(mk_k)]
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(xb)
    for h_ in hooks:
        h_.remove()
    if was_training:
        m.train()
    return {k: sum(v) / len(v) for k, v in acc.items()} if acc["alpha"] else {}

def probe(m, gb):
    # lightweight post-train probe: NO optimizer steps (V4's re-eval corrupted
    # numbers by training inside the metrics cell -- never again).
    m.eval()
    with torch.no_grad():
        tloss = sum(float(m(*gb("train"))[1]) for _ in range(20)) / 20
        xb, _ = gb("train")
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(50):
            m(xb)
        torch.cuda.synchronize()
        tps = BS * SEQ / ((time.time() - t0) / 50)
    m.train()
    torch.cuda.reset_peak_memory_stats()
    xb, yb = gb("train")
    _, loss = m(xb, yb); m.zero_grad(); loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.empty_cache()
    return tloss, tps, peak

results = {}
for seed in SEEDS:
    print(f"===== SEED {seed} =====", flush=True)
    models = build_models(vocabA, nlA, FFN["A-corpus"], seed)
    opt = {n: torch.optim.AdamW(m.parameters(), lr=LR) for n, m in models.items()}
    val_hist = {n: [] for n in models}
    state_hist = {n: [] for n in models}
    grad_hist = {n: [] for n in models}
    last_gn = {}
    for step in range(1, STEPS + 1):
        for name, m in models.items():
            m.train()
            xb, yb = get_batch("train")
            _, loss = m(xb, yb)
            opt[name].zero_grad(); loss.backward()
            last_gn[name] = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            opt[name].step()
        if step % EVAL == 0 or step == 1:
            msg = f"step {step:5d}"
            for name, m in models.items():
                vl = val_loss(m, get_batch)
                val_hist[name].append(vl)
                sn = state_rms_norms(m)
                if sn:
                    state_hist[name].append((min(sn), sum(sn) / len(sn), max(sn)))
                grad_hist[name].append(last_gn[name])
                check_finite(f"seed {seed} step {step} {name} val_loss", [vl])
                check_finite(f"seed {seed} step {step} {name} state_norms", sn)
                check_finite(f"seed {seed} step {step} {name} grad_norm", [last_gn[name]])
                sn_str = f" state_rms[{min(sn):.2f}/{sum(sn)/len(sn):.2f}/{max(sn):.2f}]" if sn else ""
                xb0, _ = get_batch("val")
                gs = gate_stats(m, xb0)
                gs_str = (f" gates[a={gs['alpha']:.3f}/b={gs['beta']:.3f}/k={gs['k']:.3f}]"
                          if gs else "")
                msg += (f" | {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})"
                        f"{sn_str} grad_norm {last_gn[name]:.2f}{gs_str}")
            print(msg, flush=True)
    seed_res = {}
    for name, m in models.items():
        tloss, tps, peak = probe(m, get_batch)
        bi = min(range(len(val_hist[name])), key=lambda i: val_hist[name][i])
        seed_res[name] = {
            "final_val": val_hist[name][-1],
            "best_val": val_hist[name][bi],
            "best_step": CKPT_STEPS[bi],
            "step250_val": val_hist[name][1],
            "train_loss": tloss, "tok_s": tps, "vram_mb": peak,
            "val_hist": list(val_hist[name]),
        }
        print(f"seed {seed} {name}: final {val_hist[name][-1]:.3f} "
              f"best {val_hist[name][bi]:.3f}@{CKPT_STEPS[bi]} "
              f"train_loss {tloss:.3f} tok/s {tps:.0f} vram {peak:.0f}MB", flush=True)
    results[seed] = seed_res
    if seed == 0:
        torch.save(models["v8a-delta"].state_dict(), "/content/v9a_seed0_v8a.pt")
        torch.save(models["attention"].state_dict(), "/content/v9a_seed0_attn.pt")
        print("seed-0 weights saved for zero-shot eval")
    del models, opt
    torch.cuda.empty_cache()
print("MULTI-SEED REPLICATION DONE")
""")

md("## 6. Dataset generalization (enwik8 + text8, V8-A + attention, seed 0)")
code("""def train_pair(models, gb, steps, tag):
    # compact training loop with the campaign watch-items: pre-clip grad-norm
    # logging (clip 1.0 unchanged), NaN/Inf SystemExit guard, best-val tracking.
    opt = {n: torch.optim.AdamW(m.parameters(), lr=3e-4) for n, m in models.items()}
    val_hist = {n: [] for n in models}
    last_gn = {}
    for step in range(1, steps + 1):
        for name, m in models.items():
            m.train()
            xb, yb = gb("train")
            _, loss = m(xb, yb)
            opt[name].zero_grad(); loss.backward()
            last_gn[name] = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            opt[name].step()
        if step % EVAL == 0 or step == 1:
            msg = f"[{tag}] step {step:5d}"
            for name, m in models.items():
                vl = val_loss(m, gb)
                val_hist[name].append(vl)
                check_finite(f"{tag} step {step} {name}", [vl, last_gn[name]])
                msg += f" | {name}: {vl:.3f} (ppl {math.exp(vl):.1f}) gnorm {last_gn[name]:.2f}"
            print(msg, flush=True)
    return val_hist

ds_results = {}
for dname in ("B-enwik8", "C-text8"):
    d, vocab, nl_id = DATA[dname]
    print(f"===== {dname} (seed 0) =====", flush=True)
    gb = make_get_batch(d, SEQ, BS)
    models = build_models(vocab, nl_id, FFN[dname], seed=0)
    val_hist = train_pair(models, gb, STEPS, dname)
    dh = {}
    for name in models:
        bi = min(range(len(val_hist[name])), key=lambda i: val_hist[name][i])
        dh[name] = {"final_val": val_hist[name][-1], "best_val": val_hist[name][bi],
                    "best_step": CKPT_STEPS[bi], "step250_val": val_hist[name][1]}
        print(f"[{dname}] {name}: final {val_hist[name][-1]:.3f} "
              f"best {val_hist[name][bi]:.3f}@{CKPT_STEPS[bi]}", flush=True)
    ds_results[dname] = dh
    del models
    torch.cuda.empty_cache()
print("DATASET GENERALIZATION DONE")
""")

md("## 7. Zero-shot length generalization (T=256/512/1024/2048, seed-0 weights)")
code("""# Fresh models built at SEQ_MAX=2048 (no rebuild needed), loaded with the
# T=128-trained seed-0 weights. Both models share the same (documented)
# position-embedding caveat past T=128, so the v8a-vs-attention ranking at
# each T stays apples-to-apples.
models_zs = build_models(vocabA, nlA, FFN["A-corpus"], seed=0)
models_zs["v8a-delta"].load_state_dict(torch.load("/content/v9a_seed0_v8a.pt"))
models_zs["attention"].load_state_dict(torch.load("/content/v9a_seed0_attn.pt"))
dA_full = DATA["A-corpus"][0]
zs = {128: {n: round(results[0][n]["final_val"], 3) for n in ("attention", "v8a-delta")}}
print("=== ZERO-SHOT LENGTH EVAL (val loss, no grad) ===")
for T_ in (256, 512, 1024, 2048):
    bs_e = 8 if T_ <= 512 else 4
    gb = make_get_batch(dA_full, T_, bs_e)
    row = {}
    for name, m in models_zs.items():
        try:
            row[name] = round(val_loss(m, gb, n=6), 3)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                row[name] = "OOM"
                torch.cuda.empty_cache()
            else:
                raise
    zs[T_] = row
    print(f"T={T_:5d}: " + " | ".join(f"{n}: {v}" for n, v in row.items()), flush=True)
print("ZERO-SHOT DONE")
del models_zs
torch.cuda.empty_cache()
""")

md("## 8. Train at T=512 (token-matched: 375 steps)")
code("""# Token budget held constant: 375 steps x 32 x 512 == 1500 x 32 x 128.
# An attention OOM where v8a fits is recorded as a result, not a failure.
SEQ512, STEPS512 = 512, 375
gb512 = make_get_batch(dA, SEQ512, BS)
t512 = {}
for name in ("v8a-delta", "attention"):
    print(f"===== T=512 {name} =====", flush=True)
    try:
        ms = build_models(vocabA, nlA, FFN["A-corpus"], seed=0)
        m = ms[name]
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        for step in range(1, STEPS512 + 1):
            m.train()
            xb, yb = gb512("train")
            _, loss = m(xb, yb)
            opt.zero_grad(); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            opt.step()
            check_finite(f"T512 {name} step {step}", [float(loss), gn])
            if step % 125 == 0 or step == 1:
                print(f"  step {step}: loss {float(loss):.3f} gnorm {gn:.2f}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        m.train(); xb, yb = gb512("train")
        _, loss = m(xb, yb); m.zero_grad(); loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e6
        vl = val_loss(m, gb512)
        t512[name] = {"final_val": round(vl, 3), "vram_mb": round(peak)}
        print(f"T=512 {name}: final val {vl:.3f} peak {peak:.0f} MB", flush=True)
        del ms, m, opt
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            t512[name] = {"final_val": "OOM", "vram_mb": "OOM"}
            print(f"T=512 {name}: OUT OF MEMORY -- recorded as a result, not a failure", flush=True)
        else:
            raise
    torch.cuda.empty_cache()
print("T=512 TRAINING DONE")
""")

md("## 9. Verdict — confirm or overturn (criteria from the brief §1a)")
code("""import statistics
print("=== V9-A VERDICT ===")
v8 = [results[s]["v8a-delta"] for s in SEEDS]
at = [results[s]["attention"] for s in SEEDS]
def mean_std(xs):
    return sum(xs) / len(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0
for key, label in [("final_val", "final val"), ("best_val", "best val"),
                   ("step250_val", "step-250 val")]:
    mv, sv = mean_std([r[key] for r in v8])
    ma, sa = mean_std([r[key] for r in at])
    pooled = math.sqrt((sv ** 2 + sa ** 2) / 2)
    print(f"{label}: v8a {mv:.3f}+/-{sv:.3f} | attn {ma:.3f}+/-{sa:.3f} | "
          f"margin {ma - mv:.3f} (2xpooled={2 * pooled:.3f})")
mv_f, sv_f = mean_std([r["final_val"] for r in v8])
ma_f, sa_f = mean_std([r["final_val"] for r in at])
pooled_f = math.sqrt((sv_f ** 2 + sa_f ** 2) / 2)
margin_f = ma_f - mv_f
flips = [s for s in SEEDS
         if results[s]["v8a-delta"]["final_val"] >= results[s]["attention"]["final_val"]]
print(f"seeds with v8a>=attn (ranking flips): {flips if flips else 'none'}")
confirm = (margin_f > 2 * pooled_f) and (mv_f + pooled_f < 1.339)
if confirm and not flips:
    print("VERDICT: CONFIRM -- seeded mean holds the gap to attention (>2x pooled std),")
    print("         the v8a band sits below V7's 1.339, no seed flips the ranking.")
else:
    print("VERDICT: OVERTURN (crown provisional) --")
    if flips:
        print(f"  - ranking flipped on seeds {flips}")
    if not (margin_f > 2 * pooled_f):
        print("  - seeded margin <= 2x pooled std (gap inside noise)")
    if not (mv_f + pooled_f < 1.339):
        print("  - v8a band not below V7's 1.339 (gap to V7 inside noise)")
    print("  V9 quality work is premature; scope claims to the current evidence.")
print("--- dataset scope findings ---")
for dname, dh in ds_results.items():
    for name in ("attention", "v8a-delta"):
        print(f"{dname} {name}: final {dh[name]['final_val']:.3f} best {dh[name]['best_val']:.3f}")
    lead = "v8a-delta" if dh["v8a-delta"]["final_val"] < dh["attention"]["final_val"] else "attention"
    flag = "  (RANKING FLIP -- crown is corpus-specific)" if lead == "attention" else ""
    print(f"  -> {dname} leader: {lead}{flag}")
print("--- length scope findings (zero-shot val) ---")
for T_ in sorted(zs):
    print(f"T={T_:5d}: " + " | ".join(f"{n}: {v}" for n, v in zs[T_].items()))
print("--- T=512 token-matched training ---")
for name, r in t512.items():
    print(f"{name}: {r}")
print("VERDICT DONE")
""")

md("## 10. Save everything (triple-redundant)")
code("""lines = ["V9-A replication & generalization -- run record"]
for s in SEEDS:
    for name in ("attention", "v8a-delta"):
        r = results[s][name]
        lines.append(f"seed {s} {name}: final {r['final_val']:.3f} "
                     f"best {r['best_val']:.3f}@{r['best_step']} step250 {r['step250_val']:.3f}")
lines.append("--- datasets ---")
for dname, dh in ds_results.items():
    for name in ("attention", "v8a-delta"):
        lines.append(f"{dname} {name}: final {dh[name]['final_val']:.3f} best {dh[name]['best_val']:.3f}")
lines.append("--- zero-shot lengths ---")
for T_ in sorted(zs):
    lines.append(f"T={T_}: " + " ".join(f"{n}={v}" for n, v in zs[T_].items()))
lines.append("--- T=512 ---")
for name, r in t512.items():
    lines.append(f"T512 {name}: {r}")
open("/content/v9a_results.txt", "w").write("\\n".join(lines))
print(open("/content/v9a_results.txt").read())
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    import shutil, datetime
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    shutil.copy("/content/v9a_results.txt",
                f"/content/drive/MyDrive/v9a_results_{stamp}.txt")
    print("Drive copy saved")
except Exception as e:
    print("Drive copy skipped:", e)
try:
    from google.colab import files
    files.download("/content/v9a_results.txt")
    print("download triggered")
except Exception as e:
    print("download skipped:", e)
print("SAVE DONE -- download the .ipynb via File > Download > .ipynb")
""")

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v9a_replication.ipynb", "w") as f:
    json.dump(NB, f, indent=1)
print("V9-A notebook written")
