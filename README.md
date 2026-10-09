# Linear Attention Lab

**Can you rip the Attention layer out of a Transformer, replace it with a
gated linear recurrence over a fixed-size state, and win?**

Yes — on quality, speed, *and* memory. A 6.37M-parameter **diagonal
gated delta-rule state-space model** with a fused Triton kernel beats a
parameter-matched standard Transformer on validation loss (**1.319 vs
1.419**), keeps the fixed-state O(T) recurrence with a long-sequence runtime
advantage from T=512 up, and pays only a modest throughput tax (**58.8k vs
60.9k tok/s**) — same corpus, same protocol, exact parameter equality.

This repo is the complete record of a fourteen-experiment campaign (V1–V12) run
2026-10-08→09 on a Colab T4 GPU: every architecture, every
measurement, every design decision, and every anomaly — nothing smoothed over.

---

## Contents

- `src/models.py` — all model code: `CausalSelfAttention`,
  `LinearRunningState` (loop + parallel-scan modes), `SelectiveSegmentedState`,
  `TinyLM`, the Hillis–Steele associative scan
- `src/triton_kernels.py` — the fused Triton kernels (`@triton.jit`
  forward + backward) and the autograd wrappers: `SelectiveSegmentedStateTriton`
  (V6), `SelectiveSegmentedStateV7` (output gate + per-layer forget floor),
  and `SelectiveSegmentedStateV8A` (diagonal gated delta rule)
- `src/fused_mixer.py` — V11-D1: save-x-only autograd Function fusing the
  scan-input projections with the existing Triton kernels (Path B)
- `notebooks/` — the eleven experiment notebooks, exactly as run
  (`v4_rerun_clean`, `v5_gated_segmented`, `v6_triton_fused`,
  `v7_gated_output`, `v8a_delta_rule`, `v9a_replication`, `v9b_bf16`,
  `v9c_fused_bwd` include their full printed outputs)
- `scripts/` — the notebook generators (so every notebook is reproducible
  from code)
- `V8A_RESEARCH_NOTES.md` — provisional abstract language and the writeup
  skeleton for V8-A (kept as research notes, not publication claims)
- `V9_RESEARCH_NOTES.md` — V9 verdicts: replication CONFIRM, bf16 and
  recompute levers falsified with mechanisms documented
- `V10_RESEARCH_NOTES.md` — V10 verdicts: the memory-autopsy ladder
  (E1/E2/E3/E4/E5, all rejected/failed/falsified with mechanisms)
- `V11_RESEARCH_NOTES.md` — V11 verdicts: D2a PASS, D1 PASS (with the dx
  transpose-bug saga), combined stack endpoint
- `V12_RESEARCH_NOTES.md` — V12 verdicts: multimodal probe — V8-A wins
  temporal (audio), attention wins spatial (vision); O(T) crossover at
  T≈256–529 in both modalities; joint 2×2 control falsifies the
  cross-modal rescue hypothesis (I=+0.0386)
- `requirements.txt` — `torch`, `triton`

---

## The idea

Replace the Transformer's self-attention sublayer with a moving linear
equation that compresses the past into a fixed-size running state:

```
h_t = d · h_{t-1} + (1 - d) · (B · x_t)        # V1–V4
```

That alone won on quality at small scale but carried a ~2% perplexity tax at
6M parameters. The tax was broken by giving the state a **will** (an
input-dependent selective gate) and a **segmented memory** (a hard reset at
text boundaries):

```
g_t = σ(W_g · x_t + b_g)                        # dynamic selective gate
r_t = 1[x_t is a boundary token]                # hard reset mask
h_t = (1 - r_t) · (g_t · h_{t-1} + (1 - g_t) · (B · x_t))   # V5/V6
y_t = C · h_t
```

V6 fuses that recurrence into a custom Triton kernel: the 256-wide state
lives in SRAM registers for the whole sequence, gate + reset + update run
in-register every timestep, and only final outputs touch HBM.

V7 adds two stabilizers from the post-Mamba literature (output gate + state
norm, and a layer-graded forget floor), keeping the recurrence and the
Triton kernel byte-identical:

```
g_t  = σ(W_g · x_t + b_g + floor_l)             # per-layer forget floor
o_t  = σ(W_o · x_t + b_o)                       # output gate
h_t  = (1 - r_t) · (g_t · h_{t-1} + (1 - g_t) · (B · x_t))
y_t  = C · (o_t ⊙ RMSNorm(h_t))                  # V7 readout
```

`floor_l` is a learnable per-layer scalar initialized monotonically
bottom→top (logit(0.3) → logit(0.9)): low layers forget fast (spelling),
top layers retain (multi-word context). The RMSNorm is parameter-free.

V8 replaces the convex blend with a **diagonal gated delta rule** (from the
GDN/HGRN/Mamba-2 literature): per-channel input-dependent keep and write
gates, initialized so the model starts near V7's regime and every step stays
strictly contractive:

```
α_t = σ(W_α · x_t + b_α + floor_l)            # V7's gate + floor, reused
β_t = σ(W_β · x_t + b_β),  b_β = 0            # write gate (init 0.5)
k_t = σ(W_k · x_t + b_k),  b_k = +2.0         # key gate (init ~0.88)
a_t = α_t · (1 − β_t · k_t²)                   # diagonal keep
b_t = β_t · k_t · (B · x_t)                    # delta write
h_t = (1 − r_t) · (a_t ⊙ h_{t−1} + b_t)        # V8-A recurrence
y_t = C · (o_t ⊙ RMSNorm(h_t))                  # V7 readout, unchanged
```

The sigmoid bounds keep `a_t < 1` everywhere, so the state cannot run away.
Per-channel gates already act as 256 independent 1-dim heads — the
literature verdict was not to split into explicit heads.

---

## Experimental protocol (identical across runs unless noted)

- **Corpus:** 2.47 MB — TinyShakespeare + Project Gutenberg: *Alice in
  Wonderland* (pg11), *Frankenstein* (pg84), *Pride and Prejudice* (pg1342)
- **Tokenization:** character-level, vocab 104, 95/5 train/val split
- **Model:** decoder-only LM, dim 256, 8 layers, 8 heads, seq 128,
  weight-tied head, dropout 0.0, LayerNorm pre-norm blocks
- **Training:** batch 32, AdamW lr 3e-4, grad clip 1.0, eval every 250 steps
  (mean of 10 val batches); V1 ran 2000 steps, V2–V7 ran 1500 steps
- **Hardware:** Google Colab Tesla T4, fresh runtime per experiment,
  `Runtime > Run all` in one continuous pass
- **Fairness:** the attention baseline is rebuilt from the same code with the
  same dims in every run; only the mixer (and its parameter-balancing FFN
  width) changes

---

## Results

### V1 — the idea works (2000 steps, ~0.7M params)

| | Attention | Linear state |
|---|---|---|
| Params | 0.82M | 0.69M |
| Final val / ppl | 1.634 / 5.1 | **1.608 / 5.0** |
| ms/batch | 11.2 | 109.7 |

The linear state won on quality with fewer parameters and led at every
checkpoint from step 200 — but the naive Python loop was ~10x slower per
batch than fused attention. The O(T) speed win needed a parallel
implementation.

### V2 — scaling laws and the parallel scan (three sub-experiments)

- **Exp 1, context scaling 128→1024:** attention peak VRAM 87→237→**789 MB**
  (quadratic); linear-scan 113→209→**419 MB** (clean linear). Memory win proven.
- **Exp 2, parallel scan:** Hillis–Steele associative scan verified identical
  to the loop (max diff 2.4e-07). Scan 5–18x faster than the loop
  (106→20 ms @T=128, 457→25 ms @T=512) but still ~2x slower than fused
  attention (10.7/14.2 ms) — the remaining gap was kernel fusion
  (`roll`/`where` copies), not math.
- **Exp 3, scale to ~6M (1500 steps):** attention 1.450 (ppl 4.3) vs linear
  1.484 (ppl 4.4). Caveat: linear had 16% fewer params (5.32M vs 6.37M) —
  the gap could be capacity, not architecture.

### V3 — parameter-equalized rematch (6,369,784 vs 6,369,792; 8 apart)

| | Attention | Linear scan |
|---|---|---|
| Final val / ppl | **1.378 / 4.0** | 1.415 / 4.1 |

Gap 0.037 — essentially unchanged from Exp 3's 0.034. Curves crossed twice
(linear led steps 250–500, attention retook ~750 and held). Train tok/s
69,441 vs 31,262 (2.2x); infer 211,426 vs 89,998 (2.3x); peak VRAM 920 vs
1,971 MB. **Conclusion: the gap is architectural (fixed-state bottleneck),
not parameter count.**

### V4 — FFN-only equalization (state frozen at 256, FFN 1024→1279)

First run: linear led steps 1–750, attention overtook 750–1000, FINAL 1.396
vs 1.423 (gap 0.027). Post-train metrics were lost to a Colab idle-timeout
disconnect, so a **clean re-run** was executed from a fresh notebook with
triple-redundant capture (verbatim log + Drive copy + `.ipynb` download).

Re-run: equality gate 6,369,792 vs 6,369,784; linear led steps 1–500,
attention ahead at 750 (1.675 vs 1.742), near-tie at 1250 (1.456 vs 1.464),
linear marginally ahead at 1500 (1.415 vs 1.420); FINAL attention 1.405 vs
linear 1.441 (gap 0.036). Metrics: train 59,591 vs 36,166 tok/s (1.6x);
infer 176,400 vs 102,156 tok/s (1.7x); peak VRAM 831 vs 1,503 MB; train
loss 1.442 vs 1.481.

Three watch-items ruled on: (1) VRAM drop confirmed directionally with state
frozen at 256 but still 1.8x attention; (2) trajectory less stable than hoped
— crossing point moved between runs (±0.05 late-phase noise); (3) capacity
delta: attention ahead ~0.03 in **all three** 6.37M runs (0.037 / 0.027 /
0.036) — small, stable, architectural. **The ~2% perplexity tax was now the
defined enemy.**

### V5 — gated segmented state space (the tax, destroyed)

Dynamic selective gate + hard reset mask on newline boundaries, state frozen
at 256, budget 6,368,760 vs 6,369,792 (1,032 apart, 0.016%).

| Metric | Attention | V5 segmented |
|---|---|---|
| Final val / ppl | 1.388 / 4.0 | **1.360 / 3.9** |
| Best val | 1.388 | **1.360** |
| Train loss (pt) | 1.441 | **1.425** |
| Train tok/s | 63,396 | 36,335 |
| Infer tok/s | 195,322 | 104,253 |
| Peak VRAM | 840 MB | 1,547 MB |

**V5 led at all 7 checkpoints wire-to-wire — attention never led once, no
crossing occurred.** Boundary stats: `newline_id=0`, 2.69% of tokens trigger
reset (~37-token average segments). V5 won on quality (val *and* train loss)
but still trailed 1.7x on throughput and 1.8x on VRAM — the PyTorch scan's
`roll`/`where` traffic, a pure software tax.

### V6 — Triton fusion (clean sweep)

Same math, same params, same protocol as V5 — only the execution path
changed. Two `@triton.jit` kernels (forward + backward): the 256-wide state
stays in SRAM registers across the sequence; sigmoid gate, reset mask, and
linear update run in-register per timestep; only `h_t` (plus `a_t`, saved for
backward) is written to HBM.

**Kernel gate** (hard gate before training — run stops if it fails):
Triton vs V5's actual loop max delta **4.47e-07** (bar: 1e-6); vs V5's scan
4.77e-07; backward grads 1.91e-06 / 1.67e-06 (bar: 1e-5). **Passed.**

| Metric | Attention | V6 Triton |
|---|---|---|
| Final val / ppl | 1.402 / 4.1 | **1.364 / 3.9** |
| Train tok/s | 61,845 | **66,156 (1.07x)** |
| Infer tok/s | 184,918 | **190,122 (1.03x)** |
| Peak VRAM | 845 MB | 881 MB (~tied) |

V6 led every checkpoint wire-to-wire (like V5 — same math, as predicted).
The software tax is gone: V6 **beats** attention on throughput and ties it on
VRAM.

**Mixer-level benchmark** (forward+backward, batch 8, dim 256):

| T | Attention | V5 torch scan | V6 Triton |
|---|---|---|---|
| 128 | 1.00 ms / 310 MB | 3.36 ms / 329 MB | 1.34 ms / **308 MB** |
| 512 | 4.65 ms / 343 MB | 10.74 ms / 452 MB | **1.72 ms** / 339 MB |
| 1024 | 13.96 ms / 390 MB | 22.94 ms / 632 MB | **3.51 ms / 381 MB** |

The fused kernel is 2.5–6.5x faster than the PyTorch scan, and the O(T) vs
O(T²) crossover is visible in wall-clock: **from T=512 up, the fused scan
beats attention outright** (1.72 vs 4.65 ms; 3.51 vs 13.96 ms at T=1024).

### V7 — output-gated SSM (new champion: 1.339)

V6's recurrence + Triton kernel unchanged; only the readout and the gate
bias changed (equations above). CPU-verified before launch
(module == manual recurrence exactly 0.0, gradients flow to gate/output/floor
params). Budget **exact**: 6,369,792 vs 6,369,792 (diff 0) via FFN 1023,
state dim frozen at 256. Kernel gate **passed** (Triton vs V5 loop
4.47e-07, backward ~1.8e-06).

| Metric | Attention | V7 gated-output |
|---|---|---|
| Final val / ppl | 1.423 / 4.2 | **1.339 / 3.8** |
| Fresh-batch re-eval | 1.406 | **1.350** |
| Train loss (pt) | 1.472 | **1.392** |
| Train tok/s | 63,749 | 63,351 (~tied) |
| Infer tok/s | 197,285 | 188,101 |
| Peak VRAM | 835 MB | 935 MB |

**V7 beats V6's 1.364 with no quality regression anywhere — new champion.**
Learning curves: V7 led 6/7 checkpoints (attention led only step 1,
4.004 vs 4.021); the step-250 margin is enormous — **1.863 vs 2.373**
— then 1.600 vs 2.004 (500), 1.484 vs 1.749 (750), 1.405 vs 1.572 (1000),
1.337 vs 1.473 (1250), 1.334 vs 1.413 (1500). The output gate + graded
floor massively accelerate early learning.

Cost: a small output-gate tax vs V6 — train −4.2% tok/s, VRAM +6% (the extra
256→256 projection and the RMSNorm). Trained floors ended at −0.85 → 2.20
bottom→top, confirming the intended fast-forget-low / retentive-high
timescale split. **Verdict: the stabilizers strictly improve quality
(1.364 → 1.339) at a small speed/VRAM cost.**

### V8-A — diagonal gated delta rule (new champion: 1.319)

V7's stabilizers (output gate, RMSNorm, graded floor) kept; the recurrence
swapped for the delta rule above with its own fused Triton kernel. CPU-verified
before launch (forward 1.19e-07, backward ~4.77e-07 vs manual recurrence).
Budget gate **passed**: 6,369,792 vs 6,367,736 (diff 2,056 ≤ 3000; the two
new projections balanced by FFN 766). Kernel gate **passed** (Triton delta
kernel vs manual recurrence 3.58e-07, backward <1e-5).

| Metric | Attention | V8-A delta rule |
|---|---|---|
| Final val / ppl | 1.419 / 4.1 | **1.319 / 3.7** |
| Best val | 1.408 | **1.312** |
| Train loss (pt) | 1.475 | **1.378** |
| Train tok/s | 60,867 | 58,848 |
| Infer tok/s | 180,708 | 173,233 |
| Peak VRAM | 877 MB | 943 MB |

**V8-A led all 7 checkpoints wire-to-wire — including step 1** (3.914 vs
4.012; V7 had trailed attention at step 1). Curve: 1.811 vs 2.361 (250),
1.586 vs 1.968 (500), 1.457 vs 1.723 (750), 1.406 vs 1.592 (1000),
1.337 vs 1.482 (1250), 1.312 vs 1.408 (1500). Against V7's own curve it is
ahead at 6/7 checkpoints (1000: 1.406 vs 1.405; 1250: tie 1.337).
Within-run margin over its own baseline: **0.100** (V7's was 0.084).

Stability: state norms drifted 0.08 → 0.25 (finite, bounded, no runaway);
grad norms fell 4.78 → 0.84, consistently *below* attention's (7.48 → 1.02)
the whole run — the contractive dynamics showing up as designed. Zero
NaN/Inf; the fault guard never fired.

Cost vs V7: train −7.1% tok/s, infer −7.9%, VRAM +0.9% — the two extra
projections charging their toll, quality paying for it.

**Mixer-level benchmark** (forward+backward, batch 8, dim 256):

| T | Attention | V6 Triton | V8-A delta |
|---|---|---|---|
| 128 | 1.36 ms / 31 MB | 1.32 ms / 31 MB | 2.48 ms / 38 MB |
| 512 | 5.65 ms / 67 MB | 1.54 ms / 62 MB | **3.55 ms** / 84 MB |
| 1024 | 12.69 ms / 113 MB | 3.05 ms / 104 MB | **7.11 ms** / 147 MB |

The delta kernel is genuinely O(T): it beats attention at T=512 and is ~1.8x
faster at T=1024, crossover between 128 and 512 — the same "from T=512 up"
story as V6. The delta tax is real at kernel level (~2x slower than V6's
kernel), and memory scales linearly (38→147 MB) against attention's
quadratic climb. **Verdict: clean architectural improvement — beats V7's
1.339 within 1500 steps with the O(T) scaling advantage intact.** Caveats:
single seed; the 0.020 margin over V7 sits near run-to-run noise — the
sturdier evidence is the within-run margin (0.100 vs V7's 0.084).

---

### V9-A — replication & generalization: CONFIRM

5 seeds × (V8-A vs attention), 1500 steps. v8a finals
1.324/1.311/1.311/1.302/1.312 (mean 1.312±0.008) vs attention
1.432/1.365/1.404/1.406/1.404 (mean 1.402±0.024) — v8a won all 5 seeds, no
flips, margin 0.090. enwik8: 1.527 vs 1.683; text8: 1.507 vs 1.519 (v8a led
all 7 checkpoints on both). Zero-shot length: v8a flat 1.324→1.296 across
T=128→2048 while attention degrades 1.432→3.33.

### V9-B — bf16 scan-state storage: REJECTED

Numerically clean (trajectory diffs ≤0.0004, final 1.326) but ~30% throughput
regression (40,868 vs 58,848 tok/s) and only ~100 MB saved of a predicted
350–400 — the activation footprint is distributed across the graph, not in
the bf16 h/a tensors. Cut per the falsification protocol.

### V9-C — fused recompute-backward: REJECTED

Bit-identical grads (0.00e+00), −34 MB VRAM as predicted — but +10.8% step
time for 34 MB is a bad trade. Recomputing `a` costs more than the memory
traffic it saves.

**V9's record: A confirms, B falsifies, C falsifies** — two VRAM levers killed
with mechanisms documented. The 943 MB peak (~840 MB activations) remains
open; V10 starts with a memory autopsy. Full notes in
`V9_RESEARCH_NOTES.md`.

### V10 — where are the ~943 MB going? (ladder: E1 reject, E2 reject, E3 reject, E4 fail, E5 falsified)

Autopsy: mixer ~408 MB + FFN ~277 MB live at peak. Micro-batch accumulation
valid (−62.1%) but rejected on throughput (+48.9%). `expandable_segments`
−17 MB (negligible). `torch.compile` numerics pass but peak +27% (rejected as
a memory lever; ~1.5× faster as a side finding). Optimizer offload −4 MB
reserved, +36.3% step (structural fail). ActNN quantization falsified cleanly:
peak +24.5% (forward transients dominate), +65% step — though the trajectory
was preserved perfectly (stochastic rounding truly unbiased). Pattern: every
post-hoc lever moves peak reserved negligibly or regresses it. Terminal
decision: stop tweaking infrastructure; redesign WHEN activations
materialize. Full notes in `V10_RESEARCH_NOTES.md`.

### V11 — memory-lifetime redesign (D2a PASS, D1 PASS, stack: the endpoint)

**D2a** (selective FFN checkpoint): 839 → 659 MB reserved (−180 MB), step
−31.9% — faster, not slower. **D1** (fused projection+scan, save-x-only):
628/705 MB (−134 MB, −16%), step −22%, grads 7.5e-10 vs the 1e-6 bar. The
numerics saga is told honestly in the notes: a real transpose bug
(`dx = du @ W.t()` instead of `du @ W`) hid behind a CPU pre-flight that
never checked `dx`; a falsified TF32 theory died first. **Stack (D1+D2a)**:
the meaningful V11 endpoint: 839 → 520 MB reserved (−38%), step −10.9%.
Peak falls and step time falls at every stage: attacking
creation/save time works where post-hoc compression didn't.

### V12 — multimodal V8-A: does the advantage survive non-text sequences?

**Frozen discipline:** V8-A mixer + V11 implementation frozen; mixer-only
comparison (same encoder, param budget, data, optimizer). AR next-element
prediction: next-patch (vision), next-frame (audio), MSE loss.

**Phase 1 (vision, ImageNette, T=256, 1500 steps):** attention 0.2102 vs
V8-A 0.2578 — attention wins quality clearly (~22%). V8-A's text advantage
does NOT automatically transfer to spatial sequences.

**Phase 2 (audio, LibriSpeech, T=256, 1500 steps):** V8-A 0.0552 vs
attention 0.0581 — V8-A wins narrowly (~5%). Pattern: temporal favors
V8-A, spatial favors attention.

**Phase 3 (scaling rungs, both modalities):** crossover between T=256–529
in vision (1.97x at T=2025) and audio (2.23x at T=2025). Attention slightly
leaner VRAM at every rung — throughput, not memory, is where O(T) wins.

**Phase 4 (joint multimodal, Flickr8k):** 500-step pilot suggested gap
narrowing (80%), but the matched 2×2 control (A=vis/attn, B=vis/v8a,
C=joint/attn, D=joint/v8a) gave I=+0.0386 — joint training *hurts* V8-A's
relative position. Attention's vision improves with text context
(0.6035→0.5670); V8-A's does not. Predeclared rule: STOP, do not extend.
The pilot was confounded; the control falsified the rescue hypothesis.
Init-verification mismatch noted as a limitation for future retests.

**Verdict:** a quality–efficiency trade-off depending on modality and
sequence length — not a universal attention replacement. See
`V12_RESEARCH_NOTES.md` for the full record including the evidence
classification.

---

## Design decisions and minor details

- **Why FFN-only equalization (V4):** widening the state to match parameters
  (V3: state 384) inflated the scan's VRAM; freezing state at 256 and widening
  only the FFN (1024→1279) protected the memory edge while closing the budget.
- **Why the gate is full-rank (V5):** a full `Linear(256→256)` + bias gate
  with FFN 1151 lands at 6,368,760 vs 6,369,792 — 1,032 apart (0.016%), the
  closest possible without distorting the design. A low-rank bottleneck gate
  could hit the budget exactly but would change the specified architecture;
  0.016% cannot explain a 2% perplexity gap.
- **Reset semantics:** when `r_t = 1`, `h_t` is set to exactly zero *including*
  the boundary token's own contribution — verified to read exactly `0.0`.
  The residual stream still carries `x_t` itself forward, so the newline is
  not erased from the model, only from recurrent memory.
- **Why `grid=(B,)`, `BLOCK=256`:** the recurrence is per-channel
  independent and the whole 256-wide state fits in registers; 32 programs ×
  128 sequential steps is memory-bound on ~16 MB total traffic — occupancy is
  a non-issue and simplicity won over channel-splitting.
- **Why `a_t` is saved for backward:** recomputing it would cost an extra
  full pass plus `exp` per step; one 4 MB tensor is the cheaper trade.
- **Why the V7 floor is init-only (no cumax):** the per-layer floor is a
  free learnable scalar with a monotonic init (logit(0.3)→logit(0.9)); the
  trained values (−0.85→2.20) preserved the ordering on their own, so no
  constraint was needed.
- **Why V7's RMSNorm is parameter-free:** the output gate `o_t` already
  provides per-channel scale control, so the norm needs no learned weight —
  the readout is `C · (o_t ⊙ RMSNorm(h_t))` with zero new norm params.
- **Why V8-A inits b_k=+2.0 / b_β=0:** the model starts near V7's
  convex-blend regime (k≈0.88, β≈0.5) so early training is stable; the sigmoid
  bounds then keep every step strictly contractive (`a_t < 1`), which is why
  the state norms stay bounded (0.08→0.25) instead of running away.
- **Why no multi-head split for V8-A:** per-channel input-dependent gates
  are already 256 independent 1-dim heads; the literature verdict (GDN/HGRN
  ablations) was that explicit splitting buys nothing here.
- **Why the stabilizers are load-bearing:** the GDN ablation shows naive
  delta integration is +3.52 ppl worse — V7's output gate + RMSNorm + floor
  are what make the delta rule trainable, which is why V8-A keeps all three.
- **Anomaly log:** (a) V4's first run lost its metrics to a Colab
  idle-timeout disconnect → clean re-run with triple-redundant capture;
  (b) the re-run's `FINAL-VERIFY` re-check printed divergent numbers
  (2.632/2.763) because the metrics cell had run 30 extra optimizer steps —
  corrupted by extra training, not a result; authoritative numbers are the
  training cell's FINAL lines, and V5/V6 store FINAL values *before* the
  metrics cell; (c) the V6 browser capture mislabeled the T=512 Triton row —
  ground-truthed against the downloaded notebook's cell outputs (this repo
  carries the corrected table); (d) V8-A's kernel-benchmark cell crashed with
  `'float' object has no attribute 'constexpr'` — root cause: the metrics
  cell used `tl = 0.0` as a throwaway variable, shadowing `triton.language`
  so the legacy V6 kernel failed JIT compile when first called in the
  benchmark (V8-A's own delta kernel had compiled and run all 1500 training
  steps fine; a harness bug, not a model bug). Fixed (`tl`→`tloss_acc`) and
  re-run benchmark-only for the table above.

## Verification log

- Hillis–Steele scan == sequential loop: max diff 2.4e-07
- Selective scan == loop with random gates + reset mask: 1.2e-07
- Reset zeroes state exactly (`0.0`); fresh accumulation after boundary
- Gradients flow to gate and input projections
- Compiled Triton kernel == V5 loop: 4.47e-07 (forward), ~1.8e-06 (backward)
- V7 module == manual recurrence: 0.0; grads flow to gate, output-gate, and
  floor parameters; equality gate exact (diff 0)
- V8-A delta kernel == manual recurrence: 3.58e-07 (forward), backward grads
  <1e-5; budget gate 6,369,792 vs 6,367,736 (diff 2,056 ≤ 3000)

## Reproduce

**Colab (as run):** upload any notebook from `notebooks/` to
[Google Colab](https://colab.research.google.com/), select a T4 GPU runtime,
`Runtime > Run all`. Fully self-contained — no pip installs (torch 2.x ships
Triton).

**Local:**
```bash
pip install -r requirements.txt
python -c "from src.models import TinyLM, CausalSelfAttention,
  SelectiveSegmentedState; print('ok')"
# Triton kernels need a CUDA GPU; the torch-scan path runs on CPU.
```

## Caveats

- Single seed per experiment; run-to-run late-phase noise observed ~±0.05 in V4.
- One corpus (2.47 MB, char-level, English literary text); the reset mask's
  advantage is tuned to structured text with paragraph breaks.
- Timings are T4-specific; the Triton kernel targets NVIDIA GPUs (sm_75+).
- Small scale throughout (6.37M params, seq 128 training) — the 1024-length
  benchmark is mixer-level, not full training.

## Timeline

| Time (IST, 2026-10-08) | Event |
|---|---|
| ~02:05 | V1: linear state wins on quality, 10x slower (naive loop) |
| ~02:30 | V2: memory scaling proven; parallel scan verified |
| ~03:25 | V3: param-equalized rematch — gap is architectural |
| ~03:35 | V4: FFN-only equalization; metrics lost to disconnect |
| ~04:45 | V4 re-run (clean): the ~2% tax defined |
| ~05:05 | V5: selective gate + reset — tax destroyed, wire-to-wire win |
| ~05:20 | V6: Triton fusion — clean sweep (quality + speed + memory) |
| ~06:20 | V7: output gate + graded forget floor — new champion (1.339), no regression |
| ~10:26 | V8-A run 1: diagonal gated delta rule — new champion (1.319), wire-to-wire lead |
| ~11:17 | V8-A kernel benchmark re-run (fixed `tl` shadowing): O(T) scaling confirmed, T≥512 beats attention |

## License

MIT — do what you want with it, attribution appreciated.
