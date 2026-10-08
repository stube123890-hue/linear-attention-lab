# V9 Research Notes — replication, and two falsified VRAM levers

V9 (2026-10-08, Colab T4): V8-A's crown put under replication, then two
VRAM-optimization paths tested and killed experimentally. Record: A confirms,
B falsifies, C falsifies — with mechanisms documented for both kills.

## V9-A — replication & generalization: CONFIRM

5 seeds × (V8-A vs attention), 1500 steps, fp32, T=128, batch 32.
v8a finals: 1.324/1.311/1.311/1.302/1.312 (mean 1.312±0.008);
attention finals: 1.432/1.365/1.404/1.406/1.404 (mean 1.402±0.024).
v8a won all 5 seeds, no flips; margin 0.090 > 5× pooled noise; below V7's 1.339.
Step-250 margin 0.561 (early-learning claim holds across seeds).
enwik8: 1.527 vs 1.683 (v8a led all 7 checkpoints). text8: 1.507 vs 1.519
(v8a led all 7; 27-char vocab compresses the gap).
Zero-shot length (T=128-trained): v8a flat 1.324→1.296 across T=128→2048;
attention degrades 1.432→3.33. T=512 token-matched: v8a 1.478 @ 2796 MB vs
attention 2.380 @ 2414 MB, no OOM either side (v8a peak VRAM > attention at
T=512 — activations dominate at length; this motivated V9-B/C).

## V9-B — bf16 scan-state storage: REJECTED (falsified)

Kernel casts bf16→fp32 on load, compute stays fp32 in SRAM, only stored h/a go
to bf16. Micro-batch equivalence PASSED (2.98e-08); bf16 kernel gate PASSED
(rel fwd 1.05e-04, rel bwd 4.44e-03); 250-step probe PASS → full 1500-step run.
Full run: final val 1.326 (within 0.03 of 1.319 ✓); trajectory gate PASS
(per-checkpoint diffs ≤0.0004 — numerically identical path). BUT: train tok/s
40,868 vs bar 58,848 (~30% throughput regression → lever cut per protocol);
peak alloc 660 / reserved 749 MB vs ≤500 target (only ~100 MB saved, not the
predicted 350–400). Mechanism: the missing 300–400 MB wasn't primarily in the
bf16 h/a tensors — the activation footprint is distributed across the training
graph. T4 lacks native BF16 Tensor Cores, making the strategy unattractive on
the hardware used (no claim about Ampere+ until tested).

## V9-C — fused recompute-backward: REJECTED (falsified)

Recompute the saved `a` tensor in backward instead of storing it.
Numerical gate PASSED — recompute-vs-standard grad deltas bit-identical
0.00e+00 (vs reference math ≤1.43e-06). VRAM −34 MB alloc (908→875; reserved
981→948) — hit the corrected ~32 MB expectation exactly (the saved `a` tensor,
4 MB/layer × 8). Step-time **+10.8%** (62.04→68.76 ms) vs <2% target and >5%
falsification bar. 500-step trajectory gate FAILED (1.55e-02 vs 1e-3; final val
anchor held 1.585 vs 1.586, bit-identical grads implicate data/RNG
nondeterminism between runs, not the lever). Mechanism: recomputing `a` in the
Triton backward costs more than the memory traffic saved — 34 MB for +10.8%
step time is a bad trade on T4.

## What V9 leaves open

The 943 MB peak is ~840 MB activations + workspace; the scan state V9-B/C
fought over is only ~67 MB of it. V10 starts with a memory autopsy (peak *live*
memory by tensor/category, not cumulative allocation) before any new lever.
See `v10_study_brief.md` (not in this repo) for the ranked plan.
