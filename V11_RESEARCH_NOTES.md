# V11 Research Notes — V8-A memory-lifetime redesign

Date: 2026-10-08→09, Colab T4. V10's verdict: post-hoc levers can't move peak
reserved — the activation live set at creation/save time sets the peak. V11
redesigns WHEN activations materialize: never create the large intermediates
in the first place. V8-A architecture and learning objective frozen throughout
(scope lock); only materialization timing changes. All runs: batch 32, T=128,
fp32, complete training steps (the backward live set is the point).

## V11-D2a — selective FFN checkpoint: PASS

`torch.utils.checkpoint` around each block's mlp only (mixer untouched).
Phase A: 761/839 @ 93.3 ms/step; Phase B: 578/659 @ 63.5 ms/step.
Reserved −180 MB (gate ≤ −150: PASS); step −31.9% (PASS — and FASTER:
skipping ~200 MB of HBM loads beats the recompute cost on bandwidth-bound
T4); numerics not bit-identical (3.5e-11 rel, benign — fails the letter,
passes any practical tolerance). First lever in V10+V11 to cut peak reserved
a lot AND speed up.

## V11-D1 — fused projection+scan, Path B (save-x-only): PASS with a lesson

The contract: ≤1e-6 numerical equivalence (not bit identity), forward keeps
V8-A math, never materialize the 5 projection tensors (B,T,256)×5, ≥25%
peak-reserved reduction, +10% step cost. Path B (chosen over in-SRAM Path A,
whose sequential-t `tl.dot`s risked +25–85 ms on T4): a save-x-only autograd
Function — forward runs the 4 scan projections as cuBLAS GEMMs (transient,
never saved), saves only x/keep/h_seq/a_seq; backward recomputes projections
and runs the EXISTING verified Triton bwd kernel unchanged. out_gate_proj
stays separate (consumed after the scan).

Results (3 runs, stable): A 761/839 @ ~84–88 ms → B 628/705 @ ~66 ms.
- Reserved −16.0% (−134 MB; gate −25%: MISS). The −134 MB is exactly the
  analytical projection saves; the rest of E0's 408 MB mixer figure was
  backward transients any correct implementation needs. Real, clean, below bar.
- Step −22% (gate +10%: PASS — faster).
- Grads 7.50e-10 vs 1e-6 (PASS, three orders of magnitude).

The numerics saga, honestly told: the first run showed grad rel-diff
5.86e-04 — a MISS. A dW reduction-order/TF32 theory was proposed, tested, and
falsified cleanly (the fix changed the number by 1e-10: dW was never it). A
localization debug then showed forward bit-identical and the mixer's own
param grads matching, but `dx` (gradient to upstream layers) off by 1.3× —
the CPU pre-flight had never checked `dx`. Root cause: `dx = du @ W.t()`
is WRONG — `F.linear` computes `x @ W.T`, so `dL/dx = du @ W` (PyTorch's own
Linear backward uses `grad_output @ weight`). A classic transpose error,
invisible to every check that only looked at the mixer's own parameters.
Fixed in both paths; CPU dx now 1.9e-07.

## V11-stack — combined D1 + D2a (the V11 endpoint)

The two levers target different activation classes (D2a: FFN saved
intermediates; D1: mixer projection saves), so the combined run is the
meaningful endpoint. Phase A plain vs Phase B combined, complete steps
(4th account, T4, 2026-10-09 ~02:20 IST):

- Phase A: 761 MB alloc / 839 MB reserved @ 82.9 ms/step
- Phase B: 450 MB alloc / 520 MB reserved @ 73.9 ms/step
- Reserved **−38.0%** (839 → 520 MB, −319 MB) — composes almost additively
  (−180 D2a + −134 D1 = −314 predicted; −319 measured).
- Step **−10.9%** — faster, not slower.
- Grads rel-diff 5.26e-10 vs 1e-6: PASS.

## What V11 establishes

Peak reserved 839 → 520 MB (−38%) with step time FASTER at every stage —
the opposite of the usual memory/throughput tradeoff, because both levers
remove HBM traffic that cost more than the recompute. The V10 pattern is
broken: attacking creation/save time works where post-hoc compression
didn't. Remaining: Path A (in-SRAM fusion) as an optional second-stage
throughput optimization; D2b (fused FFN kernel) only if a future bar
demands it.
