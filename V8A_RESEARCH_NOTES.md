# V8-A Research Notes — provisional abstract language

Status: PROVISIONAL (his words, 2026-10-08). Not for publication as-is.
Single-seed result; validation across seeds/datasets/lengths still open.

## Provisional abstract sentences (verbatim, his drafting)

1. V8-A improved the benchmark quality over V7 while preserving the fixed-state O(T) recurrence and recovering a long-sequence runtime advantage over attention at T≥512, with only a modest full-model throughput/VRAM penalty at T=128.

2. A fixed-size, selectively gated delta-state sequence model can outperform a parameter-matched attention baseline on this benchmark while retaining linear sequence-scan complexity and favorable long-context memory scaling.

## Writeup skeleton (his outline, 2026-10-08)

- **Problem:** attention's sequence-dependent memory/computation costs.
- **Approach:** fixed-size selectively gated delta-state recurrence.
- **Method:** controlled parameter-matched comparisons and fused implementation.
- **Results:** V8-A's 1.319 vs 1.339 V7 / 1.419 attention, plus T=512/1024 scaling.
- **Tradeoff:** ~7–8% full-model throughput cost versus V7.
- **Validation:** multi-seed + additional datasets/lengths.
- **Conclusion:** what the evidence actually supports.

## Numbers backing the claims (Run 1, 2026-10-08, Colab T4, 1500 steps)

- Final val: v8a **1.319** (ppl 3.7) vs attention 1.419 (ppl 4.1); V7: 1.339 (ppl 3.8). Best val 1.312 vs 1.334 (V7).
- Param match: 6,369,792 vs 6,367,736 (diff 2,056 ≤ 3000 gate).
- Checkpoints: v8a led all 7 wire-to-wire (3.914 → 1.312); ahead of V7's curve at 6/7.
- Kernel gate: 3.58e-07 (≤1e-6). State norms 0.08→0.25 bounded; grad norms 4.78→0.84; no NaN/Inf.
- Scaling (fwd+bwd, batch 8, dim 256): T=512 → v8a 3.55ms vs attn 5.65ms; T=1024 → 7.11ms vs 12.69ms (~1.8x). Memory: 38→84→147MB linear vs attention 31→67→113MB quadratic.
- Cost vs V7: train 58,848 tok/s (-7.1%), infer 173,233 (-7.9%), VRAM 943 vs 935 MB (+0.9%).

## Open before any strong conclusion

- Multi-seed repeat (current margin over V7, 0.020, sits near run-to-run noise).
- Additional datasets and sequence lengths.
