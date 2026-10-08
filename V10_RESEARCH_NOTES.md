# V10 Research Notes — where exactly are the ~943 MB going?

Date: 2026-10-08, Colab T4. V9-B/C proved the recurrence was never the memory
problem (the fused kernel did its job); V10 starts from measurement, not
another lever. Every experiment below ran with V9-style gates: numerical
≤1e-6 where dynamics are exact, explicit falsification clauses, measured
numbers tagged from live runs.

## V10-E0 — memory autopsy (prerequisite)

Peak: 761 MB alloc / 822–839 MB reserved. Live at peak: mixer ~408 MB, FFN
~277 MB — the two chunks are ~685 MB of the peak. The analytical autopsy had
undercounted the mixer 3× (134 predicted vs 408 measured — the projections
are saved in multiple places). Methodological rule for the whole campaign:
record peak *live* memory by tensor/category — bytes alive at the moment of
peak, not cumulative allocation. Lifetimes decide the map, not sizes.

## V10-E1 — micro-batch 4×8 + grad accumulation: VALID, REJECTED on throughput

Batch-32 peak 761 MB reproduces E0 exactly; 4×8 accumulation peaks at 288 MB
→ −62.1% (gate PASS). Mathematically valid (grad delta 8.4e-09, bit-identical
500-step trajectories). REJECTED as a production lever: +48.9% step cost.
Lesson: the next levers must attack activation tensors without multiplying
kernel launches.

## V10-E2 — `expandable_segments`: REJECTED

E2a baseline 761/839/77 MB @ 386.6 ms/step; E2b 760/822/62 MB @ 381.2 ms/step.
Reserved −17 MB (gate ≥30 MB: FAIL); fragmentation gap 77→62 MB (−15 MB,
~19%, not substantial); peak alloc — the OOM number — unchanged; no slowdown.
A real, free, but negligible setting — not a lever.

## V10-E3 — `torch.compile`/inductor: REJECTED as a memory lever

Numerics PASS (grad rel-delta 2.92e-07 ≤ 1e-6, losses identical 4.691391).
But peak alloc 970 MB (+27%), reserved 1097 MB (+31%) — memory went the wrong
way. Mechanism: inductor's fused kernels trade memory for speed; the Triton
scan kernel graph-breaks, so the recurrence is untouched while compiled
FFN/norm kernels hold more live memory than eager. Side finding (not a
lever): ~1.5× training-throughput win (54.9 vs 83.7 ms/step properly-warmed
eager) if a future objective values speed over memory. Correction worth
recording: an early "~7× speedup" claim was overstated — the baseline's timed
window included first-backward Triton compilation; the true number is ~1.5×.

## V10-E4 — optimizer CPU offload: FAILED (all four criteria)

E4-A 761/839 @ 83.7 ms/step; E4-B (50.9 MB states on CPU) 710/835 @
114.1 ms/step. Reserved −4 MB (gate ≈ −51): FAIL — alloc dropped the full
51 MB but reserved didn't (peak reserved is max-over-time; the step's
prefetch re-materializes states and the allocator retains freed blocks).
Throughput +36.3% (gate <+2%): FAIL — the PCIe round-trip is structural.
Numerics not bit-identical (1.4e-11 rel, benign FP noise): fails the letter,
passes any practical tolerance. Training stable. Two structural mechanisms,
both working as designed — the scheme cannot move peak reserved.

## V10-E5 — ActNN-style quantized activation storage: FALSIFIED (clean kill)

The big swing, falsified with two documented mechanisms:
1. Peak VRAM ROSE 24.5% (947 vs 761 MB; gate was ≥25% reduction). The
   theoretical 4× applies to the backward saved-set, but peak = max over the
   whole step — producing the int8 copy costs ~4 full-size fp32 temporaries
   per tensor in forward (abs, x/scale, rand, floor) while the fp32 original
   is still live. Post-hoc compression is badly timed: the forward live-set
   grows more than the backward saved-set shrinks.
2. Step overhead +65% (gate ≤15%): ~7 extra kernel launches per tensor per
   step — the probe multiplied launches, violating the no-multiplied-launches
   criterion. Only a single fused quantize kernel per tensor could revisit this.

Notably, the trajectory was preserved PERFECTLY (500-step dev 0.0001,
1500-step dev 0.0000, anchor 0.0129 vs 1.319) — stochastic rounding is truly
unbiased in practice. A beautiful negative result: the quantizer works, the
timing doesn't.

## V10 close-out

Scorecard: E1 valid but rejected on throughput; E2/E3 rejected; E4 failed;
E5 falsified with mechanisms. Pattern: every post-hoc lever either moves
peak reserved negligibly or regresses it, while the activation live set at
creation time sets the peak. Terminal decision: stop tweaking infrastructure.
The next design attacks WHEN activations are created/saved — fused kernels
that never materialize the large intermediates. That is V11.
