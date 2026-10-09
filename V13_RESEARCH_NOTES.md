# V13 Research Notes — Vision hyperparameter probe

**Question (from V12 brief):** Is V8-A's ImageNette vision quality gap
(0.2578 vs 0.2102, ~22%) a hyperparameter artifact of text-tuned settings,
or a fundamental limitation of compressive state for spatial detail?

**Design:** LR grid {1e-4, 3e-4, 1e-3} × {V8-A, ATTN} = 6 arms, 1500 steps
each, ImageNette T=256, param-matched, mixer-only difference. Both arms
receive EQUAL tuning budget. Tensor-level init verification on shared
encoder (Phase 4 lesson applied): "INIT OK" 6/6, no mismatches.

**Frozen:** models.py, triton_kernels.py, fused_mixer.py, v12_multimodal.py
— untouched.

## Results (2026-10-09, gitmusic0, T4, ~25 min)

| LR | V8-A val MSE | ATTN val MSE | gap (v8a − attn) |
|---|---|---|---|
| 1e-4 | 0.2911 | 0.3383 | −0.0472 (V8-A wins) |
| 3e-4 | 0.2567 | 0.2331 | +0.0236 |
| 1e-3 | 0.2413 | 0.1964 | +0.0449 |

Best per arm: V8-A 0.2413 (lr 1e-3) vs ATTN 0.1964 (lr 1e-3).
gap = +0.0449; abs(gap) = 0.0449 > 0.02 → **GAP PERSISTS.**

Phase 1 baseline gap was +0.0476 (v8a − attn). The V13 best-gap (+0.0449)
is essentially unchanged.

## Interpretation (his)

V13 is a **useful negative result for the tuning-artifact hypothesis**:
the gap persisted under this LR grid with equal tuning budgets and
verified initialization. It does NOT establish the gap as a proven
architectural limitation — only that it is persistent under the tested
conditions.

Notable: at lr 1e-4, V8-A actually beat attention (−0.0472), suggesting
different LR sensitivities between the architectures. But both arms'
respective bests (at lr 1e-3) preserve the Phase 1 ordering and magnitude.

## Stopping decision (his)

V13 is sufficient to document a persistent vision disadvantage under the
tested learning-rate grid. A wider sweep (optimizer, augmentation, patch
size, duration) would answer a NEW question, not change V13's validity.
Another vision experiment requires a concrete predeclared hypothesis, a
controlled protocol, and a stopping criterion — and must not change
multiple factors simultaneously.

## Methodological note

The notebook's original verdict logic (`best_a − best_v < 0.02`) was buggy:
it returned "CLOSED" for a gap of −0.0449. Corrected to `abs(gap) < 0.02`
before publication. The six runs are preserved unchanged; only the verdict
logic was fixed.
