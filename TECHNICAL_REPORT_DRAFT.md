# Linear Attention Lab: A Controlled Study of Diagonal Gated Delta-Rule Sequence Mixing

**Status:** Technical report draft. Headline numbers are reported
experimental results from the committed campaign (V1–V13); independent
reproduction via the GPU audit is pending. Do not cite numerical values
as independently reproduced until the audit completes.

---

## 1. Research question and scope

**Question:** Can the self-attention sublayer of a Transformer be replaced
with a gated linear recurrence over a fixed-size state — and under what
conditions does the replacement win or lose?

**Scope.** This study tests a specific architecture (V8-A: diagonal gated
delta-rule state-space model) against a parameter-matched causal
Transformer baseline on autoregressive sequence modeling. It measures
quality (validation loss/MSE), peak memory, and throughput across text,
vision, and audio modalities, plus a joint image–caption condition. It
does not claim a general architectural superiority law, a universal
modality-dependent law, or a field-wide research gap.

**What this report is not claiming.** The measured throughput crossover
and speedups apply to the tested implementations under the documented
benchmark conditions — not to every linear-time model or workload. The
modality-dependent quality pattern (temporal favors V8-A, spatial favors
attention) is supported in the evaluated configurations; broader claims
require more datasets and controlled replications.

---

## 2. Architecture and implementation

**V8-A mixer.** The attention sublayer is replaced with a diagonal gated
delta-rule recurrence: per-channel forget/input gating over a fixed-size
state, updated with a delta rule. The recurrence is O(T) in sequence
length under fixed-state, fixed-width assumptions.

**Fused Triton kernels.** Forward and backward passes are implemented as
custom `@triton.jit` kernels with autograd wrappers
(`SelectiveSegmentedStateV8A`). The V11-D1 optimization fuses scan-input
projections via a save-x-only autograd Function (Path B).

**Numerical validation (mandatory).** Gradient correctness is checked by
relative difference against a reference implementation, threshold 1e-6:
- V11-D1 standalone: rel-diff 7.50e-10 (PASS, three orders of magnitude).
- V11 D1+D2a stack: rel-diff 5.26e-10 vs 1e-6 (PASS).
- During development, a transpose bug (`dx = du @ W.t()` instead of
  `du @ W`) was caught by this check after a CPU pre-flight that only
  validated the mixer's own parameters missed it; fixed in both paths.

**Memory-lifetime optimization (V11).** Two composable levers:
- D2a (selective FFN checkpoint): 839 → 659 MB reserved (−180 MB), step
  −31.9% (faster, not slower).
- D1 (fused projection+scan, save-x-only): −134 MB, −16%, step −22%.
- Combined stack: 839 → 520 MB peak reserved (−38%), step −10.9%.
  Peak falls and step time falls at every stage.

**Frozen baseline.** `models.py`, `triton_kernels.py`, `fused_mixer.py`
were frozen for the V12–V13 campaign. All comparisons are mixer-only:
identical encoders, parameter budgets, data ordering, optimizer, and
schedule — only the sequence mixer changes.

---

## 3. Controlled text evaluation

**Parameter matching.** Attention baseline: 6,369,792 parameters. V8-A:
6,369,784 parameters. Difference: eight parameters (V4-style FFN
equalization: attention arm uses wider FFN to compensate for its larger
QKV projections). Described as "parameter-matched to within eight
parameters."

**Results (reported).**
- V8-A final: validation loss 1.319 vs attention 1.419 (single run).
- V9 replication (five seeds): V8-A 1.312±0.008 vs attention 1.402±0.024,
  V8-A winning all five seeds under tested conditions.

**Assessment.** The text quality advantage has multi-seed support under
the tested configuration. This is the strongest quality claim in the
study.

---

## 4. Systems evaluation

**Throughput crossover.** Measured per-step latency (median of 30 timed
steps after 5 warmup; Triton compile time reported separately as a
one-time cost):
- Crossover between T=256 and T=529 in both vision and audio modalities.
- At T=2025: ~1.97× (vision) / ~2.23× (audio) V8-A advantage.
- Compile tax: V8-A Triton JIT ~5s vs attention ~1–3s, paid once.

**Memory.** Attention used slightly less peak VRAM than V8-A at every
tested scaling rung (quadratic attention matrix freed per layer; peak
dominated by other activations). The O(T) advantage appears in
throughput, not peak memory, under these conditions.

**Timing methodology.** Compile, warmup, and steady-state are measured
and reported separately. Steady-state (median of timed steps) is the
compared number.

**Assessment.** The throughput crossover is supported under documented
benchmark conditions (T4 GPU, tested sequence lengths, fp32). It does not
establish that every linear-time implementation outperforms every
attention implementation at the same lengths.

---

## 5. Multimodal results

### 5a. Vision (ImageNette, next-patch prediction, T=256, 1500 steps)

**Reported:** attention 0.2102 vs V8-A 0.2578 validation MSE (~22%
relative advantage for attention).

**V13 follow-up** (LR grid {1e-4, 3e-4, 1e-3} × both mixers, 1500 steps
each, tensor-verified init 6/6): best V8-A 0.2413 vs best attention
0.1964; gap +0.0449, essentially unchanged from Phase 1's +0.0476.
Gap persists under the tested grid. Useful negative result for the
tuning-artifact hypothesis; not proof of architectural limitation.

**Interpretation.** The vision results are consistent with a concern
identified in the vision state-space literature (single-direction scans
disadvantaged on spatial structure; cf. Vision Mamba's bidirectionality,
VMamba's 4-way cross-scan). However, the experiments differ in task
(classification/detection/segmentation vs next-patch prediction),
objective, and architecture, and the quality results are single-seed.
The literature motivates the scan-topology hypothesis; it does not
establish causality. A controlled scan-topology ablation would be needed.

### 5b. Audio (LibriSpeech, next-frame prediction, T=256, 1500 steps)

**Reported:** V8-A 0.0552 vs attention 0.0581 validation MSE (~5%
relative advantage for V8-A, narrow).

### 5c. Joint image–caption (Flickr8k, 500 steps)

**Pilot** suggested 80% gap narrowing vs Phase 1 reference, but the
baseline was unmatched (different dataset, steps, task).

**Matched 2×2 control** (vision-only and joint arms, both mixers, same
data): interaction I = +0.0386. Attention's vision improved with joint
training (0.6035→0.5670); V8-A's did not (0.5934→0.5955).

**Assessment.** The rescue hypothesis was not supported by the decision
test. However: tensor-level init verification failed (confound), and
near-zero text loss (~0.0003) makes the experiment uninformative about
temporal preservation. Correct summary: cross-modal exploitation by V8-A
is **not established** — not "disproven."

### 5d. Modality pattern (narrow interpretation)

The tested temporal tasks (text, audio) favored V8-A on quality; the
tested spatial vision task favored attention. This is a
modality-dependent pattern in the evaluated configurations — not yet a
general property of temporal vs spatial data.

---

## 6. Related work and contribution boundary

See `literature_review_draft.md` for the full related-work table
(Vision Mamba, VMamba, Cerruti et al. ETH Zurich comparative study,
Gated DeltaNet).

**What the surveyed works do not provide** (bounded claim): the exact
combination of (a) mixer-only comparison with parameter matching to
within eight parameters, (b) cross-modal autoregressive probing with
predeclared decision rules, (c) validated fused-kernel implementation
with measured memory optimization, and (d) published negative results
with explicit "not established" statements. A broader literature search
would be needed for a field-wide absence claim.

**Claim assessment:**

| Claim | Status |
|---|---|
| Validated fused-kernel implementation | Supported by numerical checks (rel-diff 5.26e-10 vs 1e-6), subject to reproducibility audit |
| Text val loss improvement over param-matched baseline | Supported (five-seed: 1.312±0.008 vs 1.402±0.024) |
| Throughput advantage at longer tested lengths | Supported under documented benchmark conditions |
| Generally better for temporal sequences | Not established (insufficient tasks/datasets) |
| Cannot exploit cross-modal context | Not established (Phase 4 inconclusive on causality) |
| Combination absent from literature | Plausible positioning, requires broader verification |

---

## 7. Reproducibility and limitations

**Audit status.** A reproducibility audit plan is committed
(`reproducibility_audit_plan.md`) covering: V9 five-seed rerun, V11
memory benchmark rerun, V12 scaling rerun with separated
compile/warmup/steady-state timing, and V13 verdict logic verification
(completed locally). GPU execution is pending Colab quota availability.
Until the audit runs, all headline numbers in this report are **reported
experimental results**, not independently reproduced results.

**Seeds.** V9 text: five seeds. V12/V13: single seed (V13 strengthens
persistence across three LRs but does not replicate across seeds).

**Frozen artifacts.** All notebooks, generators, and the three frozen
source files are committed. Reruns use committed artifacts as-is;
rerun values are reported beside originals, never replacing them.

**Unresolved questions.**
- Vision gap cause: hyperparameter transfer, scan topology, preprocessing,
  or other factors? (V13 rules out LR-grid artifact; topology untested.)
- Does the modality pattern generalize across datasets and seeds?
- Does joint training change V8-A with verified init and informative text
  objective?
- How much throughput advantage is recurrence complexity vs specific
  Triton implementation and hardware?

**What would strengthen this report.** Completed GPU audit; a controlled
scan-topology ablation for the vision gap; multi-seed V12 replication;
broader literature survey for the novelty boundary.

---

*Report assembled 2026-10-09 from the committed V1–V13 campaign record.
Literature-review structure frozen. No new architecture experiments were
run to fill space. The reproducibility audit is the outstanding empirical
gate.*
