# V12 Research Notes — Multimodal V8-A

**Question:** Does the V8-A advantage survive when the sequence is no longer text-only?

**Discipline (frozen):** V8-A mixer + V11 implementation frozen throughout.
Mixer-only comparison: same modality encoder (arch + init), same parameter
budget (V4-style matched), same training data/order, same optimizer (AdamW
3e-4) and schedule. Only the sequence mixer changes (V8-A vs causal
attention). Starting point: the 520 MB V11 stack.

**Task:** Autoregressive next-element prediction — next-patch for vision
(iGPT-style), next-frame for audio. MSE loss on continuous patches/frames.

---

## Phase 1 — Vision (ImageNette, T=256, 1500 steps)

128px, 8×8 patches, param-matched (FFN 766 vs 1024), batch 32, ~8 min on T4.

| arm | val MSE | peak alloc/res | ms/step |
|---|---|---|---|
| V8-A | 0.2578 | 823/960 MB | 155.0 |
| ATTN | 0.2102 | 772/895 MB | 152.0 |

**Attention wins quality clearly (~22% relative).** VRAM 0.93x (attention
leaner at short T — the quadratic is still cheap; V8-A's state/gates cost
more). Throughput tied. Honest finding: V8-A's text advantage does NOT
automatically transfer to vision AR. Open: hyperparameter artifact vs
fundamental (compressive state vs exact pairwise lookup for fine spatial
detail).

## Phase 2 — Audio (LibriSpeech train-clean-100/dev-clean, T=256, 1500 steps)

2.56 s clips (256 mel frames), batch 32, ~37 min on T4.

| arm | val MSE | peak alloc/res | ms/step |
|---|---|---|---|
| V8-A | 0.0552 | 805/916 MB | 415.5 |
| ATTN | 0.0581 | 755/849 MB | 388.2 |

**V8-A wins quality narrowly (~5% relative).** Attention leaner/faster.

**Emerging pattern:** temporal modalities (text, audio) favor V8-A;
spatial (vision) favors attention.

## Phase 3 — Long-sequence scaling (his rung protocol)

One rung per execution, fixed preloaded batch (seed 42), 5 warmup + 30 timed
steps, MEDIAN, immediate save. No OOMs at any rung.

**Vision rungs:**

| T | V8-A | ATTN | V8-A res | ATTN res | Winner |
|---|---|---|---|---|---|
| 256 | 142.0 ms | 137.6 ms | 973 MB | 870 MB | ATTN ~1.03x |
| 529 | 303.3 ms | 345.0 ms | 1749 MB | 1686 MB | V8-A ~1.14x |
| 1024 | 539.2 ms | 808.0 ms | 3129 MB | 3028 MB | V8-A ~1.50x |
| 2025 | 1177.4 ms | 2314.5 ms | 6054 MB | 5933 MB | V8-A ~1.97x |

**Audio rungs** (with explicit warmup/compile reporting):

| T | V8-A timed | V8-A compile | ATTN timed | ATTN compile | Winner |
|---|---|---|---|---|---|
| 256 | 137.4 ms | 5251 ms | 132.9 ms | 915 ms | ATTN ~1.03x |
| 529 | 271.6 ms | 5770 ms | 315.5 ms | 1060 ms | V8-A ~1.16x |
| 1024 | 559.7 ms | 4514 ms | 811.8 ms | 1546 ms | V8-A ~1.45x |
| 2025 | 1059.1 ms | 4945 ms | 2364.9 ms | 3004 ms | V8-A ~2.23x |

**Crossover confirmed between T=256 and T=529 in BOTH modalities.**
Attention slightly leaner VRAM at every rung (quadratic matrix freed per
layer; peak dominated by other activations) — throughput, not memory, is
where O(T) wins. Compile tax: V8-A Triton JIT ~5s vs attention ~1–3s, paid
once. Warmup medians ≈ timed medians (steady state confirmed).

## Phase 4 — Joint multimodal (Flickr8k, paired image–caption)

Paired data only; [caption tokens, image patches] order; loss =
(vis_mse + txt_mse)/2 normalized per modality. 500-step pilot first.

**Pilot** (500 steps): val vis V8-A 0.5955 vs ATTN 0.5861; val txt both
~0.0003 (too easy, uninformative). Gap narrowed 80% vs Phase 1 reference —
consistent with cross-modal rescue, BUT: ImageNette vs Flickr8k + 1500 vs
500 steps = unmatched baseline; 500 steps may show transient compression;
text task too easy. "Interesting signal, do not overclaim."

**Matched 2×2 control** (A=vis/attn, B=vis/v8a, C=joint/attn, D=joint/v8a,
all 500 steps, identical vision-loss definition):

| Arm | Config | VAL vis | ms/step | VRAM a/r |
|---|---|---|---|---|
| A | vis-only / ATTN | 0.6035 | 143.7 | 860/979 |
| B | vis-only / V8-A | 0.5934 | 167.8 | 885/1028 |
| C | joint / ATTN | 0.5670 | 177.5 | 895/990 |
| D | joint / V8-A | 0.5955 | 186.2 | 938/1053 |

Δ_vis-only = −0.0101; Δ_joint = +0.0285; **I = +0.0386.**

**Correction (his):** this does NOT conclusively falsify cross-modal rescue
as a general hypothesis. It shows the hypothesis was not supported by the
observed experiment and its predeclared decision rule. Tensor-level
initialization verification FAILED (shared encoder weights did not match
across arms), introducing an initialization confound that prevents clean
causal attribution to the mixer architecture. Defensible conclusion: the
hypothesis was not supported by the Phase 4 decision test; the observed
interaction favored attention, but the init mismatch prevents clean causal
attribution; the experiment was stopped per its predeclared rule. This does
not establish that cross-modal rescue is universally impossible.

**Evidence classification (his):**
- OBSERVED: joint attention outperformed joint V8-A on the Flickr8k vision
  objective.
- SUGGESTED: attention may exploit linguistic context more effectively in
  this setup.
- NOT ESTABLISHED: that V8-A's spatial weakness cannot be rescued, or that
  this is a general architectural property.

---

## Campaign verdict

**Seed robustness:** V9 text has five-seed support (V8-A 1.312±0.008 vs
attention 1.402±0.024, V8-A winning all five seeds). V12's vision, audio,
scaling, and Phase 4 results are single-seed and preliminary — they require
controlled replication before broadening.

**Directly measured:** V8-A text 1.319 vs 1.419 (V9: 1.312±0.008 vs
1.402±0.024, 5 seeds); V11 stack 839→520 MB (−38%); V12 ImageNette
attention 0.2102 vs V8-A 0.2578; V12 LibriSpeech V8-A 0.0552 vs attention
0.0581; V12 throughput crossover T≈256–529 reaching ~1.97x (vision) /
~2.23x (audio) at T=2025 under tested implementations and conditions;
attention leaner VRAM at every rung.

**Supported but narrow:** the tested temporal tasks favored V8-A on
quality; the tested spatial vision task favored attention — a
modality-dependent pattern in the evaluated configurations, not yet a
general property.

**Complexity scope:** fixed-state recurrence is O(T) under stated
assumptions; measured speedups apply to these implementations, lengths,
hardware, and conditions — not every linear-time model or workload.

**Novelty boundary:** the evidence supports reporting a specific
architecture, its implementation, measured results, and the experimental
record. It does not yet establish general superiority, a universal
modality law, or a field-wide multimodal SSM research gap.

The strongest contribution is the combination of a concrete recurrent
architecture, validated custom-kernel mathematics, measured systems
improvements, parameter-controlled comparisons, multi-seed text evidence,
and an explicit record of where the observed advantages do and do not
appear.

## Future work (if revisiting)

1. Fix initialization at tensor level FIRST (assert equality on shared
   components across arms).
2. Rerun the 2×2 with the same protocol.
3. Not needed immediately unless the answer would change research direction.

## Write-up wording discipline (his)

- The measured crossover and speedups are established for THESE
  implementations and benchmark conditions — not for every linear-time
  model or workload.
- The results support a modality-dependent pattern, but broader claims
  need more datasets and controlled replications.

## Next-experiment protocol (his standing rule)

When resuming: the brief determines the next experiment's question,
baseline, controls, and stopping rule BEFORE implementation begins.
