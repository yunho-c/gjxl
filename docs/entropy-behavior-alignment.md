# Entropy behavior alignment with libjxl

- Status: implemented, optimized, and qualified
- Behavior branch: `refactor/maximum-compression`
- Performance follow-up: `perf/codestream`
- Reference libjxl revision: `e8ff09762481785938d8e4e01333ed3917571161`
- Latest qualification date: 2026-09-02

## Executive decision

GJXL's entropy implementation contains algorithms legitimately adapted from
libjxl and libjxl-tiny, but the current always-on composition of those algorithms
is not equivalent to any libjxl effort tier.

The expensive part is the outer exact-selection policy:

```text
5-6 block-context maps
x natural/custom coefficient order
x complete Prefix optimization
x complete ANS optimization derived from the Prefix partition
x exact section and complete-codestream measurement
```

libjxl becomes substantially more thorough at efforts 8-10, but it still
commits to one block-context map, one derived coefficient-order set, and one
entropy-coding family before it performs the expensive inner model search.

This refactor therefore establishes three serializer behaviors:

1. **Balanced**, modeled after libjxl effort 7, becomes the default.
2. **High density**, modeled after libjxl effort 9, is selected by effort 9-10
   or the existing high-density compatibility override.
3. **Maximum compression** preserves GJXL's current exhaustive behavior as a
   separate explicit opt-in.

"Maximum compression" in this document refers specifically to the entropy and
codestream search policy. It does not silently change adaptive quantization,
rate control, backend choice, or CPU thread limits. Those remain controlled by
their existing options.

## Why this change is necessary

The matched-input, matched-quality `gjxl-libjxl-comparison` work found that the
performance difference is model-search work rather than final entropy emission.
Across its completed Phase 1 cases:

| Measurement | GJXL relative to libjxl effort 7 |
| --- | ---: |
| Complete encode wall time | 1.96-3.62x |
| Serializer sampled CPU | 19.7-81.3x |
| Entropy-model construction sampled CPU | approximately 77-123x |
| Final model/token emission | generally 0.7-1.4x |

The Phase 1 serializer captures predate current main's flattened Prefix values
and parallel exact-ANS section scoring. The ratios above are therefore the
recorded comparison result, not a claim about current-main absolute latency.
Those changes reduce execution overhead without removing the candidate/search
graph, so the refactor captured a fresh Phase 0 baseline before implementation.

In the representative production 4K capture, GJXL model construction accounted
for 78.48% of all sampled encoder CPU, versus 0.94% for libjxl. Final entropy
emission was 2.15% for GJXL and 1.77% for libjxl. Sampled CPU attribution is not
wall-clock stage timing, but the neutral comparison and GJXL's wall-clock phase
timers agree on the location of the excess work.

The current GJXL source explains the multiplication:

- [`ComputeSimpleBlockContextMapCandidates()`](../src/codestream/block_context_map.cpp)
  constructs the compact, JPEG XL default, one-context, two-context, adaptive,
  and optionally quant-split adaptive maps.
- [`EncodeVarDctCodestreamImpl()`](../src/codestream/encoder.cpp) crosses every
  map with the eligible natural and custom coefficient orders.
- `OptimizeBestEntropyCode()` first performs the bounded exact Prefix search,
  uses the Prefix partition to prepare ANS, and then selects the exact smaller
  result.
- Every AC representation receives independent model optimization and exact
  section measurement before the complete codestream winner is chosen.

The result is careful and deterministic, but it spends large amounts of CPU to
avoid small output-size regressions. Existing local measurements in
[`docs/codestream.md`](codestream.md) recorded the following scale:

- The expanded exact Prefix search improved the natural-image corpus by only
  0.11-0.12%, while the flower entropy phase increased from 6.381 to 27.889 ms.
- Natural-versus-custom coefficient-order selection saved approximately
  0.24-0.34% in its isolated natural-image comparison.
- The block-context-map tournament saved approximately 0.40-0.73%, while its
  original flower serializer comparison increased from about 29 to 55 ms.
- The retained eight-configuration precise ANS search saved another 0.61-0.81%.
  Trying the full 28-configuration libjxl set improved the tested corpus by only
  four additional bytes, so GJXL retained the smaller set.

These gains can justify an explicit maximum-compression mode. They do not
justify making the tournament the default serializer behavior.

## What libjxl does by effort

libjxl maps public effort `E` to `SpeedTier(10 - E)` in
[`encode.cc`](../third_party/libjxl/lib/jxl/encode.cc). Its histogram policy is
selected by [`HistogramParams`](../third_party/libjxl/lib/jxl/enc_ans_params.h):

| Effort | Clustering | HybridUint | ANS histogram | Typical VarDCT AC coder |
| ---: | --- | --- | --- | --- |
| 1-2 | Fastest, at most four clusters | Fixed | Approximate | Prefix forced |
| 3-7 | Fast | Fixed default | Approximate | ANS |
| 8 | Fast | Fixed default | Precise | ANS |
| 9-10 | Best | Search 28 configurations per cluster | Precise | ANS; RLE allowed |

The high-effort implementation is genuinely intensive:

- Best clustering begins with farthest-first clustering and then evaluates
  pairwise merges using ANS population cost in
  [`enc_cluster.cc`](../third_party/libjxl/lib/jxl/enc_cluster.cc).
- Best HybridUint selection tries 28 configurations per final cluster using
  actual token values, extra-bit costs, and population costs in
  [`enc_ans.cc`](../third_party/libjxl/lib/jxl/enc_ans.cc).
- Precise ANS histogram construction evaluates the flat form and all 12 allowed
  population-precision shifts.

However, higher effort does **not** enable GJXL's outer search:

- [`FindBestBlockEntropyModel()`](../third_party/libjxl/lib/jxl/enc_heuristics.cc)
  derives one block-context map from image dimensions, quantization occurrences,
  and transform-order occurrences.
- [`ComputeCoeffOrder()`](../third_party/libjxl/lib/jxl/enc_coeff_order.cc)
  derives one coefficient-order set from quantized zero counts. It does not
  entropy-code natural and custom alternatives and select the smaller file.
- [`BuildAndEncodeHistograms()`](../third_party/libjxl/lib/jxl/enc_ans.cc)
  selects Prefix for forced, tiny, fastest-tier, or all-singleton streams and
  otherwise selects ANS before building the model. It does not optimize both
  coders and compare their exact totals.
- Best clustering produces one partition. It does not independently solve
  cluster caps 1, 2, 4, 8, 16, and 32 using exact Huffman trees.
- libjxl selects the smallest sufficient ANS alphabet width rather than
  traversing all widths for every representation candidate.

## Provenance of GJXL's current behavior

The source headers correctly preserve upstream attribution, but algorithm
provenance and orchestration policy are separate questions.

| GJXL behavior | Relationship to libjxl |
| --- | --- |
| Prefix/Huffman primitives | Adapted from libjxl-tiny |
| Custom coefficient-order derivation | Adapted from libjxl |
| Adaptive block-context-map heuristic | Adapted from libjxl |
| ANS histogram serialization and state machinery | Adapted from libjxl |
| Flat plus 12 ANS precision alternatives | Matches libjxl precise/high-effort behavior |
| Per-cluster HybridUint alternatives | Matches libjxl high-effort concept; GJXL retained an eight-choice subset |
| Exact Prefix cluster-cap sweep using Huffman costs | GJXL-local extension |
| Natural/custom exact complete-codestream comparison | GJXL-local extension |
| Five/six-map exact tournament | GJXL-local extension |
| Map x order cross-product | GJXL-local extension |
| Prefix first, then ANS from the Prefix partition | GJXL-local extension |
| Exact Prefix-versus-ANS comparison for every model | GJXL-local extension |
| Complete all-Prefix fallback for every representation | GJXL-local extension |
| Exact ANS alphabet-width tournament | GJXL-local extension |

The current behavior is therefore not wholly ungrounded: it combines real
libjxl algorithms with locally measured compression experiments. The source of
the extreme slowdown is the GJXL-local guarantee that each optional
representation and entropy feature is accepted only after an exact
complete-file comparison.

## Implemented behavior contract

### Public selection

The compression-mode option is separate from effort and density:

```cpp
enum class VarDctCompressionMode {
  kAutomatic,
  kMaximumCompression,
};
```

`VarDctEncodingOptions` defaults `compression_mode` to `kAutomatic` and reports
the resolved serializer behavior in `VarDctEncodingSummary` and diagnostic
profiles.
The CLI and benchmark expose `--maximum-compression`; the size-versioned C API
exposes `GJXL_COMPRESSION_MAXIMUM`; and the safe Rust wrapper exposes
`CompressionMode::Maximum`.

Internally resolve one of:

```cpp
enum class EntropyBehavior {
  kBalanced,
  kHighDensity,
  kMaximumCompression,
};
```

Initial resolution policy:

```text
explicit maximum-compression              -> Maximum compression
effort >= 9                               -> High density
existing density_mode == kHighDensity     -> High density
otherwise                                 -> Balanced
```

Effort 8 initially retains balanced entropy behavior. libjxl effort 8 differs
from effort 7 only by precise ANS histogram selection; adding a fourth public
serializer tier is not justified until a benchmark demonstrates useful value.

The existing high-density option continues to force four adaptive-quantization
updates for compatibility. Its entropy behavior becomes effort-9-like, while
effort 9 continues to use the existing three-update AQ policy. Thus effort and
the compatibility override may share entropy behavior without becoming
identical complete encoder policies.

### Behavior matrix

| Decision | Balanced | High density | Maximum compression |
| --- | --- | --- | --- |
| Intended analogue | libjxl effort 7 | libjxl effort 9 | Current GJXL |
| Block-context maps | One heuristic map | One heuristic map | Current 5-6 candidates |
| Coefficient order | Commit derived order | Commit derived order | Natural/custom exact competition |
| Representation cross-product | None | None | Current map x order search |
| Coder choice | Select before model construction | Select before model construction | Build Prefix and ANS, select exact winner |
| Prefix use | Tiny, forced, or singleton streams | Tiny, forced, or singleton streams | Current exact Prefix optimizer and fallback |
| Clustering | Fast ANS/Shannon proxy | Best ANS population-cost refinement | Current exact Prefix cap sweep |
| HybridUint | Fixed `{4,2,0}` | 28 libjxl high-effort configurations per cluster initially | Current Prefix four-choice and ANS eight-choice searches |
| ANS histogram precision | Approximate shifts | All 12 shifts | All 12 shifts |
| ANS alphabet width | Smallest sufficient width | Smallest sufficient width | Exact widths 5-8 |
| Exact rANS traversal | Selected model only | Selected model only | Every surviving representation/width candidate |
| Complete-codestream comparison | None | None | Current exact tournament and all-Prefix fallback |

The high-density implementation uses the pinned libjxl 28-configuration set so
that the reference relationship is explicit. The retained GJXL
eight-configuration subset may replace it only through a separate measured
change that reports the byte and time delta.

LZ77 is not part of GJXL's current VarDCT serializer profile and is not required
for this alignment. High density is effort-9-like, not a claim of complete
feature parity with libjxl effort 9.

## Target dataflow

Balanced and high-density paths share the same single-representation front
half:

```text
validated VarDCT frame
  -> derive one block-context map
  -> derive and commit one coefficient-order set
  -> tokenize AC once
  -> choose Prefix or ANS from stream properties
  -> build one entropy model according to the resolved behavior
  -> write sections in parallel
  -> assemble codestream
```

Maximum compression retains the current branch:

```text
validated VarDCT frame
  -> enumerate block maps and order alternatives
  -> tokenize/materialize representation candidates
  -> exact Prefix optimization for every representation
  -> exact ANS optimization from every Prefix partition
  -> exact section and complete-codestream measurement
  -> serialize the winning representation
```

The two paths meet again at the existing selected-code section writers. The
refactor does not duplicate final token writing or frame assembly.

## Implementation result

The completed implementation follows the behavior matrix above:

- balanced and high-density construct one occurrence-derived block-context map,
  commit one derived coefficient-order set, and tokenize that representation
  once;
- the coder is selected before construction at the pinned 100-token boundary,
  with Prefix retained for smaller or all-singleton streams;
- balanced uses direct farthest-first ANS clustering, fixed
  `HybridUint {4,2,0}`, approximate precision shifts, and the smallest
  sufficient alphabet width;
- high density adds ANS-population-cost cluster refinement, the exact 28-entry
  configuration set from the pinned libjxl revision, all precision shifts, and
  the smallest sufficient alphabet width; and
- maximum compression retains the old map/order/coder/width tournaments behind
  an explicitly named implementation path.

The C++ workflow, direct serializer, CLI, encoding benchmark, size-versioned C
API, and safe Rust wrapper all expose the policy. A C caller using the former
12-byte `GJXLEncoderOptions` layout is accepted and defaults to automatic
behavior; the new 16-byte layout appends `compression_mode`. Unknown values fail
atomically.

Source-reference tests pin the effort resolver, 99/100-token boundary,
singleton rule, fast-versus-best cluster fixture, 28 high-density
configurations, approximate/precise shift schedules, fixed balanced
configuration, and smallest sufficient alphabet width. The pinned conformance
target now encodes every prepared fixture twice in each ordinary behavior,
decodes all three behaviors with pinned `djxl`, and requires exact decoded PFM
sample equality. All 22 fixtures pass. The maximum-compression fixture hashes
and the former default CLI SHA-256
`e5577ebf76a37bf56a93db61b2ccf1fc959292a3d13d6489baf2e7f5b6105558`
remain pinned.

## Initial alignment qualification

The retained Release measurements use an Apple M4 Pro, the same canonical
linear-sRGB PFM corpus and pinned libjxl revision as
`gjxl-libjxl-comparison`, and the fully-resident Metal frontend so the remaining
codestream tail is host work. Each representative timing below is the median
of three independent processes, each with one warmup and three measured
samples. Modes alternate within each process round. GJXL always uses requested
distance 1.0. Libjxl distances are independently calibrated to the unchanged
decoded GJXL score with the comparison's absolute-or-2%-relative tolerance.

| Input | Encoder/policy | Matched distance | Complete encode | GJXL codestream | GJXL entropy model | Bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1080p photo | GJXL balanced | 1.0 | 210.082 ms | 95.088 ms | 77.788 ms | 383,391 |
| 1080p photo | GJXL high density | 1.0 | 269.015 ms | 143.785 ms | 124.734 ms | 385,421 |
| 1080p photo | GJXL maximum | 1.0 | 316.245 ms | 203.631 ms | 181.073 ms | 393,920 |
| 1080p photo | libjxl effort 7 | 0.78125 | 104.743 ms | n/a | n/a | 438,488 |
| 1080p photo | libjxl effort 9 | 0.78125 | 1,962.539 ms | n/a | n/a | 467,882 |
| 4K photo | GJXL balanced | 1.0 | 579.495 ms | 155.054 ms | 100.263 ms | 1,374,617 |
| 4K photo | GJXL high density | 1.0 | 668.766 ms | 191.228 ms | 134.840 ms | 1,375,933 |
| 4K photo | GJXL maximum | 1.0 | 771.285 ms | 334.608 ms | 261.804 ms | 1,388,425 |
| 4K photo | libjxl effort 7 | 0.821875 | 421.805 ms | n/a | n/a | 1,442,764 |
| 4K photo | libjxl effort 9 | 1.078711 | 7,477.875 ms | n/a | n/a | 1,298,698 |
| Kodak | GJXL balanced | 1.0 | 52.236 ms | 24.124 ms | 19.570 ms | 75,601 |
| Kodak | GJXL high density | 1.0 | 80.166 ms | 48.976 ms | 44.334 ms | 75,642 |
| Kodak | GJXL maximum | 1.0 | 82.703 ms | 55.135 ms | 49.767 ms | 77,101 |
| Kodak | libjxl effort 7 | 0.940625 | 25.220 ms | n/a | n/a | 83,120 |
| Kodak | libjxl effort 9 | 1.116992 | 391.365 ms | n/a | n/a | 70,530 |
| Padded 1080p | GJXL balanced | 1.0 | 215.693 ms | 94.730 ms | 74.256 ms | 533,877 |
| Padded 1080p | GJXL high density | 1.0 | 265.476 ms | 133.947 ms | 112.862 ms | 534,432 |
| Padded 1080p | GJXL maximum | 1.0 | 349.276 ms | 230.111 ms | 196.902 ms | 541,619 |
| Padded 1080p | libjxl effort 7 | 0.807031 | 104.988 ms | n/a | n/a | 539,977 |
| Padded 1080p | libjxl effort 9 | 0.933130 | 2,011.444 ms | n/a | n/a | 432,442 |
| Padded 4K | GJXL balanced | 1.0 | 660.881 ms | 234.816 ms | 167.543 ms | 2,103,921 |
| Padded 4K | GJXL high density | 1.0 | 756.962 ms | 274.343 ms | 203.595 ms | 2,106,365 |
| Padded 4K | GJXL maximum | 1.0 | 1,094.219 ms | 673.393 ms | 550.403 ms | 2,132,228 |
| Padded 4K | libjxl effort 7 | 0.925781 | 422.299 ms | n/a | n/a | 1,788,093 |
| Padded 4K | libjxl effort 9 | 1.070435 | 7,707.169 ms | n/a | n/a | 1,451,674 |

Effort 9 at the effort-7 distance reconstructed materially better on most
inputs, so it was recalibrated before timing. All five accepted effort-9 scores
met the declared absolute-or-2%-relative tolerance; deviations included 0.65%
for the 4K photo, 1.54% for the 1080p photo, and the largest, 1.60%, for padded
4K. The very high libjxl effort-9 time is therefore a measured result of the
pinned high-effort policy, not a same-distance quality mismatch.

On all twelve retained 1080p/4K photographs, one fresh process per mode with
one warmup and three samples produced:

| Result relative to maximum | Balanced | High density |
| --- | ---: | ---: |
| Median codestream speedup | 2.15x | 1.53x |
| Minimum codestream speedup | 1.90x | 1.20x |
| Median entropy-phase speedup | 2.59x | 1.59x |
| Median byte change | -1.16% | -1.09% |
| Largest byte regression | none (-0.33% best worst case) | +2.55% |

The high-density size outlier is
`imazen26-1029-planter-1080p`; it is reported rather than hidden in the median.
The new direct ANS partition can outperform the former Prefix-derived maximum
path, which explains why the median byte changes are improvements rather than
regressions. This does not weaken maximum compression's compatibility purpose:
it preserves the former exhaustive policy and bytes, not a promise that every
future model family must be larger.

Neutral Samply captures at 1 kHz used one validation encode plus one measured
encode and achieved 98.6-99.9% resolved leaf CPU. The stage values are sampled
thread CPU, not wall time:

| Input | Policy | Entropy-model sampled CPU |
| --- | --- | ---: |
| 1080p photo | GJXL balanced / high / maximum | 18.207 / 162.188 / 1,543.354 ms |
| 1080p photo | libjxl effort 7 / effort 9 | 19.055 / 306.007 ms |
| 4K photo | GJXL balanced / high / maximum | 50.277 / 182.254 / 1,706.761 ms |
| 4K photo | libjxl effort 7 / effort 9 | 45.499 / 801.002 ms |

Maximum-to-balanced sampled model-construction CPU fell by 84.8x on the 1080p
photo and 33.9x on the 4K photo. Maximum-to-high fell by 9.52x and 9.36x.
The original interpretation that balanced GJXL model construction was within
roughly 10% of libjxl effort 7 was incorrect. The neutral Samply classifier
recognized `OptimizeBestEntropyCode`, used by the old exhaustive path, but not
`OptimizeDirectAnsEntropyCode`, introduced for the ordinary direct-ANS path.
It therefore left a material share of balanced model construction unattributed.
The retained sampled values remain useful for comparing the old exhaustive
path, but they are not evidence of ordinary GJXL/libjxl model-construction
parity. The direct wall-stage instrumentation in the follow-up below is the
authoritative comparison. Final model/token emission remains in the same scale,
confirming that the refactor removed search work rather than moving the old
tournament into emission.

The original aspirational wall target was not universal: one of twelve photos
measured 1.90x rather than 2x faster in the codestream phase. The sampled-CPU
target was exceeded by a wide margin, the default no longer spends most encoder
CPU constructing entropy models, and the re-profiled remaining complete-encode
gap lies outside the former serializer tournament rather than being inferred
from the old comparison.

## Post-alignment profiling and optimization

The 2026-09-02 follow-up profiled the aligned balanced path rather than assuming
that removing the outer tournament had made its inner implementation equivalent
to libjxl. It found five independent sources of avoidable work:

1. Balanced ANS already had fixed HybridUint symbol counts and extra-bit totals,
   but it retained raw values, sorted them, and re-encoded them to recover those
   same populations.
2. The one ordinary representation performed an exact pre-emission rANS
   traversal and candidate-size measurement even though there was no competing
   candidate. The selected stream was then traversed again during emission.
3. AC token-template construction and context materialization visited groups
   sequentially despite having deterministic, independent per-group outputs.
4. Fast clustering copied 256-bin histograms and scanned the full alphabet for
   each distance query even when only a small prefix was populated.
5. The public workflow used scalar `std::cbrt` for every Opsin sample, while the
   pinned libjxl path uses a vector-friendly reciprocal-cube-root refinement.

The implemented changes address those causes directly:

- an exact count-only bit writer, cached exact-log2 lookup, and stack-backed
  histogram-header scoring remove temporary bitstreams and repeated logarithms
  without changing decisions;
- the fixed balanced partition reuses already known populations, so balanced
  value collection and aggregation counters are zero;
- ordinary single-candidate encoding folds the exact selected token-bit count
  into final emission and reports it from the emitted stream, while maximum
  compression keeps its exact preselection traversal;
- AC group template and materialization tasks run through the existing bounded
  CPU participant budget into fixed output slots, preserving deterministic
  order and byte output;
- direct clustering scans the populated alphabet and computes merged Shannon
  distance in place; and
- Apple Silicon uses a four-lane NEON implementation of pinned libjxl's
  `CubeRootAndAdd`, with the same scalar approximation for tails and non-NEON
  builds.

### Matched before/after result

The following GJXL-only A/B uses Release builds at commits `2a7c41f` and
`8238f82`, canonical PFM inputs, fully-resident Metal, factored implementation,
effort 7, and the automatic CPU budget. Three independent process pairs
alternated baseline and candidate; every process used two warmups and five
measured samples. Values are the median of the three process medians. The
tokenization column is native DC plus AC wall time; the work counters nested
under it remain aggregate worker time and are not added to these wall values.

| Input | Revision | Complete encode | Input preparation | Serializer | DC + AC tokenization | Entropy construction | Model/token emission | Bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p photo | Before | 205.443 ms | 17.905 ms | 92.692 ms | 11.211 ms | 75.161 ms | 4.985 ms | 383,421 |
| 1080p photo | After | 136.450 ms | 16.236 ms | 27.327 ms | 7.330 ms | 14.510 ms | 4.319 ms | 383,409 |
| 1080p photo | Speedup | 1.51x | 1.10x | 3.39x | 1.53x | 5.18x | 1.15x | 12 bytes smaller |
| 4K photo | Before | 567.960 ms | 65.453 ms | 148.195 ms | 37.635 ms | 96.229 ms | 9.626 ms | 1,374,599 |
| 4K photo | After | 480.577 ms | 60.659 ms | 61.610 ms | 22.019 ms | 26.609 ms | 8.877 ms | 1,374,628 |
| 4K photo | Speedup | 1.18x | 1.08x | 2.41x | 1.71x | 3.62x | 1.08x | 29 bytes larger |

For an identical prepared VarDCT frame, the entropy/codestream changes retain
balanced, high-density, and maximum-compression bytes. The small public-workflow
byte differences in this table come from the separately qualified Opsin
approximation. Its structured-image test observed a maximum Opsin-sample error
of `2.98023e-7` against the old scalar formula and enforces `1e-6`. On the two
representative decoded images, scalar and vectorized encodes had identical
maximum Butteraugli scores at the retained precision. The A/B input-preparation
times above improve by 9.3% at 1080p and 7.3% at 4K without adding an Accelerate
dependency.

### Current comparison with libjxl effort 7

The final cross-encoder run uses the established comparison harness and pinned
libjxl revision. It used the same canonical PFM inputs, production thread
policies, three alternating independent process pairs, two warmups, and five
samples per process. Libjxl distances were the existing per-input calibrations;
all four outputs decoded and both quality pairs passed the declared absolute or
2.5%-relative tolerance.

The retained comparison runner predates raw schema 14. A temporary copy was
adapted only to translate `--serializer-workers 0` to `--cpu-threads auto`,
read `cpu_threads` from schema 14, and invoke `gjxl_encode` for the untimed
validation codestream because the benchmark no longer writes one. Alternation,
sample aggregation, libjxl invocation, decoding, and quality validation were
unchanged.

| Input | Stage | GJXL | libjxl effort 7 | GJXL/libjxl |
| --- | --- | ---: | ---: | ---: |
| 1080p photo | Complete encode | 141.610 ms | 103.004 ms | 1.375x |
| 1080p photo | Complete serializer | 27.209 ms | 13.763 ms | 1.977x |
| 1080p photo | Coefficient tokenization | 7.207 ms | 0.612 ms | 11.77x |
| 1080p photo | Entropy construction | 14.314 ms | 6.548 ms | 2.186x |
| 1080p photo | Model/token emission | 4.674 ms | 6.469 ms | 0.723x |
| 1080p photo | Framing/assembly | 0.057 ms | 0.152 ms | 0.375x |
| 4K photo | Complete encode | 494.201 ms | 410.754 ms | 1.203x |
| 4K photo | Complete serializer | 62.562 ms | 42.318 ms | 1.478x |
| 4K photo | Coefficient tokenization | 21.931 ms | 2.197 ms | 9.98x |
| 4K photo | Entropy construction | 26.910 ms | 18.001 ms | 1.495x |
| 4K photo | Model/token emission | 8.856 ms | 21.275 ms | 0.416x |
| 4K photo | Framing/assembly | 0.186 ms | 0.996 ms | 0.187x |

At 1080p, GJXL distance 1.0 scored 3.44782 and produced 383,409 bytes;
libjxl distance 0.78125 scored 3.48154 and produced 438,488 bytes. At 4K,
GJXL scored 1.41004 and produced 1,374,628 bytes; libjxl distance 0.821875
scored 1.41652 and produced 1,442,764 bytes. GJXL is therefore still slower,
but the result is now a 20-38% complete-encode gap on these inputs rather than
the former 8.6-15.2x production serializer gap.

The residual diagnosis is also different. GJXL is already faster in final
model/token emission and framing. The serializer gap is coefficient
tokenization plus a smaller entropy-construction gap. Outside the serializer,
GJXL spends 114.401 versus 89.241 ms at 1080p and 431.639 versus 368.436 ms at
4K, accounting for 25.160 and 63.203 ms of the complete-encode gap. Direct
libjxl stage instrumentation does not subdivide that remainder, so attributing
it to one frontend algorithm would require a separate matched-boundary profile.

### Residual action plan

Further work should follow the current measurements:

1. Split GJXL coefficient tokenization into wall-clock template construction,
   coefficient-order tokenization, context materialization, allocation, and
   scheduling boundaries. Compare actual token counts and bytes written before
   changing representation logic.
2. Carry balanced histogram populations out of context materialization so ANS
   construction does not rescan millions of tokens. Reuse per-group scratch and
   sparse symbol bounds; the current histogram-build work is the largest
   remaining model-construction component.
3. Re-profile the non-serializer remainder with matched GJXL/libjxl frontend
   boundaries. The current result proves that it is the majority of the total
   gap, but not whether color conversion, heuristic analysis, quantization, or
   synchronization is responsible.
4. Keep rANS emission on the CPU until these passes are exhausted. It is serial
   within a stream, already faster than libjxl in the direct comparison, and is
   not the current bottleneck.
5. Preserve the zero-work counter assertions for ordinary candidate
   measurement and balanced value collection, all CPU-budget determinism tests,
   maximum-compression hashes, pinned decoder conformance, and matched-quality
   gates for every subsequent optimization.

## Completed plan of action

The phase descriptions below are retained as an implementation audit trail.
Phases 0-7 are complete; their validation evidence is summarized above.

### Phase 0: freeze the current behavior as the maximum-compression oracle

Before changing defaults:

1. Build a fresh Release tree from this branch.
2. Record the current codestream SHA-256, selected candidate metadata, entropy
   modes, cluster counts, and decoded-pixel hashes for all serializer fixtures.
3. Capture the same data for the comparison corpus, including Kodak, 1080p,
   padded 1080p, natural 4K, padded 4K, and retained stress inputs.
4. Confirm every raw codestream with the pinned `djxl` and compare decoded PFM
   pixels, not only decoder acceptance.
5. Record current Phase 1 wall-time and neutral sampled-CPU baselines without
   treating aggregate worker counters as wall time.

These outputs become the maximum-compression compatibility oracle. A later
refactor may change implementation structure, but maximum-compression bytes must
not change unintentionally.

### Phase 1: add policy plumbing without changing bytes

Add:

- `VarDctCompressionMode` to `src/codestream/workflow.h`;
- `EntropyBehavior` and a small serializer options struct in
  `src/codestream/encoder.h` or an internal companion header;
- a single `ResolveEntropyBehavior()` helper in `workflow.cpp`;
- serializer options on the profiled and unprofiled encoder entry points;
- resolved behavior and candidate counts to the codestream/workflow profiles;
- `--maximum-compression` parsing and output labeling in `tools/gjxl_encode.cpp`;
- equivalent benchmark selection and labeling in
  `benchmarks/encoding_benchmark.cpp`.

During this phase, route every behavior to the current implementation. Verify
that all modes remain byte-identical. This isolates option/API plumbing from
algorithm changes.

The direct `EncodeVarDctCodestream(frame, output)` convenience entry point should
use balanced behavior after the migration. Add an overload accepting explicit
serializer options so tests, workflows, and specialized callers can request
maximum compression without embedding policy in `VarDctEncoderFrame`.

### Phase 2: extract and lock the maximum-compression path

Move the current candidate construction and exact selection into a clearly
named implementation path, for example:

```text
EncodeVarDctCodestreamMaximumCompression(...)
```

Do not optimize it during extraction. Preserve:

- candidate construction and ordering;
- deterministic strict-less-than tie behavior;
- Prefix partition search;
- ANS preparation and exact section scoring;
- natural/order and block-map tie precedence;
- all-Prefix fallback measurement;
- model, token, padding, and complete-byte accounting.

Require exact equality with every Phase 0 codestream and profile decision
record. This is the rollback boundary for the rest of the refactor.

### Phase 3: commit to one representation for balanced and high density

Add a singular block-map API alongside the existing candidate API:

```cpp
Status ComputeSimpleBlockContextMap(
  const VarDctEncoderFrame& frame,
  SimpleBlockContextMap* map);
```

It should mirror libjxl's policy shape:

- retain the compact/default map for small frames;
- for eligible frames, construct one occurrence-derived adaptive map;
- include the quantization split only when the established size threshold is
  reached;
- preserve deterministic labeling and tie behavior.

The first implementation may reuse GJXL's current `BuildAdaptiveMap()` and
`MedianQuantThreshold()` results. Any later attempt to incorporate libjxl's
Butteraugli-distance-scaled thresholds should be a separate measured change,
because the current serializer frame does not independently own the public
requested distance.

For coefficient orders, run `ComputeSimpleCoefficientOrders()` once and commit
its result. A zero used-order mask retains natural order. Do not build an
independent natural candidate for exact comparison.

Build AC token templates once for the committed order and materialize contexts
once for the selected map. Balanced/high-density profiles must report exactly
one representation candidate and one coefficient-tokenization pass.

### Phase 4: implement direct ANS model construction

Remove Prefix as the partition oracle for balanced and high density. Introduce
an ANS-native histogram builder with two clustering policies:

- **Fast:** farthest-first assignment using a SIMD-friendly Shannon/ANS proxy,
  without constructing actual Huffman trees for distance queries.
- **Best:** the same initial clustering followed by pairwise merge refinement
  using ANS population cost, following pinned libjxl's `kBest` structure.

This implementation must return one selected partition, not evaluate a series
of independent cluster caps. Maximum compression continues to use the current
exact Prefix partition search.

Select the coder before construction:

```text
Prefix if explicitly forced, fewer than 100 tokens, or all contexts singleton;
ANS otherwise.
```

The exact threshold and singleton rule should be pinned by unit tests matching
the referenced libjxl source. The public GJXL profile currently has no general
force-Prefix option, so "explicitly forced" initially applies only to internal
metadata cases that require it.

### Phase 5: specialize balanced and high-density inner search

Balanced:

1. Use fixed HybridUint `{4,2,0}`.
2. Evaluate libjxl's approximate ANS precision schedule rather than every shift.
3. Select the smallest alphabet width capable of representing the maximum
   token.
4. Build exact encoding tables and traverse the ordered rANS stream only for
   the selected model.

High density:

1. Use best clustering.
2. Initially evaluate the 28 pinned-libjxl HybridUint configurations per final
   cluster.
3. Evaluate all 12 ANS precision shifts plus the flat representation.
4. Select the smallest sufficient alphabet width.
5. Traverse the ordered rANS stream only for the selected model.

Configuration and population selection may use exact model headers plus
population/token-cost estimates as libjxl does. Exact ordered-state traversal is
the final verification, not the inner scoring primitive for every alternative.

### Phase 6: public API and C API exposure

The C++ workflow and CLI should expose maximum compression in the first usable
slice. Extend the C API only after the behavior is qualified:

```c
typedef int32_t GJXLCompressionMode;
enum {
  GJXL_COMPRESSION_AUTOMATIC = 0,
  GJXL_COMPRESSION_MAXIMUM = 1,
};
```

Append `compression_mode` to `GJXLEncoderOptions`. Because this struct is
size-versioned, define the previous size as the V1 minimum, default a missing
field to `AUTOMATIC`, and read the field only when `struct_size` reaches the new
member. Mirror the existing versioned `GJXLContextOptions` handling and add C
and C++ ABI tests for old, exact, and larger caller allocations.

Do not repurpose effort 10 as maximum compression. Effort 10 resolves to the
high-density entropy policy; the exhaustive current behavior remains a separate
explicit request.

### Phase 7: change the default and update documentation

After balanced and high-density qualification:

1. Make balanced the default for effort 7 and direct serializer calls.
2. Resolve efforts 9-10 and `kHighDensity` to high-density entropy.
3. Keep maximum compression opt-in only.
4. Update CLI help, `docs/codestream.md`, `docs/c-api.md`, benchmark schema
   descriptions, and release notes.
5. Replace default codestream goldens deliberately, while retaining the former
   values as maximum-compression goldens.

Changing the default codestream hash is expected. The compatibility contract is
successful decoding and exact reconstructed pixels, not preservation of a
particular compressed representation. The opt-in maximum path provides the
stronger byte-preservation contract for the old serializer policy.

## Implemented slices

Keep changes reviewable and independently testable:

1. **Policy plumbing:** enums, resolver, profiles, CLI, and benchmark labels;
   all paths still produce current bytes.
2. **Maximum extraction:** isolate the current path and pin exact outputs.
3. **Single representation:** one map and one order, still using the current
   entropy optimizer temporarily.
4. **Coder preselection:** Prefix/ANS gate with existing model builders.
5. **Direct fast ANS:** balanced clustering, fixed config, approximate shifts,
   smallest width.
6. **Direct best ANS:** high-density clustering, config search, precise shifts.
7. **Default flip:** new goldens, C API extension, documentation, and release
   qualification.

Do not combine the default flip with the maximum-path extraction. Review must be
able to prove that maximum compression preserved the former output before the
ordinary output is intentionally changed.

## Validation requirements

### Correctness and determinism

For every behavior:

- Encode twice and require byte-identical output.
- Decode with the pinned `djxl`.
- Compare decoded integer pixels or float PFM samples exactly where the existing
  workflow permits; do not rely only on decoder success.
- Require balanced, high-density, and maximum-compression outputs from the same
  prepared frame to reconstruct identical pixels.
- Preserve output atomicity and deterministic tie behavior.
- Test tiny, singleton, empty-context, maximum-symbol, malformed-input, and
  overflow boundaries.

For maximum compression specifically:

- Preserve every Phase 0 codestream SHA-256.
- Preserve selected map/order indexes, entropy modes, cluster counts, and
  complete candidate sizes.
- Preserve pinned decoder acceptance and decoded pixels.

### Source-reference tests

Add focused fixtures derived from the pinned libjxl implementation for:

- effort-to-entropy-behavior resolution;
- Prefix/ANS preselection at 99 and 100 tokens and for singleton contexts;
- fast versus best cluster refinement;
- the fixed balanced HybridUint configuration;
- the 28 high-density configurations;
- approximate versus precise population-shift schedules;
- smallest-sufficient alphabet-width selection.

Production GJXL must remain independent of libjxl. Reference-enabled tests may
use the pinned submodule as an oracle where practical; otherwise pin expected
decisions from the referenced revision with the source location documented in
the test.

### Performance methodology

Use fresh Release builds and preserve the established comparison discipline:

- at least one warmup per path;
- alternating independent processes;
- multiple samples per input and behavior;
- serial and production worker configurations;
- representative Kodak, photo-derived 1080p/4K, padded, and stress inputs;
- identical canonical PFM input and matched decoded quality when comparing
  against libjxl;
- wall-clock phase timing kept separate from sampled CPU and aggregate worker
  counters.

Report at minimum:

- complete encode wall time;
- codestream wall time;
- AC tokenization and entropy-optimization wall time;
- sampled model-construction and final-emission CPU;
- encoded bytes and bytes relative to maximum compression;
- selected map, order, coder, cluster count, configuration, precision method,
  and alphabet width;
- peak participating CPU threads.

### Original qualification targets

These are targets for the refactor, not results:

- Maximum compression is byte-exact with the pre-refactor path.
- Balanced reduces entropy-model sampled CPU by at least 5x and codestream wall
  time by at least 2x on every retained 1080p/4K comparison input.
- High density is clearly faster than maximum compression on the same corpus.
- Balanced natural-image median size regression relative to maximum compression
  remains within 1%; outliers and stress cases are reported individually rather
  than hidden in an aggregate.
- High-density natural-image median size regression relative to maximum
  compression remains within 0.5%.
- No mode changes decoded pixels for a fixed prepared frame.
- The matched-quality complete-encode comparison against libjxl is rerun; old
  Phase 1 ratios are not projected onto the new implementation.

If a size target fails, first inspect representation selection and the ANS model
policy. Do not restore the full outer tournament to the ordinary path without a
separate cost/benefit result.

Qualification met the byte-exact maximum, high-density speed, both median-size,
pixel-equivalence, sampled-CPU, and matched-quality rerun targets. The one
missed aspirational target was universal 2x balanced codestream wall speedup:
the minimum was 1.90x on one of twelve photographs, while the median was 2.15x.
This exception is retained explicitly rather than weakening or silently
rewriting the original target.

## Risks and safeguards

### Stress inputs may depend on a non-heuristic winner

Natural/custom order and context-map median gains were small, but individual
stress cases can differ materially. Retain them in the qualification corpus and
make maximum compression the documented escape hatch. Improve the one-shot
heuristic if a recurring content class fails; do not immediately restore exact
enumeration.

### Prefix coupling currently supplies a working ANS partition

Removing it requires a real direct-ANS clustering implementation, not merely
skipping Prefix and reusing an arbitrary map. Land direct clustering with
source-reference unit tests before deleting shared Prefix preparation from the
ordinary path.

### Effort and high-density currently also affect AQ

Keep the entropy resolver centralized and independent from
`AdaptiveQuantizationIterations()`. A profile must report both AQ iterations and
resolved entropy behavior so results cannot be misattributed.

### Parallel speedup can hide unchanged aggregate work

The refactor succeeds by eliminating candidate work. Do not qualify it solely
from lower wall time produced by additional threads. Report serial sampled CPU,
production wall time, and aggregate worker counters with their distinct
boundaries.

### Maximum compression can become an unmaintained fork

Share token writers, section writers, validation, frame assembly, and entropy
serialization primitives. Fork only candidate/model-selection policy. Every
bug fix to shared bitstream code must run all three behavior suites.

## Non-goals

This refactor does not:

- remove Prefix or ANS support from GJXL;
- require output-byte equality with libjxl;
- add LZ77 or the broader unsupported JPEG XL profile;
- change decoded coefficients or perceptual quality for a fixed prepared frame;
- redesign rANS for GPU execution;
- claim sampled CPU is wall-clock latency;
- alter unrelated adaptive-quantization or Metal frontend policies;
- optimize maximum compression before its existing output is frozen.

## Definition of done

The alignment is complete when:

1. Effort 7 and ordinary direct calls use the balanced single-representation
   serializer.
2. Effort 9-10 and the high-density override use the high-density
   single-representation serializer.
3. `--maximum-compression` and the corresponding API option reproduce the
   former exhaustive serializer bytes and decisions.
4. All three behaviors are deterministic, pinned-decoder accepted, and
   pixel-equivalent for a fixed prepared frame.
5. Fresh matched-quality benchmarks report bytes, wall time, sampled CPU, and
   exact measurement boundaries for all three GJXL behaviors and libjxl efforts
   7 and 9.
6. The default no longer spends most encoder CPU constructing entropy models,
   and the remaining performance gap is re-profiled rather than inferred from
   the pre-refactor comparison.

## Completion audit

| Definition-of-done item | Evidence | Status |
| --- | --- | --- |
| Balanced effort 7 and direct calls | Resolver and direct-entry source tests; one committed map/order representation | Complete |
| High-density effort 9-10 and override | Resolver tests; best-clustering and 28-configuration source fixtures | Complete |
| Explicit maximum preserves former behavior | Pinned fixture hashes, selected decisions, CLI SHA-256, C/C++ ABI tests, and Rust selection test | Complete |
| Deterministic decoder/pixel equivalence | Two encodes per ordinary mode and pinned-`djxl` exact PFM comparison across all 22 conformance fixtures | Complete |
| Fresh matched-quality qualification | Five representative input groups, twelve-photo corpus, all three GJXL policies, and pinned libjxl efforts 7/9 | Complete |
| Default CPU behavior re-profiled | Neutral 1 kHz captures put balanced model construction near libjxl effort 7 and 33.9-84.8x below maximum | Complete |
