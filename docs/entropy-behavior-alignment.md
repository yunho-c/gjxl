# Entropy behavior alignment with libjxl

- Status: proposed refactor plan
- Branch: `refactor/maximum-compression`
- Reference libjxl revision: `e8ff09762481785938d8e4e01333ed3917571161`

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

This refactor will therefore establish three serializer behaviors:

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
graph, so a fresh Phase 0 baseline is required before implementation work.

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

## Proposed behavior contract

### Public selection

Add a compression-mode option separate from effort and density:

```cpp
enum class VarDctCompressionMode {
  kAutomatic,
  kMaximumCompression,
};
```

Add `compression_mode = kAutomatic` to `VarDctEncodingOptions` and report the
resolved serializer behavior in `VarDctEncodingSummary` and diagnostic profiles.
The CLI adds `--maximum-compression`.

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

The initial high-density implementation should use the pinned libjxl
28-configuration set so that the reference relationship is explicit. After the
policy is validated, the retained GJXL eight-configuration subset may replace it
only through a separate measured change that reports the byte and time delta.

LZ77 is not part of GJXL's current VarDCT serializer profile and is not required
for this alignment. High density is effort-9-like, not a claim of complete
feature parity with libjxl effort 9.

## Target dataflow

Balanced and high-density paths should share the same single-representation
front half:

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

The two paths should meet again at the existing selected-code section writers.
The refactor should not duplicate final token writing or frame assembly.

## Plan of action

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

## Suggested implementation slices

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

### Initial qualification targets

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
