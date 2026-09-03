# Amdahl leakage after entropy-behavior alignment

- Status: active implementation roadmap
- Original profile revision: `af3a9e6`
- Priority-5 qualification baseline: `e34efca`
- Reference libjxl revision: `e8ff09762481785938d8e4e01333ed3917571161`
- Profile date: 2026-09-02
- Related analysis: [entropy behavior alignment](entropy-behavior-alignment.md)

## Executive finding

After GJXL stopped running its former exhaustive entropy/codestream tournament
at ordinary efforts, the remaining effort-7 gap to libjxl was no longer caused
by one dominant serializer search. It was distributed across host preparation,
full-image validation and copying, chromaticity statistics, GPU-to-host frame
handoff, coefficient-token preparation, and ANS histogram construction.

The main Metal algorithms are not the evidence-backed problem:

- AC-strategy candidate arithmetic runs efficiently on Metal. The host still
  performs the exact non-overlap merge and deterministic tie-breaking, but the
  measured combined GJXL path was not slower than libjxl's CPU AC search.
- Initial adaptive quantization is fast on Metal.
- GJXL's resident Butteraugli-controlled AQ is much faster than libjxl's
  effort-8-and-higher CPU quantizer search when those iterative policies are
  compared directly.
- Final rANS/model emission and final codestream framing are already faster
  than the corresponding instrumented libjxl phases.

The problem is Amdahl leakage around those kernels: work that remains serial or
host-resident becomes the end-to-end limit even when the central image kernels
are accelerated. The current effort-7 profile also performs two AQ updates,
unlike libjxl effort 7. Changing effort 7 to a zero-update policy is priority 1,
but that work is intentionally outside this document's implementation scope.

The roadmap priorities are:

| Priority | Area | Current signal or status | Next implementation target |
| ---: | --- | ---: | --- |
| 2 | Preparation and storage | Purgeable exact-capacity AQ leases: warm padded-4K complete encode 371.62 -> 340.15 ms | Complete; keep Butteraugli storage per-encode |
| 3 | Color conversion, validation, and copies | Direct workflow transform plus identity-bound finite-input provenance implemented; redundant scans cost 4.61 ms at 1080p and 18.49 ms at 4K | Qualified; proceed to priority 4 |
| 4 | Quantization-matrix chromaticity statistics | Gate plus trusted NEON pass implemented; retained effort-7 preparation improved by 5.39 ms at 1080p and 25.49 ms at 4K | Qualified; incorporate effort-7 update policy and re-profile |
| 5 | Metal readback and frame assembly | Mapped-source assembly implemented; padded-4K total improved 9.05 ms and peak RSS fell 73.8 MiB | Qualified; defer GPU packing and serializer views until after priority 6 |
| 6 | Residual serializer | 62.6 ms GJXL versus 42.3 ms libjxl | Give one-representation encoding a one-pass token path |

These signals are not additive estimates. The sampled function values include
parallel aggregate CPU, several operations are nested, and allocator/memory
traffic appears beneath multiple stages. Each retained change must therefore be
qualified with an alternating end-to-end wall-clock comparison.

## Measurement snapshot and interpretation

The current matched-quality comparison used canonical PFM inputs, production
thread policies, three alternating independent process pairs, two warmups, and
five measured samples per process. The table reports the median of the process
medians:

| Input | Encoder | Complete encode | Serializer | Coefficient tokenization | Entropy construction | Model/token emission |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1080p photo | GJXL | 141.610 ms | 27.209 ms | 7.207 ms | 14.314 ms | 4.674 ms |
| 1080p photo | libjxl effort 7 | 103.004 ms | 13.763 ms | 0.612 ms | 6.548 ms | 6.469 ms |
| 4K photo | GJXL | 494.201 ms | 62.562 ms | 21.931 ms | 26.910 ms | 8.856 ms |
| 4K photo | libjxl effort 7 | 410.754 ms | 42.318 ms | 2.197 ms | 18.001 ms | 21.275 ms |

At 4K, GJXL spent approximately 61.5 ms in input preparation, 369.6 ms in the
quantization pipeline, and 62.6 ms in codestream serialization. The total gap
was 83.4 ms. The serializer accounted for 20.2 ms of it; the remaining 63.2 ms
was outside the serializer.

Three timing distinctions are essential:

1. Complete encode and explicitly instrumented stage durations are wall time.
2. Samply thread-CPU deltas and GJXL `_work` counters are aggregate worker time.
   They can exceed the containing wall stage and cannot be summed into an
   end-to-end latency claim.
3. The GJXL and libjxl coefficient-tokenization labels are not identical.
   GJXL's phase includes coefficient-order derivation, while libjxl computes
   its orders immediately before entering its tokenization timer. The raw
   9.98x 4K ratio is therefore not a pure token-loop ratio.

The current sampled capture nevertheless supplies useful causal evidence. Per
4K encode, it attributes about 42 ms of aggregate CPU to `__bzero` and 29 ms to
`memmove`. GJXL's own work counters attribute about 14.5 ms to coefficient-order
derivation, 20.9 ms to AC template construction, 4.1 ms to context
materialization, and 19.3 ms to direct-ANS histogram construction. These values
overlap in wall time, but they identify actual repeated memory passes.

## Priority 2: prepare only the selected execution path

### Landed first slice: two output contracts

Commit `3d1b93b` and the first storage slice recorded in `40b6a11` made fully
resident Metal the default for qualified automatic single-target encodes and
removed unused host storage from that path. The implementation does not carry a
six-way execution plan. It uses two concrete output contracts, with the existing
materialization flags handling only optional results inside those contracts:

| Caller or mode | Output contract | Host behavior |
| --- | --- | --- |
| Metal exact coefficients | Lean encoding | Materializes preprocessed Opsin lazily, then returns the frame, score history, and control result |
| Metal fully resident or throughput | Lean encoding | Returns the frame, score history, and control result without a host preprocessed-Opsin image |
| CPU codestream adapter | Complete compatibility | Allocates `PipelineStorage` only if CPU is actually selected |
| Metal maximum-throughput adapter | Complete compatibility | Allocates `PipelineStorage` only when this explicit frame-only mode is selected |
| Public AQ and quantization-pipeline APIs | Complete diagnostics | Preserves the promised quant fields, masks, block-distance map, reconstructed RGB, frame, and scores |
| Maximum-error control or final-score collection | Modifier on either contract | Adds only the required control result or terminal perceptual evaluation; it is not another storage plan |

`PreparedWorkflow` now keeps a small `EncodingArtifacts` result for the normal
encoding path. Its complete `PipelineStorage` is a lazy compatibility allocation
used only by the CPU and maximum-throughput adapters. The default resident path
therefore does not construct host diagnostic quant fields, block distances, or
reconstructed RGB.

`PreparedQuantizationPipeline` no longer owns a second set of final quantization,
block-distance, reconstructed-RGB, frame, or score-history results. Adaptive-
quantization providers already promise atomic output, so they commit directly
to the selected caller-owned destination. The host preprocessed-Opsin image is
also allocated only when the exact or CPU implementation needs it. CPU
Butteraugli reference preparation is eager only for an explicitly forced CPU
encode and remains available lazily for automatic fallback.

Some initial quantization, strategy, and pixel-mask fields remain because the
resident frontend currently reads them back and uses them for AC search and AQ
reconfiguration. Their retention follows actual use rather than the final
diagnostic materialization flags.

Automatic backend policy is narrower than the original plan assumed. Qualified
automatic single Butteraugli-target encoding may select resident Metal and fall
back to CPU if Metal is unavailable. Automatic target-byte, target-BPP, and
maximum-error searches stay entirely on CPU so one search never mixes CPU and
resident rate curves. Forced Metal continues to support resident maximum-error
control. Lazy compatibility storage is consequently needed for genuine fallback,
not for arbitrary backend changes between target-size attempts.

### First-slice measurements

The retained A/B used Release builds and five alternating independent process
pairs per workload, with two warmups and five measured samples per process. The
table reports the median of process medians. Measurements were captured on
patch-equivalent commits `535acfb` and `7732b9f`; their changes are recorded in
the `3d1b93b` and `40b6a11` branch history named here.

| Workload | Stage | Parent | First storage slice | Change |
| --- | --- | ---: | ---: | ---: |
| padded 1080p | Input preparation | 23.359 ms | 18.953 ms | -4.406 ms (-18.9%) |
| padded 1080p | Complete encode | 174.466 ms | 171.169 ms | -3.297 ms (-1.9%) |
| padded 4K | Input preparation | 97.576 ms | 82.201 ms | -15.375 ms (-15.8%) |
| padded 4K | Complete encode | 699.764 ms | 654.247 ms | -45.517 ms (-6.5%) |

Complete-encode timing was noisier than the preparation-stage result, so its
improvement is directional rather than a precise throughput promise. Every
sample selected Metal and retained its encoded size: 410,072 bytes at 1080p and
1,606,911 bytes at 4K. The resident encoding test also compares the lean and
complete-output frame and codestream byte-for-byte.

Three alternating padded-4K process pairs, each with one warmup and one measured
sample under `/usr/bin/time -l`, reduced median maximum resident set size from
1.56 GiB to 1.22 GiB: approximately 342 MiB, or 21%. The pinned-libjxl Release
matrix passed 66 of 67 tests; the only failure was the pre-existing CPU
quantization golden mismatch, reproduced unchanged on the parent.

### Landed second slice: borrow the coding Opsin image

Commit `55048f2` makes `PreparedQuantizationPipeline::coding_opsin` borrow the
workflow-owned immutable Opsin view for the synchronous prepared lifetime.
Exact and CPU paths
still own a separate preprocessed image because Gaborish inversion changes their
search-domain pixels; resident paths no longer create a duplicate of the
unfiltered coding image.

Each successful pipeline preparation receives a monotonic source generation.
`PreparedAdaptiveQuantization` records that generation and discards its
source-dependent evaluator and resident views when a new preparation is bound,
even if the caller reused identical host addresses. AC-search scratch is retained
because it caches capacity rather than source pixels. A focused regression test
rewrites one image in place, re-prepares through the same views, and verifies the
reused path against a fresh resident frame and codestream.

Five alternating Release process pairs measured the following preparation-stage
change relative to `40b6a11`, with two warmups and five samples per process:

| Workload | Parent | Borrowed Opsin | Change |
| --- | ---: | ---: | ---: |
| padded 1080p | 19.031 ms | 17.196 ms | -1.835 ms (-9.6%) |
| padded 4K | 89.961 ms | 76.447 ms | -13.514 ms (-15.0%) |

All five paired 4K preparation results favored the borrowed view. Complete
encode timing remained GPU-noise-limited: the median paired difference was
-6.6 ms at 1080p and -1.6 ms at 4K, with wide ranges, so this slice makes no
standalone end-to-end speed claim. Every sample selected Metal and retained the
same encoded size as its parent cohort.

Three alternating padded-4K RSS pairs reduced median maximum resident set size
from 1.24 GiB to 1.17 GiB, approximately 70 MiB or 5.5%. The pinned-libjxl
Release matrix again passed 66 of 67 tests; the sole failure was the unchanged,
pre-existing CPU quantization golden mismatch.

### Implemented third slice: direct padded Opsin preparation

`PreparedWorkflow` now borrows the caller's immutable linear-RGB view for the
synchronous encode and all target-size attempts. It no longer owns
`padded_linear`. The internal `LinearRgbToPaddedOpsin()` preparation primitive
transforms source rows directly into the workflow-owned padded Opsin image,
fills each transformed right edge, and duplicates only the final transformed
row into bottom padding.

Input and transformed-output finite checks are fused into the scalar or NEON
conversion loop. The internal destination stays private and is discarded after
failure, so it does not require another full-image staging buffer. The public
`LinearRgbToOpsin()` entry point still converts through private scratch and
commits only after success, preserving its in-place support and failure
atomicity.

Five alternating independent Release process pairs compared this change with
the `55048f2` baseline. Each process used two warmups and five measured samples;
the table reports the median of process medians:

| Workload | Stage | `55048f2` | Direct padded Opsin | Change |
| --- | --- | ---: | ---: | ---: |
| padded 1080p | Input preparation | 17.240 ms | 9.375 ms | -7.865 ms (-45.6%) |
| padded 1080p | Complete encode | 175.513 ms | 157.849 ms | -17.664 ms (-10.1%) |
| padded 4K | Input preparation | 75.088 ms | 54.422 ms | -20.666 ms (-27.5%) |
| padded 4K | Complete encode | 655.957 ms | 615.225 ms | -40.732 ms (-6.2%) |

All five paired preparation and complete-encode results favored the candidate
at both sizes. Every sample selected Metal and preserved encoded size: 410,072
bytes at 1080p and 1,606,911 bytes at 4K. A separate forced exact-coefficient
Metal smoke retained its 434,392-byte output; its one-sample timing is not a
performance claim. A cross-binary CLI smoke over the repository fixture was
byte-identical before and after the slice for CPU, fully resident Metal, and
exact-coefficient Metal.

Three alternating padded-4K RSS pairs reduced median maximum resident set size
from 1.161 GiB to 1.043 GiB, approximately 121.5 MiB or 10.2%. A targeted test
compares strided direct output bit-for-bit with the former pad-then-transform
result, checks row-stride padding and non-finite rejection, and verifies public
atomicity when finite input overflows during conversion. It passes in both the
native NEON build and a forced-scalar build. The Release matrices
passed 61 of 62 tests without libjxl and 66 of 67 with pinned libjxl; in each,
the sole failure was the unchanged pre-existing CPU quantization golden.

### Completed investigation: persistent workspace leases

The direct-write and duplicate-ownership reductions have landed, and the fresh
profile still shows material allocator/lifetime cost. The first experiment on
`perf/last-mile` therefore preserves only the exact-capacity persistent and
staging arenas owned by `MetalPreparedAqEvaluation`; it does not cache an
entire `PreparedWorkflow` unchanged. See
[the last-mile roadmap](last-mile-optimization.md#first-workspace-lease-experiment)
for the retained A/B measurements and qualification boundary.

The C API documents that `gjxl_encode` may run concurrently on one context.
Consequently, one context-wide mutable workspace would either race or serialize
all calls. A safe design is a leaseable pool:

- acquire and return a workspace under a short lock;
- keep all encode and GPU work outside that lock;
- give each batch worker one natural workspace lease;
- key or reconfigure workspaces by backend and required capacity;
- discard poisoned state after a device, upload, or completion error; and
- enforce a high-water memory policy so a single large image is not retained
  forever by every worker.

The retained implementation follows those rules with one idle slot per AQ
arena class, an exact planned-capacity key, failure poisoning, and a `1 GiB`
per-arena ceiling. Returned buffers are purgeable-volatile; an acquire that
observes an emptied resource discards it and allocates cold storage.

Live-backend measurements found the no-pressure cost that peak RSS concealed:
the median idle physical footprint rose from `160` to `425 MiB` at 1080p and
from `169` to `1,226 MiB` at 4K. Controlled pressure reclaimed it: the paused
4K candidate fell to `76 MiB`, versus `81 MiB` for the parent under the same
procedure, and the next encode recovered through the cold path with unchanged
bytes. This qualifies the AQ latency cache, but it also rules out leasing the
larger prepared Butteraugli arena without a broader cache budget and trim API.

Prepared AQ currently uses backend, view identity, and options to decide whether
an evaluation is compatible. Reusing the same host allocation for a different
image can preserve its pointer and geometry while changing its contents. A
persistent workspace therefore requires an explicit image generation/reset
operation. Pointer identity alone must not allow a new image to reuse the old
device pixels or Butteraugli reference.

### Remaining gates

The direct-transform change must preserve borrowed-view lifetime, failure
atomicity, and prepared-GPU invalidation. Required coverage includes automatic
CPU fallback, forced CPU and Metal, target-size repeated attempts, maximum-error,
final-score diagnostics, concurrent C API calls, batch workers, allocation
failures, and device-error invalidation.

## Priority 3: validate and move image pixels once

### State after the direct-write and provenance slices

The workflow previously ran [`EdgeExtend()`](../src/codestream/workflow.cpp)
over the complete padded destination, then called the atomic public
[`LinearRgbToOpsin()`](../src/codec/color_transform.cpp), which allocated and
copied another full padded result. Both workflow-only passes and the owned
padded-linear image are now gone.

The retained internal path:

- validates source values while transforming them;
- writes transformed values directly into workflow-owned Opsin storage;
- combines output-finite detection with the transform loop or its vector
  reduction; and
- fills only the transformed right edge and duplicates the final transformed
  row into bottom padding.

Quantization-matrix statistics still scan Opsin. The ordinary fully resident
workflow no longer repeats finite-value validation inside Metal evaluator
preparation. `PreparedQuantizationPipeline` records the exact immutable RGB and
Opsin views validated by the direct transform, and the backend-neutral resident
frontend checks both identities before using a private validated-preparation
capability. A different source view falls back to the public validating entry
point. Geometry, layout, strategy, EPF, option, and resident-device-view checks
remain unconditional.

The trust contract is not present on `AqEvaluationPreparation` or the public
`PrepareAqEvaluation()` API. Direct public GPU calls, exact-coefficient paths,
and other callers without matching preparation provenance retain their finite
scans. This keeps an optimization established by one synchronous workflow from
becoming a caller-controlled validation bypass.

Finite input alone does not prove finite output. The public linear-float API can
receive very large finite values that overflow during matrix arithmetic, so the
output check must be fused, not silently deleted. The public
`LinearRgbToOpsin()` function should also retain failure atomicity: a failed
standalone call must not leave a partially transformed destination.

The color transform currently creates fresh `std::thread` workers for each
parallel invocation. That overhead is lower priority. Earlier shared-executor
experiments were not unconditionally faster, so full-image allocations and
passes should be removed before scheduling is redesigned.

### Result and validation

The direct-write slice exceeded the original 5-8 ms estimate because it removed
both the persistent padded-linear image and the public transform's padded
scratch/copy from the workflow. At padded 4K, input preparation improved by
20.7 ms in the retained cohort and complete encode by 40.7 ms, with a 121.5 MiB
peak-RSS reduction. The larger end-to-end result includes allocator and memory-
traffic effects outside the removed loop and must not be interpreted as color-
transform CPU time alone.

Temporary timers around the production `ValidateFiniteImage()` calls measured
21 preparations at each resolution across seven independent processes. The
medians were:

| Finite validation | Padded 1080p | Padded 4K |
| --- | ---: | ---: |
| Original linear RGB | 2.29 ms | 9.38 ms |
| Coding Opsin | 2.33 ms | 9.16 ms |
| Combined critical-path scan | 4.61 ms | 18.49 ms |

Seven alternating stage-profile pairs compared the retained implementation
with `3313bbe`. Median `frontend.prepare_evaluator` wall time improved from
29.89 to 26.39 ms at padded 1080p (11.7%) and from 129.94 to 108.88 ms at
padded 4K (16.2%). All seven 1080p pairs and six of seven 4K pairs favored the
provenance path. Complete-encode timing remained noisier than this bounded
stage, so the scan and evaluator numbers are the causal qualification for the
slice rather than an inflated end-to-end claim.

Public-preparation tests inject NaN into original RGB and infinity into coding
Opsin and require rejection before any GPU allocation or submission. A prepared
pipeline test also substitutes a different non-finite RGB view and verifies
that provenance does not follow it. Effort-7-like, high-density, and
maximum-error CLI smoke encodes remained byte-for-byte identical to `3313bbe`
and were accepted by pinned `djxl` 0.12.0.

Tests must cover NaN, infinity, huge finite values, strided inputs, overlapping
views where allowed, unchanged public output on failure, scalar/NEON parity, and
complete codestream hashes. Any numerical approximation change also requires
decoded Butteraugli comparison; direct-write and pass-fusion changes should aim
for exact bytes.

## Priority 4: align and accelerate chromaticity statistics

### The heuristic is grounded

GJXL's quantization-matrix scale statistics are not an ungrounded local search.
Pinned libjxl's
[`PixelStatsForChromacityAdjustment`](../third_party/libjxl/lib/jxl/enc_frame.cc)
computes the same maximum X edge, B-minus-Y edge, and exposed-blue statistic and
uses the same decision thresholds.

The behavioral difference was when and how the pass ran. Libjxl skips the pixel
scan at tiers faster than effort 7 and returns before it in maximum-error mode.
GJXL previously computed the statistics during `PrepareWorkflow()` even when
`SelectQuantizationMatrixScales()` would ignore them for maximum-error.

### Implementation

The first slice now matches libjxl's control policy:

- efforts 1-6: use the distance/default scale path without pixel statistics;
- efforts 7-10 and the explicit effort-9-like high-density override: retain
  pixel statistics;
- maximum-error: skip the pass entirely.

Maximum compression continues to control only entropy/codestream search, so it
does not independently enable the pixel pass at low effort. The gate is an
explicit internal policy queried by `PrepareWorkflow()`; the standalone
statistics helper retains its complete validation contract.

The second slice optimizes the retained effort-7 and effort-9-like path:

- current and previous row pointers are hoisted outside the pixel loop;
- per-sample finite checks are removed only for the exact Opsin view validated
  by the synchronous RGB-to-Opsin transform;
- horizontal and vertical X differences, B-minus-Y differences, and
  exposed-blue products use four-wide NEON reductions; and
- scalar handling remains for vector tails and non-NEON targets.

The standalone internal helper retains its checked scalar behavior. A separate
finite-Opsin entry point makes the trust boundary explicit and leaves results
unchanged on failure. Independent row stripes were not added: the vectorized
pass brought preparation close enough to the no-statistics effort-6 floor that
thread-launch and reduction complexity are not justified by the retained
profile.

Do not initially fuse this pass into Opsin conversion. Vertical-neighbor state
and parallel row scheduling make fusion harder to qualify, while a standalone
SIMD maximum reduction is comparatively contained.

### Effort/mode-gate result

Seven alternating independent-process pairs used a Release build, two warmups,
and five samples per process with fully resident Metal at effort 6. The table
reports the median of each seven-process cohort:

| Input | Baseline preparation | Gated preparation | Removed wall time |
| --- | ---: | ---: | ---: |
| Padded 1080p | 9.28 ms | 3.25 ms | 6.03 ms (64.9%) |
| Padded 4K | 49.60 ms | 15.29 ms | 34.32 ms (69.2%) |

Every paired preparation comparison favored the gate. Complete-encode timing
also improved in every pair, but those results combine the eliminated pass with
the intentional low-effort matrix-scale and codestream change, so they are not
used as the causal timing claim. On the synthetic benchmark, effort-6 output
became 10.7% smaller at both resolutions; final Butteraugli changed from 1.597
to 1.589 at 1080p and from 1.647 to 1.699 at 4K. These are policy-alignment
observations, not a matched-quality density claim.

Effort 7, explicit high density, and maximum-error smoke encodes remained
byte-for-byte identical to `1e8bb0e`. The maximum-error result demonstrates
that the formerly unused scan was removed without changing its fixed 2/2
matrix scales. Candidate codestreams, including the deliberately changed
effort-6 output, were accepted by `djxl` 0.12.0.

### Retained-pass result and gates

Seven alternating independent-process pairs compared the optimized path with
`c81ff95`, again using Release builds, two warmups, and five samples per process.
Both sides used fully resident Metal at effort 7, so the matrix statistics were
retained and codestream behavior was identical:

| Input | Scalar preparation | Trusted NEON preparation | Removed wall time |
| --- | ---: | ---: | ---: |
| Padded 1080p | 9.96 ms | 4.57 ms | 5.39 ms (54.1%) |
| Padded 4K | 41.41 ms | 15.92 ms | 25.49 ms (61.6%) |

Every preparation pair favored the optimized path. Complete-encode cohort
medians were noisy at 1080p (159.23 versus 164.09 ms) and improved at 4K
(590.24 versus 561.01 ms); five of seven 1080p pairs and six of seven 4K pairs
favored the optimized build. The bounded preparation stage is therefore the
causal result, while a fresh matched end-to-end profile should wait for the
independently owned effort-7 update-policy change.

The checked scalar and trusted vector paths produce exactly equal statistics on
degenerate, strided, vector-width, and scalar-tail fixtures. Efforts 6, 7, 9,
and 10, explicit high density, maximum compression, and maximum-error smoke
encodes were byte-for-byte identical to `c81ff95`; every candidate was accepted
by `djxl` 0.12.0. The previously sampled 16.8 ms statistic attribution used a
different workload and sampled-CPU boundary, so it should not be compared
directly with the 25.49 ms 4K preparation-stage reduction.

The thresholds `0.015`, `0.022`, `0.026`, `0.28`, `0.33`, `0.38`, and `0.13`
directly affect the frame header. Existing `nextafter` selection fixtures cover
every boundary, and the retained SIMD path is checked against the scalar oracle.
Any future change to operation contraction or rounding requires renewed
codestream qualification.

## Priority 5: shorten the Metal-to-serializer handoff

### Completed mapped-source handoff

Encoding-only fully-resident execution already omits reconstructed-RGB,
block-distance, and final-quant-field materialization. The remaining final frame
handoff in
[`MetalPreparedAqEvaluation`](../src/gpu/metal/metal_aq_evaluation.cpp) contains
primarily:

- strategy-batch-major quantized AC coefficients;
- quantized DC coefficients;
- raw quantization; and
- small color-correlation maps.

Metal buffers use `MTL::ResourceStorageModeShared`. On Apple Silicon the former
[`CopyDeviceToHost()`](../src/gpu/metal/metal_backend.cpp) calls were therefore
host `memcpy`s from shared storage rather than PCIe transfers. They nevertheless
created an entire strategy-batch-major host representation immediately before
the frame assembler copied the same values into group-major storage.

[`AssembleVarDctEncoderFrame()`](../src/codec/vardct_frame.cpp) then allocates
and zeros group-major AC storage, copies raw quantization and DC, reconstructs
floating DC, scatters each transform's coefficients into its AC group, and calls
the deep `VarDctEncoderFrame::valid()` scan. The evaluator separately scans the
readback for poison values before assembly.

The layout conversion is structural: Metal's final coefficient buffer is
organized by strategy batch, channel, and anchor, while the serializer consumes
fixed-capacity AC-group-major channel storage. That conversion remains
necessary. Priority 5 removes the redundant representation around it rather
than changing either layout.

The completed implementation:

1. records source-independent transform layouts as checked coefficient offsets;
2. waits at the existing completion boundary, then borrows read-only AC, DC,
   and raw-quant ranges directly from completed shared Metal buffers;
3. consumes those borrowed ranges synchronously while building a local owned
   `VarDctEncoderFrame`, so no mapped span escapes and output remains atomic;
4. folds unwritten-value detection into the DC copy and AC repack loops;
5. replaces the constructor's terminal deep `valid()` self-audit with explicit
   construction checks for geometry, strategy metadata, coefficient ranges,
   group capacity and coverage, raw quantization, EPF sharpness, finite DC, and
   complete transform consumption; zero-initialization still guarantees AC
   padding; and
6. allocates exact-coefficient and diagnostic readback vectors lazily. The
   remaining diagnostic raw-quant readback is one checked contiguous copy rather
   than one copy per block row.

The final owned frame is still required by the current serializer contract, so
its group-major AC allocation and repack remain. Giving assembly writable final
spans would only avoid raw-quant and DC temporaries, about 2 MiB at padded 4K,
while complicating atomic construction. The mapped-source design instead
eliminates the much larger strategy-batch-major AC staging allocation, about
95 MiB for the measured padded-4K workload.

Testing-only readback accounting now distinguishes intermediate bytes copied
to host storage from completed shared-buffer bytes synchronously mapped during
assembly. A frame-producing resident path reports zero intermediate frame bytes
and the exact mapped AC+DC+raw footprint. Exact-coefficient paths continue to
report neither class because their coefficients originate in authoritative host
storage.

### Measured result

Five alternating independent-process Release pairs compared this slice with
`e34efca`. Both sides used the synthetic padded-image effort-7 fully resident
workflow. The table reports cohort medians:

| Input | Metric | `e34efca` | Mapped-source handoff | Change |
| --- | --- | ---: | ---: | ---: |
| Padded 1080p | Complete encode | 122.638 ms | 119.273 ms | -3.365 ms (-2.74%) |
| Padded 1080p | Quantization pipeline | 89.749 ms | 86.298 ms | -3.451 ms (-3.85%) |
| Padded 4K | Complete encode | 441.600 ms | 432.552 ms | -9.048 ms (-2.05%) |
| Padded 4K | Quantization pipeline | 336.315 ms | 326.055 ms | -10.260 ms (-3.05%) |

Every one of the ten matched pairs favored the mapped-source build. Three
alternating padded-4K `/usr/bin/time -l` pairs measured median peak RSS falling
from 1,157,054,464 to 1,079,672,832 bytes: 77,381,632 bytes, or 73.8 MiB and
6.69%.

New stage-profile boundaries place pointer/range acquisition at 0.000166 ms
median on padded 4K. That number is intentionally narrow: shared-memory
consumption happens in assembly. The remaining assembly median is 4.446 ms at
1080p and 19.058 ms at 4K. There is no added command-buffer submission or wait.

Frame parity and atomic rejection tests cover coefficient order, poison values,
invalid raw quantization, exact versus resident operation, mapped-byte
accounting, and profiled versus ordinary output. Parent/candidate CLI encodes
of `testdata/codestream_sample.pfm` were byte-for-byte identical at effort 7,
effort 9, explicit high density, maximum compression, maximum error, exact
coefficients, throughput, maximum throughput, and with final-score diagnostics.
The complete candidate/parent SHA-256 values were:

| Mode | SHA-256 |
| --- | --- |
| Effort 7, throughput, or final-score diagnostics | `56d3b52d1bb80d2b7ea260a4b6cd937d6858e9b3619dcdd35368dad7aa800e5d` |
| Effort 9 or explicit high density | `9dd9af4b1d80e3b2376e457c7940fead8a1b4445e31eb521af6adbaff47df3d8` |
| Maximum compression | `31c06d354659a6b79179046850b0182a2dc5344e7a5448cff15d01b0c3f08728` |
| Maximum error | `f0c4779e5742db21d436fe19c71989162de0056b003d9469aaf729ac56a3895a` |
| Exact coefficients | `e4566239f5e15dd67a4716d26da662728c88ffcae19bdf93ff28c2b8df6c8504` |
| Maximum throughput | `7d33f12ed509dac53d60d438297905adfb01f77bbc0039dca0a16498c9941e12` |

All candidate codestreams were accepted by `djxl` 0.12.0.

### Deferred follow-ups

A Metal group-packing kernel remains a possible optimization for the measured
19.1 ms 4K assembly. It is not the next step: it must pack into the serializer's
eventual stable representation, run in the existing final command buffer to
avoid a new synchronization boundary, and beat the current CPU repack in
end-to-end wall time rather than kernel time alone.

A serializer view over mapped Metal storage is also deferred. It would require
retaining a backend buffer lease through serialization, a backend-independent
read-only frame contract, coherence and alignment rules, cancellation-safe
lifetimes, and concurrency and memory-pressure qualification. The current
synchronous borrow deliberately adds none of that lifetime machinery.

Neither design is GPU entropy coding; rANS remains on the CPU. Re-profile after
priority 6 before choosing either one.

## Priority 6: separate ordinary tokenization from maximum compression

### Previous residual tournament architecture

Balanced and high-density encoding now select one block-context map, one
coefficient-order representation, and one entropy family. Their AC path still
uses the reusable intermediate representation built for maximum compression:

1. [`BuildSimpleAcGroupTokenTemplateForEncoder()`](../src/codestream/ac_group.cpp)
   validates and collects anchors, allocates three nonzero maps, appends each
   value, and appends one four-byte descriptor per token.
2. `MaterializeSimpleAcGroupContextsForEncoder()` builds block-context values
   and walks every descriptor again to emit a separate `uint16_t` context
   array.
3. Entropy construction walks the resulting split value/context streams again
   to build fixed-HybridUint histograms.
4. Final emission traverses the selected token streams, as it necessarily must.

This split representation was a major improvement for maximum compression:
many candidate context maps can reuse coefficient values and traversal order.
For a one-representation encode, however, the descriptor allocation and second
context pass are unnecessary.

Pinned libjxl's
[`TokenizeCoefficients()`](../third_party/libjxl/lib/jxl/enc_entropy_coder.cc)
computes the final block context and immediately appends the `(context, value)`
token in one coefficient traversal. Although the function is dispatched through
Highway, its central advantage here is not an extraordinary vectorized
coefficient loop. It performs less generalized work and uses per-group scratch.

### Ordinary one-pass token path

Add a dedicated internal ordinary encoder path that receives the already chosen
coefficient orders and block-context map and, for each AC group:

- traverses anchors and coefficients once;
- writes split `values` and final `contexts` arrays directly;
- reuses one nonzero-map and anchor scratch object per participating worker;
- precomputes natural orders once per encode rather than lazily in every group;
  and
- optionally produces cache-local balanced histogram statistics while token
  values are hot.

Keep the current template/context-materialization path for maximum compression,
where its reuse is justified. The two paths should share the actual coefficient
context formulas and validation helpers so behavior does not drift.

### Effort-aware coefficient-order work

Pinned libjxl's
[`ComputeCoeffOrder()`](../third_party/libjxl/lib/jxl/enc_coeff_order.cc)
deterministically samples approximately half of eligible blocks when deriving a
custom coefficient order at the effort-7-like Squirrel tier if only DCT8 is in
use. GJXL currently scans every coefficient at every ordinary effort.

Resolve coefficient-order effort explicitly rather than inferring it solely
from entropy behavior:

- effort 7: use the same deterministic DCT8-only half-sampling policy;
- effort 8: retain full order statistics, matching libjxl's slower tier;
- efforts 9-10/high density: retain full statistics; and
- maximum compression: retain full statistics.

This is deliberately byte-changing. The objective is effort behavior and
speed/density alignment, not byte identity with GJXL's previous effort-7
codestream. Reproducing libjxl's PRNG and traversal policy does not imply that
the two encoders will select identical orders when their quantized frames or
strategies differ.

Also remove redundant trusted-frame validation inside serializer helpers. The
top-level encoder already validates the complete frame; internal calls for DC
tokenization, block-context calculation, and coefficient orders should use
validated variants rather than repeatedly invoking a deep frame audit.

### Balanced histogram construction

The balanced direct-ANS partition uses one fixed HybridUint configuration. Its
current histogram builder re-encodes every ordered token to accumulate symbol
counts and extra-bit totals. The ordinary tokenization task can accumulate
those exact integer statistics concurrently with final-context construction.

Do not scatter every token directly into a process-wide table of approximately
1 KiB histograms. A previous experiment removed the nominal histogram pass but
increased context-materialization work by an order of magnitude because writes
jumped across thousands of cache lines. The viable design is:

1. accumulate group-local populations only for contexts touched by that group;
2. retain compact touched-context and sparse-symbol bounds;
3. reduce groups in deterministic group order into the global context
   histograms; and
4. pass those populations into `PrepareDirectAnsPartition()` so it skips its
   full token scan.

For high-density effort 9, retain the legitimate search across multiple
HybridUint configurations and precise histogram forms. The one-pass token
representation still applies, but an effort-7 fixed-configuration population
shortcut must not silently remove effort-9 search work.

### Expected result and gates

The raw 4K phase labels report 21.9 ms for GJXL and 2.2 ms for libjxl, but the
boundary mismatch assigns coefficient-order work only to GJXL. After correcting
that mismatch, the earlier profile estimated approximately 8.7 ms of 4K delay
from GJXL's coefficient-order/token-preparation behavior. The separate
entropy-construction gap was approximately 8.9 ms. Those are the intended
targets.

Final rANS/model emission should not be redesigned as part of this work. At 4K
GJXL spends 8.9 ms there versus libjxl's 21.3 ms. rANS is serial within a stream,
and moving it to Metal would add dispatch and synchronization to a phase that is
already ahead.

The one-pass representation and prepared-population changes should preserve
exact ordinary codestream bytes. Effort-7 coefficient-order sampling should be
qualified through encoded size, decoded pixels, Butteraugli, and deterministic
repetition rather than old-byte equality. Effort 9 and maximum compression must
retain their current search intensity and pinned hashes unless explicitly
changed by a separate proposal.

### Implemented result

Priority 6 now gives the two policies distinct data paths:

- balanced and high-density encoding emit final split values and contexts in
  one AC traversal, using precomputed natural orders and one reusable anchor,
  nonzero-map, and population scratch object per participating worker;
- balanced tokenization also emits compact touched-context/sparse-symbol
  populations, reduces them in deterministic group order, and passes the
  resulting fixed-HybridUint populations into direct ANS construction;
- high density keeps its 28 HybridUint configurations and precise histogram
  search instead of consuming the effort-7 population shortcut; and
- maximum compression alone retains reusable token templates and the separate
  context-materialization pass needed by its representation tournament.

The direct and template paths share the nonzero prediction, zero-density, block
context, and final context-layout helpers. Serializer-only DC, block-context,
and coefficient-order entry points now trust the top-level frame audit; their
public counterparts retain independent validation and atomic output behavior.

Coefficient-order effort is explicit in `VarDctCodestreamOptions`. Automatic
effort 7 selects the pinned xorshift128+ DCT8-only half-sampling policy. Effort
8, efforts 9-10, explicit high density, and maximum compression use all order
statistics. Maximum compression forces the full policy even if a low-level
caller supplies the sampling value. The standalone serializer defaults to full
statistics so its existing API remains byte-compatible.

Seven alternating independent-process parent/candidate pairs were measured per
workload from fresh Release builds. Every process performed one internal warmup
before its retained sample, and pair order alternated. The padded fixtures use
multiple order families, so their effort-7 bytes remain identical and isolate
the direct-token/population change:

| Effort 7, Metal fully resident | Parent median | Priority 6 median | Change |
| --- | ---: | ---: | ---: |
| 1080p codestream wall time | 28.702 ms | 21.782 ms | -24.0% median paired ratio |
| 1080p complete workflow | 117.791 ms | 111.670 ms | -5.2% from cohort medians |
| 1080p AC-tokenization wall time | 6.351 ms | 7.795 ms | histogram work moved here |
| 1080p entropy wall time | 15.302 ms | 7.017 ms | -54.1% |
| 4K codestream wall time | 89.623 ms | 56.670 ms | -36.9% median paired ratio |
| 4K complete workflow | 426.376 ms | 397.603 ms | -6.7% from cohort medians |
| 4K AC-tokenization wall time | 23.455 ms | 27.721 ms | histogram work moved here |
| 4K entropy wall time | 43.575 ms | 8.105 ms | -81.4% |

All fourteen effort-7 codestream pairs favored Priority 6. Encoded sizes were
unchanged at 533,163 bytes for 1080p and 2,103,900 bytes for 4K. The apparent
increase in aggregate coefficient-tokenization worker time is intentional:
balanced symbol populations are now produced while group-local values are hot.
It replaces the later serialized token scan; aggregate worker time is not added
to phase wall time.

Five additional alternating padded-4K effort-9 pairs retained the exact
2,106,485-byte codestream. Median codestream wall time was effectively neutral
at 160.963 ms versus 160.442 ms, with a 0.994 median paired ratio. This is the
intended boundary: the direct representation remains, but legitimate
high-density entropy search still dominates.

The explicit sampling gate was qualified on a deterministic 512x512
three-channel random fixture containing 4,096 DCT8 transforms and no other
strategy. Across five alternating CPU pairs, coefficient-order work fell from
1.449 ms to 0.773 ms; the median paired ratio was 0.537. The sampled codestream
was 280,663 bytes versus 280,631 bytes for full statistics, an increase of 32
bytes or 0.0114%. Repeated sampled encodes were byte-identical. `djxl` 0.12.0
accepted both streams, their decoded PFM files were byte-identical, and their
independently measured Butteraugli scores were identical (`3.0025761127`, with
the same `3-norm` of `1.553354`).

The small public-workflow fixture also retained exact parent bytes at efforts
7, 8, and 9 and maximum compression. Their SHA-256 hashes are respectively
`56d3b52d1bb80d2b7ea260a4b6cd937d6858e9b3619dcdd35368dad7aa800e5d`,
`1dfca10baf3f4e33beaaedb5a85feebce463c288e2256a9a0fc7628eb5828f35`,
`9dd9af4b1d80e3b2376e457c7940fead8a1b4445e31eb521af6adbaff47df3d8`,
and `31c06d354659a6b79179046850b0182a2dc5344e7a5448cff15d01b0c3f08728`.
All four were accepted by `djxl` 0.12.0. The diagnostic raw-sample schema is now
version 15 and names the direct versus template AC-tokenization path explicitly.

## Recommended implementation sequence

The priority numbers describe causal importance, but the implementation order
separates low-risk work elimination from lifetime and layout redesign:

1. **Completed in `3d1b93b`:** make fully resident Metal the ordinary default
   while retaining exact coefficients as an explicit compatibility path.
2. **Completed in `40b6a11`:** give encoding and public diagnostics two concrete
   output contracts; remove dead resident host materialization and defer CPU-only
   preparation until CPU is selected.
3. **Completed in `55048f2`:** remove duplicate workflow-to-prepared Opsin
   ownership, with explicit borrowed-view lifetime and prepared-state
   invalidation.
4. **Completed in `3313bbe`:** remove padded linear RGB and the
   workflow's atomic color-transform scratch/copy through a checked direct-write
   path; preserve the public transform contract.
5. **Completed in `1e8bb0e`:** bind finite-input provenance to the
   exact workflow RGB/Opsin views and skip the measured duplicate scans only in
   private fully resident preparation; retain public GPU validation.
6. **Completed in `c81ff95`:** apply priority 4's effort/mode gate;
   preserve high-density behavior and keep maximum compression orthogonal.
7. **Completed in the current slice:** vectorize priority 4's retained
   effort-7/high-density statistics under explicit finite-Opsin provenance.
8. Incorporate the independently owned effort-7 zero-update change and capture
   a fresh matched profile.
9. **Completed in the current priority-5 slice:** assemble synchronously from
   completed shared Metal buffers, remove intermediate frame readback storage,
   fuse validation into repacking, and retain atomic owned-frame construction.
10. **Completed in the current priority-6 slice:** add one-pass ordinary
    tokenization, per-worker scratch, precomputed natural orders, trusted
    serializer entry points, and effort-aware coefficient-order sampling.
11. **Completed in the current priority-6 slice:** reduce cache-local balanced
    populations and reuse them during direct ANS construction without changing
    high-density or maximum-compression search.
12. **Completed for priority 6:** re-profile with alternating Release processes;
    defer more counters, persistent workspace
    pooling, a Metal group-packing kernel, or a mapped zero-copy frame view.

This sequence prevents two common forms of wasted optimization: building a
persistent pool around allocations that should not exist, and redesigning the
GPU coefficient layout before the ordinary serializer's required representation
has stabilized.

## Qualification contract

Every retained optimization should use a fresh Release build, alternating
independent processes, warmups, multiple samples, and representative 1080p, 4K,
and higher-resolution inputs. Report complete-encode wall time separately from
stage wall time, aggregate worker time, and GPU timestamps.

The functional matrix must cover:

- efforts 7, 8, 9, and 10;
- explicit maximum compression;
- forced CPU, forced Metal, and automatic fallback;
- fully-resident, exact-coefficients, and supported throughput paths;
- single-target, target-size repeated attempts, and maximum-error control;
- final Butteraugli diagnostics on and off;
- explicit CPU thread budgets and automatic scheduling;
- batch encoding and concurrent calls on one C API context; and
- injected allocation, upload, completion, readback, and numeric failures.

For byte-preserving changes, retain selected block-map/order metadata, entropy
modes, cluster counts, model/token bit accounting, complete SHA-256 hashes,
pinned-`djxl` acceptance, and decoded-pixel comparison. For intentionally
byte-changing effort alignment, retain deterministic output and compare encoded
size and decoded perceptual quality against both the parent GJXL build and the
matched libjxl effort.

## Explicit non-goals

- Do not restore the maximum-compression representation tournament at an
  ordinary effort.
- Do not call aggregate worker time wall-clock delay.
- Do not infer that a Metal kernel is slow from the complete pipeline without a
  matched operation boundary.
- Do not cache one mutable prepared workflow in `GJXLContext` and serialize
  concurrent calls.
- Do not remove public finite-input or failure-atomicity guarantees to speed up
  a trusted internal caller.
- Do not replace effort-9 HybridUint and precise-histogram search with the
  effort-7 fixed-population shortcut.
- Do not prioritize GPU entropy coding while CPU final emission is already
  faster than libjxl and repeated host passes remain.
