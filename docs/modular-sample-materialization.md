# Native histogram sample materialization

This follows the [Metal row prototype](modular-sample-rows-prototype.md), committed
as `677ccdd`. That prototype left most of its gathering time in CPU replay into
the native learner. The change here reduces that work for all three row routes:
raster CPU, selected CPU, and CPU replay of Metal-generated compact rows.

## Implementation

Previously, every selected candidate was appended to the learner's separate
predictor/property columns. The preparer then read those columns twice to hash
the new row, compared all fields against occupied lookup slots, and removed the
appended row if it matched a previous row.

The preparer now holds one small candidate on the stack. It computes the two
native hashes from that candidate, sharing their property traversal while
retaining their different residual order and arithmetic. Comparison stops at
the first unequal field. Only candidates that need a new retained row are
appended to the learner's columns. This adds no heap allocation or stored hash
array and preserves the existing worst-case admission bound.

This is an ordered native deduplication optimization. Sampling RNG, quantization
cuts, bin mapping, predictor tokens, table growth and retained prefixes, first
and second slot preference, saturation at 65,535, overflow rejection, and output
row order remain unchanged. Hash reuse from `5e3e4d7` was already in the baseline;
this work does not claim that earlier optimization again.

Metal still produces the same compact rows. GPU dispatch count, buffer layout,
feature ownership, and weighted-feature producer are unchanged. Public backend
selection is unchanged; the native CPU materialization improvement applies
without opting into Metal rows. The learner still owns its final columns, so
each retained Metal row must still be copied into them.

## Correctness

The pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161` preparation oracle
passed all **119 differential cases**, comparing exact sample columns, tokens,
weights, thresholds, and learned trees. The saturated-alias weight overflow case
was rejected as required. This provides an independent oracle for the changed
CPU code, rather than only comparing routes that share it.

All six focused CTest cases passed: histogram preparation, histogram learning,
reference hybrid, reference hybrid palette, Metal sample rows, and Metal
reference workflow. The Metal row test covers 16 fixture/configuration
combinations, including all 14 predictors, custom property ordering, partial
static prefixes, signed wide values, metadata/excluded channels, group holes,
saturated weights, bounded allocation, and failure publication.

The final paired qualification accepted **84 encodes**: 48 large-photo calls
(12 warmups and 36 measured calls), 30 Best-predictor coverage calls, and six
all-predictor calls. Every accepted output matched bytes between builds and row
modes and decoded exactly. Selected/distinct counts and tree-node counts also
matched. Two additional attempts overlapped another tree-fitting build/test,
were retained and excluded, then succeeded when repeated after waiting.

Coverage includes `photo-macan`, `alpha-ria`, `palette-358`, `alpha16-ramp`, and
the retained synthetic 769x513 RGBA16 `alpha16-local` fixture. All three row modes
run on both binaries. The four small original inputs use shared/global trees,
where a Metal-row request falls back to selected CPU preparation. The synthetic
RGBA16 input exercises 12 local GPU scopes and 1,030,284 selected rows, including
an additional all-14-predictor comparison. Single-pass coverage timings are not
used as speed evidence.

## Measurement method

The baseline is the frozen `677ccdd` prototype profile executable. The candidate
uses the same Release `-O3 -g -DNDEBUG` build; its only production change is
`histogram_sample_preparation.cpp`. Every call uses native Best predictors,
estimated ANS, 14 participants, `fit-off`, and automatic buffering. Both row
modes use the existing automatic Metal weighted-feature route and GPU
tokenization. Tree fitting remains the same CPU implementation.

The paired runner freezes executables, hashes linked dependencies and inputs,
archives source and configuration, rotates row mode and before/after order,
records power/thermal/process snapshots, and independently decodes every
encoded result. It waits for other codec builds/qualifications and retains but
excludes any overlapping attempt. Each output must be byte-identical across
both builds and all requested row modes for its input/predictor setting.

The timed corpus is the same three large photographs, `photo-large-1029`,
`photo-large-1039`, and `photo-large-1049`. There is one warmup and three measured
repetitions per input, row mode, and build. Complete-call wall and process CPU
times exclude backend creation, input/output I/O, and independent decode.
Preparation and replay counters sum per-scope wall intervals that overlap
across workers; they are not additional elapsed time or direct CPU-time
measurements.

This is a single Apple M4 Pro, 14 CPU cores, 48 GiB, AC-powered desktop session.
Spotlight and media analysis background work are recorded in the snapshots.
The results characterize these matched runs, not peak machine throughput.

## Results

Median complete-call wall time, seconds; the reduction is relative to the
matching pre-change row mode:

| Input | Row mode | Before | After | Reduction |
|---|---|---:|---:|---:|
| 1029 | Raster CPU | 4.372 | 4.286 | 2.0% |
| 1039 | Raster CPU | 3.784 | 3.646 | 3.6% |
| 1049 | Raster CPU | 3.176 | 3.105 | 2.2% |
| 1029 | Metal | 4.372 | 4.272 | 2.3% |
| 1039 | Metal | 3.805 | 3.703 | 2.7% |
| 1049 | Metal | 3.197 | 3.119 | 2.4% |

Median preparation scope time, seconds (overlapping sums across workers):

| Input | Raster CPU before / after | Reduction | Metal before / after | Reduction |
|---|---:|---:|---:|---:|
| 1029 | 4.633 / 3.589 | 22.5% | 5.057 / 3.751 | 25.8% |
| 1039 | 3.783 / 2.921 | 22.8% | 4.200 / 3.190 | 24.0% |
| 1049 | 3.736 / 2.924 | 21.7% | 4.156 / 3.123 | 24.9% |

Metal's CPU replay scope time fell from 4.485 to 3.166 seconds for 1029,
3.661 to 2.666 for 1039, and 3.641 to 2.574 for 1049: **27.2–29.4% lower**.
GPU generation scope time remained approximately 0.36–0.41 seconds per image.
GPU calls, selected rows, compact output bytes, and dispatch bytes are identical
between the two builds for each mode.

Summing the three per-input medians, raster CPU preparation is **22.4% lower**,
complete-call wall time **2.6% lower**, and process CPU time **3.1% lower**.
For Metal rows, preparation is **25.0% lower**, CPU replay **28.7% lower**,
complete-call wall time **2.5% lower**, and process CPU time **3.7% lower**.
These are sums of per-input medians, not a measured batch throughput result.
Raw repetitions and per-input ranges are retained in the machine-readable
results.

The large photos retain every selected row, so avoiding duplicate append/pop
cannot explain their improvement. The result is consistent with cheaper hashing
from the temporary row and early mismatch rejection. The combined change does
not separately measure those two effects. Duplicate-heavy correctness is covered
by the independent oracle and constant/palette fixtures.

## Decision and remaining work

Keep the native CPU materialization improvement. It reduces preparation work
without a new buffer, changed sampling policy, or changed learned output.

Keep Metal row generation opt-in. With the improvement applied to both paths,
the sum of Metal complete-call medians is still **0.5% higher** than raster CPU
rows. This does not establish a Metal row speed advantage.

CPU replay still accounts for approximately **84–86%** of Metal gathering scope
time. Remaining work includes reading compact GPU rows, hashing candidates,
ordered lookup/weight updates, copying retained rows into learner columns, and
rehashing retained rows when the native table grows. Sampling and quantization
also stay on CPU. A further experiment should isolate retained-row copying and
table rehash cost before deciding between storage/ownership changes, hash
caching, or moving more work onto the GPU. Their added storage, traffic, and
ordered native behavior must be included in that comparison.

## Reproduction and evidence

```sh
# Freeze the baseline executable before building the candidate.
python3 tools/modular_reference_learning/run_sample_materialization_qualification.py \
  --baseline BASELINE_PROFILE --baseline-revision 677ccdd \
  --binary build/release/gjxl_modular_hybrid_profile --out NEW_CAPTURE \
  --input PHOTO_1029 --input PHOTO_1039 --input PHOTO_1049 \
  --modes raster metal --warmups 1 --repeats 3
```

- [Paired runner](../tools/modular_reference_learning/run_sample_materialization_qualification.py).
- [Native preparer](../src/codec/modular/histogram_sample_preparation.cpp).
- [Pinned preparation oracle log](../build/sample-rows-cpu-materialization/reference-tests.log).
- [Focused test log](../build/sample-rows-cpu-materialization/tests.log).
- [Timed raw manifest](../build/sample-rows-cpu-materialization/timed/manifest.json).
- [Coverage manifest](../build/sample-rows-cpu-materialization/coverage/manifest.json)
  and [all-predictor manifest](../build/sample-rows-cpu-materialization/all-predictors/manifest.json).
- [Portable results](modular-sample-materialization-results.json), including
  per-input medians/ranges, input and output hashes, build provenance, and
  accepted/excluded counts.

The independent preparation oracle uses the existing
`tools/modular_reference_learning` harness, generated from the pinned reference
with `prepare.py`. Its CMake configuration and build logs are retained under
`build/sample-rows-cpu-materialization/reference-*`. Because the production
archive was built with Metal, this standalone research harness additionally
links `libgjxl_metal.a`, `libgjxl_gpu_butteraugli.a`, and the Metal, Foundation,
and CoreGraphics frameworks through `CMAKE_CXX_STANDARD_LIBRARIES`.

Build-directory evidence is local. Integration into `feat/modular` is separate
from this isolated branch.
