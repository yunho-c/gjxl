# Deferred materialization of Metal learning rows

This experiment follows the CPU optimization at `d523147` and the
[previous measurements](modular-sample-materialization.md). It targets the
remaining CPU replay of completed Metal sample rows. The private
`metal-deferred` mode retains native ordered deduplication and builds the
learner's columns after the retained rows have been identified.

## Implementation and scope

Immediate replay reads each compact GPU row, hashes it, compares occupied native
lookup slots, and appends its fields to the learner's individual columns when
retaining the row. Deferred replay instead appends the source-row index. Hashing
and equality comparisons for retained rows read the immutable completed GPU
columns through those indices. Table growth, retained table prefixes, lookup
order, row weights, saturation, and overflow rejection are unchanged.

After deduplication, the preparer resizes and fills each learner column once. If
all selected rows were retained, the source and destination row order are
identical and each column uses a contiguous copy. Otherwise it compacts the
retained indices in order. Resizing value-initializes the destination before
copying; this cost is included in the materialization timer.

The retained-index list reuses the selection-position vector after synchronous
GPU completion. No additional index allocation is required. The GPU request
contract explicitly ends its position/slice/mapping borrows before return.
Completed row buffers remain immutable, and the learner receives its usual
owned columns before fitting can reorder them. Worst-case preparation storage
includes the existing selection vector, GPU rows, and learner columns.

The default remains immediate replay. `metal-deferred` is an additional private
profile mode. CPU row preparation and shared/global fallback remain available;
the fallback does not exercise deferred GPU replay. Sampling, quantization,
weighted features, row kernels, and fitting algorithms are unchanged.

## Attribution

The frozen `d523147` executable has a matching frozen dSYM and preparer source.
Samply captures all workers at 1 kHz. The analyzer resolves inline functions and
reports CPU deltas assigned to both the current and preceding sampled stacks.
This sensitivity matters near phase transitions; sampled CPU is not caller wall
time or GPU duration. Inclusive function totals overlap.

The verified 1029 baseline capture attributes 22.6–24.7% of sampled preparation
CPU to table growth/rehashing, 10.0–10.3% to candidate hashing, and 8.6–12.0% to
row comparison, depending on delta assignment. Per-row column appends were also
visible. The deferred capture still attributes 28.6–31.7% to table growth and
11.6–15.5% to comparisons. These percentages have different denominators and
overlap with their callees; they are not independent speedup predictions.

The optional `rows-on` diagnostics add lookup/occupied-probe counts, duplicate
hits, saturated-row counts, table-growth counts, retained-row rehash counts, and
a coarse table-growth timer. Timing comparisons use `rows-off`, which selects
template instantiations without the hot-loop diagnostic increments. Deferred
replay always records its single materialization interval per scope.

The separate 1029 counter captures agree exactly in both replay modes:

| Diagnostic | Count |
|---|---:|
| GPU scopes | 192 |
| Selected and retained rows | 23,460,354 |
| Lookup probes / occupied probes | 46,920,708 / 23,663,289 |
| Table growths / retained-row rehashes | 565 / 22,873,570 |
| Duplicate hits / saturated rows | 0 / 0 |

All 192 deferred scopes use contiguous column copies. Their materialization
intervals sum to 0.020 seconds, including destination initialization. The
instrumented table-growth intervals sum to 1.003 seconds for immediate replay
and 0.870 seconds for deferred replay. These are single diagnostic captures,
excluded from the paired performance results below.

The first timing pilot is excluded in full: another codec executable named
`profile` was observed while the original process guard recognized only
`gjxl_` names. The guard now recognizes GJXL executable paths, profiling tools,
and `qualify_` runners, including their idle gaps. Raw pilot results and an
explicit exclusion record are retained; passing decode/byte checks do not make
those timings uncontaminated.

An early diagnostic wrapper also mistook its own newly spawned profiler target
for competing work. Its attempts are retained separately and excluded; the
verified captures use the corrected process exclusion. The first full baseline
timing capture stopped at a failed post-encode AC-power check. That partial
capture is excluded in full, and the replacement starts from fresh warmups.

## Paired measurement

The two comparisons use the same three large photographs, Best predictors,
estimated ANS, automatic buffering and weighted-feature/tokenization routing,
CPU tree fitting, and a 14-participant CPU ceiling and phase limits; feature
production keeps its existing four-participant limit. There is one warmup and three measured
repetitions per input, build label, and row mode. The machine is an Apple M4 Pro
with 48 GiB memory, on AC power. Ordinary Spotlight/media-analysis and desktop
activity is recorded in the raw snapshots. This is not a peak-throughput claim.

The first comparison uses the **same candidate executable** for both labels,
switching only between immediate `metal` and `metal-deferred`. Although its raw
manifest carries the base revision `d523147`, both binary hashes are identical
and both include this experiment's source overlay. It isolates the replay mode
from compilation changes. The second comparison uses the frozen `d523147`
executable and includes raster CPU rows as a control.

Complete-call wall time excludes backend creation, input/output I/O, and
independent decoding. Preparation and replay times are sums of overlapping
per-scope wall intervals across workers. They cannot be added to, or directly
subtracted from, complete-call elapsed time.

Same-build medians, seconds:

| Input | Complete call immediate / deferred | Reduction | Preparation immediate / deferred | Reduction |
|---|---:|---:|---:|---:|
| 1029 | 4.224 / 4.165 | 1.4% | 3.715 / 2.810 | 24.4% |
| 1039 | 3.698 / 3.577 | 3.3% | 3.228 / 2.372 | 26.5% |
| 1049 | 3.015 / 2.964 | 1.7% | 2.976 / 2.327 | 21.8% |

CPU replay intervals fall from 3.146 to 2.250 seconds, 2.690 to 1.893 seconds,
and 2.463 to 1.814 seconds respectively: 26.4–29.6% lower. The whole-encode
change is smaller, and individual timing ranges overlap.

Frozen `d523147` immediate Metal replay versus deferred replay:

| Input | Complete call before / after | Reduction | Preparation before / after | Reduction |
|---|---:|---:|---:|---:|
| 1029 | 4.244 / 4.085 | 3.7% | 3.605 / 2.758 | 23.5% |
| 1039 | 3.643 / 3.569 | 2.0% | 3.074 / 2.340 | 23.9% |
| 1049 | 3.027 / 2.953 | 2.4% | 3.003 / 2.261 | 24.7% |

Raster CPU control medians are 4.158 / 4.157, 3.630 / 3.625, and
3.008 / 3.043 seconds before / after. Summing the three input medians, raster
preparation changes by less than 0.1% and complete-call wall increases by 0.3%.
The corresponding Metal totals fall by **24.0% for preparation, 29.2% for replay,
and 2.8% for the complete call**. The same-build comparison gives 24.3%, 28.2%,
and 2.1% respectively. These are sums of per-input medians, not measured batch
throughput. All raw repetitions and ranges are retained in the portable results.

Within the candidate, deferred Metal rows give complete-call medians 1.6–3.0%
below raster CPU rows in the second capture. This is a modest gain on this
three-photo corpus. The GPU row kernels, sampling policy, and fitted output
did not change. This experiment combines deferred writes with lookup/rehash
reads from the GPU column layout; it does not separately measure those effects.

## Correctness

The expanded Metal row test compares raster CPU, selected CPU, and each GPU
replay mode across 32 fixture/configuration combinations. It checks exact
columns, tokens, weights, thresholds, trees, diagnostic population identities,
allocation-free admission, actual backing within the bound, and failure
publication. A separate GPU-compatible two-channel case checks rejection of a
saturated native table alias in raster, immediate, and deferred replay.

All six focused CTest cases passed. The independent pinned libjxl preparation
oracle passed all 119 cases with exact samples and trees, including rejection
of reference weight overflow. This protects the CPU control path affected by
the shared helper templates; GPU replay is additionally compared with that CPU
path in the expanded Metal test.

The completed paired captures accept **108 encodes**: 24 in the same-build
comparison, 48 against the committed baseline, 30 coverage calls, and six
all-predictor calls. Every output is byte-identical for its input/predictor
setting across builds and requested row modes, and every encode independently
decodes exactly. Selected/distinct populations and tree-node counts also agree.
The final captures have no overlapping attempts. The excluded pilot, wrapper
attempts, and power-check interruption are separate records.

Coverage includes `photo-macan`, `alpha-ria`, `palette-358`, `alpha16-ramp`, and
the synthetic 769x513 RGBA16 `alpha16-local`, in raster, selected CPU, and Metal
modes. The first four use shared/global CPU fallback. The synthetic input uses
12 local GPU scopes and 1,030,284 selected rows; it retains 1,030,283 rows, so
11 scopes copy contiguously and one exercises compaction. Both Best and all-14
predictor runs match exactly. These single-pass coverage timings are not speed
evidence. Duplicate-heavy, saturation, and custom-property cases are also
covered by the preparer tests.

## Decision and next experiment

Keep deferred materialization as a private opt-in mode. It removes the per-row
column appends without an additional index allocation, but still performs
ordered native lookup, weight updates, and rehashing. This is not a parallel
GPU deduplication implementation, and the measured gain does not establish a
public default or a peak end-to-end GPU advantage.

The next useful experiment is a guarded proof that rows are unique. With the
default 256-bin quantizer, x/y bins preserve coordinates within a 256x256
channel. If the actual coordinate mappings are injective and quantized static
channel/group values distinguish every included slice, selected rows cannot
match. A preparer can then retain them directly with weight one. The proof must
inspect the active mappings, properties, shapes and static bins and fall back
to ordered native deduplication whenever it cannot establish uniqueness.
The current experiment does not implement this shortcut.

That common-case optimization could remove both lookup and rehash work on CPU
and GPU row routes. For scopes that cannot use it, retained hash caching is a
separate candidate; its buffer cost and memory traffic must be measured. Native
sampling and cut generation remain on CPU. GPU batching and integration into
the advancing public branch should follow measured qualification separately.

## Reproduction and evidence

```sh
# Final optional argument enables diagnostic counts; omit it for timing.
build/release/gjxl_modular_hybrid_profile INPUT OUT auto auto best \
  fit-off 14 14 14 14 14 metal-deferred rows-on

# Frozen d523147 versus the candidate's deferred mode.
python3 tools/modular_reference_learning/run_sample_materialization_qualification.py \
  --baseline FROZEN_D523147 --baseline-revision d523147 \
  --binary build/release/gjxl_modular_hybrid_profile \
  --candidate-metal-mode metal-deferred --out NEW_CAPTURE \
  --input INPUT --modes raster metal --warmups 1 --repeats 3

# Use the same candidate executable for --baseline and --binary to isolate
# immediate versus deferred replay within one build; request --modes metal.
```

- [Sampled CPU analyzer](../tools/modular_reference_learning/analyze_sample_preparation_samply.py).
- [Native preparer](../src/codec/modular/histogram_sample_preparation.cpp).
- [Exact row test](../tests/metal_histogram_sample_rows_test.cpp).
- [Initial sampled profile](../build/sample-rows-deferred/baseline-metal.json.gz)
  and [inline attribution](../build/sample-rows-deferred/baseline-attribution.json).
- [Excluded pilot timing record](../build/sample-rows-deferred/pilot-timing-exclusion.json).
- [Focused test log](../build/sample-rows-deferred/tests.log).
- [Pinned preparation oracle log](../build/sample-rows-deferred/reference-tests.log).
- [Same-build paired manifest](../build/sample-rows-deferred/within-build/manifest.json).
- [Committed-baseline paired manifest](../build/sample-rows-deferred/timed-ac-retry/manifest.json).
- [Coverage manifest](../build/sample-rows-deferred/coverage/manifest.json)
  and [all-predictor manifest](../build/sample-rows-deferred/all-predictors/manifest.json).
- [Verified baseline attribution](../build/sample-rows-deferred/baseline-verified-attribution.json)
  and [deferred attribution](../build/sample-rows-deferred/deferred-verified-attribution.json).
- [Diagnostic counters](../build/sample-rows-deferred/diagnostics.json).
- [Portable results](modular-deferred-sample-materialization-results.json),
  including binary/source/dependency/input/output hashes, medians and ranges,
  population checks, environment, diagnostic attribution and exclusions.

Build-directory artifacts are local. This experiment remains on the isolated
`feat/modular-sample-rows` branch.
