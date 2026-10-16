# Proving unique histogram rows before deduplication

This experiment follows `bc75912` and the
[deferred materialization measurements](modular-deferred-sample-materialization.md).
The diagnostic capture for `photo-large-1029` retained every selected row while
still hashing candidates, probing the native table and rehashing retained rows.
The new private option proves uniqueness from the active quantized properties
before gathering. It applies to raster CPU, selected CPU and Metal rows.

## Proof and fallback

The proof examines the actual property list and quantizer maps after native cut
generation. Both x and y properties must be present. Their bins must strictly
increase over every coordinate in each included nonempty channel. A byte column
cannot distinguish more than 256 coordinates; a smaller bin count can still
qualify when its actual map is injective over the channel's dimensions.

Across included channels in an image, the actual clamped static channel bins
must strictly increase. Across nonempty images, static group bins must strictly
increase. A single channel or image does not need its corresponding static
property. Empty channels/group holes and the first-oversized-channel inclusion
rule are handled exactly as in gathering. Metadata is included despite the
ordinary size limit. The proof reads no pixels and allocates no storage.

Distinct `(group, channel, y, x)` bin tuples imply distinct full rows regardless
of residual tokens. Native selection visits each pixel at most once, so every
selected row must be retained in order with weight one. The proof is sufficient,
not necessary: colliding static bins, missing coordinates, wide channels and
other unproved scopes retain the complete ordered native deduplication path.
Sampling, cut construction, thresholds, multiplier ranges and fitting remain
unchanged.

## Implementation

Proven CPU scopes use a separately compiled gathering loop without hashing,
lookup or table growth. They still compute and validate every required property
and predictor residual. Proven Metal scopes fill unit weights and copy the
completed GPU columns directly into the learner's owned columns. They do not
extract candidates on CPU or create retained-row indices. Unproved Metal scopes
use the previous deferred replay, including native saturation and overflow
rejection.

Hash-table allocation is deferred until after the proof. Proven scopes avoid
that allocation; admission keeps the conservative worst-case table bound.
The proof's timer is separate from gathering and included in total sample
preparation. New coarse counters record attempted/proven scopes and covered
rows. No per-pixel proof instrumentation is added.

The default remains unchanged. The private profile modes are `raster-unique`,
`selected-unique`, and `metal-unique`; the last uses deferred native replay when
the proof fails. Integration into `feat/modular` remains separate.

## Correctness

All six focused CTest cases pass. The CPU preparer checks proof acceptance and
fallback for group holes, partial/missing static prefixes, reordered or missing
coordinate properties, all predictors, 255/256/257 coordinate boundaries,
width-one quantization, multiplier-cut collisions, clamped group/channel IDs,
and metadata/excluded channels. Both CPU routes match exact rows, weights,
thresholds and learned trees. Storage bounds and allocation-failure publication
are tested with the proof enabled and disabled.

The expanded Metal test passes 80 fixture/configuration combinations, including
proven scopes and fallbacks with immediate/deferred replay. It checks exact
columns/tokens/weights/cuts/trees, diagnostic identities, managed backing bounds,
and allocation/submission/completion failure publication. A saturated native
table alias is rejected in all three routes with the proof on and off.

The independent pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161`
preparation oracle passes 121 cases in both proof modes: 242 native preparations
with exact sample columns, weights, thresholds and learned trees. The two new
fixtures explicitly require proof acceptance for distinct coordinate/static
bins and fallback for coarse coordinate bins. The native weight-overflow
reproduction remains rejected with the proof enabled.

- [Focused test log](../build/sample-rows-unique/tests.log).
- [Independent preparation oracle log](../build/sample-rows-unique/reference-tests.log).
- [Native preparer](../src/codec/modular/histogram_sample_preparation.cpp).
- [CPU proof cases](../tests/modular_histogram_preparation_test.cpp)
  and [Metal comparisons](../tests/metal_histogram_sample_rows_test.cpp).

## Diagnostic coverage

Separate `rows-on` captures compare committed deferred replay with the proof:

| Input | Proven scopes | Rows covered by proof / selected | Lookup probes before / after | Rehashed rows before / after |
|---|---:|---:|---:|---:|
| photo-large-1029 | 192 / 192 | 23,460,354 / 23,460,354 | 46,920,708 / 0 | 22,873,570 / 0 |
| alpha16-local | 9 / 12 | 1,028,216 / 1,030,284 | 2,060,567 / 4,135 | 772,764 / 1,584 |

The proof intervals sum to 0.061 ms for 1029 and 0.004 ms for the RGBA16 fixture.
Three small RGBA16 scopes remain unproved and retain native lookup; the output
still retains 1,030,283 rows, including one merged duplicate. This illustrates
why the guard must inspect the actual mappings instead of assuming that every
local scope is unique. These single diagnostic captures are not used as paired
speed evidence.

## Qualification and measurement

The comparison uses the frozen `bc75912` profile executable and a same-build
control with the proof disabled. Both keep automatic weighted-feature and
tokenization routing, CPU tree fitting, Best predictors, estimated ANS,
automatic buffering and 14 CPU participants, with feature production's existing
four-participant limit. The timed inputs are `photo-large-1029`,
`photo-large-1039`, and `photo-large-1049`, each 4000 x 3000 RGB8. One warmup and
three measured repetitions per input/mode/build are interleaved on an Apple M4
Pro with 48 GiB of memory, on AC power, on 2026-10-16.

Complete-call wall excludes backend creation, input/output I/O and independent
decoding. Preparation/gather/replay times are overlapping per-scope wall sums,
not serial elapsed phases. The runner requires AC power, waits for other codec
qualifications/builds, records environment/process snapshots, and excludes and
retries observed overlaps. Ordinary desktop activity remains visible in the
snapshots; these are matched session results, not peak-throughput measurements.

```sh
python3 tools/modular_reference_learning/run_sample_materialization_qualification.py \
  --baseline FROZEN_BC75912 --baseline-revision bc75912 \
  --binary build/release/gjxl_modular_hybrid_profile \
  --baseline-metal-mode metal-deferred --candidate-unique-rows \
  --out NEW_CAPTURE --input INPUT --modes raster metal --warmups 1 --repeats 3

# Pass the candidate executable as --baseline too for the same-build control.
# Append rows-on to a direct profile invocation for lookup/rehash diagnostics.
```

All 132 accepted paired encodes match complete encoded bytes and independently
decode exactly. This includes 48 same-build calls, 48 previous-commit comparison
calls, 30 Best-predictor coverage calls, and six all-predictor calls. Two
same-build warmup attempts overlapped a separate weighted-selection build/test;
their timings are excluded and both were repeated after waiting. The four
separate diagnostic calls also pass byte/decode checks.

Coverage inputs are `photo-macan`, `alpha-ria`, `palette-358`, `alpha16-ramp`, and
the 769 x 513 RGBA16 `alpha16-local`. All three row routes are checked. The first
four use the existing global CPU row route, including when Metal is requested;
the local RGBA16 fixture exercises 12 GPU scopes. Both Best and all 14 predictors
prove nine of those scopes and retain native fallback for three. The palette
and small 16-bit ramp fixtures exercise complete fallback. Single coverage
calls are correctness evidence, not speed estimates.

## Measured results

The table compares the frozen `bc75912` executable with the candidate. CPU means
raster rows with the existing automatic weighted-feature route; Metal means
deferred replay before and guarded uniqueness after. Values are medians in
seconds. Preparation values are sums across overlapping scopes.

| Input | Row route | Encode wall before / after | Wall reduction | Preparation before / after | Preparation reduction |
|---|---|---:|---:|---:|---:|
| 1029 | CPU | 4.217 / 4.096 | 2.9% | 3.502 / 1.821 | 48.0% |
| 1039 | CPU | 3.636 / 3.522 | 3.1% | 2.899 / 1.590 | 45.1% |
| 1049 | CPU | 3.014 / 2.887 | 4.2% | 2.860 / 1.562 | 45.4% |
| 1029 | Metal | 4.144 / 3.910 | 5.7% | 2.772 / 0.608 | 78.1% |
| 1039 | Metal | 3.575 / 3.358 | 6.1% | 2.367 / 0.610 | 74.2% |
| 1049 | Metal | 2.962 / 2.762 | 6.7% | 2.263 / 0.637 | 71.8% |

The same-build control uses the identical candidate executable on both sides,
changing only whether the proof is enabled. Aggregating by the sum of the three
per-input medians gives:

| Comparison | Row route | Encode wall reduction | Preparation reduction |
|---|---|---:|---:|
| Previous commit to candidate | CPU | 3.3% | 46.3% |
| Same binary, proof off to on | CPU | 4.3% | 47.2% |
| Previous commit to candidate | Metal | 6.1% | 74.9% |
| Same binary, proof off to on | Metal | 5.6% | 75.3% |

This supports the shortcut in both row producers without attributing a compiler
layout change to the algorithm. Three repetitions and three photographs support
a matched-session result, not a corpus-wide or peak-throughput guarantee.

The proof covers different fractions of the three images' learning scopes:

| Input | Proven / checked scopes | Proven / selected rows |
|---|---:|---:|
| 1029 | 192 / 192 | 23,460,354 / 23,460,354 |
| 1039 | 177 / 192 | 19,841,050 / 20,432,794 |
| 1049 | 164 / 192 | 18,283,891 / 19,470,891 |

These counts match between CPU and Metal. Remaining scopes run native
deduplication, whether or not they happen to contain duplicate rows. The proof
takes about 0.06 ms summed across all scopes per image.

Metal's replay interval falls from 1.75–2.22 s to 0.08–0.12 s in scope sums.
The column-materialization subinterval itself increases from roughly 0.02 s to
0.06–0.08 s; the improvement comes from removing the surrounding candidate
extraction, lookup and rehash work. The isolated copy interval is not faster.
Overall managed peak backing varies by less than 1% in the previous-commit
comparison, and conservative admission limits are unchanged. This experiment
does not establish a meaningful whole-encode memory reduction.

## Decision and remaining work

Keep this as a qualified opt-in implementation on `feat/modular-sample-rows`.
The next integration step is to receive the sample-row changes into the current
public workflow, qualify its chosen CPU/Metal routing, then profile the combined
pipeline. Public defaults have not changed in this experiment.

On these photographs, remaining Metal preparation sums are about 0.61–0.64 s:
GPU generation accounts for 0.36–0.38 s, native selection for 0.075–0.088 s, and
replay/materialization for 0.08–0.12 s. Generation includes host setup, transfers,
submission and waiting. Native preflight, cut generation and allocation also
remain within preparation. These are overlapping scope measurements, so they
do not predict an equivalent reduction in encode wall time. CPU tree fitting
remains separate and unchanged.

This evidence makes general GPU deduplication a lower priority for these local
scopes. Further preparation work should first measure the residual selection,
dispatch/wait and copy costs in the integrated pipeline, including representative
cases where the proof fails. Preserve the native fallback while deciding
whether caching selection, batching row dispatches or changing column ownership
is worth its complexity.

- [Portable results, raw values, hashes and environment](modular-unique-sample-rows-results.json).
- Local raw manifests: [same build](../build/sample-rows-unique/within-build/manifest.json),
  [previous commit](../build/sample-rows-unique/timed/manifest.json),
  [coverage](../build/sample-rows-unique/coverage/manifest.json),
  [all predictors](../build/sample-rows-unique/all-predictors/manifest.json),
  and [diagnostics](../build/sample-rows-unique/diagnostics.json).

Build-directory captures are local to the isolated `feat/modular-sample-rows`
worktree; the portable results are tracked with this report.
