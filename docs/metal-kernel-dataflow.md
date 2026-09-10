# Metal kernel dataflow sprint

This sprint is based on resident execution at
`1f16769bef87b42094fcd6a3f6dc50722662e20b`. It targets unnecessary intermediate
traffic and communication while retaining the existing encoding policy.
Results below apply only to the recorded machine, revisions and workloads.

## Scope and acceptance

Investigate, implement experiments, qualify, and record a disposition for:

1. AC residual inverse DCT followed by weighted per-channel loss reduction.
2. Per-thread Y/X/B coefficient ownership, intermediate output suppression,
   and final encoding-only materialization.
3. Resident forward/inverse DCT directly reading/writing image rectangles.
4. Pass-specialized EPF, cooperative neighborhood loading, and multiple pixels
   per thread.
5. Prepared kernel/shape/image/device dispatch selection based on measurements.

Keep AQ iterations, AC candidates, tie rules, quantization, rate control,
validation and atomic failure behavior. Compare numerical intermediates and
parent/candidate codestream bytes; independently decode representative files.
Keep the independent CPU oracles and their existing tolerances. Do not accept
a numerical regression because aggregate quality looks similar.

Measure each experiment and the final combination. Use Release builds, warmups,
alternating independent process pairs, natural and padded synthetic inputs at
small, 1080p and 4K sizes. Separate GPU stage/dispatch measurements, complete
public-call wall time, and memory accounting. Retain raw samples, source and
binary identities, commands, failures, and rejected experiments. A local
kernel improvement alone does not establish a complete encode speedup.

## Starting evidence

The local device is Apple M4 Pro (20 GPU cores, 48 GiB), macOS 15.6. A frozen
archive of the parent and separate Release build are under
`build/dataflow/baseline-src` and `build/dataflow/baseline`. Optional libjxl
reference builds are disabled; metal-cpp is pinned at the repository revision.
The existing resident qualification protocol requires identical shader payloads,
so it cannot certify these kernel changes without a separately reviewed protocol.

Initial dispatch-boundary timestamp profiling is unavailable on this device
(`GPU dispatch-boundary timestamp sampling is unavailable`). Use supported
stage timing and Metal System Trace or isolated submissions for kernel evidence;
do not report unsupported dispatch counters as measured.

## Design sources

- [FlashAttention](https://arxiv.org/abs/2205.14135): eliminate materialized
  intermediates by consuming them while local.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): partition work to reduce
  inter-thread communication, as well as memory traffic.
- [Apple shader performance guidance](https://developer.apple.com/videos/play/tech-talks/111373/):
  registers, threadgroup storage and device data share parts of the Apple GPU
  memory hierarchy; staging is a hypothesis to measure, not an automatic win.
- CUDA's tracked `docs/cuda-optimization-s1.md` on `origin/feat/cuda` supplies
  implementation and rejected-experiment references. Its timings are historical
  CUDA results, not Metal forecasts. The local CUDA checkout predates S37-S39.

## Experiment ledger

These are separate incremental experiments. Percentage changes must not be
added. The combined qualification below is a separate parent-versus-bundle experiment.

| Experiment | Incremental correctness evidence | Incremental measurement |
| --- | --- | --- |
| Coefficient ownership | 336 original guarded bitwise cases; later 448 expanded cases, plus existing AQ tests | All seven large isolated shapes improve about 4-9%; five paired workflow processes are inconclusive (+1.17% 1080p, +0.44% 4K). |
| AC inverse loss reduction | Exact cost bits against the retained wide path; existing search/policy tests; three canonical image codestreams | Smaller AC families improve 9-17% in stage measurements. Five workflow pairs: -0.58% 1080p and -1.75% 4K; all five 4K pairs improve. |
| Scored/final coefficient outputs | 448 guarded bitwise cases per full/scored/final variant, including unused-output poison and invalid inputs; AQ integration; three canonical codestreams | Per-shape final stages improve 3-20%, scored stages 2-16% in instrumented 4K profiles. No separate whole-workflow win is claimed for these modes. |
| Direct DCT image I/O | 56 paired forward/inverse cases including strides, offsets, shuffled anchors and padding; AQ integration; three canonical codestreams | Transform/gather/scatter subset improves 68.9-69.2% in three instrumented 4K pairs. Five workflow pairs: -2.14% 1080p, -3.19% 4K; every pair improves. |
| EPF specialization/tiling | 432 guarded bitwise cases across three passes and six variants | Triplet cooperative loads improve large isolated pass 0/1/2 by 41.6%/34.4%/22.1%. Integrated 4K pass 1/2 stages improve 33.5%/15.5%; five workflow pairs show -1.28% 4K, mixed 1080p attribution. |

The workflow comparisons use three warmups and seven measured calls per
independent process, alternating side order over five pairs. GPU probes use
four warmups and seven samples per side in five alternating cohorts, with three
kernel dispatches per GPU-timed command buffer. These isolated probes share a
process and are not independent-process workflow results. Stage comparisons use
two warmups and three samples in three independent alternating pairs.

Raw records live under `build/dataflow/evidence`. `coefficient-wall`,
`ac-loss-wall`, `dct-image-wall`, and their `*-stage` directories contain runtime
identities, commands, raw samples and recomputable paired summaries. Each frozen
intervention has its own executable/metallib under `build/dataflow/`.

Coefficient ownership eliminates 24 bytes of source-level device accesses per
three-channel coefficient index per pass. Direct DCT I/O eliminates the packed
gather/scatter store/read traversals. These are access counts, not measured DRAM
traffic. The Metal gathered-pixel arena is subsequently borrowed by Butteraugli;
direct I/O therefore does not eliminate that allocation as the CUDA experiment
did. Existing lifetimes and memory admission remain in force.

The first direct-DCT probe omitted the two device-basis bindings required by the
old wide inverse kernels. Its failure was a probe defect. After binding bases
with the production host formula, all direct/packed outputs match bitwise. The
failed log and corrected runs are retained. No numerical tolerance was changed.

## Numerical and lifetime contracts

AC loss fusion borrows the completed magnitude-reduction threadgroup arena for
inverse pixels. The inverse helper's basis-staging barrier separates the last
rate read from reuse; every lane rejoins before loss reduction. The weighted
power-eight operations, binary reduction tree and cross-channel finalizer order
match the retained wide control. There is no SIMD-sum reassociation or change to
CPU candidate merging and ties.

Coefficient fusion assigns Y/X/B at a coefficient index to the same thread.
Rounded reconstructed Y stays local. Separate dequantization and color-restore
operations retain their rounding boundaries under the existing safe Metal math
mode and disabled contraction. DC extraction/quantization and the LLF overwrite
still have the necessary publication barriers. Scored mode suppresses integer
AC stores only when no consumer needs them; final mode omits reconstructed AC
and LLF work while still calling X/B dequantization for its finite-value/error
checks. DC and integer outputs retain their existing contracts.

Direct DCT changes image addressing around the original matrix operations,
bases, scales and coefficient layouts. The validated anchor plan owns disjoint
image rectangles, and all owned image planes use the prepared stride. Retained
forward coefficients still survive between AQ iterations. Packed scalar and
factored routes retain gather/scatter and their original bindings.

EPF retains the SAD accumulation, FMA and weight order. Halos are 3/2/1 pixels
for passes 0/1/2. Cooperative loads apply repeated reflection at partial edge
tiles; no lane exits before the publishing barrier. Output bounds are checked
only afterward. Existing input/output ping-pong buffers remain distinct.

## Prepared dispatch and rejected choices

The EPF selection is resolved in preparation, with no per-pixel host decisions.
On devices reporting Apple GPU family 9, passes 0 and 1 use a 32x8 output tile
(32x4 threads, two sequential output pixels per thread) at area >= 256x256.
Pass 2 uses that tile at area >= 1024x1024; below it, the pass-specialized direct
kernel retains an 8x8 launch. Tiling also requires both dimensions >= 64.
Other families retain the original runtime-pass kernel. Unsupported optional
threadgroup geometry falls back without disabling the backend. Measurements
cover this M4 Pro only; family membership is a dispatch guard, not multi-device
performance qualification.

The sweep compared direct, 16x8 with 1/2/4 pixels per thread, and 32x4 with 2/4.
It covered 256x256, 512x512, 1280x720, 1920x1080 and 3840x2160. A first cooperative
load implementation independently computed coordinates for each channel.
Loading the three channels together and avoiding reflection arithmetic on
interior tiles improved that implementation substantially. Four pixels per
thread offered only a small large-image advantage for some passes, and lost
more on small images; the production table uses two. Shared staging is not
selected indiscriminately: direct pass 2 was better through the measured 720p
cohort. The area threshold interpolates between measured cohorts; it is not a
claim that 1024x1024 is a precisely measured crossover.

The direct EPF and all experimental tile entry points remain available to the
isolated probe. Production creates only the six selected optional pipelines.
The transforms retain their existing per-shape geometry and generic packed
implementations; the new image entry points apply to the seven resident shapes
when the SIMD-matrix implementation is selected.

## Reproduction and evidence boundaries

Build the parent revision above and this candidate separately in Release, with
`GJXL_BUILD_TESTS=ON`, `GJXL_BUILD_BENCHMARKS=ON`, and the same Apple Clang/SDK and
pinned metal-cpp. The optional libjxl reference build remains disabled. Record
source changes as well as revision IDs when measuring an uncommitted candidate.
The benchmark must receive each build's own `--metallib`; copying an executable
alone can otherwise retain a baked-in build path.

`tools/metal_dataflow/compare.py` runs independent alternating process pairs,
checks raw sample inventories and duration validity, and retains commands,
runtime/input/protocol hashes, logs and every process median. Grouped stage
costs are summed *within each sample* before computing medians. Phase fields
ending in `work` are aggregate worker counters, not additional wall latency.
Only `total` denotes the documented complete public-workflow timing boundary.
Backend creation is outside that warmed boundary; the resource driver records
setup separately. Do not compare instrumented stage times with uninstrumented
workflow wall time as though they were an additive decomposition.

`qualify.py` discovers both complete CTest inventories, accepts only a freshly
reproduced inherited CPU quantization golden mismatch, checks all guarded
probes, and runs 22 fixtures per revision through the pinned decoder. Its
explicit `--resume` revalidates identical runtime artifacts and each completed
command/log before reuse and records the resumed protocol hash. `parity.py`
covers the canonical corpus/policies and representative decoded pairs.
`resources.py` compiles the existing resident qualification driver separately
against revision-matched headers/static libraries. It exercises changed images,
mixed shapes, two callers, bounded CPU/memory admission, shutdown and trimming,
retaining and decoding 24 output pairs. It records footprint diagnostics but
does not turn those short correctness runs into throughput estimates.

## Combined warm public-workflow result

Seven alternating independent process pairs per workload, each with three
warmups and seven measured public calls, compare the untouched parent against
the complete bundle. Both builds use distance 1.2, effort 7, fully resident AQ,
SIMD transforms and the tuned AC path. Each row's percentage is the median
*paired* change; it is not the ratio of the two separately pooled medians.

| Workload | Parent median ms | Bundle median ms | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 9.222 | 9.074 | -0.44% | 4/7 |
| Padded 1919x1079 | 71.225 | 67.714 | -4.61% | 6/7 |
| Padded 3839x2159 | 240.374 | 226.134 | -5.76% | 7/7 |
| Kodak 17 (512x768) | 22.315 | 21.855 | -2.19% | 7/7 |
| Planter photograph (4K) | 256.187 | 243.907 | -5.11% | 7/7 |

Both 4K cohorts improve consistently, by about 5-6% in warm elapsed time.
The 1080p cohort has one regression; Kodak improves in all seven pairs.
The small case is inconclusive: its paired changes range from -18.3% to +25.0%.
These are measured improvements on this M4 Pro, not the much larger
hypothetical speedups from an Amdahl illustration or CUDA's historical results.

## Qualification

- Both fresh Release inventories contain 122 tests. Each passes 121 and
  reproduces only the inherited `quantization_pipeline` mismatch:
  actual `0.24919039011001587`, expected `0.24914586544036865`.
- All 56 canonical corpus/policy codestream pairs match byte-for-byte, including
  efforts 1-10, density/compression options, final scoring, alternate AQ routes,
  target size and maximum-error policy. Three representative pairs also have
  identical decoded PFMs through pinned libjxl `e8ff0976`.
- Both revisions pass all 22 pinned codestream conformance fixtures.
- All 1,344 coefficient cases, 56 direct/packed DCT pairs and 432 EPF cases pass
  with guard/padding checks. The same complete probe matrix also passes with
  Metal API and shader validation enabled, with error reporting to stderr and
  abort-on-fault. [Apple validation controls](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage/).
- Eleven scoped CPU and thirteen scoped Metal ASan/UBSan tests pass. C++ is
  instrumented; Objective-C++ and leak detection are not. The unsuppressed Metal
  probe reproduces the existing metal-cpp `NSObject.hpp:112` null-member-call
  diagnostic. Only the repository's existing narrow Metal suppression is used
  for the subsequent Metal tests. This is not unsuppressed-Metal cleanliness.
- Eight retained/resource scenarios pass, including mixed sizes, two callers,
  tight/full memory admission, constrained CPU participation, shutdown and trim.
  All 24 changed-image retained pairs match both codestream and decoded bytes.
- Three controller tests reject missing/duplicate samples, invalid durations,
  non-Metal/incomplete evidence and unavailable stage timing; they also verify
  summing stage durations within a sample before taking a median.

The first full candidate suite caught stale profiling expectations: old AC
kernel suffixes and empty scatter stage records. The host now omits eliminated
scatter stages, and the CLI tests explicitly check the new loss kernels while
retaining ordered intervals and nonempty dispatch requirements. The corrected
full suite was rerun. Earlier logs remain in the local evidence tree.

## Stage attribution and setup/resource tradeoffs

Separate instrumented cohorts use three alternating process pairs, two warmups
and three samples. At padded 4K the stage groups change as follows; aggregation
sums the actual stages within each sample before taking medians.

| GPU stage group | Parent ms | Bundle ms | Median paired change |
| --- | ---: | ---: | ---: |
| AC search | 41.170 | 37.953 | -7.82% |
| Scored and final coefficients | 9.116 | 7.581 | -16.84% |
| DCT plus gather/scatter | 10.465 | 3.166 | -69.78% |
| EPF | 6.080 | 4.440 | -27.02% |
| All recorded GPU stages | 155.638 | 142.622 | -8.82% |

Direct image DCT provides the largest absolute saving among these groups.
These instrumented intervals explain where GPU work changed; they are not an
exact decomposition of the separately measured public-call wall time. The
recorded 4K dispatches confirm the 32x4/two-pixel EPF variants; per-dispatch
counter timestamps remain unavailable on this device.

A separate seven-pair process-cold cohort records backend setup and the first
public call independently. At padded 4K, setup medians are 47.484 -> 48.305 ms
(median paired +1.57%), while first-call medians are 295.991 -> 279.105 ms
(median paired -5.25%, all seven improve). This excludes executable launch,
input creation and driver bookkeeping, and does not reset Metal disk/driver
caches. It is not a claim about a cold boot or total application startup.

The metallib grows from 1,489,228 to 1,999,580 bytes, including the retained
experimental entry points. The ordinary image arena plan is unchanged. In the
same short 4K process cohort, median peak physical footprint is 2778.720 ->
2780.173 MiB; post-trim is 2035.251 -> 2034.048 MiB with inputs and backend still
alive. These are process footprints, not managed-admission limits or a memory
saving. The corresponding small 17x9 peak is 120.657 -> 121.329 MiB.



An additional all-bypass-sigma EPF probe checks the shortcut-heavy extreme.
The selected large-image tile is 25.8-26.4% faster across the three passes there;
at 512x512, the tiled changes range from -7.3% to +2.2% around roughly 0.02 ms.
This does not establish a complete-encode win for flat images, but it did not
reveal a large-image penalty hidden by the active-filter synthetic input.

## Running the qualification tools

Use frozen builds and the canonical corpus described in
[`tools/resident_qualification/corpus.md`](../tools/resident_qualification/corpus.md).
For example, with `BASE`, `CAND`, `CORPUS`, `DJXL` and `JXLINFO` pointing to those
local paths, and `OUT` naming a new evidence directory:

```sh
python3 tools/metal_dataflow/qualify.py --baseline-build "$BASE" \
  --candidate-build "$CAND" --output "$OUT/qualified" \
  --decoder "$DJXL" --info "$JXLINFO"
python3 tools/metal_dataflow/parity.py --baseline-build "$BASE" \
  --candidate-build "$CAND" --corpus "$CORPUS" \
  --output "$OUT/final-parity" --decoder "$DJXL"
python3 tools/metal_dataflow/resources.py --baseline-build "$BASE" \
  --candidate-build "$CAND" --baseline-source "$BASE_SOURCE" \
  --candidate-source . --corpus "$CORPUS" --decoder "$DJXL" \
  --output "$OUT/final-resources"
python3 tools/metal_dataflow/compare.py --baseline-build "$BASE" \
  --candidate-build "$CAND" --output "$OUT/combined-wall" \
  --workload synthetic_128x96 --workload padded_1080p --workload padded_4k \
  --input "$CORPUS/kodak-kodim17.pfm" \
  --input "$CORPUS/imazen26-1029-planter-4k.pfm" \
  --pairs 7 --warmups 3 --samples 7
python3 tools/metal_dataflow/compare.py --baseline-build "$BASE" \
  --candidate-build "$CAND" --output "$OUT/combined-stage" \
  --workload padded_4k --input "$CORPUS/kodak-kodim17.pfm" \
  --profile --pairs 3 --warmups 2 --samples 3
python3 tools/metal_dataflow/setup.py --drivers "$OUT/final-resources" \
  --output "$OUT/setup-comparison" --pairs 7
python3 tools/metal_dataflow/test_compare.py
```

Native probes are built with `GJXL_BUILD_BENCHMARKS=ON`. Each takes the frozen
parent and candidate metallib paths as positional arguments. The coefficient
probe supports `--mode full|scored|final` and `--threads`; the EPF probe supports
`--extent WxH` and `--pattern`. Both support `--timing`, which measures three
repeated dispatches per command buffer, four warmups and seven samples in five
alternating cohorts. These are isolated GPU submissions in one process.

The compact tracked [result ledger](metal-kernel-dataflow-results.json) includes
all final paired percentages and artifact digests. Full commands, binaries'
identities, raw samples, failed attempts, sanitizer/validation logs and decoded
outputs remain locally under `build/dataflow/evidence`. Its `manifest.json`
anchors every retained file by SHA-256. `report.py` validates the completed
sprint gates and exports the ledger. The recorded ASan build/run commands and
Metal validation environment are included in that evidence; optional libjxl
reference and thread sanitizer qualification are outside this sprint.

The final disposition is to keep all four dataflow changes and the measured EPF
selection table. Further per-transform launch sweeps, devices beyond this M4
Pro, and broader storage-plan changes remain separate experiments. This sprint
establishes an exact-output 5-6% warm 4K improvement, with substantial local GPU
savings and no claimed arena reduction.
