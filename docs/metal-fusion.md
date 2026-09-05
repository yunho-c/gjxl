# Metal Butteraugli fusion investigation

Date: 2026-09-04

- Planning branch: `perf/metal-fusion`
- Baseline commit: `4eda200f8cf62db6f2544f47f504bc4c3b4131cf`
- Primary target: ordinary effort 7, fully resident Metal, multiscale
  Butteraugli, final diagnostic score disabled
- Initial qualification device: Apple M4 Pro
- Status: completed through Phase 5

## Executive outcome

The completed study supports a small set of selective Butteraugli
**superkernels**, not one monolithic kernel and not indiscriminate launch
fusion. The retained path combines 5-tap blur with Opsin conversion, computes
33-tap low/medium filtering through a direct-load tile, and writes the final
metric directly into resident block and score sinks. Launch-only Malta and
channel fusions were neutral or slower and were removed.

The fully resident AQ path already records reconstruction, filtering,
Butteraugli, block reduction, and policy updates into one ordered compute
encoder and waits once. Its remaining Butteraugli cost comes from a deep serial
graph of full-image kernels. The current multiscale implementation records:

- 44 Butteraugli dispatches to prepare the invariant reference;
- about 72 dispatches for each distorted-image comparison; and
- 144 comparison dispatches at ordinary effort 7, whose current policy performs
  two scored evaluations when no final diagnostic evaluation is requested.

The investigated fusion work was therefore inside the command buffer: launch
fewer thread grids, keep short-lived values in registers or threadgroup memory,
and avoid writing and rereading full-image intermediates. The investigation
order was:

1. fuse all six Malta stages for one scale into one tiled dispatch;
2. fuse independent channel work in psycho-image construction and subsampling;
3. test tile-local separable convolution only where saved global traffic exceeds
   halo duplication and resource pressure;
4. add a resident-only final-distance and reduction path that need not
   materialize a public diagnostic map; and
5. tune each resulting kernel's threadgroup shape from counter evidence.

This began as an investigation plan, not a forecast. Its working target was to reduce a
comparison from about 72 dispatches to 25--35 and its approximate launched
thread count from about 350 million to 150--200 million at padded 4K. Those
numbers are design goals to falsify, not measured results. A 1.5--2x speedup of
Butteraugli evaluation is plausible only if the fused kernels retain healthy
occupancy and do not replace device-memory traffic with excessive halo work,
register pressure, or threadgroup barriers.

## What the counter result does and does not mean

The earlier phrase "launch-dominated" needs a precise interpretation. Apple's
Compute Shader Launch limiter includes both useful work that launches threads
and stalls caused by launch backpressure. A high value can mean that enough
threads are being launched, or that resource pressure prevents more threads
from launching. It is not a direct measurement of CPU API calls, command-buffer
submission latency, or a fixed cost per dispatch. Apple explicitly recommends
checking threadgroup-memory use, occupancy-manager targets, cache behavior, and
other resource pressure after observing a high launch limiter.

This distinction is important here:

- the resident AQ loop already uses one command buffer, one compute encoder,
  and one final wait in
  [`metal_aq_evaluation.cpp`](../src/gpu/metal/metal_aq_evaluation.cpp);
- each comparison still launches hundreds of millions of shader threads over
  dozens of dependent passes; and
- the generic plane helper currently chooses an `8 x 8` threadgroup for nearly
  every Butteraugli plane kernel in
  [`metal_primitives.cpp`](../src/gpu/metal/metal_primitives.cpp).

The counters therefore justify investigating fusion, but they do not prove
that merely turning six dispatch API calls into one will recover a large fixed
overhead. A successful superkernel must reduce at least one of:

- launched thread volume;
- global intermediate reads and writes;
- redundant address, bounds, and per-thread setup work;
- launch backpressure caused by the old kernels' resource profiles; or
- the number of unavoidable serial full-frame passes.

The interpretation follows Apple's descriptions of the
[Shader Launch limiter](https://developer.apple.com/videos/play/tech-talks/111374/?time=332),
[GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy),
and
[threadgroup/grid sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes).
Apple's
[visual-timeline guidance](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-a-visual-timeline)
also recommends avoiding unnecessarily small passes because work exists
between passes, but the GJXL capture must determine how much that consideration
matters relative to actual Butteraugli arithmetic.

## Evidence and its limits

### Valid System Trace capture

The retained clean Performance Limiters capture attributed the following GPU
activity on Apple M4 Pro:

| Stage | GPU activity | Mean occupancy | Mean bandwidth | Main signals |
| --- | ---: | ---: | ---: | --- |
| Butteraugli reference preparation | `22.017 ms` | `80.5%` | `160.1 GB/s` | shader launch, instruction throughput |
| Resident AQ command buffer | `120.225 ms` | `66.3%` | `148.4 GB/s` | shader launch, instruction throughput |

The resident number includes reconstruction, filters, Butteraugli, reductions,
policy work, and final-frame work. It is not a Butteraugli-only timing. About
`17.4 ms` of counter intervals overlapping other processes' GPU activity was
conservatively excluded from that capture. The trace predates this planning
branch, so it supports the architectural diagnosis but must be repeated on the
current revision before promotion.

### Intrusive stage profile

A retained padded-4K stage profile at `3839 x 2159` recorded this sample:

| Submission or group | GPU timestamp time | Dispatches |
| --- | ---: | ---: |
| Reference preparation | `29.665 ms` | `44` |
| Resident AQ command buffer | `179.813 ms` | `316` |
| Butteraugli groups inside resident AQ | `127.788 ms` | `144` |
| Resident reconstruction | `26.277 ms` | `113` |
| Final frame | `6.379 ms` | `34` |
| Resident block reduction | `1.704 ms` | `14` |
| Resident quantizer work | about `2.17 ms` | `60` |

The stage instrumentation inflates execution and its absolute times must not be
substituted for complete-encode or System Trace timings. Its dispatch list and
relative composition are still useful. Across two resident comparisons it
attributed:

| Butteraugli group | GPU timestamp time | Dispatches |
| --- | ---: | ---: |
| Main-scale psycho construction | `49.590 ms` | `38` |
| Main-scale Malta | `24.792 ms` | `12` |
| Subscale psycho construction, including subsampling | `14.611 ms` | `44` |
| Main-scale mask and final metric | `11.262 ms` | `10` |
| Main-scale L2 | `10.726 ms` | `2` |
| Subscale Malta | `6.929 ms` | `12` |
| Subscale mask and final metric | `4.869 ms` | `18` |
| Subscale L2 | `2.782 ms` | `2` |
| Scalar score reduction | `2.228 ms` | `6` |

These values make psycho construction and Malta the best first fusion targets.
They do not predict the speedup of a fused implementation.

The local evidence retained when this plan was written is:

- `/private/tmp/gjxl-metal-trace.SFPRQe/gjxl-effort7-4k-performance-limiters.trace`;
- `/private/tmp/gjxl-metal-trace.SFPRQe/gjxl-metal-performance-limiters.tracetemplate`;
  and
- `/private/tmp/gjxl-metal-trace.SFPRQe/handoff-stage-profile.json`.

These temporary paths are provenance, not durable repository fixtures. Phase 0
must produce a fresh current-revision capture rather than depending on them.

### Amdahl bound from the earlier public-workflow baseline

The earlier complete padded-4K effort-7 baseline was `390.358 ms`, with about
`82.032 ms` of timestamped resident Butteraugli stages and `22.054 ms` of
reference preparation. If a hypothetical implementation halved exactly that
`104.086 ms` and changed nothing else, the arithmetic upper estimate would be a
`52.043 ms`, or `13.3%`, complete-workflow saving:

```text
T_new = T_total - T_target + T_target / speedup
```

That is an illustration of scale, not a current benchmark. Subsequent changes,
queue overlap, instrumentation, and any change to the effort-7 evaluation
policy all alter the real bound. Every slice below must be judged at the current
complete public-workflow boundary as well as inside the GPU stage.

## Current Butteraugli topology

The implementation in
[`metal_butteraugli.cpp`](../src/gpu/metal/metal_butteraugli.cpp) uses 33
reusable working planes, five blur kernels with lengths `5`, `33`, `15`, `7`,
and `13`, a `32 x 8` Malta output tile with radius 4, and a 256-thread scalar
maximum reduction. The shaders in
[`butteraugli.metal`](../src/gpu/metal/kernels/butteraugli.metal) are compiled
with safe math, precise FP32 functions, and contraction disabled by
[`CMakeLists.txt`](../CMakeLists.txt). The shader also uses explicit
`unfused_multiply_add` expressions where evaluation order matters.

### Reference preparation

For a normal multiscale image without expansion:

```text
main psycho image                              19 dispatches
main reference-mask precompute and blur         3
three-channel 2x subsample                       3
subscale psycho image                           19
                                                --
reference preparation                           44
```

The subscale reference psycho planes are cached. Its blurred mask is not; the
current comparison reconstructs that mask at subscale.

### One distorted-image comparison

```text
main psycho image                              19
main difference: Malta 6 + L2 1 + mask/final 5 12
three-channel 2x subsample                       3
subscale psycho image                           19
subscale difference                              15
main/sub composition                              1
maximum reduction                                ~3
                                                 --
one comparison                                  ~72
```

The 19 psycho-image dispatches per scale are:

```text
three 5-tap horizontal channel blurs              3
fused vertical blur and Opsin conversion           1
three 33-tap transposed channel convolutions       3
fused low/medium-frequency output                  1
two high-frequency channels, two passes each       4
medium-B separable blur                            2
X suppression                                      1
two ultra-frequency channels, two passes each      4
                                                  --
                                                  19
```

The six Malta stages already fuse scale calculation and neighborhood response
within each stage. Each stage nevertheless traverses its scale independently
and reads and updates one full accumulation plane. The strict accumulation
order is `4, 5, 2, 3, 0, 1`; stages 4 and 5 initialize the two accumulation
channels before the remaining four additions.

### Approximate launched-thread volume

Let `N = 8,288,401` main-scale pixels and `M = 2,073,600` subscale pixels for
the retained padded-4K image. Ignoring edge rounding and counting reduction
threads approximately:

```text
reference preparation     = 22N + 22M  = 227,964,022 threads
one comparison            ~ 33N + 37M  = 350,240,433 threads
reference + two compares                 928,444,888 threads
```

This is why launch machinery can be a strong limiter even though the work is
recorded in one command buffer: the GPU is launching close to a billion
Butteraugli threads through a serial pass graph. Many of those threads do only
one channel or one small transform stage, then commit an intermediate that a
later grid reads back.

## Design rules

1. **Fuse semantic neighborhoods, not the entire metric.** Use several
   independently measurable superkernels. A single megakernel would accumulate
   too much live state, threadgroup memory, control flow, and compile-time
   complexity.
2. **Preserve strict arithmetic order.** Fusion may change rounding even with
   strict compiler flags. Malta accumulation, convolution sum order, nonlinear
   transforms, and final powers must retain their existing expression order.
3. **Optimize the ordinary resident path first.** Public distance-map and
   intermediate-stage diagnostics have different materialization requirements.
   They must remain correct, but they should not force the encoder to write
   planes it does not consume.
4. **Keep A/B machinery temporary and internal.** Use two Release build
   directories selected by a private compile definition during an experiment.
   Do not add a public execution-plan API. After qualification, retain one
   normal implementation and remove the experiment switch.
5. **Do not infer a win from dispatch count.** Record elapsed GPU time,
   complete encode time, occupancy, launch limiter, instruction limiter,
   threadgroup-memory use, register pressure indicators, bandwidth, and output
   correctness for every slice.
6. **Promote slices independently.** A failed tiled 33-tap convolution must not
   block a successful six-way Malta or three-channel subsample fusion.

## Proposed superkernels

### Slice A: six-way Malta fusion

This is the recommended first experiment. It is bounded, the existing tile
implementation is already available, and Malta represented `31.72 ms` across
main and sub scales in the intrusive two-evaluation sample.

Add `gjxl_butteraugli_malta_sixway_f32`. One `32 x 8` threadgroup owns one
output tile. It processes the six psycho-plane pairs serially in the existing
order and retains the two output-pixel accumulators in FP32 registers:

```text
ac[0] = 0
ac[1] = 0
for stage in [4, 5, 2, 3, 0, 1]:
    cooperatively load and scale this stage's haloed tile
    threadgroup barrier
    response = existing LF or full Malta neighborhood expression
    channel = (stage is even) ? 1 : 0
    if stage >= 4: ac[channel] = response
    else:          ac[channel] = ac[channel] + response
    threadgroup barrier before reusing tile storage
store ac[0] and ac[1] once
```

The kernel should use one `(32 + 8) x (8 + 8)` FP32 threadgroup tile, or 2,560
bytes, exactly as one current stage does. It does **not** need six tiles live at
once. The second barrier is necessary so early threads cannot overwrite the
shared tile while late threads still read it.

Expected structural change per comparison:

- main Malta: 6 dispatches to 1;
- subscale Malta: 6 dispatches to 1;
- six accumulation-plane writes/read-modify-writes to two final writes per
  scale, or 12 to four across both scales; and
- no change to Malta neighborhood arithmetic or source-plane reads.

Across the current two effort-7 comparisons, this removes 20 dispatches. It
does not reduce the six sets of input samples or the dominant neighborhood
math, so a large speedup is not guaranteed.

Implementation notes:

- add a parameter block containing six reference/distorted strides, weights,
  norms, low/full selectors, and offsets, or bind the twelve plane views
  directly for the first experiment;
- preserve `kMaltaAccumulationOrder` on the host and duplicate it in a tested
  shader constant only if avoiding dynamic indexing is measurably better;
- when a Malta response stage is being captured for diagnostics, either write
  that selected response from the fused loop or temporarily use the baseline
  implementation; and
- compare both `32 x 8` and `16 x 8` tiles if counters show the six-stage live
  range reducing occupancy. Do not assume the current shape remains optimal.

Primary failure modes are instruction-cache growth, register spills, lower
occupancy, and barrier cost. Stop this slice if GPU time fails to improve
reliably even after trying the two bounded tile shapes; do not keep it merely
because it lowers the dispatch count.

### Slice B: channel-fused psycho construction

Several current grids differ only by channel. One thread can compute all
independent channels at a pixel and write planar outputs. This reduces grid
launches and thread setup while preserving the same loads, stores, and
arithmetic.

Implement these kernels separately so each can be accepted or rejected:

1. three-channel 5-tap horizontal blur: 3 dispatches to 1;
2. three-channel 33-tap transposed convolution: 3 to 1;
3. X/Y 15-tap transpose pass: 2 to 1;
4. X/Y high-frequency vertical pass: 2 to 1;
5. X/Y 7-tap transpose pass: 2 to 1;
6. X/Y ultra-frequency vertical pass: 2 to 1; and
7. three-channel 2x subsample: 3 to 1.

Without changing the separable-convolution topology, items 1--6 can reduce
`EncodePsychoImage` from 19 to about 11 dispatches. Item 7 removes two more
dispatches wherever subsampling occurs. The rough Butteraugli graph would move
from 44 to about 26 dispatches for reference preparation and from 72 to about
54 for one comparison.

Use explicit scalar channel accumulators initially. A `float3` spelling is
acceptable only if generated code and strict results match; planar buffers do
not guarantee beneficial vector loads. Every variant needs a Shader Cost Graph
or counter comparison because tripling per-thread live values can reduce
occupancy enough to offset the lower thread count.

The 33-tap pass is the most likely channel-fusion winner because it amortizes
address/bounds setup over substantial arithmetic. The 5-tap and subsample
kernels are lower-risk implementation probes but may have a smaller GPU-time
effect.

### Slice C: tile-local separable convolutions

The current generic convolution writes a transposed full-image intermediate
and a later kernel reads it to complete the other axis. A true convolution
superkernel can stage a haloed tile in threadgroup memory, perform both axes,
and emit only the semantic output planes.

Investigate by radius, not with one generic kernel:

| Kernel length | Radius | Candidate | Risk |
| ---: | ---: | --- | --- |
| 5 | 2 | fuse input blur and Opsin conversion | low-to-medium |
| 7 | 3 | fuse high-to-ultra separable filtering | medium |
| 13 | 6 | fuse mask blur | medium |
| 15 | 7 | fuse medium-to-high filtering | medium-to-high |
| 33 | 16 | fuse low/medium filtering | high |

The 5-tap path is attractive: a three-channel halo tile can remove all three
horizontal intermediate writes and reads before the existing Opsin transform.
The 7- and 13-tap paths may also fit comfortably.

The 33-tap path must be treated skeptically. For an `8 x 8` output tile, three
FP32 input-channel tiles with a 16-pixel halo require roughly 19.2 KiB before
any additional shared state, and adjacent threadgroups redundantly load large
halos. Larger output tiles reduce halo duplication but raise thread count and
working-set pressure. Alternatives to test are:

- channel fusion while retaining the global transpose intermediate;
- a two-superkernel strip design rather than a one-dispatch 2D blur;
- one channel per dispatch with a larger reusable tile; and
- a sliding-window implementation that trades registers for fewer loads.

For each radius, calculate and record:

```text
global bytes avoided
halo-load amplification
threadgroup bytes per group
estimated registers or compiler occupancy report
measured occupancy and occupancy-manager target
measured GPU stage time
```

Reject any variant that wins only on dispatch count while increasing the
complete psycho stage.

### Slice D: resident final metric and reductions

The ordinary encoder consumes block distances and a scalar score. It does not
need the complete per-pixel distance map unless a final diagnostic is requested.
The public Butteraugli API and stage-capture tests do need materialized planes,
so this slice must be a narrow resident-only entry point rather than a public
mode switch.

The candidate dataflow is:

```text
subscale final map + main-scale difference intermediates
                         |
                         v
strategy-aware anchor threadgroups
  - evaluate the main-scale final masked distance
  - compose it with the cached subscale value on demand
  - accumulate distance^16 for the transform footprint
  - emit one block distance
  - emit one partial scalar maximum
                         |
                         v
small final maximum reduction
```

Each pixel belongs to one non-overlapping selected transform footprint, so the
block-reduction grid can become the consumer of the final metric instead of
rereading a stored distance map. The transform reduction must remain:

```text
1.2 * pow(mean(pow(distance, 16)), 1.0 / 16.0)
```

The first implementation should retain the quarter-area subscale final map so
that its expensive final expression is evaluated once per subscale pixel, not
four times while processing main-scale pixels. It can potentially remove:

- the final full-resolution distance-map store and reread;
- the separate compose pass;
- the first full-image scalar-maximum pass; and
- some or all of the seven per-strategy block-reduction dispatch boundaries.

Do not start here. It couples Butteraugli to the strategy-anchor layout, has a
larger correctness surface, and risks divergent work if all transform shapes
share one kernel. First establish whether Malta and channel fusion move the
measured budget as predicted.

A clean interface is a new internal function in
`metal_butteraugli_encoding.h` that accepts the resident reduction sinks and
anchor metadata. The existing public comparison descriptor remains unchanged.
If `collect_final_butteraugli_score`, map capture, or a public diagnostic plane
is requested, use the complete-map implementation. Once qualified, this is a
single semantic split—materialized diagnostics versus encoder sinks—not a
general execution-plan framework.

### Slice E: threadgroup specialization

Threadgroup tuning is a follow-up to each fusion, not a substitute for it. The
current `8 x 8` generic plane shape should be compared with `16 x 8`, `16 x 16`,
and any algorithm-native tile only when supported by the pipeline's
`threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`.

For each fused kernel, capture:

- achieved occupancy and occupancy-manager target;
- compute shader launch limiter;
- FP32, instruction-throughput, texture/buffer, L1, LLC, and MMU limiters;
- threadgroup-memory allocation;
- compiler-reported register or spill evidence where available; and
- edge inefficiency for Kodak, 1080p, and odd padded dimensions.

Reduced occupancy is not automatically a regression. A more productive kernel
can be faster with fewer resident SIMD groups if its limiting execution unit is
already busy. Judge the shape by elapsed stage and workflow time, using counters
to explain the result.

## Implementation sequence

### Phase 0: refresh and freeze the baseline

1. Build the current branch in a fresh Release directory.
2. Record commit, tree hash, compiler/Xcode version, macOS version, device, GPU
   family, command line, input hashes, output hashes, and thermal state.
3. Capture ordinary effort-7 padded 1080p and padded 4K public-workflow
   baselines with the final diagnostic score disabled.
4. Capture one stage profile to verify that the current source still records
   44 reference and about 72 per-comparison Butteraugli dispatches.
5. Capture a clean Metal System Trace with the same Performance Limiters
   counter set used previously.

Use two build directories for temporary A/B selection, for example
`build/release-metal-baseline` and `build/release-metal-fused`. A private CMake
compile definition may select the experiment. It must not enter the public C,
C++, or benchmark behavior contract, and it should be deleted after promotion.

### Phase 1: six-way Malta

Files expected to change:

- `src/gpu/metal/kernels/butteraugli.metal`: parameter layout and kernel;
- `src/gpu/metal/metal_backend_internal.h`: temporary pipeline state;
- `src/gpu/metal/metal_butteraugli.cpp`: pipeline creation, binding, dispatch,
  capture fallback, and resource checks;
- `tests/metal_butteraugli_test.cpp`: stage and end-result comparisons; and
- profiling/benchmark code only as needed to expose the private build identity.

Gates before retaining the slice:

- exact or existing-tolerance agreement for all six captured Malta responses;
- complete Metal-vs-CPU distance map and score tests without wider tolerance;
- no NaN, bounds, odd-size, or small-image failures;
- deterministic repeated output;
- lower Malta GPU time in at least four of five alternating independent-process
  pairs at padded 4K; and
- no statistically credible regression on Kodak or padded 1080p.

If successful, commit this slice independently before beginning channel fusion.

#### Phase 0 and Phase 1 result (2026-09-04)

The current-revision Phase 0 baseline was captured from commit `25f3cbd` on an
Apple M4 Pro with a fresh Release build. Five independent processes, each with
two discarded warmups and one retained effort-7 fully-resident sample, produced
these medians:

| Workload | Complete encode | Quantization pipeline | Codestream bytes |
| --- | ---: | ---: | ---: |
| padded 1080p | `91.715 ms` | `72.101 ms` | `410072` |
| padded 4K | `315.240 ms` | `271.036 ms` | `1606911` |

A five-sample padded-4K stage profile measured `22.301 ms` for reference
preparation and `117.653 ms` for resident AQ. Inside resident AQ, the medians
were `13.114 ms` for main-scale Malta, `3.246 ms` for subscale Malta,
`32.012 ms` for main-scale psycho construction, and `8.660 ms` for subscale
psycho construction. The associated capture retained 44 reference dispatches
and 316 resident-AQ dispatches. A fresh System Trace and Performance Limiters
capture were also recorded; their workflow timings remain diagnostic rather
than public benchmark results.

The Phase 1 prototype fused all six Malta stages at a scale into one kernel,
preserved the established accumulation order, and reused one haloed
threadgroup tile. Both `32 x 8` and `16 x 8` variants compiled with strict Metal
settings and passed the focused Metal Butteraugli test. The `32 x 8` variant
removed exactly 20 dispatches from the profiled two-evaluation resident AQ
submission, reducing the resident count from 316 to 296. That structural win
did not translate into a meaningful timing win:

- the median Malta time changed from `16.259 ms` to `16.167 ms` (`0.6%`);
- the median resident-AQ time changed from `117.563 ms` to `116.871 ms`
  (`0.6%`); and
- the `16 x 8` shape was not materially better.

In five alternating independent-process complete-encode pairs, `32 x 8` won
three of five padded-1080p pairs, but only two of five padded-4K pairs. The 4K
median changed from `308.607 ms` to `311.022 ms`, a `0.78%` regression, while
all output byte counts remained identical. This fails both the majority-pair
and practical-value promotion gates. The prototype has therefore been removed
rather than leaving an inactive pipeline or build selector in production.

This rejection narrows the diagnosis: the number of Malta dispatches is not by
itself the fundamental cost. The fused kernel still performs all six
neighborhood evaluations and introduces repeated tile loads and barriers, so
its useful arithmetic dominates the saved inter-dispatch work. Phase 2 should
proceed as a sequence of independently measured channel-fusion experiments,
starting with subsampling and the 5-tap horizontal blur, rather than pursuing a
larger Malta superkernel.

### Phase 2: channel fusion

Implement and measure in this order:

1. three-channel subsample;
2. three-channel 5-tap horizontal blur;
3. three-channel 33-tap transpose;
4. paired 15-tap/high kernels; and
5. paired 7-tap/ultra kernels.

The order starts with simple correctness probes, then reaches the highest-work
channel-fusion candidate, and leaves coupled in-place frequency updates last.
Do not aggregate all five changes before measuring; resource interactions make
it necessary to know which kernel actually wins.

#### Phase 2.1 result: three-channel subsample (2026-09-04)

The three per-channel subsample dispatches were replaced experimentally by one
kernel whose thread processed all three channels. Independent input and output
strides were preserved, as was the scalar arithmetic order. Strict Metal
compilation and the full focused Metal Butteraugli stage-capture test passed.

On the five-sample padded-4K stage profile, the experiment reduced reference
preparation from 44 to 42 dispatches, resident AQ from 316 to 312 dispatches,
and the two subscale psycho groups from 44 to 40 dispatches. Despite that,
median subscale psycho time increased from `8.601 ms` to `8.996 ms` (`4.6%`),
while resident AQ was effectively flat (`117.145 ms` versus `117.360 ms`).

Five alternating independent-process complete-encode pairs were consistent at
4K: the fused path lost all five, and its median increased from `311.913 ms` to
`317.442 ms` (`1.77%`). At 1080p it won only three of five pairs and the
medians were effectively equal (`87.449 ms` versus `87.385 ms`). Codestream
sizes were identical in every pair.

The slice is rejected and its kernel and build selector have been removed.
Serializing three independent channel calculations within each thread reduced
launch count but did not remove arithmetic or source traffic, and exposed less
independent work to the GPU scheduler. This is further evidence that dispatch
count alone is a poor proxy for useful cost. Phase 2.2 should test the 5-tap
horizontal blur separately because it can at least share normalized weights;
it must not inherit the subsample result by assumption.

#### Phase 2.2 result: three-channel 5-tap horizontal blur (2026-09-04)

The second prototype shared 5-tap weight loading and normalization across all
three channels while preserving each channel's input and output stride and
convolution operation order. Strict Metal compilation and both control and
fused focused Butteraugli tests passed.

In seven alternating independent-process padded-4K stage-profile pairs, the
fused path reduced reference preparation from 44 to 40 dispatches and resident
AQ from 316 to 308. Main-scale psycho construction improved in six of seven
pairs, with its median moving from `32.284 ms` to `32.158 ms` (`0.39%`). The
subscale median moved from `8.904 ms` to `8.870 ms` (`0.38%`) but improved in
only three of seven pairs. Resident AQ improved in five of seven pairs and its
median moved from `117.736 ms` to `117.515 ms` (`0.19%`). Reference preparation
instead regressed from `22.329 ms` to `22.541 ms` (`0.95%`) and improved in
only three of seven pairs.

Complete-encode evidence did not establish a user-visible win. At 4K the fused
path won four of seven pairs, but the medians were effectively equal
(`312.551 ms` control and `312.015 ms` fused). At 1080p it lost five of seven
pairs and the medians were indistinguishable (`90.109 ms` and `90.106 ms`).
Codestream sizes remained identical.

The slice is rejected and removed. Its small per-pixel weight reuse is not
enough to make the total reference-plus-resident GPU path reliably faster, and
the complete-encode result fails the measurable-improvement gate. The next
channel experiment targets the 33-tap transpose, where sharing the truncated
weight-sum loop across channels removes substantially more arithmetic.

#### Phase 2.3 result: three-channel 33-tap transpose (2026-09-04)

This prototype retained three independent convolution accumulators but shared
the clipped source interval, weight loads, and weight-sum accumulation across
channels. It preserved independent plane strides and the established
per-channel summation order. Strict Metal compilation and the complete focused
Metal Butteraugli stage-capture test passed in both control and fused builds.

Seven alternating independent-process padded-4K stage-profile pairs showed a
repeatable GPU improvement:

| Scope | Control median | Fused median | Change | Fused wins |
| --- | ---: | ---: | ---: | ---: |
| reference preparation | `22.070 ms` | `21.318 ms` | `-3.40%` | 6/7 |
| resident AQ | `117.952 ms` | `115.996 ms` | `-1.66%` | 6/7 |
| main-scale psycho construction | `31.919 ms` | `29.991 ms` | `-6.04%` | 7/7 |
| subscale psycho construction | `8.813 ms` | `8.410 ms` | `-4.58%` | 7/7 |

The change reduces reference preparation from 44 to 40 dispatches and resident
AQ from 316 to 308. Unlike the rejected smaller fusions, the speedup is not
attributed to dispatch count alone: two of the three clipped 33-tap weight-sum
loops are eliminated for every output while the three channel sums remain
independent.

Complete padded-workload results also passed the gate. The fused path won all
seven 4K pairs, with the median moving from `316.119 ms` to `311.340 ms`
(`-1.51%`), and five of seven 1080p pairs, with the median moving from
`91.790 ms` to `89.686 ms` (`-2.29%`). Codestream sizes were identical in every
pair.

The content-diversity run covered all 24 Kodak images plus two real 1080p and
two real 4K photographs. Five-sample Kodak process medians were mixed, with 13
faster and 11 slower, because the GPU saving is small relative to process and
host-tail noise at 512x768. Targeted profiles of two apparent wall-time losers
resolved that ambiguity: on Kodak 03 the main/subscale psycho medians improved
from `1.438/0.490 ms` to `1.308/0.449 ms`; on Kodak 12 they improved from
`1.397/0.486 ms` to `1.275/0.446 ms`. Reference preparation and resident AQ
also improved on both.

All four real-photo profiles improved their reference, resident-AQ, main
psycho, and subscale psycho medians. Main psycho improved by `6.3%` to `7.4%`
and subscale psycho by `7.4%` to `9.5%`. Complete real-photo timing remained
noisier: each 1080p image won four of five paired processes, while the 4K
images were neutral-to-mixed despite their consistent GPU-stage wins.

A focused Performance Limiters capture repeatedly exercised the 1919x1079
Butteraugli path and sampled the new kernel directly. The Shader Timeline
interval was `92.167 us` and represented `9.0%` of its sampled GPU kick. Across
the three fully contained `20.792 us` compute-counter samples, mean kernel
occupancy was `47.0%` against a `50.6%` occupancy-manager target. Mean shader
launch and instruction-throughput limiters were `87.0%` and `61.7%`; mean L1
limiter was `33.2%`. Stack L1 reads were effectively zero, stack writes
averaged `0.002%`, and no ray-tracing scratch activity was present. The counter
sample is narrow, but it rules out a pathological occupancy collapse or spill
regression and remains consistent with launch/instruction pressure.

Control and fused CLI builds produced byte-identical codestreams on
representative Kodak, 1080p, and 4K photographs. The slice is promoted as the
unconditional three-channel 33-tap path; its private build selector and host
fallback have been removed. The generic single-channel transpose pipeline
remains because later high, ultra, and masking passes still use it.

#### Phase 2.4 result: paired 15-tap/high kernels (2026-09-04)

The 15-tap experiment evaluated its two axes independently. The transpose
prototype processed X and Y with separate scalar accumulators while sharing the
clipped interval, weight loads, and weight-sum loop. The vertical prototype did
the same before applying the existing channel-specific high-frequency updates.
Both retained the full-image transposed intermediate, including its FP32
materialization point. Two already-provisioned psycho work planes held the
simultaneous X/Y intermediates, so the experiment did not increase scratch
capacity.

Control, transpose-only, vertical-only, and combined Release builds all passed
strict Metal compilation and the complete focused Metal Butteraugli test. Each
reported the same maximum CPU-reference errors: `0.000549316` for the distance
map and score and `0.000396729` for captured stages.

The first four-way padded-4K stage matrix showed that the vertical-only variant
lost most reference, resident, main-psycho, and subscale-psycho pairs. Two
direct seven-pair experiments then isolated each decision. Relative to the
unfused control, transpose-only produced:

| Scope | Control median | Transpose median | Change | Transpose wins |
| --- | ---: | ---: | ---: | ---: |
| reference preparation | `20.940 ms` | `21.326 ms` | `+1.84%` | 1/7 |
| resident AQ | `117.094 ms` | `116.538 ms` | `-0.47%` | 4/7 |
| main-scale psycho construction | `30.703 ms` | `30.474 ms` | `-0.75%` | 5/7 |
| subscale psycho construction | `8.505 ms` | `8.635 ms` | `+1.54%` | 3/7 |

This removed two reference dispatches (`40` to `38`) and four resident
dispatches (`308` to `304`), but the small main-scale saving did not overcome
the reference and subscale regressions. Adding the paired vertical kernel to
the transpose prototype removed the same counts again (`38` to `36` and `304`
to `300`) but also failed its incremental gate:

| Scope | Transpose median | Combined median | Change | Combined wins |
| --- | ---: | ---: | ---: | ---: |
| reference preparation | `20.926 ms` | `21.188 ms` | `+1.25%` | 2/7 |
| resident AQ | `116.254 ms` | `117.349 ms` | `+0.94%` | 3/7 |
| main-scale psycho construction | `30.474 ms` | `30.599 ms` | `+0.41%` | 2/7 |
| subscale psycho construction | `8.661 ms` | `8.514 ms` | `-1.70%` | 5/7 |

Complete public-workflow timing resolved the marginal transpose result against
retention. Transpose-only won three of seven padded-1080p pairs and one of
seven padded-4K pairs. Its median changed from `88.370` to `88.782 ms`
(`+0.47%`) at 1080p and from `305.054` to `307.175 ms` (`+0.70%`) at 4K.
Codestream sizes remained identical at `410072` and `1606911` bytes,
respectively.

Both 15-tap fusions are therefore rejected and their shaders, pipeline states,
scratch scheduling, and private build selectors have been removed. As with the
5-tap result, sharing one short normalization loop does not compensate for
serializing X/Y work within a thread on this device. Phase 2.5 should still
measure the 7-tap/ultra pair independently: its arithmetic is shorter, but its
different nonlinear output work and lower per-pass cost make the launch versus
parallelism tradeoff a separate empirical question.

The retained experiment artifacts are:

- `/private/tmp/gjxl-metal-fusion-high2-stage.kE1WDB` (four-way stage matrix);
- `/private/tmp/gjxl-metal-fusion-high2-transpose-pairs.zvVMew` (transpose
  isolation);
- `/private/tmp/gjxl-metal-fusion-high2-vertical-pairs.E8WN2y` (vertical
  incremental isolation); and
- `/private/tmp/gjxl-metal-fusion-high2-wall-pairs.I4yAbI` (public-workflow
  pairs).

#### Phase 2.5 result: paired 7-tap/ultra kernels (2026-09-04)

The final channel-fusion experiment repeated the independent-axis design for
the 7-tap ultra-frequency passes. The transpose prototype shared the clipped
interval, weights, and normalization while retaining scalar X/Y accumulators.
The vertical prototype shared only that convolution setup; it preserved X's
two `remove_range` operations and Y's clamp, scale, and `amplify_range`
operations in their established order. As in Phase 2.4, two existing psycho
work planes held the simultaneous intermediates without increasing arena
capacity or removing the full-image FP32 materialization boundary.

Control, transpose-only, vertical-only, and combined builds passed strict Metal
compilation and the complete focused Metal Butteraugli test. All four again
reported maximum CPU-reference errors of `0.000549316` for the distance map and
score and `0.000396729` for captured stages.

Seven rotated padded-4K stage-profile rounds produced mixed results rather than
a stage-wide improvement:

| Variant and scope | Control median | Variant median | Change | Variant wins |
| --- | ---: | ---: | ---: | ---: |
| transpose: reference | `21.770 ms` | `21.101 ms` | `-3.07%` | 6/7 |
| transpose: resident AQ | `116.914 ms` | `117.140 ms` | `+0.19%` | 3/7 |
| transpose: main psycho | `30.708 ms` | `30.736 ms` | `+0.09%` | 4/7 |
| transpose: subscale psycho | `8.479 ms` | `8.323 ms` | `-1.85%` | 4/7 |
| vertical: reference | `21.770 ms` | `21.433 ms` | `-1.54%` | 5/7 |
| vertical: resident AQ | `116.914 ms` | `116.446 ms` | `-0.40%` | 5/7 |
| vertical: main psycho | `30.708 ms` | `30.576 ms` | `-0.43%` | 5/7 |
| vertical: subscale psycho | `8.479 ms` | `8.629 ms` | `+1.77%` | 2/7 |
| combined: reference | `21.770 ms` | `21.209 ms` | `-2.58%` | 5/7 |
| combined: resident AQ | `116.914 ms` | `117.669 ms` | `+0.65%` | 3/7 |
| combined: main psycho | `30.708 ms` | `30.892 ms` | `+0.60%` | 3/7 |
| combined: subscale psycho | `8.479 ms` | `8.592 ms` | `+1.33%` | 2/7 |

Each individual half reduced reference dispatches from `40` to `38` and
resident dispatches from `308` to `304`; the combined form reached `36` and
`300`. The combined form nevertheless regressed the broader resident and
psycho scopes, showing again that the saved grids are not free throughput when
independent channel work is serialized within one thread.

Complete public-workflow measurements did not establish an individual winner:

| Workload | Control median | Transpose median | Transpose wins | Vertical median | Vertical wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| padded 1080p | `89.510 ms` | `89.836 ms` (`+0.36%`) | 4/7 | `89.582 ms` (`+0.08%`) | 3/7 |
| padded 4K | `311.243 ms` | `310.495 ms` (`-0.24%`) | 5/7 | `313.412 ms` (`+0.70%`) | 3/7 |

All variants emitted the same `410072`-byte 1080p and `1606911`-byte 4K
codestream sizes. The small, contradictory changes are below a credible
promotion threshold, and neither variant improves both resolutions. Both
7-tap fusions are rejected; their shaders, pipeline states, scratch scheduling,
and private selectors have been removed. Counter and full-corpus qualification
were intentionally skipped after the stage and public-workflow gates failed.

This completes Phase 2. Of its five channel-fusion candidates, only the
three-channel 33-tap transpose is retained. The result supports a narrower
principle than "fewer launches": channel fusion pays when it removes enough
duplicated normalization work to offset the loss of independently schedulable
channel threads.

The retained experiment artifacts are:

- `/private/tmp/gjxl-metal-fusion-ultra2-stage.EiuKYZ` (four-way stage matrix);
  and
- `/private/tmp/gjxl-metal-fusion-ultra2-wall.dNo3zb` (public-workflow pairs).

### Phase 3: tile-local convolution

Prototype the 5-tap blur-plus-Opsin kernel first. Continue through 7, 13, and 15
taps only after each previous radius demonstrates lower GPU time. Treat the
33-tap path as an independent research experiment with explicit memory/halo
accounting. Retain the channel-fused separable implementation whenever it is
faster than the fully tiled version.

#### Phase 3.1 result: tiled 5-tap blur plus Opsin (2026-09-04)

The first tile-local prototype is retained. One `16 x 8` threadgroup now loads
a reflected radius-2 RGB tile, materializes the three horizontal convolution
results in FP32 threadgroup memory, applies the vertical convolution, and
performs the unchanged Opsin transform. It replaces three full-image
horizontal grids and the subsequent Opsin grid with one tiled grid. The
threadgroup uses `5,184` bytes: three `20 x 12` raw tiles and three `16 x 12`
horizontal tiles. Its raw halo amplification is `1.875x` for a full tile.

The first profiling prototype uncovered an important scheduling correctness
constraint. Expanded and subsampled RGB initially occupied the same three
working planes to which the fused kernel wrote Opsin output. That was safe with
separate whole-image horizontal passes, but not with independent
threadgroups: one group could overwrite an input pixel before a neighboring
group loaded it as halo. GPU-profile validation detected this by reporting a
changed encoded result. Expansion and subsampling now place temporary RGB in
three already-free psycho work planes, disjoint from the Opsin output until the
tiled grid completes. The later convolution stages reuse those planes, so the
fix does not increase scratch capacity. The invalid profile was discarded.

The retained implementation preserves the established convolution weight
normalization, pair-addition order, explicit unfused multiply-adds, reflected
edge behavior, and the FP32 horizontal materialization point. Control and
fused builds reported the same focused Metal Butteraugli maxima:
`0.000549316` for the map and score and `0.000396729` for captured stages.
Profiled padded-4K encodes were stable after the scratch remap.

Seven rotated padded-4K stage-profile rounds compared `8 x 8`, `16 x 8`,
`16 x 16`, and `32 x 8` tiles with the separable control. Every tile improved
all seven pairs in reference preparation, resident AQ, main psycho, and
subscale psycho time. The retained `16 x 8` shape produced:

| Scope | Control median | Tiled median | Change | Tiled wins |
| --- | ---: | ---: | ---: | ---: |
| reference preparation | `21.764 ms` | `18.720 ms` | `-13.99%` | 7/7 |
| resident AQ | `116.741 ms` | `111.527 ms` | `-4.47%` | 7/7 |
| main-scale psycho construction | `30.680 ms` | `25.948 ms` | `-15.43%` | 7/7 |
| subscale psycho construction | `8.860 ms` | `7.812 ms` | `-11.83%` | 7/7 |

Reference preparation falls from `40` to `34` dispatches and the
two-evaluation resident submission from `308` to `296`. On the padded-4K
geometry, this removes about `93.3 million` launched threads across reference
preparation and resident AQ. It also eliminates three horizontal intermediate
writes and fifteen nominal vertical intermediate reads per psycho invocation,
or about `2.084 GiB` of shader-request traffic over those six psycho images.
That traffic estimate is not a DRAM-bandwidth measurement.

The other shapes exposed a resolution-dependent tradeoff. `32 x 8` had the
lowest padded-4K stage time, but `16 x 8` gave the best padded-1080p workflow
median and uses less threadgroup memory and half as many threads. Seven rotated
complete-workflow rounds favored the balanced `16 x 8` default:

| Workload | Control median | Tiled median | Change | Tiled wins |
| --- | ---: | ---: | ---: | ---: |
| padded 1080p | `90.820 ms` | `86.989 ms` | `-4.22%` | 6/7 |
| padded 4K | `307.763 ms` | `306.024 ms` | `-0.57%` | 6/7 |

Control and production builds emitted byte-identical codestreams for a small
odd-sized fixture and representative Kodak, real 1080p, and real 4K images.
The old Opsin pipeline and private build selectors have been removed. The
generic 5-tap horizontal and vertical pipelines remain because diagnostic
sigma-1.2 blur capture still uses them. Pipeline construction validates the
new kernel's thread-count and dynamic threadgroup-memory requirements.

This is the first fusion in the study to remove large full-image intermediate
traffic as well as dispatches, and its stage-wide improvement is much larger
than the rejected launch-only channel fusions. Phase 3.2 should now prototype
the radius-3 7-tap separable path under the same arithmetic and aliasing rules.

The retained experiment artifacts are:

- `/private/tmp/gjxl-metal-fusion-opsin5-stage-safe.5s7NTW` (five-way stage
  matrix);
- `/private/tmp/gjxl-metal-fusion-opsin5-wall.Ks9grt` (four-way workflow
  matrix); and
- `/private/tmp/gjxl-metal-fusion-opsin5-codestream.SufU1X` (exact-output
  comparisons).

#### Phase 3.2 result: tiled 7-tap ultra filtering (2026-09-04)

The radius-3 prototype fused each channel's transposed horizontal pass,
vertical pass, and ultra-frequency nonlinear update into one tiled dispatch.
Because the existing path updates its high-frequency input in place, the
prototype first redirected the pre-ultra X/Y planes into two free image work
planes and emitted final high and ultra planes disjointly. This avoided the
cross-threadgroup halo race found during Phase 3.1 without adding a copy,
allocation, or scratch plane.

The candidate removed two dispatches per psycho image. Reference preparation
fell from `34` to `30` dispatches and resident AQ from `296` to `288`, removing
about `62.2 million` launched threads at padded 4K. It also eliminated one
horizontal intermediate write and seven nominal vertical intermediate reads
per channel, about `1.853 GiB` of shader-request traffic across both channels
and all six psycho images. That traffic estimate excludes cache effects and is
not measured DRAM traffic.

Four tile shapes compiled with strict Metal settings. Their threadgroup memory
and full-tile raw halo amplification were:

| Tile | Threadgroup memory | Raw halo amplification |
| --- | ---: | ---: |
| `8 x 8` | `1,232 B` | `3.0625x` |
| `16 x 8` | `2,128 B` | `2.40625x` |
| `16 x 16` | `3,344 B` | `1.890625x` |
| `32 x 8` | `3,920 B` | `2.078125x` |

Control and tiled builds passed the complete focused Metal Butteraugli test
with identical maxima: `0.000549316` for map and score and `0.000396729` for
captured stages. Both profiled workflows retained the same `1606911`-byte
padded-4K result. Correctness and resource use therefore passed; performance
did not.

Seven rotated padded-4K stage-profile rounds produced these median changes
relative to the separable control:

| Tile | Reference | Resident AQ | Main psycho | Subscale psycho |
| --- | ---: | ---: | ---: | ---: |
| `8 x 8` | `+8.81%` (1/7) | `+1.95%` (0/7) | `+5.44%` (0/7) | `+9.27%` (0/7) |
| `16 x 8` | `+8.04%` (1/7) | `+0.99%` (0/7) | `+2.57%` (0/7) | `+6.64%` (0/7) |
| `16 x 16` | `+15.08%` (1/7) | `+0.04%` (2/7) | `-0.52%` (7/7) | `+4.02%` (0/7) |
| `32 x 8` | `+1.42%` (2/7) | `+0.90%` (0/7) | `+1.94%` (0/7) | `+6.25%` (0/7) |

Parentheses report tiled wins out of seven paired rounds. The `16 x 16`
shape's small main-scale improvement did not survive at subscale and did not
improve the enclosing resident submission. All other shapes regressed every
resident, main-psycho, and subscale-psycho pair. The short convolution's two
barriers, cooperative halo work, clipped-boundary control, and partial-tile
cost outweigh the removed global intermediate on this device.

The slice is rejected and all prototype shaders, pipeline states, scratch
routing, resource checks, and private selectors have been removed. Complete
workflow and counter qualification were intentionally skipped after the GPU
stage gate failed. Under the plan's explicit radius gate, the 13- and 15-tap
tile-local variants are not pursued: they have larger halos and cannot advance
after the shorter 7-tap form failed. The independent 33-tap research question
remains separate because its much greater arithmetic and global traffic could
change the tradeoff.

The retained experiment artifact is
`/private/tmp/gjxl-metal-fusion-ultra7-stage.pNIbhE`.

#### Phase 3.3 result: direct-load tiled 33-tap low/medium filtering (2026-09-04)

The independent radius-16 experiment is retained. It replaces the
channel-fused horizontal transpose and the vertical low/medium pass with one
`16 x 64` tiled kernel. Threads cooperatively compute the three horizontal
planes directly from the planar XYB inputs, store only those FP32 results in
threadgroup memory, synchronize once, and then apply the vertical convolution
and the existing low/medium transforms. The horizontal materialization point
therefore remains FP32, but it no longer occupies three full-image scratch
planes or crosses device memory.

The production threadgroup uses 1,024 threads and `18,432 B` of memory for
three `16 x 96` horizontal planes. Its interior vertical-halo amplification is
`1.5x`: each 64-row output tile computes 32 extra horizontal rows. Pipeline
creation validates both the 1,024-thread limit and the dynamic memory
requirement before the Metal backend becomes available.

The first family of prototypes cached the complete three-channel input halo as
well as the horizontal results. Even its least redundant tested geometry had
an `11.67x` raw-halo amplification, and all four shapes regressed every one of
seven padded-4K stage pairs:

| Raw tile | Threadgroup memory | Raw-halo amplification | Reference mean | Resident mean | Main psycho mean | Subscale psycho mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `8 x 16` | `27,648 B` | `15.0x` | `+63.56%` | `+38.00%` | `+130.84%` | `+112.76%` |
| `16 x 8` | `30,720 B` | `15.0x` | `+98.83%` | `+55.11%` | `+189.13%` | `+161.73%` |
| `8 x 24` | `32,256 B` | `11.67x` | `+39.21%` | `+20.76%` | `+72.26%` | `+61.44%` |
| `4 x 32` | `30,720 B` | `18.0x` | `+64.22%` | `+32.57%` | `+112.57%` | `+96.81%` |

The direct-load design avoids that raw-tile replication. Four geometries then
improved every resident, main-psycho, and subscale-psycho pair. The `16 x 64`
shape was the clear psycho-stage winner:

| Direct tile | Threadgroup memory | Reference median | Resident median | Main psycho median | Subscale psycho median |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | n/a | `29.929 ms` | `107.360 ms` | `25.024 ms` | `7.330 ms` |
| `8 x 64` | `9,216 B` | `29.009 ms` (`-3.07%`, 4/7) | `105.908 ms` (`-1.35%`, 7/7) | `23.763 ms` (`-5.04%`, 7/7) | `6.982 ms` (`-4.75%`, 7/7) |
| `16 x 32` | `12,288 B` | `28.952 ms` (`-3.27%`, 5/7) | `105.992 ms` (`-1.27%`, 7/7) | `23.975 ms` (`-4.19%`, 7/7) | `7.052 ms` (`-3.79%`, 7/7) |
| `16 x 64` | `18,432 B` | `28.294 ms` (`-5.46%`, 6/7) | `104.419 ms` (`-2.74%`, 7/7) | `22.576 ms` (`-9.78%`, 7/7) | `6.709 ms` (`-8.48%`, 7/7) |
| `32 x 32` | `24,576 B` | `27.956 ms` (`-6.59%`, 5/7) | `106.175 ms` (`-1.10%`, 7/7) | `23.928 ms` (`-4.38%`, 7/7) | `7.059 ms` (`-3.70%`, 7/7) |

Parentheses report change from the separable control and tiled wins out of
seven rotated rounds. The retained path reduces reference preparation from 34
to 32 dispatches and the two-evaluation resident submission from 296 to 292.
At padded 4K it removes about `30.8 million` launched threads across reference
preparation and resident AQ. A nominal shader-request accounting also removes
about `6.08 GiB` across the six psycho images after charging the direct path
for its `1.5x` duplicated horizontal input work. This estimate is not measured
DRAM traffic and excludes edge rounding and cache effects.

Seven alternating independent-process complete-workflow pairs confirmed a
smaller but consistent end-to-end benefit:

| Workload | Control median | Tiled median | Change | Tiled wins |
| --- | ---: | ---: | ---: | ---: |
| padded 1080p | `85.118 ms` | `84.387 ms` | `-0.86%` | 7/7 |
| padded 4K | `303.917 ms` | `302.157 ms` | `-0.58%` | 5/7 |

The complete focused Metal Butteraugli test passed with the same established
maxima as the control: `0.000549316` for map and score and `0.000396729` for
captured stages. Control and tiled builds emitted byte-identical codestreams
for a small odd-sized fixture and representative Kodak, real-1080p, and
real-4K inputs.

A focused Performance Limiters micro-capture repeated the new subscale grid to
obtain clean counter intervals between the preceding subsample shader and the
following high-frequency shader. Across seven trimmed intervals, mean kernel
occupancy was `56.7%` against a `64.6%` occupancy-manager target. Mean compute
shader launch, instruction-throughput, F32, and L1 limiters were `86.8%`,
`73.3%`, `59.0%`, and `19.8%`; buffer L1 miss rate was `8.6%`. Stack L1 reads
and writes each averaged `0.0001%`. This intrusive counter workload is not an
elapsed-time benchmark, but it rules out stack spilling and a pathological
occupancy collapse in the 1,024-thread production shape.

The direct kernel supersedes the Phase 2.3 transpose implementation, so the
old three-channel transpose, vertical low/medium pipeline, full-image
intermediate routing, raw-tile prototype, and private build selectors have all
been removed. The retained source is smaller than the separable implementation
it replaces. The slice therefore qualifies under the plan's exception for a
simple, independently measured change even though Amdahl dilution keeps its
complete-workflow improvement below 5%.

The retained experiment artifacts are:

- `/private/tmp/gjxl-metal-fusion-low33-stage.0hLOxo` (raw-tile stage matrix);
- `/private/tmp/gjxl-metal-fusion-low33-direct-stage.IIZY7x` (direct-tile stage
  matrix);
- `/private/tmp/gjxl-metal-fusion-low33-direct-wall.M0dWmH` (complete-workflow
  pairs);
- `/private/tmp/gjxl-metal-fusion-low33-codestream.7b1EAN` (exact-output
  comparisons); and
- `/private/tmp/gjxl-metal-fusion-low33-counters.xXqOdz` (focused counter
  micro-capture).

### Phase 4: resident sink fusion

After the perceptual kernels stabilize:

1. define a resident-only descriptor containing block-distance and score sinks;
2. implement composed-distance evaluation inside one strategy-family reducer;
3. emit scalar-max partials from the same threadgroups;
4. compare seven family-specialized dispatches with one metadata-driven anchor
   dispatch; and
5. retain the least divergent measured implementation.

The public complete-map path remains the correctness oracle during this phase.

#### Phase 4 result: resident final metric and reductions (2026-09-04)

The retained implementation adds one Metal-internal resident comparison
descriptor with exactly the outputs consumed by the encoder: strategy anchors,
block distances, scalar-score partials, the scalar score, and the shared
numeric-error word. The public `DeviceButteraugliComparisonDescriptor` and its
complete distance map are unchanged.

For a multiscale resident comparison, distorted subscale processing now runs
first and retains its quarter-area final map in the existing final-staging
plane. Main-scale processing then reuses the ordinary psycho scratch but stops
before storing the final full-resolution metric. Seven family-specialized
anchor dispatches evaluate that final expression, compose the subscale value,
accumulate the established `distance^16` transform norm, fill the covered
block distances, and emit one scalar maximum per anchor. The existing maximum
reducer scans only those anchor partials. Small and single-scale comparisons
continue to use the complete-map path.

This removes, per comparison:

- the full-resolution final-metric store;
- the separate full-resolution compose pass;
- the full-resolution first maximum-reduction pass; and
- the later full-resolution reread by the AQ block reducer.

The profiled path follows the same resident graph and exposes a single
`butteraugli.resident_reduction` stage, while the public map and capture APIs
remain the materialized correctness oracle. At padded 4K, five stage samples
measured the affected final/reduction region as follows. The control sum is
main mask/final, sub mask/final plus compose, scalar reduction, and block
reduction; the resident sum is sub mask/final, main mask preparation, and the
resident reduction.

| Scope | Complete-map median | Resident-sink median | Change |
| --- | ---: | ---: | ---: |
| affected final/reduction region | `14.300 ms` | `11.124 ms` | `-3.176 ms` (`-22.2%`) |
| resident-AQ command-buffer GPU time | `103.950 ms` | `100.712 ms` | `-3.238 ms` (`-3.12%`) |
| resident-AQ dispatches, two evaluations | `292` | `286` | `-6` |

Seven alternating independent-process public-workflow pairs, each with two
discarded warmups, confirmed that the GPU-stage gain survives the complete
encoder boundary:

| Workload | Complete-map median | Resident-sink median | Change | Resident wins |
| --- | ---: | ---: | ---: | ---: |
| padded 1080p, complete encode | `82.852 ms` | `81.597 ms` | `-1.255 ms` (`-1.51%`) | 6/7 |
| padded 1080p, quantization | `64.000 ms` | `63.165 ms` | `-0.835 ms` (`-1.30%`) | 6/7 |
| padded 4K, complete encode | `297.625 ms` | `292.323 ms` | `-5.302 ms` (`-1.78%`) | 6/7 |
| padded 4K, quantization | `252.135 ms` | `246.809 ms` | `-5.326 ms` (`-2.11%`) | 5/7 |

All compared codestream sizes were identical. The focused Metal AQ test also
compared the resident policy with its serial complete-map oracle, including
profiled and unprofiled execution. Its maximum block-distance discrepancy was
`2.38419e-07`, versus the existing `5e-4` gate; score, quantization, failure
injection, and readback-contract checks passed without changing tolerances.

A second implementation dispatched every anchor through one metadata-driven
kernel. It removed six family dispatches per comparison, but every threadgroup
had to find its batch and dynamically index five geometry arrays. In the first
seven-pair padded-4K run it regressed quantization by `1.70%` and complete
encoding by `0.91%`, losing 5/7 pairs at both boundaries. A repeat was noisier:
quantization still regressed by `0.29%` and lost 4/7, while complete encoding
was statistically neutral (`-0.17%`, 4/7 wins). Across both runs, the unified
variant's quantization median was `0.77%` slower and the complete boundary was
flat. It was removed along with its private selector and pipeline. The retained
family form has uniform compile-time-like geometry within each dispatch and is
the less divergent implementation.

The first Phase 4 slice deliberately reuses the prefix of the already
allocated full-resolution distance-map plane for score partials. Phase 5 can
replace that allocation with an anchor-sized plane once every non-diagnostic
caller has moved to the sink path; doing so is a memory consolidation, not
part of the measured compute win above.

The retained experiment artifacts are:

- `/private/tmp/gjxl-metal-fusion-resident-wall.I3IvRD` (complete-workflow and
  repeated metadata-dispatch comparisons);
- `/private/tmp/gjxl-metal-fusion-resident-control-stage.json` (complete-map
  stage profile); and
- `/private/tmp/gjxl-metal-fusion-resident-stage.json` (resident-sink stage
  profile).

### Phase 5: consolidate

Status: completed and retained.

All private baseline selectors and rejected experimental pipelines had already
been removed at their phase boundaries. The final consolidation removes the
remaining unconditional full-resolution distance-map allocation from prepared
AQ evaluation and makes its storage metric-specific:

- multiscale Butteraugli production owns only one score partial per strategy
  anchor plus the scalar score;
- the small-image Butteraugli fallback owns only the scalar score;
- maximum-error evaluation owns only its transform-maximum plane; and
- frame-only evaluation owns none of those sinks.

The ordinary fully resident path already used the Phase 4 sink. Phase 5 also
migrates the generic, exact-coefficient production path to it, so no production
multiscale caller materializes the complete distance map. Public diagnostics
still return that map. They alias the prefix of the coefficient-reconstruction
staging plane, which owns three values per coding pixel and is therefore larger
than one unpadded source plane. Diagnostic uploads happen before reconstruction
or after it has completed. The expanded-small-image production fallback records
the map-writing dispatch after reconstruction in the same command encoder, so
Metal dispatch ordering prevents the two lifetimes from overlapping. No lazy
allocation, additional arena, or new lease class is required.

The block reducer now receives its input view explicitly. This avoids retaining
a false object-wide distance-map identity and lets both the diagnostic alias and
the small production fallback reuse the same encoder without changing the
public API. A dedicated `score_partials_` member replaces the earlier practice
of treating the prefix of the complete-map plane as reduction storage.

#### Memory result

The exact Phase 4 parent and Phase 5 candidate reported the following prepared
AQ storage for the existing 1920-by-1080 test fixture:

| Storage | Phase 4 | Phase 5 | Change |
| --- | ---: | ---: | ---: |
| staging capacity | `453,679,924 B` | `445,126,452 B` | `-8,553,472 B` (`-8.16 MiB`, `-1.89%`) |
| peak scratch | `341,704,992 B` | `333,151,520 B` | `-8,553,472 B` (`-8.16 MiB`, `-2.50%`) |

For padded 4K, the structural lower bound is `34,190,592 B` (`32.61 MiB`):
the aligned 3839-by-2159 map and the unused three-values-per-block
maximum-error plane disappear, while at most one aligned partial per 8-by-8
block is added. Larger transforms produce fewer anchors and therefore save
slightly more. This is a capacity reduction, not a claim that every byte was
physically resident at the same instant.

#### Timing and counter result

Because Phase 5 changes storage selection rather than the fully resident
compute graph, its ordinary-path timing gate is non-regression. Seven
alternating independent-process pairs used two warmups and the median of three
retained Release samples per process:

| Workload | Boundary | Phase 4 | Phase 5 | Change | Phase 5 wins |
| --- | --- | ---: | ---: | ---: | ---: |
| padded 1080p | complete encode | `83.119 ms` | `82.746 ms` | `-0.373 ms` (`-0.45%`) | 4/7 |
| padded 1080p | quantization | `63.931 ms` | `63.962 ms` | `+0.031 ms` (`+0.05%`) | 3/7 |
| padded 4K | complete encode | `299.907 ms` | `297.430 ms` | `-2.477 ms` (`-0.83%`) | 6/7 |
| padded 4K | quantization | `251.797 ms` | `251.528 ms` | `-0.269 ms` (`-0.11%`) | 4/7 |

The quantization boundary is effectively neutral, as expected. The complete
boundary has a favorable but modest noisy direction and is not attributed to
removed GPU arithmetic.

The generic exact-coefficient path does remove complete-map computation. Seven
alternating independent-process pairs, each with two warmups and one retained
sample, measured:

| Workload | Boundary | Complete map | Resident sink | Change | Sink wins |
| --- | --- | ---: | ---: | ---: | ---: |
| padded 1080p | complete encode | `562.244 ms` | `550.879 ms` | `-11.365 ms` (`-2.02%`) | 6/7 |
| padded 1080p | quantization | `323.684 ms` | `320.046 ms` | `-3.638 ms` (`-1.12%`) | 5/7 |
| padded 4K | complete encode | `2149.946 ms` | `2119.598 ms` | `-30.348 ms` (`-1.41%`) | 4/7 |
| padded 4K | quantization | `1221.040 ms` | `1193.319 ms` | `-27.721 ms` (`-2.27%`) | 5/7 |

The exact-coefficient mode remains a compatibility path with a large host tail;
these numbers establish a local improvement, not a reason to optimize that mode
ahead of the default.

A matched Performance Limiters capture compared the exact Phase 4 parent with
the candidate using the same symbolized Release configuration and custom
counter template. The two resident-policy command buffers averaged
`101.030 ms` in the parent and `101.092 ms` in the candidate, a `0.062 ms`
(`0.06%`) difference. Mean occupancy was `61.9%` and `60.8%`, mean instruction
limiters were `63.2%` and `62.2%`, and mean L1 limiters were `10.4%` and
`10.3%`, respectively. This rules out a new occupancy, instruction, or cache
regression and confirms that the resident dispatch graph is unchanged within
measurement noise. Thermal state remained nominal. The final candidate trace
still classified reference preparation as launch-limited in `83.0%` of clean
limiter samples, but resident AQ was now instruction-throughput-limited in
`70.9%`; the retained earlier fusions changed the optimization balance inside
the larger submission.

#### Correctness and pressure qualification

The candidate produced byte-identical codestreams to the exact Phase 4 parent
for all 38 canonical images: 24 Kodak images, six photographic scenes at both
1080p and 4K, and the padded 1080p/4K fixtures. Pinned libjxl `djxl` 0.13.0
decoded padded 4K, Kodak 17, and a photographic 4K representative. Their
external linear-RGB Butteraugli scores were `0.683604`, `0.729423`, and
`0.718311`, respectively.

Efforts 1 through 10, high density, maximum compression, target-size search,
maximum-error control, final-score collection, exact coefficients, throughput,
and maximum throughput all remained byte-identical to the parent. The focused
AQ test compares generic sink scores and block maps with the public complete-map
oracle, compares profiled and unprofiled outputs exactly, and adds a 7-by-5
expanded-source test for the aliased small-image fallback. The existing
`5e-4` block-map gate was not widened; the measured maximum reduction error
remains `2.38419e-07`.

Failure and lease coverage passed allocation, upload, device, completion,
readback, poisoning, forced-purge, and recovery cases. A real pressure run
paused one process between two padded-4K encodes on the same backend, applied
`memory_pressure -p 60`, and then resumed it. RSS fell from `794,576 KiB` to
`517,232 KiB`, while `footprint`-reported reclaimable storage fell from
`1,462 MiB` to `1,248 MiB`. The recovery encode took the expected cold path and
reproduced the exact `1,606,911`-byte codestream.

The complete Release suite passes 61 of 62 tests. The only failure is the
pre-existing pinned CPU `quantization_pipeline` score mismatch
(`0.24919039011001587` actual versus `0.24914586544036865` expected), reproduced
on the parent before attribution.

The retained Phase 5 artifacts are:

- `/private/tmp/gjxl-metal-fusion-phase5-corpus.8EN01s` (corpus, policy,
  decoder, and external-quality validation);
- `/private/tmp/gjxl-metal-fusion-phase5-profile/20260905T004944Z-padded_4k-fully-resident-4a75008a735b`
  (candidate stage profile and Performance Limiters trace);
- `/private/tmp/gjxl-metal-fusion-phase5-control-limiters.B713sy` (matched
  Phase 4 Performance Limiters control); and
- `/private/tmp/gjxl-metal-fusion-phase5-pressure-recovery.json` (same-process
  pressure recovery samples).

## Correctness contract

Fusion is not purely a scheduling transformation. Removing a device-memory
round trip can remove an FP32 rounding point; combining loops can permit
reassociation; and changing a reduction tree necessarily changes evaluation
order. The following gates are mandatory.

### Kernel and metric parity

- Compare every capturable Butteraugli stage against the baseline Metal path.
- Compare full distance maps and scalar scores against the native CPU
  implementation using the existing tolerances.
- Do not widen tolerances to make fusion pass. Investigate the first differing
  expression and restore its operation order.
- Cover constant, impulse, edge, gradient, noise, odd-size, minimum-size,
  expanded-small-image, and multiscale fixtures.

### Encoder behavior

- Compare raw quant fields and quantizer parameters.
- Compare AC strategy grids and color-correlation state.
- Compare final coefficient planes and owned frame state.
- Require deterministic codestream bytes across repeated encodes where the
  baseline is deterministic.
- Decode with the pinned `djxl`, compare decoded pixels, and calculate external
  Butteraugli on a representative corpus.
- Exercise effort 1--10, high density, maximum compression, target-size search,
  maximum-error control, and final diagnostics even though effort 7 is the
  performance target.

### API and failure behavior

- Public diagnostic APIs must still return every promised plane.
- Stage capture must not expose stale scratch after a fused submission.
- Allocation, pipeline-creation, command encoding, completion, and readback
  failures must preserve current invalidation and atomic-output behavior.
- Concurrent encodes sharing a Metal backend must not share mutable
  threadgroup parameters or capture state.

## Measurement protocol

### Wall time

- fresh Release builds with no profiler attached;
- at least two discarded warmups;
- alternating baseline/fused order in independent processes;
- at least five retained pairs for development and seven for promotion;
- identical already-generated linear-RGB inputs and timed boundaries;
- nominal thermal state and no overlapping foreground GPU workload; and
- report median, every paired ratio, output bytes, and output hash.

Primary workloads:

- padded 1080p and padded 4K synthetic photographic workloads for controlled
  scaling;
- all Kodak images for edges, small dimensions, and content variation; and
- at least two real photographic 1080p images and two real photographic 4K
  images.

### GPU attribution

Use stage timestamps to identify the affected family, but do not treat the
instrumented workflow time as the public benchmark. Use Metal System Trace and
Performance Limiters for causal claims. For each candidate record:

| Metric | Why it matters |
| --- | --- |
| target-stage GPU time | proves the intended family improved |
| resident command-buffer GPU time | catches downstream resource regressions |
| reference-preparation GPU time | separates invariant and per-evaluation wins |
| dispatch count and grid sizes | verifies the structural change |
| approximate launched threads | distinguishes launch-volume from call count |
| occupancy and occupancy target | identifies register/threadgroup pressure |
| launch and instruction limiters | separates backpressure from arithmetic |
| buffer bandwidth, L1, LLC, and MMU | checks whether intermediates were removed beneficially |
| complete encode wall time | establishes user-visible value through Amdahl's law |

Counter intervals overlapping unrelated GPU processes must be excluded by the
same conservative rule used in the earlier capture.

## Promotion and rollback criteria

A slice is eligible for the default path only when:

- all focused and full Release tests pass without a new or widened tolerance;
- output determinism and pinned-decoder acceptance pass;
- no tested corpus image has a material decoded-quality regression;
- padded 4K target-stage GPU time improves in a majority of independent pairs;
- complete encoding improves measurably at 4K and does not regress materially
  on Kodak or photographic 1080p; and
- the counter capture explains the result without new spills, pathological
  occupancy collapse, or severe LLC/MMU pressure.

As a practical review threshold, a complex slice should target at least a 10%
resident-AQ improvement or a 5% complete-encode improvement. A smaller change
can be retained when it is simple, clearly measured, and composes with later
fusion. Dispatch reduction alone is not a promotion criterion.

Rollback remains straightforward because each superkernel is introduced and
qualified separately. If a slice fails, remove its pipeline state and host
branch while leaving successful earlier slices intact.

## Work deliberately excluded from this branch

- changing the number of effort-7 AQ evaluations;
- relaxed or fast Metal math;
- overlapping reference preparation with AC on another queue;
- AC strategy search, coefficient tokenization, entropy coding, or codestream
  serialization;
- expanding the persistent-workspace lease; and
- adding a general public execution/materialization-plan API.

Those can change the Amdahl budget, but they answer different questions. In
particular, relaxed math must be evaluated as a quality/correctness policy, not
smuggled into a scheduling optimization.

## Historical first implementation checkpoint

The first checkpoint deliberately contained only:

1. a refreshed current-revision baseline and counter capture;
2. `gjxl_butteraugli_malta_sixway_f32` behind a temporary private build
   selector;
3. exact stage-capture and full-map differential tests;
4. `32 x 8` and `16 x 8` counter-qualified variants; and
5. an alternating Release benchmark on padded 4K, padded 1080p, and Kodak.

That experiment directly tests the central hypothesis: whether keeping the
six-stage accumulation resident within one threadgroup materially reduces
launch pressure and global accumulation traffic without creating a worse
resource bottleneck. Its result determines whether to proceed to broader
channel and convolution fusion or redirect attention to arithmetic itself.
