# Metal Butteraugli fusion investigation

Date: 2026-09-04

- Planning branch: `perf/metal-fusion`
- Baseline commit: `4eda200f8cf62db6f2544f47f504bc4c3b4131cf`
- Primary target: ordinary effort 7, fully resident Metal, multiscale
  Butteraugli, final diagnostic score disabled
- Initial qualification device: Apple M4 Pro

## Executive decision

The next architectural Metal experiment should be a small set of
Butteraugli **superkernels**, not one monolithic kernel and not another
command-buffer or allocation optimization.

The fully resident AQ path already records reconstruction, filtering,
Butteraugli, block reduction, and policy updates into one ordered compute
encoder and waits once. Its remaining Butteraugli cost comes from a deep serial
graph of full-image kernels. The current multiscale implementation records:

- 44 Butteraugli dispatches to prepare the invariant reference;
- about 72 dispatches for each distorted-image comparison; and
- 144 comparison dispatches at ordinary effort 7, whose current policy performs
  two scored evaluations when no final diagnostic evaluation is requested.

The promising fusion work is therefore inside the command buffer: launch fewer
thread grids, keep short-lived values in registers or threadgroup memory, and
avoid writing and rereading full-image intermediates. The recommended order is:

1. fuse all six Malta stages for one scale into one tiled dispatch;
2. fuse independent channel work in psycho-image construction and subsampling;
3. test tile-local separable convolution only where saved global traffic exceeds
   halo duplication and resource pressure;
4. add a resident-only final-distance and reduction path that need not
   materialize a public diagnostic map; and
5. tune each resulting kernel's threadgroup shape from counter evidence.

This is an investigation plan, not a forecast. A working target is to reduce a
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

### Phase 3: tile-local convolution

Prototype the 5-tap blur-plus-Opsin kernel first. Continue through 7, 13, and 15
taps only after each previous radius demonstrates lower GPU time. Treat the
33-tap path as an independent research experiment with explicit memory/halo
accounting. Retain the channel-fused separable implementation whenever it is
faster than the fully tiled version.

### Phase 4: resident sink fusion

After the perceptual kernels stabilize:

1. define a resident-only descriptor containing block-distance and score sinks;
2. implement composed-distance evaluation inside one strategy-family reducer;
3. emit scalar-max partials from the same threadgroups;
4. compare seven family-specialized dispatches with one metadata-driven anchor
   dispatch; and
5. retain the least divergent measured implementation.

The public complete-map path remains the correctness oracle during this phase.

### Phase 5: consolidate

1. Remove the private baseline/fused build selector.
2. Remove superseded pipeline states and scratch planes that are no longer
   required by either production or diagnostic capture.
3. Keep a separate diagnostic implementation only where the API truly promises
   an intermediate plane that the fused graph cannot expose cheaply.
4. Repeat the full corpus, System Trace, failure-injection, and memory-pressure
   qualification.
5. Update `docs/last-mile-optimization.md` with measured results rather than
   design targets.

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

## Recommended first implementation checkpoint

The first checkpoint should contain only:

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
