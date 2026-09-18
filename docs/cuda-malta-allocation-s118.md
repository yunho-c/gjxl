# Malta execution context and allocation (S118)

Date: 2026-09-08. Starting revision: `3a0d1fa`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Question and controls

[S117](cuda-malta-preload-s117.md) found that current-input, original-stride
Malta replay does not reproduce the resident encoder's much slower kernel
durations. Early and late same-size calls differ strongly in the encoder.
Kernel traces already rule out treating the entire difference as event
intervals enclosing host launch gaps. They do not isolate SM execution,
clock behavior, preemption, stalls or memory residency.

Source inspection finds no device/function cache or shared-memory preference
setters in the CUDA backend. S117's early/late kernel records also agree on
registers, static shared memory, block/grid geometry and shared carveout:
48 registers / 64 KiB carveout for full response, 40 / 100 KiB for LF, with
11,520 static shared bytes and 256 threads. These controls do not establish
that every aspect of the execution context is equal.

One concrete difference is allocation. Production defaults to
`CudaBackendOptions::use_stream_ordered_allocation=true`, using a private
`cudaMallocFromPoolAsync` pool when supported, with default cache retention
of min(half device memory, 4 GiB). S117 replay uses `cudaMalloc`. S118 tests
the existing backend option with pooling enabled or disabled. It changes no
production allocator, kernel, system setting or compatibility behavior.

The diagnostic reuses S117's GPU object and S114's qualified libraries. Both
new release/host-ASAN executables preserve all 94 production/prototype GPU
bodies exactly. Each process selects one allocator for its backend; allocator
comparisons therefore cross processes and must not be described as paired
same-process contrasts. Backend allocation state is not switched mid-encode.

Within each process, S117's four labels retain two production and two preload
copies, four warm and twelve measured balanced Williams rows. Both allocators
run in forward and reverse process orders, with the same seed within each
repetition. Every encode checks the retained 4K codestream, baseline summary
and unchanged AC storage width/size. The resident path remains the primary
subject; exact-coefficient mode is not substituted for it.

Uninstrumented timings and CUDA tracing run separately. Traces associate
targeted kernel durations with all 24 call geometries and specializations and
verify actual allocation APIs. An after-encode `cudaMemGetInfo` observation
is outside timing and is not a measurement of peak or physical residency.

## Allocator results

| Repetition | Pooled baseline encode | `cudaMalloc` baseline encode | Pooled / `cudaMalloc` quantization |
|---|---:|---:|---:|
| 0 | 304.10 ms | 401.54 ms | 223.75 / 294.51 ms |
| 1 | 315.65 ms | 398.52 ms | 234.03 / 294.82 ms |

These are medians of the within-row baseline-copy means. Disabling pooling
costs 97.43/82.87 ms, about 32.0/26.3%, across the separate process runs.
This is not a throughput improvement. Preload contrasts remain inconsistent:
whole-call paired medians range from improvements to regressions under either
allocator, with large duplicate-copy variation. S117's preload is not promoted.

The traces confirm the intended allocation paths rather than relying only on
an option value. Each pooled process makes 325 `cudaMallocFromPoolAsync` and
325 `cudaFreeAsync` calls; each non-pooled process makes 325 `cudaMalloc` and
325 `cudaFree` calls. There are 65 encodes per trace, five allocation/free
pairs per encode. Profiled allocation/free API interval sums average about
1.35/0.71 ms per pooled encode versus 26.27/32.87 ms without pooling, including
reference and warm encodes. These averages are not paired measured-row times
and do not account for the entire cross-campaign encode difference.

| Repetition | Pooled Malta kernel sum | `cudaMalloc` Malta kernel sum |
|---|---:|---:|
| 0 | 26.46 ms | 25.15 ms |
| 1 | 27.85 ms | 25.87 ms |

Ordinary allocation lowers these traced baseline sums about 4.9/7.1%, but
both remain far above S117's constructed replay sum of about 9.8 ms. More
importantly, both retain the early/late slowdown: full-response call 0 is
about 0.93–1.10 ms, whereas same-size call 12 is 2.69–2.89 ms. Removing the
pool does not remove that phenomenon. Pooling is not established as its cause.

The after-encode observations show approximately 3.02 GB free with pooling
versus 5.37 GB without in the first timing repetition. This is consistent
with retained allocation state, but says nothing definitive about peak
resident working set, physical-page placement, paging or device eviction.

## Counter comparison

Nsight Compute captures production full-response call 0 and call 12 under
pooling, ordinary allocation and the corresponding current-input replay,
in two opposite orders: twelve reports, one matching kernel and four replay
passes each. Kernel-name filtering restricts matching to paired full-response
32x64 kernels. Skipping four matching launches selects call 12, after calls
0/1 and half-scale 6/7. The installed tool's help explicitly confirms that
`--launch-skip` counts only matching kernels. Clocks/cache controls are `none`,
and profiler warnings are retained. Counter and ordinary timing campaigns
do not overlap.

Every one of the twelve reports has exactly 71,963,205 executed warp
instructions and 139,303,116 predicated-on FFMA thread instructions. Shared
stores are also identical at 367,200 wavefronts. Shared-load wavefronts vary
by less than 0.003%; DRAM traffic stays around 66.4 MB read and 33.2 MB written.
L2 traffic is about 150–151 MB. These captures do not support an explanation
based on several times more arithmetic or data traffic in the later call.

| Counter context | Reported SM cycle rate | Profiled duration |
|---|---:|---:|
| Pool, early | 0.984–1.067 GHz | 0.758–0.821 ms |
| Pool, late | 0.968–0.997 GHz | 0.817–0.836 ms |
| `cudaMalloc`, early | 0.930–0.954 GHz | 0.847–0.867 ms |
| `cudaMalloc`, late | 0.821–0.852 GHz | 0.952–0.985 ms |
| Replay, either call | 1.271–1.284 GHz | 0.638–0.641 ms |

Reported DRAM cycle rates remain about 5.495 GHz throughout. The product of
reported duration and SM cycle rate is approximately 0.806–0.819 million
cycles across all twelve captures. This is a derived consistency check from
profiler metrics, not a new direct clock measurement in unprofiled encoding.

The counter collection materially changes the phenomenon: late pooled calls
are no longer around 2.8 ms, and early/late differences nearly disappear.
Clock-rate differences are present in the counter captures, but those captures
cannot establish that clocks explain the full unprofiled slowdown. Attributing
the missing time to power throttling, cache effects or preemption now would
overstate the evidence.

## Decision and next test

Keep production pooling and kernels unchanged. Neither disabling pooling nor
promoting the preload prototype is justified by this study. The allocation
control narrows the search: the pool is not necessary for the early/late
slowdown, and the sampled kernels do not execute extra instructions or traffic.

The next bounded experiment should observe device cycle rate close to early
and late launches in the normal stream, with lightweight probes and duplicate
controls to quantify perturbation. Compare against replay without locking
clocks or changing power/security settings. This is a measurement problem
before another kernel rewrite; the current evidence still does not identify
the cause of the unprofiled execution-context difference.

## Qualification and evidence

Release and scoped host-ASAN preflights pass twenty exact encodes, ten under
ASAN. Four uninstrumented jobs add 260 checks, four system traces add 260, and
eight in-encode counter jobs add forty: 580 exact encode checks in total.
The four replay counter jobs also check their six variant bursts against the
scalar GPU reference. System traces contain 6,240 targeted kernels, with
labels, call counts, geometries and specializations verified. No GPU code was
changed or newly sanitized; the reused GPU object retains S117's differential
and sanitizer qualification, and production retains S114's full qualification.

Evidence is retained under `U:/gjxl-cuda-diagnostics/s118`, including binaries,
native dumps, logs, system/counter reports, source snapshots, `inputs.json`,
`within_analysis.json`, `trace_analysis.json`, `allocator_analysis.json`,
`allocation_api_times.json`, `counters_analysis.json` and artifact hashes.
The validator checks all 41 job statuses/log hashes, four timed-job isolation
windows, source/input/library identities, native equivalence, trace and encode
counts, counter selections and all forty retained runtime hashes. All recorded
jobs are terminal. No admin, firewall or permission blocker was observed and
no system/security settings were changed. User scratch files remain untouched.
