# CUDA optimization study S1

- Status: profiling complete; S1.1-S1.4 implemented
- Profile revision: `a474937`
- Profile date: 2026-09-04
- Build: Release, CUDA 11.8, `CMAKE_CUDA_ARCHITECTURES=86`
- Device: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6,
  6 GiB device memory
- Driver: 577.00
- Related analysis: [CUDA backend support analysis](cuda-support.md)

## Executive finding

The current CUDA backend has two different performance profiles.

- Fully-resident encoding is compute-bound. Forward and inverse DCT kernels
  consume `133.7 ms`, or about 67% of measured GPU kernel time, in the warmed
  padded-1080p trace.
- Maximum-throughput encoding is not GPU-compute-bound. It executes only
  `10.1 ms` of kernels at 1080p. Its remaining quantization time is dominated
  by host preparation and validation, 27 device allocations and frees,
  synchronous transfer boundaries, readback, and host frame assembly.

The highest-confidence next optimization is to consolidate per-image device
allocations into a small number of arenas. It applies to both encoding modes,
has a measured CUDA API ceiling of `23.9 ms` in maximum-throughput and
`73.9 ms` in fully-resident encoding, reduces fragmentation and batch memory
pressure, and establishes stable pointers for later CUDA Graph capture.

After allocation consolidation, the next work should remove encoding-only
readbacks and host materialization. The main compute optimization should then
specialize the 32x32 and 16x32/32x16 DCT kernels. Moving linear-RGB-to-Opsin
preparation to CUDA has a larger architectural scope but becomes especially
valuable at 4K.

Adding more production streams is not currently justified. The two-lane pool
improves throughput, while the fully-resident path already shows contention
for shared GPU compute resources.

## Scope and method

The primary boundary was the public in-memory workflow from a caller-owned
linear RGB image through a completed codestream. Input generation, file I/O,
backend construction, and optional final-score diagnostics were excluded.

The measurements used:

- `gjxl_cuda_encoding_benchmark` for public-workflow wall time by stage;
- `gjxl_image_batch_benchmark` for paired serial and batch throughput;
- Nsight Systems 2023.2.3 for CUDA API, kernel, launch-geometry, and memory
  operation traces; and
- Nsight Compute 2022.3 as an attempted source of occupancy and hardware
  counter data.

Each latency workload was warmed before measurement. The principal 1080p
comparison used two warmups and seven samples. The 4K comparison used one
warmup and three samples. Nsight traces captured one warmed 1080p sample per
policy inside the benchmark's `cudaProfilerStart`/`cudaProfilerStop` range.
Profiler start time is excluded from every CUDA API total below.

The workload is synthetic and has odd source dimensions (`1919x1079` and
`3839x2159`) so that the normal padding path remains covered. Kernel geometry
and fixed per-pixel work are representative; entropy size and host
serialization behavior can vary for natural images.

This laptop changes clocks and power state aggressively. Absolute batch times
varied even after warmup, so the alternating paired speedup is more useful
than comparing isolated serial and batch medians. One abandoned batch-8 sweep
continued running after the command runner yielded and thermally contaminated
intermediate observations. That process was stopped, the GPU was returned to
an idle state, and none of the contaminated batch numbers are reported here.

## Public-workflow wall profile

### Padded 1080p

Warmed median times in milliseconds:

| AQ mode | Total | Input preparation | Quantization pipeline | Codestream encoding |
|---|---:|---:|---:|---:|
| Exact coefficients | 2756.6 | 1035.9 | 1661.3 | 69.1 |
| Fully resident | 430.9 | 36.8 | 317.9 | 71.2 |
| Throughput | 441.3 | 41.1 | 325.1 | 78.4 |
| Maximum throughput | 260.3 | 42.5 | 132.5 | 69.9 |

The resident modes remove most of the CPU-compatible preparation performed by
the exact-coefficient path. `throughput` is intentionally equivalent to
fully-resident inside the encoding workflow, so its small difference is run
variance rather than a separate optimization opportunity.

Quantization accounts for about 74% of fully-resident wall time and 51% of
maximum-throughput wall time. In maximum-throughput mode, input preparation
and CPU codestream generation are already material Amdahl limits.

### Padded 4K

Warmed median times in milliseconds:

| AQ mode | Total | Input preparation | Quantization pipeline | Codestream encoding |
|---|---:|---:|---:|---:|
| Fully resident | 2405.3 | 237.8 | 1718.3 | 442.9 |
| Maximum throughput | 1044.6 | 226.6 | 496.0 | 252.4 |

Fully-resident quantization remains dominant at 4K, at about 71% of total
time. Maximum-throughput quantization falls to about 47%; input preparation
and serialization together account for nearly another half of total wall
time. Optimizing only CUDA kernels therefore cannot produce a large multiple
of end-to-end maximum-throughput performance.

## CUDA timeline profile

### Fully resident

The warmed padded-1080p trace contains:

| Item | Measured value |
|---|---:|
| Kernel launches | 497 |
| GPU kernel execution | 200.0 ms |
| CUDA event synchronizations | 5 |
| Device allocations / frees | 27 / 27 |
| Allocation and free API time | 73.9 ms |
| Kernel launch API time | 12.7 ms |
| Memory operations | 60 copies, 3 memsets |
| Host-to-device payload | 54.6 MB |
| Device-to-host payload | 34.6 MB |
| Device copy execution | 13.7 ms |
| Blocking copy API time | 41.8 ms |

CUDA API time and GPU execution time are not additive. In particular,
`cudaEventSynchronize` is the host waiting for already-counted GPU work, and a
synchronous free can expose earlier outstanding work. The evaluator waits for
its submissions before destruction, however, so the repeated post-completion
allocation/free cost remains a meaningful optimization target.

The kernel-time distribution is:

| Kernel group | Time | Share of kernel time |
|---|---:|---:|
| Forward DCT | 68.1 ms | 34.1% |
| Inverse DCT | 65.6 ms | 32.8% |
| Malta response | 9.0 ms | 4.5% |
| Residual | 7.2 ms | 3.6% |
| Adjusted-quant selection | 5.7 ms | 2.9% |
| Four convolution variants | 12.1 ms | 6.1% |
| EPF | 4.9 ms | 2.4% |
| Initial CfL | 4.5 ms | 2.3% |
| All other kernels | 22.9 ms | 11.4% |

The generic DCT kernels use one 256-thread CUDA block per transform, 40
registers per thread, and one shared-memory intermediate plane. The largest
shape groups were:

| Transform group | Combined forward/inverse time |
|---|---:|
| 32x32 | approximately 66.1 ms |
| 16x32 and 32x16 | approximately 38.8 ms |
| Remaining shapes | approximately 28.8 ms |

The first two groups therefore account for approximately `105 ms`, 79% of DCT
time and 52% of all measured kernel time. They are the correct first targets
for shape-specific DCT work.

### Maximum throughput

The corresponding maximum-throughput trace contains:

| Item | Measured value |
|---|---:|
| Kernel launches | 32 |
| GPU kernel execution | 10.1 ms |
| CUDA event synchronizations | 2 |
| Device allocations / frees | 27 / 27 |
| Allocation and free API time | 23.9 ms |
| Kernel launch API time | 0.8 ms |
| Memory operations | 16 copies, 1 device copy, 2 memsets |
| Host-to-device payload | 24.9 MB |
| Device-to-host payload | 34.0 MB |
| Device copy execution | 10.5 ms |
| Blocking copy API time | 16.4 ms |

Only one kernel is individually large:

| Kernel | Time | Share of kernel time |
|---|---:|---:|
| Initial CfL | 6.2 ms | 61.1% |
| Forward DCT8 | 1.5 ms | 15.1% |
| Inverse Gaborish convolution | 0.7 ms | 6.9% |
| Quant adjustment | 0.7 ms | 6.4% |
| All other kernels | 1.0 ms | 10.5% |

`InitialCflKernel` launches only two 256-thread blocks for 510 color tiles.
Each active thread traverses one complete 64x64 tile twice and accumulates its
statistics serially. This explains why it dominates an otherwise short GPU
sequence.

Maximum-throughput also performs substantial host work around the CUDA calls:

- finite-value scans over full input images;
- copies of coding Opsin planes into contiguous host vectors;
- construction of one transform-layout entry per DCT8 block;
- allocation and initialization of compatibility output maps;
- validation and copying of downloaded quantization maps; and
- conversion of downloaded raw coefficient arrays into the final encoder
  frame.

The required final AC readback is 24.9 MB at 1080p. It cannot be eliminated
without moving codestream construction to the GPU, but it can write directly
into final frame storage and share one packed transfer with the other frame
metadata.

## Batch throughput

The clean 1080p paired measurements were:

| AQ mode | Batch | Batch throughput | Median paired speedup |
|---|---:|---:|---:|
| Maximum throughput | 1 | 2.58 images/s | 1.02x |
| Maximum throughput | 2 | 2.40 images/s | 1.19x |
| Maximum throughput | 4 | 4.44 images/s | 1.70x |
| Fully resident | 1 | 1.68 images/s | 0.98x |
| Fully resident | 2 | 1.85 images/s | 1.20x |
| Fully resident | 4 | 1.77 images/s | 1.45x |

The absolute values retain laptop power-state variance. The important signal
is that fully-resident scaling saturates earlier than maximum-throughput
scaling because its two streams compete for SM execution. The current two-lane
production pool should remain bounded at two. Allocation consolidation and
lower per-image memory pressure should be measured before reconsidering that
cap.

## Ranked optimization plan

The ceilings below are measured portions of the current trace, not promised
speedups. They overlap and cannot be summed. Every retained change needs a new
wall profile because removing one synchronization point can move time into a
later wait.

Each completed checkpoint should append a performance snapshot to its section.
The snapshot should identify the commit and workload, compare the same wall
profile before and after the change, report relevant CUDA timeline counters,
and call out noise or regressions as explicitly as improvements. Build and
test qualification belongs in the same snapshot when it materially defines
the result.

### S1.1: consolidate and reuse device allocations

**Priority:** highest; applies to both resident policies.

The prepared AC-strategy search owns three device buffers for each of seven
strategy stages, plus three shared scratch buffers. The maximum-throughput
prepared evaluator similarly allocates each image plane and working array as a
separate `DeviceBuffer`. At the public workflow boundary these objects are
created and destroyed for every image. The relevant allocation fan-out is in
`src/gpu/ops/ac_strategy_search.cpp` and `src/gpu/cuda/cuda_aq.cpp`.

Recommended implementation:

1. Plan candidates, matrices, costs, and common AC-search scratch into one or
   two `DeviceScratchArena` allocations.
2. Convert the maximum-throughput evaluator to persistent and staging arenas,
   matching the resident evaluator's ownership model.
3. Preserve arena capacity across compatible target-size attempts.
4. Consider a backend size-bucketed cache or `cudaMallocAsync` pool only after
   object-local arenas have removed the known allocation fan-out.

Object-local arenas are the preferable first step because they work with the
current CUDA 11.8 baseline, keep ownership explicit, preserve deterministic
failure behavior, and create stable pointers for later graph capture.

Measured reducible API ceiling:

- maximum throughput: `23.9 ms`, about 9% of the 1080p public wall time;
- fully resident: `73.9 ms`, about 17% of the 1080p public wall time.

The acceptance gate should require identical frames and codestreams for every
mode that promises identity, unchanged injected-failure behavior, zero
steady-state allocation for reused prepared operations, and a materially lower
allocation count for fresh public encodes.

### S1.2: remove encoding-only materialization and readbacks

**Priority:** high; lower risk than a kernel rewrite.

The encoding path currently downloads initial results that are not serialized:

- an 8.29 MB pixel mask;
- a 129.6 KB quant field; and
- a 129.6 KB strategy mask.

Maximum-throughput uses those values only to satisfy the complete public
pipeline adapter before continuing with resident frame encoding.
Fully-resident AC search already has resident Opsin, quant-field, and pixel-mask
views, but its common provider interface still requires valid host maps. The
readback implementations are in `src/gpu/cuda/cuda_aq.cpp` and
`src/gpu/cuda/cuda_aq_resident.cpp`.

Recommended implementation:

1. Add an internal encoding-only initial-quantization contract. Keep the
   complete public AQ and pipeline contracts unchanged.
2. Let resident AC search validate and consume device views without requiring
   populated host quant and pixel maps.
3. Keep numeric error checking on the device and download only the scalar
   error/quantizer values required by host control flow.
4. Allocate the final host frame arrays before readback and copy coefficients
   directly into their final destinations.
5. Pack final raw-quant, CfL, DC, AC, and error data where practical so that
   one ordered transfer replaces several synchronous copies.
6. Batch immutable metadata and small constant uploads into one staging
   payload during preparation.

The current copy API ceiling is `16.4 ms` for maximum-throughput and `41.8 ms`
for fully-resident encoding. Actual device transfer occupies `10.5 ms` and
`13.7 ms`, respectively, so eliminating host synchronization and copying is
more important than attempting to tune raw PCIe bandwidth alone.

Pinned reusable staging can improve the remaining large transfers and allow
them to overlap across the two production lanes. It should not weaken the
backend's synchronous host-buffer lifetime contract.

#### S1.2 completion snapshot (2026-09-04)

Commit `f2732d3` keeps initial quantization, masking, and initial CfL data on
the device for encoding-only operation. Resident AC search consumes those
device views directly, and fully-resident Butteraugli control reduces and
adjusts the initial field without a host field handoff. Small immutable
uploads and related readbacks share one stream synchronization per batch.
The complete public diagnostic APIs continue to materialize their maps.

The wall comparison uses the same synthetic `1919x1079` padded-1080p workload,
distance `1.2`, effort `7`, two warmups, seven GPU-only samples, and an RTX
3060 Laptop GPU. Times are warmed medians in milliseconds:

| AQ mode and stage | Before S1.2 | After S1.2 | Change |
|---|---:|---:|---:|
| Maximum throughput, total | 260.3 | 191.0 | -26.6% |
| Maximum throughput, quantization | 132.5 | 86.5 | -34.7% |
| Fully resident, total | 430.9 | 435.2 | +1.0% |
| Fully resident, quantization | 317.9 | 303.1 | -4.7% |

The post-change sample ranges were `178.8-270.3 ms` total and
`80.2-118.2 ms` quantization for maximum throughput, and `394.9-461.0 ms`
total and `297.2-317.7 ms` quantization for fully resident. The laptop's clock
and thermal variance make the fully-resident total effectively unchanged;
the smaller quantization-stage median is directionally consistent with the
copy reduction but should not be treated as a stable 4.7% end-to-end gain.

Warmed Nsight Systems captures confirm the intended traffic reduction:

| AQ mode and transfer | Before S1.2 | After S1.2 | Change |
|---|---:|---:|---:|
| Maximum throughput, DtoH | 34.0 MB / 11 copies | 25.4 MB / 8 copies | -25.2% bytes |
| Fully resident, DtoH | 34.6 MB / 23 copies | 25.9 MB / 18 copies | -25.1% bytes |
| Fully resident, HtoD | 54.6 MB / 37 copies | 54.2 MB / 34 copies | -0.7% bytes |

Maximum throughput benefits directly because the removed 8.55 MB initial-map
readback was a large share of its short GPU path. Fully resident removes the
same maps plus three host quant-field handoffs, but generic DCT execution still
dominates its wall time; this reinforces S1.3 as its next high-impact target.
The checkpoint passed all 53 tests in the CUDA build and all 47 tests in the
CPU-only build. The ignored trace artifacts are
`s12_maximum_throughput_1080p.nsys-rep` and
`s12_fully_resident_1080p.nsys-rep` under `build-cuda-ninja/profiles`.

### S1.3: specialize the dominant DCT shapes

**Priority:** highest compute optimization for fully-resident encoding.

The current transform kernel performs a generic separable matrix multiply for
all supported shapes. A shape-specific path should start with 32x32, then
16x32/32x16. The current implementation is in
`src/gpu/cuda/cuda_kernels.cu`. Useful candidates include:

- tiled shared-memory row and column passes;
- warp-specialized fixed-size transforms;
- factored radix-2 transforms when they preserve the accepted result; and
- shape-specific block sizes instead of reserving 256 threads for every
  transform.

A 2x improvement over the two dominant shape groups would save approximately
`52 ms` at 1080p. A 2x improvement over all DCT work would save approximately
`67 ms`, about 16% of quantization time and 15% of total fully-resident wall
time in this profile.

Fast math and tensor-core arithmetic should remain disabled until a separate
decision-sensitivity and output-quality contract explicitly permits them.

Before selecting a kernel design, enable NVIDIA GPU performance counters and
capture achieved occupancy, eligible warps, issue stalls, shared-memory bank
conflicts, L1/L2 traffic, and arithmetic throughput separately for the 32x32
and rectangular groups. Nsight Compute currently reports `ERR_NVGPUCTRPERM`,
so those counters were unavailable for this study.

#### S1.3 completion snapshot (2026-09-04)

This checkpoint adds compile-time CUDA paths for 32x32, 16x32, and 32x16
transforms. The arithmetic, basis values, and accumulation order remain the
same as the generic separable transform. The specialized kernels preload the
horizontal basis from a coalesced global-memory copy into a shared-memory tile
with one padding column. This replaces the generic horizontal pass's
warp-divergent constant-memory accesses and prevents shared-memory bank
conflicts. The vertical basis continues to use constant-memory broadcasts.
Fast math, factored transforms, and tensor-core arithmetic remain disabled.

Launch metadata also corrected an ambiguity in the initial profile. A 2,048
byte intermediate identifies only a 512-coefficient transform; it does not
identify its dimensions. The production call-site metadata shows that those
dominant launches are 16x32 and 32x16, not 8x64 and 64x8. The earlier labels
in this document have been corrected accordingly.

Nsight Compute 2022.3 was retried on the 32x32 forward transform, but the
driver again returned `ERR_NVGPUCTRPERM`. Kernel selection therefore used
Nsight Systems launch metadata and matched before/after timing rather than
hardware counters. In one warmed padded-1080p trace, combined forward and
inverse times were:

| Transform scope | After S1.2 | After S1.3 | Change |
|---|---:|---:|---:|
| Primary 32x32 batch | 53.0 ms | 4.0 ms | -92.5% |
| Primary 16x32 batch | 12.4 ms | 1.9 ms | -85.1% |
| Primary 32x16 batch | 40.3 ms | 2.0 ms | -94.9% |
| All occurrences of the three shapes | 138.9 ms | 10.4 ms | -92.5% |
| All DCT kernels | 176.4 ms | 39.1 ms | -77.8% |

The public wall comparison uses the same synthetic `1919x1079` workload,
distance `1.2`, effort `7`, two warmups, seven GPU-only samples, and RTX 3060
Laptop GPU as the S1.2 snapshot. Times are medians in milliseconds:

| AQ mode and stage | After S1.2 | After S1.3 | Change |
|---|---:|---:|---:|
| Fully resident, total | 435.2 | 317.7 | -27.0% |
| Fully resident, quantization | 303.1 | 208.1 | -31.3% |
| Maximum throughput, total | 191.0 | 186.5 | -2.4% |
| Maximum throughput, quantization | 86.5 | 78.8 | -8.9% |

The post-S1.3 fully-resident ranges were `296.9-367.0 ms` total and
`200.9-215.4 ms` quantization. Maximum-throughput ranges were
`173.7-256.6 ms` total and `77.3-99.1 ms` quantization. Maximum-throughput is
a control here: its DCT8-only encoding path does not dispatch any of the new
kernels, so its difference should be treated as clock and host variance.

The odd `3839x2159` 4K checkpoint measured `1161.3 ms` total and `766.5 ms`
quantization for fully resident, and `616.1 ms` total and `252.1 ms`
quantization for maximum throughput. Separate `1919x1079` and `3839x2159`
fully-resident CLI encodes were decoded successfully by the pinned `djxl`,
which reported the original dimensions. Paired 1080p batch qualification
produced these medians:

| AQ mode | Batch | Batch ms | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 354.9 | 2.82 | 0.95x |
| Fully resident | 2 | 557.7 | 3.59 | 1.19x |
| Fully resident | 4 | 974.6 | 4.10 | 1.51x |
| Maximum throughput | 1 | 230.5 | 4.34 | 1.00x |
| Maximum throughput | 2 | 307.9 | 6.50 | 1.42x |
| Maximum throughput | 4 | 737.8 | 5.42 | 1.29x |

The batch run began at 63 C, P8, and a 210 MHz SM clock and ended at 68 C,
P5, and 892 MHz, reinforcing that paired speedups are more reliable than its
absolute times. The checkpoint passed all 53 tests in the CUDA build and all
47 tests in the CPU-only build. CUDA coverage includes per-shape comparison
with the double-precision DCT reference, exact-mode CPU/CUDA codestream
identity, and iteration-zero fully-resident codestream identity. The ignored
trace artifacts are `s13_fully_resident_1080p.nsys-rep` and its SQLite export
under `build-cuda-ninja/profiles`.

### S1.4: move input preparation to CUDA

**Priority:** high end-to-end potential, broader architectural change.

`PrepareWorkflow` currently performs `LinearRgbToPaddedOpsin` before backend
selection. This costs about `43 ms` at 1080p and `227-238 ms` at 4K in the
measured GPU workflows. Fully-resident preparation then uploads both the
original linear image and CPU-produced Opsin image, accounting for six large
plane uploads. The current workflow boundary is in
`src/codestream/workflow.cpp`.

A CUDA-native frontend should:

1. select or resolve the forced CUDA backend before materializing Opsin;
2. upload the three linear RGB planes once;
3. perform padding and linear-RGB-to-Opsin conversion on the assigned lane;
4. expose the resident Opsin views to initial AQ and AC strategy search; and
5. compute any required quantization-matrix scale statistics without forcing a
   complete Opsin image back to the host.

For maximum-throughput this moves a large CPU stage to an otherwise lightly
loaded GPU. For fully-resident it also removes roughly half of the initial
full-image H2D payload. The change touches workflow preparation, provenance,
validation, and matrix-scale statistics, so it should follow the more local
arena and readback work.

#### S1.4 completion snapshot (2026-09-04)

Forced CUDA workflows now resolve the backend before host Opsin preparation,
upload the three source RGB planes once, and perform edge padding plus the
linear-RGB-to-Opsin transform on the selected CUDA lane. The prepared source
and Opsin views remain resident through initial quantization, AC strategy
search, and AQ evaluation. Quantization-matrix scale selection downloads only
three scalar maxima and the device error word. CPU, Metal, and CUDA exact-
coefficient workflows retain the established host preparation path.

The wall comparison is paired against revision `7d846bb` on the same RTX 3060
Laptop GPU. It uses the same synthetic odd-sized workloads, distance `1.2`,
effort `7`, and GPU-only measurement boundary as the earlier snapshots. The
1080p runs used two warmups and seven samples; 4K used one warmup and three
samples. Times are warmed medians in milliseconds:

| Workload and stage | Before S1.4 | After S1.4 | Change |
|---|---:|---:|---:|
| 1080p fully resident, total | 377.7 | 294.3 | -22.1% |
| 1080p fully resident, input preparation | 36.8 | 7.7 | -79.1% |
| 1080p fully resident, quantization | 254.4 | 215.7 | -15.2% |
| 1080p maximum throughput, total | 183.6 | 106.9 | -41.8% |
| 1080p maximum throughput, input preparation | 40.0 | 8.2 | -79.6% |
| 1080p maximum throughput, quantization | 79.8 | 33.3 | -58.3% |
| 4K fully resident, total | 1469.9 | 1177.2 | -19.9% |
| 4K fully resident, input preparation | 157.8 | 30.3 | -80.8% |
| 4K fully resident, quantization | 1110.0 | 950.9 | -14.3% |
| 4K maximum throughput, total | 629.9 | 303.6 | -51.8% |
| 4K maximum throughput, input preparation | 175.4 | 30.8 | -82.4% |
| 4K maximum throughput, quantization | 282.2 | 99.2 | -64.8% |

The post-change 1080p total ranges were `283.8-312.8 ms` for fully resident
and `96.3-123.9 ms` for maximum throughput. The 4K ranges were
`1150.2-1208.1 ms` and `293.6-314.9 ms`, respectively. These paired runs are
the appropriate S1.4 comparison; their absolute values should not be compared
directly with the earlier S1.3 snapshot because laptop clocks and thermals
varied between sessions.

Warmed 1080p Nsight Systems captures show the intended transfer change:

| AQ mode and counter | Before S1.4 | After S1.4 |
|---|---:|---:|
| Fully resident, HtoD | 54.220 MB / 34 copies | 29.336 MB / 31 copies |
| Fully resident, DtoH | 25.924 MB / 18 copies | 25.924 MB / 19 copies |
| Fully resident, kernel launches | 499 | 501 |
| Fully resident, allocations / frees | 4 / 4 | 5 / 5 |
| Maximum throughput, HtoD | 24.917 MB / 5 copies | 24.881 MB / 5 copies |
| Maximum throughput, DtoH | 25.403 MB / 8 copies | 25.403 MB / 9 copies |
| Maximum throughput, kernel launches | 32 | 34 |
| Maximum throughput, allocations / frees | 2 / 2 | 3 / 3 |

The extra allocation is the single input arena, replacing evaluator-owned
source/coding planes. The extra DtoH operation is a 16-byte statistics/error
result. At 1080p the new conversion and statistics kernels took `0.198 ms` and
`0.171 ms` in fully-resident mode, and `0.199 ms` and `0.193 ms` in maximum-
throughput mode. Fully resident therefore removes 45.9% of initial HtoD bytes;
maximum throughput replaces the former Opsin upload with a same-sized RGB
upload, but still removes the host transform and duplicate evaluator work.

Post-change paired 1080p batch qualification produced:

| AQ mode | Batch | Batch ms | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 305.9 | 3.27 | 1.03x |
| Fully resident | 2 | 571.7 | 3.50 | 1.30x |
| Fully resident | 4 | 1080.0 | 3.70 | 1.24x |
| Maximum throughput | 1 | 120.3 | 8.31 | 1.07x |
| Maximum throughput | 2 | 172.1 | 11.62 | 1.53x |
| Maximum throughput | 4 | 287.6 | 13.91 | 2.14x |

The fully-resident batch-4 absolute result is about 10% slower than the older
S1.3 session despite the paired single-image improvement, consistent with
power-state variance and increased contention when preparation moves onto the
GPU. It should be watched in the next checkpoint rather than attributed to a
stable regression from these non-paired sessions.

The CUDA input test compares the uploaded source, padded Opsin result, and all
three matrix statistics exactly with the CPU oracle on a `257x17` source
padded to `264x24`. Baseline/current codestreams were byte-identical for the
sample image in both resident policies and for the existing odd 1080p/4K
qualification outputs. The checkpoint passed all 53 CUDA tests and all 47
CPU-only tests. The ignored traces are `s14_fully_resident_1080p.nsys-rep`,
`s14_maximum_throughput_1080p.nsys-rep`, and their paired `s14_baseline_*`
captures under `build-cuda-ninja/profiles`.

### S1.5: parallelize initial CfL

**Priority:** useful but bounded.

Assign a cooperative CUDA block or warp group to each 64x64 color tile and
reduce the means, quadratic term, and two linear terms in parallel. A retained
implementation must define its floating reduction order and demonstrate that
the resulting quantized CfL maps remain within the policy's output contract.
The serial-per-thread implementation is in
`src/gpu/cuda/cuda_aq_kernels.cu`.

The entire current ceiling is only `4.5-6.2 ms` at 1080p, approximately 1% of
fully-resident wall time and 2% of maximum-throughput wall time. This should
not displace allocation, transfer, or DCT work.

### S1.6: capture stable resident submissions with CUDA Graphs

**Priority:** follow-on optimization after arenas.

Fully-resident encoding launches 497 kernels and spends `12.7 ms` in launch
APIs. Once arena packing guarantees stable pointers and the execution plans
have fixed geometry, graph capture can reduce host launch work and repeated
submission setup. Dynamic strategy batches and control-mode differences may
require a small graph cache keyed by geometry and policy shape.

Maximum-throughput has only 32 launches and `0.8 ms` of launch API time, so it
does not justify a graph-specific implementation by itself.

## Work that should not lead the next cycle

### More execution lanes

The current two-lane pool is sufficient. Four in-flight requests improve
overlap of CPU preparation, GPU work, and serialization, but fully-resident
GPU throughput no longer scales with the request count. More streams would
increase simultaneous resident memory without creating additional SM
capacity.

### Exact-coefficient optimization

Exact mode preserves the CPU-compatible coefficient decision path and spends
about 2.7 seconds at 1080p. It is valuable as a compatibility and differential
oracle, not as the first throughput target.

### GPU entropy coding

CPU codestream generation is already visible at about `70 ms` at 1080p and
`252-443 ms` at 4K. Moving entropy coding to CUDA could eventually raise the
end-to-end ceiling, but it is a much larger project than removing current
allocation and handoff overhead. Batch workers already overlap serialization
for one image with GPU work for another.

### Tensor cores or global fast math

The DCT, quantization, and selection paths are decision-sensitive. Throughput
gains that change threshold decisions need an explicit mode and quality
contract; they should not silently alter fully-resident behavior.

## Suggested implementation checkpoints

1. **Arena checkpoint:** reduce fresh-encode allocation count and re-run the
   1080p Nsight API summary.
2. **Readback checkpoint:** account for every transfer by semantic payload and
   prove that encoding-only output omits diagnostic maps.
3. **DCT checkpoint:** enable counters, specialize 32x32, then 16x32/32x16, with
   per-shape differential tests and public-workflow timing.
4. **Frontend checkpoint:** add resident CUDA Opsin preparation and compare
   both 1080p and 4K wall profiles.
5. **Small-kernel checkpoint:** parallelize initial CfL and evaluate graph
   capture only after the preceding pointer and dataflow changes settle.

Each checkpoint should run:

- CPU-only configuration and tests;
- the complete CUDA functional suite;
- exact-coefficient CPU/CUDA codestream comparison;
- repeated fully-resident and maximum-throughput concurrency stress;
- odd padded 1080p and 4K encode/decode qualification; and
- paired batch sizes 1, 2, and 4, with GPU temperature and clock state noted.

## Reproduction commands

Public wall profiles:

```powershell
.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_1080p --gpu-aq fully-resident `
  --warmups 2 --samples 7 --gpu-only

.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_1080p --gpu-aq maximum-throughput `
  --warmups 2 --samples 7 --gpu-only

.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_4k --gpu-aq fully-resident `
  --warmups 1 --samples 3 --gpu-only

.\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_4k --gpu-aq maximum-throughput `
  --warmups 1 --samples 3 --gpu-only
```

Batch profiles:

```powershell
.\build-cuda-ninja\gjxl_image_batch_benchmark.exe `
  --workload 1080p --batch-sizes 1,2,4 `
  --gpu-aq maximum-throughput --backend cuda `
  --warmups 1 --samples 5

.\build-cuda-ninja\gjxl_image_batch_benchmark.exe `
  --workload 1080p --batch-sizes 1,2,4 `
  --gpu-aq fully-resident --backend cuda `
  --warmups 1 --samples 3
```

Nsight Systems capture, using the installed 2023.2.3 executable path:

```powershell
$gjxlNsysRoot =
  'C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.2.3'
$gjxlNsys = Join-Path $gjxlNsysRoot 'target-windows-x64\nsys.exe'
& $gjxlNsys `
  profile --trace=cuda --sample=none --cpuctxsw=none `
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown `
  --force-overwrite=true `
  --output=build-cuda-ninja\profiles\cuda_fully_resident_1080p `
  .\build-cuda-ninja\gjxl_cuda_encoding_benchmark.exe `
  --workload padded_1080p --gpu-aq fully-resident `
  --warmups 2 --samples 1 --gpu-only --profile-range
```

The local study artifacts are:

- `build-cuda-ninja/profiles/issue6_fully_resident_1080p.nsys-rep`;
- `build-cuda-ninja/profiles/issue6_fully_resident_1080p.sqlite`;
- `build-cuda-ninja/profiles/issue6_maximum_throughput_1080p.nsys-rep`; and
- `build-cuda-ninja/profiles/issue6_maximum_throughput_1080p.sqlite`.

They live under the ignored build directory and are intentionally not source
artifacts.
