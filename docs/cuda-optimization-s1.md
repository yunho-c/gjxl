# CUDA optimization study S1

- Status: S1.1-S1.5, packed DCT, tiled Malta, and specialized/tiled blurs
  implemented; optimization ongoing
- Profile revision: `a474937`
- Profile date: 2026-09-04
- Build: Release, CUDA 11.8, `CMAKE_CUDA_ARCHITECTURES=86`
- Device: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6,
  6 GiB device memory
- Driver: 577.00
- Related analysis: [CUDA backend support analysis](cuda-support.md)

## Executive finding

The opening measurements describe revision `a474937`; the completion snapshots
below supersede them. The latest convolution checkpoint against `736dbd5`
halves Butteraugli blur execution and reduces paired 4K total encoding time
by 5.4%. AC-search residual evaluation, DCT, reconstruction filtering, host
work, and transfers remain material targets; the resident path has not
reached its performance limit.

The initial measurements showed two different performance profiles.

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

#### S1.3 follow-up: pack the remaining AC-search DCT shapes (2026-09-04)

The current padded-4K trace exposed a different transform mix after the first
S1.3 specializations removed the formerly dominant 32-wide cost. The generic
8x8 kernel assigned one 256-thread block to only 64 coefficients, while 16x8
and 8x16 used only 128 lanes. The still-generic 16x16 path occupied every lane
but retained the horizontal pass's divergent constant-memory basis access.
Together those four shapes consumed `119.601 ms` in AC search, 77.4% of its
DCT time and 28.4% of all GPU kernel execution.

This follow-up adds compile-time kernels that execute four independent 8x8
transforms or two independent 128-coefficient transforms in each 256-thread
block. The 16x16 kernel uses one transform per block. All four variants load a
single padded horizontal-basis tile into shared memory and give every thread
one coefficient. A partial final block is explicitly masked. The dense basis,
coefficient layout, scaling, per-output accumulation order, and public
standalone transform contract remain unchanged.

Seven alternating independent-process parent/candidate pairs used the
synthetic odd `3839x2159` workload, distance `1.2`, effort `7`, fully-resident
AQ, one internal warmup, and one retained GPU-only sample. Times are cohort
medians in milliseconds; the paired ratio is also reported because host work
and laptop boost state remained noisy:

| Stage | Parent `39397b7` | Packed DCT | Cohort change | Median paired change |
|---|---:|---:|---:|---:|
| Complete workflow | 929.948 | 776.815 | -16.5% | -12.8% |
| Quantization pipeline | 635.794 | 530.151 | -16.6% | -14.4% |

All seven quantization pairs favored the packed kernels. Parent quantization
ranged from `613.765-664.463 ms`; the candidate ranged from
`523.044-590.200 ms`. Complete-workflow ranges were `830.120-1043.372 ms`
and `763.330-987.404 ms`, respectively. Two complete-workflow pairs regressed
despite faster quantization because the CPU codestream stage varied
independently; the quantization and GPU-timeline results are the reliable
signals for this checkpoint.

Matched warmed padded-4K Nsight Systems captures isolate the intended change:

| GPU scope | Parent | Packed DCT | Change |
|---|---:|---:|---:|
| All kernel execution | 420.943 ms | 332.657 ms | -21.0% |
| AC-strategy search | 210.652 ms | 123.390 ms | -41.4% |
| AC-search DCT | 154.592 ms | 64.583 ms | -58.2% |
| Targeted 8x8/16x8/8x16/16x16 DCT | 119.601 ms | 26.521 ms | -77.8% |
| Resident AQ and reconstruction | 180.324 ms | 179.315 ms | -0.6% |

Kernel launch count remains 516 because packing reduces the grid size of each
small-shape dispatch rather than its launch count. AC search is no longer the
largest phase in the candidate trace: resident AQ and reconstruction now lead
at `179.315 ms`. The unchanged 32-wide DCT families were collectively about
3 ms slower in the candidate trace, consistent with the session's clock and
thermal variation; the packed families still saved about 93 ms on their own.

All 53 tests passed in the CUDA build. The CUDA transform test covers every
supported shape against the double-precision reference, and the CUDA
AC-strategy test covers CPU cost parity, resident quant-norm evaluation,
invalid descriptors, and submission failure behavior. Compute Sanitizer
memcheck reported zero errors for the CUDA AC-strategy suite. Parent and
candidate effort-7 and effort-9 fully-resident encodes of
`testdata/codestream_sample.pfm` were byte-identical, with SHA-256 values
`61be086bab4db87984699245cc0fe2eef107050d667b1070c5d93e3a25a37f5d`
and `14dfb18d19fd11e590513b3a2188fc97bd2c8402ceaad61a46f3a65cac32eab3`,
respectively. Ignored trace artifacts use the `s17_ac_packed_4k` prefix under
`build-cuda-ninja/profiles`.

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

#### S1.5 completion snapshot (2026-09-04)

`InitialCflKernel` now assigns a four-thread cooperative group to each 64x64
color tile and packs 32 groups into each 128-thread block. Each thread owns one
of the CPU-compatible four accumulator lanes and visits that lane's samples in
the original order. Fixed-width subgroup shuffles then reproduce the original
`(lane 0 + lane 1) + (lane 2 + lane 3)` horizontal sum for the three means and
three regression terms. This defines the parallel floating-point order and
preserves the existing quantized-map contract rather than introducing a new
reduction approximation.

Paired Nsight Systems captures compare revision `4a29b1d` with the completed
kernel on the same RTX 3060 Laptop GPU. Each trace contains one warmed sample:

| Workload and AQ mode | Before S1.5 | After S1.5 | Change |
|---|---:|---:|---:|
| 1080p fully resident | 4.555 ms | 0.470 ms | -89.7% |
| 1080p maximum throughput | 6.175 ms | 0.539 ms | -91.3% |
| 4K maximum throughput | 5.904 ms | 1.190 ms | -79.8% |

The 1080p launch changes from two 256-thread blocks with one serial tile per
thread to sixteen 128-thread blocks with four cooperating threads per tile.
At 4K it changes from eight 256-thread blocks to sixty-four 128-thread blocks.
Register use rises from 40 to 42 per thread, no shared memory is introduced,
and the total workflow launch counts remain unchanged at 501 for fully
resident and 34 for maximum throughput.

The paired padded-1080p public wall profile used two warmups, nine samples for
maximum throughput, and seven samples for fully resident. Times are warmed
medians in milliseconds:

| AQ mode and stage | Before S1.5 | After S1.5 | Change |
|---|---:|---:|---:|
| Fully resident, total | 261.6 | 246.3 | -5.8% |
| Fully resident, quantization | 171.0 | 161.6 | -5.5% |
| Maximum throughput, total | 120.2 | 103.2 | -14.1% |
| Maximum throughput, quantization | 37.2 | 29.8 | -19.8% |

The post-change total ranges were `232.3-291.0 ms` for fully resident and
`99.9-131.0 ms` for maximum throughput; the corresponding baseline ranges
were `248.5-342.0 ms` and `109.8-141.0 ms`. The wall-stage changes are larger
than the isolated 4.1-5.6 ms kernel savings and therefore include clock and
host-timing variance. The kernel trace is the causal measurement; the wall
profile demonstrates that the gain survives the public workflow boundary.

The final post-change paired batch qualification was:

| AQ mode | Batch | Batch ms | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 251.1 | 3.98 | 1.06x |
| Fully resident | 2 | 430.9 | 4.64 | 1.24x |
| Fully resident | 4 | 790.1 | 5.06 | 1.67x |
| Maximum throughput | 1 | 110.5 | 9.05 | 0.99x |
| Maximum throughput | 2 | 170.1 | 11.76 | 1.42x |
| Maximum throughput | 4 | 281.9 | 14.19 | 1.67x |

The GPU warmed from 65 C/P8 to 74 C/P0 during this sweep. Separate baseline
and post-change batch sessions moved inconsistently by roughly 0-10% across
batch sizes, so they do not establish an S1.5 batch-throughput change. They do
show that batch sizes 1, 2, and 4 remain functional and that the two-lane pool
continues to overlap work; the isolated kernel traces remain the reliable
performance comparison for this bounded optimization.

CUDA tests now compare the resulting initial CfL maps directly against the CPU
four-lane oracle for both structured and deterministic noisy images. The test
geometry includes a partial right tile and partial bottom rows. Both maps and
complete maximum-throughput codestreams remain byte-identical, while repeated
four-worker runs cover fully-resident and maximum-throughput workflows. Odd
1080p fully-resident and maximum-throughput codestreams, plus the odd 4K
fully-resident codestream, are SHA-256 identical to their S1.4 outputs. The
pinned `djxl` decoded all three and reported the original `1919x1079` and
`3839x2159` dimensions. The checkpoint passed all 53 CUDA tests and all 47
CPU-only tests; Compute Sanitizer memcheck also reported zero errors for the
CUDA AQ suite. Ignored trace artifacts use the `s15_*` prefix under
`build-cuda-ninja/profiles`.

### S1.6: capture stable resident submissions with CUDA Graphs

**Priority:** follow-on optimization after arenas.

Fully-resident encoding launches 497 kernels and spends `12.7 ms` in launch
APIs. Once arena packing guarantees stable pointers and the execution plans
have fixed geometry, graph capture can reduce host launch work and repeated
submission setup. Dynamic strategy batches and control-mode differences may
require a small graph cache keyed by geometry and policy shape.

Maximum-throughput has only 32 launches and `0.8 ms` of launch API time, so it
does not justify a graph-specific implementation by itself.

### Follow-up: tile Butteraugli Malta response (2026-09-05)

After packed DCT, resident AQ and reconstruction became the largest GPU phase.
A fresh warmed odd-4K trace at revision `2b91775` attributes `46.621 ms` to
24 `MaltaResponseKernel` launches. Each output reads an overlapping radius-4
neighborhood from global memory and repeats bounds handling for its samples.
The compiled kernel uses 105 registers per thread.

The retained implementation cooperatively loads a 32x8 output tile and its
four-pixel halo into a 40x16 shared-memory array. Out-of-image halo entries
are zero, and partial-tile threads participate in loading and synchronization
before exiting. Each warp evaluates one row using fixed shared-memory offsets.
The response formulas, addition trees, explicit unfused square accumulation,
and stage ordering are unchanged. The kernel uses 40 registers per thread and
2,560 bytes of shared memory. It adds no device allocations or launches.

Matched warmed Nsight Systems captures on the RTX 3060 Laptop GPU show:

| GPU scope | Parent `2b91775` | Tiled Malta | Change |
|---|---:|---:|---:|
| Malta response, main scale | 37.332 ms | 12.181 ms | -67.4% |
| Malta response, subscale | 9.289 ms | 3.186 ms | -65.7% |
| All Malta response | 46.621 ms | 15.367 ms | -67.0% |
| All kernel execution | 336.691 ms | 307.198 ms | -8.8% |

Both traces contain 516 launches, 117,079,320 HtoD bytes, and 103,699,012
DtoH bytes. A 32x16 alternative was also measured: it uses 512 threads and
3,840 shared bytes per block but takes `16.365 ms` for Malta response, 6.5%
longer than 32x8 in these captures. The 32x8 shape was retained. These timings
and compiler resource counts establish the improvement without attributing
it to unmeasured occupancy or cache-hit counters.

Public workflow measurements use seven alternating parent/candidate process
pairs per workload, each with one warmup and one retained sample, distance
`1.2`, effort `7`, fully-resident AQ, automatic CPU threads, and no final-score
diagnostic. Cohort medians and median paired changes are:

| Workload and stage | Parent | Tiled Malta | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 752.910 ms | 716.324 ms | -4.9% |
| Odd 4K, quantization | 526.240 ms | 498.863 ms | -7.0% |
| Odd 1080p, total | 224.350 ms | 240.598 ms | +1.9% |
| Odd 1080p, quantization | 140.054 ms | 137.667 ms | -5.3% |

Five of seven 4K pairs improved both total and quantization time. Parent 4K
totals ranged from `718.114-805.252 ms`, versus `686.161-765.478 ms` for the
candidate. At 1080p, six of seven quantization pairs improved, but only three
total-time pairs improved. CPU codestream serialization ranged from
`64.589-82.083 ms` in the parent and `63.538-135.582 ms` in the candidate,
despite unchanged output bytes. This session does not establish an end-to-end
1080p improvement; the 4K paired result and isolated kernel savings support
retaining the change. GPU state moved from 61 C/P8/210 MHz before the paired
runs to 71 C/P0/1762 MHz afterward.

Post-change 1080p batch checks used one warmup and three alternating
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 219.364 ms | 4.559 | 1.005x |
| Fully resident | 2 | 348.133 ms | 5.745 | 1.331x |
| Fully resident | 4 | 649.428 ms | 6.159 | 1.412x |
| Maximum throughput | 1 | 108.430 ms | 9.223 | 0.995x |
| Maximum throughput | 2 | 161.797 ms | 12.361 | 1.280x |
| Maximum throughput | 4 | 312.849 ms | 12.786 | 1.482x |

These qualify batch behavior, not a before/after batch-performance claim.
Maximum-throughput does not run Malta response. GPU state at the end of batch
qualification was 66 C/P0/1282 MHz.

All 53 CUDA-build tests and all 47 CPU-only tests pass. Butteraugli coverage
now includes `31x8`, `32x9`, `65x33`, and `127x65` inputs to exercise tile
boundaries, multiple interior tiles, and odd multiscale geometry with poisoned
host/device padding. The worst CPU-reference distance-map error is
`2.0504e-5`, below the existing `1.5e-3` tolerance. Compute Sanitizer memcheck,
racecheck, synccheck, and initcheck report zero errors or hazards on this
suite. Existing CUDA AQ tests cover exact CPU codestream identity, failure
atomicity, and repeated four-worker resident/maximum-throughput encoding.

Parent and candidate fully-resident codestreams are byte-identical for the
sample image at efforts 7 and 9 and the odd 1080p/4K qualification images at
effort 7, all with final-score collection enabled. Reported final scores also
match. The pinned `djxl` independently decodes all four at the original
dimensions. The odd 1080p SHA-256 is
`454cd84cc7ee6e915075ab741e7eec4bcc04a250c5f82d238c8ec4979f9f45f3`;
the odd 4K SHA-256 is
`86be75ba11d76a780edc6dea4b80daec848ddddfcf9452fe60f29f77afac0bc9`.

Ignored artifacts under `build-cuda-ninja/profiles` use the `s18_` prefix:
`s18_parent_4k`, `s18_malta_4k`, and `s18_tile32x16_4k` traces/SQLite exports,
`s18_paired_4k.json`, `s18_paired_1080p.json`, and qualification codestreams.

The next measured targets in the candidate 4K trace are Butteraugli
convolutions (`55.071 ms`), AC-search residual evaluation (`34.214 ms`), and
EPF (`20.597 ms`). CUDA launch API time is only `6.525 ms` for 516 launches,
so graph work should be compared against these larger compute opportunities.
CPU serialization and the remaining synchronous transfers also need attention
as GPU execution shrinks. This checkpoint does not establish a performance
ceiling or qualify other GPU generations and a natural-image corpus.

### Follow-up: specialize and tile Butteraugli blurs (2026-09-05)

At parent revision `736dbd5`, Butteraugli blur is the largest named kernel
family in a fresh odd-4K capture: `115.886 ms` over 146 launches. The 7-, 13-,
15-, and 33-tap filters use runtime loop bounds and repeatedly address
overlapping global-memory samples. The same weight sum is recomputed for
every interior pixel.

The retained implementation makes filter sizes compile-time parameters and
uses 256-thread blocks to evaluate 256x4 horizontal or 32x64 vertical output
tiles. Each block cooperatively loads its directional halo and weights into
shared memory. Four horizontal or eight vertical outputs per thread amortize
loading and synchronization. The complete weight sum is calculated once per
block in the original addition order; edge pixels still accumulate only
included weights in their original order. Padding does not contribute to
edge normalization. The separate mirrored 5-tap filter is unrolled without
tiling and preserves its reflection behavior. No approximate arithmetic or
new allocations are introduced.

The experiment compared fixed-size direct reads with two tiled layouts before
selecting the final kernels. All captures below use two warmups and one
profiled odd-4K sample, distance `1.2`, effort `7`, fully-resident AQ, and no
final-score diagnostic:

| Implementation | 7/13/15/33-tap blurs | All Butteraugli blurs |
|---|---:|---:|
| Parent, runtime bounds/direct reads | 100.142 ms | 115.886 ms |
| Fixed sizes/unrolled direct reads | 78.057 ms | 93.285 ms |
| Tiled 128x8 horizontal / 32x32 vertical | 49.854 ms | 65.447 ms |
| Tiled 256x4 horizontal / 32x64 vertical | 46.928 ms | 61.652 ms |
| Retained tiles plus unrolled mirrored 5-tap | 46.702 ms | 58.149 ms |

Specialization alone removes 22.1% of targeted time; tiling gives a further
substantial improvement. This supports reducing repeated sample access,
address calculation, and normalization work rather than attributing the
entire result to loop unrolling. It does not establish a DRAM-bandwidth or
occupancy bottleneck without hardware counters. The retained tiled kernels
use 31-55 registers per thread and at most 12,424 shared bytes per block.

Total blur time falls 49.8%. All GPU kernel execution falls from `565.944 ms`
to `487.564 ms` (-13.8%), but non-blur kernels also change from `450.058 ms`
to `429.415 ms` (-4.6%), so the entire total-kernel reduction should not be
attributed to this edit. Laptop execution state differs substantially from
the preceding Malta session; absolute times across those sessions are not
comparable. Both current parent/candidate traces still contain 516 launches,
117,079,320 HtoD bytes, and 103,699,012 DtoH bytes.

Public workflow measurements use seven alternating independent-process pairs
per workload, one warmup and one retained sample per process, and the same
distance/effort/AQ settings. Cohort medians and median paired changes are:

| Workload and stage | Parent | Retained blurs | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 1045.051 ms | 976.917 ms | -5.4% |
| Odd 4K, quantization | 839.785 ms | 771.840 ms | -7.5% |
| Odd 1080p, total | 254.085 ms | 254.615 ms | +0.3% |
| Odd 1080p, quantization | 178.039 ms | 161.209 ms | -10.9% |
| Flower 510x532, total | 49.864 ms | 49.461 ms | -1.6% |
| Flower 510x532, quantization | 33.739 ms | 33.641 ms | -1.0% |

All seven 4K pairs improve both total and quantization time. Parent totals
range from `1010.224-1077.771 ms`, versus `948.344-1062.143 ms` for the
candidate. Five of seven 1080p quantization pairs improve, while total time
remains effectively unchanged. CPU serialization ranges from
`59.513-95.637 ms` in the parent and `60.977-105.995 ms` in the candidate.
The small Flower differences are not a reliable performance gain. GPU state
moves from 65 C/P8/210 MHz before these runs to 69 C/P3/825 MHz after the
4K/1080p pairs.

Flower uses the pinned `flower_small.rgb.depth8.ppm`, converted once outside
measurement to linear sRGB float32 PFM with the standard sRGB transfer
function. That PFM's SHA-256 is
`2ad3bf99e39d8b2d5e18130e8ab51dfb9b1ff360627a414a49646904ca3ee9cd`.
Its fully-resident frame selects all seven production AC strategies. Parent
and candidate codestreams are byte-identical for Flower at efforts 7 and 9,
the small sample at efforts 7 and 9, and the odd 1080p/4K fixtures at effort 7,
all with final-score collection enabled. Reported final scores match, and
the pinned `djxl` decodes all six at their original dimensions. Flower
codestream SHA-256 values are
`41c30c28169e09ff763cc242cce9e9b5b50db8841f2e44185bc91acc864c3b57` (effort 7)
and `e20404ed5eda52afc59cf1ab75d54d48433abcd31243e84266b04e28afd65978`
(effort 9). Synthetic fixture hashes remain those of the Malta checkpoint.

All 53 CUDA-build tests and all 47 CPU-only tests pass. Added `255x63`,
`257x67`, and `33x129` Butteraugli cases exercise complete and partial tiles,
vertical tile boundaries, wide filter clipping on small images, multiscale
evaluation, and poisoned strides. The worst CPU-reference map error is
`2.35289e-5`, within the unchanged `1.5e-3` tolerance. Compute Sanitizer
memcheck, racecheck, synccheck, and initcheck report zero errors or hazards.
Existing tests also verify exact CPU/CUDA codestream identity, failure
atomicity, allocation invariants, and repeated four-worker resident and
maximum-throughput workflows.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 268.952 ms | 3.718 | 0.898x |
| Fully resident | 2 | 498.002 ms | 4.016 | 1.078x |
| Fully resident | 4 | 866.495 ms | 4.616 | 1.245x |
| Maximum throughput | 1 | 119.275 ms | 8.384 | 1.008x |
| Maximum throughput | 2 | 158.959 ms | 12.582 | 1.540x |
| Maximum throughput | 4 | 266.801 ms | 14.992 | 1.957x |

These are functional/overlap checks, not a before/after batch-performance
claim. In particular, resident batch-1 is slower than serial in this session.
The final GPU state is 65 C/P3/1282 MHz. Maximum-throughput without final-score
collection does not execute the changed blurs.

Ignored artifacts under `build-cuda-ninja/profiles` use `s19_` prefixes. The
paired trace baseline is `s19_parent_warm2_4k`; the earlier one-warmup
`s19_parent_4k` capture is not the comparison used above. Alternative traces
are `s19_specialized_4k`, `s19_tiled_4k`, and `s19_large_tiles_4k`; the retained
trace is `s19_final_4k`. Paired wall results are in
`s19_paired_{4k,1080p,flower}.json`.

The next trace target is AC-search `ResidualKernel` at `60.879 ms`. Source
inspection shows that every coefficient thread repeats the same candidate
validation, quant-norm field reduction, and CfL lookup. Moving this uniform
work out of individual coefficient lanes, and reducing the 1024-thread
launches, merits a measured experiment. Remaining DCT, EPF, host preparation,
serialization, and synchronous transfer costs also prevent a performance-
ceiling claim. Launch API time is only `6.572 ms` in the retained trace.

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
