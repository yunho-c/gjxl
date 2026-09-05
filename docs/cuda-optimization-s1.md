# CUDA optimization study S1

- Status: S1.1-S1.5, packed/register-tiled DCT, tiled Malta,
  specialized/tiled blurs, packed AC-search residuals, tiled EPF, and fused
  AC gather/DCT plus residual/inverse/loss fusion, compact AC-search scratch,
  factorized resident DCT, and cooperative quantization adjustment
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
below supersede them. The latest cooperative-adjustment checkpoint against
`f26e2ef` reduces its targeted kernel time by 84.4% at 4K, 92.4% at 1080p,
and 96.7% on Flower, with unchanged launches/transfers. All 56 CUDA and 47
CPU tests pass, as do eight sanitizer runs (full-AQ race instrumentation is
kernel-filtered). All 23 decoded image pairs remain byte-identical. Separate
parent/candidate cohorts, an identical-binary control, and a same-executable
policy probe show substantial wall-time variability: no stable end-to-end
gain is claimed. The same-executable probe also verifies exact output on the
three actual benchmark inputs. This is a kernel-level improvement, not a
demonstrated overall speedup or performance ceiling.

The preceding factorized-DCT checkpoint against `c1a75bb`
reduces 4K DCT kernel time from 107.9 to 23.1 ms (78.6%), without adding launches
or transfers. Paired fully-resident total time improves 5.1% at 4K, 7.2% at
1080p, and 7.8% on Flower; quantization improves 10.6%, 8.9%, and 5.9%.
All 55 CUDA and 47 CPU tests, 13 sanitizer runs, batch checks, and 23 decoded
before/after image pairs pass qualification. Rounding changes: only ten image
pairs are byte-identical, while independently decoded Butteraugli scores are
unchanged in 22 pairs and improve slightly in one. The largest size increase
is 0.028%. Laptop timing variance and corpus/device limits remain explicit.

The preceding compact-scratch checkpoint against `5ba4d86`
reduces the padded-4K AC-search arena from 626.8 to 323.8 MB (48.3%), saving
288.9 MiB per search. Nsight confirms that peak tracked device allocations fall
from 3.110 to 2.808 GB, with identical launches and transfer volumes. Paired total
time improves 3.2% at 4K, is effectively unchanged at 1080p (-0.2%), and improves
5.0% on Flower; substantial laptop timing variance remains. Batch checks pass,
but do not establish a batch-throughput gain. Qualified codestreams remain
byte-identical.

The preceding residual/inverse/loss fusion reduced targeted AC stages by 23.3%
at 4K, removing seven launches and a 2.55 GB coefficient-buffer round trip.
These are incremental gains, not a demonstrated performance ceiling.
Filtering, resident coefficient encoding, remaining resident allocations, host work, and
transfers remain material targets; the resident path has not reached its
performance limit.

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

### Follow-up: pack AC-search residual evaluation (2026-09-05)

At parent revision `5513704`, a fresh odd-4K trace spends `61.815 ms` in
`ResidualKernel`. Each coefficient thread independently validates the same
candidate, computes the same strategy-aware quant norm, and loads the same
CfL factor. The norm includes a field reduction and logarithm/power
approximation for larger transforms. A 32x32 candidate repeats this work
3,072 times across its three channels, then computes the norm again in the
cost kernel. The residual reduction also uses a block-wide barrier at every
halving step, with one 64-1,024-thread block per channel transform.

The retained implementation:

- prepares one quant norm per candidate in a small separate kernel;
- temporarily stores those norms in the existing cost output, reads them
  during residual evaluation, then replaces each norm with its final cost;
- evaluates candidate validity and CfL once per channel transform;
- packs independent transforms into 256-thread blocks; and
- preserves the original halving addition tree across registers, shared
  memory where needed, and full-warp shuffles.

Cost storage is already disjoint from inputs and other scratch. Preparation,
residual evaluation, and final cost writes execute on the same ordered
stream; batch outputs are only valid after their submission completes.
This needs no new allocation, transfer, or public API. Invalid
descriptors still produce non-finite costs. Inactive tail transforms join
all required barriers but do not read candidates or touch outputs.

The selected geometry bounds register-held work to eight coefficients per
lane:

| Coefficients | Threads/transform | Transforms/block | Coefficients/thread |
|---|---:|---:|---:|
| 64 | 32 | 8 | 2 |
| 128 | 32 | 8 | 4 |
| 256 | 32 | 8 | 8 |
| 512 | 64 | 4 | 8 |
| 1,024 | 128 | 2 | 8 |

All experiments use two warmups and one profiled odd-4K sample, distance
`1.2`, effort `7`, fully-resident AQ, and no final-score diagnostic. The
AC total below includes gather, norm preparation when present, residual,
and cost kernels, but not DCT:

| Experiment | Residual | AC total |
|---|---:|---:|
| Parent: uniform work per coefficient | 61.815 ms | 102.775 ms |
| Uniform work once/block; original geometry | 53.141 ms | 92.601 ms |
| Packed residuals; at most 256 threads/transform | 30.990 ms | 73.591 ms |
| Cache norms in gather; same packed layout | 25.550 ms | 65.171 ms |
| Separate norm preparation; same packed layout | 26.286 ms | 64.558 ms |
| Separate preparation; 64 threads/transform | 16.693 ms | 58.538 ms |
| Separate preparation; 32 threads/transform | 19.503 ms | 61.280 ms |
| Retained shape-dependent layout | 13.705 ms | 54.443 ms |

Moving norm preparation out of gather keeps gather at 16 registers/thread
instead of 40. Seven preparation launches cost `0.104 ms` in the retained
trace. The all-32-thread experiment improves smaller transforms, but its
32x32 kernel takes `7.196 ms`; `cuobjdump` reports a 256-byte stack frame.
The retained 128-thread 32x32 group takes `2.317 ms` with no stack frame.
All retained residual variants use 26-33 registers/thread, 96-2,096 shared
bytes/block, and zero stack bytes. These observations support the work
sharing and register-footprint choices without claiming hardware-counter
proof of an occupancy or bandwidth bottleneck.

Residual execution falls 77.8%, and the four-stage AC total falls 47.0%.
All GPU kernels fall from `502.994 ms` to `471.199 ms` (-6.3%); unrelated
kernel time rises from `400.219 ms` to `416.756 ms` (+4.1%), so the complete
kernel total is affected by execution-state variation. Kernel launches
increase from 516 to 523. Transfers remain 117,079,320 HtoD bytes,
103,699,012 DtoH bytes, and 518,400 device-to-device bytes in both traces.

Public workflow measurements use seven alternating independent-process
pairs per workload, one warmup and one retained sample per process, with
the same distance/effort/AQ settings:

| Workload and stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 946.004 ms | 909.196 ms | -3.8% |
| Odd 4K, quantization | 738.959 ms | 715.452 ms | -3.7% |
| Odd 1080p, total | 232.420 ms | 228.542 ms | +0.2% |
| Odd 1080p, quantization | 159.536 ms | 156.485 ms | -3.4% |
| Flower 510x532, total | 50.418 ms | 49.974 ms | -3.5% |
| Flower 510x532, quantization | 33.167 ms | 32.382 ms | -1.3% |

All seven 4K quantization pairs and six total-time pairs improve. Parent
4K totals range from `910.241-964.777 ms`, versus `875.354-945.962 ms` for
the candidate. Four of seven 1080p quantization pairs improve, but there is
no reliable total-time improvement at 1080p. Flower's small differences
are also noisy: total-time ranges are `43.910-59.409 ms` and
`43.288-65.436 ms`. GPU state moves from 61 C/P8/210 MHz before the pairs
to 66 C/P3/1282 MHz afterward. Absolute measurements from the preceding
convolution session should not be used as this checkpoint's baseline.

All 53 CUDA-build tests and 47 CPU-only tests pass. New candidate-prefix
tests cover partial packed blocks, exact result independence from batch
length, and untouched output guards. Invalid-descriptor tests now also
check that neighboring valid candidates remain correct. Existing host-norm
and device-norm CPU-reference cost checks retain their tolerances; the
largest absolute/relative errors remain `0.00585938` / `4.03204e-7` for
32x32. Compute Sanitizer memcheck, racecheck, synccheck, and initcheck on
the AC-strategy test report zero errors or hazards.

Parent/candidate codestreams and reported final perceptual scores are
identical for the small sample at efforts 7 and 9, odd 1080p/4K at effort 7,
and Flower at efforts 7 and 9, with final-score collection enabled. The
pinned `djxl` decodes all six at their original dimensions. The six hashes
remain those recorded by the Malta/convolution checkpoints. Flower still
selects all seven production AC strategies. Full suites retain exact-mode
CPU/CUDA identity, failure-atomicity, allocation, and concurrent workflow
coverage.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 243.159 ms | 4.113 | 0.981x |
| Fully resident | 2 | 500.675 ms | 3.995 | 1.098x |
| Fully resident | 4 | 869.448 ms | 4.601 | 1.172x |
| Maximum throughput | 1 | 102.415 ms | 9.764 | 1.000x |
| Maximum throughput | 2 | 175.958 ms | 11.366 | 1.448x |
| Maximum throughput | 4 | 259.002 ms | 15.444 | 1.781x |

These validate output stability and overlap, not a before/after batch
performance improvement. Resident batch-1 is slightly slower than serial,
and batch-2 does not improve images/s over batch-1 in this run. The final
GPU state is 67 C/P3/1282 MHz.

Ignored artifacts under `build-cuda-ninja/profiles` use `s20_` prefixes.
Trace names are `s20_{parent,hoisted,packed,cached_norm,separate_norm,
packed64,packed32,final}_4k`; each has an `.nsys-rep` and exported `.sqlite`.
Paired wall results are `s20_paired_{4k,1080p,flower}.json`. The saved parent
benchmark/encoder and `s20_verify.ps1` reproduce the measured comparison
and six-file identity/decode checks against the retained executables.

The retained profile still spends `161.739 ms` in forward/inverse DCT,
`31.603 ms` in EPF, `21.193 ms` in AC gather, and `19.441 ms` in AC cost.
Gather repeats runtime index division and candidate validation per pixel;
cost still has one coefficient-sized block per candidate and shared-memory
reductions. These are concrete follow-up targets alongside host assembly,
serialization, and transfer boundaries. This checkpoint does not establish
that fully-resident encoding is maxed out.

### Follow-up: stage and register-tile large DCTs (2026-09-05)

At parent revision `3c08276`, DCT again leads the fresh odd-4K profile:
`158.311 ms` across all shapes, including `111.060 ms` in 16x32, 32x16,
and 32x32 transforms. Their 256-thread blocks compute two or four outputs
per thread in separate dot-product loops. That reloads the same horizontal
basis for each output, then reloads the same intermediate samples for each
vertical output. Horizontal passes repeatedly read global input, and forward
column-major coefficient stores are strided across lanes.

The retained kernels cooperatively load input into a padded shared tile,
then accumulate independent outputs simultaneously in two or four registers
per thread. Each horizontal basis value and each vertical intermediate
sample is shared by those accumulators. The forward kernel reuses its input
tile for the final layout conversion and coalesced global stores, with one
additional barrier after the vertical pass. The inverse kernel loads native
coefficient order contiguously before broadcasting horizontal-pass samples.
Neither direction changes the dense basis, scaling, coefficient layout, or
any output's multiply-add order. No fast-math or factored-transform policy
is introduced.

For 16x32, two vertical basis addresses occur per warp, so that basis is also
staged in padded shared memory. The 32-wide transforms keep their vertical
constant-memory broadcasts. The final kernels retain 256 threads/block,
use 39-40 registers/thread and zero stack bytes, and require 9,536 shared
bytes for 16x32, 8,384 for 32x16, and 12,544 for 32x32. There are no new
device allocations, transfers, or kernel launches.

Experiments use two warmups and one profiled odd-4K sample, distance `1.2`,
effort `7`, fully-resident AQ, and no final-score diagnostic:

| Experiment | Large-shape DCT | All GPU kernels |
|---|---:|---:|
| Parent | 111.060 ms | 469.739 ms |
| Staged input/output, separate accumulators | 102.629 ms | 472.340 ms |
| Staging plus simultaneous accumulators, 256 threads | 70.447 ms | 417.325 ms |
| Same approach, 128 threads | 74.341 ms | 441.793 ms |
| 256 threads plus shared vertical basis for 16x32 | 69.993 ms | 420.766 ms |
| Also share vertical bases in smaller packed shapes | 71.169 ms | 429.120 ms |
| Retained large-only change, post-validation capture | 75.345 ms | 438.921 ms |

The small-shape extension raises packed-DCT time from `47.888 ms` to
`57.628 ms` in those trial captures and is not retained. Reducing the large
block size to 128 also fails to establish a win over 256. Staging alone has
limited benefit; sharing loads across independent accumulators supplies the
larger gain. These timings do not establish a hardware-counter diagnosis of
occupancy, bandwidth, or issue stalls.

A separate CUDA-event probe controls for noisy individual workflow calls.
It runs 65,536 transforms per dispatch, three warmup dispatches, then seven
samples of three dispatches each. Parent/candidate processes run in
parent-candidate-candidate-parent order. Ranges below span the two process
medians, not individual samples:

| Shape/direction | Parent median range | Retained median range |
|---|---:|---:|
| 16x32 forward | 6.551-6.745 ms | 5.956-6.024 ms |
| 16x32 inverse | 7.311-7.522 ms | 5.781-5.841 ms |
| 32x16 forward | 8.227-8.394 ms | 5.370-5.452 ms |
| 32x16 inverse | 8.401-8.488 ms | 5.440-5.491 ms |
| 32x32 forward | 23.507-24.166 ms | 12.321-13.190 ms |
| 32x32 inverse | 18.580-18.813 ms | 11.469-12.196 ms |

The post-validation workflow capture is less favorable for 16x32: its
combined forward/inverse time rises from `23.136 ms` to `24.109 ms`.
The event probe supports retaining that shape, but its workflow gain is
not established by a single trace. Combined 32x16 time falls from
`24.674 ms` to `17.538 ms`, and 32x32 from `63.250 ms` to `33.699 ms`.
Large-shape DCT falls 32.2%; all DCT falls from `158.311 ms` to
`126.923 ms` (-19.8%). Unchanged packed shapes rise from `47.251 ms` to
`51.577 ms`, while non-DCT execution is nearly unchanged at
`311.428 ms` versus `311.999 ms`. All kernel time falls 6.6%.
Both captures contain 523 launches, 117,079,320 HtoD bytes,
103,699,012 DtoH bytes, and 518,400 device-to-device bytes.

Public workflow measurements use seven alternating independent-process
pairs per workload, one warmup and one retained sample per process, with
the same distance/effort/AQ settings:

| Workload and stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K, total | 915.952 ms | 874.546 ms | -4.5% |
| Odd 4K, quantization | 709.443 ms | 670.049 ms | -6.1% |
| Odd 1080p, total | 245.894 ms | 214.149 ms | -11.2% |
| Odd 1080p, quantization | 162.712 ms | 142.060 ms | -11.5% |
| Flower 510x532, total | 46.744 ms | 42.446 ms | -7.3% |
| Flower 510x532, quantization | 32.041 ms | 27.749 ms | -4.7% |

All seven 4K quantization pairs improve; five total-time pairs improve.
The 4K total ranges are `895.220-1053.213 ms` and `839.043-948.671 ms`.
All seven 1080p pairs improve both stages, with total ranges of
`220.312-272.967 ms` and `211.792-232.445 ms`. Six of seven Flower pairs
improve both stages. GPU state moves from 63 C/P0/1282 MHz before the
wall pairs to 66 C/P3/1282 MHz afterward. Host serialization and execution
state still contribute to the wall-time changes; do not equate those
percentages with isolated kernel savings or compare absolute times across
earlier sessions.

All 53 CUDA-build tests and 47 CPU-only tests pass. The standalone CUDA DCT
test now uses 19 transforms per supported shape: impulses at distinct tile
positions, a constant, a checkerboard, unequal horizontal/vertical ramps,
and deterministic noise. It checks forward output and round trips, then
independent inverse output against the double-precision reference. Existing
tolerances remain `3e-5 + 3e-4 * abs(reference)` for forward output and
`5e-4 + 5e-4 * abs(reference)` for inverse/round-trip output. All nine
supported DCT shapes are covered, including the unchanged generic shapes.
AC-strategy cost errors remain unchanged.

Compute Sanitizer memcheck, racecheck, synccheck, and initcheck on the CUDA
backend test report zero errors or hazards. Initial initcheck found that
the pre-existing completion-failure fixture submitted uninitialized DCT8
input; the fixture now initializes that input without changing its failure
assertions. Runs use `--report-api-errors no` because a separate existing
test deliberately issues an invalid zero-grid launch and verifies error
consumption; memory and synchronization checking remain enabled.

Parent/candidate codestreams and reported final perceptual scores are
identical for the small sample at efforts 7 and 9, odd 1080p/4K at effort 7,
and Flower at efforts 7 and 9, with final-score collection enabled. The
pinned `djxl` decodes all six at their original dimensions. Hashes remain
those recorded by the preceding checkpoints; Flower still selects all seven
production strategies. Existing full-suite checks retain exact-mode
CPU/CUDA identity, allocation, failure-atomicity, and concurrent workflow
coverage.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 233.947 ms | 4.274 | 0.944x |
| Fully resident | 2 | 425.952 ms | 4.695 | 1.183x |
| Fully resident | 4 | 802.579 ms | 4.984 | 1.193x |
| Maximum throughput | 1 | 97.867 ms | 10.218 | 0.998x |
| Maximum throughput | 2 | 146.258 ms | 13.674 | 1.417x |
| Maximum throughput | 4 | 284.660 ms | 14.052 | 1.757x |

These are output-stability and overlap checks, not a before/after batch
performance claim. Fully-resident batch-1 remains slower than serial.
The final GPU state is 66 C/P3/1282 MHz.

Ignored artifacts under `build-cuda-ninja/profiles` use `s21_` prefixes.
Trace names are `s21_{parent,staged_large,register_large,register128,
shared_vertical,small_shared,retained}_4k`, with `.nsys-rep` and `.sqlite`
files. The `small_shared` trace was initially captured as `s21_final_4k`
before that experiment was rejected and renamed; `s21_retained_4k` is the
final comparison. Paired wall data are `s21_paired_{4k,1080p,flower}.json`.
The saved parent executables and `s21_verify.ps1` reproduce identity/decode
qualification. `s21_dct_probe.cu`, `s21_parent_kernels.cu`, and the
`s21_probe_{parent,shared}` executables retain the independent event probe.

The retained profile still spends `126.923 ms` in DCT, `48.853 ms` in tiled
Butteraugli blurs, `42.133 ms` in Malta response, `32.227 ms` in EPF,
`20.024 ms` in AC gather, and `18.927 ms` in AC cost. Remaining DCT packing
and load-sharing opportunities, AC gather/cost reductions, neighborhood
filtering, host serialization, and transfer boundaries remain open. The
resident path is not yet at a demonstrated performance ceiling.

### Follow-up: specialize and tile EPF neighborhoods (2026-09-05)

The fresh parent is `6e4925f`, including the preceding large-DCT changes.
At odd 4K, four edge-preserving-filter (EPF) calls cost `33.330 ms`, behind
tiled Butteraugli convolution, Malta response, and the largest DCT families.
The old kernel chooses the pass at runtime and repeatedly samples mirrored
global coordinates inside the candidate, channel, and plus-patch loops.
This duplicates both neighborhood loads and coordinate work across pixels.

The retained implementation specializes all three passes and cooperatively
loads three channel planes into a shared-memory tile. A 256-thread block
produces a 32x32 output tile, with each warp processing one row at a time
and each thread processing four pixels in a non-unrolled outer loop. This
amortizes the halo without keeping four pixels' accumulators live. Halo
radii are 3, 2, and 1 for passes 0, 1, and 2. A flattened block grid avoids
introducing the CUDA grid-Y limit for tall images. No new device allocation,
transfer, or synchronization between kernels is needed.

The complete tile is loaded before any thread takes an out-of-image or
sigma-bypass branch. Bypass continues the per-thread pixel loop rather than
discarding later rows. Halo coordinates are mirrored from the original
patch coordinates: mirroring a candidate center first and then its patch
would differ at boundaries. The existing general modulo reflection remains
unchanged, including repeated reflection for one- and two-pixel dimensions.
Candidate/channel/patch accumulation order, explicit FMAs, division, sigma
threshold, border weighting, and non-finite error handling remain unchanged.
There is no fast-math or quality-policy change. Exact mode uses this shared
kernel too, but fully-resident performance is the optimization target.

Exploratory 4K traces use two warmups and one captured effort-7/distance-1.2
encode. They contain two calls each of passes 1 and 2, not pass 0:

| Variant | EPF total | Gaborish total |
|---|---:|---:|
| Parent, runtime pass | 33.330 ms | 3.032 ms |
| Pass specialization only | 10.880 ms | 3.531 ms |
| Specialized, branch-based mirror shortcut (rejected) | 39.809 ms | 5.565 ms |
| Shared 32x8 output tile | 6.611 ms | 3.036 ms |
| Shared 32x16 output tile | 5.792 ms | 3.488 ms |
| Shared 32x32 output tile | 5.673 ms | 3.519 ms |
| Retained 32x32, fresh capture after cleanup | 6.253 ms | 3.479 ms |

The mirror shortcut tried an interior fast return and one reflection before
falling back to modulo. It increased register use and slowed EPF as well
as Gaborish, which shares that helper; it is not retained. Gaborish is
otherwise unchanged and is an execution-state control. The small difference
between 16- and 32-row tiles is not a robust standalone speedup claim;
32 rows is the lowest exploratory total and avoids more duplicate halo loads.

The final EPF reduction is `27.077 ms` (81.2%). All-kernel time moves from
`435.721` to `418.439 ms` (4.0%), while unchanged DCT work moves from
`123.636` to `128.497 ms` and tiled blurs from `49.899` to `47.046 ms`.
These controls demonstrate the continuing clock/scheduling noise: do not
attribute the entire GPU-time difference to EPF. The retained compiler
reports 56/40/38 registers per thread and 17,328/15,552/13,872 shared bytes
for passes 0/1/2, with zero local bytes or stack. The parent's runtime kernel
uses 78 registers. Pass 0 has functional and sanitizer coverage but is not
timed by this default-profile comparison.

Initial public-workflow measurements use seven alternating independent
parent/candidate process pairs, one warmup and one measured encode per
process, GPU-only fully-resident mode, no final-score collection. Cohort
medians and median within-pair changes are separate statistics:

| Workload/stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K total | 894.389 ms | 849.697 ms | -5.2% |
| Odd 4K quantization pipeline | 686.358 ms | 644.961 ms | -6.3% |
| Odd 1080p total, initial | 217.774 ms | 228.658 ms | +6.2% |
| Odd 1080p quantization pipeline, initial | 143.347 ms | 149.054 ms | +4.8% |
| Flower total | 52.578 ms | 47.331 ms | -1.6% |
| Flower quantization pipeline | 33.572 ms | 31.560 ms | -2.3% |

Six of seven 4K total-time pairs and all seven quantization pairs improve;
total ranges are `866.884-972.024 ms` and `821.979-910.112 ms`. Flower's
small paired result is not strong evidence of a consistent wall-time gain.
The initial 1080p regression is retained in the record, not discarded:
candidate total times span `206.979-297.130 ms` versus `211.716-240.062 ms`
for the parent, and one candidate's unchanged CPU serialization takes
`125.373 ms` versus its paired parent's `66.116 ms`. Both host and
quantization-stage variation require further qualification.

A fresh three-warmup 1080p trace reduces EPF from `7.600` to `1.323 ms`
(82.6%) and all kernels from `88.353` to `80.271 ms`, while unchanged DCT
totals are `16.777` and `16.684 ms`. The follow-up wall comparison keeps
seven alternating process pairs but uses three warmups and five measured
encodes per process, comparing each process's median. All seven pairs
improve both stages. Total cohort medians are `230.794` and `222.892 ms`,
with a median paired reduction of 3.6%; quantization medians are `151.872`
and `142.812 ms`, with a paired reduction of 5.3%. The respective total
process-median ranges are `224.082-260.724` and `214.478-236.515 ms`.
This supports retaining the kernel change, not treating the initial wall
regression as a reproducible EPF regression. GPU state moves from
63 C/P8/210 MHz before the initial pairs to 67 C/P3/1282 MHz afterward.
Absolute timings should not be compared across checkpoints.

All 54 CUDA-build tests and 47 CPU-only tests pass. The new standalone
`cuda_epf` test compares 114 sequences with the independent CPU EPF
implementation: 19 image shapes, one/two/three iterations, and default/custom
weights. Shapes cover 1x1, single rows/columns, repeated mirror reflection,
8-pixel sigma boundaries, 32-pixel tile boundaries, and partial tiles.
Every observed reference error is zero, within the test tolerance
`2e-6 + 2e-6 * abs(reference)`. Input, both scratch images, and the sigma
field have distinct padded strides and guarded offsets; guards and read-only
inputs must remain intact. Sigma values include mixed active/bypass blocks
and the exact bypass threshold.

Additional cases cover NaN payloads, positive/negative infinity, all-bypass
and mixed-bypass rows, preservation of existing error bits, and invalid
pass rejection without output writes. Compute Sanitizer memcheck, racecheck,
synccheck, and initcheck each complete this test with zero errors/hazards.
The full suite retains exact-mode differential, allocation, failure-atomicity,
and concurrent-workflow checks.

Parent/candidate codestreams and reported final perceptual scores remain
identical for the small sample at efforts 7/9, odd 1080p/4K at effort 7,
and Flower at efforts 7/9, with final-score collection enabled. The pinned
`djxl` decodes all six at their source dimensions. Hashes remain those
recorded at the preceding checkpoints, and Flower selects all seven
production strategies.

Ignored artifacts under `build-cuda-ninja/profiles` use the `s22_` prefix.
They include parent and exploratory benchmark executables, parent encoder,
4K traces named `parent`, `specialized`, `fast_mirror`, `tiled`, `medium_tile`,
`large_tile`, and `retained`, plus parent/retained 1080p traces. Each trace
has `.nsys-rep` and `.sqlite` files. Paired results are
`s22_paired_{4k,1080p,flower}.json` and `s22_paired_1080p_warmed.json`;
`s22_compare_warmed.py` retains the longer-warmed protocol. Verification
uses `s22_verify.ps1`; sanitizer logs are `s22_epf_*.txt`.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples, with identical outputs at batch sizes 1, 2, and 4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 226.141 ms | 4.422 | 0.980x |
| Fully resident | 2 | 426.733 ms | 4.687 | 1.131x |
| Fully resident | 4 | 831.689 ms | 4.809 | 1.275x |
| Maximum throughput | 1 | 101.691 ms | 9.834 | 1.038x |
| Maximum throughput | 2 | 147.719 ms | 13.539 | 1.472x |
| Maximum throughput | 4 | 287.812 ms | 13.898 | 1.689x |

These are output-stability and overlap checks, not before/after batch
performance claims. Fully-resident batch-1 is still slower than serial.
Logs are `s22_batch_{resident,maximum}.txt`; the final GPU state is
69 C/P3/1282 MHz.

The retained 4K profile still has `128.497 ms` of DCT, `47.046 ms` of tiled
Butteraugli convolution, `42.634 ms` of Malta response, `26.186 ms` of resident
coefficient encoding, `20.788 ms` of AC gather, and `19.476 ms` of AC cost.
EPF is now `6.253 ms`, rather than a leading hotspot. DCT, AC gather/cost,
coefficient work, other neighborhood filters, host serialization, and transfer
boundaries remain open targets. The resident path is not yet maxed out.

### Follow-up: register-tile packed small DCTs (2026-09-05)

The fresh parent is `49d6707`, including the EPF optimization. Small packed
DCT shapes still consume `48.082 ms` in the odd-4K trace; 16x16 alone costs
`29.386 ms` across forward and inverse calls. Those kernels compute one
output per thread, repeating basis and sample loads across threads and
launching one block per 16x16 transform. The large-DCT register tiling from
the earlier checkpoint does not cover these shapes.

The retained small kernels accumulate eight independent outputs per thread
without changing any output's sequence of multiply-adds. A 256-thread block
now packs 32 DCT8, 16 DCT16x8, 16 DCT8x16, or 8 DCT16x16 transforms, versus
4/2/2/1 previously. Each horizontal basis load feeds eight accumulators;
each vertical intermediate sample feeds eight accumulators. Input and
coefficient-layout conversion use a padded shared tile with coalesced global
I/O. The forward kernel reuses the input tile for output conversion.

Sub-warp transforms receive padding between their intermediate arrays so
neighboring transforms do not map their same-index samples to the same
shared-memory banks. The 16-high shapes stage their vertical basis because
each warp addresses multiple basis entries. The 8-high shapes retain
constant-memory broadcasts: with eight accumulators, each warp now uses one
vertical basis address at a time. Inactive tail transforms initialize shared
cells and participate in the required barriers without accessing global
input/output. The inverse kernel can return inactive lanes after its second
barrier; forward lanes remain through the output-conversion barrier.

The dense basis, coefficient layout, scaling, and arithmetic order remain
unchanged. No factored transform, fast-math option, quality-policy change,
device allocation, transfer, or extra kernel launch is introduced. Large
specialized and generic DCT kernels are unchanged. Shared-memory use is
18,720 bytes for DCT8, 19,008 for DCT16x8, 19,808 for DCT8x16, and 19,072
for DCT16x16. Compiler-reported registers/thread are respectively 48/48,
48/56, 40/40, and 39/40 for forward/inverse, with zero local or stack bytes.
The changed block/resource balance is not a hardware-counter occupancy claim.

Exploratory odd-4K profiles use two warmups, one captured fully-resident
encode, distance 1.2, effort 7, and no final-score collection:

| Variant | Packed DCT | Unchanged large DCT | All kernels |
|---|---:|---:|---:|
| Parent, one accumulator | 48.082 ms | 75.021 ms | 401.359 ms |
| Two accumulators, original global I/O | 36.341 ms | 80.551 ms | 398.331 ms |
| Two accumulators, shared I/O | 35.094 ms | 73.872 ms | 395.022 ms |
| Four accumulators, shared I/O | 30.114 ms | 75.784 ms | 386.447 ms |
| Four accumulators, all vertical bases shared | 34.623 ms | 72.798 ms | 382.522 ms |
| Eight accumulators, sub-warp padding, all vertical bases shared | 26.226 ms | 72.279 ms | 367.628 ms |
| Eight accumulators, selective vertical bases, forced dot-loop unrolling | 30.414 ms | 73.957 ms | 379.043 ms |
| Retained, selective vertical bases, compiler-chosen dot-loop unrolling | 28.220 ms | 76.317 ms | 391.223 ms |

Sharing every vertical basis with four accumulators helps 8x16 inverse but
regresses the aggregate result, so it is not selected. Forced unrolling of
the dot-product loops also loses; only the independent-accumulator loops
are explicitly unrolled in the retained kernel. Individual captures are
noisy, so these trials select candidates rather than prove additive gains
for every sub-change.

The final packed-DCT reduction is `19.862 ms` (41.3%). All DCT falls from
`123.103` to `104.538 ms` (15.1%); unchanged large DCT rises 1.7%, while
non-DCT execution rises from `278.256` to `286.685 ms`. All-kernel time falls
2.5%, not 41.3%. Both traces contain 523 kernel launches, 117,079,320 HtoD
bytes, 103,699,012 DtoH bytes, and 518,400 device-to-device bytes. These
controls limit attribution of total-workflow variation to the changed code.

An initial 65,536-transform event probe shows large relative timing swings
for short kernels. The final probe instead uses 67,108,864 elements for every
shape: 1,048,576 DCT8 transforms, 524,288 8x16/16x8 transforms, and 262,144
16x16 transforms. It performs three warmup dispatches, then seven samples
of three dispatches each. Parent/candidate processes run in
parent-candidate-candidate-parent order. The following ranges span the two
process medians, not individual sample ranges or confidence intervals:

| Shape/direction | Parent median range | Retained median range |
|---|---:|---:|
| 8x8 forward | 7.438-7.767 ms | 2.160-2.165 ms |
| 8x8 inverse | 8.669-8.941 ms | 5.388-5.539 ms |
| 16x8 forward | 9.789-10.209 ms | 5.854-5.919 ms |
| 16x8 inverse | 10.129-10.620 ms | 5.508-5.575 ms |
| 8x16 forward | 8.970-9.142 ms | 6.152-6.260 ms |
| 8x16 inverse | 10.367-10.488 ms | 6.348-6.415 ms |
| 16x16 forward | 10.993-11.242 ms | 7.929-8.264 ms |
| 16x16 inverse | 13.094-16.615 ms | 7.468-7.542 ms |

The unchanged large-shape controls still vary; 32x32 inverse, for example,
has parent process medians of `11.715-12.889 ms` and candidate medians of
`13.485-14.348 ms`. The packed-shape gains survive that unfavorable control,
but absolute probe and workflow times should not be interchanged. The
retained workflow's 16x16 inverse improvement is also smaller than the
isolated-probe improvement (`16.480` to `12.723 ms`).

Public wall measurements use seven alternating independent-process pairs
per workload, three warmups and five measured encodes per process, comparing
each process's median. Fully-resident GPU-only mode, distance 1.2, effort 7,
and skipped final-score collection match the profile boundary. Cohort
medians and median within-pair changes are distinct statistics:

| Workload/stage | Parent median | Candidate median | Median paired change |
|---|---:|---:|---:|
| Odd 4K total | 990.519 ms | 943.740 ms | -2.6% |
| Odd 4K quantization pipeline | 694.657 ms | 653.460 ms | -4.4% |
| Odd 1080p total | 230.694 ms | 248.354 ms | +2.3% |
| Odd 1080p quantization pipeline | 147.469 ms | 144.065 ms | -1.2% |
| Flower total | 46.465 ms | 47.426 ms | +3.2% |
| Flower quantization pipeline | 28.823 ms | 28.824 ms | +0.1% |

All seven 4K quantization pairs and six total-time pairs improve. Parent
and candidate total process-median ranges are `917.113-1045.410 ms` and
`907.796-970.551 ms`. At 1080p, five quantization pairs improve but only
three total-time pairs do; total ranges are `225.565-256.775 ms` and
`221.650-260.331 ms`. Flower total ranges are `45.601-55.813 ms` and
`47.016-53.850 ms`; only two total-time pairs improve. Thus this checkpoint
establishes a 4K wall-time gain, not a universal encoding-time improvement.
The smaller-workload wall regressions are retained in the record.

Additional three-warmup, one-sample traces investigate those smaller
workloads. At 1080p, packed DCT falls from `8.135` to `4.710 ms`; unchanged
large DCT moves from `10.017` to `9.507 ms` and all kernels from `81.087`
to `71.388 ms`. For Flower, packed DCT falls from `1.204` to `0.752 ms`,
unchanged large DCT stays near `1.21 ms`, and all kernels fall from `17.880`
to `17.416 ms`. These fresh traces support retaining the small-shape kernel
change, but do not explain away total-time regressions measured in different
runs. Host/driver work and scheduling variance remain material, especially
when the absolute kernel saving is below one millisecond. GPU state moves
from 65 C/P8/210 MHz before the wall pairs to 75 C/P3/1575 MHz afterward.

All 54 CUDA-build tests and 47 CPU-only tests pass. The standalone CUDA DCT
test now covers 14 batch counts per supported shape: 1/2/3/4, 7/8/9,
15/16/17/19, and 31/32/33. That exercises full and partial blocks around all
packing sizes, including the 32-transform DCT8 block. Across nine shapes,
126 configurations check forward output, round trips, independent inverse
output, and concurrent/repeated submission waits. Random inputs and the
existing impulse, constant, checkerboard, and unequal-ramp fixtures remain.
The double-precision reference tolerances are unchanged: forward
`3e-5 + 3e-4 * abs(reference)` and inverse/round-trip
`5e-4 + 5e-4 * abs(reference)`.

Compute Sanitizer memcheck, racecheck, synccheck, and initcheck each complete
the expanded backend test with zero errors/hazards. As before,
`--report-api-errors no` suppresses reporting of the separate, deliberately
invalid zero-grid launch used to test stale-error consumption; memory and
synchronization checking remain enabled. Existing suite checks retain
exact-mode differentials, allocation, failure-atomicity, and concurrent
fully-resident/maximum-throughput workflow coverage.

Six parent/candidate codestreams, selected strategies, and reported final
perceptual scores remain identical: the small sample at efforts 7/9,
odd 1080p/4K at effort 7, and Flower at efforts 7/9. All collect the final
score, and the pinned `djxl` decodes each candidate at its source dimensions.
Hashes remain those of the preceding checkpoints; Flower still exercises
all seven production strategies.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples and preserves identical output at sizes 1, 2, and 4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 251.674 ms | 3.973 | 0.958x |
| Fully resident | 2 | 430.302 ms | 4.648 | 1.066x |
| Fully resident | 4 | 746.158 ms | 5.361 | 1.404x |
| Maximum throughput | 1 | 121.540 ms | 8.228 | 1.068x |
| Maximum throughput | 2 | 157.160 ms | 12.726 | 1.524x |
| Maximum throughput | 4 | 285.467 ms | 14.012 | 1.749x |

These are correctness and overlap checks, not before/after batch speedup
claims. Fully-resident batch-1 remains slower than serial. GPU state after
batch qualification is 70 C/P3/1282 MHz.

Ignored study artifacts use `build-cuda-ninja/profiles/s23_` prefixes.
The eight 4K trace names are `parent`, `register2`, `staged2`, `staged4`,
`vertical4`, `vertical8`, `unrolled8`, and `retained`; parent/retained traces
also cover `1080p` and `flower`. Each has `.nsys-rep` and `.sqlite` files.
Saved kernel sources and benchmark executables retain the explored variants.
`s23_dct_probe.cu` is the final equal-element probe, with final comparison
executables `s23_probe_parent_full.exe` and `s23_probe_retained.exe`, and
four `s23_probe_{1,2,3,4}_*.txt` logs. Earlier probe executables predate the
equal-element protocol and should not be used for that comparison.
`s23_compare_warmed.py` produces `s23_paired_{4k,1080p,flower}.json`;
`s23_verify.ps1` and `s23_identity.txt` record codestream/score/decode checks.
Sanitizer logs are `s23_dct_*.txt`; batch logs are
`s23_batch_{resident,maximum}.txt`.

The retained 4K profile still spends `76.317 ms` in large DCT,
`49.196 ms` in tiled Butteraugli convolution, `40.399 ms` in Malta response,
`25.818 ms` in resident coefficient encoding, `20.563 ms` in adjusted
quantization, `19.575 ms` in AC gather, and `19.553 ms` in AC cost. Larger
transforms, coefficient work, remaining filters, host work, and transfer
boundaries remain material targets. This is not a demonstrated performance
ceiling for the resident path.

### Follow-up: fuse AC candidate gathering into forward DCT (2026-09-05)

The parent is `1d72c2b`. AC search materializes each candidate's three
image rectangles in `scratch_a`, then immediately reads that buffer into
forward-DCT shared memory. The subsequent residual kernel overwrites all
active `scratch_a` elements, and cost evaluation consumes reconstructed
residual pixels, not the original packed pixels. The gathered intermediate
is therefore unnecessary. The seven 4K gather launches represent roughly
1.274 GB of packed pixels, or 2.55 GB of logical writes plus rereads. This
is an address-volume estimate, not a measured DRAM-counter result.

The retained forward kernels accept either a contiguous pointer or an
AC-candidate image source. The latter resolves the descriptor and channel
once per participating thread, then reads its strided rectangle directly
into the existing padded shared input tile. Packed/specialized geometry,
basis staging, output layout, and each output's multiply-add sequence are
unchanged. No lower-precision arithmetic or factored transform is introduced.
Candidate layout and validation now live in a shared CUDA-only header.
Invalid descriptors yield NaN without reading pixels; inactive packed
transforms do not fetch descriptors and still participate in barriers.

Both scratch allocations remain necessary for residual coefficients and
inverse reconstruction. Allocation sizes and host/device transfers do not
change. The optimization removes a buffer pass, not the scratch allocation.

One intermediate loader precomputed an offset contiguous pointer. NVCC
raised the ordinary forward 16x8 register count from 48 to 54 and 16x16
from 39 to 40. Retaining the original `input[base + index]` expression
restores all ordinary forward register counts. Fused forward counts are
48/48/40/40 for 8x8/16x8/8x16/16x16, and 40/39/40 for
32x16/16x32/32x32. All have zero local memory and stack; shared-memory
sizes are unchanged. Only fused 16x16 adds one register versus its parent.

#### GPU evidence and controls

Clock variation is large enough to invalidate isolated absolute comparisons.
The first saved parent trace totals 374.932 ms, whereas the first fused trace
totals 198.485 ms; that apparent near-halving is **not** attributed to fusion.
A fresh parent captured adjacent to the fused run provides a much closer
unchanged-kernel control. A later final-source pair provides a second check:

| 4K capture pair | Metric | Parent | Fused |
|---|---|---:|---:|
| Adjacent initial pair | Gather | 12.685 ms | 0 |
| Adjacent initial pair | All forward DCT | 24.365 ms | 25.964 ms |
| Adjacent initial pair | All inverse DCT | 25.181 ms | 25.117 ms |
| Adjacent initial pair | Gather + all DCT | 62.230 ms | 51.082 ms |
| Adjacent initial pair | Other kernels | 146.988 ms | 147.404 ms |
| Adjacent initial pair | All kernels | 209.218 ms | 198.485 ms |
| Final-source pair | Gather + all DCT | 130.173 ms | 103.036 ms |
| Final-source pair | Other kernels | 267.850 ms | 275.755 ms |
| Final-source pair | All kernels | 398.023 ms | 378.791 ms |

The initial pair reduces gather plus DCT by 17.9% and all kernels by 5.1%,
with other kernels within 0.3%. The final pair reduces those totals by
20.8% and 4.8%, respectively, but its other kernels increase 3.0%.
The initial fused trace predates the ordinary-pointer expression cleanup;
the final-source pair includes it. Both pairs eliminate exactly seven
launches, from 523 to 516, and retain 117,079,320 HtoD bytes,
103,699,012 DtoH bytes, and 518,400 D2D bytes.

Final-source diagnostic pairs also reduce gather plus DCT from 18.580 to
13.444 ms at 1080p and 2.431 to 1.879 ms on Flower. However, their other
kernels decrease 23.8% and 5.5%, respectively. Those are not controlled
estimates of an end-to-end gain, nor evidence that fusion explains every
observed DCT reduction.

A separate same-process CUDA-event probe compares the parent's gather body
plus the ordinary forward DCT against fused forward DCT. It uses deterministic
image data, legal synthetic candidate positions, seven alternating-order
pairs after three warmups, and three dispatch repetitions per measurement
(64 for 33-candidate batches). Events measure the device timeline, including
launch gaps, rather than host workflow latency. Counts are 33, 4096, and the
large per-shape counts inferred from the 4K trace. Every coefficient is checked
bitwise before timing. Two independent processes each pass all 21 checks and
favor fusion in every configuration's median paired ratio.

| Shape | Large candidate count | Paired gather + forward time reduction, two runs |
|---|---:|---:|
| 8x8 | 129600 | 51.6-52.8% |
| 16x8 | 113280 | 57.8-58.4% |
| 8x16 | 113400 | 31.9-32.1% |
| 16x16 | 99120 | 31.1-32.1% |
| 32x16 | 24240 | 18.2-19.0% |
| 16x32 | 24300 | 6.2-11.9% |
| 32x32 | 18180 | 29.9-30.2% |

The 33-candidate reductions range from 14.8% to 50.1%, and the 4096-candidate
reductions from 21.8% to 48.9%. Individual event samples still vary substantially;
these synthetic timings isolate the two implementations, not production
search locality. They support retaining fusion for every shape without a
small-batch fallback, but do not establish a Flower workflow latency win.

#### Warmed public workflow

Seven alternating parent/candidate process pairs use three warmups and five
samples per process, effort 7, distance 1.2, fully resident, no final score.
Each process contributes its median. All pairs, including outliers, are kept.
Negative paired changes mean faster:

| Workload | Total median, parent / fused | Median paired total change | Quantization median, parent / fused | Median paired quantization change |
|---|---:|---:|---:|---:|
| Odd 4K | 639.041 / 653.210 ms | -3.7% | 407.479 / 396.414 ms | -3.4% |
| Odd 1080p | 203.917 / 198.267 ms | -1.1% | 115.748 / 111.874 ms | -3.9% |
| Flower | 44.179 / 44.806 ms | +2.6% | 27.770 / 27.605 ms | +0.5% |

Medians of cohorts and medians of paired ratios are different statistics;
the 4K total medians move in the opposite direction to the paired ratio.
4K improves in five of seven total-time pairs and six quantization pairs;
1080p improves in six pairs for both metrics. Flower improves in only one
total-time pair and three quantization pairs. Its measured total-time
regression remains part of the result, not a discarded inconvenient sample.
4K parent/fused total ranges are 618.465-714.695 / 593.595-821.915 ms;
quantization ranges are 405.481-423.054 / 389.839-449.741 ms. GPU state during
the 4K sweep reached 78 C/P0/1395 MHz graphics/6000 MHz memory. Clocks were
not locked; absolute times must not be compared to preceding checkpoints.

#### Qualification and artifacts

The final build passes all 54 CUDA tests and the CPU-only build passes all
47 tests. AC coverage now checks 231 CPU-referenced candidate costs, including
both image corners, twelve prefix lengths crossing packed-DCT boundaries,
and exact per-candidate/output-guard identity. Padded rows, plane gaps,
nonzero offsets, reordered resident planes, and NaN input guards are tested
against contiguous costs. Ten invalid descriptor cases cover coordinate
overflow/footprint bounds, quant norm, entropy multiplier, and CfL factors;
unrelated candidates must remain valid. Resident CfL maps must supersede
non-finite descriptor factors. Resident quant norms, scratch alias rejection,
and independent submission waits remain covered.

Both AC-strategy and general CUDA-backend tests pass memcheck, racecheck,
synccheck, and initcheck with zero errors/hazards. Only the general backend
test uses `--report-api-errors no`, for its intentional invalid zero-grid
launch; memory/synchronization checking remains active. Its 126 ordinary
DCT configurations still check CPU-reference forward, roundtrip, and
independent inverse results.

Six parent/fused codestreams, strategy summaries, and final perceptual scores
remain identical: sample efforts 7/9, odd 1080p/4K effort 7, and Flower efforts
7/9. The pinned decoder reads every candidate at its original dimensions;
hashes remain those of prior checkpoints. Flower covers all seven strategies.

Post-change 1080p batch qualification uses one warmup and three paired
serial/batch samples, preserving identical output at sizes 1/2/4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 222.608 ms | 4.492 | 0.980x |
| Fully resident | 2 | 382.894 ms | 5.223 | 1.265x |
| Fully resident | 4 | 777.363 ms | 5.146 | 1.333x |
| Maximum throughput | 1 | 112.129 ms | 8.918 | 1.028x |
| Maximum throughput | 2 | 165.620 ms | 12.076 | 1.468x |
| Maximum throughput | 4 | 300.619 ms | 13.306 | 1.719x |

These qualify correctness and overlap, not before/after batch performance.
Fully-resident batch-1 remains slower than serial. Ending GPU state is
70 C/P3/1282 MHz graphics/5500 MHz memory.

Ignored artifacts use `build-cuda-ninja/profiles/s24_` prefixes. Initial
traces are `parent_4k` (unmatched control), `parent_now_4k`, and `fused_4k`;
final traces are `final_{parent,retained}_{4k,1080p,flower}`, each with
`.nsys-rep` and `.sqlite` files. `profile_final.ps1` and
`profile_summary.py` reproduce capture and analysis. `compare_warmed.py`
produces `warmed_{4k,1080p,flower}.json` with raw process output.
`gather_dct_probe.cu` and `gather_probe_{1,2}.txt` retain the event comparison.
Compile the probe against `gjxl_cuda.lib` with NVCC CUDA 11.8,
`-std=c++17 -O3 -arch=sm_86 -Xcompiler=/MD --cudart=shared -I src`,
under the MSVC 14.37 developer environment. `verify.ps1`/`identity.txt`,
`{ac,dct}_{memcheck,racecheck,synccheck,initcheck}.txt`, and
`batch_{fully-resident,maximum-throughput}.txt` retain qualification results.
All artifact names in this paragraph include the `s24_` prefix.

The final 4K trace still spends 78.452 ms in large DCTs, 49.604 ms in tiled
convolution, 43.977 ms in Malta, 27.463 ms in resident coefficient encoding,
22.037 ms in adjusted quantization, and 19.121 ms in AC cost. Cost reduction
and remaining coefficient/filter work, plus host work and transfer boundaries,
remain substantial targets. This checkpoint does not demonstrate that the
fully-resident path is maxed out.

### Follow-up: pack and register-reduce AC costs (2026-09-05)

The parent is `255a706`. After gather fusion, AC cost still reads every
inverse-transformed residual pixel. Its original kernel launches one block
per candidate with 64/128/256/512/1024 threads, stages all three channels in
shared memory, and executes a block-wide barrier after every halving level.
Increasingly few threads participate as the reduction shrinks. At 32x32,
the kernel uses 1024 threads, 12 KiB of shared memory, and eleven barriers
for one cost. Runtime transform dimensions also require general integer
division/remainder for mask addressing.

The retained kernel specializes width/height and packs candidates into
256-thread blocks. Candidate groups have 32 threads through 16x16, 64 for
the 512-pixel shapes, and 128 for 32x32; each lane holds two, four, or eight
pixels for all three channels. The original halving order first combines
register-held values, then uses shared memory between warps where needed,
and finishes with five shuffle levels. Small shapes need no shared memory
or block barrier; 512/1024-pixel shapes use 3 KiB and two/three barriers.
Inactive tail candidates and invalid footprints participate safely without
reading pixels or writing outside the cost range. The allocation, transfer,
launch-count, and arithmetic-precision contracts do not change.

An initial eight-value tile reduces 4K cost execution from 17.074 to
6.089 ms, with other kernels moving from 344.337 to 339.071 ms. A sixteen-value
tile takes 6.601 ms despite lower other-kernel time (330.242 ms). Its large
shapes consume 84-86 registers instead of 40. A same-process comparison also
tests four-value tiles: they do not offer a consistent advantage over eight,
while sixteen is consistently worse on the large shapes. The retained final
register counts are 24/30/30/40/40/40/40 for physical widths/heights
8x8/16x8/8x16/16x16/32x16/16x32/32x32, with no stack or local-memory use.

#### Numerical pitfall caught by the direct comparison

The first packed version passes CPU-tolerance and batch-consistency tests,
but a direct parent-kernel comparison finds occasional one-ULP cost differences
(first observed at 16x8 with 4096 candidates). Reverting constant-count
normalization to the parent's runtime-count expression does not fix them.
PTX identifies the difference in channel combination, not the halving tree:
the parent contracts X weighting with the following Y addition into one
FMA, whereas register-resident channel totals cause NVCC to emit a multiply
and a separately rounded addition across the branch. The same source-level
summation order alone is insufficient to preserve the parent's result.

The final channel combination explicitly uses the parent's FMA sequence,
including weighted X entropy plus Y magnitude. A fresh direct probe compares
the original cost body with four/eight/sixteen-value variants across all seven
shapes and counts 33, 4096, and the production-size counts from the 4K trace.
All 63 variant/configuration comparisons pass bitwise in each of two
independent processes, including output tail guards. The probe uses strided
mask input, deterministic residuals and rates, and varying positive quant
norms/entropy multipliers. Its timing phase gives every implementation the
same warmup and repetition counts; each has independent in/out cost storage.
During timing, each cost becomes the following dispatch's norm. That feedback
is identical across implementations and keeps the timing loop focused on
the cost kernel; it does not model an AQ iteration.

Seven within-process forward/reverse-order sweeps follow three warmups,
using five repetitions per measurement (64 for 33-candidate batches).
Events measure the device timeline, including dispatch gaps. The retained
eight-value variant wins every configuration's median paired comparison in
both processes. Large-count reductions range from 62.9% to 83.4%, while
4096-candidate reductions range from 34.6% to 75.1%. The small 33-candidate
results range from 2.1% to 40.5% and are more sensitive to dispatch/clock
variation. Absolute timings and variant-order differences remain noisy;
these are isolated cost comparisons, not complete workflow speedups.

#### Warmed public workflow

Seven alternating parent/candidate process pairs use three warmups and five
samples per process, effort 7, distance 1.2, fully resident, no final score.
Each process contributes its median; all pairs and outliers are retained.
Negative paired changes mean faster:

| Workload | Total median, parent / packed | Median paired total change | Quantization median, parent / packed | Median paired quantization change |
|---|---:|---:|---:|---:|
| Odd 4K | 876.867 / 856.186 ms | +1.5% | 611.071 / 596.081 ms | -0.9% |
| Odd 1080p | 210.995 / 213.637 ms | +0.6% | 131.949 / 131.110 ms | -2.0% |
| Flower | 54.761 / 46.620 ms | -5.9% | 32.898 / 28.287 ms | -5.1% |

There is no demonstrated total-latency win on the large synthetic workloads:
only two of seven 4K total-time pairs and three 1080p pairs improve. Their
quantization stages improve in six and five pairs, respectively. Flower
improves in five total-time pairs and all seven quantization pairs, but its
parent total range is particularly broad (43.597-78.558 ms versus
44.508-55.674 ms). Its larger percentage must not be attributed entirely to
the cost kernel without an unchanged-work control.

The 4K total cohorts range from 826.371-916.619 / 839.592-948.853 ms;
quantization ranges are 579.212-625.331 / 586.411-614.051 ms. The cohort
medians and median paired ratios are different statistics: the former move
in the opposite direction to the 4K total paired ratio. During the 4K sweep
a device-state sample reads 71 C/P3/667 MHz graphics/5500 MHz memory.
Clocks are not locked, and preceding checkpoints' absolute times are not a
valid control. The isolated kernel improvement is retained, but this is not
claimed as an established 4K end-to-end speedup.

#### Final-source profiles and qualification

Adjacent parent/final-source traces use three warmups and one profiled sample:

| Workload | Cost, parent / packed | Other kernels, parent / packed | All kernels, parent / packed |
|---|---:|---:|---:|
| Odd 4K | 19.609 / 6.220 ms | 391.858 / 384.748 ms | 411.467 / 390.968 ms |
| Odd 1080p | 2.955 / 1.571 ms | 67.746 / 71.878 ms | 70.700 / 73.449 ms |
| Flower | 0.440 / 0.268 ms | 16.542 / 16.545 ms | 16.982 / 16.813 ms |

Cost execution decreases 68.3%, 46.8%, and 39.1%, respectively. At 4K, all
kernels decrease 5.0%, but other kernels also decrease 1.8%; the full change
is not attributed solely to cost packing. At 1080p, other kernels increase
6.1% and total kernel time increases 3.9% despite the cheaper cost stage.
Flower's other kernels are nearly level, with total kernel time decreasing
1.0%, much less than its paired workflow change. These controls reinforce
the distinction between a kernel gain and a universal latency gain.

The seven 4K cost launches now issue 78,150 blocks instead of 522,120.
Both complete traces still have 516 launches and transfer 117,079,320 bytes
HtoD, 103,699,012 bytes DtoH, and 518,400 bytes D2D. 1080p remains at 501
launches and Flower at 516. No scratch allocation or residency policy changes.

The final CUDA build passes all 54 tests; the CPU-only build passes all 47.
The AC fixture now checks independent padded mask rows and nonzero mask
offsets in legacy and resident modes, exact cost equality with contiguous
inputs, and untouched input/guard storage. Eighty-four bad-mask batches
cover zero, negative, NaN, and infinity values at the first, last, and an
interior pixel across all seven strategies. Each candidate must return NaN
iff it covers that pixel, otherwise retain its original cost. Existing
231 CPU-referenced costs, twelve prefix lengths, invalid descriptors,
resident quant norms/CfL, and independent-submission checks remain active.

After the FMA correction, the expanded AC test passes memcheck, racecheck,
synccheck, and initcheck with zero errors or hazards, without suppressing
API errors. Six parent/final codestreams, selected-strategy summaries, and
final perceptual scores remain identical: sample efforts 7/9, odd 1080p/4K
effort 7, and Flower efforts 7/9. The pinned decoder reads every output at
its original dimensions. Hashes remain those of the preceding checkpoints.

Post-change batch qualification at 1080p uses one warmup and three paired
serial/batch samples and preserves identical output at sizes 1/2/4:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 254.816 ms | 3.924 | 0.993x |
| Fully resident | 2 | 403.184 ms | 4.961 | 1.204x |
| Fully resident | 4 | 818.508 ms | 4.887 | 1.163x |
| Maximum throughput | 1 | 110.644 ms | 9.038 | 1.025x |
| Maximum throughput | 2 | 155.350 ms | 12.874 | 1.459x |
| Maximum throughput | 4 | 343.143 ms | 11.657 | 1.673x |

These qualify correctness and overlap, not before/after batch performance.
Ending GPU state is 74 C/P3/1282 MHz graphics/5500 MHz memory.

Ignored artifacts under `build-cuda-ninja/profiles/` use the `s25_` prefix:
`parent_4k`, `packed8_4k`, and `packed16_4k` retain the initial traces;
`final_{parent,retained}_{4k,1080p,flower}` retain the final traces, all as
`.nsys-rep`/`.sqlite`. `packed{8,16}_source.cu` are **pre-FMA-correction**
experiments, not the retained source. `cost_probe.cu` is the corrected
four/eight/sixteen-value probe, with `cost_probe_fma_{1,2}.txt` results.
`cost_probe_constant_count.txt`, `cost_probe_runtime_count.txt`, and
`cost_probe_before_fma.ptx` retain the failed numerical investigation.
The probe compiles with NVCC 11.8 under MSVC 14.37 using
`-std=c++17 -O3 -arch=sm_86 -Xcompiler=/MD --cudart=shared -I src`.
`compare_warmed.py` produces `warmed_{4k,1080p,flower}.json` with raw output;
`verify.ps1`/`identity.txt`, `ac_final_{memcheck,racecheck,synccheck,initcheck}.txt`,
and `batch_{fully-resident,maximum-throughput}.txt` retain final qualification.
`qualify.ps1` runs the benchmarks and GPU checks sequentially;
`profile_final.ps1`/`profile_summary.py` reproduce capture and extraction.
Every artifact name in this paragraph includes the `s25_` prefix.

The cost stage still logically reads 1.274 GB of residual pixels plus
0.425 GB of masks per 4K encode; overlapping mask reads can hit cache, so
this is not a DRAM-counter measurement or proof of a bandwidth ceiling.
Further packing has diminishing returns. A next dataflow experiment should
reduce weighted residual loss directly from inverse-DCT outputs and retain
only per-channel sums, removing the full residual-pixel buffer round trip.
It must measure any added inverse-kernel register/barrier cost and preserve
the now-explicit final FMA behavior. The final trace also retains 82.046 ms
of large DCT, 26.048 ms of packed DCT, 55.177 ms of convolution, 48.878 ms
of Malta, 29.036 ms of coefficient encoding, and 23.318 ms of adjusted
quantization. Host work and transfer boundaries remain important given the
mixed wall-time results. The fully-resident path is not demonstrated maxed out.

### Follow-up: fuse AC inverse transforms with residual loss (2026-09-05)

Parent: `85d7f42` (packed AC costs). AC search only needs a weighted
eighth-power loss sum for each candidate/channel, but the parent writes
every inverse residual pixel to global scratch and then reads it back in
the cost kernel. Packing the cost kernel did not eliminate that dataflow.

`AcStrategyDctLossOutput` now consumes the inverse transform's output
registers. The same templated inverse kernels retain their ordinary pointer
output for reconstruction and transform callers. The AC sink preserves the
inverse scaling's FP32 rounding explicitly, applies the mask/channel offset,
squares three times, and reduces with the original halving tree. It writes
only three floats per candidate. `FinalizeCostKernel` combines those sums
with the existing channel rates and cached quant norm, retaining the explicit
final FMAs introduced in the parent.

Small shapes reduce eight values per lane in registers and then use exact
8/16/32-lane shuffle masks. Inactive tail transforms do not participate in
those groups. Large shapes reuse the dead horizontal-basis shared storage:
after one barrier, the first warp loads eight warp-partials per lane and
evaluates the 128/64/32 halving tree in registers, followed by warp shuffles.
The horizontal-basis reads have ended at an earlier block barrier; the
remaining inverse work reads different shared regions. No extra shared
allocation is required. Invalid mask values or footprints still yield NaN.

Both large scratch buffers remain necessary for the forward/residual stages;
this is a traffic reduction, not an allocation-size reduction. At padded 4K,
522,120 candidates formerly wrote and reread 2,547,671,040 bytes of residual
pixels. The new loss-sum round trip is 12,530,880 bytes. Unlike the old
three-channel cost loop, separate inverse channels logically reread the
mask, adding 849,223,680 bytes of mask loads. The net logical reduction is
therefore 1,685,916,480 bytes, not the entire removed residual-buffer traffic.
Overlapping/coalesced mask accesses can hit cache; these are algorithmic byte
counts, not measured DRAM traffic or a bandwidth-ceiling claim.

Register counts remain 48/56/40/40 for inverse shapes 8x8/16x8/8x16/16x16 and
40/40/40 for 32x16/16x32/32x32, identical to the ordinary inverse kernels.
There are no stack/local spills, and dynamic shared-memory sizes are unchanged.
The finalizer uses 22 registers and no shared memory. Kernel launches and
host/device transfers are not removed: seven small finalizer launches replace
the seven old cost launches.

#### Experiments and numerical checks

The first fused implementation used four additional large-transform barriers
for shared-memory halving. Its fresh 4K trace reduced the AC inverse/cost
stages from 48.882 to 41.081 ms; other kernels also improved 3.0%, so the
full 4.8% kernel-total reduction was not wholly attributable to fusion.
A later trace of that version measured 52.631 to 44.458 ms for the targeted
stages (-15.5%), with other kernels nearly unchanged (+0.5%). Its paired
wall-time results were mixed: quantization improved 1.7%/1.8%/2.4% at
4K/1080p/Flower, while total time changed +2.0%/-1.5%/-1.0%.

The second implementation moves the cross-warp halving into first-warp
registers, removing three barriers. Disassembly of the normal builds confirms
identical instructions for all four small inverse-loss shapes; large shapes
have three total barriers instead of six, with unchanged instruction counts
and register footprints. This is an arithmetic-order-preserving reduction,
not a reassociated warp-first sum.

Two isolated processes compare the parent inverse plus packed cost, the first
fusion, and the one-barrier reduction at seven shapes and candidate counts
33, 4,096, and approximately the production 4K counts. All final costs and
output guards match bitwise, both initially and after the timed iterations.
The two-variant probe also passes all 21 configurations in both processes.
Timing uses seven alternating event-timed pairs, three warmups, and 64 repeats
for count 33 or five for larger counts. As in the preceding cost probe, each
timed result becomes the next quant norm equally for each implementation;
these iterations measure the kernel sequence, not an AQ iteration.

The three-way probe favors the one-barrier variant on large production-sized
batches, but even unchanged small-shape controls move substantially. At 4,096
candidates the small controls are much steadier and the large-shape benefit
is modest. Do not interpret the largest isolated ratios as causal speedups;
normal-build profiles and public-workflow measurements are the primary check.

`cuda_backend_test` now directly compares fused channel sums with a materialized
ordinary CUDA inverse followed by an independent host FP32 halving tree.
Seven shapes, 12 candidate counts (1/2/3/5/7/8/9/10/11/16/17/33), and five mask
variants exercise 420 fused batches and 12,810 channel-sum comparisons. The
inputs include noise, zeros, and impulses; mask rows have padding and a
nonzero offset; output ranges have prefix/suffix guards. Image corners,
overlapping tiles, out-of-bounds/maximum coordinates, and zero/negative/NaN/
infinite masks are covered. Finite results require bitwise equality, while
invalid results require NaN. The existing 126 ordinary DCT configurations
and expanded AC-cost invalid-input/resident-view tests remain active.

#### Retained-build measurements

Final one-barrier build versus `85d7f42`, effort 7, distance 1.2,
fully resident, no optional final-score pass. Each Nsight range captures one
encode after three warmups. Times are GPU execution milliseconds; AC inverse
and cost are measured together so that moving work into the inverse is not
misreported as a standalone cost-kernel speedup.

| Workload | AC inverse/cost parent | Final | Change | Other kernels parent/final | Total kernels parent/final |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 49.465 | 44.159 | -10.7% | 331.312 / 332.473 | 380.777 / 376.631 |
| Odd padded 1080p | 7.177 | 5.991 | -16.5% | 63.145 / 64.869 | 70.322 / 70.860 |
| Flower | 1.086 | 0.860 | -20.8% | 15.705 / 15.720 | 16.792 / 16.580 |

The 4K total improves 1.1% with other kernels within 0.4%; Flower improves
1.3% with its other kernels within 0.1%. At 1080p, a 2.7% increase in other
kernel time exceeds the targeted saving. Launch counts remain 516/501/516
for 4K/1080p/Flower. The 4K trace retains 117,079,320 host-to-device bytes
in 31 copies, 103,699,012 device-to-host bytes in 19 copies, and 518,400
device-to-device bytes in one copy. None of that transfer traffic is removed.
The final trace still contains 79.528 ms of large DCT (including fused loss),
21.222 ms of packed DCT, 68.752 ms of convolution, 62.571 ms of Malta,
30.148 ms of resident coefficient encoding, and 23.699 ms of adjusted
quantization.

Six of seven shape-specific AC inverse/cost pairs improve in the final 4K
trace; 16x32 increases from 13.628 to 14.131 ms (+3.7%). Thus the aggregate
gain is not a demonstrated universal per-shape win. The earlier fused trace
and isolated comparisons do not show a consistent 16x32 regression.

The original `s26_v2_warmed_4k` run overlapped an unexpectedly long CPU
disassembly check. Its complete log is retained, but it is not used for
performance conclusions; `s26_v2_clean_warmed_4k` repeats the entire experiment
without concurrent profiling, disassembly, builds, or tests. The uncontended
1080p/Flower runs remain in `s26_v2_warmed_{1080p,flower}`. Every reported
wall comparison uses seven alternating independent-process pairs, three
warmups and five samples per process, and the median of each process's samples.
All outliers within these uncontended experiments are retained.

| Workload | Total median parent/final | Paired total change | Quantization median parent/final | Paired quantization change | Faster total/quantization pairs |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 1065.965 / 1027.257 ms | -2.6% | 669.518 / 646.048 ms | -3.5% | 4/7, 7/7 |
| Odd padded 1080p | 230.436 / 223.762 ms | -3.2% | 138.728 / 133.548 ms | -3.1% | 6/7, 5/7 |
| Flower | 50.419 / 48.277 ms | -0.7% | 29.961 / 29.195 ms | -2.6% | 4/7, 5/7 |

The paired changes are medians of paired ratios, not ratios of the cohort
medians. Total-time ranges are 966.404-1153.517 / 940.211-1185.483 ms at 4K,
224.418-241.990 / 217.084-246.075 ms at 1080p, and 46.605-63.119 /
47.310-62.713 ms for Flower (parent/final). The 4K quantization improvement
is consistent across all pairs, but total-time wins are not universal and
the observed benefit is modest. Ending clean-run GPU state is
77 C/P3/1282 MHz graphics/5500 MHz memory. The uncontended 4K repeat is
reproduced by `s26_v2_compare_warmed.py` with the saved parent benchmark,
the final benchmark, workload `padded_4k`, and output
`s26_v2_clean_warmed_4k.json`.

#### Validation and remaining work

The final build passes all 54 CUDA tests; the CPU-only build passes all 47.
Both the AC-strategy test and expanded CUDA backend test pass memcheck,
racecheck, synccheck, and initcheck with zero errors/hazards. Only the backend
test uses `--report-api-errors no`, because its pre-existing launch-error test
deliberately submits a zero-grid launch. No AC-test API errors are suppressed.
Six final codestreams remain identical to the parent, as do selected strategies
and final perceptual scores: sample efforts 7/9, odd 1080p/4K effort 7, and
Flower efforts 7/9. All decode with the pinned decoder at the original
dimensions; hashes remain those recorded in earlier checkpoints.

Final 1080p serial/batch qualification uses one warmup and three alternating
samples per size. All batch outputs remain identical to serial outputs.

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 268.690 ms | 3.722 | 0.980x |
| Fully resident | 2 | 532.634 ms | 3.755 | 0.983x |
| Fully resident | 4 | 975.005 ms | 4.103 | 1.525x |
| Maximum throughput | 1 | 243.101 ms | 4.114 | 1.023x |
| Maximum throughput | 2 | 369.940 ms | 5.406 | 1.379x |
| Maximum throughput | 4 | 447.753 ms | 8.933 | 2.048x |

These qualify correctness and overlap, not before/after batch performance.
Timing variance remains substantial, including no demonstrated two-image
fully-resident throughput benefit in this run. Ending batch GPU state is
75 C/P5/765 MHz graphics/810 MHz memory.

The preceding residual-coefficient buffer is still materialized between AC
quantization/rate evaluation and the inverse transform. A next dataflow
experiment can quantize residual coefficients while loading inverse-DCT shared
memory, reduce channel rates before the transform, and write loss summaries
to the opposite scratch buffer. That could remove another seven launches and
the residual-coefficient round trip, but requires explicit checks of reduction
order, buffer aliasing, invalid descriptors, and register/shared-memory costs.
Large direct-matrix DCTs, filtering, host work, and transfers also remain
material targets. This checkpoint does not demonstrate a maxed-out encoder.

Ignored artifacts use `build-cuda-ninja/profiles/s26_`:
`parent_4k`/`fused_4k` are the initial traces; `final_{parent,retained}_*`,
`warmed_*`, `ac_final_*`, `backend_final_*`, `identity.txt`, and `batch_*`
describe the first four-barrier fusion, not the final reduction.
`inverse_loss_v1_source.cu` and `fused_v1_{benchmark,encode}.exe` preserve that
version. `inverse_loss_probe.cu`/`inverse_loss_probe_{1,2}.txt` compare it with
the parent. `inverse_loss_variant.cu`/`inverse_loss_variants_probe.cu` and
`inverse_loss_variants_probe_{1,2}.txt` preserve the three-way experiment.
The probes compile with NVCC 11.8/MSVC 14.37 using
`-std=c++17 -O3 -arch=sm_86 -Xcompiler=/MD --cudart=shared -I src` and link
`gjxl_cuda.lib`; the three-way probe also compiles the variant source.
Final one-barrier artifacts use `s26_v2_`: `qualify.ps1` runs warmed pairs,
eight sanitizer passes, encode/decode identity, final profiles, and both batch
modes sequentially. `s26_profile_summary.py` extracts the first seven AC
inverse/cost pairs separately from later ordinary reconstruction inverses.

### Follow-up: fuse AC residual evaluation into inverse loading (2026-09-05)

Parent: `b46bea9` (inverse/loss fusion). AC search still materialized the
quantized residual coefficients between its standalone residual/rate kernel
and inverse transform. The fresh parent 4K trace spends 12.070 ms in that
residual kernel and 53.862 ms across residual, inverse, and final cost stages.
Removing only the residual launch would miscount any work moved into inverse
loading, so the comparisons below measure the complete affected sequence.

`AcStrategyResidualDctSource` now computes color decorrelation, quantization,
round-away-from-zero, and residual dequantization while loading the inverse
shared tile. It reduces magnitude and nonzero rates before the DCT starts.
The inverse transform and the preceding checkpoint's weighted loss sink are
shared with the ordinary pointer-input path; that path retains its original
load loop and transform arithmetic. Device candidate validation and the
host/device CfL rules are unchanged. The device channel-rate layout and CfL
helper move to the shared CUDA AC header.

Small transforms keep their rate arrays in registers and use exact subwarp
shuffle masks. Large transforms borrow the not-yet-populated intermediate
tile for float magnitude and integer count partials. One extra barrier makes
those writes visible; the first warp evaluates the original high-stride
halving tree and shuffles the final levels. The existing input-tile barrier
ensures rate reads have ended before horizontal DCT output can overwrite that
storage. No extra dynamic shared-memory allocation is required.

The batch now runs quant-norm preparation, forward DCT, fused residual/inverse/
loss, and final cost. Forward coefficients stay in scratch B; compact loss
sums go to scratch A. Writing them to B would alias coefficients that other
inverse blocks might still read. Existing public range/overlap validation is
unchanged. Seven standalone residual launches disappear, along with
2,547,671,040 bytes of full residual-coefficient writes/reads at padded 4K.
Scalar metadata loads are now per inverse lane instead of one leader per
old residual group, with same-address coalescing/cache reuse; the removed
buffer-byte count is not a net DRAM-traffic or bandwidth-ceiling measurement.

Resource usage for fused inverse input/output, in width-by-height order:

| Shape | Parent registers | Fused registers | Stack/local spill bytes |
|---|---:|---:|---:|
| 8x8 | 48 | 48 | 0 / 0 |
| 16x8 | 56 | 52 | 0 / 0 |
| 8x16 | 40 | 47 | 0 / 0 |
| 16x16 | 40 | 47 | 0 / 0 |
| 32x16 | 40 | 40 | 0 / 0 |
| 16x32 | 40 | 39 | 0 / 0 |
| 32x32 | 40 | 39 | 0 / 0 |

The first fused 4K trace measures 53.862 to 40.989 ms for the affected stages
(-23.9%), with other kernels essentially unchanged (283.662 to 283.832 ms).
The kernel total falls from 337.524 to 324.822 ms (-3.8%), and launches fall
from 516 to 509. This is an initial trace, not the final wall-time result.

#### Direct numerical checks

`CheckResidualInverseLoss` extends the CUDA backend test with an independent
host FP32 residual/rate oracle. The host forms residual coefficients, then
the existing pointer-input inverse/loss path supplies reference loss sums;
the new fused path receives the original coefficients. Magnitude sums and
nonzero counts are checked independently of the final scalar cost.

Seven shapes, candidate counts 1/8/11/16/32/33, and both host/device CfL modes
cover 84 fused batches and 4,242 channel comparisons each for loss, magnitude,
and nonzero count. Inputs include random coefficients, exact positive/negative
half-way values, signed zero, varying matrices/norms, invalid footprints,
strided/offset masks and device CfL planes, and prefix/suffix output guards.
Device CfL must supersede deliberately NaN host factors. Finite float results
must match bitwise, invalid results must be NaN, and counts must match exactly.
The earlier 420 direct inverse/loss batches, 126 ordinary DCT configurations,
and expanded AC-cost validation tests also remain active.

The initial oracle used separately rounded multiply/add for the device B
factor and failed an exact loss comparison. The existing CUDA factor expression
contracts that operation; using `std::fma` in the host oracle fixes the test
without changing the kernel formula. All direct comparisons then pass.

#### Paired public-workflow timing

Seven alternating parent/final process pairs, three warmups and five measured
samples per process, fully resident, effort 7/distance 1.2, no final-score
diagnostic. Each process contributes its sample median; all outliers are
retained. No GPU tests, builds, or disassembly checks overlap these runs.

| Workload | Total median parent/final | Paired total change | Quantization median parent/final | Paired quantization change | Faster total/quantization pairs |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 863.886 / 810.982 ms | -6.1% | 582.479 / 557.841 ms | -4.2% | 7/7, 7/7 |
| Odd padded 1080p | 212.853 / 207.994 ms | -0.9% | 130.792 / 127.351 ms | -3.9% | 6/7, 7/7 |
| Flower | 44.449 / 44.299 ms | +2.0% | 27.030 / 26.719 ms | -0.3% | 2/7, 4/7 |

Paired percentages are medians of per-pair ratios, not ratios of cohort medians.
Total-time ranges are 818.522-893.568 / 793.445-833.494 ms at 4K,
206.170-227.782 / 202.230-225.643 ms at 1080p, and 43.024-45.073 /
43.890-54.217 ms for Flower (parent/final). The large-image quantization
improvement is consistent across every pair, but the natural-image total
regression remains in the report. This is not a universal latency win.

#### Final profiles and qualification

Each final profile captures one effort-7, distance-1.2 fully-resident encode
after three warmups, without the optional final-score pass. GPU times are
milliseconds and include the entire affected residual/inverse/cost sequence.

| Workload | AC stages parent/final | Change | Other kernels parent/final | Kernel total parent/final | Launches parent/final |
|---|---:|---:|---:|---:|---:|
| Odd padded 4K | 56.439 / 43.304 | -23.3% | 319.454 / 297.916 | 375.893 / 341.220 | 516 / 509 |
| Odd padded 1080p | 8.829 / 7.054 | -20.1% | 59.129 / 56.244 | 67.958 / 63.298 | 501 / 494 |
| Flower | 1.276 / 1.025 | -19.6% | 15.232 / 15.285 | 16.508 / 16.310 | 516 / 509 |

All seven 4K shape pairs improve in the final trace. Its 9.2% kernel-total
reduction is not solely caused by this patch: other kernels also run 6.7%
faster. The initial 4K trace with nearly stable other kernels is a better
controlled observation of the aggregate saving. Other-kernel time also
improves 4.9% in the final 1080p trace, but is nearly stable for Flower (+0.3%).
Host/device transfer payloads and copy counts do not change: 4K still has
117,079,320 HtoD bytes/31 copies, 103,699,012 DtoH bytes/19 copies, and
518,400 D2D bytes/one copy.

The final build passes all 54 CUDA tests and the CPU-only build passes all 47.
Both the AC-strategy and expanded CUDA backend tests pass memcheck, racecheck,
synccheck, and initcheck with zero errors/hazards. Only the backend test
suppresses API-error reporting for its deliberately invalid zero-grid launch;
the AC test uses no API-error suppression. Six parent/final codestreams,
selected-strategy summaries, and final perceptual scores match exactly:
sample efforts 7/9, odd 1080p/4K effort 7, and Flower efforts 7/9. The pinned
decoder reads all six outputs at their original dimensions. Their hashes
remain those of the preceding checkpoints.

Final 1080p batch qualification uses one warmup and three alternating paired
serial/batch samples per size, with identical output required throughout:

| AQ mode | Batch | Batch median | Images/s | Paired speedup |
|---|---:|---:|---:|---:|
| Fully resident | 1 | 223.724 ms | 4.470 | 1.155x |
| Fully resident | 2 | 416.248 ms | 4.805 | 1.177x |
| Fully resident | 4 | 890.944 ms | 4.490 | 1.382x |
| Maximum throughput | 1 | 158.049 ms | 6.327 | 1.005x |
| Maximum throughput | 2 | 303.406 ms | 6.592 | 1.399x |
| Maximum throughput | 4 | 400.195 ms | 9.995 | 1.667x |

These are correctness/overlap qualifications, not before/after batch claims.
Ending GPU state is 74 C/P5/697 MHz graphics/810 MHz memory. Power-state and
host timing variation remain important; no clock or power policy was changed.

Ignored artifacts under `build-cuda-ninja/profiles/` use the `s27_` prefix:
`parent_4k`/`fused_4k` hold the initial `.nsys-rep`/`.sqlite` traces;
`final_{parent,retained}_{4k,1080p,flower}` hold final traces.
`compare_warmed.py` and `warmed_{4k,1080p,flower}.{json,txt}` retain paired
results and raw benchmark stdout. `verify.ps1`/`identity.txt`,
`{ac,backend}_final_{memcheck,racecheck,synccheck,initcheck}.txt`, and
`batch_{fully-resident,maximum-throughput}.txt` retain qualification results.
`qualify.ps1` runs GPU benchmarks and checks sequentially;
`profile_final.ps1`/`profile_summary.py` capture and extract measurements.
The profile extractor counts the first seven complete AC groups, recognizing
that parent groups have a residual launch and fused groups do not.

#### Remaining opportunities

The shared search planner still reserves two `maximum_packed_bytes` ranges,
and public CUDA batch validation still requires those full ranges. After this
fusion, A only needs three floats per candidate. At padded 4K, its maximum
could fall from 304,496,640 to 1,555,200 bytes, saving about 303 MB per search
instance before alignment. That reduction is not implemented here: it needs
an explicit scratch-size contract/query, corresponding arena planning and
range/alias validation, and coverage of the other backends and batch lifetimes.

The final 4K trace still contains 70.292 ms of large DCT (now including residual
and loss work), 23.203 ms of packed DCT, 64.880 ms of convolution, 54.398 ms of
Malta, 29.158 ms of coefficient encoding, and 23.708 ms of adjusted quantization.
The transforms still use direct basis-matrix dot products; a factorized-DCT
experiment would need numerical and perceptual qualification. Host work and
transfer boundaries also remain material. The fully-resident path is not
demonstrated maxed out.

### Follow-up: compact CUDA AC-search scratch allocation (2026-09-05)

Parent: `5ba4d86` (residual/inverse/loss fusion). This checkpoint completes the
allocation follow-up described above; it changes no device arithmetic or kernel
launches.

#### Allocation contract and implementation

The fused inverse evaluator only writes one loss sum per candidate channel to
scratch A. The shared frontend nevertheless reserved two complete packed
coefficient ranges, and CUDA validation enforced both obsolete ranges, including
in its alias checks. Reducing only the frontend allocation would therefore be
rejected (or leave misleading overlap checks when suballocating a larger arena).

`GpuAcStrategyEvaluation::GetAcStrategyScratchRequirements` now supplies the
per-strategy/per-count contract without allocating or submitting work. Its
checked conservative default retains the original A/B/rate sizes for Metal and
other implementations. CUDA overrides A to `candidate_count * 3 * sizeof(float)`;
B and rate scratch are unchanged. Unknown strategies, null outputs, overflow,
and unsupported CUDA strategies are rejected without changing the output.
Sizing alone does not certify device indexing or launch limits; submission
validation still checks those limits and buffer ownership.

The shared planner takes three independent maxima across stages and uses them
for both capacity planning and suballocation. CUDA uses the same query for
minimum buffer ranges and overlap validation. Old full-size allocations remain
valid, but adjacent compact suballocations no longer appear to overlap. The
existing ordered submission, scratch reuse, completion waits, and ownership
lifetimes are unchanged. Search statistics expose the three logical scratch
ranges and the actual retained owning arena capacity, excluding external
resident inputs and legacy input staging. Prepared searches retain capacity
when their logical geometry shrinks.

At padded 4K (3840x2160), the maximum coefficient range comes from the 99,120
16x16 candidates, while maximum loss storage comes from 129,600 DCT8 candidates:

| Range | Parent bytes | Compact bytes |
|---|---:|---:|
| Scratch A | 304,496,640 | 1,555,200 |
| Scratch B | 304,496,640 | 304,496,640 |
| Rate scratch | 3,110,400 | 3,110,400 |
| Complete search arena, including tables/costs/alignment | 626,787,328 | 323,845,888 |

The complete arena saves 302,941,440 bytes (288.9 MiB), or 48.3%. This is an
allocation-capacity reduction, not another reduction in device memory traffic:
the preceding fusion had already eliminated the unused intermediate writes.
The rest of the encoder's resident state is outside these arena totals.

#### Contract and regression coverage

CPU-only tests exercise the conservative default across all strategy metadata,
zero and ordinary counts, the largest non-overflowing count and its successor,
unknown strategies, null outputs, and missing capabilities. CUDA tests check
compact query results for all seven supported shapes and allocation/submission-
free error handling. Candidate tests now allocate exact-size A buffers, retain
the existing strided/resident/tail coverage, and also compare conservative A
buffers against compact results exactly.

For each supported shape, counts 1/8/11/16/32/33 exercise adjacent A/B/rate/cost
ranges with nonzero offsets, outer guards, and repeated ordered submissions.
Every output range is tested one float short and misaligned; all six output
pairs are tested for partial overlap, with input alias and foreign-backend A
rejected as well. Invalid range requests must not commit a submission.

The existing CPU/GPU search-parity test is now also built for CUDA. It covers
small and non-full-tile geometries, dependency-minimal candidate counts, atomic
failure, repeated prepared searches, and shrink/restore reuse. It reconstructs
the arena capacity independently from the candidate counts and checks it against
the actual retained capacity. Its optional `--memory-4k` mode runs the same
allocation and reuse checks at padded 4K. All 55 CUDA-enabled and 47 CPU-only
CTest tests pass, including install consumers. Metal runtime testing is not
available on this Windows host; its conservative sizing default is exercised
by the portable contract test.

#### Warmed public-workflow timing

Seven alternating parent/candidate process pairs each ran three warmups and
five measured fully-resident samples, without optional final-score diagnostics.
No builds, other GPU work, or heavy disassembly overlapped these runs. All
samples and outliers are retained. Medians below are medians of the seven
process medians; paired changes are independently computed from each pair's
ratio (not from the ratio of the two cohort medians).

| Input | Parent total ms | Compact total ms | Median paired total change | Parent quantization ms | Compact quantization ms | Median paired quantization change |
|---|---:|---:|---:|---:|---:|---:|
| Padded 4K | 837.477 | 811.058 | -3.2% | 570.773 | 562.826 | -1.7% |
| Padded 1080p | 209.757 | 211.064 | -0.2% | 129.130 | 126.483 | -1.7% |
| Flower, 510x532 | 47.045 | 46.153 | -5.0% | 29.025 | 27.848 | -3.0% |

Compact allocation wins 7/7 total and quantization pairs at 4K, 4/7 total and
6/7 quantization pairs at 1080p, and 6/7 total and quantization pairs for Flower.
Total-time ranges are 807.969–906.389 versus 778.103–842.434 ms at 4K,
204.266–234.241 versus 200.953–226.513 ms at 1080p, and 45.482–75.145 versus
42.603–49.127 ms for Flower. In particular, the long Flower parent process is
not discarded. These are observed wall-time changes on a clock-variable laptop,
not faster-transform claims; 1080p total time is effectively unchanged. The
allocation saving is the stronger deterministic result.

#### Allocation traces and final qualification

Separate Nsight captures enable `--cuda-memory-usage=true`. These are used for
allocation and launch accounting, not latency claims: the profiler documents
potentially significant overhead for memory tracking. The export also includes
warmup allocation/free events at timestamp zero, so the single-encode extractor
only counts positive-timestamp device-memory events. It verifies matched
allocation/free addresses and sizes, and an empty live-allocation set at the end.

| Input | Parent search arena bytes | Compact search arena bytes | Parent peak tracked device bytes | Compact peak tracked device bytes |
|---|---:|---:|---:|---:|
| Padded 4K | 626,787,328 | 323,845,888 | 3,110,455,104 | 2,807,513,664 |
| Padded 1080p | 156,741,248 | 81,005,952 | 777,459,334 | 701,724,038 |
| Flower | 20,602,112 | 10,675,712 | 102,332,176 | 92,405,776 |

Each encode still makes five device allocations and frees. Only the search
arena size changes; the other four allocations match exactly. The 4K peak of
tracked device allocations falls 9.7%. This is requested CUDA allocation memory,
not total board/driver memory or a direct measurement of physical VRAM residency.
Kernel counts, ordered launch geometry, registers, dynamic shared memory, and
transfer counts/bytes match exactly for each input. At 4K this remains 509 kernels,
31 HtoD copies (117,079,320 bytes), 19 DtoH copies (103,699,012 bytes), and one
518,400-byte D2D copy. Memory-tracked kernel totals are 356.652/364.086 ms for
parent/compact 4K, 61.595/60.618 ms at 1080p, and 16.333/16.308 ms for Flower;
no compute-speed improvement is inferred from these traces.

Both candidate and prepared-search executables pass memcheck, racecheck,
synccheck, and initcheck, with zero errors or race warnings. The complete 4K
prepared-search reuse check also passes memcheck. Six scored parent/compact
encodes (sample and Flower at efforts 7/9; odd padded 1080p/4K at effort 7)
retain identical SHA-256, strategy summaries, and final scores. All six compact
outputs decode with the pinned independent decoder at the expected dimensions.

Batch sizes 1/2/4 pass at 1080p in both fully-resident and maximum-throughput
modes. Compact fully-resident batch medians are 281.150/493.102/923.458 ms
(3.557/4.056/4.332 images/s); maximum-throughput medians are
118.138/289.011/583.037 ms (8.465/6.920/6.861 images/s). Fully-resident 4K
batch sizes 1/2 pass for both builds: parent medians 1128.849/2046.184 ms,
compact 1139.528/2134.260 ms. Those sequential cohorts are qualification runs,
not a controlled before/after batch speedup result. They follow instrumented
checks and show substantial clock/host variance. All batch outputs match their
serial references. End-of-sweep telemetry is 77 C, P3, 1282 MHz graphics,
5500 MHz memory.

A full-process memory trace of batch size 2 (one warmup and one paired serial/
batch sample) confirms both lanes' search arenas are live together. Peak tracked
device allocations fall from 6,222,823,552 to 5,616,940,672 bytes, saving
605,882,880 bytes (577.8 MiB). Both peaks contain ten allocations: two copies
of each of the five per-image allocations. All nine encodes in each captured
process have matching allocation-size counts, except for the smaller search
arena. These aligned 3840x2160 batch inputs have slightly different non-search
allocation sizes from the odd-source single-image workload above. Memory
tracking confirms the capacity benefit under concurrency without establishing
a batch latency improvement.

#### Artifacts and remaining work

Ignored artifacts are under `build-cuda-ninja/profiles/s28_*`. Saved
`parent_{benchmark,encode,batch}.exe` binaries precede the change. The
`warmed_{4k,1080p,flower}.{json,txt}` files retain every process/sample;
`final_{parent,retained}_{4k,1080p,flower}.{nsys-rep,sqlite}` and
`memory_summary.{py,json}` retain the launch/allocation accounting. Full-process
batch captures are `memory_batch2_{parent,retained}.{nsys-rep,sqlite}`, with
`batch_memory_summary.{py,json}`. The `ac_strategy_cuda{,_search}_*check.txt`,
`memory_4k.txt`, `identity.txt`, and `batch_*.txt` files retain sanitizer,
codestream, and batch qualification. `qualify.ps1` sequences the checks, and
`profile_batch_memory.ps1` records the separate concurrent-allocation traces.

No further full-sized dead AC-search intermediate remains in this allocation
layout. The retained forward coefficients in B are still required by the fused
inverse, including cross-channel CfL. Reducing B needs a different execution
schedule or more fusion, not merely a smaller range. Other resident arenas,
direct basis-matrix DCT arithmetic, reconstruction filters, host work, and
transfers remain material. A bounded factorized-DCT experiment can start from
the existing Metal radix-2 implementation, first checking CUDA register/spill
behavior and standalone transform accuracy, then integrating with fused AC
input/output paths and qualifying perceptual/size behavior. Byte identity is
not assumed to be the acceptance ceiling for that arithmetic experiment.
The fully-resident backend is not demonstrated maxed out.

### Follow-up: factorize resident CUDA DCT arithmetic (2026-09-05)

Parent: `c1a75bb` (compact AC-search scratch). This experiment replaces direct
basis-matrix arithmetic for the seven resident transform shapes through 32x32
with the radix-2 DCT-II/III factorization already used by the Metal backend.
It applies to ordinary forward/inverse transforms, descriptor-gathered AC
forward transforms, and fused residual/inverse/loss evaluation. The 64x32 and
32x64 ordinary transforms retain their matrix implementations. The internal
`LaunchCudaDctMatrix` entry point preserves the old arithmetic as a numerical
and performance oracle, including all smaller shapes.

#### Arithmetic and scheduling

The direct separable transform performs work proportional to
`W*H*(W+H)`. Factorization reduces that to `W*H*(log2(W)+log2(H))`, using
compile-time butterflies and constants instead of basis loads and long dot
products. Each lane owns a full one-dimensional register vector; recursion
and loops are forcibly inlined/unrolled. Forward output is normalized by
`1/(W*H)`, and inverse output is already in pixel units. The fused loss adapter
therefore omits the matrix inverse's orthonormal-to-pixel rescaling, while
retaining its eighth-power arithmetic and FP32 halving reduction.

Blocks contain 64 threads, packing `64/max(W,H)` transforms, each with a
`H*(W+1)` padded shared tile. Tall/square forward transforms process columns
first so input and coefficient transactions coalesce. Wide forward transforms
stage coalesced input reads, then process rows first to retain coalesced native
row-major coefficient stores. Inverse transforms use the same native
coefficient layout and coalesced pixel stores. Tall fused inverse outputs
are redistributed through the existing tile before loss reduction to preserve
the established row-major mask addressing and summation order. Inactive tail
groups participate in block barriers without reading or writing global data.
No new global intermediate, allocation, or submission boundary is introduced.

Standalone prototypes tested 64/128/256-thread blocks over 16,777,216 floats,
with three warmups and seven alternating rounds of three launches. The first
wide-forward variant had scattered stores; merely moving the scattered side
to input reads was insufficient. Coalesced staging fixed that issue. Selected
64-thread prototype medians (matrix to factorized) were 4.298 to 0.549 ms for
32x16 forward, 5.150 to 0.552 ms for 16x32 inverse, and 4.265 to 0.553 ms for
32x32 inverse. These are raw-kernel observations, not whole-encoder speedups;
clock variance was especially large in early small-transform rounds.

The initial integrated 32x32 residual/inverse/loss kernel compiled to a
384-byte stack frame, despite the standalone transform being spill-free.
SASS showed 32 square-root helper call sites and local array traffic.
Forcing the input/output adapters inline, spelling the root `__fsqrt_rn`, and
using a narrowly scoped FTZ square-root instruction did not remove that frame;
none of those changes is retained. Instead, coefficient rates now perform the
first FP32 halving step as each lower/upper pair arrives, retaining only half
the magnitude array. Exact integer nonzero counts accumulate immediately.
This preserves the magnitude tree and count semantics, uses ordinary `sqrtf`,
and reduces the 32x32 kernel to zero stack/local bytes (89 registers per
thread, 8,448 shared bytes per block). The helper call sites remain, but the
local loads/stores disappear. The same change reduces 16x16 residual/inverse
registers from 76 to 43 and both 512-coefficient shapes from 80 to 64.
No global fast-math option is enabled.

#### Numerical and quality qualification

All 55 CUDA CTests and all 47 CPU-only CTests pass. Existing transform
tolerances are unchanged: forward absolute/relative `3e-5/3e-4`, inverse and
round-trip `5e-4/5e-4`. The backend test now additionally runs both raw launch
paths against the independent double reference with seven leading and eleven
trailing sentinels, offset pointers, and an unchanged-input check. Coverage is
nine shapes times fourteen batch counts (1/2/3/4/7/8/9/15/16/17/19/31/32/33),
with independent forward/inverse inputs, impulses, constants, checkerboards,
unequal horizontal/vertical structure, and noise. Both raw paths and both
directions are checked in each configuration (504 guarded launches).

The existing fused oracles also pass: 420 inverse/loss batches (12,810 channel
loss comparisons), and 84 residual/inverse batches (4,242 channel comparisons)
with exact FP32 loss/rate reduction checks, exact nonzero counts, non-finite
handling, mask and descriptor boundaries, host/device CfL, output guards,
and partial blocks. Full AC-cost CPU-reference and ordered-search tests pass
without loosening their tolerances. The standalone double-reference probe's
largest 32x32 inverse absolute error was `1.2445e-4` (matrix `4.7849e-5`), with
factorized RMS error `5.25e-6`; factorization changes rounding and is not
claimed byte-identical to the matrix path.

Compute Sanitizer memcheck/racecheck/synccheck/initcheck all pass for the
expanded CUDA backend test, AC-candidate test, and AC-search test (12 runs).
The large `--memory-4k` search also passes memcheck. No errors or race hazards
are reported. Only the backend test suppresses CUDA API-error reporting,
because it intentionally launches a zero-sized grid to verify stale-error
consumption; memory/race/synchronization/uninitialized-access detection remains
enabled. Runs are serialized, with no overlapping build or benchmark.

The decoded-image comparison uses the pinned libjxl `djxl` and
`butteraugli_main` tools. Decoder output and metric input are explicitly
linear sRGB (`RGB_D65_SRG_Rel_Lin`), with default SDR metric intensity 80 nits.
Seven inputs comprise the 17x13 sample, odd-source synthetic 1080p/4K,
510x532 Flower, and three 500x500 color photographs from libjxl testdata's
[Wesaturate corpus](https://github.com/libjxl/testdata/tree/73695d303670c90e4d506ea89d9901b081385089/external/wesaturate/500px).
The latter's pinned README/license identify the originals as CC0. The exact
files are `cvo9xd_keong_macan_srgb8.png`,
`tmshre_riaphotographs_srgb8.png`, and `u76c0g_bliznaca_srgb8.png`;
source hashes, linear-PFM hashes, and license copies accompany the artifacts.
Every input is encoded at effort 7 and distances 0.5/1.2/3.0; sample and Flower
also run effort 9 at distance 1.2 (23 before/after pairs). A greater-than-0.5%
increase in either bytes or independently decoded Butteraugli score was
declared an investigation threshold before collecting results, not an assumed
acceptance result. All cases, including any outliers, are retained.

All 46 codestreams decode with the expected dimensions, and no pair exceeds
either investigation threshold. The independently decoded Butteraugli score
is identical to the tool's printed precision in 22 pairs; Bliznaca at distance
3 improves from 3.2424831390 to 3.2351341248 (-0.227%). File-size ratios range
from 0.9998273 to 1.0002802. The only size changes are:

| Input / distance / effort | Parent bytes | Factorized bytes | Change |
|---|---:|---:|---:|
| Synthetic 1080p / 0.5 / 7 | 207,139 | 207,173 | +0.0164% |
| Synthetic 1080p / 1.2 / 7 | 46,952 | 46,959 | +0.0149% |
| Synthetic 4K / 0.5 / 7 | 798,858 | 798,720 | -0.0173% |
| Bliznaca / 3.0 / 7 | 17,846 | 17,851 | +0.0280% |

Strategy histograms match in every pair. Ten codestream pairs are byte-identical;
the other thirteen are not, including some with identical length and metric.
Equal metric scores are not a claim of pixel identity. These results qualify
this SDR corpus and the tested device/toolchain, not arbitrary image content,
HDR inputs, or other GPU architectures. The decoder/metric libjxl revision is
`e8ff09762481785938d8e4e01333ed3917571161` (Clang 22.1.8).

#### Final warmed wall measurements

Seven alternating parent/candidate process pairs per workload, each with three
warmups and five measured samples. Each process contributes its sample median;
the reported percentage is the median of the seven paired candidate/parent
ratios, not the ratio of independently pooled medians. No build, sanitizer,
second GPU workload, or profiling overlaps these runs. Every sample is kept.

| Workload / stage | Parent median [range], ms | Factorized median [range], ms | Paired change | Winning pairs |
|---|---:|---:|---:|---:|
| Padded 4K / total | 838.434 [799.568-875.159] | 782.615 [736.428-822.077] | -5.1% | 7/7 |
| Padded 4K / quantization | 577.619 [555.050-584.848] | 511.189 [495.734-522.603] | -10.6% | 7/7 |
| Padded 1080p / total | 214.800 [210.419-220.980] | 199.259 [195.440-238.996] | -7.2% | 5/7 |
| Padded 1080p / quantization | 129.420 [126.951-134.078] | 116.943 [115.522-132.189] | -8.9% | 6/7 |
| Flower / total | 49.989 [45.986-68.615] | 44.917 [42.227-53.744] | -7.8% | 5/7 |
| Flower / quantization | 29.749 [27.704-35.669] | 27.309 [25.423-31.164] | -5.9% | 6/7 |

The large 1080p candidate and Flower parent/candidate outliers are not removed.
An earlier 4K cohort of the initial integrated factorization, before the
rate-storage fix, measured -5.8% total and -9.6% quantization. It is retained
as an experiment record, not pooled with the final implementation. Laptop
clock/host variability prevents interpreting these sequential cohorts as a
precise isolated end-to-end benefit of the stack fix.

#### Final kernel traces and batch checks

Nsight captures one fully-resident encode after three warmups, without memory
tracking. The unchanged-kernel column excludes all ordinary and AC DCT kernels,
so it does not accidentally count another improved transform as a control.
Static and dynamic shared memory are both accounted for.

| Workload | All DCT, parent -> factorized | Other kernels, parent -> factorized | Total kernel time, parent -> factorized |
|---|---:|---:|---:|
| Padded 4K | 107.922 -> 23.051 ms (-78.6%) | 258.020 -> 262.462 ms | 365.941 -> 285.514 ms |
| Padded 1080p | 16.145 -> 5.345 ms (-66.9%) | 47.084 -> 43.816 ms | 63.229 -> 49.160 ms |
| Flower | 2.185 -> 0.771 ms (-64.7%) | 14.125 -> 14.153 ms | 16.310 -> 14.924 ms |

The 4K DCT split is AC forward 46.182 -> 8.793 ms, fused residual/inverse/loss
48.255 -> 11.737 ms, and ordinary transforms 13.485 -> 2.522 ms. Other kernels
are 1.7% slower in that trace, effectively unchanged on Flower (+0.2%), and
6.9% faster at 1080p. The latter control movement means that not all of the
1080p total-kernel reduction can be attributed to this change. Kernel duration
is not end-to-end latency; the separately warmed wall results above govern
that claim.

The seven 4K fused inverse kernels show:

| W x H | Matrix, ms | Factorized, ms | Factorized registers/thread | Factorized static shared bytes/block |
|---|---:|---:|---:|---:|
| 8x8 | 1.228 | 1.257 | 40 | 2,304 |
| 8x16 | 3.855 | 1.632 | 46 | 2,304 |
| 16x8 | 2.034 | 1.512 | 40 | 2,176 |
| 16x16 | 7.484 | 2.307 | 43 | 4,352 |
| 16x32 | 12.893 | 1.289 | 64 | 4,352 |
| 32x16 | 9.494 | 1.236 | 64 | 4,224 |
| 32x32 | 11.266 | 2.505 | 89 | 8,448 |

All factorized instantiations have zero stack/local bytes in `cuobjdump`.
The 8x8 fused inverse is essentially flat, including a small regression in
this particular 4K trace; the retained improvement is primarily in larger
shapes. The initial integrated 32x32 trace was 8.356 ms with stack traffic,
versus 2.505 ms after the pairwise rate-storage change. These are separate
captures, but the resource and SASS changes independently explain the removed
local traffic.

Launch counts are unchanged: 509 at 4K and Flower, 494 at 1080p. All transfer
counts and byte totals match each parent. The 4K trace retains 31 HtoD copies
(117,079,320 bytes), 19 DtoH copies (103,699,012 bytes), and one DtoD copy
(518,400 bytes). This checkpoint changes arithmetic and on-chip scheduling,
not the compact search arena or global transfer boundaries.

Batch sizes 1/2/4 pass at 1080p in both modes, and fully-resident sizes 1/2
pass at aligned 3840x2160. Every batch codestream matches its serial reference.
Fully-resident 1080p batch medians are 220.137/362.894/859.030 ms;
maximum-throughput medians are 117.586/246.722/332.225 ms. Fully-resident 4K
medians are 815.233/1622.771 ms. These are qualification runs with substantial
host/clock variance, not an isolated before/after batch-throughput measurement.

#### Artifacts and next bottlenecks

Ignored artifacts are under `build-cuda-ninja/profiles/s29_*`:
`parent_{benchmark,encode}.exe` preserve the parent; `factored_probe*` retain
the standalone layout/block-size experiments; `initial_*` retain the first
integration; `final_{parent,retained}_{4k,1080p,flower}.{nsys-rep,sqlite}` and
`profile_summary.{py,json}` retain the final launch/shape accounting.
`warmed_{4k,1080p,flower}.{json,txt}` preserve every timing sample.
`final_resources.txt` and `final_32_spills.txt` retain resource/call-site
inspection (the latter contains calls but no local loads/stores).
`quality.py`, `quality.json`, `quality_*.{jxl,pfm}`, and `corpus/` retain all
decoded-image comparisons and provenance. `sanitize.ps1`, `*check.txt`
(filenames end in the individual `memcheck`/`racecheck`/`synccheck`/`initcheck`
mode), `memory_4k.txt`, `measure.ps1`, and `batch_*.txt` retain qualification
scripts and results.

DCT is now only 8.1% of kernel time in the retained 4K trace. Convolutions
consume 73.862 ms and Malta 58.477 ms, with coefficient encoding at 29.477 ms
and quantization adjustment at 23.010 ms. The adjustment stage also consumes
8.480 ms at 1080p and 7.280 ms (48.8% of kernels) on Flower. Its current
`SelectAdjustedQuantizationKernel` assigns one thread to an entire transform
and scans all three channels serially through `AdjustQuantForChannel`;
cooperative coefficient processing is a concrete next experiment, subject to
its floating-point reduction/threshold decisions and decoded-quality gates.
Filtering locality/fusion, remaining resident allocations, host work, and
transfers also remain material. Further AC fusion could remove the retained
forward-coefficient buffer, but requires a new cross-channel schedule, not
another scratch-size adjustment. This is a verified checkpoint, not evidence
that fully-resident encoding is maxed out.

### Follow-up: cooperate across quantization-adjustment coefficients (2026-09-05)

Parent: `f26e2ef` (factorized resident DCT). The next profile identified
`SelectAdjustedQuantizationKernel` at 23.010 ms for 4K, 8.480 ms for 1080p,
and 7.280 ms on Flower. The latter was 48.8% of all GPU kernel duration.
The old kernel assigns one thread to each transform anchor, serially scans
all coefficients in each of three channels, and dynamically indexes quadrant
arrays. For Flower, six of seven shape batches launched just one 256-thread
block. The 32x32 batch took about 0.96 ms per invocation. Resource/SASS
inspection found a 144-byte thread stack and repeated local loads/stores.
This combines underfilled grids, scattered coefficient reads across threads,
long serial scans, and local-array traffic.

#### Cooperative implementation and experiments

Each anchor now launches one 96-thread block: one complete warp per channel,
with eight lanes assigned to each canonical coefficient quadrant. Lanes read
and quantize separate coefficients, accumulate scalar statistics, then reduce
within quadrants and across quadrant leaders. The warp leader applies the
existing raw-quant and Y-threshold policy. Three warp decisions pass through
12 bytes of shared memory and one block barrier before publishing their
maximum. This barrier also ensures that every channel has read the initial
raw quant before it is overwritten. No global partial buffer, new allocation,
extra launch, or host synchronization is introduced.

Width/height and channel are compile-time parameters. Seven physical strategy
orientations share five canonical coefficient shapes: 8x8, 16x8, 16x16,
32x16, and 32x32. Fixing the strategy class also resolves the policy tables to
constants. An initial cooperative prototype still used dynamic class indexing
and allocated 96 local bytes for the two multiplier tables on larger shapes;
specialization removed them. Final registers per thread for those five shapes
are 39/41/38/40/40, with zero stack/local bytes and 12 shared bytes per block.
The serial implementation remains available through the internal
`LaunchCudaAqSelectAdjustedQuantizationScalar` oracle, and is the fallback for
internal batches outside the specialized shape/strategy combinations.

The prototype compared one/two/four anchors per block (96/192/384 threads),
with three warmups and seven alternating rounds of three event-timed launches.
Initial raw-quant arrays were reset outside the timed interval. Small batches
had 129 anchors; large cases used 8,388,608 coefficients per channel. The
96-thread schedule consistently helped small grids and was competitive on
large grids, so it is the retained launch policy. For 129 32x32 anchors,
the final prototype measured 1.082 ms scalar versus about 0.022 ms cooperative;
for 8,192 such anchors, 2.781 versus 0.744 ms. These are raw-kernel observations
with broad clock/event variance, not public-workflow speedups.

Each prototype version checked 63 batches spanning all seven strategies,
counts 1/2/3/7/17/129/259 plus shape-dependent large counts, offset buffers,
guards, and AC non-finites. Across 338,917 anchor decisions per version,
all three cooperative launch geometries matched scalar raw quants and Y
threshold bits. This is measured fixture parity, not a universal bit-identity
claim: error, border, and magnitude sums use a fixed parallel FP32 tree instead
of serial row-major addition. No global fast math or reduced precision is used.

#### Direct and integrated correctness checks

All 56 CUDA CTests and 47 CPU-only CTests pass. The new
`cuda_quantization_adjustment` test checks 182 batches (9,667 anchors) against
the retained scalar CUDA implementation. Raw quant values and threshold bits
match exactly in these fixtures. Its 9,429 finite-input anchors also compare
against the independent CPU policy: exact raw quant, Y thresholds within
`2e-6` absolute. The latter permits the pre-existing CUDA float versus CPU
double policy constants; the existing test tolerances are not relaxed.

Coverage includes seven strategies, counts 1/2/3/7/17/33/129/257, global scales
1/3541/32768, unequal channel multipliers, raw values through 255/256, flat,
sparse, active, high-frequency-border, threshold-tie, half-integer, signed-zero,
and quantizer-limit patterns. Inputs use the real default quantization matrices;
unused table entries are NaNs. Anchor/coefficient/raw pointers are offset,
raw fields are strided and guarded, threshold output has prefix/suffix guards,
and coefficients must remain bitwise unchanged. AC NaN/infinity inputs set
error bit 16 without clearing a pre-existing bit; low-frequency non-finites
are skipped as in the scalar kernel. The public AQ tests retain their
bounded/full, maximum-error, repeatability, arena-reuse, and frame checks.

Eight sanitizer runs complete with zero errors/hazards: all four modes on
the focused adjustment test, plus full-AQ memcheck, synccheck, initcheck,
and kernel-filtered racecheck. The latter executes the entire existing AQ
test, including four-worker public encodes in both policies, while
`--kernel-regex kns=SelectAdjustedQuantization` instruments the changed
cooperative instantiations and retained scalar kernel. All other listed
runs are unfiltered; no CUDA API errors are suppressed.

The initial unfiltered full-AQ racecheck was deliberately aborted after
approximately 52 minutes without a final summary. It is incomplete, not a
pass or a detected-race result. Its original log is preserved. A Windows
firewall/elevation prompt was subsequently reported, but its role in the
delay was not established. The kernel-filtered full-workflow racecheck
completed in approximately 27 seconds. This qualification does not claim
an unfiltered full-AQ racecheck pass.

All 23 before/after image pairs are byte-identical to parent `f26e2ef`;
encoded sizes, independently decoded Butteraugli scores, selected-strategy
summaries, and reported final scores match. The corpus and explicit linear
color protocol are the same as the preceding DCT checkpoint: seven inputs
at distances 0.5/1.2/3.0 and effort 7, plus sample/Flower effort-9 checks at
distance 1.2. Pinned libjxl revision
`e8ff09762481785938d8e4e01333ed3917571161` decodes all outputs at their source
dimensions and measures them with `RGB_D65_SRG_Rel_Lin`, SDR 80 nits.
No case crosses the predeclared 0.5% size/score investigation threshold;
every measured change is zero. This corpus result is not a universal
bit-identity guarantee for the reordered floating-point reductions.

Final parent/retained Nsight captures use three warmups and one captured
fully-resident encode, with no overlapping GPU test or benchmark work:

| Workload | Adjustment parent | Cooperative | Change | Other kernels parent / cooperative | All kernels parent / cooperative |
|---|---:|---:|---:|---:|---:|
| Odd 4K | 25.507 ms | 3.969 ms | -84.4% | 275.729 / 297.507 ms | 301.236 / 301.476 ms |
| Odd 1080p | 8.818 ms | 0.668 ms | -92.4% | 41.897 / 41.716 ms | 50.715 / 42.384 ms |
| Flower | 7.272 ms | 0.242 ms | -96.7% | 7.634 / 7.644 ms | 14.905 / 7.886 ms |

The 4K unchanged-kernel control grows 7.9%, offsetting the 21.538 ms
adjustment reduction; no total-GPU-time gain is claimed from that capture.
At 1080p and Flower, unchanged work is within 0.5% and total kernel time
falls 16.4% and 47.1%. Launch counts remain 509/494/509 respectively.
Each before/after pair has identical HtoD, DtoH, and D2D transfer volumes
and counts (31/19/1 copies); the optimization adds no transfers or launches.

All batch runs complete with codestream and full-summary identity against
their serial references. Current-build batch results below qualify behavior,
not an improvement over the parent:

| Workload / policy | Batch size | Batch median | Images/s | Median serial/batch speedup |
|---|---:|---:|---:|---:|
| 1080p / fully resident | 1 | 249.021 ms | 4.016 | 0.917x |
| 1080p / fully resident | 2 | 449.865 ms | 4.446 | 1.223x |
| 1080p / fully resident | 4 | 979.541 ms | 4.084 | 1.515x |
| 1080p / maximum throughput | 1 | 196.711 ms | 5.084 | 1.185x |
| 1080p / maximum throughput | 2 | 390.839 ms | 5.117 | 1.542x |
| 1080p / maximum throughput | 4 | 481.612 ms | 8.305 | 2.053x |
| 4K / fully resident | 1 | 1071.586 ms | 0.933 | 0.877x |
| 4K / fully resident | 2 | 1956.245 ms | 1.022 | 1.164x |

These batch checks use one warmup and three alternating serial/batch samples.
The GPU-state snapshots around the first timing cohort span 73-80 C,
P8/P3, and 210-1282 MHz; clocks were not locked. A post-measurement snapshot
reports AC power, the Windows Balanced scheme, and concurrent CPU activity
from Windows security, Office, and editor processes. This is evidence of
background activity, not proof that it caused every timing difference.

Wall-time qualification uses seven alternating independent-process pairs,
three warmups and five retained samples per process, distance 1.2, effort 7,
automatic CPU threads, and no final-score diagnostic. A second complete
cohort was run because the initial parent itself varied substantially.
All samples and both cohorts are retained; none are dropped as outliers.
Percentages below are medians of paired candidate/parent ratios, not ratios
of cohort medians. The pooled column contains all fourteen pairs:

| Workload / stage | Initial paired change | Repeat paired change | All 14 pairs |
|---|---:|---:|---:|
| Odd 4K / total | +20.8% | +3.7% | +7.6% |
| Odd 4K / quantization | -7.4% | +0.5% | -0.4% |
| Odd 1080p / total | +8.8% | -1.4% | -1.3% |
| Odd 1080p / quantization | -1.3% | -11.4% | -10.5% |
| Flower / total | -5.3% | -10.1% | -7.7% |
| Flower / quantization | -19.8% | -19.5% | -19.6% |

For context, initial 4K process medians span 755.503-2304.972 ms in the
parent and 1039.518-2202.232 ms in the candidate. A seven-pair control using
the identical parent executable in both positions reports a +2.2% median
paired total difference, with individual pairs from -4.4% to +28.4%.
Its quantization difference is +0.9%, ranging from -2.8% to +20.0%.
The unchanged CPU codestream stage is 8.7% slower in the pooled 4K pairs;
input preparation is +0.2%. These controls demonstrate measurement
variability, not that the measured 4K regression can be ignored or that a
particular background process caused it.

A further diagnostic compiles the retained scalar and cooperative policies
into one executable and chooses between them at process initialization.
Both modes use the same host code, libraries, binary layout, executable
path, and timing boundary. This ignored-build probe adds no production
environment switch. It confirms exact codestream identity on the actual
benchmark inputs: 1,048,983 bytes for odd 4K, 265,570 for odd 1080p, and
37,018 for Flower. The first-sample output dump is enabled only during
identity qualification and is outside the timed encode; timing runs do
not write codestream files. The same seven-pair protocol produces:

| Same-executable workload | Median paired total change | Median paired quantization change |
|---|---:|---:|
| Odd 4K | -0.2% | -0.5% |
| Odd 1080p | -0.1% | -5.5% |
| Flower | +2.6% | -15.8% |

This removes binary/path differences from that diagnostic but not host
contention, timing variation, or automatic GPU power-state changes. The
retention claim is therefore limited to the large, directly measured
adjustment-kernel reduction and qualified correctness. Stable whole-encode
latency or throughput gains are not established by this checkpoint.

The fresh retained 4K trace spends 54.798 ms in Malta response, 13.637 ms
in Malta scaling, and 33.347 ms in resident coefficient encoding. Tiled
convolutions and host serialization/launch boundaries remain substantial.
Read-only inspection confirms Malta has no local/stack storage and already
reuses repeated response sums; scale/response fusion would duplicate scale
arithmetic in tile halos and needs measurement, not an assumed win. The
fully-resident path is not demonstrated maxed out.

Ignored artifacts under `build-cuda-ninja/profiles` use the `s30_` prefix.
`parent_{encode,benchmark,batch}.exe` are hash-verified copies of the preceding
retained executables; `retained_{encode,benchmark,batch}.exe` preserve the
qualified current build. `quant_probe{,_v1}.exe`, `quant_probe{,_v2}.txt`,
`cooperative_quant{,_v1}.cuh`, and `parent_resident_kernels.cu` preserve the
event-probe variants and frozen baseline; `final_resources.txt` records
production resource usage. `quality.py`/`quality.json` and `quality_*`
retain the decoded comparisons, reusing the provenance-pinned `s29_corpus`.
`sanitize{,_remaining}.ps1`, individual `*check.txt` files, and
`sanitizer_abort.txt` preserve the exact sanitizer scopes and aborted run.
`measure.ps1`, `warmed_*`, `recheck.ps1`, `repeat_*`, `control_aa_4k.*`,
`wall_summary.*`, and GPU-state logs retain all wall-time observations.
`profile_final.ps1`, `profile_summary.*`, `final_{parent,retained}_*`,
`kernel_totals.*`, and `batch_*.txt` retain the profiles and batch checks.
`switch_probe.cu`, `switch_benchmark.cpp`, `switch_measure.py`, and
`switch_*` preserve the same-executable diagnostic and exact outputs.

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

The DCT, quantization, and selection paths are decision-sensitive. Reduced
precision or globally relaxed math needs an explicit mode and quality
contract; it should not silently alter fully-resident behavior. The retained
FP32 radix-2 factorization uses ordinary math, preserves existing numerical
tolerances, and separately qualifies rounding changes against decoded images.

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
