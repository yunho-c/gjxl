# Last-mile effort-7 optimization roadmap

Date: 2026-09-02

- Profiled revision: `1d5d886`
- Integrated `main` revision: `a437d72`
- Source-tree identity: both revisions have tree
  `4add35822d7e8217c70e068088a868688b22061b`
- Machine: Apple M4 Pro
- Path: ordinary effort 7, fully resident Metal, SIMD DCT, Butteraugli
  distance `1.2`, automatic CPU thread budget, final diagnostic score disabled

## Executive conclusion

The ordinary codestream is no longer the dominant reason the Metal encoder is
slow. At padded 4K, the complete encode takes `390.36 ms`: `324.06 ms`, or
`83.0%`, is in the quantization pipeline, while `52.86 ms`, or `13.5%`, is in
codestream encoding.

The next large opportunity is the effort-7 AQ update policy. With the current
two-update policy, the second resident evaluation accounts for `54.27 ms` of
measured GPU time at 4K. That policy change is independently owned and is not
part of this roadmap's implementation scope.

After that policy change, the recommended GJXL-owned order is:

1. retain the qualified, purgeable exact-capacity Metal AQ workspace leases;
   keep Butteraugli storage per-encode because its idle high-water cost is not
   justified;
2. parallelize coefficient-order zero statistics;
3. optimize the existing effort-7 AC-strategy kernels without pruning the
   libjxl-grounded candidate search;
4. prototype group-major packing in the existing final Metal submission;
5. address smaller codestream and input-transform costs only after re-profiling.

GPU entropy coding is not indicated. The remaining entropy-construction wall
time is only about `8 ms` at 4K, and the independently parallel section-writing
boundary is about `11 ms`.

## Measurement contract

### Authoritative wall-time cohort

Five independent Release processes were captured for each workload. Workload
order alternated by process pair. Each process performed one unretained warmup
and one retained sample. The benchmark boundary starts with an already generated
linear-RGB image and ends with the in-memory codestream. Workload generation,
process startup, and backend creation are excluded. Metal-only validation avoids
including a CPU reference encode in the retained measurement.

The command shape was:

```sh
build/release/gjxl_encoding_benchmark \
  --workload padded_4k \
  --scope metal-public-workflow \
  --implementation simd \
  --gpu-aq fully-resident \
  --validation metal-only \
  --distance 1.2 \
  --effort 7 \
  --cpu-threads auto \
  --warmups 1 \
  --samples 1 \
  --raw-samples raw.json
```

Every retained 1080p sample produced `410,072` bytes, and every retained 4K
sample produced `1,606,911` bytes.

### Attribution captures

The finer measurements serve different purposes and must not be substituted
for complete-encode wall time:

- GPU stage profiles used two warmups and seven samples per workload. Their
  timestamp-separated stage values explain where GPU time is spent, but the
  profiled path has instrumentation overhead.
- One 1 kHz Samply capture covered a 4K process with two warmups and twelve
  retained encodes. It contained 4,588 samples, `4,406.466 ms` of sampled
  thread CPU, and `99.80%` weighted leaf-symbol resolution. Its percentages are
  aggregate CPU attribution across all participating threads, not latency.
- A thread-budget sweep used two warmups and five samples in one process per
  setting. A follow-up comparison used seven alternating independent-process
  pairs for 8 workers versus automatic scheduling.
- Dispatch-boundary GPU timestamps were requested but are unavailable on this
  device/driver path. Kernel-level occupancy, bandwidth, and cache claims
  therefore require Metal System Trace or explicit kernel instrumentation.

## Current wall-time budget

The independent-process medians are:

| Effort-7 fully resident stage | Padded 1080p | Padded 4K | 4K share |
| --- | ---: | ---: | ---: |
| Complete encode | `111.466 ms` | `390.358 ms` | `100%` |
| Input preparation | `3.527 ms` | `12.525 ms` | `3.2%` |
| Quantization pipeline | `86.422 ms` | `324.061 ms` | `83.0%` |
| Codestream encoding | `21.078 ms` | `52.863 ms` | `13.5%` |
| Summary and residual overhead | about `0.44 ms` | about `0.91 ms` | `0.2%` |

The largest instrumented quantization wall stages scale close to image area:

| Instrumented stage | Padded 1080p | Padded 4K | Interpretation |
| --- | ---: | ---: | --- |
| `resident.aq` | `36.802 ms` | `135.047 ms` | reconstruction, filtering, Butteraugli, policy updates, and final frame work |
| `frontend.prepare_evaluator` | `20.982 ms` | `80.726 ms` | host planning/allocation/uploads plus asynchronous reference preparation |
| `frontend.ac_strategy.wait` | `18.324 ms` | `73.292 ms` | critical wait for AC candidate evaluation |
| `frontend.initial_quantization` | `7.631 ms` | `27.364 ms` | queued initial-quantization completion; includes queue dependencies |
| `resident.frame_assembly` | `4.113 ms` | `16.698 ms` | host construction of the owned serializer frame |

These profiled stage medians are diagnostic. They should not be added to or
subtracted directly from the uninstrumented complete-encode medians.

## GPU work inside the quantization pipeline

The stage timestamps expose real GPU work even though per-dispatch timestamps
are unavailable:

| GPU stage family | Padded 1080p | Padded 4K |
| --- | ---: | ---: |
| Butteraugli stages across resident evaluations | `19.759 ms` | `82.032 ms` |
| AC-strategy candidate evaluation | `14.588 ms` | `59.011 ms` |
| Resident reconstruction stages | `6.895 ms` | `18.526 ms` |
| Gaborish, EPF, and Opsin-to-linear stages | `2.561 ms` | `11.708 ms` |
| Butteraugli reference preparation | `5.435 ms` | `22.054 ms` |
| Initial quantization | `1.950 ms` | `4.536 ms` |

The current effort-7 path executes two AQ scoring iterations followed by final
frame coefficient generation:

| Resident work | Padded 1080p GPU | Padded 4K GPU |
| --- | ---: | ---: |
| First scored iteration | `15.659 ms` | `59.004 ms` |
| Second scored iteration | `14.142 ms` | `54.272 ms` |
| Final-frame coefficient work | `1.891 ms` | `3.394 ms` |

This makes the effort-7 update-policy change the largest isolated candidate.
The `54.27 ms` second-iteration figure is a GPU-stage cost, not a promise of an
equal end-to-end saving. Queueing, CPU overlap, and the replacement policy path
must be measured in a matched build.

Butteraugli's largest individual stage is `butteraugli.psycho.main` at
`31.807 ms` aggregated over the 4K resident evaluations. Within AC search, the
largest strategy stages are DCT16 (`13.168 ms`) and DCT32 (`11.400 ms`). The
four rectangular 16x8, 8x16, 32x16, and 16x32 families together account for
about `29.45 ms`.

## CPU attribution and scaling

The leading encoder-relevant Samply leaves were:

| Flat sampled thread CPU | Function or activity |
| ---: | --- |
| `10.70%` | `WriteAnsTokenStream` |
| `9.10%` | direct AC-token append and balanced-population accumulation |
| `6.63%` | `memmove` |
| `5.31%` | direct AC-group tokenization outside the append helper |
| `5.07%` | coefficient-order computation |
| `3.85%` | SIMD linear-RGB-to-Opsin inner loop |
| `3.51%` | HybridUint encoding |
| `3.47%` | zero filling |
| `3.14%` | owned VarDCT frame assembly |

These values overlap wall stages through parallel execution. In particular,
rANS section emission is expensive in aggregate CPU but is distributed across
independent sections.

The 4K thread sweep confirms which codestream work already scales:

| CPU budget | Total | Codestream | Coefficient order | AC tokenization | Entropy | Section writing |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `481.60 ms` | `133.28 ms` | `15.64 ms` | `55.28 ms` | `11.55 ms` | `58.09 ms` |
| 2 | `424.78 ms` | `85.32 ms` | `14.99 ms` | `35.74 ms` | `7.88 ms` | `31.54 ms` |
| 4 | `397.24 ms` | `62.71 ms` | `15.96 ms` | `27.50 ms` | `7.92 ms` | `17.22 ms` |
| 8 | `385.28 ms` | `50.37 ms` | `15.68 ms` | `23.42 ms` | `7.97 ms` | `10.83 ms` |
| auto, 12 peak participants | `386.77 ms` | `52.14 ms` | `16.10 ms` | `24.29 ms` | `7.99 ms` | `10.87 ms` |

AC tokenization and section writing benefit strongly through eight workers.
Entropy construction saturates by two workers. Coefficient-order work is flat,
which identifies it as the clearest remaining serial codestream stage.

The apparent one-process advantage for an 8-worker cap did not reproduce.
Across seven alternating independent-process pairs, the 8/auto median paired
ratio was `1.0151` for complete encode with only one 8-worker win, and `0.9997`
for codestream encoding with four wins. Automatic scheduling should remain the
default.

## Recommended optimization sequence

### 0. Independently owned: effort-7 AQ update policy

The current mapping assigns two adaptive-quantization iterations to effort 7.
Removing the second scored evaluation targets `54.27 ms` of measured 4K GPU
work. Qualification must cover encoded size, decoded pixels, Butteraugli,
determinism, target-size search, maximum-error behavior, and normal natural
images. The result must be measured at the complete public-workflow boundary.

### 1. Lease exact-capacity evaluator arenas (completed)

Status: implemented, memory-pressure-qualified, and retained. Prepared
Butteraugli storage remains deliberately per-encode.

`frontend.prepare_evaluator` is the largest non-evaluation wall boundary at
`80.73 ms` profiled 4K. Only `22.05 ms` is attributed to the submitted
Butteraugli-reference GPU stages. Samply independently shows material time in
buffer uploads, memory copying, zeroing, virtual-memory advice, allocation,
and prepared-state destruction.

Add nested timers for:

1. input validation and buffer-size planning;
2. strategy, anchor, transform-layout, and color-correlation metadata packing;
3. persistent and staging arena allocation;
4. host-to-device population by buffer class;
5. Butteraugli-reference encoding and submission;
6. waits caused by reference-submission dependencies; and
7. arena and prepared-object teardown.

If allocation and teardown remain material, implement the leaseable workspace
pool described in `amdahl-leakage.md`. Reuse capacity, not image identity:

- acquire and return a workspace under a short lock, while keeping encode and
  GPU work outside the lock;
- allow concurrent calls and give each batch worker a natural lease;
- key leases by backend and required capacities rather than raw view pointers;
- explicitly reset image generation, uploads, and Butteraugli-reference state;
- discard poisoned leases after device, upload, or completion failures; and
- enforce a high-water policy so unusually large images are not retained by
  every worker indefinitely.

The difference between the `80.73 ms` wall boundary and `22.05 ms` of reference
GPU work is only an upper bound on removable overhead. The substages must prove
the causal saving before the pool is retained.

#### First workspace-lease experiment

The first prototype leases only the two `DeviceScratchArena`s owned by
`MetalPreparedAqEvaluation`: its persistent arena and its operation-staging
arena. The production backend holds at most one idle arena of each class.
Acquisition and return use a short mutex; preparation, uploads, submissions,
waits, and host work remain outside it, so concurrent calls still receive
exclusive storage rather than serializing on the pool.

The lease key is the exact planned capacity as well as the backend and arena
class. This prevents a full diagnostic evaluator from inflating the memory
contract of a later frame-only evaluator. Every successful preparation resets
the slice layout and reuploads image-specific state. An evaluator invalidated
by an upload, device, numeric, readback, or completion failure discards both
arenas. Each arena also has a `1 GiB` retention ceiling, and concurrent returns
retain the smaller arena when only one idle slot is available.

Returned Metal buffers are marked purgeable-volatile. Acquisition first locks
the resource by restoring the nonvolatile state. If Metal reports that the
resource became empty under memory pressure, acquisition destroys the stale
resource and performs a cold allocation; its contents are never consumed.
Focused coverage forces this transition, checks the cold-allocation count, and
compares the recovered evaluation with a fresh changed-image oracle. This uses
Apple's documented
[`MTLResource::setPurgeableState` contract](https://developer.apple.com/documentation/metal/mtlresource/setpurgeablestate%28_%3A%29).

This is deliberately narrower than caching a complete `PreparedWorkflow`.
Host metadata, image generations, resident-view identity, AC-search resources,
and the prepared Butteraugli allocation remain per-image. The latter is
especially important: extending the lease to Butteraugli could remove another
large allocation, but would also keep its 33-working-plane allocation alive
while the backend is idle.

Five alternating independent-process pairs compared the parent and prototype
Release builds. Each process performed one warmup and three retained samples;
the table reports the median of the five per-process medians:

| Workload | Stage | Parent | AQ arena leases | Change |
| --- | --- | ---: | ---: | ---: |
| padded 1080p | Complete encode | `105.917 ms` | `96.092 ms` | `-9.825 ms` (`-9.3%`) |
| padded 1080p | Quantization pipeline | `82.100 ms` | `72.690 ms` | `-9.410 ms` (`-11.5%`) |
| padded 4K | Complete encode | `371.616 ms` | `340.145 ms` | `-31.471 ms` (`-8.5%`) |
| padded 4K | Quantization pipeline | `309.003 ms` | `274.584 ms` | `-34.419 ms` (`-11.1%`) |

All ten paired complete-encode comparisons favored the prototype. Every sample
preserved the expected encoded size: `410,072` bytes at 1080p and `1,606,911`
bytes at 4K.

A separate five-sample stage-profile diagnostic reduced
`frontend.prepare_evaluator` from `19.872` to `14.675 ms` at 1080p and from
`71.750` to `54.586 ms` at 4K. The Butteraugli-reference GPU work remained
`5.505` versus `5.457 ms` and `22.428` versus `22.094 ms`, respectively, so
the change is removing host allocation/lifetime cost, not GPU perceptual work.
The larger quantization-pipeline saving also includes avoided arena destruction
outside the preparation wall stage and must not be subtracted mechanically
from the stage-profile result.

Three alternating live-context measurements paused each process only after all
encoding had completed, while keeping the same backend alive. Without memory
pressure, median physical footprint increased from `160` to `425 MiB` at 1080p
and from `169` to `1,226 MiB` at 4K. This is real idle cache residency which the
earlier peak-RSS measurement could not expose.

The volatile state makes that memory reclaimable rather than immediately
discarding it. A controlled `memory_pressure -p 60` run reduced the paused 4K
candidate from `1,226` to `76 MiB`; the parent moved from `168` to `81 MiB`
under the same procedure. A second encode on the pressured candidate detected
the emptied leases, reallocated them, and reproduced the same `1,606,911`-byte
codestream. It was predictably cold (`379.851 ms` after a `340.156 ms` warm
sample), so the lease is a latency cache rather than guaranteed residency.

This closes the AQ-arena experiment but rejects adding the much larger prepared
Butteraugli allocation. Two AQ arenas already produce a substantial no-pressure
high-water mark. Pooling another 33 working planes is not justified without a
broader cache budget, eviction policy, and an explicit application trim seam.

### 2. Parallelize coefficient-order zero statistics

`ComputeSimpleCoefficientOrdersForEncoder()` scans every AC transform into
per-family, per-channel zero counts on one thread. It takes `17.28 ms` in the
independent-process 4K cohort and remains around `15-16 ms` throughout the
thread sweep.

A bounded design is:

1. partition AC groups over the existing CPU budget;
2. give each worker compact per-family, per-channel integer counts;
3. reduce counts in deterministic group or worker order;
4. preserve the current stable tie behavior and emitted order bytes; and
5. run Lehmer tokenization only after the reduction.

For effort-7 DCT8-only sampling, precompute sampling decisions in the current
serial traversal order or otherwise prove identical xorshift consumption.
Efforts 8-10, high density, and maximum compression must retain their full
statistics. The integer reduction itself is exact; overflow and allocation
failures must remain atomic.

An ideal four-way result would save roughly `13 ms`, but memory bandwidth and
worker setup will reduce that ceiling. The actual acceptance criterion is a
repeatable complete-encode improvement with unchanged bytes outside the
already-qualified effort-7 sampling case.

### 3. Optimize AC-strategy kernels, not the effort-7 search policy

The AC-strategy submission accounts for `59.01 ms` of 4K GPU time and a
`73.29 ms` host wait. This is genuine accelerated work on the critical path,
not a hidden CPU fallback.

The candidate family is grounded in libjxl's effort-7-like hierarchical
DCT8/16/32 and rectangular merge search. Removing families would be a density
or quality policy change, not a routine optimization. Preserve the candidate
set while investigating:

- reuse of intermediate transforms across related candidates;
- redundant gather, residual, or quantization work between hierarchy levels;
- SIMD-group occupancy and register pressure in DCT16/DCT32 kernels;
- rectangular-kernel memory layout and coalescing; and
- whether candidate batches can share scratch without extra dispatches or
  synchronization.

Because dispatch timestamps are unavailable, use Metal System Trace/counters
or temporary in-kernel instrumentation before attributing a cause to occupancy,
bandwidth, or cache behavior. A kernel improvement must reduce the complete AC
submission and public workflow, not merely an isolated timestamp.

#### Expected payoff from moving hierarchical selection to Metal

The hierarchical decision itself is a much smaller target than the candidate
evaluation above. The measured CPU/GPU handoff after candidate evaluation is:

| Boundary component | 1080p | padded 4K |
| --- | ---: | ---: |
| Candidate-cost readback | `0.125 ms` | `0.509 ms` |
| CPU hierarchical decision | `0.789 ms` | `3.382 ms` |
| AQ reconfiguration | `0.235 ms` | `1.085 ms` |
| Gross handoff boundary | `1.149 ms` | `4.976 ms` |

Moving only `SearchTile`'s decision logic to a GPU kernel, then reading the
selected strategy grid back, would likely save about `2.5-3.5 ms` at 4K. A
fully resident continuation that produces the selected strategy records,
grouped anchors, and metadata in the representation consumed by AQ could save
about `4-6 ms` at 4K. The corresponding 1080p opportunity is roughly
`0.7-1.0 ms`; either version is only around a one-percent complete effort-7
encode improvement. A larger `6-8 ms` 4K result is possible if indirect
dispatch also removes an otherwise-idle submission gap, but that is speculative
until dispatch-level tracing exists.

This does **not** eliminate the `73.29 ms` AC wait: `59.01 ms` of that interval
is candidate-evaluation GPU work required before any winner can be chosen. The
change removes the post-evaluation readback/CPU-decision/AQ-setup bubble, not
the underlying search cost.

The computation is GPU-suitable at the outer level: padded 4K has about 2,040
independent 8-by-8-block search tiles. Within each tile, however, the exact
hierarchical merge order is serial and its additions, comparisons, and tie
rules must be preserved if byte-identical output remains a requirement. A
small decision kernel is therefore expected to cost roughly `0.2-0.8 ms` at
4K, but that is a projection rather than a measurement.

The decision-only form is a modest optimization and may not justify a new
materialization path by itself. Prefer combining it with a resident strategy
representation that AQ and the final group-major packing stage can consume;
otherwise much of the measured AQ reconfiguration cost is merely moved rather
than removed.

### 4. Pack the serializer's group-major frame in the final submission

Mapped-source handoff removed the intermediate coefficient readback, but host
assembly still takes `16.70 ms` at 4K. It allocates and zero-initializes the
owned group-major AC store, validates transform metadata, and repacks
strategy-batch-major shared coefficients into serializer order.

Prototype a Metal group-packing stage that:

- executes in the already-existing final command buffer;
- writes the stable group-major serializer layout;
- preserves deterministic coefficient and padding bytes;
- exposes the completed shared buffer only after the existing wait; and
- retains atomic construction of the public owned frame.

Also measure a narrower host-only experiment that avoids zeroing coefficient
ranges which are immediately overwritten and clears only required group tails.
The current `bzero` attribution shows potential, but changing the owned storage
type merely to obtain uninitialized allocation may not justify its complexity.

A mapped zero-copy serializer view remains architectural work. It would retain
a backend buffer lease through tokenization and section writing and would add
coherence, cancellation, and concurrent-call lifetime rules. It should follow,
not precede, the group-packing experiment.

### 5. Re-profile the remaining codestream tail

After coefficient-order parallelization, the likely residuals are:

- direct AC tokenization: `24.83 ms` wall at 4K;
- section writing: `10.77 ms` wall;
- ordinary entropy construction: `8.02 ms` wall; and
- DC tokenization and final assembly: small single-digit or sub-millisecond
  boundaries.

Possible bounded experiments include reusing reverse rANS chunk storage per
section worker and retaining already encoded balanced HybridUint symbols for
final emission. The latter trades more token memory and memory traffic for less
integer work, so it needs a matched prototype rather than an assumed win.

Do not redesign rANS for Metal at this stage. It is serial within each stream,
the encoder already parallelizes independent sections, and a GPU design would
add transfer, dispatch, and synchronization to a comparatively small wall
boundary.

### 6. Combine RGB-to-Opsin with resident input ownership

Input preparation is `12.53 ms` at 4K, and the SIMD RGB-to-Opsin loop accounts
for `3.85%` of sampled aggregate CPU. A standalone GPU color transform would
add an input upload and dispatch, so moving only the arithmetic is unlikely to
be compelling.

The stronger design would upload original linear RGB once, retain it for the
Butteraugli reference, and produce coding Opsin into resident Metal storage.
That can remove both the CPU transform/write and redundant original-image
population. Revisit this only after evaluator-preparation substages identify
the current upload and allocation costs.

## Explicitly rejected or deferred directions

- **Cap automatic CPU scheduling at eight workers:** rejected by seven
  independent pairs; complete encode was worse in six.
- **Prune AC strategies at effort 7 without a quality study:** rejected as
  ungrounded; the current hierarchical search is derived from libjxl behavior.
- **Move entropy coding to Metal:** not justified by the remaining wall budget.
- **Optimize the exact-coefficient compatibility path first:** it is not the
  ordinary default and should not drive the resident architecture.
- **Introduce a zero-copy frame view before group packing:** deferred because
  its lease lifetime and concurrency contract are substantially broader than
  its current `16.70 ms` target.

## Qualification gates

Every retained change must use fresh Release builds, alternating independent
processes, warmups, multiple samples, and representative 1080p and 4K inputs.
Report complete public-workflow wall time separately from stage wall time,
aggregate worker CPU, and GPU timestamps.

Functional coverage must include:

- efforts 7, 8, 9, and 10 plus explicit maximum compression;
- forced CPU, forced Metal, and automatic fallback;
- fully resident, exact coefficients, throughput, and maximum throughput;
- target-size retries and maximum-error control;
- final Butteraugli diagnostics enabled and disabled;
- explicit and automatic CPU thread budgets;
- batch workers and concurrent calls on one C API context;
- allocation, upload, device, and completion failures; and
- deterministic bytes where behavior is intended to remain identical.

Independently decode representative candidate codestreams with pinned `djxl`
and verify decoded pixels, encoded size, and Butteraugli where a policy or
ordering change can alter bytes.

The integrated Release suite passed all 61 applicable tests. The separately
known pinned CPU `quantization_pipeline` golden mismatch remains excluded and
was not introduced by the integration.
