# Resident frame handoff

This document records completed milestones 1 and 2 of the broader
[resident execution refactor](resident-execution.md). That roadmap owns the
remaining integration, resource-admission, reuse, and scheduling work and its
completion criteria. The implementation and measurements below retain their
original milestone baselines.

## Milestone 1: ownership-independent consumers

This refactor removes the owned-frame requirement from the internal codestream
consumer interface. It does not yet eliminate Metal's final frame assembly or
change the GPU coefficient layout. AQ iterations, AC selection, quantization,
coefficient-order policy, and entropy policy are unchanged.

Base revision: `4ea12ab` (the Metal fusion merge). Original development branch:
`refactor/resident-frame-handoff`, now named `refactor/resident-execution`;
worktree: `../gjxl-resident-frame-handoff`.

## Interface and ownership

`codec/vardct_frame_view_internal.h` defines `VarDctFrameViewData` and
`VarDctFrameView`. The view copies small value metadata and borrows strategy,
quantizer, color-correlation, raw-quantization, sharpness, DC, and AC storage.
It neither allocates storage nor materializes coefficients. Block planes may
have independent row strides. AC storage retains the existing fixed-capacity,
group/channel-major layout, row-major transform anchors, and zero edge tails.
The decoder-equivalent floating DC cache is still required and validated; its
removal or lazy construction is not part of this milestone.

`BorrowFrame(const VarDctEncoderFrame&)` adapts existing owned frames without
copying their planes. It checks owned allocation sizes before exposing core
plane views; value validation is shared with independently backed views.
Borrowing temporary owners is explicitly rejected at compile time. The public
owned-frame APIs and class storage layout remain intact.

`codestream_internal::EncodeVarDctCodestreamFromView` is the synchronous internal
entry point, with optional profiling. Serializer helpers, coefficient-order
derivation, block-context derivation, and DC/AC tokenizers consume the common
view. Public owned-frame entry points adapt to these implementations. The
shared validator preserves structural and initial-profile gates, and failed
serialization leaves both codestream output and optional profile unchanged.
Trusted `ForEncoder` helpers continue to require previously validated inputs.

The zero-tail invariant uses an exact unsigned integer-OR reduction. This avoids
a scalar early-exit scan introduced by the first view-based validator; the
final compiler output vectorizes the check. Negative coefficients and both ends
of every partial group's channel tail are covered by rejection tests. No
floating-point encoding calculation or policy is changed by this check.

The borrowing contract is explicit:

- Every backing allocation and referenced metadata object stays alive,
  address-stable, and immutable until the last consumer returns.
- Device-backed producers must complete before the view is consumed. A future
  output lease must prevent writes, reuse, and purging until serialization and
  every serialization worker have finished.
- The view does not perform synchronization, retain a device lease, or extend
  lifetimes when copied. As with core plane views, callers supply sufficient
  live backing for every addressed row; validation cannot detect dangling or
  undersized raw-pointer allocations.
- Completed output lifetime is distinct from transient AQ scratch lifetime.
  This interface does not require keeping an entire evaluator arena alive.

## Correctness qualification

Both builds use AppleClang 17, Ninja, Release, tests and benchmarks enabled, and
optional libjxl-reference fixtures disabled. `build/baseline` was built from the
unmodified parent before source edits and was not rebuilt afterward.

- Parent full CTest: **61/62**. Final candidate full CTest: **62/63**, including
  the new `vardct_frame_view` test and installed C/C++ consumer checks.
- Both fail only the inherited CPU `quantization_pipeline` golden:
  `0.24919039011001587` actual versus `0.24914586544036865` expected.
  This was reproduced on the fresh parent, not inferred from older records.
- The new test covers tiny/padded images, mixed transforms, multiple AC/DC
  groups, quant-field context splitting, independent strided backing,
  borrowed-pointer identity, all three entropy modes, full/sampled coefficient
  orders, 1/8 CPU-thread budgets, and concurrent readers. External backing is
  serialized after the original owned frame is destroyed.
- Invalid/missing metadata, malformed planes and group storage, nonzero tails,
  incorrect DC reconstruction, unsupported profiles, cross-group strategies,
  and output preservation are checked. AddressSanitizer and UBSan passed this
  test using a separate RelWithDebInfo build.
- All **38 canonical images** (Kodak, natural 1080p/4K photographs, and padded
  stress images) produced byte-identical parent/candidate codestreams at Metal
  fully-resident, distance 1.2, effort 7.
- **18 additional policy cases** matched bytes: efforts 1–10, high density,
  maximum compression, final-score diagnostics, exact coefficients, both
  throughput modes, target bytes, and CPU maximum error. Target bytes uses the
  small repository sample; the other additional cases use Kodak17.
- Both builds passed all **22 full pinned conformance fixtures**, including
  stored codestream hashes, decoded-reconstruction comparisons, ordinary
  entropy variants, workflow cases, and corruption checks. Decoder:
  `djxl` 0.13.0, revision `e8ff09762481785938d8e4e01333ed3917571161`.
- Independently decoding parent and candidate Kodak17, planter 4K, and padded
  stress 4K also produced identical PFM bytes. The Metal libraries themselves
  are byte-identical; no shaders changed.

Local qualification artifacts are under `build/qualification/`: parent and
candidate CTest/conformance logs, `parity.json`, retained codestreams and decoded
PFMs, and the qualification runner. These are local build artifacts, not durable
repository fixtures. `parity.json` records input and executable hashes and exact
commands; the permanent regression coverage is in `tests/vardct_frame_view_test.cpp`.

## Performance qualification

Apple M4 Pro, 48 GiB, macOS 15.6. For each workload, seven alternating
parent/candidate process pairs used two warmups and five retained samples per
process. Both builds used fully-resident Metal, tuned AC kernels, SIMD CPU
implementation, distance 1.2, effort 7, automatic CPU threads, and Metal-only
validation. Timings are profiled public-workflow wall time, excluding input
generation/loading and process startup; they are not GPU duration or aggregate
worker time. Corpus byte/decoder comparisons ran separately from timing.

| Workload | Parent total | Candidate total | Change | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| Padded 1080p, 1919×1079 | 83.799 ms | 82.789 ms | -1.20% | -0.72% |
| Padded 4K, 3839×2159 | 282.683 ms | 282.309 ms | -0.13% | +0.06% |
| Kodak17, 512×768 | 24.619 ms | 24.795 ms | +0.71% | +0.01% |
| Planter 4K, 3840×2160 | 290.945 ms | 291.636 ms | +0.24% | +0.13% |

The total columns are medians of the seven process medians. Paired change is
the median of the seven candidate/parent ratios, not a ratio of aggregate
worker counters. These results support **approximately neutral end-to-end
performance**, not a meaningful overall speedup. Codestream wall time changed
by -0.68%, -1.54%, -0.79%, and +0.52%, respectively, using process medians.

The initial view validator made the large-image validation phase roughly
0.2 ms slower by losing the parent's unrolled zero-tail scan. The final exact
integer reduction removes that regression: validation changed from
0.356→0.240 ms (padded 1080p), 0.721→0.622 ms (padded 4K), and
0.733→0.659 ms (planter 4K). Kodak17 validation is tiny and changed from
0.023→0.027 ms. These are substage observations, not additional overall gains.

All timing samples retained identical codestream sizes within and across
builds for each workload. Recorded process-boundary snapshots showed no other
GJXL encode/benchmark/test jobs. Raw samples, executable hashes, exact commands,
and process-pair summaries are retained in `build/qualification/performance/`,
`performance.json`, and `performance-summary.json`.

Example command (substitute `baseline` for the parent):

```sh
build/release/gjxl_encoding_benchmark --workload padded_4k \
  --scope metal-public-workflow --implementation simd \
  --gpu-aq fully-resident --validation metal-only \
  --distance 1.2 --effort 7 --cpu-threads auto \
  --warmups 2 --samples 5 --raw-samples /tmp/frame-handoff-samples.json
```

For natural inputs, replace `--workload padded_4k` with `--input INPUT.pfm`.

## Milestone 2: completed Metal output

Milestone 1 was committed as `ca440d1`. Milestone 2 connects the resident
Butteraugli encoding workflow to an exclusive `CompletedVarDctFrame` lease.
The serializer and the workflow's strategy summary both read its frame view.
The existing public owned-frame and diagnostic paths remain available; exact
coefficients, maximum-error control, and maximum throughput retain their owned
handoffs. No AQ iteration, coefficient decision, AC search, entropy policy,
thread admission, or batch scheduling policy is changed.

The final coefficient kernel writes AC coefficients directly to the existing
group/channel-major serializer layout. Transform dispatch and floating-point
reconstruction remain strategy-batch-major. A destination table maps each batch
anchor to its group-local, row-major transform offset. It is constructed from
the authoritative final strategy grid for each output request, including after
reconfiguration. Both final-score and final-frame-only requests use direct
output writes. Intermediate AQ iterations and the owned diagnostic path keep
their previous layout.

There is no GPU packing kernel, extra submission, or full AC readback/repack into
`VarDctEncoderFrame`. Only unused fixed-capacity group tails are initialized on
the host; the final quantization dispatch writes every used coefficient.
Integer DC and adjusted raw quantization are snapshotted at block resolution.
Floating DC is reconstructed using the same CPU steps as owned assembly, and
strategy, sharpness, quantizer and final CfL metadata are independently owned.
These small copies are intentional: they sever the output's dependence on AQ
scratch without requiring a separate GPU metadata-copy pass.

The lease owns one non-purgeable shared Metal allocation containing final AC
coefficients and its destination table, plus the block-resolution metadata.
Its allocation is bounded by frame geometry: `3 * group_count * 65536` int32
coefficients and one uint32 destination per transform anchor. It retains no
backend, prepared evaluator, submission, or scratch arena. It is published only
after the existing synchronous GPU completion and output validation. Failed
requests preserve the previous caller output. It is freed when the encoding
attempt's consumers finish; no new idle cache is introduced.

The existing prepared workflow may still retain its evaluator for target-size
attempt reuse. That is an independent preparation lifetime, not a requirement
of the completed frame. Releasing or reusing the evaluator does not invalidate
the output. Existing batch in-flight limits remain the workflow admission
boundary; this milestone adds no process-wide active-memory budget.

### Relationship to `perf/metal-preparation`

The separate worktree's `e1010fe` shares transient AQ storage with Butteraugli
and defers host preparation; `306f153` adds a bounded volatile preparation cache.
At the milestone-2 checkpoint those changes were inspected, not merged. This
output allocation does not alias
their borrowed filter/gather planes or participate in their scratch caches.
The two changes are architecturally complementary. Subsequent integration and
combined qualification are recorded in [milestone 3](resident-execution-integration.md).
Current milestone-2 measurements use `ca440d1` as their baseline and do not
include the preparation branch's speed or memory gains.

### Qualification

Permanent tests exercise direct output against owned output with exact DC/AC,
raw quantization, reconstructed DC, score-history and codestream comparisons.
They cover tiny/padded images, partial AC groups, multiple DC groups, all seven
supported transforms, reconfiguration away from a provisional DCT8 grid,
0/2-update policies, final-score on/off, stage-profiled/unprofiled parity,
concurrent reads during evaluator
reuse, forced idle-scratch reclamation, and serialization after both evaluator
and backend destruction. Simultaneous owned/leased output requests are rejected
before allocation/submission. Upload, submission, completion, device-numeric,
and readback failures preserve an existing lease and diagnostic outputs.

Qualification artifacts and commands are retained locally under
`build/handoff-qualification/`. The fresh committed-milestone-1 build is frozen
at `build/milestone1`; `build/baseline` remains the older pre-milestone-1 build.

Final Release CTest passes **63/64**, versus **62/63** on milestone 1. Both
retain only the CPU `quantization_pipeline` mismatch documented above; no
golden or tolerance was changed. The completed-frame, AQ-evaluation and full
GPU-pipeline tests also pass **3/3** with AddressSanitizer and UBSan. Leak
checking is disabled, and UBSan null checks are suppressed only in the vendor
`third_party/metal-cpp` headers: their nil retain/release wrapper fails before
encoding in the unsuppressed run. That failure and the narrowly scoped
suppression are retained. No GJXL source checks are suppressed.

All **56** corpus/policy cases match milestone-1 codestream bytes, including
efforts 1–10, final-score diagnostics, throughput, exact coefficients, and
target-size retries. Independent pinned decoding of Kodak17, planter 4K and
padded stress 4K matches PFM bytes. Pinned full conformance continues to pass
all **22** fixtures. These output checks are separate from the timing matrix.

### Complete-call performance and memory

Apple M4 Pro, 48 GiB, macOS 15.6; fresh Release libraries, SIMD/fused-tuned Metal,
fully resident, distance 1.2, effort 7, automatic CPU threads. The driver is
adapted from the preparation worktree's complete-call harness. It measures the
entire synchronous encode call **including evaluator teardown**, excluding
backend creation, image loading, hashing and output writes. This boundary is
slightly wider than the milestone-1 public-workflow profile table above.

Seven independent process pairs alternate parent/candidate order for each
workload. Each process alternates the named input and a deterministically
changed-image companion, discards its first three encodes, and retains six
encodes per image. The table shows the named input's medians of seven process
medians and the median of seven paired latency ratios. It is not a ratio of
aggregate worker counters. All 840 encodes retain identical per-image output
hashes and sizes across variants/repetitions; process-boundary checks found no
other GJXL encoding jobs during the accepted timing matrix.

| Input | Milestone 1 | Milestone 2 | Median paired latency change |
| --- | ---: | ---: | ---: |
| Padded 1080p, 1919×1079 | 85.104 ms | 84.047 ms | -0.59% |
| Padded 4K, 3839×2159 | 294.037 ms | 286.912 ms | -2.19% |
| Kodak17, 512×768 | 25.452 ms | 24.731 ms | -2.58% |
| Planter 4K, 3840×2160 | 300.730 ms | 292.970 ms | -2.76% |

The changed-image companions improve by 1.56%, 2.63%, 2.14% and 3.36%,
respectively. The primary 1080p case improves in 5/7 pairs; the other primary
inputs improve in 7/7. This supports a modest end-to-end gain on this cohort,
not a general corpus-wide performance claim.

The downstream cost matters: primary padded-4K serializer wall time increases
from 38.088 to 39.963 ms, and planter from 44.419 to 46.593 ms, while their
quantization-pipeline wall time decreases from 241.143 to 233.313 ms and
241.592 to 232.107 ms. The benchmark does not establish the hardware cause of
the serializer slowdown. Eliminating the AC copy does not make all its saved
host time an end-to-end win.

A separate padded-4K stage-profile probe (one warmup, three samples per build)
retains **five submissions and 341 dispatches** in both builds. Median host
frame assembly changes from **11.894 ms** to **0.800 ms** of mapping/metadata
finalization plus **0.582 ms** of output preparation. Final-frame GPU stages
are **3.399→3.489 ms** in this small probe. These instrumented substage numbers
explain the handoff boundary; they are not additional independent end-to-end
performance samples. The new `resident.frame_output_prepare` wall stage records
allocation, destination-table construction and tail initialization explicitly.

Three alternating memory-probe pairs at padded 4K retain two source images,
perform three encodes and sample the task physical-footprint counters. Median
process peak is **3344.4→3347.0 MiB**; footprint one second idle with the backend
alive is **1194.3→1196.3 MiB**. These are effectively unchanged, not a peak-memory
reduction or a claim that the existing AQ caches retain no memory. The output
lease itself is not cached. Freshly relinking the final Release libraries
reproduces the timed candidate driver byte-for-byte; commands, library and
driver hashes, all process samples, and memory logs are retained with the
qualification artifacts.

## Later milestones

The [resident execution roadmap](resident-execution.md#milestones-and-dependencies)
defines the remaining milestones:

- Integrate and jointly qualify the preparation branch with completed output.
- Establish coordinated resource accounting and memory admission.
- Audit and qualify further policy-preserving reuse and lifetime reductions.
- Coordinate CPU participation and CPU/GPU scheduling across admitted images.

These extend original proposals #3, #4A, and #5. Candidate pruning and selective
refinement (#4B) remain a separate policy track, not a prerequisite for this
refactor. Milestone 2 removes the compulsory full AC copy; it does not eliminate
every frame-metadata copy, every intermediate coefficient allocation, or the
CPU entropy/codestream tail. Small metadata snapshots remain intentional where
they keep completed output independent of large temporary arenas.
