# Final resident scheduling qualification

Status: **complete**, with the costs and limitations below. This record closes
[milestone 6](resident-execution.md#6-coordinated-cpugpu-scheduling--complete)
and the bounded policy-preserving structural effort (milestones 3–6).

## Comparison and boundaries

The comparison is the integrated preparation/handoff baseline
`ec4d4c5317d983b5f6df29d2474923443c9cfe63` against final runtime `07dd92e`.
It therefore measures the combined accounting, planning/admission, last-use and
scheduling work after integration, not the CPU coordinator alone. Historical
checkpoint percentages must not be added to these results.

Both builds are Release, Apple Clang 17 (`clang-1700.6.4.2`), with tests and
benchmarks enabled, optional libjxl reference and Metal profiling disabled.
Hardware is Apple M4 Pro / Mac16,7, 48 GiB, 14 logical CPUs, macOS 15.6 (24G84).
The candidate uses Ninja. The complete archived baseline uses Make; its known
generated-shader ordering defect was repaired by regenerating the embedding
after shader completion and relinking, as recorded in
[the CPU qualification](resident-cpu-coordination.md). Its runtime source is not
substituted with candidate source. Both linked shader payloads are identical,
SHA-256 `9dcbc4dff15fd81c0793a1df3d68fd06ac821d0f34929a0e397f1de4e9cafae4`.

The identical qualification driver is separately compiled against each revision's
complete matching headers and libraries. No current C++ layout is linked against
an old archive. Frozen source, library, driver, corpus and decoder identities are
checked before each phase. The baseline predates managed-resource accounting and
the new CPU-domain API: its managed counters and image scheduling fields are
**unavailable**, not zero. Process physical-footprint measurements are available
for both revisions.

Every performance call forces fully-resident Metal, effort 7, distance 1.2.
The default managed limit is unlimited but accounted on the candidate; automatic
shared CPU participation resolves to 14. Per-image CPU settings are automatic
except the explicitly named one/four-thread cases. Batch in-flight count remains
a separate control. Backend availability/setup is timed and reported separately.

Inputs are prepared before timing. Natural inputs are canonical linear-RGB PFMs;
synthetics use the repository benchmark's deterministic generator. Each process
also makes an independently owned changed version, `0.91f * x + 0.017f`, and
alternates original/changed images. Input float-bit fingerprints and output
size/hash sequences must agree between revisions, including warmups.

- Single-image and single-caller batch timers surround the complete public API,
  including its own planning, admission, evaluator teardown and publication.
  Driver construction, external pressure-cap forecasting, input preparation,
  output validation/hashing and file I/O are outside that timer.
- Two-caller cohorts use two precreated driver objects and two gated caller
  threads. Each public call has its own wall time. The cohort timer spans gate
  release through joining both callers; it additionally includes dispatch and
  thread-exit/join overhead. It is not labeled single-image latency.
- Queue is batch arrival through initial image CPU admission. Service is that
  admission through internally retained-result readiness, including teardown.
  Ready equals queue plus service, and must not exceed the enclosing API wall
  time. It precedes whole-array publication. Distributions are retained pooled
  and by source shape; summed worker or GPU durations are not latency.
- Physical footprint uses `task_vm_info`. Peak is the process-lifetime physical
  high-water mark, including inputs, backend and benchmark infrastructure.
  Backend-alive idle is sampled immediately after output release with drivers
  still alive. Post-trim follows synchronous driver stop/destruction and cache
  trimming; backend and inputs remain alive. Candidate stopped driver objects
  remain allocated until process exit; baseline drivers are destroyed before
  trim. Neither measurement is a managed-memory cap or one-second-idle result.

## Correctness and sanitizer gates

The frozen final suites reproduce only the known `quantization_pipeline` golden
mismatch: baseline **63/64**, candidate **115/116**. Actual is
`0.24919039011001587`, expected `0.24914586544036865`. Thirteen selected CPU,
admission, publication, completed-frame, cache, lifecycle and scheduling tests
also pass three repetitions each. Installed-consumer coverage is part of the
full candidate suite. No numerical threshold is widened.

All **56 corpus/policy comparisons are byte-identical**: 38 canonical PFMs,
efforts 1 through 10, high density, maximum compression, final score,
exact-coefficient/throughput/maximum-throughput modes, target size and native CPU
maximum error. Both revisions pass all **22 pinned conformance fixtures**.
The pinned `djxl` accepts separately retained outputs from both revisions and
produces identical decoded PFMs for Kodak17, natural planter 4K and padded-stress
4K, plus **24 changed-image mixed-batch/two-caller image pairs**.
Decoder SHA-256 is
`9782c3474e41e5e5415da5fc68e7103d27047661385dc5b863dcad50ec4474ac`.

The [launch-failure matrix](resident-worker-launch.md) passes all seven CPU
ThreadSanitizer and unsuppressed ASan/UBSan groups, and all four Metal ASan/UBSan
groups with the existing metal-cpp-only null-call suppression. Against the same
final runtime, CPU scope and CPU-only lifecycle/scheduling additionally pass
ten TSan repetitions each and three unsuppressed ASan/UBSan repetitions each;
Metal-inclusive lifecycle/scheduling pass three ASan/UBSan repetitions each
with that suppression. The unsuppressed vendor report remains at
`Foundation/NSObject.hpp:112:49`. ASan leak detection is disabled and
Objective-C++ ASan/UBSan flags are unchanged. These are scoped sanitizer results,
not an unsuppressed Metal or leak-sanitizer claim.

## Performance and pressure protocol

The final matrix has 27 workloads, seven alternating independent-process pairs
each. Each process makes nine complete calls/cohorts; the first two are warmups,
leaving seven measured samples. The paired statistic compares the two process
medians within a pair, then reports the median of seven percentage changes.
Positive wall-time change means slower. Setup, individual public-call walls,
image queue/service/ready distributions and physical memory remain separate.

Coverage includes 17x9, Kodak17, padded 1919x1079 and 3839x2159, natural planter
4K, mixed shapes, batches of four/eight, in-flight counts one/two/four/eight
where applicable, explicit per-image CPU one/four, and two simultaneous callers.
Before each measured process, the controller refuses launch if another encoder,
test, compiler or build process is active. This is controlled local-process isolation, not a
guarantee that macOS or unrelated applications do no work.

Pressure has 42 independent candidate processes, five samples with two warmups
each. It crosses CPU limits one/three with minimum-fit and full planned limits
for large singles, large batches, mixed batches and simultaneous callers. A
minimum-fit batch must select one work slot. Every call checks protected CPU
capacity and managed committed peak against the configured limits, and checks
zero active reservations/waiters after completion. All managed storage and CPU
participation must be zero after final trim. Bytes must match across caps.

### Complete-call results

All 189 performance pairs completed with matching output sequences and passing
CPU/memory/timing invariants. Wall columns below are medians of the seven warm
process medians, in milliseconds. Percentage change is the **median paired
change**, not the ratio of those two displayed medians. Ranges show all seven
paired changes, not confidence intervals; no outlier is discarded.

| Workload | Integrated ms | Candidate ms | Paired change | Paired range, % |
| --- | ---: | ---: | ---: | ---: |
| 17x9 single | 6.840 | 6.487 | -10.71% | -49.18 to +0.30 |
| Kodak17 single | 22.439 | 22.503 | +0.86% | -2.73 to +5.04 |
| Padded 1080p single | 70.338 | 71.787 | +2.19% | -0.80 to +3.23 |
| Padded 4K single | 238.689 | 237.215 | -0.61% | -5.49 to +2.79 |
| Natural planter 4K single | 255.995 | 257.969 | +0.76% | -2.13 to +2.59 |
| 17x9 batch8 / slot1 | 52.558 | 53.393 | +1.55% | -2.80 to +70.70 |
| 17x9 batch8 / slots4 | 11.728 | 11.879 | +2.59% | -8.52 to +5.86 |
| 17x9 batch8 / slots8 | 7.852 | 8.294 | +5.22% | -1.39 to +9.38 |
| Kodak17 batch8 / slot1 | 182.401 | 181.513 | +0.12% | -2.04 to +1.07 |
| Kodak17 batch8 / slots2 | 125.109 | 126.174 | +1.16% | -0.33 to +1.88 |
| Kodak17 batch8 / slots4 | 107.606 | 107.766 | +0.17% | -1.49 to +0.60 |
| Kodak17 batch8 / slots8 | 110.786 | 110.035 | -0.69% | -1.91 to +0.92 |
| Padded 1080p batch4 / slot1 | 282.128 | 285.908 | +1.34% | +0.22 to +3.15 |
| Padded 1080p batch4 / slots2 | 219.791 | 227.077 | +3.40% | +2.86 to +3.87 |
| Padded 1080p batch4 / slots4 | 220.518 | 220.483 | +0.63% | -0.43 to +2.54 |
| Padded 4K batch4 / slot1 | 1004.553 | 999.066 | -1.80% | -3.03 to -0.55 |
| Padded 4K batch4 / slots2 | 815.402 | 850.329 | +4.45% | +2.55 to +5.89 |
| Padded 4K batch4 / slots4 | 807.099 | 811.921 | +0.44% | -0.30 to +1.58 |
| Mixed batch4 / slot1 | 399.962 | 399.137 | -0.21% | -2.15 to +1.44 |
| Mixed batch4 / slots2 | 324.499 | 324.835 | +1.07% | -2.04 to +1.83 |
| Mixed batch4 / slots4 | 318.198 | 316.670 | -1.60% | -3.50 to +1.39 |
| Padded 4K single / CPU1 | 341.056 | 350.533 | +3.07% | +0.86 to +4.43 |
| Kodak17 batch8 / slots4 / CPU1 | 107.521 | 109.865 | +1.71% | +1.07 to +3.64 |
| Padded 4K single / CPU4 | 254.723 | 259.237 | +1.71% | +0.77 to +2.81 |
| Kodak17 batch8 / slots4 / CPU4 | 104.939 | 105.953 | +1.14% | -0.39 to +1.59 |
| Two callers, each Kodak17 batch4 / slots4 | 107.414 | 107.244 | -0.13% | -1.11 to +0.43 |
| Two callers, each mixed batch4 / slots4 | 542.984 | 541.823 | -0.26% | -0.61 to +0.29 |

The automatic-CPU natural/padded singles are close to the integrated baseline,
with median paired changes from -0.61% to +2.19%; simultaneous-caller makespans
are effectively unchanged in this experiment. This is **not a general speedup**.
The two-slot padded batches regress consistently by +3.40% and +4.45%, and the
explicit CPU-one/four singles also have consistently positive changes. Keep
those costs visible when adopting the resource/ownership guarantees. This
comparison includes several milestones and does not isolate a particular
allocation, lock, kernel or CPU-coordination mechanism as their cause.

The tiny single and one-slot tiny batch are particularly variable. Their broad
ranges do not justify a robust tiny-image speedup claim. Additional in-flight
slots also do not imply monotonic throughput: Kodak17 is faster with four than
eight slots in both builds. Workload-specific tuning remains appropriate.

For scale, candidate Kodak17 batch8/slots4 is about **74.23 images/s**, and padded
4K batch4/slots4 about **4.93 images/s**, derived from the displayed makespans.
These are throughput figures, not service times or newly introduced overlap.
The corresponding integrated throughputs are about 74.35 and 4.96 images/s.
Backend setup medians across processes are 53.496 ms integrated and 53.736 ms
candidate (ranges 44.449–60.222 and 44.243–60.765 ms), outside the warm-call table.

### Image latency and physical footprint

Representative candidate per-image distributions, pooled across measured calls,
in milliseconds. Mixed workloads also have source-specific distributions in
`performance-report.json`; pooling is not a same-size-image latency claim.

| Workload | Queue median / p95 | Service median / p95 | Ready median / p95 |
| --- | ---: | ---: | ---: |
| Kodak17 batch8 / slots4 | 15.999 / 60.824 | 46.710 / 60.778 | 70.354 / 107.988 |
| Padded 4K batch4 / slots4 | 0.047 / 0.068 | 662.585 / 818.170 | 662.638 / 818.217 |
| Mixed batch4 / slots4 | 0.046 / 0.060 | 136.536 / 326.074 | 136.585 / 326.128 |
| Two Kodak17 callers | 0.051 / 0.098 | 77.645 / 107.427 | 77.694 / 107.475 |
| Two mixed callers | 0.058 / 0.080 | 237.908 / 542.887 | 237.973 / 542.979 |

Service includes GPU waits and any later CPU resumption queueing; the initial
queue field does not measure every internal wait. Each sample satisfies
queue + service = ready and ready <= its own enclosing API wall time. Medians
and p95s of different fields need not add. Baseline image fields are unavailable,
so these are final observability results, not an old/new per-image comparison.

Median process physical footprints across seven processes, MiB, with the exact
boundaries above. Each cell is integrated → candidate:

| Workload | Process peak | Backend-alive idle | Post-trim |
| --- | ---: | ---: | ---: |
| Kodak17 single | 262.3 → 239.5 | 228.0 → 215.3 | 228.0 → 206.8 |
| Padded 4K single | 2973.2 → 2819.5 | 2806.3 → 2307.8 | 2806.3 → 2172.5 |
| Natural planter 4K single | 2921.8 → 2751.2 | 2738.4 → 2245.4 | 2738.4 → 2117.1 |
| Padded 4K batch4 / slot1 | 2971.5 → 2822.7 | 2517.0 → 2308.4 | 2517.0 → 2171.8 |
| Padded 4K batch4 / slots4 | 10605.6 → 10223.0 | 3003.7 → 2310.3 | 2713.1 → 2012.6 |

Single-image 4K peaks decrease by about 5.2% synthetic and 5.8% natural; the
four-slot 4K batch peak decreases about 3.6%. Idle/trim improvements reflect
the combined release and cache-management work, not an isolated scheduler
optimization. Trimming the managed ledger to zero does not imply zero process
footprint: inputs, backend/driver/runtime state and allocator/OS retention remain.

### Pressure results and disposition

All **42 processes, 270 public calls and 990 images** completed. Every CPU
protected peak was at most the configured one/three slots, every managed
committed peak stayed within its hard cap, and every final trim left zero
managed capacity/reservations/waiters and CPU participation. Output signatures
match across all caps. Each minimum-fit batch used one work slot; full planned
limits permitted four per driver. The two-caller full cap covers both drivers,
whereas its tight cap admits one complete minimum-fit batch at a time.

Wall columns are medians of three independent warm process medians (three
measured samples per process), in milliseconds. The CPU limit is **aggregate**;
per-image threads remain automatic. For two callers, wall is cohort makespan.

| Workload / limit | Managed cap, MiB | CPU1 wall ms | CPU3 wall ms | Median physical peak MiB, CPU1 / CPU3 |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K single / tight | 7028.15 | 408.345 | 328.586 | 2829.0 / 2820.5 |
| Padded 4K batch4 / tight | 7612.73 | 1627.719 | 1284.278 | 2832.0 / 2822.9 |
| Padded 4K batch4 / full | 30553.87 | 1111.205 | 826.560 | 9922.0 / 10153.3 |
| Mixed batch4 / tight | 7218.14 | 566.603 | 442.108 | 2881.6 / 2877.4 |
| Mixed batch4 / full | 30159.29 | 427.003 | 341.392 | 3544.7 / 3420.8 |
| Two Kodak17 callers / tight | 473.54 | 229.538 | 203.745 | 242.0 / 242.5 |
| Two Kodak17 callers / full | 3794.43 | 129.990 | 108.829 | 1036.7 / 1067.9 |

The configured cap and committed peak include **unbacked reservation**, not
only allocated storage. These are conservative policy/layout bounds, not fitted
physical-memory measurements. For example, minimum-fit 4K single admission
reserves 7028.15 MiB, while its maximum observed managed backing is 2481.48 MiB
and process physical peak is about 2820–2829 MiB. Full mixed batches reserve
30159.29 MiB but have at most 3219.73 MiB observed managed backing in this study.
These categories cannot be interchanged or interpreted as exact required RSS.
The batch plan bounds concurrent slots conservatively; it is not a new
shape-aware optimal packing scheduler.

The cost of the hard boundary is visible: reducing 4K batch4 from the full plan
to minimum fit lowers physical peak substantially, but increases CPU3 makespan
from 826.560 to 1284.278 ms. CPU1 goes from 1111.205 to 1627.719 ms. Matching-domain
cache reclamation and fewer work slots remain intentional, qualified tradeoffs.
For CPU3, median initial image queue is 479.093 ms in the tight 4K batch versus
4.268 ms full; tight two-caller Kodak queue is 89.658 ms versus 0.787 ms full.
The pressure cap is not promised to be free in throughput or latency.

Disposition: retain the conservative reservations and synchronous coordinator.
They provide explicit ownership, hard managed/CPU bounds, FIFO progress and
defined failure/drain behavior while preserving bytes. The measured regressions
above are accepted costs of this bounded structural checkpoint, not hidden
speedups or a reason to weaken admission. Tighter analytically proved bounds,
transactional idle-cache credits or different packing could be separately
measured future optimizations; they are not unbounded unfinished requirements.

## Requirement audit

The final full suite covers the current runtime; earlier milestone records
retain their own isolated comparisons rather than lending their percentages to
this comparison.

| Requirement | Implementation and evidence |
| --- | --- |
| Integrated preparation/output ownership (M3) | Independent completed-frame owner and direct final destinations survive evaluator/backend destruction and changed-image reuse. `metal_completed_frame`, AQ/quantization, preparation lifetime and cache tests run in the final suite; [joint integration](resident-execution-integration.md) records the original alias/failure reconciliation. |
| Complete managed admission (M4) | Shared C/C++ domain, checked policy plans, whole-call reservation, cache accounting and retained batch result envelopes. Resource/storage-plan tests and `workflow_admission`/`workflow_admission_cpu` cover under-budget rejection, typed underplans, caller-output atomicity and domain reuse. [Public admission](resident-public-admission.md) states managed exclusions and cache-pressure tradeoffs. |
| Fairness and progress | `resource_budget` and `workflow_admission` gate a large FIFO waiter ahead of a smaller request that would otherwise fit. `cpu_budget` checks FIFO callers and prevents extra worker reservations from bypassing queued callers. Admission does not grow an active reservation; GPU/join waits yield CPU participation. Progress assumes active work finishes and external holders release required resources; shutdown is not cancellation. |
| Bounded reuse/fusion inventory (M5) | Shared scratch, retained forward coefficients, direct completed output and last-use release remain implemented. [Reuse dispositions](resident-reuse-dispositions.md) give explicit reasons for every retained/deferred member of the audited set. No open-ended obligation to fuse more or cache mutable image contents is implied. |
| Aggregate CPU coordination (M6) | Per-image limits draw from one shared domain. Caller, worker reservation and suspended-worker capacity remain protected through nested work and waits. `cpu_execution` and C/C++ `workflow_cpu_coordination` tests cover shared identity, nested work and one-slot progress. Final performance/pressure jobs also enforce the configured aggregate limit. |
| Ordering, failures and lifecycle (M6) | Ordered per-image results, atomic array publication, explicit drain/rejection and unchanged queued outputs are covered by publication, lifecycle, scheduling and real partial-launch tests. Final retained mixed/two-caller outputs add independent decoded parity. Object destruction still requires all external callers to have returned. |
| Final performance/pressure (M6) | Complete 189-pair / 42-process matrix above; exactness, timing arithmetic, caps, raw records and artifact identities independently rechecked by the controller's audit/verify phases. CPU/GPU overlap already existed in the integrated batch driver; this branch coordinates its resource use rather than claiming to introduce overlap. |

The architecture remains a small synchronous heterogeneous execution domain.
It does not promise an OS-thread-count bound, a process-RSS cap, cancellation,
asynchronous public APIs, new encoding decisions or GPU entropy coding.
Screening/pruning, selective refinement and predictive AQ remain the separate
quality-time policy track (#4B and #1/#2), not unfinished structural milestones.

## Retained evidence

For a fresh reconstruction, use the tracked
[qualification package](../tools/resident_qualification/README.md). It takes
explicit corpus and decoder paths, reconstructs matching source/library pairs,
and runs fresh correctness and sanitizer gates. Its `--historical` configuration
selects the revisions measured here. New results remain a separate experiment;
the original measurements and artifact identities below are unchanged.

The following artifacts are a local historical archive, not tracked prerequisites
for that package.

`build/resident-scheduling-qualification/` contains `driver.cpp`, resumable
`run.py`, frozen `build.json`, per-process raw logs/JSON, append-only
`execution-events.jsonl`, retained compressed and decoded outputs, and phase
manifests. Completed measured jobs are reused only with matching arguments,
driver identities and log hashes. The completed `validation.json` seals **1611**
artifact identities, including launch-failure/final-runtime sanitizer evidence. `audit` checks
matrix membership, sample semantics, exactness, declared limits and timing
identities; `verify` also checks the sealed file hashes and recomputes the report.
Both phases pass. Validation manifest SHA-256:
`451848146149f37824a6bc1934985ae5f833e9bc3d452abf0619f105936344cf`.

To recheck the original evidence where that local archive is still available:

```sh
python3 build/resident-scheduling-qualification/run.py verify
```
