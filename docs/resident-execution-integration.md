# Preparation and completed-output integration

This is the qualification record for milestone 3 of
[resident execution](resident-execution.md). The comparison parent is `a747fca`
(the `dabe129` handoff implementation plus its roadmap); the incoming branch is
`perf/metal-preparation` at `306f153`, including `e1010fe`. Fusion was already in
both parents. This is not a new AQ, AC-search, entropy, or scheduling policy.

## Integration review

The production files auto-merged. The one textual conflict was the AQ test
driver's list of checks; resolution retains the handoff's five completed-output
failure cases as well as the preparation branch's deferred-frontend and filter
combination coverage.

The reviewed lifetime boundaries are:

- The resident frontend can defer final transform metadata until successful
  `Reconfigure`. Completed-output destination generation uses that authoritative
  strategy grid; ordinary evaluation validation rejects pending metadata before
  allocating output or submitting work.
- Butteraugli borrows only transient filter/gather planes. Its cached arena owns
  capacity, not image-specific prepared state or borrowed AQ views.
- Completed AC output and snapshotted block metadata remain independently owned.
  No output aliases the borrowed planes, cache, evaluator, or backend lifetime.
- AQ destruction waits for outstanding work and destroys its Butteraugli borrower
  before returning backing AQ arenas. Failure invalidation and generation-aware
  trimming prevent unsafe cache reuse.

The original preparation records remain authoritative for those individual
changes: [shared storage](metal-preparation-integration.md) and
[volatile caching](metal-volatile-preparation-cache.md). Their timings must not
be added to the handoff's timings as evidence for this combination.

## Correctness

Both configurations are fresh Release builds with tests and benchmarks enabled,
libjxl-reference fixtures and compile-time Metal profiling disabled. The frozen
parent is `build/resident-baseline`; the integrated build is
`build/resident-integrated`. The standalone parent's complete-call driver was
also compiled before merging, against its own headers and libraries.

All 56 corpus/policy cases match codestream SHA-256: 38 canonical natural/padded
images plus efforts 1–10, density/compression controls, final-score diagnostics,
exact coefficients, both throughput modes, target-size retries, and maximum
error. Both builds pass all 22 pinned conformance fixtures. Separate pinned
`djxl` decoding of Kodak17, planter 4K and padded stress 4K produces identical
linear-RGB PFM hashes. The decoder revision is
`e8ff09762481785938d8e4e01333ed3917571161`.

The completed-frame test now combines deferred preparation, mixed final
strategies, changed images, same-size cache reclamation, shrinking/growing
layouts, concurrent serialization and trimming, and serialization after backend
destruction. Reacquisition after forced `Empty` uses the same extent first so
the test exercises reclamation recovery, not just oversized-capacity shedding.
Existing tests retain tiny/padded images, multiple groups, score/no-score output,
profile parity, atomic failures, all filter combinations, and independent callers.

The initial integrated CTest exposed an invalid timing assertion: the measured
host output-commit span was zero nanoseconds, while coefficients, allocations,
submissions, and the other profile fields were correct. A clock can report zero
for work finishing within a tick. The test now poisons that field and checks
successful publication instead of requiring a positive duration. Detailed
diagnostics are retained; numerical tolerances and output comparisons are
unchanged. Twenty repetitions of the frozen parent's test did not reproduce the
timing failure, so this is not reported as a reproduced parent failure.

The final full Release suites pass 63/64 on both the frozen parent and integrated
candidate. Both reproduce the same CPU `quantization_pipeline` golden mismatch:
actual `0.24919039011001587`, expected `0.24914586544036865`. No numerical golden
or tolerance was changed. The corrected AQ and completed-frame tests passed ten
repetitions each; the final completed-frame test, including same-size forced
reclamation, was then rebuilt and included in the final full suite.

The final AQ-evaluation, GPU-quantization-pipeline and completed-frame tests pass
3/3 with AddressSanitizer and UBSan. Leak detection is disabled. UBSan null checks
are suppressed only in `third_party/metal-cpp` headers, whose Objective-C nil
retain/release wrappers fail the unsuppressed run before encoding. That failure
and suppression file are retained; no GJXL source checks are suppressed.

## Measurement protocol

Apple M4 Pro, 48 GiB, macOS 15.6; SIMD/fused-tuned Metal, fully resident, effort 7,
distance 1.2, automatic CPU threads. The timer surrounds the complete synchronous
encode including evaluator teardown. Backend creation, input loading, hashing,
and output writes are excluded. Cold-backend scenarios recreate the backend
outside that timer; they are not backend-initialization latency measurements.

The warm matrix uses seven alternating independent parent/candidate process
pairs for padded 1080p, padded 4K, Kodak17, and planter 4K. Each process alternates
the named image with a deterministically changed-image companion, runs 15 encodes,
discards the first three, and retains six observations per image. Report process
medians and paired ratios, not a ratio of unrelated minima or aggregate worker
time. All 840 outputs are checked for stable per-image size and hash.

Separate three-pair probes cover peak/one-second idle footprint, fresh backends,
alternating 1080p/4K geometry, and the first encode after five seconds idle. These
are smaller diagnostic cohorts, not additional warm-stream samples. Process
checks reject a run when another GJXL encoder/test is observed; they do not
establish exclusive machine ownership or fixed power/thermal conditions.

## Complete-call results

Times below are medians of seven process medians for each named input. Percentage
changes are medians of seven paired ratios, not ratios of the displayed medians.
Negative changes mean lower latency. These are combined preparation/handoff
results against the handoff parent, not sums of earlier measurements.

| Input | Parent ms | Integrated ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 25.161 | 23.554 | -7.05% | 7/7 |
| Padded 1080p, 1919x1079 | 86.235 | 72.458 | -16.07% | 7/7 |
| Padded 4K, 3839x2159 | 290.387 | 242.951 | -16.12% | 7/7 |
| Planter 4K, 3840x2160 | 292.150 | 246.813 | -15.20% | 7/7 |

The changed-image companions also improve in every pair: median paired changes
are -8.11%, -16.10%, -16.65% and -15.79%, respectively. All 840 encodes preserve
each image's output hash. These four workloads establish this checkpoint, not a
universal speedup across all image sizes or policies.

The smaller three-pair scenarios preserve all 162 output hashes. Fresh-backend
4K encodes improve by 7.40%, alternating-size 4K encodes by 3.34%, and the first
4K encode after five seconds idle by 5.58%; each improves in all three pairs.
The 1080p half of the alternating-size scenario has a 0.93% median paired
regression and improves in only one of three pairs. Fresh-backend timing still
excludes backend construction. Cache reuse depends on geometry and idle state;
warm-stream gains must not be substituted for these scenarios.

## Physical footprint and retention tradeoff

Three alternating padded-4K process pairs retain two source images throughout
the probe. Values are median process physical-footprint counters in MiB, not
requested allocation capacities, free system memory, or a hard memory limit.

| Boundary | Parent MiB | Integrated MiB |
| --- | ---: | ---: |
| Process peak after encoding | 3342.314 | 2970.095 |
| One second idle, backend alive | 1195.408 | 2023.314 |
| One second after backend destruction | 260.517 | 258.454 |

Shared scratch reduces the measured peak by 372.219 MiB, while the additional
volatile cache raises the one-second backend-alive footprint by 827.906 MiB.
The separate idle/resume probe reports 268.627/268.501 MiB after five seconds
idle. Volatile storage is reclaimable, but it is not physically free immediately.
Backend destruction also need not produce an immediate physical-counter drop.

Permanent tests verify that explicit trim drops idle Butteraugli capacity to
zero and preserves active work and completed frames. This checkpoint does not
measure physical footprint after explicit trim; the resource milestone must add
that boundary alongside coordinated accounting. Existing AQ pools are independent
of the Butteraugli cache and are not trimmed by this API.

## Artifacts and reproduction

Local evidence is in `build/resident-integration-qualification/`: `run.py`,
`summarize.py`, build/library/driver hashes, all parity outputs and commands,
decoder logs, both conformance logs and their command/binary manifest, CTest/sanitizer logs, process
snapshots, timing samples, and memory observations. The runner reuses the retained
`build/handoff-qualification/{qualify.py,full_workflow.py,driver.cpp}` harness.
These are local qualification artifacts, not installed tools or durable test
fixtures; regression coverage is in the repository's permanent tests.
The conformance executable deletes its successful per-fixture scratch files;
the separate corpus JXL/PFM outputs and decode logs remain retained.

Before committing, all fourteen parent/candidate library hashes, both qualified
driver hashes, and the driver-source hash were checked against `driver-build.json`
and matched. Only tests and documentation changed after the timed libraries were
built. Final suite logs are `final-integrated-ctest.log` and `final-sanitize.log`;
the initial failure, baseline, and repeated-test logs remain separate.

After freezing the parent and building the candidate as described above:

```sh
python3 build/resident-integration-qualification/run.py build
python3 build/resident-integration-qualification/run.py parity
python3 build/resident-integration-qualification/run.py conformance
python3 build/resident-integration-qualification/run.py performance
python3 build/resident-integration-qualification/run.py memory
python3 build/resident-integration-qualification/run.py scenarios
python3 build/resident-integration-qualification/summarize.py
```

Do not rebuild the frozen parent against the now-integrated source. The next
resource/scheduling milestones need this combined implementation as their own
fresh, identified baseline; milestone 3 introduces no whole-encoder budget.
