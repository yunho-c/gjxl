# Diagnostic and retained-result accounting

This milestone-4 attachment follows `1c6b3ce`. It closes the score-history,
summary-history, timing-attempt and GPU-profile backing gaps identified in
[publication accounting](resident-publication-accounting.md). It does not finish
whole-workflow admission, last-use optimization, or coordinated scheduling.

## Ownership and publication

| Storage | Internal owner and accounting boundary |
| --- | --- |
| CPU and resident Metal AQ score history | A fixed-capacity `PublicationVector<double>`, charged as AQ scratch. Moves into workflow artifacts and the selected summary keep the same backing and charge. |
| Selected summary | `OwnedEncodingSummary` wraps the unchanged public summary record and owns its history ticket. Candidate replacement and target-size tie-breaking preserve the history until the candidate is discarded or published. |
| Attempt timings | A bounded publication vector reserves one record for a single target, or the validated maximum attempt count for target-size search. Appends never allocate. `OwnedEncodingTiming` keeps the backing charged through the outer result handoff. |
| Batch results | The result table, internal result owners, and three-ticket-per-image publication escrow are managed. Successful bytes, summaries and timings become retained-result storage until the entire ordered public result array is published. Failed images keep no candidate backing. |
| GPU profile graph | Managed diagnostic vectors for wall stages, submissions, stages and dispatches; managed strings for their identifiers, including non-inline labels and empty containers retaining capacity. Submission snapshots and child profiles retain their charges. |
| Private CPU AQ profile | Its evaluation array is a managed diagnostic vector. This internal profile retains backing through moves until destruction; it is not the public workflow timing record. |

The public workflow summary/timing records and C byte-buffer ABI are unchanged.
`PublicationOutput<T>` lets an AQ producer commit either to an existing public
`std::vector<T>*` or an internal publication owner. Existing pointer-initialized
call sites remain valid; the low-level C++ output member's type/layout changes,
so consumers must rebuild. Private profile vector/string types also change.
Frozen comparison drivers must use their matching headers and libraries.

`PublicationRecord` keeps a public record's vector backing and resource ticket
together. Its field replacement, move assignment and destruction release backing
before the old charge. A multi-output adapter escrows all tickets until every
output is assigned, with no fallible work after publication starts. This removes
the old summary-history copy; it does not change scores or candidate selection.

Bounded append uses the previously reviewed fresh-vector constructor contract
in macOS libc++ C++20. It constructs the exact requested capacity, clears logical
elements, and rejects append beyond that capacity with a terminal resource-plan
error. Empty logical vectors can still own capacity: transfer/reclassification
must inspect capacity, not emptiness. CPU AQ bounds remain the existing iteration
count plus one, or six evaluations for maximum-error control. No policy changes.

GPU diagnostics use allocator-owned tickets because all nested profile types
are internal. At the complete workflow's successful public/testing handoff,
recursive helpers release those tickets without copying or freeing backing.
Allocator headers remain valid for later external destruction. Only allocated
string character storage has a ticket; inline characters are already part of
their containing record. The release helper relies on the reviewed libc++ C++20
contract that short-string data points into the string object and long-string
data is the original allocator pointer. A compile-time platform/library guard
requires that contract to be reviewed before porting. Tests cover both sides of
the inline-storage boundary, long and cleared labels, moves, copies and growth.

Publication excludes caller-owned results only after the handoff. Internal child
profiles, completed-submission snapshots and selected retry summaries must not
use the release helpers early. Their producers may be gone while the backing
remains charged to the original domain.

## Failure handling

Managed failures retain their precise status through the profile session,
Metal stage construction, void dispatch recorder, completed-submission snapshot,
and primitive-profile adapter. A resource-plan overrun remains terminal through
target-size search; ordinary candidate failures keep their existing policy.

The profiled compute path now scopes both active-dispatch thread-local state
and `endEncoding()`. Metadata failure or an exception must unwind these before
returning, without committing the command buffer or leaving a dangling pointer
to stack-owned dispatch state. Every failed attempt can reuse the same backend.

The host allocation test hook can filter a resource class. It still fires only
after authorization and immediately before physical allocation; unrelated-class
allocations do not consume the counter. Existing all-class injection is retained.

The managed limit is still not a physical-footprint bound. Metal counter sample
buffers and resolved `NS::Data` are opaque driver allocations, within the existing
driver/command-internal exclusion. Their host profile arrays and labels are not
excluded. Small control objects, status messages, immutable pipeline setup,
allocator headers and stacks retain the separately documented exclusions.

## Correctness and lifetime checks

Fresh Release candidate: `build/resident-diagnostics`. Frozen parent:
`build/resident-publication` at `1c6b3ce`. Both enable tests/benchmarks and disable
libjxl reference fixtures and compile-time Metal profiling. Runtime stage
profiling is exercised separately by the permanent diagnostic test.

The full suites pass 70/71 on the parent and 72/73 on the candidate. Both reproduce
the CPU `quantization_pipeline` golden mismatch: actual `0.24919039011001587`,
expected `0.24914586544036865`. No golden or tolerance was changed.

All 56 corpus/policy codestream SHA-256 comparisons match, including efforts
1–10, exact coefficients, resident/throughput modes, final scores, target-size
search and maximum error. Both pass all 22 pinned conformance fixtures.
Independent pinned `djxl` decoding of Kodak17, planter 4K and padded stress 4K
produces matching linear-RGB PFM hashes. The decoder reports `e8ff0976`, matching
the pinned `e8ff09762481785938d8e4e01333ed3917571161` build.

Permanent tests cover:

- Fixed-capacity history accumulation, under-plan rejection before allocation,
  independently recomputed backing counts, score-based candidate ties against
  the original standard-vector search, and cross-thread retained ownership.
- Record replacement/destruction while backing remains charged, closed producer
  reservations, zero-copy output handoff and multi-field publication escrow.
- Public score/timing failure atomicity, C publication failure enumeration,
  all 818 C host-backing failure positions, one/three-worker batch publication,
  mixed success/failure, and no default-domain escape from an explicit reservation.
- Nested diagnostic graph copying, inline/long/cleared strings, retained empty
  vector capacity, cross-thread publication/destruction, idempotent release and
  all 29 allocation-failure positions in the CPU diagnostic-graph fixture.
- All 13 allocation positions in a profiled Metal affine submission and snapshot,
  and all 406 diagnostic positions in a 17x9 effort-1 resident workflow. Every
  injected failure is followed by successful reuse of the same backend.
- A real reservation overrun inside the void dispatch recorder after the stage
  array fits exactly. It preserves the terminal reason, submits no work and
  leaves device output untouched, followed by unprofiled and profiled reuse.

The existing all-host resident workflow sweep now covers 726 positions, including
the score-history allocation, with same-backend recovery after every failure.

This M4 Pro supports stage-boundary timestamp sampling but not dispatch-boundary
sampling. The native diagnostic test reports that latter mode as skipped. This
does not qualify per-dispatch timestamp execution on a supporting device.

Five CPU suites pass ASan/UBSan without suppressions: resource budget, managed
allocator, publication storage, workflow publication and diagnostic storage.
Three additional suites (batch workflow, Metal resource accounting and Metal
diagnostic storage) pass with only the existing `null:*/third_party/metal-cpp/*`
UBSan suppression for Objective-C nil retain/release wrappers. Both sanitizers
halt on error; leak detection is disabled. No GJXL source checks are suppressed.
Six CPU/concurrency suites, including batch workflow and nested diagnostics,
each pass ten ThreadSanitizer repetitions without suppressions (199.05 seconds).

## Complete-call and footprint measurements

Apple M4 Pro, 48 GiB, macOS 15.6, Apple Clang 17, arm64; Release fully resident
SIMD/fused-tuned Metal, effort 7, distance 1.2, automatic CPU threads. The timer
surrounds the synchronous C++ encode, including evaluator teardown and outer
publication. Backend creation, input loading, hashing and file writes are outside
the timer. Internal returned timings are not substituted for this boundary.
Runtime GPU counter profiling is not enabled in these latency measurements.

Seven alternating independent parent/candidate process pairs per workload each
alternate two changed images, run 15 encodes, discard the first three, and retain
six samples per image. All 840 encoded sizes/hashes match across variants and
remain stable. Times are medians of process medians; percentage changes are
medians of paired ratios, not ratios of the displayed medians. Negative means
lower latency.

| Primary input | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 23.157 | 23.429 | +0.30% | 2/7 |
| Padded 1080p, 1919x1079 | 72.322 | 71.561 | -0.50% | 5/7 |
| Padded 4K, 3839x2159 | 248.752 | 246.989 | -0.11% | 4/7 |
| Planter 4K, 3840x2160 | 258.978 | 259.076 | -0.80% | 4/7 |

Changed-image companion paired changes are -0.44%, -0.57%, +0.16% and -0.04%,
respectively. These small mixed results establish neither a consistent speedup
nor zero overhead, and are not a qualification of all policies or image sizes.

Three alternating padded-4K physical-footprint process pairs retain both inputs:

| Boundary | Parent MiB | Candidate MiB |
| --- | ---: | ---: |
| Peak after encoding | 2972.204 | 2972.439 |
| One second idle, backend alive | 2014.798 | 2206.314 |
| One second after backend destruction | 248.923 | 250.876 |

The higher candidate one-second idle observation is reported, not treated as
proof of equal physical footprint. Managed idle capacity is unchanged in the
separate trim probe; physical reclamation of volatile/driver storage has a
different boundary from allocation-owned capacity. A separate confirmation
cohort checks this observation without replacing the initial measurements.
Its three alternating pairs preserve all 18 output hashes and report
2017.908/2019.798 MiB one-second idle medians, with wide parent/candidate ranges
of 764.548–2018.830 / 1497.236–2207.236 MiB. Peak medians are
2971.892/2971.392 MiB; one second after backend destruction is
252.111/251.767 MiB. The initial idle difference is not stable across these small
cohorts. This is evidence of a variable observation boundary, not proof that
physical overhead is identically zero or that every variation is OS-caused.

Three further alternating trim-probe pairs preserve all 24 output hashes. Both
variants report 1856.690 MiB managed idle before trim and zero afterward; live
capacity and unbacked reservation bytes are zero at those boundaries. Peak
managed backing is 2786.792791/2786.792807 MiB: exactly 16 newly accounted score
bytes at this workload's peak, not new image-sized backing. One second after
trim, physical footprint is 260.298/259.642 MiB. Immediate post-trim footprint
is much higher (1950.126/2110.798 MiB); released capacity is not an immediate
physical-counter guarantee. Trim time is 6.639/3.574 ms and the first encode after
trim is 331.659/337.292 ms. Three pairs do not establish a causal trim/resume gain.

The existing batch benchmark supplies a separate three-pair regression probe,
with five samples after three warmups, two changed inputs, batch sizes 2/4 and
the same Metal policy. It verifies exact batch-versus-sequential bytes and
summaries within each variant. Cross-variant checks in this probe compare byte
counts only, not hashes; the corpus matrix supplies separate cross-variant byte
evidence. This is not milestone-6 scheduling qualification.

| Workload / batch size | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| 256x192 / 2 | 15.322 | 14.979 | -1.85% | 3/3 |
| 256x192 / 4 | 24.607 | 24.862 | +1.04% | 1/3 |
| 1080p / 2 | 116.959 | 116.143 | -0.90% | 2/3 |
| 1080p / 4 | 208.344 | 209.399 | -0.12% | 2/3 |

These mixed small-cohort results are recorded costs, not a batch-throughput
improvement claim. Timed work began after builds/tests finished. Process checks
reject another observed GJXL encoder/test, but do not establish exclusive
machine ownership or fixed power/thermal conditions.

## Qualification artifacts

The current run's local evidence is in `build/diagnostic-resource-qualification`.
It reuses the established qualification harness with frozen parent drivers and
fresh candidate drivers, each matched to its own headers/libraries. The manifest
includes all sixteen parent/candidate libraries, four drivers and both driver
sources. It is not valid to rebuild the frozen parent from this checkout.

Sanitizers use separate RelWithDebInfo builds `build/diagnostic-sanitize` and
`build/diagnostic-tsan`, frame pointers and line-table debug information. Logs are
`sanitize-cpu.log`, `sanitize-metal.log`, and `tsan.log` in the qualification
directory. `final-ownership.log` contains the allocation-site counts and the
native timestamp-capability skip. The full candidate suite log remains
`build/resident-diagnostics/final-ctest.log`; the frozen-parent rerun is
`parent-ctest.log` in the qualification directory.

`summary-verified.log` recomputes paired results and verifies libraries, drivers,
sources, encoder binaries, all retained corpus/input/decoded files and the timed
output hashes. `batch.json` records its independent executable hashes and source.
`memory-confirmation-verified.log` verifies the separate confirmation drivers,
18 output hashes and physical measurements. `validation.sha256` records the
relevant Release/sanitizer test binaries and logs, verified before and after
commit. The three CPU publication/diagnostic test sources also pass AppleClang
17 `-Wall -Wextra -Wpedantic -Werror` syntax checks.

After freezing the parent and building the candidate:

```sh
python3 build/diagnostic-resource-qualification/run.py build
python3 build/diagnostic-resource-qualification/run.py parity
python3 build/diagnostic-resource-qualification/run.py conformance
python3 build/diagnostic-resource-qualification/run.py performance
python3 build/diagnostic-resource-qualification/run.py memory
python3 build/diagnostic-resource-qualification/run.py trim
python3 build/diagnostic-resource-qualification/summarize.py
python3 build/diagnostic-resource-qualification/batch.py
python3 build/diagnostic-resource-qualification/run.py memory-confirmation
python3 build/diagnostic-resource-qualification/verify_confirmation.py
```

## Remaining scope

The next milestone-4 work is shared conservative allocation planning, public
execution-domain configuration/propagation, upfront whole-work and aggregate
retained-batch-result reservations, automatic domain-wide idle-cache eviction,
and end-to-end rejection/waiting/progress tests. Instrumented allocation coverage
is necessary but does not itself prove a complete working-set estimate or enforce
a user-visible aggregate limit.

Milestone 5 still requires the audited last-use/reuse dispositions and measured
gates. Milestone 6 still requires aggregate CPU/GPU scheduling and actual
latency/throughput qualification. The diagnostic attachment does not satisfy
those requirements or change the exclusions in the architecture charter.
