# Last-use release in the resident workflow

This milestone-5 checkpoint shortens storage lifetimes without changing encoding
decisions. Its parent is `f33fe0b`, the whole-workflow planning checkpoint.
It does not complete public resource admission or aggregate CPU scheduling.

## Ownership boundaries

The resident AC search waits for its GPU submission, resolves requested
diagnostics, performs CPU placement, and exports an independently owned strategy
grid. Its candidates, matrices, cost arrays and packed/rate buffers have no
remaining consumer after that return. `PreparedAcStrategySearch::Reset()` now
releases this state explicitly; returned grids and borrowed inputs are unchanged.
It is synchronous, not safe concurrently with search, and is idempotent. The
empty owner can be prepared again.

The internal encoding pipeline accepts a last-use indication for AC storage.
It **uses existing prepared capacity for the final search**, then resets that
owner on success or failure. Passing a null prepared owner for the last retry
would instead allocate duplicate AC storage while the previous owner survived.
The existing reusable/diagnostic APIs retain their defaults. No arithmetic,
candidate-set, placement, tie, or quantization policy changes.

Failure-path review also found that recording the AC preparation wall stage
could fail after submission but before `Wait()`. That path now drains the
submission before returning the original diagnostic error, so reset/reuse cannot
race work that still consumes AC buffers or borrowed inputs. A failed profiling
allocation must not release an allocation ticket while its GPU consumer is live.

The public workflow provides that last-use indication for single-target calls
and the final permitted target-size attempt. Earlier attempts retain reusable
AC state. A tolerance success before the attempt limit is unknowable until
serialization, so preparation remains live through that earlier serializer call
and is destroyed when the search returns. Failed candidates still consume their
existing attempt count; terminal underplans do not escape into another attempt.

After a successful final-attempt **independent completed-frame** handoff, the
workflow destroys the prepared AQ evaluator, quantization metadata, resident
input and any remaining host preparation before CPU serialization. Borrowers
are destroyed before their input owners. Geometry and scalar matrix statistics
remain available for summary construction; serialization and strategy counts
consume only the independent output view. An optional in-place AQ owner permits
early destruction without another heap allocation. Reusing this released
workflow is explicitly rejected.

Owned-frame fallback and diagnostic consumers retain their existing contracts;
the workflow does not apply the completed-output release to them. The internal
prepared encoding API can still run again after AC release, reacquiring that
owner. Cleanup is charged to the quantization phase, and complete-call timing
includes teardown on both sides of the comparison.

## Accounting and phase bounds

AC backing has no idle pool: releasing its owner actually ends its allocation
charges. AQ, input and Butteraugli backing may instead become idle cached
capacity. It remains charged until eviction, trim or backend destruction. Moving
backing from live to idle is not a managed-memory or physical-footprint saving.

The whole-workflow plan therefore retains common AQ/input/metric charges and
uses the maximum of two conservative envelopes:

- AC-search phase: common backing, AC host/device work, diagnostics, search
  control and an earlier retained best result when applicable.
- Completion/serialization phase: common backing, independent completed output,
  the complete serializer envelope and the same retained/control diagnostics.
  Multi-attempt searches also include AC here because earlier attempts retain it.

The `device_bytes` and `frontend` fields expose owner inventories, not simultaneous
peaks. `search_phase`, `completion_phase` and `working` expose the phase bounds.
No capacity is grown or waited on during allocation. The job still needs one
upfront reservation through outer publication; public admission is separate.

## Qualification

Evidence is retained in `build/last-use-qualification/`, using frozen parent
`build/resident-workflow-plans` and fresh candidate `build/resident-last-use`.
The sanitizer build is `build/resident-last-use-asan`. `run.py` and
`summarize.py` record commands, raw samples, hashes and summaries;
`validation.json` seals source, build, decoder and evidence artifacts. Earlier
builds/evidence are preserved. Initial successful parity evidence is retained
under `pre-drain/`; final-source parity is rerun after the failure-path fix.

The permanent workflow test checks 2,240 allocation-free plan shapes, 64 attempt
limits and actual interval-growth cases, plus 99 whole-workflow shapes with an
unprofiled oracle and two plan-bounded cold/warm encodes each. It checks tiny,
padded and 4K geometry, all iteration counts, score/profile requests, throughput,
high density, compression modes and size-search selection. A calling-thread-only
test observer samples the ledger immediately before serialization, verifying
completed-output independence, released final-attempt owners and retained retry
preparation. Extra searches cover one permitted attempt and early tolerance
success. Direct tests cover idempotent AC reset, surviving grids and explicit
prepared-pipeline reuse after last-use release. The hook is not installed as a
public header and does not propagate across workers.

The existing diagnostic failure enumeration covers 406 workflow allocation
positions and 13 primitive positions, with atomic output and recovery checks.
Underplan/physical failure distinctions and cache trim/owner destruction remain
covered by the whole-workflow and resource tests. Stage-boundary profiling is
available; dispatch-boundary sampling is unavailable and explicitly skipped.

All 56 final-source parent/candidate codestream pairs match exact hashes. Kodak17,
planter 4K and padded-stress 4K separately decode to matching PFM hashes with the
pinned `djxl`. Both builds pass all 22 pinned conformance fixtures. Sixty-one
common runtime objects and the metallib are byte-identical; only workflow, AC
search, encoding-pipeline plumbing and the workflow-plan object change.

Both full Release suites finish **88/89**, with only the inherited CPU
`quantization_pipeline` golden mismatch: actual `0.24919039011001587`, expected
`0.24914586544036865`. The final-source Release suite is rerun after the drain
fix. AC search, quantization pipeline, completed-frame, resource-budget,
diagnostic-storage and whole-workflow tests each pass three repetitions in
Release and ASan/UBSan: eighteen focused runs per configuration. The initial
sanitizer filter used the wrong AC CTest name; a separate three-repeat run of
`metal_ac_strategy_search` supplies that coverage. The other five sanitizer
tests already ran against the drain fix. Strict warnings pass for the planner
and expanded whole-workflow test.

The unsuppressed sanitizer run reproduces metal-cpp's Objective-C nil-reference
idiom at `Foundation/NSObject.hpp:112:49`. Successful ASan/UBSan runs use
`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:suppressions=.../ubsan.supp`, with only
`null:*/third_party/metal-cpp/*` suppressed. No GJXL source is suppressed.

### Bounds and observed backing

The fresh-domain ledger probe retains the prior synthetic recipe, fully resident
effort 7, target 1.0, four CPU participants and no diagnostics:

| Source | Parent planned bytes | Last-use planned bytes | Parent peak backing | Last-use peak backing |
| --- | ---: | ---: | ---: | ---: |
| 89x57 | 27,482,700 | 26,898,804 | 15,367,607 | 14,772,847 |
| 3839x2159 | 7,952,975,062 | 7,307,617,086 | 2,933,527,656 | 2,602,024,476 |

The 4K reservation bound falls 8.1% and observed live-plus-idle peak backing falls
11.3%. These are managed-capacity figures, not physical-memory measurements or
predicted usage. The reservation still conservatively overestimates the observed
content-dependent encode.

### Complete-call measurements

Platform: Apple M4 Pro, 48 GiB, 14 logical CPUs, Mac16,7; macOS 15.6 (24G84),
Apple Clang 17 and SDK 26.2. Release builds enable tests/benchmarks and disable
libjxl-reference fixtures and compile-time Metal profiling. Measurements use
SIMD/fused-tuned Metal, fully resident, distance 1.2, effort 7 and automatic CPU
threads. The complete synchronous-call timer includes teardown, but excludes
backend creation, input loading, hashing and writing. It uses the CPU workflow
profile wrapper, without GPU timestamp sampling.

Each workload has seven alternating parent/candidate independent-process pairs.
A process encodes original/changed images fifteen times, discards the first
three, and retains six observations per image. Tests/builds finish before timing;
process snapshots check for competing GJXL work, not exclusive machine ownership
or fixed thermal/power conditions. All 840 outputs have unchanged sizes/hashes.

Times below are medians of process medians; percentage changes are medians of
paired ratios, not ratios of the displayed times.

| Original image | Parent ms | Last-use ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 23.798 | 23.860 | +1.39% | 2/7 |
| Padded 1080p, 1919x1079 | 73.106 | 73.194 | -0.19% | 4/7 |
| Padded 4K, 3839x2159 | 249.013 | 247.359 | +0.002% | 3/7 |
| Planter 4K, 3840x2160 | 263.707 | 265.639 | +0.16% | 3/7 |

Changed-image companions are +1.45%, +0.11%, -0.55% and -1.00%, respectively.
Original-image paired ranges are -1.50% to +4.11%, -1.58% to +1.56%, -2.62% to
+1.60%, and -3.35% to +3.79%. Both builds drift upward during the planter cohort;
alternating pairs mitigate but do not eliminate machine variability. This is
not a speedup or zero-cost claim. In particular, the small-image paired
regression is retained in the record rather than hidden by the larger inputs.

### Physical footprint, idle capacity and trim

Independent padded-4K memory and trim cohorts each use three alternating
process pairs and retain both source images. Footprint is macOS
`TASK_VM_INFO.phys_footprint`; peak is `ledger_phys_footprint_peak`. These include
excluded caller images, driver/runtime state and allocator retention, unlike
the managed-backing ledger. Values below are medians in MiB:

| Boundary/cohort | Parent | Last-use |
| --- | ---: | ---: |
| Peak, no-trim cohort | 2974.251 | 2821.298 |
| One-second backend-alive idle, no-trim cohort | 2016.361 | 2005.814 |
| Peak, trim cohort before trim | 2972.251 | 2820.814 |
| One-second idle before trim, trim cohort | 2021.267 | 2017.861 |
| One-second idle after trim | 264.064 | 262.407 |

No-trim peak ranges are 2969.814–2975.580 and 2814.595–2822.486 MiB. The
independent cohorts agree on approximately **5.1% lower peak physical footprint**.
Idle readings vary substantially: the no-trim candidate ranges from 1290.189 to
2016.954 MiB, while the parent ranges from 2007.501 to 2206.830 MiB. This does
not establish a stable idle-footprint improvement or a changed cache policy.

Both builds retain exactly 1,946,881,084 idle managed bytes before trim and zero
after trim. In that cohort, peak managed backing is 2,922,164,054 parent versus
2,602,024,476 last-use bytes; its synthetic recipe/target differ from the earlier
bound probe, so those parent peaks must not be conflated. Encode-after-trim
recovers with unchanged output, and backend destruction releases retained
capacity. The physical-memory reduction is smaller than the managed-capacity
reduction; neither counter is relabeled as the other.

Disposition: **retain** the last-use release for its qualified peak-memory
benefit and explicit ownership boundaries, with the measured small-image latency
tradeoff. It is not an AQ/AC algorithm optimization or completed domain admission.

## Remaining work

Milestone 5 still needs dispositions for the remaining exact shared-calculation
and fusion inventory. Milestone 4 still requires other backend/policy/input/batch
plans and shared public-domain admission; milestone 6 requires aggregate CPU
coordination and concurrency/fairness qualification. These last-use boundaries
are useful independently but do not satisfy those larger contracts.
