# Candidate codestream and publication backing

This checkpoint extends the frontend attachment at `9a5d93d`. It is part of
milestone 4 of [resident execution](resident-execution.md), not completion of
whole-workflow accounting, admission, lifetime reduction, or scheduling.

## Ownership boundary

The serializer's candidate bytes now remain managed through representation
selection, workflow return, target-size candidate selection, retained batch
results, and the final C/C++ ownership handoff. Before this attachment, a
serializer could account its writers and models but release its result into an
ordinary untracked vector while that result was still owned by the encoder.
That boundary was too early for a whole-workflow resource limit.

`PublicationVector<T>` owns an ordinary `std::vector<T>` together with a separate
allocation ticket. It authorizes fresh backing before construction, commits the
ticket after successful construction, and frees backing before releasing the
ticket. Moves preserve both pointer and charge. Replacing a result counts old
and new backing simultaneously. A retained allocation survives its producer's
scope/reservation and can be destroyed on another thread.

This owner deliberately does not offer implicit vector growth. The reviewed
macOS libc++ C++20 implementation's fresh count/forward-range constructors use
`__init_with_size -> __vallocate -> __allocate_at_least`; the C++20 branch of
`__allocate_at_least` calls `allocator.allocate(n)` and returns capacity `n`.
That pre-allocation contract, not a post-allocation observation, establishes the
capacity bound. The implementation rejects other standard libraries/language
modes at compile time until their contract is reviewed. A runtime capacity
check also detects a changed implementation. Supporting another configuration
must not silently replace this with allocate-first accounting.

The final writer-to-vector copy already existed. This attachment preserves it
and the zero-copy vector move into the public C++ result. It does not introduce
a managed-allocator-to-default-allocator publication copy.

### Retry and batch retention

The existing target-size algorithm is shared between the ordinary internal
test interface and the managed-byte workflow. Best-so-far and new candidate
backing coexist and stay charged. Candidate ordering, score tie-breaks, search
intervals, stopping rules, and ordinary candidate-local failures are unchanged.
Interval backing now uses the managed serializer allocator.

A resource-plan overrun is different from an ordinary candidate failure.
`Status::ResourcePlanExceeded` retains `kOutOfMemory` and the existing message,
but carries a structured internal reason through status/exception adapters.
Target-size search aborts on that reason and leaves its output unchanged. It
must not keep searching for a candidate that happens to fit an underestimated
plan. Cache-transfer failures are not marked as plan overruns: dropping an
oversized idle allocation and trying the planned fresh capacity remains valid.

Batch workers inherit the caller's resource context. Each successful image
reclassifies its byte allocation as retained result without changing the
domain, reservation, physical backing, or total charge. Structural result and
ownership arrays are accounted too. Once all workers finish, a no-fail
publication phase moves nested byte vectors while retaining their tickets in
escrow. Those tickets are released only after the outer result array reaches
the caller. The escrow is declared before the unpublished array so rollback
destroys backing first. Individual encode failures and request ordering retain
their existing semantics. This is not aggregate batch-result admission.

### C API

Packed-input conversion is now inside the managed input scope. The C adapter
keeps the internal codestream charged while authorizing and allocating its
existing `new[]` output array. Both backings remain covered through the copy;
the C array leaves accounting only when its pointer/size are published. The
existing `gjxl_buffer_free`/`delete[]` ownership contract and public C ABI are
unchanged. Status translation preserves the precise managed failure message.

## Permanent checks

- `publication_storage`: exact capacity/class accounting; preauthorization;
  replacement overlap; physical-allocation failure; empty/null/overflow cases;
  zero-copy publication; same-domain transfer and cross-domain rejection;
  closed producers; cross-thread destruction; move-assignment destruction order;
  nested publication escrow; and managed target-size retention/terminal overrun.
- `resource_budget`: pending/live/idle owner transitions, unchanged totals,
  invalid/no-op transitions, closed producers, cross-thread transitions, and
  reclassification added to the independent randomized state model.
- `workflow_publication`: owned/public CPU workflow byte and summary equality;
  retained byte capacity after workflow return; exact all-workers-complete batch
  capacities before publication with one/three workers and mixed success/failure;
  batch underplan propagation, atomic setup failure and successful recovery;
  C conversion preauthorization and all 817 host-backing failure positions,
  including the final C allocation; equivalent C/C++ codestream bytes.
  Explicit reservations never allocate in the
  unlimited default domain in this test.
- Existing resident Metal failure coverage now reaches 725 host-backing
  positions, including final candidate bytes. The same backend remains usable
  after every injected failure, with all charges released after trim/teardown.

The explicit capacities in these tests are deliberately ample test envelopes,
not workload estimators or supported application memory-limit settings.

## Qualification

Fresh Release parent/candidate suites pass 68/69 and
70/71 respectively, retaining only the reproduced CPU `quantization_pipeline`
golden mismatch: actual `0.24919039011001587`, expected
`0.24914586544036865`. No golden or tolerance changed.

All 56 corpus/policy codestream SHA-256 comparisons match: canonical natural and
padded images, efforts 1–10, density/compression controls, diagnostics, exact
coefficients, both throughput modes, target-size retries and maximum error.
Both builds pass all 22 pinned conformance fixtures. Separate pinned `djxl`
decoding of Kodak17, planter 4K and padded stress 4K produces identical linear
RGB PFM hashes; decoder revision is `e8ff09762481785938d8e4e01333ed3917571161`.

Four CPU ASan/UBSan tests pass without suppressions: resource budget, publication
storage, workflow publication and rate-control search. The batch-workflow suite
also exercises automatic Metal selection; its unsuppressed run reaches the
known metal-cpp null retain/release wrapper check. Batch workflow and Metal
resource accounting pass with only the existing `third_party/metal-cpp` null
suppression. Leak detection is disabled, both sanitizers halt on error, and no
GJXL source checks are suppressed. Four suites (resource budget, publication
storage, workflow publication and batch workflow) each pass ten TSan repetitions
with halt-on-error and no suppressions. The final C/C++ byte-comparison assertion
was subsequently rebuilt and rerun in Release, ASan/UBSan and TSan.

### Complete-call and footprint results

Apple M4 Pro, 48 GiB, macOS 15.6, Apple Clang 17, arm64. Release builds enable
tests/benchmarks and disable libjxl-reference fixtures and compile-time Metal
profiling. The fully resident SIMD/fused-tuned workflow uses effort 7, distance
1.2 and automatic CPU threads. The synchronous encode timer includes evaluator
teardown; backend setup, input loading, hashing and file writes are excluded.
This is the C++ complete-call boundary, not a separate C API latency study.
The returned `VarDctEncodingTiming.total_nanoseconds` retains its internal
workflow commit endpoint; it excludes prepared-state teardown and the outer
publication adapter. Its comment is clarified, and it is not substituted for
the surrounding complete-call timer in these measurements.

Seven alternating independent parent/candidate process pairs per workload each
alternate two changed images, perform 15 encodes, discard the first three and
retain six samples per image. All 840 encoded sizes/hashes are stable. Values
are medians of process medians; percentage changes are medians of paired ratios,
not ratios of the displayed medians. Negative means lower latency.

| Primary input | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 24.292 | 24.480 | +0.99% | 3/7 |
| Padded 1080p, 1919x1079 | 74.463 | 74.785 | +0.71% | 2/7 |
| Padded 4K, 3839x2159 | 251.906 | 251.830 | -0.03% | 6/7 |
| Planter 4K, 3840x2160 | 255.755 | 255.862 | +0.04% | 3/7 |

Changed-image companion changes are +0.13%, +0.00% (rounded), +0.05% and +0.20%.
The small increases are recorded costs, not proof of zero overhead. There is
no consistent complete-call benefit and no resolved large regression in this
cohort; it does not establish performance for every image or policy.

Three alternating padded-4K physical-footprint process pairs retain both inputs:

| Boundary | Parent MiB | Candidate MiB |
| --- | ---: | ---: |
| Process peak after encoding | 2970.189 | 2970.814 |
| One second idle, backend alive | 2016.595 | 2015.736 |
| One second after backend destruction | 251.939 | 251.845 |

A separate three-pair trim probe preserves all 24 output hashes. Both variants
retain 1856.690 MiB of accounted idle capacity before trim and zero after trim;
live and unbacked capacity are zero at those boundaries. Accounted peak remains
2786.793 MiB in both: the newly covered output does not coincide with this
workload's higher scratch peak. One-second post-trim physical footprint is
260.907/262.657 MiB. Median trim durations are 15.498/8.381 ms and resumed-encode
latencies are 349.432/329.593 ms, respectively. These three-pair diagnostic
observations do not establish a causal trim/resume improvement. Physical
footprint is distinct from requested/accounted capacity and varies with OS
reclamation; no physical-memory benefit is claimed for this attachment.

### Additional batch probe

The existing batch benchmark was run in three alternating independent process
pairs for synthetic 256x192 and 1920x1080 input, at in-flight counts two/four.
Each process uses three warmups and five retained samples per configuration,
effort 7, distance 1.2, forced fully resident Metal and automatic CPU threads.
The timer covers the full blocking batch encode; input generation, driver
construction and result destruction are outside it. Every result's bytes and
summary match that variant's sequential reference; reference byte counts also
match across variants. This probe does not independently compare cross-variant
batch hashes, and is not the mixed-image scheduling qualification of milestone 6.

| Input / in-flight | Parent batch ms | Candidate batch ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| 256x192 / 2 | 15.674 | 15.513 | +1.81% | 1/3 |
| 256x192 / 4 | 25.452 | 25.685 | -0.42% | 2/3 |
| 1080p / 2 | 118.920 | 120.523 | +1.95% | 1/3 |
| 1080p / 4 | 207.488 | 209.616 | -0.31% | 2/3 |

Again, the percentage is the median of paired ratios. The small mixed changes
do not justify a throughput improvement claim or dismiss the two-worker cost.

### Artifacts and reproduction

The frozen parent is `build/resident-frontend-storage`; the candidate is
`build/resident-publication`. Do not rebuild the frozen parent with new headers.
Local evidence is under `build/publication-resource-qualification`: drivers,
commands, source/library/executable hashes, corpus outputs and decode logs,
conformance manifests, timing samples, process checks, physical-footprint and
trim observations, batch logs and summaries. The shared verifier checks all 16
library hashes (including both C API archives), four complete-call/trim drivers
and both driver sources. The batch probe separately records and verifies its
two executable hashes and records the unchanged benchmark source hash.

Release, sanitizer and TSan logs are in `build/resident-publication`, including
the original unsuppressed failure. Sanitizer builds use RelWithDebInfo, frame
pointers and line-table debug information in `build/publication-sanitize` and
`build/publication-tsan`. Timed runs began only after builds/tests had finished;
the final test-only assertion was rebuilt after all timing probes finished.
Process checks reject another observed GJXL encoder/test; they do not establish
exclusive machine ownership or fixed thermal/power conditions.

After freezing the parent and building the candidate:

```sh
python3 build/publication-resource-qualification/run.py build
python3 build/publication-resource-qualification/run.py parity
python3 build/publication-resource-qualification/run.py conformance
python3 build/publication-resource-qualification/run.py performance
python3 build/publication-resource-qualification/run.py memory
python3 build/publication-resource-qualification/run.py trim
python3 build/publication-resource-qualification/summarize.py
python3 build/publication-resource-qualification/batch.py
```

## Remaining work

This checkpoint covered codestream-byte publication, not all result storage.
The subsequent [diagnostic attachment](resident-diagnostic-accounting.md) adds
AQ score histories, returned summary histories, timing-attempt arrays and GPU
profiling containers through their own public boundaries. Small control objects and allocator/driver overhead
remain the separately documented exclusions; diagnostic arrays are not silently
reclassified as excluded overhead.

Shared conservative estimators, public execution-domain configuration,
whole-workflow and aggregate retained-batch-result admission, domain-wide cache
eviction, and end-to-end waiting/rejection tests are still pending. Milestones
5 and 6 still require the last-use/reuse audit and coordinated CPU/GPU scheduling
with their own correctness and performance gates. Publication ownership alone
does not establish a hard managed-memory limit or a process-footprint bound.
