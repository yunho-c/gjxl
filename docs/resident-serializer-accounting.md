# Managed serializer tokens and entropy models

This checkpoint builds on host attachment `aa88f13`. It accounts for serializer
container backing while preserving encoding policy, token order, candidate search,
and tie decisions. It is part of milestone 4 of [resident execution](resident-execution.md),
not completion of whole-workflow accounting or admission.

## Ownership and coverage

`codestream_internal::Storage<T>` is a vector using the allocation-owned managed
allocator with a fixed serializer owner tag. It charges the allocator's capacity
request, not logical size, and inherits the current reservation at allocation time.
The [host attachment record](resident-host-accounting.md) defines allocation-ticket
lifetime, capacity replacement, caller exclusions and worker propagation.

The converted owners include:

- AC values, contexts, map-independent templates, block keys, context/symbol
  populations, nonzero maps, strategy anchors and per-worker tokenization scratch.
- DC residual and AC-metadata tokens, their group owners and predictor scratch.
- Coefficient-order counts, scans, permutation scratch, tokens and worker outputs.
- Block-context candidate maps, occurrence clustering and quantization thresholds.
- Prefix/ANS model owners, reverse lookup maps, reciprocal tables, aggregated
  values, normalization/search scratch, prepared clusters and deferred candidates.
- Section/model arrays, candidate-size measurements, worker bookkeeping, section
  lengths and reverse ANS bit chunks. Bit-writer backing was already attached.
- Sparse aggregation's hash-map nodes and allocator-rebound bucket arrays, as well
  as its dense counts and sorted result. Uncommon uint32 values remain sparse.

Nested owner arrays and their payloads are distinct allocations; borrowed spans
and token views carry no tickets. Moving a group/model transfers its storage and
charges without reallocating it. A consumer can retain it beyond its producer's
scope and reservation, and free it on another thread. Copies allocate in the
current domain. These rules do not authorize sharing another job's allowance.

The fixed-size modular context tree is copied into a 313-token stack array and
borrowed directly by the optimizer. This replaces two heap copies and the legacy
input-view adapter, without changing token values. The JPEG XL default block map
is compared against its immutable table without constructing a function-static
owning map. The first full qualification found that such a static would otherwise
retain 39 newly accounted bytes from the first encode indefinitely. The existing
Metal workflow accounting test caught this; its zero-live-storage check remains.

All immutable ANS logarithm/population tables and fixed-size Huffman scratch remain
arrays. This is not an interception of the standard library, process RSS, thread
stacks, command-driver storage or allocator/header overhead.

## Sorting and exact tie order

`std::stable_sort` can acquire scratch outside a container's allocator. The four
serializer uses are now `std::sort` with the original stable tie order explicit:

| Sort | Preserved tie order |
| --- | --- |
| ANS normalization remainders | Ascending symbol, the original insertion order |
| Simple Huffman symbols | Ascending symbol, the caller's original order |
| Coefficient scans | Natural scan rank, not numeric coefficient offset; the inverse rank is managed storage shared across channels |
| Adaptive block clusters | Rank entering the current merge iteration, not original cluster ID; ranks fit a fixed 256-entry stack array |

In particular, merging changes cluster order, so a numeric-ID tiebreaker would
not preserve the prior policy. These changes remove hidden sort allocations;
they do not claim faster sorts or change the arithmetic used to calculate costs.

## Compatibility and failure behavior

The C ABI and public final codestream `std::vector<uint8_t>` remain unchanged.
Lower-level C++ token/model structs have different internal allocator types and
require rebuilding consumers. Code explicitly naming their former vector types
may need `auto`, spans, or iterator assignment; this is not a binary-compatibility
promise for internal headers.

Output functions retain template adapters for ordinary caller-supplied vectors.
Managed production calls select the non-template overload directly, so they do
not take the adapters' publication copies. Legacy adapters first build managed
candidates, then construct the caller-allocator candidate before committing it.
Nested legacy contexts require an explicit row copy. AC metadata's auxiliary
transform count commits only after token publication succeeds. Null outputs retain
validation rather than becoming ambiguous overloads.

Managed allocation failures preserve their original status through serializer
exception boundaries and joined workers. Aggregation, deferred ANS finalization,
and Prefix serialization's validation allocations now have allocation-error
boundaries too. Thread-creation allocation failures join already launched serializer
workers before returning. Scheduling policy is unchanged.

The allocator's thread-local test hook can fail the backing allocation after a
specified number of successful allocation boundaries. A rejected ticket does not
consume the hook; a backing failure releases pending credit. No global allocator
override or production environment switch is introduced.

## Permanent coverage

`serializer_storage` checks exact nested capacity/class accounting, move ownership,
destruction after reservation close on a different thread, and managed DC tokens
against the ordinary-vector API. It exhausts every backing-failure position in a
sparse aggregation fixture and small model fixtures: 30 for fast Prefix, 74 for
balanced ANS, 1,246 for high-density ANS, and 491 for maximum-compression ANS.
Each failure preserves output and releases candidate tickets; the terminal run
must succeed with its failure hook unconsumed and match its oracle.

An explicit allowance that fits only the dense aggregation table rejects the
first sparse node and cannot escape into the default domain. Retained Prefix/ANS
owners are reconciled against each vector capacity, including nested reverse maps.
A failing legacy allocator verifies token/count publication atomicity and recovery.
The entire explicit-domain test leaves the default domain's peak at zero.

The existing serializer test exercises all three entropy policies with one and
four CPU participants, exact bytes, insufficient-reservation rejection and recovery.
Its 64 MiB test allowance is not a production estimator. Existing token/model
goldens and whole-workflow qualification remain required; failure-fixture coverage
is not a proof of every possible allocator or OS thread-creation failure.

## Correctness qualification

The fresh candidate is `build/resident-serializer-storage`; the frozen parent is
`build/resident-managed-host` at `aa88f13`. The initial candidate full suite exposed
the static-map retention described above in addition to the known CPU quantization
golden mismatch. After fixing that retention, the final rebuilt candidate passes
67/68 Release tests and the rerun parent passes 66/67. The sole failure in each is
CPU `quantization_pipeline`: actual `0.24919039011001587`, expected
`0.24914586544036865`. No numerical golden or tolerance is changed.

Allocator, serializer storage, parallel serializer and Metal resource tests each
pass 25 Release repetitions. The first three also pass ten repetitions each in
the final rebuilt ThreadSanitizer configuration. Eight CPU ASan/UBSan targets pass
without suppressions: serializer storage, entropy, AC/DC groups, block contexts,
serializer, allocator and sections. Three additional targets pass: workflow,
completed frame and Metal resource accounting. Those three use the existing
vendor-header-only UBSan null suppression; all ASan runs disable leak detection.
The initial sanitizer runs are retained separately from the final rebuilt results.

All 56 corpus/policy JXL SHA-256 comparisons match the frozen parent. Separate
pinned `djxl` decoding of Kodak17, planter 4K and padded stress 4K produces matching
linear-RGB PFM hashes. Both configurations pass all 22 pinned conformance fixtures.
The decoder remains `e8ff09762481785938d8e4e01333ed3917571161`.

## Complete-call and memory measurements

Apple M4 Pro, 48 GiB, macOS 15.6; AppleClang 17 Release, SIMD/fused-tuned Metal,
fully resident, distance 1.2, effort 7 and automatic CPU threads. Tests and
benchmarks are enabled; libjxl-reference fixtures and compile-time Metal profiling
are off. Timers surround the complete synchronous encode including evaluator
teardown, excluding backend creation, input loading, hashing and output writes.

Seven independent process pairs alternate parent/candidate order. Each process
alternates two changed images, discards three warmups and retains six calls for
each image. All 840 outputs preserve their corresponding hashes. Times are medians
of process medians; changes are medians of paired ratios, not ratios of the shown
times. Positive changes mean higher latency.

| Primary input | Parent ms | Candidate ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 23.016 | 23.444 | +2.06% | 1/7 |
| Padded 1080p, 1919x1079 | 71.820 | 73.445 | +2.43% | 0/7 |
| Padded 4K, 3839x2159 | 243.527 | 244.580 | +0.31% | 2/7 |
| Planter 4K, 3840x2160 | 245.374 | 246.792 | +0.85% | 1/7 |

The changed-image companions regress by +2.14%, +1.18%, +1.58% and +0.59%,
respectively. Serializer-stage paired changes across all eight images range from
+3.92% to +7.62%; six are slower in every pair, and the other two in six of seven.
This is measurable accounting/representation overhead, not a zero-cost attachment
or a speedup. The experiment does not isolate allocator/ledger bookkeeping from
the changed sorting implementation and storage representation. No kernel or
encoding-policy change is included.

Three separate alternating padded-4K process pairs report physical process peak
2974.236/2969.658 MiB (parent/candidate), and one-second backend-alive idle footprint
2021.876/2020.142 MiB. These small differences do not establish a material physical
memory saving. One second after backend destruction, footprints are
258.017/255.595 MiB. Physical counters are not requested capacities or a hard limit.

The three-pair trim/resume probe retains two source images, performs A/B/A encodes,
waits one second, trims, waits another second, and encodes A again. Both builds
retain 1856.690 MiB of managed idle capacity before trim and zero afterward. Live
capacity and unbacked reservations are zero at the post-encode and post-trim
boundaries. All 24 outputs match their corresponding hashes. Accounted backing
peak increases from 2559.230 to 2755.063 MiB because coverage now includes serializer
storage; this is not an equivalent increase in physical peak.

One second after trim, physical footprints are 265.345/262.345 MiB. In this separate
cohort, the one-second pre-trim footprints are 2210.579/2003.954 MiB; the difference
from the preceding idle cohort illustrates OS timing variability, not proven cache
savings. Median trim-call times are 5.687/6.265 ms, with very unstable paired ratios
(one parent call was nearly zero). Post-trim resume medians are 306.518/320.496 ms;
the median paired regression is +4.56%, and all three candidate calls are slower
(+2.57% to +4.95%). These are small recovery cohorts, not warm-stream estimates.

All builds and tests in this worktree finished before performance measurement.
The harness records process snapshots and rejects observed overlapping GJXL
executables. This is not exclusive-machine or controlled thermal/power testing.

## Artifacts and reproduction

`build/serializer-resource-qualification` contains the driver-build manifest,
library/driver/source hashes, parity outputs and decoder logs, both conformance
records, all timing/process samples, memory/trim probes, and `summary.json`.
The final summary verifies all fourteen libraries, four drivers, both driver
sources, completed manifests and timed output hashes. Parent drivers were copied
from the previously qualified host checkpoint, not rebuilt with candidate headers.

The final full Release log is `build/resident-serializer-storage/final-ctest.log`;
`full-ctest.log` retains the initial static-map failure. Parent and repeated Release
logs are `parent-ctest.log` and `repeated-ctest.log` in the qualification directory.
Final ASan/UBSan logs are `final-cpu-ctest.log` and `final-metal-ctest.log` under
`build/serializer-storage-sanitize`; the TSan log is `final-ctest.log` under
`build/serializer-storage-tsan`. Sanitizer builds use RelWithDebInfo and
`-fno-omit-frame-pointer`, with `-fsanitize=address,undefined` or `-fsanitize=thread`.
All enable halt-on-error; ASan disables leak detection. Only the three Metal
ASan/UBSan targets use `build/handoff-qualification/ubsan.supp` for vendor-header
null checks. No GJXL source checks are suppressed.

With the frozen parent retained and the candidate built as above:

```sh
python3 build/serializer-resource-qualification/run.py build
python3 build/serializer-resource-qualification/run.py parity
python3 build/serializer-resource-qualification/run.py conformance
python3 build/serializer-resource-qualification/run.py performance
python3 build/serializer-resource-qualification/run.py memory
python3 build/serializer-resource-qualification/run.py trim
python3 build/metal-resource-qualification/summarize.py build/serializer-resource-qualification resident-managed-host resident-serializer-storage
```

These are retained local qualification artifacts, not installed tools. The
permanent source tests provide durable regression coverage. Do not rebuild the
frozen parent against the converted headers.

## Remaining work

This attachment deliberately does not claim complete managed host coverage:

- Final candidate codestream publication still uses the public ordinary-vector
  boundary. Retained batch results need accounting until public ownership handoff.
- Frontend raw quant/sharpness/mask arrays, CPU AC-search costs and metadata,
  completed-frame host metadata and other image-shaped vectors need attachment.
- C input conversion precedes the current workflow scope and needs domain/admission
  integration. Caller input and previously returned output remain excluded.
- Shared checked planners must account for policy/worker-dependent capacity,
  replacement overlap and retries. Public domain configuration, cache shedding,
  whole-work admission and aggregate CPU scheduling remain outstanding.

Unconfigured allocations still use an individual reservation state in the shared
unlimited default domain. This checkpoint must measure that overhead, not assume
that future job-level reservation sharing has already removed it. No memory
pressure path may silently reduce encoding effort, candidates, or quality policy.
