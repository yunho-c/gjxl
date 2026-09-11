# Metal completed-frame storage reuse

The completed-frame buffer can now be reused after its final owner releases it.
On the qualified M4 Pro, warm 4K workflow timing improves by **3.8–4.6 ms
(2.1–2.5%)**. A separate timer around the complete public encode call confirms a
warm benefit, with a noisier **5.4 ms (2.9%)** median paired improvement on the
4K photo. AQ's driver interval on cache hits drops from approximately **3 ms to
0.025 ms**.

This is a throughput optimization with a memory tradeoff. A 4K backend may retain
an additional **102 MiB of purgeable idle capacity**. Fresh-process first-encode
latency is approximately unchanged in this measurement; repeatedly trimming
before every encode was **7–10 ms slower** and should not be presented as a cold
latency improvement.

Base: `318893b3efdb179ea778b5ef04c02ea2a79be253`, including the preceding AC
scratch and two-plane cold-preparation changes. Branch:
`perf/metal-frame-output-cache`. Device: Apple M4 Pro, macOS 15.6 (24G84).
Qualification took place September 11–12, 2026. Artifacts are under
`build/frame-output-cache/` in this worktree.

## Ownership, accounting, and limits

Each backend retains at most one completed-frame **allocation**, without
retaining frame metadata or image results. The default
`MetalBackendOptions::completed_frame_cache_bytes` is 128 MiB; zero disables
retention. A separate process-wide counter caps idle completed-frame capacity
at 256 MiB. Active outputs are outside these idle limits and retain their own
independent storage. Oversized outputs continue to work and are freed at return.

Only successfully completed and published output can return to the cache. Its
final owner proves that GPU work and CPU serialization have finished. A failed
operation discards the buffer, including a failure after GPU completion. A weak
registry reference permits a frame to outlive its backend without retaining
that backend or its preparation arenas. Registry locking excludes backend
unregistration while a returned buffer finds its owner.

Idle buffers become purgeable-volatile. Acquisition restores nonvolatile state;
reclaimed buffers are discarded and allocated again. Explicit trimming advances
the existing preparation-cache generation, so a pre-trim live frame cannot
repopulate the cache. `GpuBackend::TrimPreparationCache()`,
`TrimVarDctPreparationCache()`, and `gjxl_trim_preparation_cache()` cover the new
capacity. Domain eviction only drops matching-domain idle storage.

Idle capacity remains charged to its resource-budget domain. Reuse transfers
that charge to the next reservation in the same domain; a cross-domain request
releases the old backing before allocating within the correct reservation.
Queued admission prevents returned frames from becoming idle. The cache does
not bypass a workflow's memory bound.

Completed-frame plans round capacity up to 1 MiB. Strategy-dependent anchor
counts can therefore change without forcing a different allocation for each
4K image. Reuse requires an exact capacity match, and admission charges that
full rounded capacity. Offsets and used lengths are unchanged; padding is never
serialized. All required contents and frame metadata are rebuilt.

| Completed-frame device capacity | Baseline | Candidate |
| --- | ---: | ---: |
| Padded 3839×2159 | 106,300,861 B | 106,954,752 B / 102 MiB |
| Planter 3840×2160 | 106,727,351 B | 106,954,752 B / 102 MiB |

Rounding adds less than 1 MiB per active output, including small outputs and
outputs that exceed the cache limit. Retained capacity is an accounting measure,
not a resident-memory measurement. OS memory-pressure reclamation remains
possible. There are no shader, AC-search cache, or asynchronous-residency changes.

## Batch admission correction

The initial cache commit, `53a5712`, omitted completed-frame capacity from the
batch planner's idle-pool inventory. The individual allocation remained charged,
but a batch could retain more idle capacity than its declared bound. A new
native regression test failed before the correction: four 257×257 resident
images left 19,637,213 B idle against a 15,442,909 B idle allowance, a **4 MiB
shortfall**. The encodes themselves succeeded; no actual allocation failure was
reproduced.

The resident and unified workflow plans now carry a fifth idle-pool contribution
through the batch accumulator. A batch reserves one maximum completed-frame
cache across its requests, independent of worker count. This adds 102 MiB to the
idle allowance for the qualified 4K geometry. CPU and compatibility routes do
not create completed-frame cache entries and contribute zero to this slot.

The bound uses the production backend's default 128 MiB limit. Variable
strategies can produce a cacheable output even when the maximum-anchor plan is
too large: at 4352×2560, dense anchors need 129 MiB but a smaller anchor table can
fit a 128 MiB bucket. That case retains a 128 MiB allowance; fixed-DCT8 output at
the same geometry has no allowance because it cannot fit. Geometries whose
minimum buffer already exceeds the limit also contribute zero.

The existing batch policy now sees the complete idle allowance when deciding
whether to preserve caches, reduce concurrency, or trim before reusing a work
slot. This corrects planning; it does not change cache ownership, shader work,
or the repeated-trim performance regression above. Batch throughput remains a
separate measurement question.

Permanent qualification includes 36 native batches and 144 byte/summary
comparisons against single-image oracles: repeated calls, one and three workers,
same-size and changing-size resident images, mixed CPU/resident/compatibility
routes, exact cache allowances, the byte below the cache-retention threshold,
and minimum one-slot budgets. At all-workers-complete publication, observed idle
capacity must fit the plan; all calls also check the shared hard limit and
reservation cleanup. Static planning checks cover the 4K allowance, 128 MiB
boundary, oversized outputs, and one-cache aggregation across requests.

The original small native probes now match their idle bounds exactly. A separate
4K check produces eight byte-identical outputs against the frozen baseline, with
Metal API/shader validation enabled: four images with two work slots and caches
retained, then four with a minimum one-slot budget and per-image trimming.
Observed idle capacity is exactly the planned 1,988,174,140 B in the first case
and zero in the second. The follow-up passes 8/8 focused tests, 5/5 Metal
validation tests, and 3/3 AddressSanitizer tests with leak detection disabled.
The full Release suite remains 126/127, with the same inherited CPU golden
mismatch documented below.

Follow-up evidence and the pre-fix failing test are retained under
`build/frame-output-cache/batch-fix/`.

## Warm timing

All performance runs use Release code, effort 7, distance 1.2, fully resident
Metal, automatic CPU participation, and no final-score diagnostic. Files are
loaded before timing. Baseline/candidate order alternates by round, with
randomized workload order. There were no competing compiler or encoder
processes in the saved before/after process snapshots.

The standard workflow campaign uses eight independent paired processes per
input, three warmups after the initial validation encode, and seven retained
samples per process: 224 measured encodes. The table reports medians of process
medians; paired savings are calculated independently from paired process medians.

| Input | Baseline workflow | Cached workflow | Median paired saving | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K | 182.288 ms | 178.572 ms | 3.839 ms / 2.11% | 8/8 |
| Planter 4K | 186.755 ms | 181.709 ms | 4.588 ms / 2.45% | 8/8 |

The workflow timer covers input preparation, quantization, serialization, frame
release, and summary construction. It excludes backend creation and stops before
the final public ownership handoff. The paired saving in the quantization phase
is 3.59 ms and 3.51 ms respectively. Independently summarized phase medians are
not additive; the entire saving is not attributed to driver work alone.

To check the complete boundary, `whole-call.cpp` times the public
`EncodeLinearRgbVarDctCodestream` call externally, including its final handoff and
cleanup. Identical drivers link the frozen baseline and candidate Release
libraries. Eight paired processes, four untimed encodes, and seven measured
encodes yield 112 samples on the Planter photo:

| Complete public call | Baseline | Candidate | Median paired change |
| --- | ---: | ---: | ---: |
| Warm Planter 4K | 188.344 ms | 181.749 ms | −5.391 ms / −2.89% |

Six of eight pairs improve; paired changes range from −10.21 to +14.62 ms. This
confirms a warm benefit under the full boundary, while showing more variance
than the standard workflow campaign. All retained final codestreams match the
fresh baseline oracle exactly.

## Allocation and driver attribution

The separate diagnostic campaign uses identical timing hooks on both arms,
three paired processes, three measured samples, and three warmups after initial
validation. It covers both inputs with warm and trimmed caches, for 72 measured
encodes. Its whole-workflow timing is noisier and is not the primary speedup
estimate. All four existing preparation arenas hit on all 18 measured warm
candidate encodes. Completed-frame allocation is absent on 17 of those 18;
one measured encode takes a fresh 102 MiB allocation. The trace does not record
the reason for that individual cache miss.

| Warm AQ policy submission | Baseline driver interval | Candidate driver interval |
| --- | ---: | ---: |
| Padded 4K | 2.857 ms | 0.023 ms |
| Planter 4K | 3.001 ms | 0.026 ms |

These are `kernelEndTime - kernelStartTime` intervals, separately measured from
GPU execution and host wait. On a hit, fresh Metal allocations fall from 24 to
23 per encode: the roughly 101–102 MiB completed output disappears, leaving the
23 AC allocations totaling 18.45 MiB. This establishes that output storage reuse
removes AQ's repeated first-use driver cost. It does not identify individual
kernel/driver functions or assign every millisecond of the total saving to that
interval. The shaders are byte-identical across all four benchmark builds.

## Cold and trim behavior

A fresh-process external-call campaign uses eight paired processes on Planter,
with one measured first encode each. Unlike the warm workflow benchmark, it
includes lazy backend creation; it excludes input loading, process startup, and
process-exit teardown. The medians are 298.402 ms baseline and 299.591 ms
candidate. The paired median change is **+0.720 ms / +0.25%**, with a broad
−99.02 to +19.51 ms range. No first-encode latency benefit is established.

The allocation-cold diagnostic campaign trims preparation caches before every
encode, outside the timed boundary. It uses six paired rounds and five measured
samples per process, totaling 120 encodes:

| Trim before every encode | Baseline workflow | Candidate workflow | Median paired change |
| --- | ---: | ---: | ---: |
| Padded 4K | 265.786 ms | 274.285 ms | +9.494 ms / +3.49% |
| Planter 4K | 267.155 ms | 277.112 ms | +6.604 ms / +2.52% |

Only one of six pairs improves for each input. The smaller traced campaign also
shows a positive median change, approximately +12.8 ms and +9.4 ms. Much of the
cold wait difference appears in reference preparation, before completed-frame
allocation; its precise cause is unresolved. Therefore these measurements do
not establish that rounding or the completed-output allocation itself costs
7–10 ms. They do establish a regression under this repeated-trim protocol.

Use the cache across normal repeated encodes; trim when the application becomes
idle or needs to release capacity. A cache hit is opportunistic, and neither
forced trimming nor OS reclamation is claimed to preserve warm performance.

## Correctness and lifecycle qualification

- Release suite: **126/127 passed**. The sole failure is the unchanged CPU
  `quantization_pipeline` golden mismatch, reproduced in the saved baseline
  suite: actual `0.24919039011001587`, expected `0.24914586544036865`.
- Focused Metal API and shader validation: **4/4 passed**.
- AddressSanitizer: completed-frame, cache-admission, and resource-budget tests
  **3/3 passed**; leak detection was disabled for the macOS runtime.
- All **38 canonical corpus images** produce byte-identical codestreams against
  fresh baseline encodes. Four pairs also have identical decoded PFM hashes.
- **16 public-API encodes** alternate three different 4K photos and a small Kodak
  image over four rounds, with an explicit trim before round three. Every output
  matches its fresh baseline oracle, with Metal API/shader validation enabled.
- New lifecycle coverage exercises live-output independence, actual producer
  reuse, post-completion failure discard, disabled/oversize caches, reclaimed
  storage, stale-generation return, same/cross-domain accounting, domain-only
  eviction, queued admission, the process cap, and backend-destruction races.

The first full-suite attempt exposed an accounting test that did not trim after
releasing a newly cacheable output; it now explicitly trims before asserting an
empty domain. An early lifecycle fixture used a sub-page buffer whose Metal
purgeability transition was ignored; it now uses the real minimum 1 MiB bucket.
An initial focused run also encountered the unchanged Butteraugli reuse check;
it passed subsequent focused, full-suite, validation-layer, and baseline runs.
All attempt logs are retained rather than presenting the first run as clean.

Reproduction scripts, raw samples, process snapshots, binaries and source
hashes, exact outputs, decoder hashes, diagnostic-only patches, and verification
commands are recorded under `build/frame-output-cache/`. Start with `audit.json`,
`verification-manifest.json`, the campaign `analysis.json` files, and
`whole-call/analysis.json`. No measurement automatically starts when importing
results.
