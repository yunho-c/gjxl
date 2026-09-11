# Reuse DC populations for ordinary entropy selection

The balanced DC entropy task now collects default-HybridUint populations once,
uses them to select Prefix versus ANS, and passes them to the existing prepared
ANS builder. This removes the separate checked-conversion selection traversal
for DC streams that select ANS. Collection remains inside the parallel entropy
phase, alongside AC work; it is not moved into serial DC tokenization.

Baseline: `aa5025c`, on `perf/entropy-bit-writing`. Measurements and qualification
were performed on 2026-09-10. The preceding investigation is recorded in
[DC histogram preparation profile](dc-histogram-preparation-profile.md).

## Implementation and contracts

`CollectDefaultEntropyPopulations` accepts token-stream views and returns owned,
per-context populations under the existing managed serializer allocator. It uses
the constexpr validated converter with the constant default configuration. Stream
validity, context range, total-token overflow and extra-bit overflow remain
checked. The checked global token total bounds every bin and per-context count;
the default configuration maps all uint32 values into the fixed ANS alphabet.
There is no per-token success `Status` construction. Failed collection leaves
its destination unchanged.

The existing population-based selector retains the exact fewer-than-100-token
and all-singleton-context rules. The prepared ANS builder still validates all
population bins and totals before borrowing views and constructing its model.
Clustering arithmetic, ties, model serialization and token emission are unchanged.

Only balanced DC takes the new route. AC already supplies populations where
supported. High-density and maximum-compression retain their previous paths.
When Prefix is selected, its existing builder still prepares its own data. Thus
this is an incremental shared-preparation refactor, not a common rewrite of both
entropy builders or a change to encoding policy.

The 45-context population payload is 93,240 bytes (about 91 KiB) on this ARM64
build, owned until the DC entropy task returns. The serializer control-storage
bound explicitly accounts for it. The existing DC stream-view allowance covers
the views now created at that call site. This is an additional scoped owner and
reservation allowance; it is not a claim of lower peak memory or RSS. No shader,
public API or runtime profiling hook is added.

Collection time contributes to the existing ANS histogram-preparation work
counter. That counter does not include the old standalone selection scan, so its
before/after delta alone understates the eliminated CPU work. For rare balanced
DC Prefix outcomes it now also includes the initial population preparation.

## Isolated DC entropy replay

Captured original DC/metadata streams from Kodak17, Planter 1080p, Planter 4K and
padded 4K use configuration `(4, 2, 0)` and 45 contexts. The Release replay includes
the unchanged old token-selection route and the new collected-population route
in one binary. Each runs the complete ordinary DC entropy optimization, including
model writing and deferred token-cost behavior. Collection allocation and release
are inside the new route's timer. Fixture loading, output equality comparisons
and returned-code destruction are outside both timers.

Five alternating independent-process pairs per input, five warmups and 31 measured
calls per process: 1,240 measured calls. Every call checks the complete selected
model and cost against the old route. Values below are medians of process medians;
percentage changes are median paired ratios and need not equal ratios of the
reported milliseconds.

| Input | Old DC task (ms) | Reused populations (ms) | Paired time change |
| --- | ---: | ---: | ---: |
| Kodak17 | 0.492 | 0.403 | -19.48% |
| Planter 1080p | 1.830 | 1.349 | -26.78% |
| Planter 4K | 4.771 | 2.947 | -38.23% |
| Padded 4K | 3.540 | 2.018 | -42.80% |

All five pairs improve on every input. The two 4K cases save roughly 1.5–1.8 ms of
CPU time per DC entropy task. These are warm isolated CPU durations, not
complete-encode latency reductions.

## Complete-workflow comparison

Apple M4 Pro, 48 GiB, macOS 15.6, AppleClang 17 Release C++20. Fully-resident Metal,
effort 7, distance 1.2, automatic CPU threads, fused-tuned inverse, no GPU-stage
profiling. Five alternating independent-process pairs per input, three warmups
and seven timed encodes per process: 350 measured encodes. The public workflow
boundary excludes initialization, file I/O and process startup.

| Input | Baseline whole (ms) | Candidate whole (ms) | Paired whole change | Paired serializer change | Paired entropy-stage wall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 9.769 | 8.491 | -4.79% | +3.49% | -0.53% |
| Kodak17 | 19.785 | 19.509 | -0.32% | -2.64% | +3.16% |
| Planter 1080p | 64.560 | 64.248 | -0.58% | -1.76% | +0.04% |
| Planter 4K | 200.030 | 200.612 | +0.27% | +0.30% | +0.04% |
| Padded 4K | 211.529 | 202.333 | -1.63% | +1.22% | +0.51% |

There is no clear general single-image latency gain. The tiny and padded whole
results vary substantially across pairs; their apparent gains are not accompanied
by consistent serializer gains. Planter 4K is effectively unchanged. This agrees
with the earlier task timelines: AC normally finishes after DC, hiding much of the
DC work from the completion boundary.

One padded-4K candidate sample failed the post-process quiet guard because an
unrelated compiler started during the measurement. Its raw files and rejection
reason are retained under `workflow/excluded/`; that pair was rerun. Completed
pairs passed the before/after guards. No unrelated process was stopped.

## Batch throughput

Five alternating independent-process pairs, three warmups and five measured
batches per process, at batch sizes two and four: 300 measured batches. Timing
covers completion of the whole batch, excluding initialization and correctness
checks. Settings match the workflow comparison. Positive changes mean greater
throughput; percentages are medians of paired throughput ratios.

| Input | Batch size | Paired throughput change | Pair range | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 2 | +1.76% | -6.33% to +18.87% | 3/5 |
| Kodak17 | 4 | +0.87% | -10.92% to +7.83% | 3/5 |
| Planter 1080p | 2 | -2.21% | -5.34% to +0.02% | 1/5 |
| Planter 1080p | 4 | +6.15% | -10.65% to +14.38% | 4/5 |
| Planter 4K | 2 | -0.08% | -17.15% to +28.41% | 2/5 |
| Planter 4K | 4 | +0.13% | -15.08% to +22.35% | 3/5 |

The 4K results are effectively unchanged and vary widely across pairs. The
four-image HD case favors the candidate, while two-image HD favors baseline.
This cohort does not establish a general batch-throughput improvement.

The two-image HD regression prompted a focused confirmation: seven alternating
pairs, five warmups and nine measured batches per process, followed by an
identical-baseline A/A control with the same protocol. The candidate confirmation
shows **+1.32%** median throughput (-0.51% to +2.58%, six of seven pairs faster).
The A/A control shows **+0.24%** (-0.74% to +2.04%, five of seven pairs positive).
These are separate cohorts, not values to subtract into a corrected speedup.
The initial -2.21% result did not reproduce; the confirmation suggests a small HD
benefit, but does not establish a general or 4K throughput gain.

Retain the refactor for its exact, repeatable reduction in DC CPU work. The extra
scoped population storage is explicit, and the study found no reproducible
complete-workflow regression. Prioritize the AC entropy critical path for further
single-image latency work.


## Qualification and artifacts

- Release: 128/129 tests pass. The only failure is the same baseline
  `quantization_pipeline` mismatch: index 1 is `0.24919039011001587` rather than
  `0.24914586544036865`. No golden or tolerance changed.
- Four ASan/UBSan suites pass: entropy, entropy storage planning, serializer
  storage planning and serializer storage. Leak detection is disabled; sanitizer
  errors halt execution. Allocation-failure injection covers the new owner.
- Independent checked-conversion population tests cover empty sections/contexts,
  tiny-stream thresholds, singleton contexts, uint32 extremes, interleaved/split
  views and atomic rejection. Prepared versus scanned ANS models and costs match.
- All 56 canonical corpus/policy output comparisons are byte-identical. Three
  representative output pairs also decode to identical pixels using pinned
  `djxl` (`e8ff0976`). All 22 pinned codestream conformance fixtures pass.
- The structured serializer oracle also compares 72 complete codestreams across
  small/padded/multi-DC-group frames, three entropy policies and two coefficient
  order policies against baseline. All outputs match byte for byte.
- Baseline and candidate Metal libraries have identical SHA-256 hashes.

Artifacts are local under `build/dc-population-reuse/`: frozen `source/`,
`identity.json`, build and test logs, `parity/`, `conformance/`, `replay/`,
`workflow/`, `batch-throughput/`, `hd-confirm/` and `hd-control/`. Scripts retain commands, binary/input hashes,
raw samples and process records. `audit.py` verifies source and binary identities,
output hashes, sample inventories and timing summaries; `manifest.json` hashes
the retained diagnostic evidence. Build artifacts are ignored by Git.
