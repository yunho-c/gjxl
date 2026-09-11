# Exact histogram-distance fast path

Retain this scalar optimization: captured-population clustering is 11–22% faster,
and entropy wall time falls 7–12% on the four corpus inputs. Complete-call gains
are modest; batch-throughput improvement is not established. Output bytes and
error behavior are preserved by the qualification below.

The selected change adds a table-only path to `DirectHistogramDistance` on
Clang ARM64. It retains ordered fused double arithmetic and uses four-way loop
unrolling. Other compiler/architecture combinations and large combined totals
use the existing loop. There are no new buffers, tables, allocations or entropy
policy changes.

## Why this path is valid

Source and merged histogram counts are already checked. Every bin is bounded
by its histogram total. Once the combined total passes the existing overflow
check, a combined total at most 65536 proves every bin sum fits the exact log2
table. This hoists the table-bound check out of the bin loop.

The table's zero entry is positive zero. Including zero-count terms in the
ordered accumulation preserves its result under the default rounding mode and
removes the zero-bin branch. `std::fma` explicitly preserves the fused rounding
of the existing Clang ARM64 scalar loop. Four-way unrolling lets the compiler
schedule independent count loads and table lookups ahead of that ordered chain.
Final Shannon-cost subtractions, seed/assignment order, tie handling, all public
validation and output publication are unchanged.

## Experiments and the precision correction

The first screen tested table-bound hoisting, four-way unrolling of the original
loop, four-bin lookup preparation, combined table/preparation and two-lane NEON
count preparation. None was consistently beneficial. The NEON screen was
13–31% slower; assembly shows vector-to-scalar transfers and spills around the
scalar logarithm fallback. This does not rule out other SIMD designs.

A branchless table-only loop initially appeared 12–30% faster on eight captured
population sets. It matched the borrowed-histogram oracle and replay hashes, but
the expanded owned-histogram oracle found a distance-bit mismatch at deterministic
triple 874 (totals 56165 and 124). That variant is rejected. Cluster/output hashes
alone do not prove every intermediate distance preserves fused rounding.

Explicit `std::fma` fixed the owned/view discrepancy but lost most of the initial
speedup. Its separate seven-pair full-workflow cohort showed no general benefit:
Planter HD entropy time regressed 2.75%, while Planter 4K improved only 0.87%.
Batch timing for that version was deferred.

The final screen compared explicit-FMA unrolling by four/eight and an implicit-
contraction fast path restricted to borrowed histograms. Four-way unrolling
provides consistent improvement across the eight sets, with explicit fused
arithmetic in both histogram types and less unrolling than the eight-way option.
The borrowed-only alternative is faster in most replay rows but continues to
rely on implicit contraction and optimizes only one representation; it is not
selected.

## Selected clustering replay

| Captured populations | Baseline ms | Selected ms | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| imazen26-1029-planter-1080p-gjxl | 4.446 | 3.935 | -10.83% | 5/5 |
| imazen26-1029-planter-1080p-libjxl | 3.976 | 3.452 | -13.86% | 5/5 |
| imazen26-1029-planter-4k-gjxl | 3.642 | 3.182 | -12.61% | 5/5 |
| imazen26-1029-planter-4k-libjxl | 3.814 | 3.174 | -15.17% | 5/5 |
| kodak-kodim17-gjxl | 0.834 | 0.656 | -21.28% | 5/5 |
| kodak-kodim17-libjxl | 0.713 | 0.569 | -21.58% | 5/5 |
| padded-stress-4k-gjxl | 3.443 | 3.084 | -11.02% | 5/5 |
| padded-stress-4k-libjxl | 2.557 | 2.280 | -12.47% | 5/5 |

## Measurement protocol

Apple M4 Pro, 48 GiB, macOS 15.6, AppleClang 17, Release ARM64. Baseline is
`1d342c7` in `perf/entropy-bit-writing`, using the existing frozen
`build/dc-population-reuse/candidate` build. The candidate is built from a frozen
archive of that revision plus only the ANS change. No unrelated work in the main
GJXL checkout is included.

Replay uses the eight fixtures from `docs/ac-entropy-profile.md`: four GJXL
captures and four libjxl captures. The 4K libjxl captures represent only their
first streaming partition. Each within-row comparison uses identical populations
and GJXL's same clustering policy. Input loading and result hashing are outside
timing; output allocation and initial Shannon-cost calculation are inside.
Five alternating/rotating independent-process pairs, five warmups and 31 timed
calls per process. Percent changes are medians of paired process-median ratios.

Whole-workflow measurements use fully-resident Metal, effort 7, distance 1.2,
automatic CPU threads, fused-tuned inverse and no GPU-stage profiling. Seven
alternating independent-process pairs per input, five warmups and eleven timed
encodes. Wall-stage times are distinguished from aggregate worker counters.

Batch measurements use the frozen benchmark's complete in-memory batch boundary,
preloaded linear RGB through output, at batch sizes two and four. Initialization,
file I/O and correctness comparisons are outside timing. Seven alternating
process pairs, three warmups and five measured batches per input/size. Each
process checks outputs against its single-image reference.

## Complete-workflow confirmation

| Workload | Whole ms, baseline → selected | Whole change | Faster pairs | Serializer change | Entropy wall ms, baseline → selected | Entropy change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_128x96 | 10.173 → 10.245 | -0.49% | 4/7 | +5.38% | 0.635 → 0.643 | +1.98% |
| kodak-kodim17 | 19.958 → 19.770 | -0.75% | 5/7 | -5.58% | 2.240 → 1.939 | -11.56% |
| imazen26-1029-planter-1080p | 67.201 → 66.350 | -1.81% | 6/7 | -3.29% | 7.165 → 6.626 | -7.67% |
| imazen26-1029-planter-4k | 204.660 → 205.603 | -0.40% | 4/7 | -0.76% | 6.460 → 6.014 | -7.42% |
| padded-stress-4k | 208.592 → 202.101 | -0.71% | 6/7 | -1.41% | 6.212 → 5.814 | -7.30% |


Displayed milliseconds are medians of process medians. Changes are medians of
within-pair ratios, so dividing the displayed times need not reproduce them.
Entropy wall time improves in every pair for all four corpus inputs. The 4K
reduction is approximately 0.4–0.5 ms. Complete-call changes are smaller: Planter
4K ranges from -1.75% to +2.47%, with four of seven pairs faster; its complete-
encode gain is not established. Planter HD favors the candidate (six faster pairs,
median -1.81%). The tiny-image screen requires the separate control below.

## Batch throughput

Positive changes mean higher images/s. Changes use paired ratios; displayed rates
use medians of process-median batch durations.

| Input | Batch size | Images/s, baseline → selected | Paired change | Pair range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| kodak-kodim17 | 2 | 61.131 → 60.870 | +1.24% | -1.73% to +5.53% | 4/7 |
| kodak-kodim17 | 4 | 89.502 → 88.756 | +1.51% | -3.25% to +4.62% | 4/7 |
| imazen26-1029-planter-1080p | 2 | 18.987 → 19.244 | +0.62% | -2.53% to +5.24% | 4/7 |
| imazen26-1029-planter-1080p | 4 | 22.540 → 23.140 | +1.33% | -14.49% to +6.20% | 5/7 |
| imazen26-1029-planter-4k | 2 | 5.447 → 5.558 | +0.70% | -3.47% to +8.65% | 4/7 |
| imazen26-1029-planter-4k | 4 | 5.920 → 6.057 | +0.12% | -7.75% to +12.67% | 4/7 |

All batch medians are positive, but only four or five pairs improve per case.
Changes range from +0.12% to +1.51%, with substantial variation; the four-image
4K pair range is -7.75% to +12.67%. These results do not establish a general batch
throughput gain or a reliable 4K throughput improvement.

## Tiny-image confirmation

The initial tiny-image serializer increase (+5.38%, six slower pairs) did not
repeat in the longer confirmation. Seven pairs, ten warmups and 31 measured
encodes per process give these paired changes:

| Cohort | Complete workflow | Serializer | Entropy wall |
| --- | ---: | ---: | ---: |
| Baseline versus selected | -0.73% | -0.24% | +2.23% |
| Identical-baseline A/A control | -0.09% | +1.15% | -2.50% |

Whole-call pair ranges are -16.00% to +14.61% for the candidate comparison and
-12.55% to +2.79% for A/A. Serializer ranges are -4.59% to +9.98% versus -12.89%
to +13.76%. These observations do not establish a reliable tiny-image gain or
regression. The A/A cohort characterizes variation; it is not subtracted from
candidate measurements.

## Qualification and artifacts

Each selected-kernel oracle run checks 100000 deterministic histogram triples,
including all alphabet extents, empty/sparse/dense populations, table-boundary
counts, values near 2^53 and uint64 limits. Both owned and borrowed template
instantiations and null-output calls are checked. Of the triples, 69539 have
both tested distances valid and 30461 include a combined-total overflow; output
bits and status match the frozen scalar reference. The final explicit loop
emits ordered ARM64 fused multiply-subtract instructions.

The selected Release suite passes 128/129. The sole failure is the inherited
`quantization_pipeline` score at index 1: actual `0.24919039011001587`, expected
`0.24914586544036865`. No golden or tolerance changed. All four ASan/UBSan suites
pass: entropy, entropy storage planning, serializer storage planning and serializer
storage (leak detection disabled, no suppressions). All 56 corpus/policy outputs
are byte-identical, three representative decoded-PFM pairs match, and all 22
pinned codestream conformance fixtures pass. Qualification includes high-density,
maximum-compression, effort and resource/error-path coverage from the existing
suites. No separate memory-saving or RSS claim is made.

Local artifacts are under
`/Users/yunhocho/GitHub/gjxl/build/entropy-distance-kernel`, while the source change
is in the sibling `gjxl-entropy-bit-writing` worktree. Controllers, source copies,
build commands, assembly, oracles, raw measurements and hashes are retained.
`screen`, `followup` and `last-screen` are separate kernel cohorts; `final-replay`
and `workflow` belong to the rejected explicit-FMA version without unrolling.
`selected-workflow` and `selected-batch` measure the final four-way version.

The first screen attempt completed its oracles but stopped before timing because
the sandbox prohibited `ps`. Timing resumed with process inspection available.
The retained controllers check for competing builds/encoders before and after
processes; this is not an exclusive-machine or controlled thermal experiment.
Frozen source archives needed their pinned Metal C++ headers copied separately
because `git archive` omits submodule contents. No timing was taken from the
initial failed build.

The retained cohorts contain 600 clustering replay processes (18600 measured
calls), 140 ordinary workflow processes (1540 measured encodes, including the
rejected non-unrolled FMA version), 14 selected batch processes (420 measured
batches), and 28 tiny-control processes (868 measured encodes). Qualification
and oracle runs are separate from these performance counts.

After collection, these commands audit recorded hashes, source/runtime identities,
raw sample inventories, recomputed paired summaries and qualification outputs:

```sh
cd /Users/yunhocho/GitHub/gjxl
python3 build/entropy-distance-kernel/audit.py
python3 build/entropy-distance-kernel/audit-first-fma.py
python3 build/entropy-distance-kernel/audit-selected.py
```

`selected-audit.json` records the final source, binary and qualification checks.
`evidence-sha256.json` freezes the controllers and retained evidence after the
audits complete. Full build directories remain local; their relevant runtime
identities are included in the benchmark manifests and audit results.
