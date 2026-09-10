# Exact direct-ANS clustering optimization

Retain the small scalar change: real-histogram clustering replay is 22–26% faster,
and 4K entropy-stage wall time falls about 1.1 ms. Complete-call improvements are
modest and workload-dependent; a 4K batch-throughput gain is not established.
The larger paired-distance experiment is not retained.

Baseline: `f719aa9`, including the qualified word writer and producer-side ANS
packing. This follow-up changes only direct-ANS histogram-distance evaluation.
The qualified change is retained on `perf/entropy-bit-writing`.

## Implementation and exactness argument

The private distance helper returns static error text, with `nullptr` denoting
success. Its two callers construct `Status::InvalidArgument` only on failure.
The public status type, error messages and output-publication behavior remain
unchanged. Release ARM64 builds inline the helper after this change; the baseline
has two outlined instantiations and materializes a 32-byte success result.

The second change removes per-bin addition-overflow tests from this helper.
Every source histogram comes from checked token accumulation or a fully validated
fixed population. Cluster copies and `AddHistogram` preserve those sums. Thus
`left_bin <= left_total` and `right_bin <= right_total`. The retained combined-
total overflow check proves each bin sum fits too. External population validation,
merge checks and the combined-total check are retained.

The exact log2 table, fallback logarithms, ordered fused multiply-subtract chain,
final cost subtractions, histogram traversal, clustering updates and tie order
are unchanged. Precomputing `count * log2(count)` was not attempted: the baseline
uses fused multiply-subtract instructions, so separately rounded products would
not establish equivalent floating-point behavior.

## Diagnostic evidence

A captured Planter 4K AC partition has 6930 contexts and makes 190,960 distance
calls over 4,461,551 bin visits. Of those visits, 45,500 (1.02%) use the large-count
log2 fallback; 52,593 combined totals also exceed the 65536-entry lookup range.
These counts come from replay with instrumentation, divided by six identical
passes (five warmups plus one retained pass). Instrumented durations are excluded
from performance comparisons.

Real histogram populations were captured from Kodak17, Planter 1080p and Planter
4K. The replay compares hashes covering cluster counts, assignments, populations
and raw double cost bits. A separate direct-distance oracle copies the frozen
baseline expression and tests 100,000 deterministic histogram triples: 69,539
fully valid triples and 30,461 with overflowing combined totals. It exercises
empty/dense/sparse arrays, all alphabet extents, table-boundary counts, values near
2^53, and uint64 limits. All three experimental variants match exact distance bits
and success/failure status.

Existing permanent entropy tests additionally reject malformed values in every
one of 256 bins, including bins beyond a declared maximum or in an allegedly
empty population. Borrowed and mapped population failures preserve models and
costs. No new API or runtime diagnostic hook is introduced.

## Screening and rejected variant

Five alternating independent-process replay pairs per input, five warmups and
51 measured clustering calls per process. Input loading and result hashing are
outside the clustering timer. Changes are median paired ratios; milliseconds
are medians of process medians. Negative means lower time.

| Variant | Kodak17 | Planter 1080p | Planter 4K |
| --- | ---: | ---: | ---: |
| Static error return only | -7.28% | -8.85% | -7.52% |
| Plus redundant-check removal | -26.04% | -23.65% | -21.73% |
| Plus paired independent distances | -19.90% | -24.92% | -20.90% |

The selected scalar candidate takes 1.098 → 0.821 ms on Kodak, 5.870 → 4.426 ms on
Planter 1080p, and 4.687 → 3.670 ms on Planter 4K in this replay. The HD case has
more distance/bin work than the 4K case; clustering work is content-dependent.
These are isolated CPU durations, not complete-encode latency savings.

The paired variant evaluates two candidate clusters in one loop, preserving two
independent ordered cost chains and applying tie comparisons in original order.
It preserves replay output but offers no consistent benefit over the simpler
scalar candidate, especially on Kodak. It is excluded from delivered source.
No explicit NEON kernel or approximate logarithm is retained.

Three-pair full-workflow screens (three warmups, seven timed samples) showed:

| Variant | Kodak whole / serializer | Planter 1080p whole / serializer | Planter 4K whole / serializer |
| --- | ---: | ---: | ---: |
| Static error return | -1.91% / -3.70% | -0.27% / -3.56% | -0.54% / -2.41% |
| Scalar candidate | +5.60% / -1.85% | -1.74% / -7.14% | -1.63% / -5.31% |
| Paired variant | -2.34% / -0.20% | -0.47% / -3.99% | -0.88% / -2.75% |

These screens establish a reason to confirm the scalar candidate; they do not
resolve noisy whole-call effects, particularly Kodak. The longer confirmation
and batch-throughput comparison are separate cohorts.

## Longer complete-workflow confirmation

Apple M4 Pro, 48 GiB, macOS 15.6, AppleClang 17 Release; fully-resident Metal,
effort 7, distance 1.2, automatic CPU threads, fused-tuned inverse, no GPU-stage
profiling. Seven alternating independent-process pairs per workload, five warmups
and eleven timed encodes per process: 924 measured encodes. The public workflow
boundary excludes initialization, file I/O and process startup.

Negative changes mean lower latency. Percentages are medians of paired process-
median ratios; dividing displayed medians need not reproduce them. Entropy and
serializer columns are stage wall times, not aggregate worker durations.

| Workload | Whole ms, baseline → candidate | Whole change | Whole faster pairs | Serializer change | Entropy optimization change |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic_128x96 | 8.585 → 9.724 | +9.16% | 3/7 | +0.94% | -2.06% |
| padded_1080p | 63.398 → 62.464 | -1.23% | 6/7 | -6.42% | -14.62% |
| padded_4k | 207.822 → 208.269 | -1.14% | 4/7 | -4.48% | -14.23% |
| kodak-kodim17 | 20.216 → 20.086 | -0.69% | 5/7 | -1.89% | -8.05% |
| imazen26-1029-planter-1080p | 72.650 → 71.950 | -0.96% | 4/7 | -3.84% | -16.22% |
| imazen26-1029-planter-4k | 216.356 → 224.645 | +0.14% | 3/7 | -2.70% | -13.03% |

Entropy optimization improves in every pair for each of the five non-tiny
workloads. Padded 4K entropy time is 7.308 → 6.230 ms; Planter 4K is 7.684 →
6.564 ms. Complete-call effects are smaller and mixed: padded 4K pair changes
range from -4.47% to +1.13%; Planter 4K from -3.42% to +5.61%. Planter 4K
whole-call improvement is not established in this cohort. The Kodak short-screen
regression does not repeat. Tiny-image results require the separate controls.

## Batch throughput and tiny-image controls

Seven alternating independent-process pairs, three warmups and five measured
samples per input/batch size. The existing benchmark times preloaded linear RGB
through complete in-memory output, including scheduling and complete workflow;
initialization, warmups and correctness comparisons are outside the boundary.
Every output matches its process's single-image reference. Cross-build bytes
are covered by the separate corpus/policy and resource qualification.

Positive changes mean higher throughput. Paired changes are medians of within-
pair ratios; displayed images/s use medians of process-median batch times, so
dividing the displayed rates need not reproduce the paired change.

| Input | Batch size | Images/s, baseline → candidate | Paired change | Pair range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| kodak-kodim17 | 2 | 58.010 → 61.115 | +4.35% | -1.43% to +11.04% | 5/7 |
| kodak-kodim17 | 4 | 87.716 → 87.401 | +0.75% | -3.37% to +4.83% | 5/7 |
| imazen26-1029-planter-1080p | 2 | 18.580 → 19.108 | +2.36% | -13.19% to +8.54% | 6/7 |
| imazen26-1029-planter-1080p | 4 | 21.969 → 21.822 | +0.57% | -10.24% to +17.02% | 4/7 |
| imazen26-1029-planter-4k | 2 | 5.138 → 5.342 | +0.12% | -1.99% to +12.21% | 4/7 |
| imazen26-1029-planter-4k | 4 | 5.528 → 5.603 | -0.83% | -5.95% to +13.00% | 3/7 |

Two-image Kodak and Planter 1080p results favor the candidate, but four-image
HD/Kodak and both 4K results remain small or inconclusive. In particular, the
four-image 4K median is slightly negative and only three pairs improve. This
study does not demonstrate a general throughput gain or a 4K batch improvement.
The retention case is the consistent entropy-stage reduction with a small,
exact implementation change and no new storage or encoding policy.

The tiny-image follow-up uses seven pairs, ten warmups and 31 samples. Its median
whole-call change is -2.26% (range -8.97% to +24.47%, four faster pairs), reversing
the original +9.16% result. A matching identical-baseline A/A control gives -3.11%
(range -12.07% to +15.56%). Serializer changes are -2.37% versus -3.17% for A/A.
These results do not establish a reliable tiny-image gain or regression.

## Correctness qualification

The final Release suite passes **128/129**, matching the recorded baseline's
failure inventory. Both report `quantization_pipeline` index 1 as
`0.24919039011001587` instead of `0.24914586544036865`. No golden or tolerance
changed. All four ASan/UBSan suites pass without suppressions (leak detection off):
entropy, entropy storage planning, serializer storage planning and serializer
storage. These include exhaustive allocation-failure tests.

All **56/56** corpus/policy outputs are byte-identical. All **22** pinned decoder
fixtures pass. Eight resident-resource scenarios pass, with 24 retained matching
output/decode pairs, including tight/full admission, concurrent callers and
release/shutdown. For the one-image 4K case, peak backing is unchanged at
2,602,024,508 bytes and peak committed allowance at 9,301,538,408 bytes. The full
concurrent cohort's backing peak differs by +12,288 bytes; all other scenario
backing peaks and all committed peaks match. No physical-memory saving is claimed.

## Evidence and exclusions

Local artifacts are in `build/entropy-clustering`: frozen sources and builds,
probe compile/link commands, captured histogram fixtures, oracle results,
assembly, a whole-workflow sample profile, raw replay/workflow/batch samples,
qualification commands and logs, and an independent summary audit.

The baseline binary is the previously qualified frozen producer build under
`build/entropy-bit-writing/producer`; its runtime source matches `f719aa9`.
Experimental probes replace only the ANS translation unit and link the same
baseline library objects. The final candidate is rebuilt from its own frozen
source tree for qualification and confirmation.

An initial trusted/paired replay overlapped correctness-oracle subprocesses and
is excluded in full (`replay-excluded`). The replacement clean cohort supplies
the screening table above. A workflow-screen attempt overlapped a separate GJXL benchmark
and was rejected by the quiet-process guard (`screen-trusted-excluded`). Its
incomplete raw files remain excluded. A batch attempt also overlapped unrelated
compilation; its first-process data is excluded (`batch-throughput-excluded`).
On resumption, another compilation interrupted pair 4: both sides of that pair
were excluded and retried, while completed pairs 0–3 were retained. No other
user's job was stopped.

Timing checks detect competing encoder/build/test processes before and after each
workflow process. They do not provide exclusive-machine or controlled thermal/
power conditions. Profiling and counters are diagnostic; only uninstrumented
complete-workflow and batch measurements support throughput conclusions.

## Reproduction and next boundary

```sh
python3 build/entropy-clustering/audit.py
python3 build/entropy-clustering/qualify.py
python3 build/entropy-clustering/confirm.py
python3 build/entropy-clustering/batch-confirm.py
python3 build/entropy-clustering/tiny-controls.py
```

The audit checks source/runtime identities, raw timing and command-log hashes,
recomputes paired summaries, checks exact replay/oracle agreement, and reconciles
test, parity and resource inventories. Frozen probe sources, binaries and input
fixtures remain available for replay. `final-evidence-sha256.json` identifies the
retained measurement records and controllers. Incomplete timing attempts remain
excluded; completed paired records can be resumed with their original identity.
Do not rebuild the frozen baseline against candidate source.

The next profiling target remains DC histogram population preparation. Explicit
NEON and approximate histogram arithmetic were not implemented in this study;
the paired scalar probe does not prove that every SIMD design would lose. A new
experiment should demonstrate additional complete-call or batch value against
this qualified baseline before adding that complexity.
