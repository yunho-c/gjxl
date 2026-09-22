# Butteraugli traffic study: phase 5 checkpoint

> Historical study checkpoint. The final disposition and commit status are in [REPORT.md](REPORT.md) and [README.md](README.md). Any pending work below refers to the time of that checkpoint.

The investigation has demonstrated achievable headroom on the M4 Pro. It has not established optimization exhaustion. The primary checkout remains unchanged; all source prototypes are in the detached study worktree. This checkpoint supersedes pending statements in `PHASE4.md` for experiments listed below.

## Ordinary complete-call results

Candidate `integrated/ac-interior` combines branch-free adjacent horizontal loads, main/subscale DC composition at its producer, and AC composition at the final Malta pass. Frozen original baseline: `4f3e414`, with shaders identical to primary `b1a7373`. Eight CPU participants; loaded linear RGB through returned codestream, including CPU serialization. These are three alternating independent-process pairs per setting, two warmups and three measured samples each. Percentages are medians of paired changes, not ratios of the displayed independent latency medians. They are an exploratory broad cohort, distinct from the seven-pair confirmations of earlier candidates.

| Setting | Median time change | Faster pairs |
|---|---:|---:|
| Campus 12 MP, e7, d1.9 | -4.845% | 3/3 |
| Alpine 24 MP, e7, d1.9 | -4.987% | 3/3 |
| Forest 48 MP, e10, d1.9 | -3.261% | 3/3 |
| Alpine 12 MP, e10, d0.8 | -3.362% | 3/3 |
| Forest 24 MP, e7, d0.8 | -4.260% | 3/3 |
| Campus 48 MP, e10, d0.8 | -4.208% | 3/3 |
| Kodak01, e7, d1.9 | **+1.640%** | 1/3 |
| Kodak17, e10, d0.8 | -1.352% | 2/3 |

All paired codestreams are byte-identical; repeated calls also preserve bytes, summaries and submission counts. The six large cases give consistent evidence of a gain. The small cases do not support a universal gain. An independent production-aligned profile cohort reports approximately 9.9% and 10.9% less total Butteraugli GPU time for Alpine24/e7 and Forest48/e10. Ordinary timing remains authoritative for whole-call claims. See `ac-interior-broad-wall/summary.json` and `ac-interior-stage/summary.json`.

## Correctness scope

- Seven Metal/resident contract tests pass under API/shader validation. This is a focused suite, not a full-suite green result.
- Fifty-six canonical/policy encoder comparisons pass exact byte parity; three independent decoded pairs have identical finite pixels with the pinned decoder.
- AC fusion passes 1,920 direct guarded Malta comparisons and 360 guarded reduction comparisons, including nontrivial strides, edges, exceptional values, distinct reference/distorted features, error flags and asymmetry factors.
- The optional workflow-admission target could not build because an unchanged C API profiling header triggers inherited unused-parameter errors with `-Werror`. The failure and byte comparison to baseline are retained in `phase5-prepare/inherited-build-failure.json`; no warnings or tolerances were weakened.

See `phase5-qualification-checkpoint.json`, `ac-interior-canonical-parity/summary.json`, and its `finite-pixels.json`.

## Further mechanisms tested

| Mechanism | Evidence | Disposition |
|---|---|---|
| Share adjacent vertical 33-tap input loads as well as horizontal loads | Four exact variants; best isolates at -32.9% padded4K and -34.8% 24MP, 5/5 each, versus original filter | Integrate and test geometry interaction |
| Direct 7-tap ultra filter with nonlinear epilogue | Disjoint-input integration passes focused tests and paired byte equality; ultra stage -35.3% / -31.1% at 24/48MP | Combine incrementally; ordinary confirmation pending |
| Malta 32x16 tile, two outputs per thread | Focused parity and exact-byte integrated comparisons; Malta stage -3.5% / -5.3% at 24/48MP | Check interaction with AC fusion |
| Eight adjacent-output Malta refinements | All pass guarded parity; best about -3.1% / -3.5% isolated against original | No clear additional large gain over larger Malta tile; cohorts are separate |
| Cache nonlinear reference-mask curves | Optimistic consumer-only screens: <0.9% of filter/reduction time before cache construction/storage; guards pass | Low priority. Applying these ratios to the saved stage budget gives a fixed-rest estimate around 0.1% of complete-call time, not a theoretical upper bound |
| Materialize composed pixel distance before block reduction | Exact bytes and focused tests pass, but charged reduction scope increases **42.1% / 26.5%** at 24/48MP | Reject this implementation for covered large workloads; retain fused reduction |

Stage improvements in this table are three-pair diagnostic results. Do not add them or equate them with ordinary encoder gains. Source load counts describe logical requests, not external-memory traffic.

## Remaining bounded work

Phase 6 separately measures vertical reuse over the earlier AC/filter candidate, ultra integration over that result, larger Malta geometry over that bundle, and power-of-two reduction addressing over AC fusion. It also screens four larger vertical-reuse variants and 23 predeclared tile/reuse combinations with guards and original-binary controls, then measures a broad ordinary cohort. These hypotheses remain plausible and prevent an exhaustion conclusion. The study does not change arithmetic, AQ policy, precision or output requirements.
