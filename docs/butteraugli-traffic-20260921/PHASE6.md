# Butteraugli traffic study: phase 6 checkpoint

> Historical study checkpoint. The final disposition and commit status are in [REPORT.md](REPORT.md) and [README.md](README.md). Any pending work below refers to the time of that checkpoint.

The newer exact-output bundle demonstrates further achievable performance. It combines AC/DC composition fusion, two-axis low/medium load reuse (two adjacent outputs on each axis), and a direct 7-tap ultra filter. It does not include the larger Malta tile, addressing changes, or four-vertical-output refinements. No production source has been changed.

## Ordinary complete-call evidence

Three alternating independent-process pairs, two warmups and three samples each, eight CPU participants, same complete-call boundary and frozen original baseline as `PHASE5.md`. The reported changes are medians of paired percentage changes. Every pair and repeated call preserves codestream bytes, summary, and submission count.

| Setting | Median time change | Faster pairs |
|---|---:|---:|
| Campus 12 MP, e7, d1.9 | -6.114% | 3/3 |
| Alpine 24 MP, e7, d1.9 | -7.404% | 3/3 |
| Forest 48 MP, e10, d1.9 | -7.389% | 3/3 |
| Alpine 12 MP, e10, d0.8 | -5.005% | 3/3 |
| Forest 24 MP, e7, d0.8 | -6.981% | 3/3 |
| Campus 48 MP, e10, d0.8 | -5.845% | 3/3 |
| Kodak01, e7, d1.9 | -3.335% | 3/3 |
| Kodak17, e10, d0.8 | -3.080% | 3/3 |

Evidence: `fusion-bundle-broad-wall/summary.json`, with input/binary hashes, raw samples, paired output hashes and process observations in the same directory. This is a broad exploratory cohort. The selected final candidate still needs a larger fixed confirmation, same-binary controls and broader policy/decoder/lifecycle checks. Do not transfer the earlier AC/interior candidate's 56-case qualification to this new bundle.

## Attributable stage comparisons

Each row uses a separate three-pair production-aligned profile cohort and the stated direct parent. These numbers cannot be added to predict ordinary latency.

| Added mechanism / direct parent | Alpine24/e7 | Forest48/e10 | Interpretation |
|---|---:|---:|---|
| Vertical load reuse / AC+DC+horizontal filter | Low/medium -17.306% | Low/medium -17.284% | Useful additional stage gain; unchanged GPU stages approximately flat in paired median |
| Direct 7-tap ultra / preceding vertical candidate | Ultra -33.793% | Ultra -31.573% | Useful gain; complete ordinary bundle above confirms combined effect |
| Malta 32x16/two-output / preceding ultra bundle | Malta -3.711% | Malta -3.528% | Small additional stage opportunity; instrumented whole-call mixed (+0.797%, -0.514%) |
| Power-of-two reducer coordinates / AC+DC fusion | Reduction -0.070% | Reduction +0.406% | No consistent benefit; reject this added complexity |

The new packed-Malta geometry passed 1,920 direct guarded bitwise cases; addressing passed 360 guarded reduction cases. All four integrated candidates passed three focused Metal/AQ tests under API/shader validation. Full-suite success is not claimed.

## Filter frontier

Four larger vertical-reuse variants and 23 predeclared tile/reuse combinations passed 48 guarded cases each. Each timing screen includes an unchanged original-kernel control and five paired microtiming rounds at padded4K and 24MP. The best new geometry, 16x96 with four vertical outputs and two horizontal outputs per thread, improves isolated filter time 38.929% and 38.953% against original. The simpler 16x64/four-output version improves 37.133% and 38.034%. Eight vertical outputs are consistently worse than four within the corresponding shapes. Wider/narrower tiles did not beat the 16-column designs.

These results justify integrating two four-output finalists, not claiming their isolated percentage as an encoder speedup. The two-output bundle in the ordinary table predates these finalists. See `reuse-geometry-timing/summary.json`, `vertical-wide-timing/summary.json`, and `TRAFFIC-NOTES.md` for the request-count and resource tradeoffs.

## Remaining work

Phase 7 measures both four-output finalists incrementally and tests main-distance composition fused into the final mask-blur pass in both grid orientations. The latter adds no dispatch and is distinct from the rejected separate materialization experiment. It requires adding mask and reduction timings before interpreting the result. Both orientations pass 192 guarded producer cases, 360 reduction/error cases and the three focused integrated tests. A final small rolling-halo/launch-bound interaction screen is generated but uncompiled. Final confirmation and a remaining-hypothesis audit are still required; exhaustion is not established.
