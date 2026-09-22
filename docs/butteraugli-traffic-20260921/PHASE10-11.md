# Geometry followups and a new short-filter opportunity

> Historical study checkpoint. The final disposition and commit status are in [REPORT.md](REPORT.md) and [README.md](README.md). Any pending work below refers to the time of that checkpoint.

The geometry-only followups do not establish a consistent ordinary gain over the supported bundle. However, the subsequent adjacent-load short-filter experiment finds substantial new isolated headroom. **Optimization exhaustion is not established; the positive short-filter candidates require integration.**

## Geometry-only results

Three alternating independent-process pairs, with the supported `fusion-bundle` as the direct parent. Stage and ordinary timing are separate cohorts.

| Candidate | 24 MP targeted stage | 48 MP targeted stage | 24 MP ordinary call | 48 MP ordinary call |
|---|---:|---:|---:|---:|
| 32x8 Opsin tile | Opsin +2.185% | Opsin -9.255% | -0.768% (3/3 faster) | +0.066% (1/3) |
| 16x16 13-tap transpose | Mask scopes -1.438% | Mask scopes -7.087% | -0.025% (2/3) | -0.265% (2/3) |

The 24 MP Opsin profile also contains changes in unmodified stages, so its small targeted effect should not be interpreted in isolation. These ordinary differences are comparable to the variation seen in the unchanged-binary controls. Neither geometry change is added to the supported bundle on this evidence. Focused validation and paired byte equality pass.

## Exact shared-load short filters

The earlier short-filter screen tested direct tiles with independent convolution outputs. The successful 33-tap mechanism suggested a concrete repair: share loads across two horizontal and two/four vertical outputs, use branch-free interior loops, and retain each output's ascending tap order and FP32 horizontal materialization.

Ten configurations cover 7/13/15 taps, 16x64 output tiles, and 16x128 alternatives for 13/15 taps. All pass 48 guarded bitwise cases each, with odd/tiny extents, nontrivial strides, constant/impulse/large/Inf/NaN patterns, input/padding comparisons and buffer guards. Five-pair microtiming includes unchanged original two-pass controls (-0.18% padded4K, +0.51% 24MP).

| 16x64, four vertical outputs, two horizontal outputs | Padded4K change | 24 MP change |
|---|---:|---:|
| 7 taps | -56.34% | -57.03% |
| 13 taps | -50.19% | -53.57% |
| 15 taps | -50.27% | -52.68% |

All rows are faster in 5/5 pairs at both extents. The 16x128 shapes do not offer a consistent advantage over 16x64 across both extents. These are isolated pure-blur timings versus the original two-pass implementation, **not additional complete-encoder speedups**. The production high/ultra stages include nonlinear epilogues, and mask/medium stages can alias input/output; integration must charge their full work and preserve data dependencies.

Evidence: `short-reuse-parity/summary.json`, `short-reuse-timing/summary.json`, frozen sources and commands under `phase11-short-reuse/`.

## Independent integrations under test

Campaign 12 builds, validates and measures three variants against the supported bundle:

- `ultra-reuse`: replace the direct 7-tap blur core while retaining its exact nonlinear/mask epilogue and already-disjoint pre-ultra inputs.
- `high-reuse`: fuse the 15-tap high-frequency blur and its exact epilogue. Because the original epilogue updates medium in place, raw medium X/Y is routed through the still-dead future ultra slots 8/9; final medium and pre-ultra high are written separately. Medium-B retains its existing path in this first experiment.
- `mask-reuse`: use the direct 13-tap blur for disjoint reference masks. For resident distorted masks, the packed AC/DC representation leaves `kDc+2` dead through Malta/final composition, so raw activity is emitted there and blurred into the ordinary `kWork+4` consumer plane. The legacy in-place path retains its original two-pass implementation.

These routes add no arena allocation or copy. They are source-backed lifetime hypotheses until the broader operation, resident storage, cache, paired-output and timing checks finish. All compilation and validation complete before timing starts. The main checkout is unchanged; the previously confirmed bundle remains the supported candidate until new ordinary evidence and correctness gates justify an update.
