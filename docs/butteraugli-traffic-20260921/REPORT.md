# Measured Butteraugli optimization headroom on M4 Pro

**The original GJXL implementation had substantial achievable headroom. This investigation is complete at the user-approved practical stopping point for exact-output Butteraugli optimization.** The strongest direct comparison with the original baseline confirms **12.70% lower warm encoding time at 24 MP/e7** and **10.71% at 48 MP/e10**, with all seven independent pairs faster at each setting. A final refinement adds approximately 0.3–1.2% in those two cases, with smaller, more variable results across the broader cohort.

This is an empirical stopping point for the tested implementation, hardware and workloads. It does **not** establish a theoretical hardware maximum or rule out sub-percent, workload-specific or future algorithmic gains. All planned campaigns and final controls are complete. The measured implementation and compact evidence are now committed on `perf/butteraugli-traffic-20260921`; the primary checkout and remote were not changed by this checkpoint. See [preservation and reproduction](README.md).

## Measurement boundary and provenance

The fixed baseline is `4f3e4147c8bde16b0b49523adbf10ddd8043585f`, the production-aligned profiling revision based on primary `b1a7373beff4c0370494c2f9584bcb1510491357`. Baseline shaders and serialization match that primary revision. Measurements use the local **M4 Pro, 20 GPU cores, 48 GB unified memory**, macOS 15.6, and eight CPU participants.

Ordinary timing covers the public encode call from loaded linear RGB through the returned codestream, including reference preparation, encoding and CPU serialization. Backend construction, input loading and destruction of the returned object are outside. The path is fully resident Metal, with diagnostic final scoring disabled. Results describe warm single-image calls, not cold startup or batched throughput. Profiling and ordinary timing use separate cohorts; validation layers are disabled during timing.

Confirmation cohorts use seven alternating independent-process pairs, three warmups and seven measured calls per process. Broad cohorts use three pairs, two warmups and three measured calls. Percentages below are **medians of paired latency changes**, not ratios of separately aggregated medians. Lower is better. Cohort identities retain sources, binaries, shader libraries, input hashes, commands and per-call evidence.

## Confirmed improvement over the original baseline

The `short-bundle` candidate combines:

- Shared adjacent loads in the 33-tap low/medium filter while preserving each output's arithmetic order and FP32 rounding points.
- DC/AC composition at its filter/Malta producers, reducing subsequent resident reduction traffic and arithmetic.
- A direct 7-tap ultra filter with its exact nonlinear epilogue.
- Shared-load 15-tap high/medium-B filters and a direct 13-tap mask path, with disjoint input/output lifetimes.

| Seven-pair ordinary confirmation | Latency change | Faster pairs |
|---|---:|---:|
| alpine24-e7 | -12.697% | 7/7 |
| forest48-e10 | -10.715% | 7/7 |
| kodak01-e7 | -5.121% | 6/7 |
| kodak17-e10-d08 | -5.468% | 7/7 |

The first two cases use distance 1.9; Kodak17 uses 0.8. An independent broad cohort covers six large settings, three photograph contents, 12/24/48 MP, efforts 7/10 and distances 0.8/1.9. **All six large settings improve 9.25–12.42%, with all 18 pairs faster.** The two Kodak controls improve 2.21% (2/3 pairs) and 3.99% (3/3). These are fixed nominal settings with identical tested codestreams, so the observed gain does not trade away tested rate or quality.

Fresh unchanged-binary controls have median changes of -0.471% at 24 MP and +0.364% at 48 MP. Individual pairs span -1.329% to +0.004% and -0.486% to +1.198%, respectively. These diagnose variability; they are not corrections to subtract. The main bundle's gain is substantially larger and repeats in every large-image confirmation pair.

Evidence: [confirmation](short-bundle-wall-confirm/summary.json), [broad cohort](short-bundle-broad-wall/summary.json), [controls](control-wall-short-final/summary.json), [frozen identity](integrated/short-bundle/identity.json), [source patch](integrated/short-bundle/source.patch).

## Separate GPU attribution

Three-pair production-aligned profiles compare `short-bundle` directly with original baseline. Stage totals sum valid, nonoverlapping invocation intervals before aggregation. Ordinary timing above supplies the complete-call claims.

| GPU scope | 24 MP/e7 | 48 MP/e10 |
|---|---:|---:|
| Total Butteraugli, including reference features | **-29.513%** | **-27.598%** |
| All timed GPU work | -16.687% | -16.312% |
| Low/medium | -34.227% | -32.733% |
| High-frequency | -51.995% | -37.537% |
| Medium B | -50.472% | -53.636% |
| Ultra | -34.983% | -31.451% |
| Main mask | -51.168% | -53.515% |
| Resident reduction | -60.226% | -60.748% |
| Malta, including moved AC work | **+9.178%** | **+8.704%** |

The increased Malta work is charged in the improved total. Scopes overlap; these percentages must not be added. The slower producer/faster consumer pair illustrates why an isolated reduction timer would overstate the gain. See [GPU attribution](short-bundle-original-stage/summary.json).

An independent seven-pair ablation compares the bundle directly with the high-only candidate. Adding medium-B and mask filtering together saves another **2.104% / 2.299%** at 24/48 MP, with all seven pairs favorable in each case. Their combined value therefore has complete-call evidence despite mixed earlier individual screens. See [ablation](short-bundle-high-increment-wall-confirm/summary.json).

## Final small refinement and practical stopping point

`short-final` increases vertical/horizontal load reuse for the 15-tap filters, vertical reuse for the 13-tap mask, and uses the wider 32x32 ultra layout with four horizontal outputs. It preserves the bundle's storage routing, arithmetic order and precision policy. The isolated worktree now contains this independently tested final prototype.

The following measurements compare **directly with `short-bundle`**, not original baseline:

| Seven-pair ordinary confirmation | Additional latency change | Faster pairs |
|---|---:|---:|
| alpine24-e7 | -1.220% | 7/7 |
| forest48-e10 | -0.296% | 5/7 |
| kodak01-e7 | -0.377% | 4/7 |
| kodak17-e10-d08 | +0.057% | 3/7 |

Independent broad direct-parent coverage is also complete:

| Broad ordinary cohort | Additional latency change | Faster pairs |
|---|---:|---:|
| alpine24-e7 | -0.905% | 2/3 |
| forest48-e10 | -0.260% | 2/3 |
| campus12-e7 | -0.245% | 2/3 |
| alpine12-e10-d08 | -0.920% | 2/3 |
| forest24-e7-d08 | -0.877% | 3/3 |
| campus48-e10-d08 | -1.347% | 3/3 |
| kodak01-e7 | -0.364% | 2/3 |
| kodak17-e10-d08 | +0.427% | 1/3 |

All six large-image medians are favorable by 0.24–1.35%, but many individual pairs are not faster. Small-image effects are mixed. Matching same-binary controls are +0.284% and +0.209% at 24/48 MP, with individual ranges of -0.797% to +0.961% and -0.968% to +0.718%. The 24 MP confirmation is the clearest incremental result; the smaller effects cannot be treated as equally robust production-wide gains.

Separate stage profiling reduces total Butteraugli another 3.364% / 2.156% and all timed GPU work 1.653% / 1.144%. These stage results support the mechanism but do not override ordinary variability. **No original-baseline total is synthesized by multiplying results from separate cohorts.** The 12.70% / 10.71% headline remains the directly measured `short-bundle` result.

With the user's explicit choice to use a practical stopping point, the study closes here. Substantial positive leads in the inventory have been integrated, rejected using complete affected scopes, or bounded by evidence. Remaining tuning effects are around one percent or less in most measured cases, with exceptions near 1.3%, and depend on workload or measurement variability. Pursuing those is a separate diminishing-return study, not required for this conclusion.

Evidence: [final confirmation](short-final-increment-wall-confirm/summary.json), [final broad cohort](short-final-increment-broad-wall/summary.json), [matching controls](control-wall-short-increment/summary.json), [final stage attribution](short-final-increment-stage/summary.json), [frozen final identity](integrated/short-final/identity.json), [final source patch](integrated/short-final/source.patch).

## Correctness, artifact audit and preserved implementation

Both `short-bundle` and `short-final` independently pass seven focused/extended Metal and resident tests under API/shader validation, **56 canonical/policy codestream byte comparisons**, and **three independently decoded pairs with identical finite pixels**, checked for dimensions and payload. Direct mechanism guards also cover odd/tiny extents, strides, exceptional values, padding and reduction error behavior. Existing tolerances were not widened.

- Main bundle: [byte/decoder summary](short-bundle-canonical-parity/summary.json), [finite-pixel checks](short-bundle-canonical-parity/finite-pixels.json).
- Final prototype: [byte/decoder summary](short-final-canonical-parity/summary.json), [finite-pixel checks](short-final-canonical-parity/finite-pixels.json).
- Independent saved-artifact audit: **23 late-study cohorts, 256 pairs and 3,356 file hashes verified**, including frozen identities, recorded byte/submission equality, ordinary medians, paired ratios and summary aggregates. The collection drivers separately validate GPU timestamps. This is an audit, not a remeasurement. See [audit record](short-artifact-audit.json).
- Restoring the final prototype re-passed the seven tests and reproduced the frozen measured shader library byte for byte. See [restoration identity](restored-qualified/identity.json).

The exact measured main bundle is commit `db8a4d67d554bcb04b651243c3660c42ec245bb8`, followed by the exact final refinement in `ef5f29003c52ec788d339f00b6f2e50a6f1818e7`. They change only `butteraugli.metal`, `metal_backend_internal.h` and `metal_butteraugli.cpp` in the isolated worktree. The primary checkout and its pre-existing dirty work are preserved. The primary checkout advanced to `3b0de6d9d47846f6aa13f33ad6cca7e24e52ada6` after measurement; this checkpoint retains the original measured base rather than implying requalification on the new revision. This is focused M4 Pro validation, not universal correctness, a full-suite success claim, or qualification on other GPUs. The inherited optional C API build warning failure and CPU quantization-pipeline golden mismatch remain documented in the phase evidence.

The storage audit explicitly preserves live reference-mask slot 30 and the reconstructed linear RGB input. New direct filters use disjoint producer/consumer storage, avoiding cross-threadgroup halo races. See [lifetime analysis](PHASE12-14.md) and [traffic accounting](TRAFFIC-NOTES.md).

## Why this is a practical plateau, not a hardware ceiling

The investigation covers 186 configuration entries across 15 guarded kernel-screen families, including repeated/current-shape controls; these are not 186 independent algorithms. It also tests integrated dataflow and reduction alternatives. Taller/rolling tiles, further 33-tap reuse, Malta geometry, Opsin geometry and transpose geometry lack consistent broad incremental ordinary gains. Separate distance-map materialization raises reduction time 26–42%; mask-producer composition raises the combined mask-plus-reduction scope 6–9%. Exact power-of-two addressing is effectively flat. Optimistic nonlinear reference-mask caching has a consumer-only benefit below 0.9% before preparation/storage costs. These measured alternatives support stopping, rather than merely assuming current kernels are optimal.

The short-filter results also show why negative results are mechanism-specific: early direct 13/15-tap layouts regressed, but adjacent-load reuse later produced the substantial gains above. Future transformations remain possible. See the final [hypothesis audit](HYPOTHESIS-AUDIT.md), [opportunity inventory](OPPORTUNITIES.md), and [complete saved-results tables](PROGRESS.md).

Desktop background activity was observed during parts of the study, including suggestion, media-analysis and indexing processes. Snapshots do not establish which calls were affected. All results were retained, no services were changed, and fresh controls quantify some of the timing variability. The environment was not claimed to be fully isolated.

The original [hardware/profile investigation](../performance-headroom-20260921/REPORT.md) contains the saved-stage audit, Metal System Trace captures, counter caveats and fixed-rest sensitivity estimates. Occupancy is not useful-work efficiency; source load counts are not measured DRAM traffic; peak advertised bandwidth is not an encoder throughput bound. A theoretical hardware-efficiency percentage remains unavailable. CPU serialization, AC strategy and broader CPU/GPU scheduling remain separate potential optimization subjects outside this Butteraugli closure. The constructive, output-equivalent gains here establish that the original encoder had headroom; they do not establish that the final encoder reaches the hardware maximum.
