# Shared-load short-filter integration followup

> Historical study checkpoint. The final disposition and commit status are in [REPORT.md](REPORT.md) and [README.md](README.md). Any pending work below refers to the time of that checkpoint.

Campaign 12 is complete. All three independent integrations pass seven focused Metal/resident contract tests under API/shader validation, and all stage/ordinary timing pairs preserve codestream bytes, summaries and submission counts. These variants have not yet passed the supported bundle's broader 56-case/decoder qualification.

Three alternating independent-process pairs, two warmups and three samples per process; eight CPU participants. Stage and ordinary timing are separate. All comparisons below use the already improved `integrated/fusion-bundle`, not original baseline.

| Variant | Target stage, 24 MP | Target stage, 48 MP | Ordinary encode, 24 MP | Ordinary encode, 48 MP |
|---|---:|---:|---:|---:|
| ultra-reuse | -10.232% | -2.339% | +0.949% (1/3) | -0.773% (3/3) |
| high-reuse | -50.658% | -37.576% | -2.907% (3/3) | -2.656% (3/3) |
| mask-reuse | -35.709% | -38.547% | -1.459% (3/3) | -0.007% (2/3) |

High/ultra target scopes sum both channel filters across reference and distorted scales. The mask target scope includes all reference mask stages and main/subscale mask/final stages. The main mask stage alone improves 51.0% and 53.5%. Do not add these stage percentages or add independent ordinary percentages to the earlier bundle's gains.

The high-frequency integration has a clear stage gain and all six ordinary pairs improve, making it the strongest new candidate. One 24 MP pair is an outlying -7.6%; the paired median is -2.9%, and a larger independent cohort remains necessary. The ultra refinement is mixed across extents and is excluded from the proposed next bundle. Mask ordinary results are smaller and mixed at 48 MP; its large stage gain does not by itself qualify a whole-call improvement.

Campaign 13 is complete. The medium-B candidate passes all seven focused tests and exact paired outputs. Its GPU stage improves 51.178% at 24 MP and 52.825% at 48 MP, while ordinary medians are -1.286% (3/3 faster) and +0.341% (1/3 faster). This is a mixed ordinary result, not a qualified independent gain. Campaign 13 independently integrates the remaining 15-tap medium-B filter. Its low/medium producer writes raw medium-B to the otherwise dead `kImage+5` plane, then the direct blur writes final psycho slot 5. The plane is consumed before suppress/ultra, and before any later DC/L2 or raw mask reuse. This avoids an extra copy or arena allocation. Build, focused contracts, stage and ordinary measurements ran sequentially after campaign 12.

Campaign 14 checks a bounded geometry/reuse interaction around the new short-filter mechanism: 7/13/15 taps each use the current 16x64/four-output shape as a control, plus 8x64, 32x64, 16x32 and 32x32 four-output shapes and 16x64/16x128 eight-output shapes. It includes guarded parity before isolated timing at padded4K and 24 MP. It ran only after campaign 13 finished and is now complete. All 21 configurations pass 48 guarded bitwise cases each. The eight-output 13/15-tap and wider ultra leads proceed to independent integrations in campaign 17, described in [PHASE15-17.md](PHASE15-17.md). These tests address the new mechanism; earlier independent-output tile failures do not exclude it.

## Lifetime audit

- High X/Y: raw medium resides in future ultra slots 8/9; both high filters consume those values before ultra overwrites them. Final medium writes slots 3/4; pre-ultra high remains disjoint in `kImage+3/+4`.
- Medium B: raw medium uses `kImage+5`, with the psycho output stride (including packed reference subscale). Its consumer precedes suppress/ultra. This raw value is dead before mask production may reuse the same plane.
- Mask: reference raw/completed planes are disjoint. Resident packed AC/DC leaves `kDc+2` dead through Malta/final composition, so the raw distorted mask can live there until direct filtering into `kWork+4`. `packed_dc` and `defer_l2` agree at resident ordinary/profiled call sites. Legacy nonresident comparisons keep the original in-place two-pass path.
- `kWork+3` is the live invariant reference-eroded mask and is not reused.

Every profiling stage is emitted in the same dependency order as ordinary execution. Source reasoning is backed by existing storage, lifecycle, prepared-operation and AQ tests, but a combined candidate still needs independent validation. The primary checkout remains untouched. Optimization exhaustion is not established.

The enclosing resident AQ owner lends already-dead filtered-XYB and gathered-pixel scratch, retaining reconstructed linear RGB separately (`metal_aq_evaluation.cpp:1608-1637`). Slot 26 maps to the sixth filtered-XYB plane in the stable 11-slot borrowed mapping. Its added raw medium-B/mask use does not overwrite the reconstructed RGB input.
