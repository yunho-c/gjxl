# Combined short-filter qualification, campaigns 12–20

Complete at the user-approved practical stopping point. Both combined candidates have independent correctness, broad and seven-pair ordinary results; controls, ablation and final refinement coverage are complete. No primary-checkout source has changed.

`short-bundle` adds the shared-load 15-tap high and medium-B filters and the disjoint 13-tap mask path. The mixed 16x64 ultra refinement is excluded. Its frozen source, binary and shader identities are under `integrated/short-bundle/`. All timing percentages below are direct paired changes against each cohort's declared parent; effects from different cohorts must not be added.

## Execution status

| Campaign | Status | Current step |
|---|---|---|
| 12 | complete |  |
| 13 | complete |  |
| 14 | complete |  |
| 15 | complete |  |
| 16 | complete |  |
| 17 | complete |  |
| 18 | complete |  |
| 19 | complete |  |
| 20 | complete |  |

## Correctness

The new combined candidate passes 56 canonical/policy byte comparisons and 3 independently decoded pairs with identical finite pixels. The build's three focused tests and four extended Metal/resident tests pass under API/shader validation. The rebuilt library is byte-identical to the frozen measured library. This is focused qualification, not a full-suite success claim.

## Ordinary complete-call cohorts

Public encoding from loaded linear RGB through returned codestream, including CPU serialization. Backend creation and input loading are outside. Eight CPU participants; validation layers disabled for timing. Initial cohorts use three alternating independent-process pairs with two warmups/three samples. Confirmation cohorts use seven pairs with three warmups/seven samples.

| Cohort / input | Direct baseline | Change | Faster pairs |
|---|---|---:|---:|
| short-bundle-increment-wall / alpine24-e7 | fusion-bundle | -4.795% | 3/3 |
| short-bundle-increment-wall / forest48-e10 | fusion-bundle | -3.629% | 2/3 |
| short-bundle-broad-wall / alpine24-e7 | baseline-v2 | -12.424% | 3/3 |
| short-bundle-broad-wall / forest48-e10 | baseline-v2 | -10.948% | 3/3 |
| short-bundle-broad-wall / campus12-e7 | baseline-v2 | -12.276% | 3/3 |
| short-bundle-broad-wall / alpine12-e10-d08 | baseline-v2 | -9.521% | 3/3 |
| short-bundle-broad-wall / forest24-e7-d08 | baseline-v2 | -11.527% | 3/3 |
| short-bundle-broad-wall / campus48-e10-d08 | baseline-v2 | -9.253% | 3/3 |
| short-bundle-broad-wall / kodak01-e7 | baseline-v2 | -2.214% | 2/3 |
| short-bundle-broad-wall / kodak17-e10-d08 | baseline-v2 | -3.995% | 3/3 |
| short-bundle-wall-confirm / alpine24-e7 | baseline-v2 | -12.697% | 7/7 |
| short-bundle-wall-confirm / forest48-e10 | baseline-v2 | -10.715% | 7/7 |
| short-bundle-wall-confirm / kodak01-e7 | baseline-v2 | -5.121% | 6/7 |
| short-bundle-wall-confirm / kodak17-e10-d08 | baseline-v2 | -5.468% | 7/7 |
| control-wall-short-final / alpine24-e7 | baseline-v2 | -0.471% | 6/7 |
| control-wall-short-final / forest48-e10 | baseline-v2 | +0.364% | 1/7 |
| short-bundle-high-increment-wall-confirm / alpine24-e7 | high-reuse | -2.104% | 7/7 |
| short-bundle-high-increment-wall-confirm / forest48-e10 | high-reuse | -2.299% | 7/7 |
| short-bundle-high-increment-wall-confirm / kodak01-e7 | high-reuse | -0.008% | 4/7 |
| short-bundle-high-increment-wall-confirm / kodak17-e10-d08 | high-reuse | -1.123% | 6/7 |
| short-eight-increment-wall / alpine24-e7 | short-bundle | +1.593% | 1/3 |
| short-eight-increment-wall / forest48-e10 | short-bundle | -1.023% | 2/3 |
| short-wide-ultra-increment-wall / alpine24-e7 | short-bundle | -0.206% | 3/3 |
| short-wide-ultra-increment-wall / forest48-e10 | short-bundle | -0.602% | 2/3 |
| short-final-increment-wall-confirm / alpine24-e7 | short-bundle | -1.220% | 7/7 |
| short-final-increment-wall-confirm / forest48-e10 | short-bundle | -0.296% | 5/7 |
| short-final-increment-wall-confirm / kodak01-e7 | short-bundle | -0.377% | 4/7 |
| short-final-increment-wall-confirm / kodak17-e10-d08 | short-bundle | +0.057% | 3/7 |
| control-wall-short-increment / alpine24-e7 | short-bundle | +0.284% | 1/7 |
| control-wall-short-increment / forest48-e10 | short-bundle | +0.209% | 3/7 |
| short-final-increment-broad-wall / alpine24-e7 | short-bundle | -0.905% | 2/3 |
| short-final-increment-broad-wall / forest48-e10 | short-bundle | -0.260% | 2/3 |
| short-final-increment-broad-wall / campus12-e7 | short-bundle | -0.245% | 2/3 |
| short-final-increment-broad-wall / alpine12-e10-d08 | short-bundle | -0.920% | 2/3 |
| short-final-increment-broad-wall / forest24-e7-d08 | short-bundle | -0.877% | 3/3 |
| short-final-increment-broad-wall / campus48-e10-d08 | short-bundle | -1.347% | 3/3 |
| short-final-increment-broad-wall / kodak01-e7 | short-bundle | -0.364% | 2/3 |
| short-final-increment-broad-wall / kodak17-e10-d08 | short-bundle | +0.427% | 1/3 |

## Separate GPU attribution

| Cohort / input | Total Butteraugli | High | Medium B | Mask scopes | Ultra |
|---|---:|---:|---:|---:|---:|
| short-bundle-increment-stage / alpine24-e7 | -14.046% | -51.201% | -50.281% | -37.257% | +0.153% |
| short-bundle-increment-stage / forest48-e10 | -12.039% | -36.860% | -52.741% | -39.795% | -0.976% |
| short-bundle-original-stage / alpine24-e7 | -29.513% | -51.995% | -50.472% | -48.160% | -34.983% |
| short-bundle-original-stage / forest48-e10 | -27.598% | -37.537% | -53.636% | -50.249% | -31.451% |
| short-eight-increment-stage / alpine24-e7 | -0.155% | +1.636% | -8.440% | -3.078% | -0.502% |
| short-eight-increment-stage / forest48-e10 | -0.253% | -2.708% | -6.140% | -2.423% | -0.230% |
| short-wide-ultra-increment-stage / alpine24-e7 | -0.843% | +1.246% | -3.465% | +0.922% | -8.501% |
| short-wide-ultra-increment-stage / forest48-e10 | -2.323% | +0.616% | +5.779% | +3.428% | -14.183% |
| short-final-increment-stage / alpine24-e7 | -3.364% | -3.267% | -12.802% | -3.986% | -11.280% |
| short-final-increment-stage / forest48-e10 | -2.156% | -0.967% | -7.415% | -1.510% | -13.576% |

## Bounded interaction followup

Campaign 14 checks 21 tile/reuse combinations, including three current-shape controls. Every configuration passes 48 guarded bitwise cases. The 15-tap eight-output variants reduce isolated two-pass blur time roughly 53-56%, compared with about 50-53% for the current four-output shape. The 13-tap increment is smaller. Wider 32x32/four-output ultra has a further isolated advantage, but its nonlinear epilogue must be included before a gain is credited.

Campaign 16 compares the new bundle directly with high-only over seven pairs, so the stronger high-frequency improvement cannot hide unhelpful medium-B/mask additions. Campaign 17 independently integrates the 16x64/eight-output 13/15-tap filters and the 32x32/four-output ultra filter, validates them, and measures each against `short-bundle`. These integrated results are complete; their ordinary effects are mixed or small and are listed below.

The smaller 16x64 eight-output shape is used because 16x128 doubles tile storage and shows no consistent advantage across both 13/15-tap radii and extents. Comparisons between separate micro-cohorts are diagnostic; their control variation and production epilogues preclude a precise encoder prediction.

Campaign 18 closes one remaining degree of freedom in the newly successful short filters: four/eight horizontal outputs versus the existing two. Eighteen configurations include six two-output controls, both leading ultra shapes, and four/eight vertical outputs for the 13/15-tap filters. It completed guarded parity and isolated timing sequentially after campaign 17. Three-channel 33-tap experiments did not favor four horizontal outputs, but the shorter one-channel kernels have different register pressure, so those earlier results do not exclude this case.

The phase 17 ordinary results are mixed or small: eight vertical outputs are +1.593% at 24 MP and -1.023% at 48 MP; wide ultra is -0.206% and -0.602%. A read-only snapshot records active suggestd, media analysis and indexing processes, without proving which calls they affected. No system service was changed. Campaign 18 passes all 18 x 48 guarded cases; four horizontal outputs modestly improve 15-tap filtering, while eight usually regress. Campaign 19 combines the remaining best choices and independently passes seven focused tests, 56 byte comparisons and three identical finite decoded pairs. Its seven-pair direct-parent comparison is complete: -1.220% at 24 MP (7/7), -0.296% at 48 MP (5/7), -0.377% on Kodak01 (4/7), and +0.057% on Kodak17 (3/7). Matching controls are +0.284% at 24 MP and +0.209% at 48 MP, diagnostic of variability rather than corrections. Campaign 20 is complete: all six large-image medians improve 0.24–1.35%, but many individual pairs are mixed; Kodak01 improves 0.364% (2/3), while Kodak17 regresses 0.427% (1/3 faster). These smaller effects do not change the confirmed original-baseline gain of the main short-filter bundle. The final prototype is restored in the isolated worktree; seven tests re-pass and the shader matches the frozen measured library. The independent artifact audit verifies 23 late-study cohorts, 256 pairs and 3,356 file hashes.

Broader interpretation and prior rejected mechanisms: [REPORT.md](REPORT.md), [HYPOTHESIS-AUDIT.md](HYPOTHESIS-AUDIT.md), [PHASE12-14.md](PHASE12-14.md). No planned comparison remains pending. The practical stopping conclusion is empirical for these workloads, not a mathematical hardware optimum or proof that no small improvement remains.

Updated: 2026-09-22T02:27:29.371747+00:00
