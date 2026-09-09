# CUDA integrated four-output-row scheduling (S151)

## Scope

S150 qualified four adjacent output rows per lane: lower shared-load and
instruction work, bitwise reference agreement, and repeatable sustained
convolution-pair gains. Its common control forced three-row tile96, which
is not the production choice on every tested plane. S151 compares the
prototype inside complete fully resident encodes against actual S148
dispatch and an explicit rolling48 control. Starting commit: `695cf0f`.
Production source is unchanged during this diagnostic study.

| Family | Ineligible planes | Eligible planes |
| --- | --- | --- |
| 0: actual S148 | Plain48 | Three-row 48 below 4M pixels; three-row 96 from 4M |
| 1: rolling48 control | Plain48 | Three-row 48 |
| 2: four-row 64 | Plain48 | Four-row 64 |
| 3: four-row 96 | Plain48 | Four-row 96 |
| 4: four-row 128 | Plain48 | Four-row 128 |

Eligibility remains width at least 32, height at least 96, and area at least
2,000,000 pixels. There is no changed small-image fallback, host weight sum,
reciprocal substitution, tap reordering, or compatibility layer. Six
geometries/contents are used: flower500, padded 1919x1079, flower2000,
padded 3839x2159, flower3200x2160, and keong3839x2159. The last two retain
the exact S149 inputs and frozen oracles; no new resampling is performed.

The complete-encode boundary is distance 1.2, effort 7, fully resident AQ,
automatic CPU threads, and final-score collection disabled. Both wide
and compact AC coefficient storage are tested; these are storage variants
of the fully resident path, not exact-coefficient mode.

## Integration and native-code controls

The diagnostic CUDA translation unit renames only the original public
low/medium function definition, leaving its normal psycho-construction
call sites intact. A replacement public wrapper chooses the requested
body and records dispatch. Thus the comparison runs through the normal
encode path, rather than only a replay harness. Six observed call shapes
and their exact selected bodies are checked on every encode.

Both ordinary and event-instrumented CUDA objects contain the same 83
device bodies as S150, instruction-exact. Full encoders contain all 223
production bodies unchanged plus the three S150 bodies, also exact.
Eight linked executables (ordinary/event, wide/compact, release/host-ASAN)
retain nine unchanged GPU modules and one expanded Butteraugli module.
Each four-executable ordinary/event group has identical module hashes.
The owned S148 weight payload and host implementation are unchanged.

The dispatch globals are confined to this serial diagnostic harness; they
are not a proposed production API or a concurrency guarantee. Production
integration must remove this harness state and qualify its selected policy.

## Pinned protocol

Each family has two identical labels, `f` and `f+5`. A ten-label randomized
Williams design uses ten warmup and twenty measured rounds, with a second
repetition in reversed process order. The six cases and two storage modes
give 24 ordinary and 24 separate event-instrumented timing processes.
Preflights exercise all eight executables on all six cases. Memcheck and
initcheck cover HD and photographic 4K in both storage modes, through the
event-instrumented encode path.

Every encode must match its frozen codestream byte-for-byte and its
in-process baseline summary exactly. AC storage width and byte count,
dispatch-family counts, body counts, and the six main/half call shapes
must also agree with the intended configuration. Diagnostic mode changes
are outside the encode interval. No clock, power, priority, affinity, or
security setting is changed.

Ordinary and event timing are never pooled. The event executable records
horizontal and vertical intervals around each of six low/medium pairs,
with no additional stream synchronization: the completed encode must make
all end events queryable. Events can include idle gaps and perturb
scheduling. Unchanged horizontal work, including the first horizontal
stage before the encode's first changed vertical kernel, is a control.
Power-limit endpoints and separate clock/power/temperature/state telemetry
are retained; none is substituted for ordinary throughput.

For each measured round, the primary difference averages each family's two
labels, then takes the median paired-round difference and percentage.
Every candidate/control cross-label comparison and every duplicate-label
difference is retained. Comparisons include actual dispatch, rolling48,
and the new tile sizes against one another. This differs from S150's
average-of-label-medians aggregation, so those estimators are not pooled.
Event logs retain the field name `tile` for the diagnostic body key:
0=plain48, 1=three-row48, 2=three-row96, 3/4/5=four-row64/96/128.

## Qualification

All 48 release/host-ASAN preflights pass, checking 528 complete encodes.
All eight CUDA memcheck/initcheck processes pass, checking another 88
encodes; memcheck reports zero leaked bytes. These integrated checks add to
S150's independent guarded-array, ordered-tap, ownership, captured-graph,
boundary, racecheck, and synccheck qualification of the identical kernels.
They do not replace that lower-level coverage with codestream checks alone.

The study uses its own explicitly created temporary directory. No launch
retry, admin approval, firewall change, or operating-state adjustment has
been needed.

All 48 timing processes also pass, adding 14,448 checked encodes. The final
total is **15,064** complete frozen-oracle checks. Validation independently
reconstructs the Williams position/preceding-label balance, dispatch counts,
six call shapes, 90,912 event intervals, 15,152 telemetry records, and 30,128
power-limit endpoints. All 104 GPU-facing jobs ran serially; all 111 recorded
jobs finished successfully. This is diagnostic integration qualification,
not a fresh production-build/CTest qualification.

## Ordinary complete-encode results

The twenty eligible-image processes (five cases, two storage modes, two
repetitions) give the following comparisons against actual S148 dispatch.
Negative percentages mean shorter complete encodes. The range is the
observed range of process-level paired-round medians, not a confidence
interval or a transferable speedup estimate.

| Candidate | Favorable primary comparisons | Median percentage | Observed percentage range |
| --- | ---: | ---: | ---: |
| Four-row64 | 12/20 | -0.290% | -3.258% to +2.226% |
| Four-row96 | 10/20 | -0.181% | -2.945% to +4.428% |
| Four-row128 | 12/20 | -0.435% | -3.483% to +2.460% |

Each candidate is favorable in only two of four flower500 processes, where
all families actually launch the same plain kernels. The largest absolute
ordinary duplicate-label difference is **16.09655 ms**, between the two
actual-dispatch labels in first-repeat compact padded 4K. These controls and
the sign changes preclude a dependable whole-encoder speedup claim. Results
from the separate event executables are not added to these counts.

## In-encode vertical intervals

The following ranges cover four process-level primary results per case:
wide/compact storage and both repetitions. Each interval sums the three
full-resolution vertical calls, not horizontal work or the whole encoder.
The control is actual S148 dispatch, including three-row48 at HD.

| Case | Actual vertical baseline (ms) | Four-row64 | Four-row96 | Four-row128 |
| --- | ---: | ---: | ---: | ---: |
| Padded HD | 1.352-1.362 | -1.13% to -0.78% | +0.53% to +0.99% | +0.82% to +2.21% |
| Flower2000 | 2.720-2.748 | -9.30% to -8.52% | -5.67% to -5.20% | -5.67% to -4.45% |
| Padded 4K | 7.727-7.832 | -9.33% to -7.60% | -10.13% to -8.95% | -10.03% to -6.98% |
| Flower3200x2160 | 5.440-5.536 | -5.94% to -3.62% | -9.22% to -4.30% | -9.18% to -6.22% |
| Keong3839x2159 | 7.390-7.506 | -8.56% to -7.12% | -10.41% to -8.57% | -8.42% to -8.03% |

All three new families improve all sixteen larger-image primary comparisons
and all 64 cross-label comparisons each, against both actual dispatch and
the explicit rolling48 control. Tile64 additionally improves all four HD
primary and sixteen cross-label comparisons against each control. Tiles96
and128 regress all four HD primaries. This is not a universal larger-tile win.

Flower2000's gain is not just correcting the old 4M threshold: against
rolling48, tile64 still improves 4.64-6.33%, tile96 1.38-2.48%, and tile128
0.80-2.76%, with all cross-label comparisons favorable. Tile64 beats tile96
on all four flower2000 primaries and all sixteen cross-label comparisons.
Conversely, tile96 beats tile64 on all four padded-4K and all four
flower3200 primaries, though only 14/16 and 15/16 cross-label comparisons
respectively; photographic 4K is 3/4 primary and 10/16 cross-label favorable.
Tile128 does not establish an advantage over tile96 at either 4K content:
all eight primary comparisons favor tile96, despite tile128's sustained
replay advantage in S150. Those different execution contexts are not pooled.
Direct pairwise medians are computed independently; subtracting two
separately summarized comparisons against the old control is not equivalent.

Both 4K cases also select rolling kernels on their half-resolution planes.
Their half-resolution vertical primary changes are favorable in all four
processes per new family. Other cases keep plain half-resolution kernels;
their changes remain controls, not additional optimized work.

## Controls and scale of the gain

Unchanged horizontal intervals are not stable enough to treat as exact
normalizers. The largest absolute full-vertical duplicate-label difference
is 0.295424 ms; the largest horizontal duplicate difference is 0.457728 ms.
Even the first horizontal stage, before the encode's first changed vertical
kernel, has a 0.102912 ms duplicate difference in first-repeat wide
photographic 4K. For tile128, that unchanged stage is 4.57-6.53% shorter
across all four photographic-4K primaries. Its favorable sign is therefore
not uniquely evidence of the current encode's changed vertical work.
The flower500 half-resolution control ranges from -12.01% to +13.08% for
tile128 labels despite identical kernels. All these observations are retained.

Across all phases, all 30,128 power-limit endpoints report 40,000 mW.
All six management-query statuses succeed on every telemetry record, yet
reported SM clocks range from 210 to 1,770 MHz. P-state 3/memory 5,500 MHz
occur on 15,149 records; three report P-state 5/memory 810 MHz. Endpoints do
not establish clocks during individual kernels or identify a unique physical
cause for timing variation. No operating-state observations were discarded.

Absolute scale matters. In tile96's event-instrumented 4K comparisons, the
six changed vertical intervals together occupy roughly 2.91-3.65% of the
baseline complete-encode time. Their paired savings are 0.820-0.936 ms for
padded 4K and 0.660-0.926 ms for photographic 4K, about 0.21-0.34% of the
corresponding instrumented outer baselines. These ratios explain the scale
of the opportunity; they are not ordinary-encode speedup estimates and do
not remove possible event or operating-state effects. Combined with S150's
exact arithmetic/native-work evidence and sustained pair results, S151
supports a targeted stage optimization, not a broad throughput percentage.

## Decision and next step

Keep S148 as the retained runtime for this study. Select a bounded candidate
for fresh production integration: preserve the plain fallback and existing
2M/4M eligibility boundaries, use four-row64 in the old rolling48 region,
and four-row96 in the old rolling96 region. Do not add tile128 or infer a new
area crossover from this small corpus. This deliberately leaves the better
flower2000 tile64 result documented rather than claiming the retained 4M
boundary is optimal.

The proposed mixed policy is not one of S151's uniform eligible-plane
families: full-4K/half-resolution encodes would combine tile96 and tile64.
It therefore needs explicit mixed-dispatch, narrow/tall/boundary, ownership,
fresh-build, and complete-encode qualification before promotion. Remove the
diagnostic selector globals and superseded production rolling bodies during
that integration; no compatibility adapter is needed. Preserve compact
storage's opt-in status and all unrelated routing/allocation behavior.

After qualification, refresh the resident critical-path profile before
further convolution micro-tuning. S151 does not establish that encoding is
maxed out, and its sub-millisecond 4K stage savings do not account for the
much larger remaining encoder time.

## Evidence

Local evidence is under `build-cuda-ninja/profiles/s151-artifacts/`:
`before.json`, `inputs.json`, `protocol.json`, `native.json`, all 111 job
records/logs, `analysis.json`, `report.json`, and `details.json`. The final
snapshot includes the diagnostic source and its dependencies. `verify_s151.py`
recomputes analysis and both reports, verifies serial job execution, native
identity, all pinned inputs/oracles, prior S150 evidence, and forty retained
runtime hashes. `freeze_s151.py` records source snapshots and the artifact
inventory; `verify_s151.py --frozen` verifies that inventory and its hashes.
Failed or historical studies and the protected user notes are untouched.
