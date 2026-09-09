# CUDA rolling-tile crossover screen (S149)

## Question and fixed experiment

Starting commit: `8295803` (S148). No production source or GPU body changes
are made in this study. S148 showed that rolling 48 beat rolling 96 on its
2000-square input. S149 asks whether that preference extends across content
and whether an area-only crossover can be justified for larger images.

The matrix contains flower and keong at 2000x2000, 2560x1800, 2560x2160,
3200x2160, and 3839x2159; a third 2000-square photographic content
(riaphotographs); and the existing padded 4K control. The areas are 4.000,
4.608, 5.530, 6.912, and 8.288 million pixels. Existing 2000-square inputs
and the padded control are reused unchanged. Other inputs use byte-exact
floor-coordinate nearest replication of frozen 500-square linear-RGB PFM
pixels. Stored bottom-up orientation is preserved, without floating-point
resampling arithmetic. The verifier reconstructs every output row.

All executables are frozen binaries: S143 creates reference bytes/summaries
and cross-checks wide/compact storage; S148 supplies normal and event-timed
within-process harnesses plus the native compact encoder for memory checks.
Their hashes remain identical to the prior frozen manifests. No new compiler
output or altered device code can explain a crossover in this study.

Ninety-six preflights cover both storage modes, normal/event harnesses, and
release/host-ASAN builds. The timed screen is compact-mode only: twelve cases
in two independent, oppositely ordered processes. Each has six warm and twelve
measured balanced Williams rounds, with six labels: plain 0/3, rolling 48 on
eligible planes 1/4, and the S148 mixed policy 2/5. Each main plane is at least
4M, so the primary comparison is rolling 96 versus rolling 48 there. Half-size
planes use identical choices in those two families. Every encode checks exact
reference bytes and summary/storage contracts.

The pinned primary is the median paired-round difference between duplicate-
averaged full-resolution vertical totals (three invocations per encode).
All four individual cross-label comparisons, duplicate controls, unchanged
horizontal/half-size stages, telemetry, and outer timings are retained.
Event intervals can include stream idle time and perturb scheduling. This
screen is not an uninstrumented whole-encoder throughput qualification.

## Results

Positive percentages below mean rolling 96 is slower than rolling 48.
Each pair gives forward/reverse process-order results. Cross wins count
how many of the four label comparisons favor rolling 96; zero favors 48
in every individual cross. The horizontal column is an unchanged-stage
control, not an optimization claim.

| Content / geometry | Full vertical change % | 96 cross wins / 4 | Horizontal control % |
| --- | ---: | ---: | ---: |
| flower_2000x2000 | +2.95 / +3.59 | 0 / 0 | -1.60 / -0.44 |
| flower_2560x1800 | +0.59 / -0.11 | 1 / 0 | +0.30 / -7.38 |
| flower_2560x2160 | +1.62 / +2.48 | 0 / 0 | +2.95 / +0.16 |
| flower_3200x2160 | -1.85 / -3.79 | 3 / 3 | +6.02 / -0.16 |
| flower_3839x2159 | -0.12 / -0.25 | 1 / 2 | +0.78 / -0.59 |
| keong_2000x2000 | +3.81 / +3.08 | 0 / 0 | +1.87 / -2.69 |
| keong_2560x1800 | +0.86 / +3.10 | 0 / 0 | -3.26 / +2.55 |
| keong_2560x2160 | +0.96 / +0.85 | 3 / 1 | +0.08 / +0.96 |
| keong_3200x2160 | -1.23 / -3.61 | 3 / 4 | +8.12 / -0.25 |
| keong_3839x2159 | +3.34 / +3.25 | 0 / 0 | +8.54 / +7.39 |
| riaphotographs_2000x2000 | +3.07 / +3.00 | 0 / 0 | -1.94 / -3.29 |
| padded_4k_control | +1.60 / +0.35 | 2 / 3 | +1.04 / +0.15 |

The 4M preference for rolling 48 repeats across all three photographic
contents and both process orders: all 24 full-vertical cross comparisons
favor 48. Rolling 96's paired primary penalty is 2.95-3.82% there.
This independently extends the S148 observation beyond its single flower
content. It does not establish a whole-encoder gain of that size.

The larger-size evidence is not monotonic or uniformly robust. At 4.608M,
flower's reverse-order paired primary slightly favors 96 (-0.11%), while
all four separate label-median differences favor 48. These summaries use
different nonlinear aggregation, so both are retained. At 5.530M, flower
consistently favors 48, but keong's cross-label results disagree. At 6.912M,
both contents' primary medians favor 96, with 13/16 favorable crosses.
At 8.288M, the flower and padded control are mixed, while keong favors 48
but also exhibits large unchanged-horizontal shifts.

Duplicate differences can exceed candidate differences. The largest
full-vertical duplicate delta is 0.643 ms; horizontal is 0.652 ms and the
combined pair 1.309 ms (keong 3200x2160, reverse order). The largest outer
duplicate difference is 11.583 ms. No rows or observations are discarded,
and no tiny primary percentage is reported as a reliable universal win.

## Post-hoc temporal control

After observing the large unchanged-horizontal shifts, a separate post-hoc
breakdown compares all twelve individual GPU intervals. It leaves the pinned
aggregate primary unchanged. Stage 0 is the first horizontal convolution and
precedes the first changed vertical kernel in the current encode.

For keong 3839x2159, the first horizontal interval shifts -0.16% in the first
process order but +10.65% in the reverse order; later horizontal stages also
vary. Keong 3200x2160's first horizontal shift changes from +9.84% to -2.80%.
Thus, some large control shifts are already present before the current
encode executes its first changed kernel. The data do not support attributing
all such shifts to that kernel's current-encode power or timing effects.
They also do not identify a unique external cause. Prior-encode state,
host/driver scheduling, and event observer effects are not separated here.

All 6,576 enforced-power-limit endpoints are 40 W, and all 5,904 detailed
telemetry records have successful query statuses. Endpoint equality does
not establish a constant effective clock or an unchanged device state
throughout each interval. No clock, power, thermal, affinity, priority, or
security setting is changed.

## Qualification and launch recovery

The study checks 3,375 oracle-exact encodes: 72 frozen reference/cross-mode
encodes, 672 preflight encodes, 2,616 timed encodes, twelve encodes in four
device memcheck/initcheck jobs, and three native launch-diagnostic encodes.
Forty-eight preflights use host ASAN. The event record contains 35,424 GPU
intervals. All successful memory checks report zero errors and leaks where
applicable. The same already-qualified S148 device bodies are reused.

After the timing campaign completed, the first memcheck attempt exited 13
with “Error launching target app.” The study preparation had omitted its
local temporary directory, although the shared runner set TEMP/TMP to that
path. A direct native encode passed. Creating only that study-local directory
made an identical sanitizer command launch and pass; the remaining checks
also passed. The failed job and failed parent campaign are preserved, and
a versioned recovery script records the missing directory and exact retry.
No completed timing or preflight campaign was restarted, no executable was
changed, and no administrator or firewall permission was required.

## Decision and next mechanism

S148 remains the retained runtime baseline. The evidence strengthens the
case for keeping rolling 48 at 4M, but does not establish an optimal universal
replacement area threshold. No production selector is changed by this
screen, and the endpoint/control disagreements remain part of the record.

The next kernel experiment targets shared-load and barrier cost directly.
The present three-output-row-per-lane body processes 24 output rows per
chunk, loading 35 input rows per lane across its three adjacent outputs.
Four adjacent outputs would need 36 input rows, reducing ideal shared input
loads per output by 22.9%. Eight warps would produce 32 output rows per chunk,
filling a 64-row ring exactly and reducing a 96-row tile from four chunks
to three. This removes the currently unused ring slot for normalization:
test an original-order sum in each warp's lane 0 and a single full-warp
broadcast before the chunk loop. That avoids adding shared memory beyond
24 KiB, at the cost of extra normalization work and likely more registers.
These are hypotheses requiring native-code, occupancy, exactness, sanitizer,
and integrated-performance qualification; no gain is assumed.

Evidence is under `build-cuda-ninja/profiles/s149-artifacts`. The before
snapshots, twelve input definitions, deterministic transformations, frozen
oracle hashes, pinned execution protocol, original failure and recovery,
every event/control sample, and derived tables are retained. Run
`python -X utf8 build-cuda-ninja/profiles/verify_s149.py --frozen` to recheck
the study, frozen S148 artifacts, unchanged production sources, and the
forty retained-runtime hashes.
