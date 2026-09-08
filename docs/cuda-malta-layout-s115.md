# Malta layout and scaling-consumption study (S115)

Date: 2026-09-08. Starting revision: `5edaca2`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Decision

Neither column-paired responses nor adjacent two-value scaling is retained as
a universal Malta replacement. Both preserve exact results, but their gains
depend on response type and image size. This is a diagnostic study, **not an
encoder speedup claim**. No production source, dispatch, API, compatibility
layer, buffer, launch, synchronization or arithmetic policy changes.
S114 horizontal pairing, S111 composition/reduction fusion, serial CPU tile
scheduling and opt-in compact coefficient storage remain unchanged.

The useful new finding is about consumption layout: reducing loop work or
using wider instructions does not necessarily reduce memory transactions.
Adjacent scaling increases global-load sector requests and shared-store
wavefronts. Full-response column pairing increases shared-load wavefronts and
register pressure. By contrast, S114's wider row-paired tile reduces L2 traffic
without materially reducing DRAM bytes, arithmetic or nominal halo work.

## Controlled variants

All diagnostic variants use 256 threads and 11,520 bytes of static shared
storage. The 32x64 and 64x32 output tiles both have a four-pixel halo and a
nominal 1.40625 scaled values per interior output. Boundary work can differ.

| Labels | Output tile | Responses per thread | Scaling loop |
|---|---|---|---|
| 0 / 4 | Production-selected, 32x64 for timed cases | Two rows | One input |
| 1 | 64x32 | Two rows | One input |
| 2 | 64x32 | Two columns | One input |
| 3 / 5 | 32x64 | Two rows | Two adjacent inputs |

Labels 0/4 call the actual production launcher. Labels 3/5 call the same
candidate; these are duplicate controls, not independent implementations.
Label 1 is instruction-identical to S114's 64x32 prototype for both response
types and both grid forms. Compare 2 against 1 to isolate response mapping;
compare 3/5 against 0/4 to isolate scaling batching on the timed HD/4K cases.
The fixed 32x64 diagnostic is not an identity control for production's smaller
tile selections on tiny inputs.

The response sum trees, direction order, initialization/addition behavior and
rounded divisions are unchanged. Full responses interleave two outputs by
direction; LF responses use two sequential response helpers, as in the retained
row-paired implementation. Every thread participates in halo loading and the
zero-tile collective before output bounds checks. Odd second columns/rows are
not stored. Scaling batching changes index assignment to `2*batch + item`;
each input still calls the original scalar scaling helper independently.

The prototype object contains all 78 current production kernels, native-
identical to the qualified S114 production object, plus 12 experiment kernels.
All four release/host-ASAN executables preserve those 90 bodies. There are no
stack frames or spills in the 16 reported target/reference resource records.
For the timed 2D route:

| Variant | Full registers | LF registers | Full/LF register-limited CTAs per SM |
|---|---:|---:|---:|
| Baseline / wider rows / scaling batch | 48 | 40 | 5 / 6 |
| Wider columns | 56 | 62 | 4 / 4 |

The profiler reports 1,024 additional driver-reserved shared bytes per CTA.
Full kernels use a 64 KiB shared carveout, baseline/wide/scaling LF kernels
100 KiB, and column LF 64 KiB. These are observed launch attributes, not a
claim that every kernel uses the device's maximum shared capacity. Static
shared-load occurrences for one response pair remain 70 scalar loads for
full columns; LF columns use 33 two-value loads plus four scalar loads, still
70 values. Static instruction counts are not dynamic per-output work counts.

## Timed screening

Twenty-four isolated jobs cover six retained S65 first-full-scale captures
at 1919x1079 and 3839x2159, with two opposite-order repetitions. These are
historical stage inputs, not fresh current-encoder captures or all AQ scales
and iterations. Stages 0/1 are full/init, 2/4 dense LF/add, and 3/5 zero LF/add.

Each job checks six untimed qualification bursts, then six warm and twelve
measured six-label Williams rows. Each burst contains four launches. Every
burst is checked bit-for-bit against the separate scalar scale/response GPU
reference. Uploads, resets and checks are outside CUDA-event timing; the
default-stream upload is explicitly synchronized before the nonblocking
replay stream is used. No other recorded agent job overlaps a timing job.
Clocks, power settings, priority and affinity are not modified.

The table gives ranges of within-row paired median percentages across stages,
repetitions and, for scaling, both duplicate labels. Negative means faster
than the mean of baseline labels 0/4; these ranges are not confidence intervals.

| Case | Wider rows | Wider columns | Adjacent scaling batch |
|---|---:|---:|---:|
| HD full | -0.28 to +0.62% | +7.02 to +9.19% | -2.95 to -0.31% |
| HD dense LF | -1.26 to -0.51% | -3.62 to -2.29% | -0.58 to +0.76% |
| HD zero LF | +1.25 to +3.50% | -2.93 to -1.06% | -1.80 to +0.63% |
| 4K full | -3.19 to -2.46% | +10.13 to +12.14% | -2.84 to -1.23% |
| 4K dense LF | -4.10 to -3.78% | -2.27 to -1.74% | +1.00 to +1.33% |
| 4K zero LF | -4.33 to -3.94% | -2.23 to -1.51% | +2.93 to +3.48% |

Baseline duplicate medians range from -0.81 to +0.95%; scaling duplicate
medians from -1.09 to +1.10% (4K only: -0.53 to +0.19%). Treat small HD
differences cautiously. Columns beat wider rows on HD LF, but lose to them
on 4K LF and on all full-response jobs. There is no universal winner and no
fresh integrated encode measurement in this study.

## Counter attribution

Nsight Compute 2025.2.1 collected six initial baseline/wide profiles using
the frozen S114 replay, followed by 24 profiles using the S115 binary:
three 4K stages (0, 2, 3), four variants, two reverse-order repetitions.
Each report profiles exactly one matching kernel. Initial captures use 16
replay passes; explicit mechanism captures use four. GPU caches and clocks
are left uncontrolled, and the resulting profiler warnings are retained.
These counters support mechanisms, not profiler-derived throughput claims.

Across the six explicit baseline/wide comparisons, the wider row tile has
4.90–5.26% less L2 traffic and 5.19–6.04% fewer global-load sector requests.
DRAM read changes are -0.08 to +0.89%, with essentially unchanged register
limits and instruction count (about +0.03–0.06%). This reproduces the initial
roughly 5% L2 difference. The 4K gain is not explained by fewer nominal halo
values, lower register use, or a comparable reduction in DRAM traffic.

Two adjacent scaling inputs per thread cause roughly twice the shared-store
wavefronts (+100–103%) and 68–93% more global-load sector requests. L2 traffic
increases 17–46%, while executed warp instructions fall only 0.75–1.57% and
FFMA thread counts are unchanged. DRAM bytes barely change. Full-response
captures report 367,200 shared-store bank conflicts versus zero, identically
in both repetitions. The source's stride-two lane mapping and separate scalar
stores are consistent with this extra shared-memory work. Fewer loop iterations
are not a sufficient reason to retain this consumption layout.

Full-response columns nearly double shared-load wavefronts (+99.91%) and
report 9,067,800 shared-load bank conflicts versus zero in both repetitions,
alongside the lower register-limited occupancy. LF columns reduce warp
instruction count by 7.75% on the dense stage, but increase L2 traffic by
9.20–9.85% and reduce the register limit from six to four CTAs. Zero LF has
no response shared loads, yet column mapping still changes accumulation
traffic and instruction count. These tradeoffs help explain why selecting
columns solely from wider static loads or from an HD result would be unsafe.

Small nonzero shared-conflict counts also appear in baseline LF captures and
vary between repetitions. They are preserved without assigning a causal
explanation; the large replicated effects above are the stronger evidence.

## Qualification and evidence

Release and scoped host-ASAN each pass 10,752 fixtures, with three successive
accumulation stages per fixture: 14 shapes, six labels, requested 2D/flat
forms, full/LF, initialize/add and 16 patterns. The candidate flat routes are
forced; production labels retain their own dispatch. Cases include odd tile
boundaries, thin/tall planes, padded independent strides, nonzero offsets,
signed zero, subnormals, threshold-adjacent values, NaN/Inf and exceptional
accumulation. Inputs, outputs and guards are checked. Host ASAN instruments
the harness, not the GPU or every linked library.

Twenty-four HD/4K replay preflights pass (release and host-ASAN). CUDA memcheck,
racecheck, initcheck and synccheck each pass 240 scoped fixtures with no errors
or race warnings. All 40 retained runtime artifacts retain their hashes.
Production source is unchanged, so no new full encoder/CTest qualification is
claimed beyond S114. No firewall, admin or permission blocker was observed.

Evidence is archived at `U:/gjxl-cuda-diagnostics/s115`, including source
snapshots, binaries, native dumps, raw observations, profiler reports,
`inputs.json`, `native.json`, `screen.json`, `counter_analysis.json`,
`mechanism_analysis.json`, and the artifact hash manifest. The frozen scripts
include `s115_campaign.py`, `s115_analyze.py`, `s115_mechanism.py`,
`s115_validate.py` and `s115_freeze.py`. Validation checks job exit codes,
log hashes, timed-job isolation, captures, native identity, sanitizer summaries
and retained binary hashes. Five failed CSV exports using unsupported
`--units base` are preserved; corrected `--print-units base` exports pass.
All launched processes are terminal, including those failed exports.

The next bounded experiment is scaling-loop batching that preserves contiguous
lane accesses, rather than assigning adjacent inputs to each lane. A broader
size/response scheduling study may also be worthwhile, but the HD/4K crossover
alone does not establish a dispatch threshold. Neither these rejected universal
replacements nor the earlier scaling/response split exhaust Malta optimization.
