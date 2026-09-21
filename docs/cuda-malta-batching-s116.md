# Lane-contiguous Malta scaling batches (S116)

Date: 2026-09-08. Starting revision: `b060927`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Decision

Neither batching variant is retained. Preserving contiguous lane accesses
removes S115's large transaction increase, but the two-iteration variant
executes more warp instructions and the four-iteration LF variant loses
occupancy. No production code, routing, API, compatibility layer, allocation,
launch or arithmetic policy changes. S114 horizontal pairing, S111 fusion,
serial CPU scheduling and opt-in compact coefficient storage remain unchanged.
These are kernel diagnostics, **not a whole-encoder speedup claim**.

## Experiment

[S115](cuda-malta-layout-s115.md) found that assigning adjacent scaling inputs
to each thread increases shared-store wavefronts and global-load sector
requests. S116 instead unrolls two or four iterations of the original
cooperative load. At each unrolled item, neighboring lanes still address
neighboring input and shared-tile values. The index is
`batch + item * blockDim.x`, and the batch advances by
`ScaleBatch * blockDim.x`.

The first item is covered by the outer loop bound. Later items have explicit
tail checks. All valid tile values, including the halo, are written before
the original zero-tile collective. The scalar scaling helper, its divisions,
the paired-row response arithmetic and accumulation semantics are unchanged.
There is no separate scaling plane, preload array or extra launch.

The diagnostic uses the existing 32x64 output tile, 256 threads and 11,520
static shared bytes. Labels 0/4 use the actual production launcher; label 1
is the copied one-iteration control; 2/5 are duplicate two-iteration labels;
3 uses four iterations. Production selects 32x64 for the timed HD/4K cases.
The fixed tile is not a production identity control on tiny inputs that use
smaller production tiles.

An initial build retained a redundant first-item bounds check even for the
one-iteration copy. Its native audit found all four copied control bodies
differed from production. That build and its passing correctness tests are
preserved, but it is not timed. The revised source excludes the first-item
check; all four copied full/LF and 2D/flat controls then match production
instruction-for-instruction. This is an experiment-control correction, not
a correction to production behavior.

Both builds retain all 78 production GPU bodies unchanged and add 12
diagnostic bodies. Each build's four release/host-ASAN executables match its
90-body object. The revised 2D baseline/two-iteration kernels use 48 registers
for full responses and 40 for LF; four iterations use 48 for both. The
reported target/reference bodies have no stack frames or register spills.
Response loads are unchanged: 70 scalar shared-load occurrences per paired
body. Static code counts do not measure executed work in the scaling loop.

## Measurement protocol

The timing campaign uses only the revised binary. Twenty-four isolated jobs
cover six S65 first-full-scale captures each at 1919x1079 and 3839x2159,
with two opposite-order repetitions. These are historical captured stage
inputs, not fresh current-encoder inputs or all AQ scales/iterations.
Stages 0/1 are full/init, 2/4 dense LF/add, and 3/5 zero LF/add.

Each job runs six untimed qualification bursts, six warm Williams rows and
twelve measured rows of six labels. A burst contains four launches; every
burst is checked bit-for-bit against the separate scalar scale/response GPU
reference. Uploads, resets and checks are outside CUDA-event intervals.
The initial default-stream upload is synchronized before the nonblocking
replay stream is used. Build, profiler and other recorded jobs are excluded
from each timing interval. Clocks, power settings, priority and affinity
remain unmodified.

For each measured row, the control is the mean of labels 0/1/4. The
two-iteration contrast uses the mean of labels 2/5 in that same row; four
iterations use label 3. The table gives ranges of paired percentage medians
across two captures and two repetitions per group. Negative means faster;
the ranges are not confidence intervals or ratios of separately pooled medians.

| Case | Two iterations | Four iterations |
|---|---:|---:|
| HD full | -0.47 to +1.16% | -0.59 to +0.50% |
| HD dense LF | -0.55 to +0.34% | +2.31 to +4.46% |
| HD zero LF | -0.39 to +0.07% | +1.45 to +1.88% |
| 4K full | +0.82 to +1.45% | +0.96 to +1.28% |
| 4K dense LF | +0.18 to +0.35% | +5.02 to +5.22% |
| 4K zero LF | -0.09 to +0.16% | +2.98 to +3.55% |

Baseline label-4/label-0 duplicate medians span -0.45 to +0.61%; two-iteration
label-5/label-2 duplicates span -1.31 to +0.79%. Much of the two-iteration
variation is small relative to these controls. It does not establish a gain.
Four iterations consistently regress LF stages, and both variants regress
the tested 4K full stages. No integrated encoder timing is attempted after
this failed screen.

## Counters

Nsight Compute 2025.2.1 captures three 4K stages (0, 2, 3), three variants and
two reverse-order repetitions: 18 reports, each with one matching kernel and
four replay passes. Clocks and caches remain uncontrolled; the profiler's
warnings are retained. Counter changes are diagnostic, not timing estimates.

Two iterations preserve memory behavior: global-load sector changes are
-0.11 to -0.01%, L2 traffic -0.06 to +0.07%, and shared-store wavefronts
-0.22 to +0.64% versus baseline in these captures. This is unlike S115's
68–93% extra global-load sectors and roughly doubled shared stores. However,
executed warp instructions increase 1.42% on the full stage, 1.45% on dense
LF and 2.95% on zero LF, identically in both repetitions. FFMA thread counts
are unchanged. Fewer outer-loop iterations did not reduce executed work.

Four iterations do not save appreciable instructions either: changes are
+0.57% full, +0.03% dense LF and +0.06% zero LF. The LF register increase
from 40 to 48 lowers the register-limited resident CTA count from six to five.
The observed LF shared carveout changes from 100 KiB to 64 KiB, with 12,544
shared bytes per CTA including 1,024 driver-reserved bytes. Full response
launch attributes are unchanged. The LF L2 decrease is only 0.35–1.27% and
does not compensate for the timing regression. Minor baseline LF conflict
counts vary; this study does not assign them a causal explanation.

## Qualification and evidence

Each of the initial and revised builds passes 10,752 release fixtures and
the same 10,752 under scoped host ASAN, with three successive accumulation
stages per fixture. These are repeated qualifications, not 43,008 distinct
images. The 14 shapes cover odd tile edges, thin/tall images, padded independent
strides and nonzero offsets. Six labels, requested 2D/flat forms, full/LF,
initialize/add and 16 patterns exercise signed zero, subnormals, thresholds,
NaN/Inf, sparse data and exceptional accumulation. Candidate flat routes are
forced; production labels retain their own dispatch. Inputs and guards are
checked. Host ASAN instruments the harness, not GPU code or every linked library.

Both builds also pass 24 HD/4K replay preflights each and four CUDA sanitizer
jobs each (memcheck, racecheck, initcheck, synccheck), with 240 scoped fixtures
per job and no errors or race warnings. All timed and profiled bursts match
the scalar GPU reference exactly. There is no new full-encoder/CTest claim:
production source is unchanged and keeps its S114 qualification.

Evidence is archived at `U:/gjxl-cuda-diagnostics/s116`: original/revised
sources and binaries, native dumps, raw timings, profiler reports, source
snapshots, `inputs.json`, `inputs_v2.json`, `native.json`, `native_v2.json`,
`screen.json`, `summary.json`, `mechanism_analysis.json` and the artifact hash
manifest. Frozen scripts include `s116_campaign_v2.py`, `s116_analyze.py`,
`s116_mechanism_v2.py`, `s116_validate.py` and `s116_freeze.py`.
Validation checks exit codes, log/source/capture hashes, native identities,
sanitizer summaries, timed-job isolation and all 40 retained runtime hashes.
All 136 recorded jobs completed successfully, including the untimed original
control experiment. All launched processes are terminal. No admin, firewall
or permission blocker was observed, and no system/security settings changed.

This closes plain guarded two-/four-iteration scaling unrolling, not Malta
optimization. The next bounded test should separate guaranteed full batches
from the scalar tail, eliminating per-item tail tests before considering
explicit input preloading. The successful earlier CfL study likewise separated
preloading and full-chunk/remainder handling from plain unrolling; it is a
reason to test that mechanism, not evidence that Malta will benefit.
