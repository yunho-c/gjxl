# S182: broader bounded-appender qualification

The campaign was interrupted by an `ENOSPC` error while the runner created
its 61st timing journal. Sixty timing jobs are fully journaled; the 61st has
a complete passing log but an unverified exit status; fifteen were not run.
This is partial evidence, not completed qualification. Completed sustained
quartets retain local token-worker improvements but mixed whole time,
including three slower Keong results. Do not promote the candidate.
Production remains unchanged and the optimization goal remains active.

## Question and fixed candidate

The [S181 screen](cuda-bounded-token-appender-s181.md) found six of six
coefficient-token worker improvements (6.45–14.64%) and six AC-tokenize wall
improvements, but only three whole-encoder wins. Both 4K whole-time results
regressed alongside changes in earlier quantization. This stage tests the
same candidate more broadly and in sustained single-policy processes.

The parent is `7b0de204fb47967560e489be16dd57d592eff93e` on `feat/cuda`.
Production remains the qualified S168 implementation. Evidence is under
`build-cuda-ninja/profiles/s182-artifacts`, with versioned `s182_*` helpers
beside it. All 343 production source files are pinned and unchanged.

S181's normal/ASAN DLLs and normal interleaved executable are reused byte
for byte. Modes 0/1 are the same original tokenizer; modes 2/3 are the same
bounded appender, for both dense and sparse groups. The existing per-group
budget check and population-collection specialization remain. Sparse rank,
reservation capacities, coefficient decisions, output format and GPU work
are unchanged. No compatibility layer or minimum-ISA change is introduced.

Only a new fixed-policy host executable is compiled, normally and under
AddressSanitizer. Its source audit reconstructs it exactly from S181's
interleaved source by changing CLI mode selection, loop multiplicity, and
log prefixes. The per-Encode body is identical, including the timer and
all correctness checks. There is no device compilation or candidate rebuild.

## Precommitted protocol

The source, inputs, oracles, census, executables, analysis and complete
chronological schedules are pinned before timing. The eleven inputs are
the existing six photographic workloads and five synthetic controls:
flower 500x500, flower-derived 1919x1079 ("1080p"), flower 2000x2000,
flower-derived 3839x2159 ("4K"), flower 3200x2160, Keong 3839x2159,
65x71 pattern 2, 513x519 patterns 0/1/2, and 1025x1031 pattern 2.
Derived photographs are not independent native-resolution image captures.
The fully resident path uses effort 7, distance 1.2 (pattern 2: 0.01),
and disables finalscore. Automatic and eight-thread configurations are
both covered by the interleaved matrix.

Each timing process has one S168 reference encode, 32 warm-up encodes and
64 measured encodes. Each measured encode creates a fresh prepared workflow
while retaining its backend. Prior-result clearing, policy configuration,
reference/setup, validation and read-only NVML endpoints are outside the
timer. All candidate work, including budget checks and common group audit
accounting, is inside Encode.

The following counts describe the planned campaign, not its completed work.

| Cohort | Process design | Planned processes | Planned whole calls | Planned measured calls |
|---|---|---:|---:|---:|
| New harness preflight | 11 inputs x 2 thread settings x normal/ASAN x 4 fixed labels; no warm-up, one checked encode plus reference | 176 | 352 | Not timing evidence |
| Interleaved matrix | 11 inputs x 2 thread settings x 2 reversed passes; 8 warm and 16 measured four-label rounds | 44 | 4,268 | 2,816 |
| Sustained policy | 4K/Keong, eight threads, 4 passes x 4 separate fixed-label processes; 32 warm and 64 measured encodes | 32 | 3,104 | 2,048 |
| Total | | 252 | 7,724 | 4,864 |

Matrix cells are shuffled once with seed 18220260910, then reversed for
pass 1. Within-process Williams orders are 0132/1203/2310/3021, with their
order reversed in pass 1. After each eleven-process matrix half, a sustained
pass runs both images. Its four separate processes follow that pass's
Williams order; image order alternates between passes. This intersperses
the two cohorts in calendar time. Preflight uses seed 18220260911.

Matrix primary statistics are medians of within-round candidate duplicate
means minus original duplicate means. Sustained primary statistics compare
the means of the two candidate process medians with the means of the two
original process medians within each image/pass quartet. Independent
processes are never paired by sample index. Both same-policy duplicate
differences and all four candidate-versus-original comparisons are retained.
Each sustained process also retains four consecutive 16-sample block medians
for every phase. Repeated encodes are not independent image replications.

All 41 phases, outer Encode time, warm-ups and NVML endpoints are retained.
No filtering, reruns, optional stopping, builds, sanitizers, profilers,
compression or heavy hash work occurs during timing. No power, clock,
cooling, affinity, priority, driver or security settings are changed.
Every recorded warm/measured power-limit endpoint must be 40,000 mW.
That is a configured power limit, not instantaneous power or clock telemetry.

No automatic promotion gate is used. A local worker gain alone cannot
qualify this candidate: promotion requires repeatable whole-encoder benefit
across representative inputs and thread settings. Mixed or noise-comparable
results remain unpromoted. Cohort differences cannot uniquely establish
cadence causality; process boundaries, time and operating state also differ.

## Correctness and preservation

The new host harness built once, from 09:11:18.591 to 09:11:35.622 UTC on
2026-09-10, with optimized warning-as-error MSVC and clang-cl/ASAN builds.
The exact mechanical source audit passed. Preflight ran from 09:17:51.894
to 09:20:37.190 UTC: all 176 processes and 352 whole calls passed. The 88
ASAN processes include 88 instrumented encodes and 88 normal reference
encodes. Preflight durations are not used as performance observations.

Every checked encode requires exact whole-output bytes, the full encoding
summary, four-byte native coefficient ownership, and the reference storage
size. The parser independently checks the frozen S172 per-input/thread
census: group/sparse/anchor counts, maximum and used tokens, value/context
capacities, and candidate dispatch. Both value and context capacities remain
the original conservative maximum. All phase durations must be finite and
nonnegative; NVML endpoint ordering is checked against each sample.

The unchanged S181 candidate retains that stage's 318,720 private fixture
checks, 8,192 public reference cases, normal/ASAN all-input qualification,
and clean CUDA memcheck/initcheck runs. Those are inherited evidence, not
new S182 tests. CUDA sanitizers are not repeated for a host-loop-only change.
The unchanged normal/ASAN candidate DLLs each contain the same eleven CUDA
native modules. S181's native audit also established original-control
instruction and normalized-relocation identity with production.

## Interruption and actual coverage

Timing began at 09:21:04.123 UTC. The last fully journaled job, reversed-pass
4K with automatic threads, finished at 09:38:05.526 UTC. The next job,
`matrix_r1_keong_3839x2159_t0`, started at 09:38:05.528 UTC (PID 34732).
The runner printed its heartbeat at 09:38:35.561 UTC, then exited with code
1 and `OSError: [Errno 28] No space left on device` at the journal-writing
line in `s106_run.py`. Its original JSON file is zero bytes and is preserved.

The original 118,964-byte log contains all 96 samples, 192 ordered NVML
endpoints, `S132 POWER PASS`, and `S181 WHOLE PASS checks=96 reference_calls=1`.
Its complete samples pass the same independent parser. This does not recover
the child exit code or finish timestamp; neither is fabricated. The runner's
failure happened after child polling and log reading, but only the captured
start/PID and raw-log evidence are recorded in the additive partial manifest.

Subsequent inspection found no remaining measured executable, an empty
task temporary directory, and 1,127,047,168 bytes free on C: (later
1,126,563,840 bytes), without deletion. U: had 30,883,840 bytes free. A
read-only pagefile snapshot showed 2,560 MB allocated, 266 MB used and a
303 MB peak. These snapshots do not establish the cause of the error or
prove C: remained full. No firewall/admin prompt was observed. No filesystem
cleanup, security change, paging change, restart or campaign resumption was
performed. Storage reliability/headroom needs resolution before more timing.

| Actual evidence | Count |
|---|---:|
| Fully accepted journals, including build | 237 |
| Accepted preflight processes | 176 |
| Accepted timing processes | 60 |
| Complete timing logs with unknown terminal status | 1 |
| Unrun timing processes | 15 |
| Whole calls in accepted preflight/timing jobs | 6,172 |
| Additional whole calls reported by terminal-unverified log | 97 |
| Measured calls in accepted timing jobs | 3,840 |
| Additional terminal-unverified measured calls | 64 |
| Retained timing NVML endpoints, including final log | 11,712 |

All observed endpoints are 40,000 mW. Accepted timing comprises 36 matrix
processes and 24 sustained processes (six complete quartets). The unverified
log is the 37th matrix process. The remaining seven matrix cells and final
eight-process sustained pass are unrun. No data are replaced or rerun; the
original schedule and empty failed journal remain intact. The collected
logs report 6,269 whole calls including preflight, but only 6,172 belong to
fully accepted jobs.

The original full-campaign analyzer/verifier/freezer are retained unused.
New additive `s182_collect_partial.py`, `s182_analyze_partial.py`,
`s182_decide_partial.py`, `s182_verify_partial.py`, and
`s182_freeze_partial.py` handle the interruption explicitly. The partial
analyzer keeps the original estimands and observed samples, permits missing
repeats, and marks the terminal-unverified matrix row. It never presents
the interrupted schedule as complete.

## Results and decision

Only descriptive conclusions are warranted. Among the 36 fully journaled
matrix cells, 33 token-worker, 31 AC-tokenize wall and 22 whole-time contrasts
are negative. The three nonnegative worker contrasts are low-work 513 p0
controls. The terminal-unverified Keong/automatic/repeat-1 row is retained
separately; including it gives 33/37, 31/37 and 22/37 respectively.

The matrix below retains every planned input/thread cell. Entries are
repeat 0 / repeat 1; `*` identifies the complete log with unknown exit status.
Whole differences use the primary median within-round contrast; worker
percentages use the median within-round ratio, so near-zero signs can differ.

| Matrix input / threads | Whole ms, r0 / r1 | Worker %, r0 / r1 |
|---|---|---|
| flower_500 / auto | +0.107 / +0.053 | -6.98 / -8.51 |
| flower_500 / 8 | -0.277 / unrun | -10.47 / unrun |
| 1080p / auto | +0.528 / +0.623 | -9.95 / -12.26 |
| 1080p / 8 | +0.710 / -0.001 | -8.83 / -10.76 |
| flower_2000 / auto | -0.323 / -0.012 | -16.02 / -14.83 |
| flower_2000 / 8 | -1.569 / unrun | -10.78 / unrun |
| 4k / auto | +1.664 / +2.037 | -5.18 / -6.41 |
| 4k / 8 | +1.093 / unrun | -8.15 / unrun |
| flower_3200x2160 / auto | +0.814 / unrun | -11.94 / unrun |
| flower_3200x2160 / 8 | +2.560 / -1.633 | -13.71 / -7.95 |
| keong_3839x2159 / auto | +1.283 / +8.101* | -10.23 / +0.09* |
| keong_3839x2159 / 8 | -0.244 / unrun | -13.66 / unrun |
| 65_p2 / auto | -0.055 / -0.027 | -12.35 / -11.53 |
| 65_p2 / 8 | -0.224 / unrun | -6.55 / unrun |
| 513_p0 / auto | -0.116 / +0.301 | -4.94 / +0.76 |
| 513_p0 / 8 | +0.030 / -0.074 | -0.02 / +7.23 |
| 513_p1 / auto | -0.285 / -1.189 | -6.32 / -12.14 |
| 513_p1 / 8 | +0.165 / -0.721 | -2.11 / -8.72 |
| 513_p2 / auto | -0.162 / -0.961 | -16.19 / -13.48 |
| 513_p2 / 8 | -1.367 / -0.926 | -15.81 / -20.35 |
| 1025_p2 / auto | -3.573 / unrun | -27.21 / unrun |
| 1025_p2 / 8 | -4.354 / -0.885 | -27.96 / -24.52 |

All six completed sustained quartets improve worker work and AC-tokenize
wall time, but only two improve whole time. Negative numbers favor the
candidate. These are contrasts of process medians, not samplewise pairs.

| Sustained image / pass | Whole ms | Whole % | Tokenize wall ms | Worker ms | Worker % | Whole duplicate differences ms (original, candidate) |
|---|---:|---:|---:|---:|---:|---|
| 4K / 0 | +2.274 | +0.887 | -0.119 | -2.065 | -8.68 | -2.633, -5.687 |
| 4K / 1 | -2.646 | -0.986 | -0.383 | -3.220 | -12.66 | -3.827, -0.713 |
| 4K / 2 | -2.689 | -0.991 | -0.386 | -3.606 | -14.20 | +1.531, +2.631 |
| Keong / 0 | +2.337 | +0.822 | -0.689 | -7.312 | -11.14 | -1.955, +0.023 |
| Keong / 1 | +1.402 | +0.479 | -0.918 | -6.522 | -9.63 | -0.629, +3.267 |
| Keong / 2 | +0.614 | +0.206 | -1.057 | -7.969 | -11.67 | +0.837, -3.302 |

Same-policy duplicate disagreement is often comparable with the whole-time
contrast. A fixed policy does not establish a repeatable whole-encoder win:
Keong is slower in all three completed quartets, despite lower token-worker
and tokenize-wall measurements. This does not prove an intrinsic Keong
regression either. Quantization and later phases move, and separately
calculated phase medians are not an additive decomposition of whole medians.
The interrupted run cannot prove cadence causality or complete S181's
broader qualification.

All 41 phase statistics, four candidate/control comparisons, duplicate
differences, and the four consecutive 16-sample block medians for every
completed sustained process are retained in `analysis_partial.json`.
Do not select favorable blocks or reconstruct the missing fourth pass.

The decision is `interrupted_unpromoted`. Resolve storage reliability and
headroom before any new independently pinned timing campaign. The bounded
appender remains a local experimental improvement, not a demonstrated
end-to-end gain. This stage makes no production, GPU, minimum-ISA or settings
change, and does not establish that the fully resident path is maxed out.

## Verification and archive

The partial verifier checks every accepted journal and its raw-log hash,
nonoverlap of accepted jobs, exact scheduled commands, all observed samples
and census values, the preserved zero-byte journal, and absence of all
fifteen unrun jobs. It regenerates the partial analysis and decision counts.
It also checks all pinned sources, support inputs, S181's unchanged 611-file
inventory, unchanged candidate DLLs and their eleven native CUDA modules.
The final inventory includes the failed journal and all additive recovery
helpers; prior artifacts and pinned full-campaign helpers are not overwritten.

Read-only frozen verification from the repository root:

```powershell
python build-cuda-ninja/profiles/s182_verify_partial.py --frozen
```

Successful verification means the interrupted evidence is intact, not that
the planned campaign completed. No material files were deleted. The report
is the only tracked change for this checkpoint.
