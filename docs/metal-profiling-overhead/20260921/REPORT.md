# Production-aligned GPU profiling overhead

Commit: `3e1ef9e` (`perf/profile-production-alignment`), local and not merged to main.
Date: 2026-09-21. Device: Apple M4 Pro, 20 GPU cores, 48 GiB; macOS 15.6.

## Decision

See `DECISION.md` for the interpretation of these measurements.

## Follow-up: complete encode

Each comparison is a within-round difference or ratio; cells in brackets are
95% paired bootstrap intervals for the median (4,000 resamples, fixed seed).
Ordinary and Full columns are separate medians, so their ratio need not equal
the median paired ratio. Negative differences do not establish a production
speedup from profiling. The baseline is an ordinary complete resident encode,
with the same small diagnostic observation scopes used by every control.

| Image / effort | Ordinary (ms) | Full (ms) | Full minus ordinary (%) | Full minus host-profile (%) |
|---|---:|---:|---:|---:|
| 0.39 MP / e4 | 9.64 | 9.86 | +0.99 [-2.81, +8.91] | +2.02 [-2.51, +6.59] |
| 0.39 MP / e7 | 22.41 | 24.17 | +6.71 [+4.89, +8.64] | +5.95 [+5.05, +7.09] |
| 0.39 MP / e9 | 42.86 | 45.53 | +6.45 [+4.67, +7.73] | +5.03 [+3.66, +6.48] |
| 12.00 MP / e4 | 117.27 | 111.53 | -4.63 [-7.02, -1.83] | -4.35 [-7.30, -0.11] |
| 12.00 MP / e7 | 498.82 | 496.60 | -1.46 [-6.09, +4.43] | -1.08 [-8.74, +3.87] |
| 12.00 MP / e9 | 909.32 | 901.96 | -1.44 [-5.36, +1.71] | -3.26 [-4.97, +1.32] |

## #2: splitting compute encoders

`split − graph` holds the stage callback graph and command-buffer count fixed.
Both controls disable dispatch recording and timestamps. This isolates the
encoder boundaries much more directly than ordinary-versus-profiled timing.

| Image / effort | Encoders: graph → split | Complete encode (ms) | GPU command spans (ms) | Submission host time (ms) |
|---|---:|---:|---:|---:|
| 0.39 MP / e4 | 3 → 10 | +0.14 [-0.13, +0.43] | -0.00 [-0.20, +0.26] | +0.01 [+0.00, +0.01] |
| 0.39 MP / e7 | 4 → 150 | +0.18 [-0.13, +0.68] | +0.00 [-0.29, +0.34] | +0.03 [+0.03, +0.04] |
| 0.39 MP / e9 | 4 → 250 | -0.29 [-0.78, +0.10] | -0.37 [-0.68, +0.13] | +0.04 [+0.04, +0.07] |
| 12.00 MP / e4 | 3 → 10 | +2.48 [-1.78, +3.75] | -0.39 [-2.80, +2.37] | +0.00 [-0.00, +0.01] |
| 12.00 MP / e7 | 4 → 150 | -2.06 [-38.02, +17.10] | -4.18 [-39.84, +9.53] | +0.03 [+0.02, +0.06] |
| 12.00 MP / e9 | 4 → 250 | +9.08 [-34.26, +77.95] | +34.10 [-25.10, +84.49] | +0.06 [+0.02, +0.08] |

GPU command spans are `GPUEndTime − GPUStartTime`, summed across completed
command buffers, including input preparation. They are elapsed GPU spans,
not isolated active-kernel occupancy, and can still reflect scheduling noise.
The submission host interval includes encoding and commit, but not the
completion wait. It is elapsed host time, not OS-accounted CPU time.

## #3: recording and counter processing

All values below are milliseconds per complete encode. Metadata recording is
the paired `record − split` submission-time delta. Metadata postprocessing is
the direct duration of the no-counter `GpuProfile` path (profile copy, indirect
grid reads, and publication). Counter-buffer creation and counter resolution
are separately timed around the Metal calls. Other timestamp setup is the
paired submission-time delta `full − record`, less direct counter-buffer
creation; it includes timestamp attachments, different pass descriptors, and
associated driver work. Full postprocess contains counter resolution, profile
copy/validation, and publication. These columns overlap and must not be summed.

| Image / effort | Metadata recording Δ | Metadata postprocess | Counter-buffer creation | Other timestamp setup Δ | Counter resolution | Full postprocess |
|---|---:|---:|---:|---:|---:|---:|
| 0.39 MP / e4 | 0.009 | 0.006 | 0.050 | 0.029 | 0.010 | 0.016 |
| 0.39 MP / e7 | 0.100 | 0.060 | 0.044 | 0.422 | 0.019 | 0.084 |
| 0.39 MP / e9 | 0.159 | 0.095 | 0.052 | 0.740 | 0.022 | 0.124 |
| 12.00 MP / e4 | 0.018 | 0.014 | 0.055 | 0.031 | 1.104 | 1.118 |
| 12.00 MP / e7 | 0.110 | 0.076 | 0.083 | 0.651 | 0.036 | 0.113 |
| 12.00 MP / e9 | 0.244 | 0.146 | 0.090 | 1.296 | 0.038 | 0.188 |

## Protocol and controls

- Metal fully resident, distance 1.2, eight CPU threads, final score disabled.
  Release / AppleClang 17; production shaders, frontier experiments disabled.
- 20-case broad experiment: four inputs (128×96 synthetic, Kodak 768×512,
  CLIC 1507×2048, Unsplash 4249×2824), efforts 2/4/7/9/10. Six balanced
  blocks, two per-mode warmups and three measured calls per block.
- Six-case follow-up: Kodak and 12 MP, efforts 4/7/9; 36 rounds, one call
  per mode per round. Six Williams schedules balance position and first-order
  carryover; schedule rows are shuffled within each six-round replicate.
  Two initial warmups per mode. Both experiments also have eight initial
  untimed ordinary burn-in calls after the reference encode.
- Every call is checked against the ordinary reference codestream and complete
  encoding summary. Submission counts are equal across all six controls.
  Encoder counts agree for ordinary/host/graph, and for split/record/full.
- `ordinary`: ordinary workflow; no host or GPU profile.
- `host`: host workflow timers only; ordinary GPU encoders.
- `graph`: GPU-profiled stage callbacks, all in one encoder per original
  submission; no dispatch graph or timestamps; dummy profile for orchestration.
- `split`: the same callbacks, with one encoder per stage; no recording or
  timestamps; dummy profile for orchestration.
- `record`: normal split path, dispatch graph, retained indirect arguments,
  profile copy and grid resolution; no counter buffer or timestamp attachment.
  It retains small counter-capability/descriptor scaffolding from the full path.
- `full`: the committed stage profiler, plus measurement scopes.
- Image loading, backend/pipeline construction, output file I/O, output
  equality checking, JSON output, and destruction of returned outputs are
  outside the timed complete API call. Host profile/GPU profile construction
  and publication inside the API remain timed. This is not CLI startup timing.
- `record − split` approximates recording work; `full − record` includes both
  timestamp sampling and processing. They should not be described as pure
  GPU costs or as exactly additive end-to-end penalties.

## Interference and scope

The initial sweep overlapped background indexing and active desktop/remote
display work. The user then left the Mac idle for the follow-up. Background
system services and WindowServer remained active; no attempt was made to
terminate them. No competing codec/build/test job was detected during either
accepted collection, and the saved thermal snapshots reported no thermal or
performance warning. The follow-up is more informative but is not a controlled
headless/device-isolated performance qualification. Bootstrap intervals capture
observed run variability, not every source of systematic bias.

The experiment has one natural image in each size stratum, one target distance,
one device, and selected representative efforts. Do not extrapolate a universal
overhead percentage, claim production gains from negative deltas, or treat
profiled timestamps as an unperturbed production trace. Direct elapsed host
scopes can include preemption. No shader arithmetic or command-buffer
boundaries were changed by the controls, and the diagnostic and production
metallib hashes match.

## Artifacts and reproduction

- `ablation/identity.json`, `ablation/summary.json`: broad-run identity and all
  paired comparisons, including GPU/serializer/quantization and host metrics.
- `followup/identity.json`, `followup/summary.json`: final alternating follow-up.
- Each case retains raw JSONL (including warmups), command, exit status,
  interference log, stderr, reference codestream, and SHA256 completion record.
- `verification.json`: output equality, submission/encoder checks, raw hashes,
  and cross-build codestream equality for the six shared cases.
- `probe.cpp`, `run.py`, `patch_ablation.py`, `ablation.patch`, and the retained
  `source/src/gpu/metal/overhead_control.h` reproduce the first controls.
  `source/` is the commit archive plus that diagnostic-only patch/header.
- `followup-code/` retains the exact second probe, replacement Metal source,
  header, archive, binary, driver, job list, build commands, and SHA256 manifest.
  `prepare_followup.py` produces these from the frozen first diagnostic build.
- `analyze.py`, `analyze_followup.py`, `make_report.py` reproduce analysis.
- `environment.txt`, `background-activity.json`, and follow-up environment
  snapshots preserve machine/thermal/activity context. `pilot/` is exploratory
  original-library evidence and is not the basis for the decision.
- All diagnostic switches and measurements live under this ignored build
  directory. Production tracked files remain unchanged after the commit.

From the worktree, rerun the frozen follow-up with a fresh collection name:

```sh
python3 build/profiling-overhead-20260921/followup-code/run.py \
  --name followup-repeat \
  --probe build/profiling-overhead-20260921/followup-code/probe \
  --jobs build/profiling-overhead-20260921/followup-code/jobs.json \
  --modes ordinary,host,graph,split,record,full --pairs 36 --samples 1 --warmups 2
python3 build/profiling-overhead-20260921/analyze_followup.py followup-repeat
```
