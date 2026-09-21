# Paired resident GPU stage captures

`capture.cpp` collects ordinary and profiled complete encode-call timings using
one loaded image and one reused Metal backend. It retains host and GPU profiles
from the **same profiled call**, along with exact output/summary/submission-count
equivalence checks. It reuses benchmark parsing, backend policy and the GPU JSON
serializer by including `benchmarks/encoding_benchmark.cpp`; encoder code is
unchanged. This is diagnostic tooling, not a production API.

The completed six-image, nominal Q80 (distance 1.9), E1–E10 study is at:

```
/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-stages-q80-20260921
```

Its `build-command.json` records the exact compiler/link arguments against the
Release static libraries. `config.json` freezes the input/job grid, source
revision, libraries, embedded metallib, capture executable and collector source
hashes. Use a new directory and freshly frozen configuration for another study;
do not overwrite a completed run or mix different builds into its ledger.

After the CUDA-main integration, the live harness uses `gpu_aq_mode`; the
archived harness still uses the earlier `metal_aq_mode` API at its recorded
source revision. Build the live harness against current libraries and keep
the archived executable for reproducing the original study.

The benchmark-style capture command requires an even `--samples` count and:

```sh
"$CAPTURE" --input "$INPUT" --scope metal-public-workflow --validation metal-only \
  --gpu-aq fully-resident --effort "$EFFORT" --distance 1.9 --cpu-threads 8 \
  --warmups 2 --samples 6 --gpu-profile stage --gpu-profile-output "$ATTEMPT/gpu.json"
```

This writes `gpu.json`, `paired.json` and `reference.jxl`. Two warmups per mode
and six alternating AB/BA measured pairs follow an ordinary reference encode.
Timed calls include everything before return; loading, backend creation,
validation, file writes and destruction of returned values are outside timing.
The ordinary call requests no profiling. Profiling still changes encoder
boundaries and adds counter work despite retaining production submission policy.

`collect.py --run RUN` executes/resumes a frozen configuration. It verifies
hashes, uses an exclusive PID lock, retains numbered attempts, rejects competing
codec/build/test processes, and marks a case complete only after `validate.py`
accepts metadata, sample coverage, GPU timestamps and additive residuals. A
crash leaves a lock: inspect the recorded PID before removing a stale lock.
Resume only one collector at a time. Completed cases are revalidated, not rerun.

For independence from the terminal, use a launchd plist with `RunAtLoad=true`
and `KeepAlive=false`, optionally through `caffeinate -i`. `launchctl submit`
can automatically restart a completed job; the retained study records its
explicit service removal in `collector-lifecycle.json`.

Plotting is separate and never starts collection. In the libjxl worktree,
`tools/scripts/cjxl_gjxl_paired_breakdown.py` reads this configuration and produces
flat complete-call partitions, ordinary-total markers and optional estimated
scaled allocations. See `doc/runtime-gjxl-profile.md` there for exact boundaries.
