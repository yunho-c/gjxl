# Fused resident AQ initialization

The ordinary, unprofiled resident Butteraugli policy now performs strategy-aware
quant-field adjustment and initial bound construction inside its existing AQ
submission. It avoids the separate adjustment submission, host wait, field
readback, CPU bounds scan and adjusted-field re-upload. The public standalone
adjustment operation remains available. Maximum-error, exact-coefficient and
profiled execution retain their established paths.

The new optional preparation capability accepts an unadjusted initial field.
After the existing per-family adjustment kernels, a capped 64-threadgroup
reduction computes positive finite extrema, followed by a one-thread bounds
calculation. Policy updates consume these device bounds. Two floats add 256 bytes
to the existing staging arena after alignment; there is no additional device
allocation. Admission plans include this storage. The unused host adjusted-field
allocation is skipped in the fused path.

Adjustment/bounds error flags survive the first reconstruction reset, including
zero-update frame-only execution. Invalid device results invalidate the prepared
evaluation and leave caller outputs unpublished. The device calculation applies
the CPU policy's finite/range checks. Metal's `precise::divide` and
`precise::sqrt` are deliberate: the first direct test found a one-bit lower-bound
difference with ordinary division even though the initial image pilot matched.

## Correctness evidence

- 256 deterministic uniform/random quant-field cases compare the fused path with
  the separate Metal adjustment plus CPU bounds and resident policy. Targets
  0.7, 2.4 and 7, seven field scales, 0–4 updates, optional final evaluation,
  strided inputs and reused preparations are covered. Bounds, output quant
  fields, scores and quantized coefficients match exactly in these cases. Each
  fused call uses one submission.
- Upload, submission, completion, numeric and readback failures are tested for
  iterative and zero-update completed-frame output, with output atomicity and
  invalidated reuse checks. A real out-of-range bounds failure is also covered.
- Normal and Metal API/shader-validation AQ evaluation tests pass. Existing AQ
  reconstruction, storage layout and host storage checks pass. The frozen
  storage oracle is retained, with an explicit 256-byte resident-bounds delta.
- Complete resident pipeline and standalone AQ tests confirm one fewer
  submission (4 versus 5, and 2 versus 3 respectively). Resident/compatibility
  workflow storage plans and workflow admission tests also pass.
- All 60 retained CPU-greedy pilot codestreams remain byte identical with the
  corrected precise-math implementation: six images, efforts 5/8, distances
  0.7, 1.2, 2, 4 and 7.

The 60-case artifact is
`build/frontier/results/fused-aq-encode-parity-20260918-v2/`; the earlier directory
without `-v2` predates the precise-division correction and is superseded.
This is tested parity, not a proof of bitwise equivalence for every binary32
field or every Metal implementation.

## Complete-call timing

Retained run: `build/frontier/results/fused-aq-timing-20260918/`. It freezes both
binaries, source snapshots/patches and inputs. The control is the previously
qualified native-selection binary; its experiment override supplies the CPU
selector arm. The candidate adds fused AQ initialization. Every arm/process
output hash matches for each image/effort. Portable summaries and hashes are in
[evidence/](evidence/).

Apple M4 Pro, distance 1.2, eight CPU threads, no profiling. Three processes per
arm/case, each with three warmups and seven retained samples; case and arm order
rotate. Values are medians of process medians. Runs were sequential on the
interactive host, with no concurrent agent build or GPU measurement job.

| Input | Effort | CPU selector | GPU selector | GPU selector + fused initialization | Combined speedup |
|---|---:|---:|---:|---:|---:|
| Alpine 3MP | 5 | 59.892 ms | 58.137 ms | 58.147 ms | 1.030× |
| Alpine 3MP | 8 | 115.327 ms | 114.441 ms | 114.695 ms | 1.006× |
| Alpine 24MP | 5 | 385.570 ms | 374.815 ms | 373.913 ms | 1.031× |
| Alpine 24MP | 8 | 731.471 ms | 723.605 ms | 724.359 ms | 1.010× |
| Forest 24MP | 5 | 419.773 ms | 411.441 ms | 406.200 ms | 1.033× |
| Forest 24MP | 8 | 845.430 ms | 829.577 ms | 827.629 ms | 1.022× |

The combined speedup is modest and mostly due to device selection. Initialization
fusion alone ranges from 0.998× to 1.013×, with most differences small relative to
the process variation. It removes a known boundary and prepares the remaining
resident handoff, but this run does not establish a consistent standalone latency
benefit. These are engineering controls, not final paper/corpus timings.

The ACS map readback and CPU transform metadata construction remain. Their
removal is scoped in [HANDOFF.md](HANDOFF.md). Profiled calls still execute the
older selector/initialization paths and must not be used to attribute the new
path's critical time. Fresh paper studies remain pending the final combined
implementation and qualification.
