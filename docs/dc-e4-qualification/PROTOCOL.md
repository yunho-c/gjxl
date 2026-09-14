# Prospective e4 DC default qualification

Revision: 8e8c49a; clean unchanged source export, fresh Release, profiling off.
No encoder defaults are changed. CPU participants: 8. Fully resident Metal.

## Inputs and collection
- Timing: all nine pinned Unsplash 12/24/48MP PFMs at e4/e7, plus the three 12MP inputs at e3. Preserve original PFMs, including documented Lanczos overshoot.
- Visual: three 2048-wide photographic derivatives of the pinned 12MP inputs, resized in linear RGB and explicitly clipped to [0,1], plus four deterministic float gradient diagnostics (gray, dark gray, sky blue, red-green chroma). These synthetic tests are not photographic benchmark replacements.
- Visual matrix: all seven images at e3/e4/e7 and distances 1.2/3/6. Four modes: weighted ordinary rounding, quantize, smooth, both. Repeated native outputs must be byte-identical; independent float decode must be finite. Record SSIMULACRA2 and diagnostic gradient errors.
- Matched-size visual checks: for every visual image at e4/e7 and distances 3/6, calibrate only both against default bytes, within 1%, at most 10 total both observations including the initial point, distances bounded to [0.1,12]. Preserve unresolved targets without resetting budgets. This is matched-size evidence, not matched-metric-quality evidence.

## Timing
- Same-distance d1.2 measures the cost of changing the policy at unchanged user settings. It is not matched-quality timing.
- Each case: 3 warmups and 20 balanced AB/BA rounds of complete public encode calls, with loaded inputs and a shared production backend. Validation, file I/O, and diagnostic scoring are excluded. Output byte stability required.
- Three independent baseline-vs-baseline control invocations on alpine 12MP/e4 precede timing; each has 20 rounds and 3 warmups. All attempts retained; no favorable-run selection.
- Record power, thermal status, and process CPU snapshots. Do not overlap builds, encoders, decoders, or scorers. Flag significant background load/control variability rather than claim sub-percent precision.
- High-resolution size/quality collection independently scores both variants. Existing e7 results may only be cross-checked, never silently pooled as new measurements.

## Decision boundaries
- Correctness requires the relevant existing CPU/Metal tests to pass and all completed streams to decode deterministically to finite pixels.
- Visual review checks gradients, chroma striping, blocking, and obvious detail loss. Agent inspection and amplified-error diagnostics cannot substitute for a blinded human study on a calibrated display.
- Report rate, quality, complete-call cost, scene dependence, all missing/unresolved cases, and material limitations. No speed or rate acceptance threshold is fitted after seeing results. Default-policy recommendation will explicitly state the observed tradeoff; no default change is part of this task.
