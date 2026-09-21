# Overhead decision

The alignment is committed locally as `3e1ef9e` on
`perf/profile-production-alignment`. The measurement controls are retained as
build artifacts and have not changed the production implementation.

**Defer a redesign for #2. For #3, prioritize timestamp setup/sampling if we
continue reducing perturbation; generic metadata recording and copying are
lower priority.** This is a prioritization from a bounded experiment, not a
claim that stage profiling is free or that large-image overhead is ruled out.

## Observed complete-call penalty

The alternating follow-up found clear overhead on the 0.39 MP Kodak image:

| Effort | Full profile vs ordinary | Full profile vs host timers only |
|---|---:|---:|
| 7 | +1.47 ms / +6.71% (95% interval: +4.89% to +8.64%) | +1.37 ms / +5.95% |
| 9 | +2.73 ms / +6.45% (95% interval: +4.67% to +7.73%) | +2.26 ms / +5.03% |

These are medians of paired differences/ratios, not ratios of separately
aggregated medians. Effort 4 on Kodak was inconclusive. The 12 MP results remain
too variable for a trustworthy general overhead estimate, even after the user
left the Mac idle. Some differences are negative, including a negative interval
at e4; these do not establish that adding instrumentation accelerates production.
GPU scheduling, background services, host scheduling, and instrumentation-induced
scheduling changes are not fully separated by this experiment.

## #2: one compute encoder per stage

The diagnostic `graph` and `split` variants run identical stage callbacks with
the same command buffers, without timestamps or dispatch recording. On efforts
7 and 9, they use 4 versus 150/250 compute encoders per complete encode.

- Added host submission time was approximately **0.03–0.06 ms** at e7/e9.
- Every follow-up confidence interval for the complete-call and GPU-span
  `split − graph` difference included zero.
- On Kodak, the complete-call differences were +0.18 ms at e7 (interval
  −0.13 to +0.68 ms) and −0.29 ms at e9 (−0.78 to +0.10 ms).
- The 12 MP e9 result is weakly constrained: the interval still permits
  approximately **+8% complete-call / +14% GPU-span** cost. Absence of a resolved
  penalty here is not evidence that a large-image penalty cannot exist.

There is no consistent demonstrated cost that justifies changing the timestamp
boundary mechanism immediately. If the paper needs a tight bound on large-image
perturbation, repeat this preserved control on a more isolated system before
declaring #2 negligible or spending effort on a replacement.

## #3: metadata versus timestamp machinery

The finer host scopes distinguish work that the original single bucket hid:

- **Metadata recording:** approximately 0.01–0.24 ms added submission time.
- **Metadata copying/indirect-grid resolution:** approximately 0.006–0.15 ms
  in the no-counter postprocessing path.
- **Counter-buffer creation:** approximately 0.04–0.09 ms.
- **Other timestamp setup:** approximately 0.42–1.30 ms at e7/e9, after
  subtracting counter-buffer creation from the `full − record` submission
  delta. This includes the timestamp attachment/pass-descriptor encoder path
  and associated Metal driver work; the experiment does not isolate each API
  call within that bucket.
- **Counter resolution itself:** approximately 0.01–0.04 ms in five follow-up
  cases, but **1.10 ms median at 12 MP/e4**. A direct timer around
  `resolveCounterRange` confirms this is the counter-resolution call, rather
  than copying the graph. Its underlying driver/scheduling cause is unresolved.

The `full − record` complete-call delta on Kodak was +1.04 ms at e7 (interval
+0.45 to +1.54 ms) and +1.65 ms at e9 (+1.04 to +2.29 ms), larger and more
consistent than the recording-only delta. This comparison includes timestamp
sampling effects on GPU execution as well as host setup/resolution.

If we optimize next, a narrow investigation of pass-descriptor/attachment setup
and its allocation/reuse opportunities has the clearest measured motivation.
It still needs an A/B validation; these measurements do not promise that caching
descriptors eliminates the observed cost. Dropping all dispatch metadata or
redesigning the profiler's data structures has less demonstrated upside.

## Implication for results figures

Use ordinary complete-call measurements for production throughput, and label
the stage breakdown as instrumented. Do not subtract summed GPU stages from
ordinary wall time to estimate CPU work, normalize away the measured overhead
without explanation, or apply the Kodak percentage to every size/effort.
This experiment collected overhead metrics; it is not a new full corpus of
per-stage notebook data.

The 20-case sweep and six-case follow-up verified **3,456 measured encodes**,
plus all logged warmups, against ordinary reference bytes and encoding summaries.
Command-buffer counts matched across all controls. The six shared cases also
have identical output hashes across the two diagnostic builds. See
[REPORT.md](REPORT.md), [verification.json](verification.json), and the retained
raw JSONL/identity/build files for measurements, caveats, and reproduction.
