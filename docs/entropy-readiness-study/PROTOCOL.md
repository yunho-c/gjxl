# Earlier AC completion and entropy readiness

This is an isolated opt-in experiment at `d71eeb034f1f1d7d8b1aee2775ed9fe18bb24e98`,
with the already audited Xcode 27 compatibility changes. The original checkout
and all of its pre-existing changes remain untouched. `source/` is a source
snapshot, not a Git worktree. There is no production default change.

## Interventions

The single experimental executable accepts `GJXL_EXPERIMENT_EAGER_ENTROPY`:

- **0:** Original dependencies. The entropy-task body is factored into a shared
  helper, so all modes use exactly the same model-building algorithms.
- **1:** After the existing GPU token submission, run CPU DC preparation and
  AC Finish concurrently. Entropy models still start at their common barrier.
- **2:** The DC branch builds its entropy model as soon as DC preparation
  finishes. The other branch builds the order model, waits for GPU completion,
  reduces/publishes populations, then builds the AC model. Writing and candidate
  selection remain at the original join. No earlier DC launch, context/order
  fork, shader change, reduction algorithm change, or writing overlap is tested.

Both parallel modes reserve one extra participant nonblockingly. No available
slot means the original schedule. DC retains `workers - 1` as its local ceiling;
the other branch has one. Resetting nesting depth with disjoint ceilings keeps
DC group parallelism and the existing shared CPU domain. All workers join before
return, error propagation, output publication, or provider destruction.

Mode 2 is initially limited to the ordinary non-rate path. High-effort dual
representations retain their original entropy schedule; mode 1 can still test
earlier Finish there. CPU-only tokens, CPU=1, exhaustive search, explicit outer
parallel scopes, and native additive profiling retain the original schedule.
The independent trace sink does not request native profiling and measures the
experimental dependencies directly. `eager.selected` records actual activation;
requested settings alone are not activation evidence.

## Comparison

The harness loads one image outside timing, obtains a mode-0 reference, and
alternates modes 0/1/2 within every sample block using six counterbalanced
permutations. Every result is compared byte-for-byte and by encoding summary.
CPU protected-slot limits and zero live participants/reservations are checked
after every call. Returned output destruction is outside the synchronous public
API timing. Internal cleanup remains inside it.

Primary results use tracing disabled, with four independent processes and 18
measured blocks per configuration after three warmup blocks. Trace, sensitivity,
pilot and forced-retry cohorts remain separate. Complete process commands,
environment overrides, hashes, power, VM and process snapshots are retained.
Inherited `GJXL_*` settings are cleared. Captures are serial; builds and tests do
not run concurrently with measurement. No system power settings are changed.

Report the median same-block latency reduction, p10/p90, and the range of
per-process median percentage reductions. Correlated samples within a process
are not independent replications. The primary comparison isolates a schedule
change in the same executable; it does not establish the cost of compiled-out
instrumentation or a production implementation with complete native profiling.

Batch requests repeat one image four times under a 36 GiB domain. Report public
batch makespan directly, not a sum of per-image modeled savings. Actual admission
and fallback can vary with the shared CPU domain. Capacity retries are forced
in a separate cohort. The 48 MP case is excluded because the previous study did
not establish a stable pressure-free baseline there.

## Correctness and preservation

- Full 160-test Release suite with mode 2 requested.
- 84 additional contract cases cover byte equality, CPU 1/2/4/8, completion and
  allocation errors, worker-launch errors, admission fallback, native-profile
  fallback, no partial publication, and no leaked participants.
- All measured results equal their mode-0 reference. References also compare
  across processes, CPU limits, batch sizes and token paths for the same input
  and coding options; independently decode each distinct reference.
- Source/binary hashes, exact experiment diff and identical metallib hash are
  retained. Check original changed-file hashes, HEAD, status and index at closure.

The pilot and complete primary matrix were recorded on battery with `pmset`
reporting `powermode 1` for battery and `powermode 0` for AC. The user chose to
continue on battery. The machine later switched to AC during trace follow-up;
this was not an agent power-setting change. A separate `ac_confirm` matrix and
`ac_trace` cohort were added. Power condition is part of every analysis group;
transition processes remain separate and do not support steady-condition claims.
Absolute timings must not be compared to the prior study as a speedup estimate.
