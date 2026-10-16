# CPU property parallelism within shared-tree fitting

This change extends the learned Modular CPU policy to evaluate independent
properties concurrently inside a shared/global histogram fit. It builds on
`ebfa2d5` (fused occupancy/histogram scans and inherited constant properties).
Local prepare-and-fit scheduling is unchanged. Automatic routing still uses
shared trees at eight groups or fewer; explicit buffered learning also uses
this implementation. Sampling, quantization and partitioning remain serial.

## Execution and deterministic decisions

`EvaluateProperty` retains the original predictor loop, bin loop, histogram
arithmetic, floating-point entropy estimates and strict comparison rules. Each
property returns four category winners: static with a constant side, static,
nonstatic, and nonstatic without weighted prediction. The calling thread
reduces these results in ascending property order, then applies the original
category preferences. Equal-cost splits therefore retain the same property,
bin and predictor. Forced multiplier splits bypass ordinary property search.

One `PropertyTeam` lives for the fit. It uses the existing execution-domain and
per-image worker grants and includes the caller as a participant. Each worker
owns reusable histogram/occupancy/cost scratch. Sample columns and parent
histograms are read-only during dispatch; the caller waits for all property
tasks before changing sample order, node storage or the inherited constant
mask. Small nodes run on the caller without waking the team.

The crossover is **8,192 distinct rows times predictor count**, with
at least two potentially varying properties. Team size is bounded by the
requested allowance, the number of properties with multiple quantization bins,
the encode thread limit and both CPU budgets. No nested team starts inside an
explicit parallel scope, or without CPU participation. Local fits explicitly
retain a one-participant fitter even when their scheduler falls back to serial.

`ReferenceWorkflowOptions::learning_cpu_participants` controls either local
prepare-and-fit jobs or shared-tree property evaluation. The public learned
policy already passes its CPU allowance through this option. No public API or
bitstream format changes are needed. Private `LearnHistogramTree` calls remain
serial unless their new `cpu_participants` option requests otherwise.

## Memory, waiting and failures

The shape-based bound includes every additional property workspace and thread
handle. Parent histograms and sample columns are shared without copies. All
managed backing, result storage and worker launches complete before the first
row partition. The workflow retries failed memory admission with half as many
participants, down to the existing serial route. It never retries an allocation,
launch or fitting failure after admission.

Workers sleep on a condition variable between nodes with suspended CPU permits;
they retain protected capacity without counting as active participants. Caller
waits suspend its global permit and release the mutex before resuming. A
partially launched team is stopped and joined before borrowed features,
reservations or scratch can be released. Tree/frame/encoded output publication
retains its existing failure atomicity.

## Diagnostics

Histogram profile schema 3 adds `peak_property_participants` and
`parallel_property_nodes`. Work counts and distributions retain the serial
meaning. Fit and node timers measure elapsed spans; property stage timers sum
individual task wall intervals and can overlap within a fit. They must not be
summed or subtracted from elapsed fitting time to infer idle/control overhead.
The reader accepts schemas 1–3. The exclusive-phase report explicitly rejects
within-fit parallel captures.

For a same-binary shared-fit scaling check, hold the other CPU allowances fixed:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT cpu auto best fit-off 4 14 14 14 14
```

Change the first `4` to `1`, `2`, `8` or `14`; use `fit-on` in place of
`fit-off` for separate detailed diagnostics. The driver verifies independent
lossless decoding and complete release of managed storage. Full public-route
measurements use `gjxl_modular_metal_workflow_profile INPUT.pam OUT 3 encode-public 14`.

## Qualification

The final source passes all seven targeted Release suites: reference workflow,
parallel learning, histogram learning/preparation, hybrid workflow, hybrid
palette and public API. Tests include Best/All predictors, prefix/ANS byte
parity, independent decoding, CPU ceilings, memory-driven concurrency reduction,
unaccounted/nested serial fallback, shared execution domains, partial worker
launches and unchanged output after failures.

Native tests compare final sample-column order as well as exact trees and all
work/distribution counters at 2/4/8/automatic participants. Cases include sparse
bins, tied properties, 1/2/14 predictors, inherited constants and forced
multiplier splits with an empty child. An allocation sweep reaches success
only after every allocation boundary has been checked for failure atomicity.
The pinned libjxl differential suite passes all 257 cases. The profile reader
validates 36 distinct historical/current captures across schemas 1–3; captured
source hashes still match the final source files.

The final native fitter passes AddressSanitizer/UndefinedBehaviorSanitizer.
ThreadSanitizer passes both the native fitter and the workflow's
`--preparation-only` mode, including concurrent images and partial launches,
with no reported races. Complete encoding and independent decoding are covered
by Release tests and captures; the focused TSan workflow mode omits entropy
encoding. `ASAN_OPTIONS=detect_leaks=0` is used for the native ASan run.

### Matched complete-call measurements

Apple M4 Pro, 14 CPU participants, 48 GiB, AC power, public learned policy,
Best predictors and estimated ANS. Baseline is `ebfa2d5`. Each input/binary runs
in a fresh process: one discarded cold call and one warmup per CPU/Metal route,
then three measured rounds with rotating route order. Before/after binary
order reverses across inputs. Detailed fitter probes are disabled. Encoder
elapsed time includes admission, selection, preparation, fitting, tokenization,
modeling, emission, cleanup and publication; file I/O, backend construction and
independent decoding are outside the timer.

Seconds below are medians. Negative reductions mean longer elapsed time.

| Input | CPU before | CPU after | Reduction | Metal before | Metal after | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| photo-macan | 0.4244 | 0.2941 | 30.7% | 0.4242 | 0.2816 | 33.6% |
| alpha-ria | 0.6041 | 0.3624 | 40.0% | 0.5908 | 0.3524 | 40.4% |
| alpha16-ramp | 0.2641 | 0.1711 | 35.2% | 0.2577 | 0.1513 | 41.3% |
| palette-358 | 0.0097 | 0.0089 | 8.4% | 0.0098 | 0.0090 | 8.2% |
| photo-large-1029 | 4.7337 | 4.8065 | -1.5% | 5.1601 | 5.2783 | -2.3% |

The three target shared-tree cases improve by **30.7–40.0% on CPU** and
**33.6–41.3% with explicit Metal**. The small palette case improves by less than
one millisecond. The large local-tree control has higher medians by 1.5% on CPU
and 2.3% on Metal; it is not an improvement claim for local trees. Its CPU
ranges overlap (4.725–4.737 s before, 4.715–4.938 s after), as do the Metal ranges
(5.082–5.355 s before, 5.234–5.375 s after). Three repetitions on an active
desktop do not resolve a change of that size confidently.

The main capture contains **169 complete encodes**: 100 public-route calls,
60 same-binary scaling calls and nine detailed diagnostics. Every output
retains exact baseline bytes and independently decoded samples. Diagnostic
work signatures, including learned node counts and histogram/bin/candidate
counts, match the baseline and the candidate's serial control.

### Same-binary fitting scalability

The following separate CPU capture varies only learning participants; all
other allowances stay at 14. Each configuration has one warmup and three
measured calls in rotating order. Values are **fitting seconds**, excluding
sample preparation, from the candidate with detailed probes disabled.

| Input | 1 participant | 2 | 4 | 8 | 14 | Fitting reduction, 1 to 14 |
|---|---:|---:|---:|---:|---:|---:|
| photo-macan | 0.2808 | 0.2072 | 0.1706 | 0.1537 | 0.1511 | 46.2% |
| alpha-ria | 0.3764 | 0.2628 | 0.2104 | 0.1860 | 0.1763 | 53.2% |
| alpha16-ramp | 0.1596 | 0.0998 | 0.0719 | 0.0554 | 0.0549 | 65.6% |

Separate profiled runs dispatch 185 of 879 nodes for `photo-macan`, 240 of 821
for `alpha-ria`, and 62 of 477 for `alpha16-ramp`, with 14 admitted property
participants. The other nodes remain on the caller. Sampling/preparation,
parent histograms, split selection and partitioning still limit whole-call
scaling. The 8,192 crossover was selected after exploratory 32,768-versus-8,192
pilots; it is a bounded heuristic, not a claim of an optimal universal cutoff.
Those pilot timings are excluded from the tables and encode total above.

Each additional participant requires admitted private scratch. Whole-call
managed committed peaks are unchanged in this set because preparation or other
stages dominate them. That does not imply free scratch: CPU peak live backing
rises from 16.05 to 18.26 MB for `alpha16-ramp`, and from 1.20 to 4.51 MB for the
palette case. CPU backing peaks for `photo-macan` and `alpha-ria` remain 24.27
and 31.31 MB respectively. Managed measurements exclude worker stacks,
allocator bookkeeping, borrowed input, oracle decoding and backend storage.
Memory admission can reduce the property team without changing learned output.

### Evidence and limits

[Raw results](parallel-shared-tree-fitting-results.json) retain every measured
and warmup row, input/binary/dependency/source hashes, process/power snapshots,
commands, diagnostics and validation logs. Full local artifacts, frozen
executables, source archive, qualification script and encoded outputs are in
`build/shared-tree-parallel-20261015`.

A first timing capture overlapped sanitizer/codec tests because macOS truncated
a nonfinal `ps comm` column. It was stopped, retained under `discarded-overlap`,
and excluded in full. The corrected runner reads the full command column
separately, waits for other codec/build/test processes, and checks again after
each call. No such processes were observed at the final capture boundaries.
Desktop services remained active; these are five-image, three-repetition
results, not an isolated-machine or general-corpus claim.
