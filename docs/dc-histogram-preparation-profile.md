# DC histogram preparation profile

DC/metadata entropy preparation repeats a substantial token scan, but it is
usually hidden behind AC entropy work in the measured single-image workflow.
Do not prioritize collecting DC populations during tokenization for latency:
that would move work from a parallel entropy task into an earlier serial phase.
Profile the remaining AC entropy task first. A narrower checked-conversion change
in DC coder selection is a plausible CPU-efficiency experiment, requiring batch
throughput measurements before claiming practical benefit.

This is a diagnostic-only study of `aa5025c` on `perf/entropy-bit-writing`, dated
2026-09-10. No encoder implementation change or optimization speedup is delivered.

## Source findings

- [`SelectOrdinaryEntropyCodingMode`](../src/codestream/encoder.cpp) scans every
  token to count tokens and determine whether each context has a single symbol.
  It validates stream/context inputs and calls the external, checked
  `EncodeHybridUint` for every value. That wrapper checks the configuration and
  output pointer, converts the value, and constructs a success `Status`.
- [`AddDirectAnsTokenHistograms`](../src/codestream/ans.cpp) scans the same DC and
  metadata streams again. It already uses the inline, validated HybridUint
  helper. Each histogram update checks symbol/count/extra-bit bounds, updates
  the selected bin and total/extra-bit counters, and tracks the maximum symbol.
- `PrepareDirectAnsPartition` allocates 45 DC-context histograms and then scans
  and clusters them. Allocation is only a few microseconds. The ordinary path
  selected ANS for every measured input.
- [`TokenizeSimpleDcGroupsForEncoder`](../src/codestream/dc_group.cpp) processes DC groups in
  sequential nested loops. In `encoder.cpp`, DC tokenization precedes AC
  preparation/tokenization; DC, coefficient-order and AC entropy optimization
  subsequently run through `RunParallelSections`. The existing population-based
  selector can avoid a token scan, but producer-side population collection would
  change which phase pays for the work.

## Live workflow measurements

Apple M4 Pro, 48 GiB, macOS 15.6, AppleClang 17, Release C++20. Fully-resident Metal,
effort 7, distance 1.2, automatic CPU threads, `metal-public-workflow` timing,
`metal-only` validation, no GPU-stage profiling. The public workflow excludes
process startup, initialization and file I/O.

Three alternating baseline/diagnostic independent-process pairs per input,
three warmups and seven timed encodes per process. The additional reference call
and warmups are excluded from diagnostic summaries. Values below are medians of
process medians; component medians need not add exactly. Histogram preparation
includes allocation, traversal and clustering; the two scan columns isolate the
DC and metadata stream traversals. These are CPU durations inside the DC task,
not independent contributions to complete-encode latency.

| Input | DC / metadata tokens | Coder selection (ms) | DC scan (ms) | Metadata scan (ms) | Histogram preparation (ms) | Baseline complete workflow (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 18,432 / 13,898 | 0.086 | 0.046 | 0.047 | 0.101 | 19.351 |
| Planter 1080p | 97,200 / 90,728 | 0.452 | 0.253 | 0.305 | 0.569 | 65.409 |
| Planter 4K | 388,800 / 347,462 | 1.739 | 1.040 | 1.165 | 2.222 | 204.328 |
| Padded 4K | 388,800 / 176,866 | 1.437 | 1.020 | 0.604 | 1.648 | 210.826 |

Allocation accounts for approximately 1.1, 3.0, 4.5 and 3.8 microseconds,
respectively. Planter 4K spends about 3.96 ms of CPU work in selection plus
histogram preparation; padded 4K spends about 3.09 ms. Neither number is an
available complete-workflow latency saving.

The diagnostic build adds clocks and logging. Median paired complete-workflow
changes relative to baseline were +0.38%, +0.38%, +0.03% and -0.98%, respectively.
These small cohorts characterize observer effects and run variability, not an
optimization. Histogram-preparation timing includes small logging overhead;
section-scan clocks exclude capture and logging. Fixture capture occurs only on
the first reference call. CLI parity logs are not used for timing.

## Is DC on the critical path?

A separate diagnostic build timestamps entry and exit of each parallel entropy
task. Three processes per input, three warmups and seven measured samples give
21 timelines per input. Each measured encode has one DC, one order and one AC
task. Finish times are relative to the start of the entropy phase.

| Input | DC finish (ms) | AC finish (ms) | Median DC slack behind last other task (ms) | DC finishes last |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 0.662 | 2.231 | 1.563 | 0 / 21 |
| Planter 1080p | 2.081 | 6.861 | 4.778 | 0 / 21 |
| Planter 4K | 5.101 | 6.507 | 1.413 | 1 / 21 |
| Padded 4K | 3.804 | 6.119 | 2.374 | 0 / 21 |

DC finishes before the last other task in 83/84 samples. In the single exception,
DC extends the task completion boundary by only 0.060 ms. Holding other task
completion times fixed, shortening DC alone therefore has a zero median direct
latency benefit in this cohort. This is a scheduling observation, not proof that
DC optimization can never help: less contention, different content, CPU thread
limits and concurrent image batches can change the result.

Baseline DC tokenization itself takes 3.594 ms on Planter 4K and 2.603 ms on
padded 4K. Adding histogram updates to this earlier serial phase could consume
latency that is currently overlapped with AC entropy work. Producer fusion needs
an explicit scheduling argument and complete-workflow evidence.

## Captured-token replay

The diagnostic build captures the original alternating DC/metadata streams,
45-context extent and HybridUint configuration `(4, 2, 0)`. The two 4K inputs have
four DC groups and eight streams; the smaller inputs have two streams. An
uninstrumented Release replay uses the frozen original histogram function.

Five independent processes per input/mode, alternating forward/reverse mode order,
five warmups and 51 measured calls per process: 160 processes and 8,160 measured
calls. I/O, preencoding and result hashing are outside the timer. Buffers are warm.
Values are medians of process medians.

| Input | Original histogram scan (ms) | Preconverted histogram scan (ms) | Selection (ms) | Walk probe (ms) | Conversion probe (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 0.0833 | 0.0795 | 0.0753 | 0.0093 | 0.0277 |
| Planter 1080p | 0.5223 | 0.5169 | 0.4355 | 0.0522 | 0.1580 |
| Planter 4K | 2.0861 | 2.0303 | 1.6986 | 0.2139 | 0.6266 |
| Padded 4K | 1.5155 | 1.4911 | 1.4075 | 0.1635 | 0.4813 |

The preconverted probe uses an eight-byte context/symbol/extra-bit record, the
same size as the original context/value token, and calls the original histogram
update. It changes both representation and traversal code. It is a counterfactual
cost probe, not an implemented producer pipeline or an exclusive conversion timer.
Its roughly 2.7% Planter 4K and 1.6% padded 4K scan improvement gives little reason
to introduce a separately stored preencoded token array merely for histogram
construction. It does not measure the benefit of eliminating the whole scan.

The walk probe reads tokens, validates views/contexts and accumulates a checksum;
the conversion probe additionally runs inline validated HybridUint conversion.
Their durations cannot be subtracted into an additive cost breakdown: generated
code, dependencies and memory access differ. A separate allocation/replacement
probe takes about 1.3 microseconds and includes releasing the previous owner.

On Planter 4K, 80.2% of DC values and 99.8% of metadata values are below 16.
Metadata updates concentrate heavily in contexts 0 and 10. Together with the
preconverted probe, this suggests histogram-update dependencies and traversal
matter more than expensive large-integer conversion. No hardware-counter evidence
isolates a particular stall mechanism, and validation has not been assigned an
exclusive percentage.

## Sampling and next experiments

Three-second macOS `sample` profiles of long Planter 4K replays confirm the
histogram loop is the hot function in histogram replay. In selection replay,
1,806 of 2,426 selector-subtree samples have `EncodeHybridUint` at the top of the
stack (about 74%). That wrapper includes validation, conversion, call overhead
and success-status construction; this is not a measurement of validation alone.
Optimized/inlined histogram source-line attribution is not an exact breakdown.
Sampling runs are separate from all reported timing cohorts.

Recommended order:

1. Profile the remaining AC entropy task, particularly model construction and
   residual clustering work, because it determines completion in this cohort.
2. If pursuing DC CPU efficiency, first test hoisting immutable configuration
   validation and using the existing validated conversion helper in ordinary
   selection. Preserve invalid/empty-stream behavior, error precedence and full
   token/context validation; an early return once ANS is inevitable is not
   automatically equivalent. Gate on exact output and concurrent batch throughput.
3. Consider combining selection and histogram collection inside the existing
   entropy task, retaining populations when ANS is chosen. Compare against the
   narrow selector change before adding ownership/storage complexity. Preserve
   Prefix selection and other entropy-policy paths.
4. Revisit population collection during tokenization only with a demonstrated
   scheduling benefit. A serial producer is a material constraint, even when
   fusing passes reduces aggregate CPU work.

No candidate speedup, SIMD benefit or batch-throughput improvement is established
by this diagnostic study.

## Validation and retained evidence

The baseline and diagnostic encoders produce byte-identical JXL files for Kodak17,
Planter 1080p and Planter 4K. All four benchmark workloads pass their internal
repeat validation. Padded 4K has no separate cross-build file comparison here.
Replay hashes cover every histogram bin, total count, extra-bit count and maximum
symbol; original and preconverted results match. This does not qualify a future
optimization or replace its tests. No new full CTest suite or decoded-quality
sweep was run for this profiling-only change.

Local artifacts are retained under `build/dc-histogram-profile/`:

- Frozen `source/`, diagnostic source copies, build scripts and exact compile/link
  command records; baseline build is `build/entropy-clustering/candidate/`.
- `workflow/`: 24 baseline/diagnostic process records, raw samples, logs, paired
  identities, summary and exact-output hashes.
- `fixtures/`, `fixture-statistics.json`, `replay/`: captured tokens, distributions,
  160 process records and raw replay samples; `replay-main.inc` defines each probe.
- `timelines/`: 12 process records, raw task traces and 84 measured timelines.
- `sample-full.txt`, `sample-selection.txt`, corresponding process records and
  ARM64 assembly extracts.
- `audit.py`, `audit.json`, `manifest.json`: verifies recorded hashes, recomputes
  timing summaries, checks timeline alignment and replay equality, confirms all
  220 frozen runtime source files match the live checkout, and confirms identical
  Metal libraries across baseline and both diagnostic builds.

The replay binary SHA-256 is
`0a26a20224b32a87e9914e00799bf00951848e4e7d478a19964b8479c67d078d`.
Benchmark/fixture hashes and exact commands are stored in the identity and
per-process records. Build artifacts are local and ignored by Git; this report
is the only delivered repository change.
