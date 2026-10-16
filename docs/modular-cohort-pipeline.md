# Local Modular fitting across feature cohorts

This implements the CPU-fitting milestone of the cohort-stall plan on top of
`d9a98506597ec03a724cf72495678cc7fe8803a4` (`feat/gpu-histogram-fitting`). The private
hybrid workflow can prepare later feature batches and tokenize completed batches
while other local fits continue. The public default remains the existing cohort
schedule and ordinary CPU fitter.

The experiment changes scheduling and ownership. Sampling, quantization,
selection order, split/tie policy, tree compaction, feature batch shapes, GPU
producer eligibility, and final serialization retain their existing rules.
Shared/global trees keep their existing scope. Weighted prediction's producer
wavefront barriers remain necessary and unchanged.

## Scheduling and admission

`ReferenceWorkflowOptions::learning_schedule = kPipeline` enables one persistent
CPU queue for local fits. Its caller coordinates feature preparation and token
generation, and executes a fit when no coordinator operation can advance.
Workers process disjoint records. There is one shared CPU grant; a second full
fitting team is never launched. CPU feature preparation can consequently receive
fewer participants than its requested four.

The queue admits up to twice its participant count (at most 256) outstanding
whole-fit reservations. Completion publication synchronizes the caller with the
worker. Each feature owner keeps a remaining-fit count; only that owner's fits
gate its token generation and release. Tokens own their backing independently.
On failure, workers join before jobs, profiles, reservations, or feature owners
are destroyed. Unpublished output remains private to the failed workflow.

Feature batches retain the old 8 Mi-sample size and eligibility boundaries.
`feature_window_samples` accepts 8 or 16 Mi samples for aggregate live owners;
the larger window is a private experiment. It does not silently shrink batches
to make them fit, so GPU routing can be compared directly. Whole-stream rounding
can prevent two large batches from overlapping within the 8 Mi window.

Every pipeline feature owner reserves enough capacity for its eventual token
output and a worst-case tree dispatch before preparing its features. The exact
and conservative token planners share layout arithmetic. This prevents
lookahead/fit reservations from consuming its completion capacity. Fit admission
is nonblocking: workers already admitted drain before more work is attempted;
when no fit is active, the original tighter per-stage fit can be used. Before any
feature or fit has started, failed pipeline metadata/completion admission can
fall back to the original cohort route. Allocation/device/worker errors after
execution begins propagate rather than retrying work.

## Private diagnostics

The existing complete-call profiler accepts two trailing arguments:

```sh
build/release/gjxl_modular_hybrid_profile INPUT.pam OUT auto auto best \
  fit-off 14 14 14 14 14 fitter-cpu 16384 64 pipeline 16
```

Use `cohorts 8`, `pipeline 8`, and `pipeline 16` for matched controls. `schedule`
in place of `fit-off` records per-job intervals and per-batch preparation start,
feature readiness, observed fit completion, token start, and feature release.
These bounded trace buffers are allocated by the profiler before timing. The
codec's default diagnostic path does not allocate trace events.

`pipeline_seconds` covers the combined feature/fit/token phase. In pipeline mode,
`learning_seconds` spans the first job start through the last job finish and can
overlap feature and token operations. Per-job preparation/fitting times are
summed work intervals. Do not add these overlapping measurements or interpret
mean active job lifetimes as hardware CPU utilization.

## Validation and measurement

The local archive is `build/cohort-pipeline-qualification/`. Its frozen control
was built from the clean base above. `interleaved/` saves both executables, binary
and input hashes, source snapshots, all output codestreams, raw profiler JSON,
commands, power state, and detected competing processes. The reusable runner is
`tools/modular_reference_learning/qualify_cohort_pipeline.py`.

The comparison uses the existing three 4000x3000 RGB8 photo inputs
1029/1039/1049, plus photo, alpha, palette, 16-bit/all-predictor, and buffered
shared-tree cases. Each variant gets one warmup and three interleaved measured
calls on AC power, with an 8 GiB domain and 14 CPU participants. Timing covers
the complete private encoder call including cleanup, excluding backend/domain
creation, file IO, and independent decoding. Detected overlapping runs are
retained and rejected. This is one machine/session/corpus, not a general speed
or automatic-dispatch claim.

All 160 accepted encodes (40 warmups and 120 measured calls) decoded exactly and
matched complete codestream bytes across all five variants for each input and
policy. One detected overlapping attempt was retained and rejected, then rerun.
Feature/token batch counts, samples, and CPU/GPU feature routing matched the
frozen control for every hybrid call.

On this Apple M4 Pro, macOS 27.0.1, AppleClang 21 Release (`-O3 -g -DNDEBUG`)
session, complete-call medians were:

| RGB8 photo | Frozen hybrid cohorts | Same-build cohorts | Pipeline, 8 Mi samples | Pipeline, 16 Mi samples | CPU only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1029 | 3.901 s | 3.889 s | 3.932 s | 3.570 s | 3.765 s |
| 1039 | 2.953 s | 2.916 s | 2.787 s | 2.593 s | 2.742 s |
| 1049 | 2.628 s | 2.616 s | 2.600 s | 2.458 s | 2.565 s |
| Sum of medians | 9.482 s | 9.420 s | 9.319 s | 8.620 s | 9.072 s |

The 16 Mi window reduces this summed median encode time by **8.5%** against
same-build cohorts, **9.1%** against the frozen base, and **5.0%** against the
same-build CPU-only path. The 8 Mi window reduces the sum by 1.1% against
same-build cohorts, but regresses 1029 by 1.1%. The meaningful improvement here
requires enough live features to admit work across the large cohort boundaries.
Keeping the worker team alive alone does not establish a consistent win.

Separate `schedule` replays in `diagnostics/` confirm the intended mechanism:
the 16 Mi pipeline overlaps all 11 boundaries between the original cohorts
across these photos. Those boundaries have positive gaps with cohorts and
negative gaps when the next cohort starts fitting before the prior one finishes.
The trace replays are diagnostic observations, separate from the interleaved
complete-call timing matrix. They also reproduce the same codestream hashes.

The cost is memory. Across these three photos, maximum observed managed backing
rises from 742/665/642 MiB with same-build cohorts to 784/745/721 MiB with the
16 Mi pipeline. Peak feature backing rises from about 155 MiB to 308/314/308 MiB.
The complete-call managed commitment peaks remain 1814/1595/1529 MiB because a
different stage dominates that envelope. These are managed backing/reservation
measurements, not process RSS or physical allocation estimates.

All five smaller/buffered qualification cases use shared trees and therefore
retain the existing schedule; they establish exact-output coverage, not a
pipeline speed result. Local small/16-bit/all-predictor and mixed-palette
coverage comes from the focused workflow tests. This corpus does not justify a
general default or a policy for trading memory against speed, so the experiment
remains opt-in.

Frozen control SHA-256:
`0a9f6040306fb89d699186937821202d1876ff84606f1376b4713dc180ae6402`.
Candidate SHA-256:
`cb6bfa2e818f5cade79ef866207bb98520fb58cfeec3e857723267de03ffd4aa`.

Validation includes exact tree/transform/sampling decisions, prefix and ANS
codestream equality, independent lossless decoding, feature-window limits,
mixed feature producers, constrained-memory admission, output preservation,
device/host allocation errors, and partial worker launches. The incremental
queue has a deterministic held-tail test: a task in batch A cannot finish until
a task newly submitted from batch B runs. It also checks publication, ring-buffer
wraparound, 1/2/4 participants, and concurrent shared-domain calls.

Release checks and the focused Metal workflow under AddressSanitizer pass. The
focused ASan run disables leak detection for the Metal/framework process;
managed-resource release is asserted separately. The CPU queue passes both
ASan/UBSan and ThreadSanitizer. The initial broad ASan/UBSan Metal workflow timed
out at 180 seconds and emitted null-member-call diagnostics in Apple's Metal
C++ wrapper headers. That run is retained as incomplete evidence, not a pass.
Full Metal workflow ThreadSanitizer coverage is not claimed.

## Following milestone

The resumable/GPU histogram fitter retains its existing queue. Combining it with
`kPipeline` currently returns an explicit unsupported status; it does not
silently run a different fitter. That integration needs incremental admission
of immutable prepared-row owners into the same CPU/GPU coordinator, grouped
node dispatch per row owner, and independent row-owner retirement. It must
preserve the existing exact evaluator and CPU cutoff. No Metal histogram kernel
or GPU-fitting default changes are included in this CPU scheduling milestone.
