# Identical-coefficient AC tokenizer replay

The controlled replay confirms an implementation gap: GJXL's checked direct AC
tokenizer takes **2.24–2.51x as long as pinned libjxl** on identical coefficients,
orders and contexts, with GJXL population collection disabled. Every emitted
value and context matches. This is serial tokenizer work, not complete-encode
latency or a SIMD-only comparison.

The selected change replaces two scalar token-output `push_back` calls with
`emplace_back`. On this AppleClang/libc++ build it brings the capacity checks and
stores into the token appender, reducing isolated photographic tokenizer time by
13–21% while preserving token, population and codestream output. No new SIMD
intrinsics or forced-inlining attribute are required. Complete-workflow and batch
measurements below determine the practical benefit.

## Revisions and hardware

- GJXL `06857bedd0467b4cb8b034d6456b0209b1db99c5`.
- libjxl `a6afcb686ba32fa321904b6d5dbf1c688c95a69c`, using the immutable Release
  archive/build from the preceding AC serialization profile, including its pinned
  Highway, Brotli and skcms sources. This is not a claim about latest upstream.
- Apple M4 Pro, 48 GiB, macOS 15.6, AppleClang 17.0.0, ARM64 Release `-O3`.
  libjxl uses its Highway implementations with compiler auto-vectorization
  disabled by its build flags; GJXL keeps its ordinary compiler vectorization.

## Identical inputs and correctness boundary

An untimed capture encoder exports all AC groups from the same four canonical
linear-sRGB PFMs used in `ac-serialization-profile.md`, effort 7, distance 1.2,
fully-resident Metal. The capture's complete codestream matches the frozen GJXL
baseline on each input. Its binary fixtures retain:

- Group dimensions, row-major transform anchors and strategy identifiers.
- All three planes of quantized coefficients and the group's raw quantization field.
- Block-context map and quantization thresholds, custom-order mask, and the exact
  resolved scan order for each active strategy family/channel.
- Expected token values/contexts and sparse populations, including totals, extra
  bits, maximum symbol, context encounter order and symbol counts.

The adapters reconstruct each library's native data structures outside timing.
Coordinates are made group-local in both adapters. libjxl receives no chroma
subsampling, one DC context, and zero qdc indices, matching GJXL's context map
without DC thresholds. GJXL's reconstructed natural scans are checked against
captured scans. libjxl receives those same physical coefficient indices.

All 316 photographic groups are replayed. A separate constant linear-gray
3840x2160 PFM (RGB 0.18) supplies 135 additional groups for sparse-output screening.
Both tokenizers match all expected tokens on this flat fixture as well. The
Python audit independently reconstructs each captured HybridUint population from
its expected tokens and verifies all stored population metadata and counts.

## Timing boundary

Each timed call serially tokenizes every group in one captured image. Output
containers are fresh for each call; group token storage allocation is inside the
timer. Worker scratch is created inside each timed call, reused between groups,
and destroyed before the timer ends. Output storage remains live until validation
after the timer. libjxl initializes its nonzero scratch image once per call.

Input loading/adaptation, natural-order preparation, output equality checks and
logging are outside timing. GJXL's direct validated implementation is compiled
from frozen source into the replay translation unit to access its private entry
point; libjxl uses its pinned object implementation. This exposes compiler and
storage differences as well as loop arithmetic, and is not the public encoder API
or a parallel wall-time benchmark.

The main comparison has five independent process rounds per input, rotating and
alternating the three modes: GJXL without populations, GJXL with populations,
and libjxl tokenization. Each process performs one reference/validation call,
five warmups and 31 measured calls: 60 processes and 1860 measured calls.
All calls compare every produced value/context with the captured reference;
GJXL population mode additionally compares every sparse population field.

Values below are medians of process medians. Ratios are medians of ratios within
paired rounds, so displayed medians need not divide to the displayed ratio.
Milliseconds represent serial work over all groups, not image latency.

| Input | GJXL no populations ms | libjxl ms | GJXL/libjxl ratio | GJXL with populations ms |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 1.064 | 0.452 | 2.330 | 1.608 |
| Planter 1080p | 9.181 | 3.655 | 2.513 | 15.112 |
| Planter 4K | 27.571 | 11.564 | 2.383 | 43.066 |
| Padded-stress 4K | 25.201 | 11.158 | 2.235 | 38.496 |

The no-population comparison resolves the ambiguity in the earlier native
profile: different token counts and the location of population collection do
not explain the entire tokenizer gap. GJXL retains its group validation,
checked storage and native split value/context output, so the ratio does not
isolate SIMD counting from those other implementation costs.

The population-enabled column performs additional collection and sparse
publication. Comparing it directly with libjxl tokenization would overstate
the matched-work implementation gap.

## Generated-code finding and first screen

The frozen production object and baseline replay both contain out-of-line
`AppendDirectAcToken` calls in the coefficient loop. Clang's missed-inlining
remarks report a helper cost of 375 against a threshold of 250. They also report
that the value and context `vector::push_back` calls exceed that threshold
(costs 255 and 250 respectively). These calls remain relevant after merely
inlining the outer appender.

A plain `inline` keyword does not force the intended code change. The explicit
`always_inline` experiment removes the appender call and preserves its integer
operations, population updates, checks, allocation behavior and errors. This experimental
variant restricts the hint to Clang ARM64; other compiler/architecture
combinations keep the original declaration.

Five paired rounds, five warmups and 31 measured calls per variant/input give
another 60 processes and 1860 measured calls. Population collection is enabled
for every row in this screen.

| Input | Baseline ms | Plain inline change | Forced inline ms | Forced inline change |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 1.580 | +2.15% | 1.494 | -4.40% |
| Planter 1080p | 15.093 | -0.37% | 13.930 | -7.31% |
| Planter 4K | 43.103 | -0.25% | 40.057 | -7.00% |
| Padded-stress 4K | 38.563 | +0.42% | 36.130 | -6.31% |

## Follow-up screen and rejected indexed writes

A bounded follow-up tests hoisting the final coefficient-context offset out of
the inner scan, pre-sizing both output vectors and writing by index, and their
combinations with forced inlining. The indexed variant uses the existing checked
capacity `3 * (64 * block_count + anchor_count)`, then shrinks to the emitted
extent. It keeps population operations and verifies all output fields, but
`resize` value-initializes the maximum-sized buffers before token production.

Three paired rounds per input, three warmups and 15 samples for eight variants
on five inputs give 120 processes and 1800 measured calls. Representative rows
show the tradeoff; `analysis.json` retains all variants.

| Input | Forced inline change | Context hoist change | Indexed + inline change | Indexed + hoist + inline change |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | -4.93% | +3.83% | -19.72% | -18.52% |
| Planter 1080p | -7.65% | +0.87% | -25.35% | -25.33% |
| Planter 4K | -6.60% | +0.84% | -18.76% | -20.94% |
| Padded-stress 4K | -7.30% | +2.50% | -19.27% | -20.89% |
| Flat 4K | +0.50% | +1.82% | +281.21% | +288.47% |

On flat 4K, baseline time is 2.444 ms while the indexed variants take approximately
9.48–9.86 ms. Initializing mostly unused token storage erases their photographic
benefit. These variants are rejected. Context hoisting alone provides no consistent
gain. The forced-inlining variant is superseded by the following simpler change.

## Selected change: scalar token emplacement

Replace only `group->values.push_back(value)` and
`group->contexts.push_back(context)` with `emplace_back` in
`AppendDirectAcToken`. Both elements are scalar integers, passed as the same
lvalues. Output allocation, growth, size, population handling, checks and error
behavior are preserved; no maximum-capacity buffers are initialized.

The native ARM64 object confirms that the two previously called append fast paths
become local capacity checks and scalar stores inside `AppendDirectAcToken`.
The outer appender itself remains out of line. This is a measured compiler and
standard-library code-generation benefit, not a general claim that emplacement is
faster for integers on every toolchain.

Three paired rounds, three warmups and 15 measured calls for four modes across
five inputs add 60 processes and 900 measured calls, all with population collection.
The combined emplacement/forced-inline variant supplies limited and inconsistent
additional benefit on larger images. Retain plain emplacement as the portable,
minimal change.

| Input | Baseline ms | Emplace ms | Emplace change | Emplace + forced inline change |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 1.519 | 1.326 | -12.72% | -20.19% |
| Planter 1080p | 14.592 | 11.863 | -18.84% | -18.69% |
| Planter 4K | 42.536 | 34.067 | -19.50% | -21.14% |
| Padded-stress 4K | 38.622 | 30.499 | -21.31% | -22.70% |
| Flat 4K | 2.551 | 2.401 | -5.89% | -7.24% |

## Native qualification

The baseline's tracked source matches `06857be`. The frozen Release candidate
changes only the two token appends; both builds have byte-identical Metal libraries.
Validation is against that candidate, not the earlier forced-inline prototype.

- All 451 groups pass exact token comparison against both implementations.
  Population-enabled variants additionally match all sparse population metadata
  and bins. Across the four screens, 300 processes and 6420 measured replay calls
  pass, with reference calls and warmups also validated.
- Complete codestreams match byte-for-byte in all 56 corpus/policy cases, plus
  the additional flat 4K case. Byte identity preserves decoded output; this pass
  does not claim a new independent decoder/conformance run.
- Release: 128/129 tests pass. The sole failure is the inherited
  `quantization_pipeline` mismatch at index 1: actual `0.24919039011001587`,
  expected `0.24914586544036865`, identical to the earlier baseline failure.
- ASan/UBSan: all four direct-tokenization, token-storage-plan,
  serializer-storage-plan and serializer-storage suites pass.

## Complete-workflow measurements

Each input has seven independent alternating baseline/candidate process pairs,
five warmups and 11 samples per process: 84 processes and 924 measured encodes.
Effort 7, distance 1.2, fully-resident Metal, SIMD backend and fused-tuned inverse
are explicit. PFM reading and filesystem output are outside timing; the measured
boundary is the existing profiled public-workflow implementation from linear RGB
to an in-memory codestream, using a warm supplied GPU backend. Its ordinary CPU
phase timers remain enabled. GPU stage/submission profiling is disabled.

These are actual complete-workflow times, distinct from serial tokenizer work.
`codestream_encoding` is the whole serialization phase, not just tokenization.
Input order is fixed and baseline/candidate order alternates within each input;
the experiment is not a randomized corpus sweep. All percentage changes below
are medians of the seven paired process-median changes. Negative means faster.

| Input | Baseline total ms | Candidate total ms | Total time change | Serialization time change |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 9.593 | 8.022 | -10.87% | -8.66% |
| Kodak17 | 19.297 | 19.096 | -1.14% | -0.80% |
| Planter 1080p | 63.727 | 63.263 | -0.73% | -1.79% |
| Planter 4K | 199.119 | 197.761 | -0.45% | -3.49% |
| Padded-stress 4K | 197.974 | 196.092 | -0.61% | -4.04% |
| Flat 4K | 219.473 | 216.876 | -1.98% | +0.19% |

| Input | Total-time pair range | Pairs faster / 7 |
| --- | ---: | ---: |
| Synthetic 128x96 | -19.56% to +1.57% | 5 |
| Kodak17 | -2.60% to +1.40% | 5 |
| Planter 1080p | -2.44% to +0.94% | 6 |
| Planter 4K | -1.60% to +0.82% | 6 |
| Padded-stress 4K | -1.76% to +1.87% | 4 |
| Flat 4K | -12.77% to +11.44% | 5 |

The native phase profile localizes the benefit. The AC-tokenization wall phase
includes ordering/context work and scheduling; coefficient-tokenization work is
summed across workers and cannot be added to wall time. All four photographic
inputs show approximately 15% less tokenization worker work. Their unmodified
quantization-pipeline medians move by only -0.26% to 0.00%.

| Input | AC-tokenization wall change | Coefficient-tokenization worker-work change |
| --- | ---: | ---: |
| Kodak17 | -11.11% | -15.04% |
| Planter 1080p | -7.53% | -14.72% |
| Planter 4K | -9.98% | -14.53% |
| Padded-stress 4K | -11.56% | -14.64% |

The tiny and flat controls have much wider whole-workflow variation. In the tiny
case, the unmodified quantization phase shifts by -12.4%; its apparent -10.9%
complete-workflow improvement cannot reasonably be assigned to two token stores.
Flat 4K's total-time pairs span -12.8% to +11.4%, with a +0.19% median serialization
change. These controls do not establish a stable whole-workflow benefit or
regression, but do rule out the obvious fourfold sparse-tokenizer penalty of
pre-sized indexed output in the replay screen.

## Batch throughput

Seven alternating baseline/candidate process pairs measure three photographic
inputs at batch sizes 2 and 4, with three warmups and five samples per combination.
The 14 processes retain 420 batched public calls and 420 sequential-control calls.
Both drivers execute `VarDctBatchEncoder::Encode` with the same request count;
the sequential control has a concurrency limit of one. The table compares the
batched candidate with the batched baseline, not with its sequential control.

The input is repeated within each batch. Encoder creation and file reads are
outside timing; scheduling and complete in-memory encode completion are inside.
Every output and summary is checked against a single-image reference outside
timing. Policy is automatic per-image CPU workers, Metal fully-resident, effort 7,
distance 1.2. This is the batch API's boundary and differs from the warm supplied
GPU-backend workflow above. `producer` in artifact filenames denotes the candidate.
Positive throughput change means faster; rates are derived from medians of
process-median batch times, and changes from within-pair ratios.

| Input | Batch | Baseline images/s | Candidate images/s | Paired throughput change | Pair range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 2 | 63.270 | 62.696 | +0.01% | -13.14% to +7.98% |
| Kodak17 | 4 | 91.935 | 91.995 | +0.38% | -5.59% to +2.03% |
| Planter 1080p | 2 | 19.616 | 19.691 | +0.33% | -7.39% to +16.32% |
| Planter 1080p | 4 | 23.837 | 23.282 | -2.33% | -8.28% to +16.81% |
| Planter 4K | 2 | 5.726 | 5.628 | +0.67% | -9.47% to +17.84% |
| Planter 4K | 4 | 6.162 | 6.047 | -0.72% | -7.57% to +4.98% |

## Decision and interpretation

Retain the two-line emplacement change as a small, exact-output tokenizer
optimization. It improves the isolated photographic tokenizer by 13–21%, reduces
native tokenization worker work by about 15%, and reduces the measured complete
photographic workflow by only 0.45–1.14% in paired medians. The AC-tokenization
wall phase improves 7.5–11.6%; the rest of the encode limits the total benefit.
Individual complete-workflow pairs still include regressions, so the small total
changes should be treated as indicative, not a guaranteed speedup for each call.

Batch throughput is not qualified as an improvement. Paired median changes span
-2.33% to +0.67% across the six input/batch combinations. In particular, Planter
1080p at batch 4 measures -2.33%, and Planter 4K at batch 4 measures -0.72%.
Every combination has both winning and losing pairs. Retain these negative rows
alongside the positive ones rather than describing the experiment as an overall
throughput win. A snapshot during collection found active desktop processes
outside the measurement guard's build/benchmark list; the small-corpus batch
results need a quieter repeat before accepting a general throughput claim.

The controlled libjxl comparison applies to the original GJXL tokenizer with
population collection disabled. It establishes a 2.24–2.51x baseline gap on the
same tokens, coefficients, orders and contexts. The selected emplacement screen
has population collection enabled, so its percentage improvement cannot be used
to calculate a new GJXL/libjxl ratio. It also does not justify replacing the
already-vectorized counting loops with a new SIMD implementation.

For further work, preserve the identical-input replay and separate the remaining
population-update, validation and token-emission costs before selecting another
kernel. Avoid pre-initializing maximum-sized token buffers: the flat-image replay
shows why the larger photographic speedup is not a viable general optimization.
The current results support keeping a small compiler-friendly source change;
they do not support a larger throughput claim or a broad SIMD rewrite yet.

## Evidence and reproduction

Artifacts are ignored under `build/tokenizer-replay` in this worktree: capture
sources, 451 fixtures and manifests; all replay/variant sources and build commands;
compiler remarks and native disassembly; raw timing logs and per-process records;
analysis/audit scripts; and isolated native source/builds. The selected qualification
and workflow/batch artifacts use the `emplace-` prefix. Unprefixed candidate
qualification files belong to the superseded forced-inline experiment.

The initial measurement guard detected another worktree's tests before collection.
A later post-process guard detected a concurrent test/build during the original
Planter 4K replay round 1. The entire partial round was retained under
`replay/excluded-after-busy-guard` and repeated. Retained processes passed
before/after guards. This does not establish exclusive-machine or thermal control.

`analysis.json` contains all paired replay ratios; `final-audit.json` verifies
native source scope, test inventories, parity hashes, raw timing records, summaries
and runtime identities. `evidence-sha256.json` freezes the supporting scripts,
reports and retained logs. These results are specific to the pinned compiler,
libraries, settings, device and small corpus; no CUDA or other-CPU throughput
claim follows from them.

Saved-data analysis can be reproduced without running encodes:

```sh
python3 build/tokenizer-replay/analyze.py
python3 build/tokenizer-replay/audit.py
python3 build/tokenizer-replay/audit-final.py
python3 build/tokenizer-replay/write-report.py
```
