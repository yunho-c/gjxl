# Metal coefficient-order population experiment

This adapts CUDA S79 (`914b42ce642faeeaa2598bcb5fb35ceb4e2f3afb`) to the
Metal completed-frame path. Baseline is `48e32ff5f7ddead71e9095bef3929033ed699e3f`
on `perf/metal-kernel-dataflow`. The candidate source, baseline source, Release
builds, source hashes, and patch are frozen under `build/coefficient-population/`.
Measured on 2026-09-08, Apple M4 Pro (Mac16,7, 48 GiB), macOS 15.6,
Apple Clang 17, Release C++20/libc++, libjxl reference disabled.

The experiment is exact and reduces the 4K serializer tail by 3.2–5.3%, with
paired whole-call gains of 0.5–1.2% on the two 4K inputs. Small-input benefit
is not established.

## Implementation

The final quantization dispatch already writes group-major integer AC into an
independently owned completed-frame buffer. A counting dispatch immediately
follows each final transform batch in the same serial compute encoder. It reads
that batch through the existing destination table and reduces zero populations
in tiles of 32 coefficient positions by 64 anchors, with 256 threads. Eight
partial rows are reduced in threadgroup memory before integer global atomics.
Both ordinary and profiled execution use the same dispatch path, including
policies with and without a final score evaluation.

The 6144 uint32 bins occupy 24 KiB: 5952 full counts across three channels and
five order families, followed by 192 sampled DCT8 counts. Both orientations of
each rectangular transform share physical coefficient positions and accumulate
into the same family. Pure DCT8 gets a byte flag per anchor, generated with the
existing xorshift sequence in AC-group-first order and remapped to batch order.
The final authoritative strategy grid determines families and sampling on every
output request. A runtime check enforces the pure-DCT8 raster mapping.

The completed output allocation retains the populations and sample flags along
with coefficients and destinations. The CPU clears counts before submission;
there is no extra GPU submission, device clear, completion wait, or AC copy.
The counts become readable only after completion, and remain immutable for the
frame lease's lifetime, even after evaluator reuse, cache trimming, or backend
destruction. Device buffer arguments cover disjoint slices of that allocation.
The mapped-frame accounting includes the extra 24 KiB of population reads.

A private frame-view field exposes optional populations to order selection.
The consumer validates shape, supported family mask, per-family anchor bounds,
and sampled-subset bounds. It then uses the existing float-scaled sort keys,
natural-rank ties, LLF prefix, and order tokenization unchanged. Exact semantic
correspondence with coefficients is a producer contract, tested by scalar
recount; validating it in production would repeat the eliminated scan.
CPU/owned frames without populations retain the previous counting path,
including deterministic sampling, parallel reduction, managed allocation
failure handling, and the uint64 counter fallback. Metal's existing uint32
coefficient-address bound also bounds populations below uint32 overflow.

## Storage

The exact completed-frame device recipe adds `24576 + anchor_count` bytes and
no allocation. Sample bytes are reserved even for mixed frames, where they are
zero and unused. The worst-case 3840x2160 plan adds 154176 bytes (0.147 MiB);
actual mixed-transform frames have fewer anchors. The source/destination AQ
arenas are unchanged. The generic serializer plan conservatively retains its
CPU recount bound. Phase-local CPU allocation savings do not imply lower RSS
or lower whole-workflow peak backing.

## Qualification and timing

The scalar population oracle compares all 6144 bins, including absent-family
and sampled bins, against final Metal coefficients. Retained-frame tests cover
pure DCT8, all seven transform shapes in a mixed grid, multiple groups, narrow
and padded edges, changed strategies and images, zero/two AQ iterations, both
final-score settings, ordinary and profiled execution, cache recovery, concurrent
readers/trimming, and destruction of the original backend. It also compares
both cached order policies with the uncached CPU path, full codestream bytes,
and the exact retained allocation count/capacity. Metal API and shader validation
pass on the frozen candidate.

The host order test exercises 218 frame/policy cases against an independent
coefficient-major scalar sort oracle, also comparing cached orders on all 218.
It checks malformed table lengths, unsupported/empty masks, overflowing full
counts, forbidden sampled bins, and unchanged outputs on errors. Four targeted
host suites pass ASan/UBSan without suppressions (leak detection disabled):
coefficient order, representation storage, codestream encoder, and VarDCT frame.

All 56 canonical corpus/policy cases are byte-identical, including efforts 1–10,
full/sampled policy selection, maximum compression, alternate Metal policies,
rate control, and the CPU fallback. Three pinned decoder pairs match; all 22
codestream conformance fixtures pass in each build. Decoder revision:
`e8ff09762481785938d8e4e01333ed3917571161`.

All eight resource/lifecycle scenarios pass, including tight/full limits and
two callers; 24 retained codestreams and their decoded pixels match. The 4K
managed peak backing is unchanged at 2602024508 bytes (2481.484 MiB). Mixed-call
peaks vary with overlap, so this experiment makes no whole-workflow memory or
RSS reduction claim. These resource runs are correctness/diagnostic runs, not
performance samples.

The coefficient-order profile field sums the elapsed spans around order
selection and order-token generation in `encoder.cpp`; it is contained within
AC-tokenization latency. The separate coefficient-tokenization field sums
worker work. Neither is added to complete-call timing. GPU stage measurements
use instrumented submissions and are a separate experiment.

The baseline full Release suite passes 123/124, failing only the inherited
`quantization_pipeline` fixture (`actual=0.24919039011001587`,
`expected=0.24914586544036865` at index 1). The candidate full run initially
reported that same failure plus an obsolete expectation in
`worker_launch_failure_metal_orders`: GPU populations remove that CPU launch.
The revised test verifies no launch and exact output for resident Metal, and
retains actual partial-launch failures through the exact-coefficient Metal
recount path. Its focused rerun passes all 24 configurations. Only this test
changed after the full run; runtime sources and binaries are identical to those
used for parity, resource checks, and final timing. Effective Release coverage
is 123/124 with the same sole inherited runtime failure. Both original JUnit
files and the test-only correction/rerun are retained.

## Warm performance

Seven alternating independent-process pairs per input, three warmups and seven
timed samples per process: 84 process records and 588 timed encodes. Policy is
fully-resident Metal, effort 7, distance 1.2, fused-tuned residual inverse, with
GPU stage profiling disabled. All build/test/resource/corpus processes finished
before this run; the controller rejects competing encoders/builds/tests.
The boundary is the encoding benchmark's public workflow, excluding file I/O
and process startup. This is not a cold-start study.

Displayed times are medians of process medians. Paired changes are medians of
the seven per-pair percentage changes; they need not equal the ratio of the
separately displayed medians. Negative means faster.

| Input | Serializer ms | Paired change | Complete workflow ms | Paired change |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 1.640 → 1.634 | -0.95% | 9.935 → 10.566 | +4.29% |
| Padded 1919x1079 | 15.492 → 15.073 | -2.09% | 65.573 → 65.741 | +0.19% |
| Padded 3839x2159 | 35.860 → 34.414 | -3.23% | 212.164 → 213.585 | -0.45% |
| Kodak 17 (512x768) | 5.394 → 5.308 | -0.74% | 20.539 → 20.667 | +0.02% |
| Planter 1920x1080 | 19.971 → 19.564 | -2.18% | 70.375 → 69.538 | -0.92% |
| Planter 3840x2160 | 39.945 → 37.965 | -5.31% | 220.835 → 217.467 | -1.15% |

| Input | Complete-call pair range | Faster pairs |
| --- | ---: | ---: |
| Synthetic 128x96 | -19.07% to +11.03% | 2/7 |
| Padded 1919x1079 | -1.13% to +2.34% | 3/7 |
| Padded 3839x2159 | -3.00% to +4.15% | 5/7 |
| Kodak 17 (512x768) | -3.50% to +3.22% | 3/7 |
| Planter 1920x1080 | -2.67% to +0.36% | 6/7 |
| Planter 3840x2160 | -3.59% to +1.22% | 6/7 |

The result is a modest larger-image optimization, not a general large speedup.
The natural HD/4K images improve in six of seven complete-call pairs; padded
4K improves in five. Kodak and padded HD are effectively flat in this run.
The tiny synthetic case regresses 4.29% at the paired median with wide spread;
its earlier three-pair screen had the opposite sign. There is no demonstrated
small-image gain, and these samples do not establish a stable small-image
regression magnitude. The padded-4K ratio of separate medians has the opposite
sign from its paired median; that is retained rather than rounded away.

At 4K, order selection plus order-token generation drops from 3.038 to 0.245 ms
on padded input and from 2.955 to 0.147 ms on Planter: 92.03% and 95.02% lower
at the paired medians. AC-tokenization elapsed time improves 15.50% and 14.99%.
However, the subsequent coefficient-tokenization aggregate worker work rises
21.60% and 15.56%, respectively. Removing the CPU scan does not translate
one-for-one into serializer savings. The measurements do not isolate whether
cache state or worker scheduling causes this offset.

The earlier three-pair instrumented GPU screen used the byte-identical candidate
metallib (`0bf114d11aae32ab853b374dcdb3a7bb46a6fd2fbc4e224127802dba06c7cee4`).
The sum of median final-coefficient stages increases from 2.440 to 2.877 ms
on padded 4K and from 2.999 to 3.454 ms on Planter 4K. These stages include
quantization and the new count dispatches, excluding the final quantizer stage;
they are not isolated count-kernel timestamps or additive whole-call estimates.
The frozen candidate subsequently adds only host metadata/mapping checks and
test coverage to that GPU implementation.

## Evidence and reproduction

All evidence is under `build/coefficient-population/`:

- `baseline-src`, `candidate-src`, `baseline`, `final`, `asan`, source hash
  manifests, `candidate.patch`, and `toolchain.json`.
- Original full-suite JUnit/logs, `launch-test-fix.xml`, `test-summary.json`,
  sanitizer results, Metal validation log, and both 22-fixture conformance runs.
- `parity/` (56 cases, three decoded pairs), `resources/` (eight scenarios,
  24 decoded retained pairs), and `wall/` (42 pairs, 84 processes, 588 samples).
- `screen/` and `profile-screen/` retain preliminary measurements separately
  from final timing. Pre-test-correction patch and source hashes are retained.

The existing `tools/metal_dataflow/compare.py`, `parity.py`, and `resources.py`
record commands, inputs, runtime identities, raw results, and artifact hashes.
Use the frozen baseline/final build directories with the pinned corpus at
`gjxl-libjxl-comparison/build/libjxl-comparison/corpus-phase1-pilot-pinned/canonical`.
Release configuration: `GJXL_BUILD_TESTS=ON`, `GJXL_BUILD_BENCHMARKS=ON`,
`GJXL_ENABLE_LIBJXL_REFERENCE=OFF`. Direct compiler/CTest subprocesses use
`SDKROOT="$(xcrun --show-sdk-path)"`. CPU sanitizer flags are
`-fsanitize=address,undefined -fno-omit-frame-pointer`; tests use
`ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1`.
