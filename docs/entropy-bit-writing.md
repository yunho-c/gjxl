# Entropy bit-writing optimization qualification

Retain aligned append, word-oriented bit writing, and producer-side ANS packing.
The final implementation preserves every tested codestream byte. On this M4 Pro,
4K serializer latency falls 6–7%; complete single-image latency falls about
0.9–1.5%. Four concurrent Planter 4K images show a 5.45% median paired throughput
gain. This is useful but smaller than the earlier 5–10 ms complete-image target;
it does not establish a general 5% throughput gain across workloads.

Baseline: `1b108ce`. Candidate: `perf/entropy-bit-writing` branch.
Apple M4 Pro, 48 GiB, macOS 15.6 (24G84), AppleClang 17, Release C++20/libc++,
Ninja, benchmarks enabled, libjxl-reference fixtures disabled. No GPU kernels,
model arithmetic, token order, entropy policy, ANS recurrence or quality settings
change. The frozen candidate is `build/entropy-bit-writing/producer-src`.

## Retained implementation

1. Ordinary `BitWriter::Append` bulk-copies when its destination is byte-aligned,
   preserving a partially filled source's exact logical bit count (S168 adaptation).
2. `WriteBitsUnchecked` uses one unaligned little-endian 64-bit store for up to
   56 bits. Seven writable padding bytes are included in managed storage bounds
   and excluded from published bytes. Rollback restores the logical final byte
   and clears overwritten padding, including nested allotments. An explicit
   byte-order fallback covers non-little-endian hosts. ARM64 disassembly confirms
   a single `str` rather than a byte loop; other architectures were not measured.
3. ANS packs reverse fragments as they are produced, storing full 56-bit words
   and keeping the partial word on the stack. It writes the final ANS state,
   the partial word, then the stored words in reverse order. Fragment splitting
   preserves the exact bit sequence. One checked allotment reserves the temporary
   output; a synchronous reference-wrapped callback avoids a callback allocation.

The first ANS experiment adapted S167's consumer-side batching. Producer-side
packing supersedes it in the final source: there is no remaining array of
individual reverse fragments or second coalescing pass.

Each token emits at most 31 extra bits and 16 renormalization bits. Runtime and
planner share the checked bound `floor(47*N/56)` uint64 words, with the partial
word on the stack and the 32-bit state outside that array. Compared with `2*N`
eight-byte fragments, the asymptotic reverse-array bound falls about 58%.
The historical `reverse_chunks` field/helper names now refer to packed words.

## Marginal screens and retention

Three alternating process pairs per stage, three warmups and five timed encodes
per process. These are screening results against the preceding implementation,
not independent contributions that can be added together.

| Step | Kodak section writing | Planter 1080p | Planter 4K | Decision |
| --- | ---: | ---: | ---: | --- |
| Aligned append | -1.33% | -3.82% | -4.12% | Retain |
| Consumer ANS batching | -6.50% | -7.67% | -8.46% | Superseded by producer packing |
| Word writer | -5.63% | -4.82% | -2.51% | Retain |
| Producer packing | -7.79% | -8.92% | -9.55% | Retain |

The initial combined append/batching/word confirmation reduced 4K section-writing
latency about 15%. The final producer implementation improves it about 21–22%.
Both cohorts and all marginal screens are retained.

## Complete single-image measurements

Seven alternating independent-process pairs, three warmups and seven measured
encodes per process, six workloads: 588 timed encodes in the final cohort.
Fully-resident Metal, effort 7, distance 1.2, automatic CPU threads, fused-tuned
residual inverse, no GPU-stage profiling. The public workflow timer excludes
file I/O, initialization and process startup. Stage columns below are wall times;
aggregate worker durations are not added to them.

Percentages are medians of paired process-median ratios. Milliseconds are medians
of process medians, so dividing the displayed times need not reproduce the paired
percentage. Negative changes mean lower latency.

| Input | Serializer ms, baseline → final | Serializer change | Section-writing change | Whole encode ms, baseline → final | Whole change | Whole faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic 128×96 | 1.586 → 1.562 | -1.50% | -8.16% | 9.079 → 9.631 | +5.02% | 3/7 |
| Padded 1919×1079 | 15.247 → 14.653 | -4.31% | -18.99% | 63.977 → 64.198 | +0.35% | 3/7 |
| Padded 3839×2159 | 35.276 → 32.920 | -7.07% | -21.92% | 208.349 → 206.394 | -0.88% | 5/7 |
| Kodak17, 512×768 | 5.494 → 5.317 | -3.90% | -11.81% | 20.408 → 20.201 | +0.21% | 3/7 |
| Planter 1920×1080 | 20.051 → 18.934 | -6.05% | -21.40% | 68.512 → 67.691 | -1.20% | 6/7 |
| Planter 3840×2160 | 39.080 → 36.709 | -6.07% | -20.95% | 211.628 → 208.017 | -1.53% | 5/7 |

Section writing improves in all seven pairs for every input. Serializer wall time
improves in every pair for both HD and both 4K inputs. Complete-call effects are
smaller: padded 4K paired changes range from -2.64% to +2.17%; Planter 4K from
-3.49% to +1.12%. This is not exclusive-machine or controlled thermal testing.

A separate longer small-image confirmation uses seven pairs, ten warmups and
21 samples per process. Kodak whole latency changes -1.42% (six faster pairs,
range -3.35% to +0.49%). The identical-baseline A/A control changes -0.13%
(range -1.08% to +0.71%). Tiny-image results reverse sign to -8.67%, while its
A/A control itself changes +5.07% with a -10.19% to +19.33% range. No reliable
tiny-image whole-call gain or regression is established. These follow-up cohorts
are separate; they do not replace the original mixed results.

## Batch throughput

The existing image-batch benchmark measures preloaded linear RGB through complete
in-memory output, including scheduling and the whole workflow. Initialization,
warmup and correctness checks are outside the timer. Every output matches the
process's single-image reference; cross-build output sizes agree, with byte
identity separately covered by the corpus/policy qualification.

Seven alternating independent-process pairs, two warmups and three samples per
batch size/input. Positive changes mean more images per second. The two rates
are derived from the medians of batch times; the paired percentage is calculated
within each independent pair and then summarized, as in the single-image study.

| Input | Batch size | Images/s, baseline → final | Paired gain | Pair range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kodak17 | 2 | 56.531 → 58.239 | -2.16% | -7.06% to +14.91% | 3/7 |
| Kodak17 | 4 | 84.061 → 86.319 | +0.71% | -5.57% to +10.69% | 4/7 |
| Planter 1080p | 2 | 18.293 → 18.757 | +1.20% | -1.35% to +5.97% | 5/7 |
| Planter 1080p | 4 | 21.772 → 21.980 | +2.73% | -4.81% to +4.81% | 4/7 |
| Planter 4K | 2 | 5.054 → 5.028 | +0.22% | -3.07% to +5.41% | 4/7 |
| Planter 4K | 4 | 5.539 → 5.576 | +5.45% | -1.31% to +19.53% | 6/7 |

The four-image 4K result is encouraging; two-image 4K batching is essentially
flat. The broad ranges and difference between paired and pooled summaries limit
how confidently the 5.45% figure can be generalized.

Because the initial Kodak two-image result suggested a regression, a focused
seven-pair cohort used five warmups and eleven samples. Gains become +0.61%
(batch 2, five faster pairs) and +2.08% (batch 4, four faster pairs). Matching
A/A controls give -0.68% and -0.25%. There is no reproducible initial Kodak
regression; its throughput improvement remains modest and noisy.

## Correctness and storage qualification

- Final full Release: **128/129**. Baseline: **125/126**. Both reproduce the sole
  `quantization_pipeline` score mismatch at index 1: actual
  `0.24919039011001587`, expected `0.24914586544036865`. No golden changed.
- The initial word build also exposed the managed-allocator fixture's obsolete
  16-byte allowance. A 32-byte allowance restores its intended growth-rejection
  scenario with charged padding. Its retest and the final full suite pass.
- Seven word-writer host ASan/UBSan suites pass; four final producer suites pass,
  including ANS, entropy and both storage planners. No suppressions; leaks off.
- Independent append oracle: 18,936 cases, 18,864 insufficient allotments and
  37,872 late-error rollbacks. Independent division-based ANS oracle: 816
  configurations and 30,240 comparisons/failure checks. Its predicted batching
  counter is a reference simulation, not instrumentation of producer writes.
- Independent bit-at-a-time writer oracle: 2,280 sequences, all widths 0–56 and
  offsets 0–7, invalid inputs and nested rollback followed by further writes.
- Final **56/56** canonical corpus/policy codestreams are byte-identical.
  Coverage includes 38 corpus images, efforts 1–10, alternate entropy/throughput/
  exact-coefficient policies and target/error modes. Earlier variants remain
  separately qualified in their retained records.
- Baseline, word and producer builds each pass all **22** pinned conformance
  fixtures. Decoder: `djxl` revision `e8ff0976`, SHA-256
  `9782c3474e41e5e5415da5fc68e7103d27047661385dc5b863dcad50ec4474ac`.
- Eight final resident-resource scenarios pass, including tight/full admission,
  multiple callers and release/shutdown. All 24 retained output/decode pairs
  match. Peak managed backing for the one-image 4K scenario remains
  **2,602,024,508 bytes**. Peak committed allowance decreases from 9,316,371,018
  to 9,301,538,408 bytes; that is reservation accounting, not RSS. No whole-
  workflow peak physical-memory saving is established.
- Baseline and all four candidate Release variants use identical Metal libraries.

## Remaining entropy opportunity

A separate diagnostic build splits direct-ANS model work into population
preparation, clustering and normalization. The quiet cohort retains three
measured calls after one reference call and one warmup. These are per-task work
durations and must not be added to concurrent whole-encode wall time.

| AC model | Contexts | Population ms | Clustering ms | Normalization ms |
| --- | ---: | ---: | ---: | ---: |
| Kodak17 | 2475 | 0.317 | 1.295 | 0.219 |
| Padded 4K | 6930 | 0.844 | 4.632 | 0.295 |
| Planter 4K | 6930 | 0.834 | 4.988 | 0.337 |

AC clustering is the larger measured model target. DC population preparation also
costs 1.58–2.17 ms in these 4K cases. Prioritize profiling their loops and data access
before approximate SIMD logarithms or normalization changes. Preserving ordered
double arithmetic and deterministic ties constrains vectorization. No histogram
optimization is retained here, and no speedup is inferred from these diagnostic
work totals. The initial diagnostic run overlapped qualification and is excluded;
only `histogram-quiet-*` records feed the table.

## Evidence and reproduction

Local evidence is under `build/entropy-bit-writing/`: frozen baseline/append/batch/
word/producer source trees, hashes and binaries; raw timing samples and command
records; corpus outputs; decoder fixtures; resource scenarios; JUnit and sanitizer
logs; ARM64 assembly; diagnostic source and logs; and study controllers.

`audit.py` rechecks runtime/input hashes, raw/log hashes, recomputes timing summaries
from every saved sample, validates test inventories and parity/resource outputs,
and confirms delivered runtime source equals the tested producer snapshot.
`audit.json` records 368 single-image timing processes (184 pairs) and 42 batch
processes (70 workload pairs). `final-evidence-sha256.json` hashes the retained
records and scripts. Timing harnesses reject observed concurrent GJXL/build/test
processes; this does not eliminate OS, power or thermal variability.

```sh
python3 build/entropy-bit-writing/audit.py
python3 build/entropy-bit-writing/run-producer-qualification.py
python3 build/entropy-bit-writing/run-producer-confirm.py
python3 build/entropy-bit-writing/run-producer-batch-throughput.py
python3 build/entropy-bit-writing/run-small-confirm.py
python3 build/entropy-bit-writing/run-kodak-batch-confirm.py
python3 build/entropy-bit-writing/run-kodak-batch-control.py
```

Controllers retain/resume completed cohorts and validate their identities. Do not
rebuild an earlier binary against the delivered producer source. The permanent
C++ tests and CMake entries provide durable coverage; the build directory contains
local reproduction artifacts rather than installed project tools.
