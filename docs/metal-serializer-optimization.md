# Shared serializer optimization for Metal

Measured on 2026-09-08 on Apple M4 Pro (Mac16,7, 48 GiB), macOS 15.6,
Apple Clang 17.0.0, Release C++20/libc++. Baseline:
`6e39e381cbd6b6b28c91f2bba6d716dde99400e5` on `perf/metal-kernel-dataflow`.
The candidate is the reviewed working-tree serializer patch, frozen under
`build/serializer-optimization/candidate-src` with per-file SHA-256 manifests.

The shared CPU serializer now uses the CUDA branch's lightweight token results,
contiguous coefficient counting, and private ANS histogram reuse. Across the
four 1080p/4K cases, median paired serializer latency falls **8.7–16.9%** and
complete warm workflow latency falls **3.7–6.7%**, with exact output parity.
Both CPU and Metal workflows use this code. The Metal shader payload is
byte-identical to the baseline.

## Implementation and contracts

- AC nonzero counting reduces the contiguous coefficient plane, then subtracts
  its LLF rectangle. Direct token accumulation returns a private error enum;
  the public Status is constructed on failure. Sparse population order, context
  selection, signed packing, overflow checks, and output publication stay exact.
- Coefficient-order counting uses branch-free contiguous updates and `uint32_t`
  populations when the validated frame's block area fits. Larger frames retain
  `uint64_t`. Anchors partition the block area, so each counter, including its
  worker reduction, is bounded by that area. Sampling decisions, worker
  orchestration, float-scaled sort keys, natural-order ties, and LLF prefixes
  are preserved. The storage planner selects the same counter width.
- ANS state advancement returns a static error message or null. The private
  constexpr HybridUint converter takes an already validated configuration;
  stream configurations are checked before the recurrence. The public converter
  keeps its checked, atomic interface. The token-scanned histogram loop also
  uses this converter after validating its configuration and context map.
- Farthest-first ANS clustering caches Shannon costs in its existing private
  source array. Log-table initialization is resolved outside the inner symbol
  sums; the ordered double arithmetic is preserved. Validated fixed populations
  without an initial map are borrowed synchronously through small views.
  Preclustered/scanned sources and selected/merged clusters own their counts.
  No borrowed population reference escapes partition construction.
- DC gradient contexts use a constexpr 1024-byte lookup expanded from the same
  pinned runs. Successful residual emission appends directly, keeping the
  original int32 overflow diagnostic and atomic destination replacement.

Adapted CUDA commits: `2260047` (S41), `0f5a7a7` (S42), `efda0a0` (S43),
`51790b8` (S44), `355180e` (S51), `5ea11e6` (S52), `7e5be16` (S53),
`1caf9a5` (S56), and `bf4968b` (S58). This adaptation preserves the Metal
branch's managed `Storage`, frame views, allocation failure handling, and
parallel coefficient-order implementation. Apple Clang's retained optimization
remarks confirm vectorization of the two contiguous counting loops.

## Qualification

| Check | Fresh result |
| --- | --- |
| Full Release CTest, baseline | 121/122 passed |
| Full Release CTest, candidate | 123/124 passed, including two new test targets |
| Host ASan/UBSan | 7/7 targeted suites passed, unsuppressed; leak detection disabled |
| Independent coefficient-order oracle | 218 frames/policies, ties, LLF prefixes, sampled/group boundaries, atomic errors |
| Direct AC oracle | 4096 group cases, exact tokens, coordinate-wise nonzero counts, fixed populations, scratch reuse |
| HybridUint inverse oracle | 816 valid configurations, 443904 values; invalid configurations preserve output |
| DC scalar run oracle | 6148 guarded/clamped cases plus positive/negative residual overflow |
| ANS validation | Exact division-based emission, scanned/borrowed/preclustered parity, sparse/empty contexts, malformed bins and 40 late-section failures |
| Frozen representation oracle | 128 frame/policy order outputs plus block-map outputs, byte-identical (547072 bytes) |
| Frozen serializer oracle | 72 fixture/policy codestreams, byte-identical (2184980 bytes) |
| Canonical corpus/policy comparison | 56/56 byte-identical, including efforts 1–10 and high-density/maximum-compression modes |
| Pinned decoder | 3 corpus decode pairs match; all 22 conformance fixtures pass per build |
| Resident resources/lifecycle | 8 scenarios pass, including tight/full limits and two callers; 24 retained outputs match |

Both full suites fail only the inherited `quantization_pipeline` score fixture:
`actual=0.24919039011001587 expected=0.24914586544036865` at index 1.
The test inventories and failure diagnostics were checked from fresh JUnit
outputs. The decoder is libjxl `e8ff09762481785938d8e4e01333ed3917571161`.

The managed tests exercise one/two/eight coefficient-order workers, admitted
bounds, rejection before physical allocation, output atomicity, failure recovery,
and charge release. Entropy storage tests additionally sweep allocation failures
on the borrowed population path. Counter-width boundary planning is checked
without constructing a multi-billion-block frame. The sanitizer scope is host
serializer primitives and token/entropy/representation storage tests.

## Warm performance

Seven alternating independent-process pairs per input, three warmups and seven
timed samples per process: 84 process records and 588 timed encodes. Policy is
fully-resident Metal, effort 7, distance 1.2, fused-tuned residual inverse, with
GPU stage profiling disabled. The existing measurement controller rejects
competing encoder/build/test processes. File I/O and process startup are outside
the timed public workflow; backend creation is not a separate cold-start study.

Displayed milliseconds are medians of the seven process medians per build.
Percentages are medians of the seven *paired* percentage changes, so dividing
the two displayed medians need not reproduce the percentage. Negative is faster.

| Input | Serializer ms | Paired change | Complete workflow ms | Paired change |
| --- | ---: | ---: | ---: | ---: |
| Synthetic 128x96 | 2.264 → 1.551 | -30.65% | 10.883 → 10.516 | -6.48% |
| Padded 1919x1079 | 19.712 → 17.858 | -8.67% | 70.543 → 67.406 | -3.96% |
| Padded 3839x2159 | 42.837 → 37.618 | -13.69% | 223.557 → 214.701 | -3.70% |
| Kodak 17 (512x768) | 8.522 → 6.492 | -24.02% | 24.281 → 22.223 | -7.45% |
| Planter 1920x1080 | 24.856 → 20.869 | -15.20% | 75.817 → 71.095 | -5.60% |
| Planter 3840x2160 | 47.835 → 39.561 | -16.87% | 231.571 → 217.911 | -6.70% |

Serializer latency improves in every one of the 42 pairs. Complete-call
measurements remain noisier: synthetic 128x96 pairs range from -25.61% to
+20.58%, padded 4K from -7.74% to +1.39%, and Kodak from -15.36% to +0.34%.
Both natural-image HD/4K runs improve in all seven complete-call pairs. These
are workload-specific warm results, not a claim about every image or cold calls.

The 4K elapsed serializer phases help locate the gains:

| Elapsed phase | Padded 4K | Planter 4K |
| --- | ---: | ---: |
| DC tokenization | -33.73% | -25.70% |
| AC tokenization | -16.86% | -27.79% |
| Entropy optimization | -12.72% | -19.42% |
| Section writing | -4.79% | -2.18% |

Coefficient-order **aggregate worker work** falls 27.28% on padded 4K and 42.91%
on Planter 4K. It is not elapsed latency and is not added to the phase timings.
The retained phase data measures the combined port; there was no per-commit
Apple ablation. Section-writing gains are modest here compared with the CUDA
study, while AC/DC tokenization and counting show clearer improvements.

## Storage and tradeoffs

The direct ANS plan now covers one owning source histogram array instead of
two. That is a conservative bound for both owning and smaller borrowed sources;
it does not assume a fixed-population call when planning generic direct ANS.
A source histogram is 2080 bytes and a borrowed view is 40 bytes on this ABI.
At 6930 contexts, replacing two original arrays with views removes 27.23 MiB
of source-array backing in that phase; the caller's populations and mutable
cluster counts remain. The generic source-array plan shrinks by 13.75 MiB.

At the five-family/eight-worker bound, narrow coefficient populations save
214272 bytes (0.204 MiB). Actual savings depend on present families and workers.
The resource qualification observed **no consistent whole-workflow peak backing
reduction**; the single-4K peak remained 2481.484 MiB in both builds, and mixed
scheduling can change overlap. Phase-local copy/allocation savings are not an
RSS or total peak-memory claim. The DC table costs 1 KiB of immutable storage,
and the recorded `gjxl_encode` executable grew by 25872 bytes on this toolchain.

## Evidence and reproduction

All fresh evidence is under `build/serializer-optimization/`:

- `baseline-src`, `candidate-src`, `baseline`, `final`, `asan`;
  `baseline-identity.json`, both `*-source-hashes.json`, `candidate.patch`,
  `toolchain.json`, and counting-loop assembly/optimization remarks.
- `baseline-tests.xml`, `final-tests.xml`, `asan-tests.xml` and matching logs;
  oracle binaries/manifests, `conformance.json`, `qualification-summary.json`.
- `parity/`, `resources/`, and `wall/`, each retaining commands, identities,
  outputs and raw logs/samples; `evidence-hashes.json` seals the completed
  evidence, and `wall-audit.json` records independent recomputation
  of all process medians, paired percentages, summary values and file hashes.

Build each frozen source with Ninja, Release, tests and benchmarks enabled,
`GJXL_ENABLE_LIBJXL_REFERENCE=OFF`. Run CTest with `SDKROOT` from
`xcrun --show-sdk-path`, serially. ASan/UBSan uses C++ flags
`-fsanitize=address,undefined -fno-omit-frame-pointer`, with
`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`.

The committed `tools/metal_dataflow/parity.py`, `resources.py`, and `compare.py`
accept the two source/build paths and an explicit canonical corpus/decoder.
For timings, supply the six inputs above to `compare.py` using `--workload` for
the three synthetic cases and `--input` for the three canonical PFMs, with
`--pairs 7 --warmups 3 --samples 7`. Exact commands and input/tool SHA-256 values
are retained in the evidence, including the original canonical corpus path.
The final runtime/build/test source files were checked against the frozen
candidate after qualification. The two existing storage-contract documents
were updated to describe the new lifetimes and counter width.
