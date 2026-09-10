# S173: reuse transform counts for sparse token reservation

## Scope and motivation

Starting revision is `03a5184df10e6220840dfde8f9aaf4ef9bff997e` (S172).
Production remains S168's fully resident CUDA runtime, with native sparse
ownership on and compact-width packing off. S173 investigates host-side
compact coefficient consumption, not new GPU composition/reduction kernels,
tile scheduling, numerical changes or a new sparse storage representation.

S172's factor-four sparse reservation reduced 4K value/context capacity
from 149,446,080 to 6,831,672 bytes. Its coefficient-token worker time
improved locally, but neither AC-tokenize wall nor complete encoding improved
robustly across the corpus. It was not promoted. That prototype counts each
group's sparse masks once to estimate capacity, then performs the existing
per-transform AC nonzero count during tokenization.

S173 tests whether those *required* transform counts can serve both purposes.
The 800-file S172 inventory and 343 unchanged production sources are verified
before preparation. Artifacts are under
`U:/gjxl-cuda-diagnostics/s173-artifacts`, with ignored
`build-cuda-ninja/profiles/s173_*` helpers. Protected untracked notes remain
outside the experiment and are not read.

## Mechanism

After the unchanged anchor validation, the candidate visits each transform
and channel, calling the existing `CountNonzerosExceptLlf`. It stores the
results in a local fixed `1024 × 3` array of `int32_t` and sums them for
reservation. The array's logical storage is **12,288 bytes**, not a claim
about the entire compiler-generated stack frame. The maximum follows from
the 65,536-coefficient group capacity and the minimum 64-coefficient
transform. An explicit anchor-count guard protects its indexing.

The initial value/context reservation is:

```text
min(original maximum, 3 * anchor_count + 4 * AC_nonzero_count)
```

The existing overflow-safe factor calculation is retained. The later token
loop reads the cached count instead of counting that transform again.
It still generates every required intervening zero token and keeps the
same coefficient order, predictor contexts, fixed populations, signed
packing, final validation and transactional output publication.

The count pass checks transform bounds before reading. LLF entries are
excluded by the same function used by production tokenization. S172's
capacity-only pass counted all nonzeros in the active span, including LLF.
Thus the reservations can differ on diagnostic groups with nonzero LLF,
although token output remains identical. The standard completed native
whole-encode frames have zero LLF, so their S172/cached capacities must match.

Dense layouts keep the original reservation and direct nonzero counting.
There is no scratch-structure or public ABI change, retained heap cache,
thread-local cache, new compatibility layer or ISA requirement. The cache
is local to each group invocation. Moving a pass earlier and adding cache
reads/writes can cost time; reduced counting is not assumed to be a speedup.

As in S172, this is an initial-capacity heuristic, not a token bound.
Ordinary vector growth handles late nonzeros and can exceed the original
dense reservation on adverse orders. It never truncates required tokens.
Capacity bytes are not process RSS, committed memory or GPU allocation.

## Same-binary controls

| Mode | Tokenizer |
| --- | --- |
| 0 | Exact original production template |
| 1 | Identical control, same original-template instruction path |
| 2 | Exact S172 extra active-span count and sparse reservation |
| 3 | Cached transform counts and sparse reservation |

The mode-2 template is checked against S172's frozen source after identifier
renaming. A source audit verifies the bounded count-cache transformation,
then reconstructs the original translation unit by removing the two
candidate templates and diagnostic wrapper. The original body remains exact.

An atomic selector is configured only between joined complete encodes.
Workers write independent slots in a common post-success group audit.
The selector, 4,096-slot audit and optional capacity output are diagnostic
only; concurrent reconfiguration is unsupported. Each mode includes its
own actual count/cache costs and the common audit inside the timed call.

Normal builds use optimized MSVC 14.37 C++20 release settings; host ASAN
uses optimized clang-cl 22. The current S168/S167/S166 host overlays and
S162 instrumented support libraries are reused, with unchanged
uninstrumented PFM input support. Nineteen inherited object/library hashes
and CUDA runtime/NVML link libraries are verified. Both whole DLLs retain
all eleven unchanged standard GPU modules; no GPU code is recompiled.

## Token and storage qualification

Each normal/ASAN fixture uses 128 frames and 320 groups: all seven supported
transform strategies plus mixed grids, full and narrow edge groups, eight
coefficient patterns, all three coefficient widths, signed extrema,
natural/custom orders, three context maps and population collection on/off.

Both candidates are compared with original dense and original sparse
tokenization. Every token, context and population field must match.
Aligned forward sparse payloads and unaligned spans with reversed word
payload order are covered. The shifted layout also overwrites each
transform's LLF rectangle with signed extrema and retains nonzero suffix
values outside the used span. This explicitly tests LLF exclusion and
used-span boundaries, not just native frames whose LLF happens to be zero.

The fixture independently checks each candidate's capacity formula, the
unchanged dense capacity, growth fallback, and rejection of short spans
and null scratch/output without changing a sentinel output.

Normal and ASAN each pass **149,760 checks**, including **9,144 growth
cases**: 299,520 checks and 18,288 growth cases total. The build and both
fixtures complete without a rejected job.

## Whole-workflow protocol

All eleven current corpus inputs run in automatic and eight-thread
configurations. Each process first encodes with frozen S168, then checks
every diagnostic encode for exact bytes, full summary, coefficient width,
native-owner size, token counts, initial/final capacity and growing groups.
The capacity oracle uses S172's hash-pinned per-group census and simulates
the current toolchain's vector growth; native frame content is independently
checked against the frozen reference each time.

Normal/ASAN preflight covers all input/thread pairs. Whole CUDA memcheck
covers 4K and initcheck Keong before ordinary timing. Their instrumented
durations are not performance evidence.

Timing uses two process repeats, eight warmup rounds and sixteen measured
rounds, each containing four modes. The four Williams orders are
`0132`, `1203`, `2310`, `3021`: positions and directed predecessors
are balanced. The second repeat reverses both case and order schedules.
No sample is discarded or automatically retried.

The outer window is the complete bridge Encode call. Prior-result clearing,
input/backend construction, reference encoding and post-call validation
are outside it. All 41 host phases and read-only NVML endpoints are recorded.
Coefficient-token worker work is nested work, not additive wall time.
The differential against mode 2 tests count reuse in the same process;
historical S172 timings are not substituted.

## Results and disposition

The ordinary cohort ran **03:46:40–03:55:21 UTC on 2026-09-10**:
44 processes and 4,268 complete encodes including references, with 2,816
measured encodes. All 8,448 warmup/measured NVML endpoints report 40,000 mW.
No GPU profiler, build, sanitizer or compression job overlaps timing.
Every sample and both repeats are retained.

Preflight passed **230 encodes in 46 processes**. Normal and host ASAN
passed all eleven inputs in both thread configurations. CUDA memcheck and
initcheck report zero errors, and memcheck reports zero leaked bytes and
allocations. S173 therefore completes **4,498 whole encodes** in total.

Each range below spans four separate input/thread/repeat cells. Changes
are medians of ratios or differences paired within four-mode rounds;
they are not ratios of independent overall medians. Negative means faster.
The S172 extra-scan and cached-count capacities agree on every standard
input, including their group-growth counts; the dense capacities are
unchanged. This is element capacity, not a measured RSS saving.

| Sparse input | Baseline capacity bytes | Extra-scan / cached capacity bytes | Growing groups |
| --- | ---: | ---: | ---: |
| 1080p | 37,364,112 | 2,021,670 | 7 / 40 |
| 4k | 149,446,080 | 6,831,672 | 0 / 135 |
| 513_p0 | 4,872,996 | 9,492 | 6 / 9 |
| flower_2000 | 72,083,286 | 3,976,902 | 2 / 64 |
| flower_3200x2160 | 124,544,070 | 7,084,302 | 3 / 117 |
| flower_500 | 4,601,016 | 986,232 | 0 / 4 |
| keong_3839x2159 | 149,869,098 | 16,181,112 | 17 / 135 |

### Coefficient-token worker work (not wall)

| Input | Baseline ms | Cached delta vs baseline ms | Cached delta vs extra-scan ms | Cached vs extra-scan | Faster than extra-scan / 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1025_p2 | 74.341 to 78.861 | -0.850 to +2.712 | -5.079 to +5.088 | -5.50 to +6.87% | 2 |
| 1080p | 10.290 to 11.616 | -0.817 to -0.095 | -0.796 to +0.155 | -7.79 to +1.53% | 3 |
| 4k | 29.995 to 32.054 | -4.811 to -3.186 | -3.107 to -0.098 | -9.86 to -0.25% | 4 |
| 513_p0 | 0.486 to 0.524 | -0.090 to -0.008 | -0.031 to +0.024 | -6.31 to +5.57% | 1 |
| 513_p1 | 11.502 to 13.152 | -0.388 to +0.412 | -0.228 to +0.523 | -1.88 to +4.94% | 2 |
| 513_p2 | 14.765 to 18.447 | -0.671 to +1.052 | +0.018 to +0.843 | +0.30 to +5.97% | 0 |
| 65_p2 | 0.297 to 0.338 | -0.028 to +0.001 | -0.008 to +0.007 | -2.25 to +2.23% | 1 |
| flower_2000 | 23.038 to 25.395 | -1.498 to -0.683 | -1.632 to +0.357 | -6.79 to +1.44% | 2 |
| flower_3200x2160 | 37.388 to 37.938 | -3.871 to +1.858 | -3.639 to +4.777 | -10.17 to +13.42% | 2 |
| flower_500 | 3.061 to 3.357 | -0.006 to +0.145 | -0.096 to +0.161 | -3.09 to +5.48% | 1 |
| keong_3839x2159 | 78.035 to 82.120 | -11.038 to -0.818 | -6.357 to +4.131 | -8.23 to +5.64% | 2 |

### AC-tokenize wall

| Input | Baseline ms | Cached delta vs baseline ms | Cached delta vs extra-scan ms | Cached vs extra-scan | Faster than extra-scan / 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1025_p2 | 14.688 to 15.409 | -0.052 to +0.594 | -0.743 to +0.947 | -4.96 to +6.58% | 3 |
| 1080p | 5.905 to 6.603 | -0.058 to +0.062 | -0.255 to +0.218 | -3.61 to +3.28% | 3 |
| 4k | 10.093 to 11.388 | -1.061 to -0.122 | -0.449 to +0.128 | -4.31 to +1.26% | 3 |
| 513_p0 | 1.742 to 2.080 | -0.041 to +0.048 | -0.017 to +0.055 | -0.95 to +2.24% | 2 |
| 513_p1 | 4.485 to 5.645 | -0.002 to +0.179 | -0.125 to +0.259 | -2.27 to +6.55% | 2 |
| 513_p2 | 5.393 to 6.978 | -0.220 to +0.105 | -0.110 to +0.175 | -2.21 to +3.13% | 1 |
| 65_p2 | 1.276 to 1.503 | -0.018 to +0.043 | -0.004 to +0.073 | -0.27 to +5.83% | 1 |
| flower_2000 | 8.519 to 9.184 | -0.327 to -0.010 | -0.575 to +0.291 | -6.11 to +3.57% | 2 |
| flower_3200x2160 | 11.316 to 12.064 | -1.112 to +0.646 | -0.449 to +1.708 | -4.12 to +15.01% | 2 |
| flower_500 | 2.434 to 2.862 | -0.031 to +0.044 | -0.073 to +0.113 | -2.72 to +4.88% | 3 |
| keong_3839x2159 | 19.634 to 20.351 | -0.812 to +0.009 | -0.974 to +1.247 | -4.44 to +6.93% | 3 |

### Complete Encode wall

| Input | Cached vs baseline | Cached vs duplicate | Cached vs extra-scan | Extra-scan vs baseline | Identical control vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1025_p2 | +0.85 to +4.69% | -3.65 to +1.04% | -1.75 to +3.36% | -0.76 to +4.38% | -0.28 to +5.23% |
| 1080p | -1.48 to -0.01% | -5.00 to -1.66% | -2.48 to +0.54% | -1.62 to +3.13% | +0.76 to +4.78% |
| 4k | -4.08 to -0.22% | -4.23 to +0.07% | -4.36 to +2.87% | -3.91 to +1.27% | -4.68 to +3.24% |
| 513_p0 | -1.02 to +2.11% | -0.05 to +2.85% | +0.04 to +1.70% | -1.73 to +0.65% | -0.99 to +0.22% |
| 513_p1 | -1.79 to +0.87% | -1.72 to +0.85% | -1.20 to +2.50% | -3.65 to +1.75% | +0.15 to +3.87% |
| 513_p2 | -2.51 to +1.51% | -4.75 to +1.13% | -1.42 to +0.22% | -0.27 to +0.53% | -1.87 to +2.24% |
| 65_p2 | -1.56 to +2.40% | -0.92 to +2.34% | +0.66 to +2.41% | -2.60 to -0.24% | -2.91 to +0.01% |
| flower_2000 | -3.51 to +2.63% | -1.52 to +1.27% | -1.29 to +4.22% | -4.94 to +6.14% | -3.34 to +3.56% |
| flower_3200x2160 | -2.91 to +2.69% | -4.41 to +3.67% | -0.24 to +6.48% | -2.04 to +2.37% | -2.88 to +3.47% |
| flower_500 | +0.17 to +1.61% | -0.61 to +1.88% | -1.94 to +0.59% | -0.49 to +1.66% | -1.20 to +2.22% |
| keong_3839x2159 | -5.04 to +0.23% | -0.74 to +2.03% | -1.64 to +2.51% | -4.08 to +2.52% | -3.17 to +0.51% |

### Interpretation

For 4K, cached coefficient-token worker work improves by **3.186–4.811 ms**
against production, or **11.65–15.31%**. It also improves against S172's
extra-scan implementation in all four cells: **0.098–3.107 ms**, or
**0.25–9.86%**. This is direct same-process evidence for a useful local
count-reuse result, not an extrapolation from the prior stage.

The 4K AC-tokenize wall improves against production in all four cells by
0.122–1.061 ms, but is mixed against extra-scan (−0.449 to +0.128 ms).
Complete Encode is faster than production in all four cells, but one cell
is +0.07% against the identical control and one is +2.87% against extra-scan.
Nested worker-time savings must not be added to wall-time savings.

Keong worker work improves against production in all four cells, but its
cached-versus-extra-scan worker comparison changes from −8.23% to +5.64%.
The larger flower's corresponding range is −10.17% to +13.42%; its whole
cached-versus-extra-scan range is −0.24% to +6.48%. Count reuse does not show
a consistent benefit over extra-scan across the corpus.

Only 1080p beats both production controls in all four whole-workflow cells.
No input beats production, identical control *and* extra-scan in all four.
The dense unchanged-reservation controls also vary; they do not establish
a sparse count-cache benefit. These results do not establish performance
equivalence or identify every cause of the residual variation.

**Do not promote the cached-count candidate or S172 reservation.**
Production defaults, token-buffer reservation, public interfaces and
scratch ABI remain unchanged. The capacity reduction and 4K worker result
are useful, qualified observations, but not a robust general
complete-encoder speedup.

The next investigation should return to the dominant resident-pipeline
work and actual synchronization boundaries in the existing profile.
Another reservation-factor sweep is not justified by this result alone.
Device gaps must not be assumed to be removable host launch overhead,
and local improvements must still be checked in the complete workflow.

## Preservation and limits

All 93 journaled jobs are accepted: one build, two host fixtures, 46 whole
preflight processes and 44 ordinary timing processes. The archive retains
sources, controls, native modules, raw logs, protocols and parsed results.
Verification checks the complete predecessor inventory, 343 production
sources, inherited support hashes, exact S172 template, cached transformation,
fixture counters, every log parse and capacity prediction, all 41 phase
statistics, derived tables and the non-promotion decision.

No fresh CTest suite, installation test, external decoder run, independent
working-set measurement, simultaneous-frame throughput experiment,
stack-frame disassembly or exhaustive cache representation sweep is claimed.
The 12,288-byte number is the logical local array size, not measured
stack-frame usage. Whole timing covers the existing default int32 ownership;
the new direct-token fixtures additionally cover int8/int16.

No firewall/admin blocker was encountered. No power, clock, thermal,
priority, affinity, firewall or security setting changed. No predecessor
was modified or rebuilt, and nothing was deleted. The optimization goal
remains active; no "maxed out" conclusion is made.
