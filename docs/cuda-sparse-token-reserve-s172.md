# S172: sparse-aware AC token-buffer reservation

## Scope and baseline

Starting revision is `4691f4dcb8ad30c2099dbb6ef6b3eb45b14f2bed` (S171).
Runtime remains S168: fully resident CUDA, native sparse ownership on and
compact-width packing off. This experiment follows compact coefficient
consumption into the CPU serializer; it changes neither sparse packing nor
GPU composition, reduction, tile scheduling, quantization or numerical results.

Artifacts are under `U:/gjxl-cuda-diagnostics/s172-artifacts`; helpers are
ignored `build-cuda-ninja/profiles/s172_*` files. Preparation verified the
714-file S171 inventory and archived 343 unchanged production files.
Protected untracked notes are outside the experiment and are not read.

## Why token capacity matters

The direct serializer primitive in `src/codestream/ac_group.cpp` reserves
`3 * (used_coefficient_count + anchors.size())` elements in each group's
`uint32_t` value and `uint16_t` context vectors, regardless of native
coefficient layout. This is a dense worst-case reservation. The encoder
moves these vectors into `direct_groups`, retaining their capacity while
building entropy streams and writing sections.

The existing tokenizer already stops after the last nonzero in the chosen
coefficient order. Zeros *before* that last nonzero still produce tokens and
contexts. Nonzero count is therefore not an exact token count or a tight
upper bound. A late-position nonzero can require a long run of zero tokens.

A baseline-only overlay preserves the complete original tokenizer body and
audits successful groups afterward. Its additional nonzero-count scan is
diagnostic overhead, not a timing result. Eleven complete inputs run in
automatic and eight-thread configurations: 22 census encodes plus 22 frozen
S168 reference encodes. Every output, full encoding summary, coefficient
width and native-owner size matches; per-group census data match between
thread modes.

The 4K case uses 592,579 tokens (3,555,474 bytes of values and contexts) but
retains 149,446,080 bytes of element capacity. Keong uses 9,092,628 bytes
against 149,869,098 bytes of capacity. These are observed vector capacities,
not measurements of resident RAM, working set, committed process memory,
allocator metadata or a proven allocation-time bottleneck.

## Candidate and controls

The candidate reserves, for native sparse groups only:

```text
min(original maximum, 3 * anchor_count + 4 * active_span_nonzero_count)
```

The sum covers all three channels and only each span's used prefix.
Multiplication is guarded by comparing nonzeros with the remaining
coefficient allowance divided by four. Dense layouts keep the original
reservation. All token generation, contexts, fixed populations, validation,
error handling and output publication remain unchanged.

This is an initial-capacity heuristic. Underestimates use ordinary vector
growth and do not truncate tokens or introduce a token limit. Geometric
growth may exceed the original reservation on adverse coefficient orders;
this is not a universal memory bound. Allocation failure still leaves the
caller output unpublished through the existing local candidate object.

The baseline census considered factors two, four and eight. Factor two
underestimates every group in both larger flower images. Factor four avoids
growth in the 4K and flower-500 cases, with some growth elsewhere. Factor
eight doubles the initial nonzero allowance but still underestimates some
Keong and constant-image groups. Only factor four is implemented and timed
in this stage; this is not an exhaustive tuning sweep or a claim of optimality.

One diagnostic DLL supplies three modes:

| Mode | Tokenizer |
| --- | --- |
| 0 | Exact original direct-token template |
| 1 | Identical control, same original-template instruction path |
| 2 | Separate template clone with the sparse initial reservation |

The candidate includes the cost of an extra sparse-mask count pass.
Baseline/control do not receive an artificial matching count pass. All
modes have the same post-success capacity audit. Each worker writes its
independent group slot; configuration and reading occur only outside joined
complete encodes. The atomic mode selector and 4,096-slot audit limit are
diagnostic-only, with no support for concurrent diagnostic reconfiguration.

A source audit reconstructs the original translation unit from each overlay
by removing instrumentation and, for the candidate, its duplicated template.
It also checks that the candidate template differs only by reservation and
an optional diagnostic initial-capacity output.

Normal builds use MSVC 14.37; host ASAN uses clang-cl 22. Both are optimized
C++20 release builds. The current S168/S167/S166 host overlays and S162 ASAN
support are reused, with the unchanged uninstrumented PFM reader.
Nineteen inherited library/object hashes and both CUDA link libraries are
verified separately. Census, normal candidate and ASAN candidate DLLs each
contain all eleven unchanged standard GPU modules. No GPU code is rebuilt.

## Qualification

The host fixture covers 128 deterministic frames and 320 groups, including
narrow right/bottom groups, all seven transform strategies and mixed grids,
eight coefficient patterns, all-zero and late-only nonzeros, signed extrema,
all three coefficient widths, natural/custom orders, three context maps,
and population collection on/off.

Each case compares original dense, original sparse, candidate sparse and
candidate dense tokens, contexts and every population field. Two storage
layouts cover aligned forward payloads and unaligned spans with reversed
word-payload order, plus nonzero suffixes outside the used span. Capacity
formulas and unchanged dense capacity are checked. Short coefficient spans
and null scratch/output are rejected without changing a sentinel output.

Normal and ASAN each pass 74,880 checks, including 4,572 reservation-growth
cases. The total is 149,760 checks and 9,144 growth cases.

The first reservation build omitted the DLL object inputs and failed with
LNK2001. The second repaired that but linked the included tokenizer twice
into the standalone fixture, failing with LNK2005. Neither reached candidate
execution. Distinct v3 build paths correct both link recipes; original
sources, partial outputs, failed logs and job records are retained.
These were diagnostic build errors, not firewall or elevation failures.

## Whole-workflow protocol

The eleven-input corpus includes six image inputs, one sparse constant
synthetic input and four dense synthetic controls. Each has automatic and
eight-thread configurations, normal/ASAN preflight, and a frozen S168
reference encode per process. CUDA memcheck covers 4K and initcheck Keong.

Ordinary timing uses two process repeats per input/thread configuration,
six warmup rounds and eighteen measured rounds, each containing all three
modes. Six balanced mode orders distribute positions and predecessors;
the second repeat reverses both the case schedule and order schedule.
No samples are discarded or automatically retried.

The outer interval covers the complete bridge Encode call. Input/backend
construction, frozen reference encoding, prior-result clearing and
post-call validation are outside it. All 41 host phases are recorded.
AC-tokenize wall is the mechanism-specific phase; nested coefficient-token
worker time is not additive wall time. NVML power-limit endpoints are outside
the timed interval and do not change settings.

Every encode checks exact bytes, full summary, coefficient width, native
storage and token capacities. The parser independently predicts initial
capacity from the per-group census, then simulates this toolchain's geometric
vector growth to check final capacity and the number of growing groups.
This observed growth policy is an experiment check, not a production
assumption or compatibility layer.

## Results and disposition

The ordinary cohort ran **03:15:49–03:22:25 UTC on 2026-09-10**:
44 processes, 3,212 encodes including references, and 2,376 measured encodes.
All 6,336 warmup/measured NVML endpoints report 40,000 mW. No build, GPU
profiler, sanitizer or compression job overlaps the ordinary cohort.
All samples and both repeats are retained.

Preflight completes 184 encodes across 46 processes. Normal and host ASAN
pass all eleven cases in both thread modes. CUDA memcheck and initcheck
report zero errors; memcheck reports zero leaked bytes/allocations.
Together with the baseline census, S172 completes **3,440 whole encodes**.

For each input the timing ranges below cover four separate input/thread/
repeat cells. Changes are medians of ratios paired within three-mode rounds,
not ratios of unpaired overall medians. Negative means faster.
Capacity is identical across repeats and thread configurations.

| Input | Used token bytes | Baseline capacity bytes | Candidate capacity bytes | Growing groups |
| --- | ---: | ---: | ---: | ---: |
| 1025_p2 | 19,163,790 | 19,469,970 | 19,469,970 | 0 / 25 |
| 1080p | 1,219,302 | 37,364,112 | 2,021,670 | 7 / 40 |
| 4k | 3,555,474 | 149,446,080 | 6,831,672 | 0 / 135 |
| 513_p0 | 7,530 | 4,872,996 | 9,492 | 6 / 9 |
| 513_p1 | 4,268,178 | 4,919,436 | 4,919,436 | 0 / 9 |
| 513_p2 | 4,864,230 | 4,943,250 | 4,943,250 | 0 / 9 |
| 65_p2 | 92,988 | 94,770 | 94,770 | 0 / 1 |
| flower_2000 | 3,056,130 | 72,083,286 | 3,976,902 | 2 / 64 |
| flower_3200x2160 | 5,171,502 | 124,544,070 | 7,084,302 | 3 / 117 |
| flower_500 | 582,432 | 4,601,016 | 986,232 | 0 / 4 |
| keong_3839x2159 | 9,092,628 | 149,869,098 | 16,181,112 | 17 / 135 |

| Input | Baseline AC-tokenize ms | Candidate AC-tokenize delta ms | Candidate AC-tokenize change | Control AC-tokenize change | Faster than both / 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1025_p2 | 14.315 to 15.193 | -0.001 to +0.866 | -0.02 to +5.88% | -2.22 to +3.75% | 1 |
| 1080p | 5.683 to 6.529 | -0.061 to +0.267 | -0.97 to +4.55% | -4.20 to +0.17% | 0 |
| 4k | 10.359 to 11.037 | -0.406 to +0.118 | -4.08 to +1.25% | -3.23 to +4.22% | 2 |
| 513_p0 | 1.802 to 2.048 | -0.108 to +0.045 | -5.55 to +2.26% | -0.28 to +3.24% | 2 |
| 513_p1 | 4.760 to 5.298 | -0.249 to -0.007 | -4.98 to -0.15% | -1.72 to +1.43% | 1 |
| 513_p2 | 5.015 to 5.392 | -0.129 to -0.010 | -2.60 to -0.19% | -2.39 to +0.04% | 1 |
| 65_p2 | 1.300 to 1.684 | +0.005 to +0.051 | +0.26 to +3.99% | +0.02 to +3.58% | 0 |
| flower_2000 | 8.227 to 9.486 | -0.579 to +0.335 | -6.29 to +3.88% | -5.37 to +4.85% | 1 |
| flower_3200x2160 | 11.598 to 11.987 | -0.249 to +0.344 | -1.87 to +3.01% | -4.22 to -0.36% | 0 |
| flower_500 | 2.530 to 2.905 | -0.143 to -0.036 | -5.52 to -1.31% | -3.96 to -1.06% | 2 |
| keong_3839x2159 | 18.540 to 20.293 | -1.370 to +0.502 | -6.75 to +2.69% | -6.06 to +1.96% | 1 |

| Input | Candidate whole vs baseline | Candidate whole vs duplicate | Identical control vs baseline |
| --- | ---: | ---: | ---: |
| 1025_p2 | -0.89 to +2.37% | -2.96 to +3.56% | -3.38 to +1.91% |
| 1080p | -2.80 to +0.02% | -1.19 to +0.72% | -1.74 to +0.73% |
| 4k | -6.28 to +2.35% | -6.21 to +1.39% | -0.29 to +4.56% |
| 513_p0 | -0.98 to +0.94% | -0.75 to -0.07% | -0.54 to +1.17% |
| 513_p1 | -1.40 to +0.41% | -0.84 to +1.57% | -0.87 to +2.10% |
| 513_p2 | -1.87 to -0.21% | -3.00 to +1.23% | -1.50 to +0.93% |
| 65_p2 | -11.21 to +2.09% | -0.21 to +0.16% | -3.56 to +1.04% |
| flower_2000 | -3.67 to +1.67% | -0.12 to +0.89% | -2.02 to +2.69% |
| flower_3200x2160 | -2.27 to -0.22% | -0.55 to +2.57% | -1.41 to +1.03% |
| flower_500 | -1.96 to +0.16% | -1.13 to +0.73% | -1.86 to +0.98% |
| keong_3839x2159 | -5.35 to +6.84% | -1.75 to +2.04% | -2.52 to -0.52% |

The 4K capacity reduction is **142,614,408 bytes (95.43%)**. Keong saves
**133,687,986 bytes (89.20%)**. These are value/context element-capacity
savings only. They do not establish an equal reduction in process RSS,
committed memory, GPU memory or total encoder peak allocation.

For 4K, coefficient-token worker work improves by 2.108–4.016 ms,
or 6.51–13.25% against baseline and 8.52–10.61% against the identical control,
across all four cells. This is a useful local signal, but nested worker work
does not translate one-for-one into wall time. AC-tokenize wall changes
from −0.406 to +0.118 ms; it beats both controls in only two of four cells.

Keong's coefficient-token worker work changes from −6.017 to +2.464 ms.
Its AC-tokenize wall changes from −1.370 to +0.502 ms, and whole Encode
changes from −5.35% to +6.84% against baseline. A robust corpus-wide speed
benefit is not established. The smaller and dense controls also show
variation even when their reservation is unchanged; they are not evidence
for a sparse-reservation gain.

**Do not promote this candidate.** No input beats both controls in all four
whole-workflow cells. The clear capacity saving and 4K worker-work result
are not presented as a complete-encoder speedup. Production reservation,
public interfaces and defaults remain unchanged. This does not prove
equivalent speed, a fundamental performance ceiling, or that smaller
reservations cannot help another workload.

A concrete follow-up is to combine reservation counting with the transform
nonzero counts already needed by tokenization, avoiding the additional
header pass. That would need its own bounded scratch-storage design,
growth/pathological-order qualification and fresh whole-workflow evidence.
It is not implemented or claimed faster in S172.

## Preservation and limits

The journal inventory contains 118 jobs: 116 accepted and two rejected
diagnostic builds. Failed outputs and corrected v3 artifacts are retained.
The source audit, inherited support hashes, census records, build/module
hashes, fixture counters, complete log parses, capacity growth simulation,
all 41 phase statistics and tables are checked before freezing.

No fresh CTest suite, installation test, external decoder run, host
working-set study, simultaneous-frame throughput test or exhaustive
reservation-factor search is claimed. This diagnostic-only stage relies on
its new exact-reference whole encodes, detailed token/population fixtures,
host ASAN and whole CUDA sanitizers; it changes no production code.

There were no firewall/admin blockers. No power, clock, thermal, process
priority, affinity, firewall or security setting was changed. Nothing was
deleted and no predecessor artifact was rebuilt or modified. The broader
optimization goal remains active; this is not a "maxed out" conclusion.
