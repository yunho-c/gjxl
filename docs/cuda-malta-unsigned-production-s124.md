# Production unsigned Malta input addresses (S124)

## Outcome

The paired Malta loader now widens validated coordinates through `uint32_t`
before forming its 64-bit input addresses. The original coordinate bounds
check proves those values nonnegative, so this is an exact integer change.
It applies directly to the existing 8-, 24- and 64-row paired kernels, without
new dispatch, allocation, synchronization, numerical approximation, dependency
or compatibility surface.

This retains a small, verified reduction in address work, not a demonstrated
universal encode speedup. Six uninstrumented 4K windows across S123/S124 favor
the both-coordinate candidate, but duplicate variation is large, HD is slower
in this study's uninstrumented samples, and event-instrumented results are
mixed. Those adverse observations are retained below. The backend is not
proved maxed out.

## Scope and native identity

[S123](cuda-malta-unsigned-address-s123.md) qualified a forced 64-row prototype.
S124 first preserves the old production unit in an immutable baseline source,
then instantiates both its original loader and unsigned loader for all three
tile heights, full/LF response modes and 2D/flat grids. All twelve controls
match their original production bodies exactly.

The production edit is only two local unsigned coordinates and their use in
the existing input-address expressions. Coordinates, halo predicates, lane
mapping, shared locations, floating-point order and initialization/addition
semantics remain unchanged. The standard loop and existing size policy are
preserved; no previous runtime implementation is kept as a new production
fallback. Experimental control code exists only in the diagnostic harness.

A clean Release build uses the same pinned MSVC 14.37, CUDA 11.8 and sm86
configuration as the retained S114 production build. Exactly twelve of the
78 Butteraugli GPU bodies change. Every changed body matches its qualified
unsigned prototype, including instruction encodings and scheduling bits;
the other 66 Butteraugli bodies and 133 linked GPU dependencies are unchanged.
Comparisons cover the fresh production object and normal/compact release and
host-ASAN encode executables. The benchmark object has 102 bodies and its
full-encoder executables have 235; the production encoders have 211.

Each changed specialization has two fewer static instructions. Register,
shared-memory and spill requirements are unchanged. Full/LF 2D kernels use
48/40 registers. Flat 8-row kernels use 48/40; flat 24/64-row kernels use
56/47. Shared storage is respectively 2,560, 5,120 and 11,520 bytes, with zero
stack/spill storage throughout.

All four 64-row candidate bodies also match S123's measured candidate exactly.
Its full 4K captured-input counter result therefore refers to the same native
code: 71,236,245 executed warp instructions instead of 71,963,205, a 1.01%
reduction, with identical FFMA counts and essentially unchanged memory traffic.
No new counter run or 1.01% timing improvement is claimed for S124, and that
percentage is not generalized to other tile heights or inputs.

## Correctness and qualification

The fresh production build passes all 78 CTest tests, with no skips. Its
ordinary Malta test checks 2,944 guarded cases through three stages; the tall
test checks 32 more, including heights 524,280/524,281 and
4,194,240/4,194,241 at width one. Those exercise both historical and current
2D/flat dispatch boundaries. The existing tests cover explicit 8/24/64-row
tiles, full/LF responses, initialization/addition, tails and scalar comparisons.

Additional qualification comprises:

- 25,088 guarded prototype fixtures across release and host-ASAN builds,
  comparing the old production loader and both all-height experimental
  implementations to a scalar oracle through three stages each.
- 1,230 exact encodes from fresh production-linked executables: 960 ordinary,
  222 host-ASAN and 48 encoder memcheck checks. Coverage includes photographs,
  padded HD/4K input, quality/search cases, compact and ordinary storage,
  serial reuse, independent concurrent contexts and the public batch driver.
  Codestream bytes and summaries match the frozen S108 oracle.
- Four production Malta sanitizer jobs, each covering 264 guarded fixtures:
  memcheck, racecheck, initcheck and synccheck report no errors or hazards.
  The four encoder memcheck jobs report no errors or leaks.
- Fourteen expected rejection cases preserve their exact retained error.
- Another 840 exact encodes in the performance/preflight harnesses, including
  30 host-ASAN checks, for **2,070 exact encodes and 252 host-ASAN checks** in
  total. Host ASAN instruments the harness/resident owner, not every linked
  library or GPU code.

All 40 previously retained runtime file hashes remain unchanged. The new
production libraries are in the separate S124 build directory; qualification
does not overwrite the old S114 control build.

## Resident timing

The matched benchmark uses the fully resident path, distance 1.2, effort 7,
automatic CPU thread count and the pool allocator. Each measured process
checks one reference encode, four warm rounds and twelve measured rounds of
four shuffled labels. Two labels use the original production implementation;
two use the unsigned loader at the same dispatch-selected tile height.
Each encode makes 24 eligible Malta calls. A second process order supplies
the other window for each image. All twelve timed jobs are isolated from
other recorded jobs, builds and native-code dumps.

The table gives matched-round medians in milliseconds. Positive deltas mean
the unsigned candidate is slower. Duplicate deltas compare two labels that
run identical baseline or candidate code; they expose substantial variation.
Percentage medians are computed per round, not by dividing table medians.

| Image/window | Baseline encode | Candidate delta | Delta % | Baseline duplicate | Candidate duplicate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flower 500 / 0 | 19.151 | -0.151 | -0.68% | -0.380 | -0.060 |
| Flower 500 / 1 | 20.635 | -0.140 | -0.71% | -0.579 | -0.020 |
| HD / 0 | 73.667 | +0.810 | +1.13% | -2.459 | -0.553 |
| HD / 1 | 84.344 | +1.489 | +1.82% | +2.226 | +1.472 |
| 4K / 0 | 329.621 | -5.358 | -1.63% | +19.734 | -1.063 |
| 4K / 1 | 339.580 | -6.844 | -1.96% | -12.520 | +12.019 |

Separate instrumented jobs sum CUDA-event intervals around eligible Malta
launches. These are not the same observation boundary or jobs as the table
above, and their whole-encode deltas must not be substituted for it.

| Image/window | Instrumented encode delta | Baseline Malta sum | Malta delta | Malta duplicates (baseline, candidate) |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 / 0 | +0.907 | 0.706 | +0.022792 | +0.023040, -0.048128 |
| Flower 500 / 1 | -0.061 | 0.697 | +0.000224 | +0.025696, -0.018944 |
| HD / 0 | -2.323 | 2.971 | +0.016640 | -0.048128, -0.020480 |
| HD / 1 | +4.472 | 2.952 | +0.046336 | -0.019456, -0.015360 |
| 4K / 0 | +5.472 | 27.746 | +1.943552 | +1.818624, +1.190912 |
| 4K / 1 | -13.588 | 29.058 | -0.055552 | -0.467456, +2.006528 |

The performance executables link the retained S114 libraries with a GPU
override; native-code comparisons prove the override's twelve candidate
bodies equal the fresh production bodies and all other GPU code is unchanged.
Fresh production qualification jobs are correctness checks, not additional
wall-time benchmarks. No new replay, Nsight Compute or Nsight Systems
performance run is claimed here.

Read-only NVML queries outside each encode's wall timer record 1,680
power-limit endpoints, all 40 W, with no endpoint changes. This rules out the
observed endpoint-limit changes from S121 in these samples, but does not prove
stable clocks or thermal behavior inside an encode. The harness adds no
artificial idle periods and changes no power, clock, process-priority or
security settings. Production has no NVML dependency. No admin, firewall or
permission blocker was encountered.

## Retention decision and evidence

Retain the two casts as a narrowly scoped, exact address-work reduction with
unchanged resources and verified native equivalence. The 4K observations are
encouraging, but the HD sample regressions and mixed instrumented observations
prevent a universal speedup claim. S123's four 4K windows and S124's two all
favor the candidate without events; they are different campaigns and are not
pooled into a causal effect estimate. Fewer instructions alone do not prove
lower elapsed time.

Evidence lives in `U:/gjxl-cuda-diagnostics/s124/`, including `summary.json`,
`native.json`, `production_native.json`, `s123_transfer.json`,
`production_qualify.json`, `production_asan.json`, `production_memcheck.json`,
`within_analysis.json`, `events_analysis.json`, `within_power_analysis.json`,
the CTest log, job records, native dumps and fresh `build-cuda/` artifacts.
Diagnostic sources are under `build-cuda-ninja/profiles/s124_*`; the archive
contains their snapshots and transitive local Python dependencies.

`s124_validate.py` checks all 168 job records (154 successes and 14 expected
rejections), log/executable/input hashes, native identities, fixture and
encode counts, sanitizer results, timing isolation and retained runtime
hashes. `s124_freeze.py` validates before archiving sources and hashing the
evidence; `s124_validate.py --frozen` checks that archive afterward.

The fully resident backend still has open optimization work. Future changes
need sustained-load and whole-encode qualification, with adverse cases
retained; this result does not settle compact coefficient consumption,
composition/reduction fusion or tile scheduling globally. No compatibility
layer or new runtime tuning policy is introduced by this change.
