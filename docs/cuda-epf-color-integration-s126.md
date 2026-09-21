# Resident EPF/color fusion integration (S126)

Starting revision: `7fb47fd`, branch `feat/cuda`; Windows, MSVC 14.37,
CUDA 11.8, RTX 3060 Laptop / sm86, 2026-09-08.

**Outcome:** the integrated candidate is bit-exact and saves a repeatable
EPF/color stage interval on actual encoder inputs. All 24 individual
candidate-versus-control stage medians are favorable; the primary paired
4K saving is 1.61 ms per encode (about 29%). Uninstrumented whole-encode
results remain mixed, including a slower 4K window. Proceed to clean
production-build qualification without claiming a general encode speedup.
The tracked runtime remains S124 in this study.

## Integration and ownership

[S125](cuda-epf-color-fusion-s125.md) established an isolated final-EPF/color
candidate with roughly half the measured boundary DRAM traffic. S126 overlays
the resident owner and its qualified GPU object on the S124 production
libraries. It uses the same binary GPU object, not a recompiled approximation
of the candidate. Production source remains unchanged during this study.

At the last EPF pass, the perceptual route either runs the existing EPF then
color conversion or invokes S125's fused kernel into `reconstructed_linear_`.
Earlier EPF passes, Gaborish, sigma preparation and reconstruction are
unchanged. Final pass 1 serves one EPF iteration; pass 2 serves two or three.
Zero EPF iterations retain ordinary color conversion. Maximum-error
evaluation retains its filtered-XYB materialization because its error
reduction consumes `FinalFilteredImage()`; this is a live data requirement,
not a compatibility fallback. No allocation, transfer ownership or public
API changes are introduced by this overlay.

All 211 original GPU bodies match S124. The full diagnostic executables add
only S125's four bodies (two controls, two fused candidates), for 215 total.
Standalone layout/replay probes contain the same fourteen-body S125 object.
The 8/24/64-row Malta address change and all other S124 kernels remain intact.
Host ASAN instruments the harness and resident owner, not every library or
GPU code. Diagnostic selectors, counters and CUDA events are not proposed
production APIs or compatibility layers.

## Layout and hot-input checks

The guarded S125 fixture is adapted to coding-stride inputs/intermediates and
packed source-width RGB output, with additional input/sigma padding as a
second layout. Release and host-ASAN each pass 27,648 guarded pipeline
executions. All four CUDA sanitizers each pass another 3,456, covering the
same exceptional values, bypass boundary, custom scales and changed-input
reuse described in S125: 69,120 executions in total. The fused intermediate
planes remain untouched; output and input/row/allocation guards remain exact.

Separate diagnostic encodes capture both actual final-EPF inputs for Flower
500, HD and odd 4K. The capture waits for the producer stream, reads only
logical rows (not uninitialized padding), and preserves original strides,
pass and float parameter bits in the header. A subsequent fused encode
matches the frozen codestream and the reference summary, adding six checked
encodes across the three capture jobs. Capture synchronization is not part
of the timing executables.

All six captures are final pass 2. At 3839x2159, observed input and
intermediate strides are 3840, RGB stride is 3839 and sigma stride is 480.
Replay restores those actual pitches with poisoned unused cells and compares
original, native-identical separate control and fused output through three
reuses. Release, host-ASAN and memcheck each pass 54 guarded executions
(162 total); memcheck reports zero errors and leaks. These hot-input checks
close S125's synthetic/padded-output gap without claiming a new hot-replay
timing campaign.

## Profile, reuse and full-encode qualification

The existing CUDA AQ test passes with the integrated candidate in both
release and host-ASAN. It retains CPU differential checks, bounded/full
resident behavior, exact and resident maximum-error paths, deferred metadata,
prepared reuse, failure atomicity and concurrent public-workflow tests.
This is a relinked test, not a new full CTest run.

A separate matrix explicitly compares baseline and candidate across two
contents, zero/one/two/three EPF iterations, Gaborish on/off, ordinary/custom
filter and intensity parameters, perceptual/maximum-error control, and
zero/two AQ updates: 128 cases per build. Quant fields, block maps, score
histories and complete RGB planes are bitwise equal; maximum-error results,
frame populations and serialization outcomes also match. Each case asserts
whether fusion, zero-EPF bypass or the maximum-error route actually ran.

Both builds pass all 128 pairs. Eight pairs per build serialize identically;
the other 120 match existing frame-writer rejections of unsupported profiles,
with unchanged empty output. Those are successful AQ-result comparisons, not
240 successful codestream encodes. The initial release/ASAN matrix harness
incorrectly demanded successful serialization for every valid AQ profile and
stopped at the existing Gaborish-profile rejection. Its source, binaries and
failure logs are retained; v2 checks matching serialization outcomes without
changing GPU or resident-owner code.

The S108 frozen-oracle campaign runs against fixed-candidate dense and compact
overlays: 960 ordinary exact encodes, 222 under host ASAN and 48 under encoder
memcheck, plus fourteen matching expected rejection cases. These cover
photographs, padded HD/4K, quality/final-score settings, byte/max-error search,
extreme finite ranges, compact overflow decisions, serial reuse, independent
contexts and the public batch driver. All four encoder memchecks report zero
errors and leaks. Compact ownership remains opt-in.

The fixed-candidate release harness does not emit the optional inline-object
destructor counter report; the host-ASAN build does. Its selector is explicitly
initialized to candidate and never changed by the S108 driver. Explicit route
assertions in the matrix and per-encode timing harnesses provide direct
branch-count checks. The missing release footer is not treated as proof of
per-request route counts in the larger S108 cohort.

## Timing protocol and audit scope

Separate within-process wall-time and CUDA-event executables compare duplicate
baseline labels 0/2 with duplicate candidate labels 1/3. Each reference and
subsequent encode must match frozen bytes, the reference summary, AC width
and owner size. Every encode asserts two eligible final-EPF/color calls;
the event executable also asserts two intervals. Its sum spans both complete
EPF/color boundaries, not just the fused kernel versus EPF alone.

Flower 500, HD and odd 4K each run twice in reversed case order, with four
warm and twelve measured balanced/shuffled rounds of four labels per job.
Distance is 1.2, effort 7, CPU thread count automatic, pool allocation and
final scoring off. Backend lifetime, image I/O, comparisons and returned
result destruction are outside the complete-call wall timer. Event results
are a separate instrumented population. Preflights use release and host-ASAN.

Some correctness preflights overlap host-ASAN qualification, and CPU native
dumps overlap correctness work. Those jobs provide output/safety evidence,
not performance measurements. The measured campaigns must start only after
the other jobs finish; the validator checks all measured windows against
every recorded job. Read-only NVML endpoint queries sit outside each encode's
wall timer. No clock, power, process-priority, affinity or security setting
is changed, and no artificial idle period is inserted. Equal power-limit
endpoints do not establish stable clocks within an encode.

## Measured results

The tables report matched-round medians in milliseconds. Negative deltas
mean faster. Each round averages baseline labels 0/2 and candidate labels
1/3 before subtraction; percentages are also formed per round. Duplicate
columns compare labels running identical code. Separate table medians are
not additive and are not confidence intervals.

Uninstrumented complete encodes:

| Input / window | Baseline encode | Candidate delta | Paired change | Baseline duplicate | Candidate duplicate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flower 500 / 0 | 18.890 | -0.013 | -0.07% | -0.438 | +0.055 |
| Flower 500 / 1 | 19.487 | -0.192 | -1.01% | +0.092 | -0.207 |
| HD / 0 | 72.513 | +0.102 | +0.14% | +1.086 | -1.229 |
| HD / 1 | 78.835 | -0.775 | -1.03% | +2.662 | +1.675 |
| 4K / 0 | 301.069 | +3.496 | +1.19% | -4.563 | -11.770 |
| 4K / 1 | 302.888 | -1.916 | -0.63% | -2.876 | -1.510 |

Separately instrumented jobs, with stage time summed over both final-EPF/color
chains in each encode:

| Input / window | Baseline stage | Stage delta | Stage change | Stage duplicates (baseline, candidate) | Instrumented encode delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flower 500 / 0 | 0.122880 | -0.052224 | -42.47% | 0.000000, +0.000512 | +0.074 |
| Flower 500 / 1 | 0.122112 | -0.051456 | -42.26% | 0.000000, -0.001024 | -0.081 |
| HD / 0 | 0.851456 | -0.399872 | -46.97% | -0.001024, +0.011776 | -2.142 |
| HD / 1 | 0.848640 | -0.402432 | -47.24% | +0.002048, +0.003584 | -0.238 |
| 4K / 0 | 5.580032 | -1.605632 | -28.69% | +0.007168, +0.120320 | -8.131 |
| 4K / 1 | 5.585664 | -1.605632 | -29.25% | +0.026112, -0.035328 | -12.947 |

All six primary stage medians improve. The four individual candidate-versus-
baseline stage comparisons per job are also all favorable: 24 of 24, with
4K savings ranging from 1.308 to 1.888 ms across those correlated medians.
This is strong evidence of a local reduction, not proof that every encode
or batch is faster. The larger instrumented whole-call savings cannot be
substituted for the mixed uninstrumented results. Duplicate whole-call
variation reaches roughly 12 ms, much larger than the target stage saving.

All 1,680 power-limit endpoints in these preflight/measured processes are
40 W, with no endpoint changes. There is no per-kernel clock measurement or
clock-normalized speedup claim. Timing occurs at 14:51:21–14:52:18 UTC
(uninstrumented) and 14:52:18–14:53:17 UTC (events), after the native queue and
all other qualification jobs have finished. No new Nsight Compute counters
are collected here; S125's traffic measurements use different inputs/layouts
and are not silently transferred as an exact S126 DRAM percentage.

## Disposition and reproduction

Carry the fusion into clean production-source/build qualification. Preserve
the unchanged zero-EPF and maximum-error data requirements, add a permanent
regression test, and verify the final compiled kernels against the qualified
prototype. Repeat full CTest and production-linked exact-output checks before
retaining the implementation. Single/batch throughput still needs honest
qualification; the existing byte-exact batch results do not constitute a
batch timing improvement. The backend has not been established to be maxed
out, and this study introduces no compatibility layer or new runtime policy.

Evidence root: `U:/gjxl-cuda-diagnostics/s126/`; diagnostic sources and tools:
`build-cuda-ninja/profiles/s126_*`. The primary reports are `generation.json`,
`within_inputs.json`, `production_inputs.json`, `native.json`, `captures.json`,
`production_qualify.json`, `production_asan.json`, `production_memcheck.json`,
`within_analysis.json`, `events_analysis.json`, `within_power_analysis.json`
and `summary.json`. The source snapshots include the qualified S125 GPU
source, derived resident owner and transitive diagnostic dependencies.

There are 2,076 frozen-oracle encode checks (252 under host ASAN), plus the
256 matrix AQ result pairs described above, 69,120 guarded layout pipeline
executions and 162 guarded hot replay executions. Nine CUDA sanitizer jobs
cover four encoder memchecks, four layout tools and one hot replay memcheck.
The existing AQ test is additional coverage, not folded into an invented
exact-encode count. All 40 previously retained runtime files remain unchanged.

The full native dump queue takes about fifteen minutes of CPU-side work. A
small follow-up audit extracts the ten embedded GPU ELF modules from four
different full-encoder/test executables in about 31–43 ms each and proves
their module-hash multisets identical. This offers a faster way to establish
identity across executables sharing an already audited GPU module set in
future studies. It does not replace this study's completed SASS comparisons.
The initial wrapper and child both chose `elf_probe.json` as their report
name: the child completed successfully, then the wrapper refused to overwrite
its report. The child report/log and separate successful `audit_elf` record
are preserved. No kernel or encode failed in that auxiliary audit.

The unused initially generated power parser retained an old field name;
`s126_power_analyze.py` corrects it before analysis. Input manifests keep both
sources. The validator checks 185 terminal job records: 169 successes,
fourteen expected encode rejections and two rejected v1 matrix harnesses.
It independently checks raw timing rows and paired medians, native bodies,
captured headers/payload sizes, hashes, matrix coverage, sanitizer markers,
timing isolation and retained-runtime identities. `s126_freeze.py` validates
and freezes evidence/source hashes; `s126_validate.py --frozen` rechecks them.
No admin, firewall or permission blocker was encountered.
