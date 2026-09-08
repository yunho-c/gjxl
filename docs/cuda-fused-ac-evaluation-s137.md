# Fused AC-candidate evaluation prototype (S137)

Disposition: qualified prototype; production integration is next. A block now
keeps each candidate's three channel transforms together, consuming forward
coefficients from shared memory during residual quantization, inverse transform
and rate/loss reduction. This removes seven kernel launches and all use of the
global forward-coefficient scratch, while retaining its allocation for this
isolated experiment. Production source and the retained runtime are unchanged.

Complete-encode primary comparisons favor the scalarized prototype in 15/16
runs; quantization favors it in 16/16. At 4K, whole-call paired deltas are
−16.3 to −25.3 ms across two repeats and both coefficient widths. The short
compact-4K trace nevertheless has slower fused GPU work, so these results are
not a universal kernel-speed claim or proof that GPU-time variation is solved.
They justify testing a production integration, not declaring the encoder maxed
out. No S135 packed consumer or S136 generated-descriptor path is combined here.

## Dataflow and scheduling

The original batch sequence is quant-norm preparation, forward DCT,
residual/rate plus inverse DCT/loss, then final cost. The prototype replaces
the middle two launches with one fused kernel for each of the seven supported
shapes: 8×8, 16×8, 8×16, 16×16, 32×16, 16×32 and 32×32. Quant-norm preparation,
final cost, candidate enumeration, CPU hierarchy/tie-breaking, host cost
scattering and coefficient-width policy stay unchanged.

Each block has 192 threads. A transform uses `max(width, height)` lanes; a
block holds `64 / max(width, height)` complete three-channel candidates. Each
channel has a padded `height × (width + 1)` shared tile. The forward transform
also copies Y coefficients into immutable per-candidate shared storage before
any channel replaces its own tile with residuals. A block-wide barrier protects
that cross-channel dependency. Inactive tail candidates participate in every
barrier without loading their descriptors.

The factorized transform order, coefficient normalization/layout, quantization,
nonzero counts, magnitude halving tree, inverse transform and mask-weighted
eighth-power loss tree match the original path. Tall transforms redistribute
pixels before the final reduction to preserve the original row-major lane
mapping. No fast-math relaxation or changed numerical formula is introduced.

The initial fused version (V1) passed exact-output preflights but materialized
reduction arrays in local memory for four shapes. Ptxas reported zero spill
loads/stores, yet SASS contained `STL`/`LDL` instructions and nonzero stack
storage. V2 replaces only the nested reduction-stage loops with compile-time
recursive halving, keeping the same FP32 operation order. All seven V2 kernels
have zero stack, local storage and static local-memory instructions.

| Shape | V2 registers | Shared bytes | V1 stack bytes | V1 static stores / loads |
| --- | ---: | ---: | ---: | ---: |
| 8×8 | 40 | 8,960 | 0 | 0 / 0 |
| 16×8 | 39 | 8,576 | 0 | 0 / 0 |
| 8×16 | 56 | 8,960 | 0 | 0 / 0 |
| 16×16 | 48 | 17,152 | 64 | 23 / 4 |
| 32×16 | 56 | 16,768 | 64 | 23 / 4 |
| 16×32 | 88 | 17,152 | 64 | 23 / 4 |
| 32×32 | 78 | 33,536 | 192 | 63 / 10 |

The 32×32 register count rises from 75 to 78; the other counts are unchanged.
These are native static resources, not measured occupancy or DRAM counters.
V1 was not put through the timed campaign; timings below use V2 only.

## Scratch opportunity, not an allocated-memory saving yet

For a strategy with N candidates and W×H coefficients, the old B range is
`N × 3 × W × H × sizeof(float)`. The prepared owner reuses the maximum range
across strategies, rather than allocating their sum. The prototype leaves
that range allocated and validated but proves it is untouched by fused work.

| Input | Maximum B bytes still allocated | Sum of forward coefficient bytes across strategies |
| --- | ---: | ---: |
| Flower 500 | 9,292,800 | 38,247,168 |
| HD | 76,124,160 | 317,260,800 |
| 4K | 304,496,640 | 1,273,835,520 |
| Flower 2000 | 145,993,728 | 610,544,640 |

The 4K maximum is DCT16×16: 99,120 candidates, or 290.39 MiB. These are
source-derived logical ranges. Eliminating forward stores and subsequent own/Y
coefficient reads does not imply that their sum was physical DRAM traffic;
caches and repeated Y reads matter. No VRAM, RSS, allocation-reserve or measured
bandwidth reduction is claimed for this still-allocated prototype.

## Exactness and native isolation

Starting revision `aed78a1`, branch `feat/cuda`, September 8, 2026. The system
is Windows with an RTX 3060 Laptop (`sm_86`), CUDA 11.8 and MSVC 14.37. Diagnostic
GPU objects contain all original kernels plus the seven candidates. A selector
exists only in diagnostic batch code. Host ABI/layout is unchanged; the tests
link frozen S134 libraries, with a freshly compiled resident-owner object for
compact coefficients.

The focused fixture covers fifteen block geometries, including thin images,
partial tiles and axes through 257 blocks: 72 nonempty geometry/strategy cases
and 14,938 descriptors per execution. Independently nested host enumeration
produces descriptors. The original two-kernel path supplies exact channel-loss,
channel-rate and final-cost bytes. Two fused evaluations reset a guarded arena
and must match all three outputs, preserving every input, descriptor, guard and
the entire poisoned B range. RGB, mask, quant and signed device-CfL views have
nontrivial strides.

V1 passes one focused execution. V2 passes standalone, harness-ASAN, CUDA
memcheck, initcheck, synccheck and racecheck: six focused executions. All four
CUDA sanitizer jobs are clean; memcheck also reports zero leaked bytes.
An exhaustive extension tests every block width/height pair from 1 through 19,
covering 2,201 nonempty cases and 134,809 descriptors in each of Release and
harness-ASAN. Together these focused/exhaustive fixtures run 4,906 baseline
and 9,812 fused batches; repetitions are not independent images.

The unchanged broad CUDA AC-strategy contract test additionally passes in
Release and harness-ASAN with fused dispatch mandatory. It covers host/device
norm and CfL inputs, invalid descriptor values and masks, candidate tails,
guarded compact output ranges, overlaps, no-work and invalid-request atomicity.
It still uses the old B requirement; a future zero-B contract requires updated
tests, not merely rerunning this executable.

Whole-encode checks total 1,436 against the frozen S133 oracles: 100 preflight
checks (twenty V1), 1,296 timed-process checks including warmups/references,
and forty trace checks. Forty are harness-ASAN whole encodes. Every encode
checks output bytes, summary, coefficient width/storage and dispatch counters.
ASAN instruments the diagnostic host harnesses only, not the frozen libraries
or compact resident object. This prototype does not claim a fresh full CTest,
broader quality/rate/batch matrix, independent decoder, Metal or Linux run.

Native analysis compares instruction and control bytes after normalizing only
anonymous-namespace names. All 214 original S134 kernel bodies are identical
in V1 and V2; each has exactly seven additional fused bodies, for 221 total.
Fourteen linked GPU executables are audited by extracted module hashes: two
V1 and twelve V2. Each has ten modules; all executables of the same variant
match exactly. Eight modules match S134 byte-for-byte, and nine match between
V1 and V2. The modified batch module's two original bodies also match S134,
despite different enclosing module bytes.

## Complete-encode measurements

Inputs follow S133: the 500-square Flower crop, HD from 1919×1079, 4K from
3839×2159, and Flower 2000, which is fourfold nearest-neighbor replication of
the crop, not a native 2000-square photograph. Distance is 1.2, effort 7,
fully resident, with wide and opt-in compact coefficients.

Labels 0/2 are duplicate baselines and 1/3 duplicate V2 candidates in the same
process. Each four-round Williams block balances positions and ordered
predecessors. A process runs one reference encode, four warm rounds and sixteen
measured rounds; a second process pass reverses case/width order. Sixteen
timed processes contain 1,024 measured encodes. No other recorded task job
overlaps those timed processes.

The primary statistic is the median across rounds of mean candidate-label
time minus mean baseline-label time, not a difference of independent medians.
Whole-call time includes input preparation, quantization and serialization.
Negative is faster.

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.190 / −0.152 | −0.236 / −0.148 |
| Flower 500 / compact | −0.071 / +0.015 | −0.119 / −0.118 |
| HD / wide | −2.377 / −2.252 | −2.251 / −0.838 |
| HD / compact | −1.066 / −1.963 | −1.503 / −1.608 |
| 4K / wide | −16.337 / −19.605 | −14.369 / −15.940 |
| 4K / compact | −24.387 / −25.293 | −19.013 / −25.560 |
| Flower 2000 / wide | −2.729 / −6.721 | −2.732 / −6.538 |
| Flower 2000 / compact | −4.609 / −8.541 | −4.364 / −6.789 |

Whole-call primary comparisons favor V2 in 15/16 runs and 60/64 cross-label
pairs; quantization is favorable in 16/16 and 63/64. The smallest compact
case's second whole-call delta is +0.015 ms. The four 4K whole-call percentage
deltas are approximately −5.43%, −5.98%, −8.46% and −8.29%, but duplicate
controls remain appreciable: equivalent baseline labels differ by +10.436 ms
in the second wide-4K process. These observations do not establish a universal
percentage speedup or justify a post-hoc image-size threshold.

## Trace crosscheck and limits

Eight complete-process Nsight Systems captures contain 32 labeled windows
plus eight reference encodes. CPU sampling/context-switch collection is off.
All CUDA calls succeed. Candidate windows replace fourteen AC forward/residual
kernels with seven fused kernels, preserving seven quant-norm and seven final
cost kernels. All other kernel-name/count multisets match. H2D, D2H and other
copy counts and payload bytes are unchanged; unused global scratch traffic
is kernel work, not a CUDA memcpy.

Launch counts decrease from 301/302 to 294/295 at 4K (wide/compact), 327/328
to 320/321 at HD, and 366/367 to 359/360 for both Flower sizes. Traced summed
AC work decreases from about 0.65 to 0.48 ms for Flower 500, 4.7 to 3.5 ms
for HD and 8.5–9.0 to 6.2 ms for Flower 2000. At 4K the wide trace changes
from 23.45 to 22.30 ms, but compact changes from 21.32 to 23.38 ms. These
short instrumented windows do not reproduce the magnitude of the unprofiled
whole-call saving and do not explain its complete cause.

All 2,872 captured power endpoints report a 40 W enforced limit. This is not
proof of constant clocks, stable cache/placement or absent external activity.
No power, clock, priority, affinity or firewall policy was changed. No admin
or firewall blocker was encountered. Existing S119–S121 rate-variation evidence
still applies; RDP or a particular hardware/driver mechanism is not established
as the cause of this campaign's remaining variation.

## Next integration and evidence

Next make the fused evaluator the normal CUDA batch path, sharing arithmetic
with real transform consumers rather than retaining an experiment selector or
compatibility layer. Remove B from the CUDA internal launch/validated state,
report a zero CUDA B requirement and teach the common owner to skip zero-byte
arena ranges while preserving Metal's real B use. Verify the changed layout,
invalid-input contracts and actual prepared-memory reduction, then perform a
fresh build/full CTest, broader production/oracle/ASAN qualification and
whole-encode controls. The prototype's still-allocated timings must not be
silently relabeled as measurements of that integration.

Evidence is frozen under `U:/gjxl-cuda-diagnostics/s137/`: `before.json`,
`inputs.json`, `trace_inputs.json`, `within_analysis.json`, `timing_summary.json`,
`scratch_plan.json`, `native_scan.json`, `linked.json`, `trace_analysis.json`,
job logs, `source_snapshot_index.json`, `final_summary.json` and
`artifact_hashes.json`. Archived `build-cuda-ninja/profiles/s137_*` files include
both prototypes, generators, differential fixtures and analysis drivers.

There are 68 terminal job records: 67 successful and one failed initial build.
That first build compiled V1 GPU code but failed in diagnostic batch code
because `cost_blocks` was scoped inside the selector branch. The original
source/log are retained; `s137_batch_v2.cu` fixes that host-launch scope issue.
It is not a numerical failure, elevation prompt or live stalled process.
Forty retained runtime files keep their recorded hashes. This checkpoint
commits documentation only; no experiment code is enabled in production.
