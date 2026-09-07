# CUDA candidate disposition and systematic survey (S104–S105)

2026-09-07, starting from S103 `da32d87`. RTX 3060 Laptop, 6 GiB, SM 8.6;
CUDA 11.8, driver 577.00, Windows. Production remains S79 `914b42c`.

**Reject S84 and S95 as default production changes; close this qualification
campaign.** S84's current dense-owner plus compact-staging design has useful
large-image latency gains but no established general dispatch/concurrency
policy. Mask fusion has no consistent whole-call benefit, alone or added to
S84. This is a promotion decision, not proof that either mechanism is useless.
Retire these implementations instead of continuing open-ended threshold,
first-touch, mask-fusion or graph-retirement sweeps. Compact coefficient
consumption and broader traffic reduction should be new candidates.

The counter survey covers the kernel specializations observed in three
retained traces. It distinguishes DRAM limits in final masking/low-medium
construction from SM load/store pressure in Malta. Direct CPU timers identify
strategy merging as a larger host gap component than metadata construction.
No new encoder speedup or exhausted optimization potential is claimed.

## S104: complete-encode and batch comparison

One executable links the existing S84 resident host implementation, S81 narrow
kernels and S95 Butteraugli kernels with retained libraries. Four families
are tested: retained control, S84, S95, and both. Each has two labels using the
previously native-identical duplicates. Eight warm and sixteen measured
eight-label Williams rounds balance position and every ordered predecessor
pair. Two replications reverse configuration order.

The outer timer covers the complete encode, including serialization and
per-image owner cleanup, or the complete public batch Encode call. Backend /
batch-driver creation, input loading, output comparisons and destruction of
returned codestreams are outside timing. Single calls use automatic CPU
participation; concurrent calls use two CPU participants per image. Batches
use the public bounded CUDA lane pool. Comparisons within a batch size have
equal CPU policy; cross-size ratios are not controlled throughput scaling.
Independent backends are separately qualified.

All 24 completed timing jobs pass: 9,624 encodes, 9,600 exact byte/summary
comparisons, and 24 frozen reference hashes. The 3,072 measured outer call
intervals represent 6,400 image encodes; warm-up calls represent 3,200 more.
A separate four-image 4K job fails before its first timing row and is excluded
from these counts, with its failure preserved.

Below are ranges of eight within-round paired-median percentage changes:
both candidate labels against both controls in both replications. Negative
is faster. The fixed descriptive gate requires all eight values negative;
it is not a significance test or confidence interval. Duplicate controls and
all slower observations remain in the raw results.

| Input | Batch | Control outer ms, rep 0 / 1 | S84 % | S95 % | Both % | Adding S95 to S84 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny 17×13 | 1 | 8.853 / 8.050 | -3.24 to +8.77 | -1.37 to +0.18 | -5.71 to +1.71 | -10.26 to +5.08 |
| Flower 500×500 | 1 | 23.783 / 30.294 | -5.79 to +2.14 | -2.08 to +4.95 | -5.38 to +7.80 | -4.14 to +4.70 |
| Flower 500×500 | 2 | 34.292 / 43.481 | -8.97 to +7.11 | -4.79 to +7.63 | -6.95 to +2.16 | -9.65 to +11.90 |
| Flower 500×500 | 4 | 52.455 / 62.942 | -12.20 to +7.51 | -9.65 to +3.95 | -8.45 to +4.34 | -6.67 to +7.98 |
| Keong 500×500 | 1 | 28.062 / 27.760 | -4.15 to +4.50 | -2.86 to +5.70 | -6.82 to +4.07 | -4.83 to +7.64 |
| Keong 500×500 | 2 | 37.790 / 41.068 | -7.62 to +6.82 | -5.82 to +7.35 | -5.03 to +10.50 | -4.32 to +14.52 |
| Keong 500×500 | 4 | 57.251 / 59.059 | -11.57 to +2.95 | -4.41 to +4.05 | -8.65 to +2.30 | -7.14 to +1.13 |
| 1919×1079 | 1 | 88.219 / 102.085 | **-9.10 to -3.44** | -2.77 to +2.47 | **-10.12 to -5.26** | -4.52 to +3.41 |
| 1919×1079 | 2 | 158.517 / 177.309 | **-7.73 to -1.18** | -3.05 to +3.82 | **-7.63 to -2.45** | -5.99 to +3.31 |
| 1919×1079 | 4 | 323.797 / 376.903 | **-5.31 to -2.60** | -1.87 to +5.37 | **-7.43 to -3.26** | -4.58 to +0.71 |
| 3839×2159 | 1 | 385.450 / 377.002 | **-9.35 to -1.61** | -2.41 to +2.73 | -5.38 to +0.69 | -0.86 to +5.24 |
| 3839×2159 | 2 | 850.525 / 854.725 | -5.79 to +2.06 | -0.24 to +2.38 | -4.42 to +1.19 | -2.18 to +6.52 |

S84 passes four of twelve configurations; eight are mixed. Both together pass
three; nine are mixed. S95 alone and S95 added to S84 are mixed in all twelve.
No family is consistently slower across all duplicate comparisons in both
replications. Mixed evidence is not proof of harm, but fails the predeclared
unconditional-promotion gate. S85 also failed to establish a general geometry
threshold. Do not turn the four favorable cases into an untested size rule.

### Memory, capacity, and correctness

Thirty-two separate same-mode process jobs isolate memory from mixed-label
process peaks. At 4K, peak process working set is 429.5–429.6 MiB for control
and 476.9 MiB for S84; batch two is 614.6–615.1 versus 708.8–709.2 MiB.
The roughly 47.4 MiB per-image increase agrees with the extra 49,766,400-byte
compact allocation. S95 alone has similar working-set peaks to control.
These are Windows working-set peaks, not peak host commit or a full physical
allocation census. S84's final dense owner still occupies 106,168,320 bytes.

Sampled NVML device-memory peaks are 2496.7 MiB for all single-image 4K
families. Batch-two peaks vary about 4864.7–4992.7 MiB across modes/repetitions.
These include pool retention and are sampled, not exact live-allocation maxima.
Source and prior audits establish that S84 adds no new device allocation;
changed scheduling can still affect retention and simultaneous liveness.

Four-image 4K is not reliable on this device. Across eight fresh-process
same-mode checks, control succeeds once/fails once; S84 fails twice; S95
succeeds once/fails once; both succeed twice. Failures report
`cudaErrorMemoryAllocation`. Together with the initial mixed-job failure,
these show a shared capacity problem with schedule-dependent outcomes. They
do not prove that S84 alone causes OOM or that the combination solves it.
There is no four-image 4K performance or general capacity claim.

All sixteen release/scoped-ASan preflights pass 240 encodes: eight labels on
tiny, Flower, 1080p and 4K single calls, 1080p public batches 2/4, 1080p with
two independent backends, and 4K batch two. Every active coefficient also
matches the original dense device representation. Both new executables match
the exact union of frozen S95/S81 native code: 212 kernels. S84 and S95 frozen
evidence validators pass, retaining their boundary, arithmetic, failure-atomic
and prepared-owner qualification.

New integrated Compute Sanitizer checks on both candidates together pass:
1080p batch-two memcheck (zero errors/leaked bytes) and Flower batch-two
initcheck (zero errors). Each checks six candidate encodes plus a reference.
The Windows wrapper loses the harness's final unflushed stdout marker;
qualification verifies all three flushed rows, emitted only after exact bytes,
summary and dense-coefficient checks, plus reference hash, zero process exit
and sanitizer summary. This logging limitation and an initial launcher-path
mistake are preserved, not counted as extra passes.

ASan covers the new harness and S84 host code, not all retained libraries.
No fresh independent decode is claimed: exact outputs match already decoded
frozen references. No production implementation is promoted, so unchanged
CPU/CUDA suites are not repeatedly run in place of the new checks above.

## S105: counters and resource map

### Counter access and measurement scope

Installed Nsight Compute 2022.3 fails on both 1080p application replay and tiny
kernel replay with `driver resource was unavailable`, rather than the earlier
permission error. NVIDIA's portable **2025.2.1.3 succeeds** with the same driver
and permissions. Its archive matches the official CUDA 12.9.1 manifest SHA-256
`e9d558654c98d83049969d133b98922b53ab8f4e3ba9e0a37bdb5e2ff300b7de`.
This resolves collection without changing a driver, power, clock, security or
system-profile setting. Old-tool compatibility is implicated; its exact
failure mechanism is not established.

The replay-safe harness links retained libraries and checks every encode
against frozen bytes and a fresh summary. Each application does two conditioning
encodes and one captured encode. Application replay, cache control `none`, clock
control `none`, and first invocation per fully demangled kernel collect
SpeedOfLight, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, SchedulerStats,
WarpStateStats and Occupancy, plus DRAM bytes, L2 bytes and shared-bank conflicts.

There are 249 main first-invocation profiles: **71 / 83 / 95** at 4K / 1080p /
Flower. Their normalized symbols exactly match every specialization in the
corresponding S103 traces, covering 100% of those traces' kernel-family time.
This is symbol coverage, not every invocation or full/half-scale cache state.
A three-kernel pilot and twelve repeated leading-kernel profiles bring the
qualified total to 264. Six captures each complete sixteen application passes:
288 frozen-byte-checked encodes. Two failed CSV export commands are corrected
by read-only re-export; successful GPU captures are not repeated for that error.

Application replay retains preceding work better than cache-flushed kernel
replay, but still serializes launches and changes gaps/device state. Replay
times are not encode latency. One primary Malta occupancy is 102.9%; it is
flagged and excluded from occupancy conclusions, not clipped. Its repeat is
95.1%. Interpret throughput, stalls and occupancy together; see NVIDIA's
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

The complete main data is in [the 249-row CSV](cuda-counter-survey-s105.csv).
The GPU milliseconds below are S103's six-trace means per encode, not sums of
replay samples or first-invocation times multiplied by launch counts. Percentages
describe sampled 4K invocations, with repeat ranges where available. LSU means
the SM load/store instruction pipeline: high SM utilization can be memory work.

| Stage | S103 4K GPU ms/encode | Counter evidence | Inferred constraint / next experiment |
| --- | ---: | --- | --- |
| Leading paired Malta variants | 23.55 | LSU 82–88% active; FMA 34–39%; DRAM 26–61% | On-chip load/store demand. Reduce repeated shared reads through warp/register reuse; budget occupancy and synchronization. |
| Low/medium convolution and construction | 10.41 | DRAM 87.4%; LSU 37–38%; FMA 11%; long scoreboard about 17 cycles/issue | DRAM traffic/load latency. Avoid intermediate writes through a bounded consumer fusion or rolling strip. |
| Erosion/L2/final masking | 9.85 | DRAM 96.0–96.2%; FMA 12–14%; no shared storage | Strong DRAM limit. Consume final statistics directly; S95 alone is rejected above. |
| Fused mirrored RGB blur/Opsin | 9.36 | LSU about 77%; DRAM 64%; FMA 31% | Mixed on-chip memory issue and external traffic. Fusion must reduce LSU work as well as DRAM stores. |
| Vertical 15-tap convolution | 7.70 | LSU 79%; DRAM 71%; FMA 20% | Mixed memory-system pressure. Reuse across adjacent stages, with explicit halo/shared-traffic accounting. |
| Paired horizontal convolution | 5.92 | DRAM 91.6%; LSU 67% | External traffic is a strong constraint. Preserve horizontal reuse while avoiding materialization. |
| EPF stages 1 + 2 | 8.39 | DRAM 77% / 90%; LSU 71% / 42%; high occupancy | Stage-dependent memory pressure; more occupancy alone offers little demonstrated headroom. |
| Gaborish | 5.08 | DRAM 83%; LSU 46%; FMA 26% | Traffic-oriented optimization coordinated with reconstruction. |
| Resident 32×32 coefficient passes | 6.91 | Reconstruction variant: DRAM 87.7%, LSU 28%, FMA 13% | Coefficient/image traffic. Treat reconstruction and final-only variants separately. |
| AC-search inverse DCTs | 14.09 | Shapes span DRAM 43–72%, LSU 40–80%; 32×32 occupancy about 41% | Shape-specific mixture. Inspect register lifetimes/communication; not a blanket tensor-core opportunity. |
| Compose + maximum + per-transform reductions | 4.97 | Separate kernel profiles and source reduction boundaries | Encoder-specific composition/reduction can avoid a write and rereads; preserve nonlinear accumulation order and invalid-value semantics. |
| Active AC packing | 0.95 | DRAM 94%; SM 12% | Locally bandwidth-bound but small. Change the consumer representation instead of more packing sweeps. |
| Exact quantizer chain | 2.66 | Eighty launches; small-image utilization differs | Small 4K ceiling. Test an exact selector only on a separately qualified small-image path. |

Stage times are ceilings, not wholly removable work. Initial engineering targets
are 10–20% of Malta (2.4–4.7 ms), 15–25% of low/medium (1.6–2.6 ms), and 1–3 ms
from composition/reduction. These are hypotheses requiring full-call validation,
not measured savings, confidence intervals or additive promises. A larger tiled
pipeline spans several rows and must count extra border work, registers,
shared traffic, synchronization and remaining global writes together.

Natural clock-state variation provides a useful cross-check. The same 4K
`<64,false,false>` Malta invocation transfers about 99.6–99.7 MB in both runs,
with memory clock about 5495 MHz. GPC clock falls from 1275 to 553 MHz and time
rises from 0.640 to 1.458 ms; LSU stays about 86–88%. Final masking stays
2.752–2.755 ms and about 96% DRAM despite GPC changing from 1260 to 945 MHz.
This supports different limiting resources. It is not a controlled clock
experiment or normalization.

At 1080p masking remains about 95.5% DRAM and low/medium about 86%. Flower
masking is about 90%, while small CfL/selection launches have low utilization
and occupy more of its trace. Large-image diagnoses do not establish a
universal launch policy. No Metal counter results are inferred from CUDA.

### CPU gaps and ownership

Diagnostic copies of retained host source separately time metadata phases,
upload, AC cost copy/scatter and strategy merge. Timers alternate off/on in one
executable. Five release and five scoped-ASan jobs pass 130 frozen-byte-checked
encodes. Counter, host and host-ASan executables retain the same 205 native
CUDA bodies. Keong 2000 below is a frozen integer-replicated derivative, not a
native high-resolution photograph.

| Direct host scope, median ms | Flower 500 | Keong 2000 | 1080p | 4K |
| --- | ---: | ---: | ---: | ---: |
| Strategy merge | 0.538 | 8.828 | 2.371 | **12.000** |
| CPU cost scatter | 0.052 | 0.984 | 0.389 | 1.831 |
| Metadata construction | 0.325 | 3.104 | 0.747 | **3.129** |
| Of which validated block scan | 0.222 | 2.085 | 0.601 | 2.635 |
| Metadata upload, including synchronization | 0.137 | 0.569 | 0.068 | 0.396 |

Metadata is built once per encode here, alongside seven cost-copy/scatter
batches and one merge. Nested scopes are not additive. Timer-on/off whole-call
medians differ roughly -0.6% to +5.6%; these are attribution observations, not
microsecond-precision overhead measurements or an exact partition of S103's
19.08 ms maximum gap from another cohort.

`FindAcStrategyGridImpl` visits 64×64 color tiles serially. A bounded next
experiment is independent tile selection with deterministic commit and
identical failure handling, using a bounded executor. The 12 ms merge is the
ceiling; 4–8 ms reduction is an unproven target. Optimizing the 3 ms metadata
builder cannot remove the whole gap; its validated block scan is secondary.

The coefficient ownership problem remains larger than packing arithmetic.
S84 reduces this 4K active AC payload from 99,532,800 to 24,883,200 bytes but
expands it back to dense int32 and adds about 47.4 MiB resident host pages per
image. A lossless narrow view consumed by CPU tokenization could remove
expansion/dense writes; sparse consumption is a larger step with scan-order,
context and zero-run invariants. Preserve int32 fallback and exact bytes.
No compact consumer is implemented here, and transfer/CPU savings cannot be
added to overlapping GPU work without a whole-call experiment.

S103's 4K mean copy time is 46.32 ms, and its post-GPU interval is 81.18 ms.
These are attribution boundaries, not wholly removable PCIe time or pure
serialization. New serializer wall time below is about 55–56 ms, including
substantial entropy optimization and section writing. Worker-sum subtimers are
not added to wall time. This survey has GPU counters, not CPU cache/DRAM
counters, and does not label every CPU phase bandwidth-bound.

## Fresh retained absolute baseline

Six fresh-process jobs use the hash-frozen S103 retained harness. Each creates
a persistent backend, checks one reference and two warmups, then measures eight
complete calls. Capture and in-window NVML sampling are off; existing NVTX
annotations remain. All 66 encodes pass: 60 comparisons and six frozen hashes.
Backend lifetime, file I/O and output checks are outside timing; per-encode
cleanup is included. Distance 1.2, effort 7, automatic CPU count, fully resident,
final score off.

| Input | Outer median ms, rep 0 / 1 | All measured min–max ms | Quantization median ms | CPU codestream median ms |
| --- | ---: | ---: | ---: | ---: |
| Flower 500×500 | **21.980 / 23.456** | 20.730–26.314 | 10.522 / 10.980 | 10.016 / 10.796 |
| 1919×1079 | **84.204 / 81.323** | 78.839–102.868 | 48.200 / 46.491 | 26.855 / 25.572 |
| 3839×2159 | **322.349 / 325.415** | 306.154–367.269 | 236.562 / 236.124 | 55.823 / 55.434 |

These are observations of retained production, not a speedup since S79 or a
denominator for S104's different diagnostic harness/cohort. Absolute drift is
preserved. No sustained-load state, vendor power-mode explanation or
machine-wide isolation is claimed.

## Next implementation order

1. Compact coefficient consumption through actual frame assembly/tokenization,
   avoiding S84's second owner and dense expansion; qualify memory and full-call
   single/batch performance together.
2. Independent host strategy-tile selection with deterministic decisions and
   atomic failure; metadata follows the measured 12 ms merge.
3. Encoder-specific composition/reduction, then a bounded perceptual strip
   spanning producers/consumers. Count removed DRAM traffic and extra LSU/halo
   work before a large fusion.
4. Warp/register reuse in Malta to reduce shared-load issue demand. Ampere
   asynchronous copies can overlap loading, but do not remove response shared
   loads; useful overlap needs work before waiting. See NVIDIA's
   [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
5. Capacity-aware batch admission/reuse before expanding concurrency. Four 4K
   encodes are not reliably admitted on this 6 GiB device.

Do not lead with more reciprocal, packing, mask-fusion or graph-retirement
sweeps. Tensor cores/global fast math affect a different numerical contract
and are not justified by these LSU/DRAM observations. The remaining opportunity
is representation, communication and scheduling, with exact-output and
complete-call gates.

## Evidence and reproduction

New scripts/sources are `build-cuda-ninja/profiles/s104_*` and `s105_*`.
Frozen evidence, binaries, raw logs, `.ncu-rep` files, exports, telemetry and
source snapshots are in `U:/gjxl-cuda-diagnostics/s104`. The compact counter
CSV is tracked beside this report. Large inputs/prior diagnostic dependencies
remain local, as in preceding checkpoints.

```powershell
python build-cuda-ninja/profiles/s104_analyze.py
python build-cuda-ninja/profiles/s105_analyze.py
python build-cuda-ninja/profiles/s104_validate.py
```

The recorded profiler command uses portable Nsight Compute 2025.2:

```text
ncu --profile-from-start off --replay-mode application
    --cache-control none --clock-control none
    --kernel-name-base demangled --rename-kernels off
    --kernel-id "::regex:.*:1"
    --section SpeedOfLight --section MemoryWorkloadAnalysis
    --section ComputeWorkloadAnalysis --section SchedulerStats
    --section WarpStateStats --section Occupancy
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
    --export REPORT counter.exe INPUT.pfm FROZEN_REFERENCE.jxl
```

GPU jobs run serially. Light inspection, analysis and some host builds overlap
the interleaved campaign; natural laptop drift and background work remain
limits. Failed exports/launches, capacity failures and slower observations are
retained. No live process is restarted because an observation timed out.
Production source and forty retained runtime files are unchanged; the user's
three untracked Markdown files are untouched.
