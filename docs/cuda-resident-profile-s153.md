# Fully resident production profile after four-row integration (S153)

## Scope and method

Starting runtime is S152, commit `2118ccb`. This is a fresh bottleneck study,
not a candidate-versus-baseline speedup test. No production source, public
API, compact default, allocation policy, or system setting changes.

Four fresh host profiling executables link the frozen S152 Release CUDA
libraries: wide/compact, each ordinary or host-ASAN. Every executable has
the exact same ten GPU-module hashes as the S152 production CLI. There are
no diagnostic GPU bodies, event brackets, or dispatch selectors. The ASAN
executables reuse S152's instrumented host support objects. Host profiling
uses the existing complete in-memory encoding entry and its phase records;
optional NVTX ranges enclose whole calls only.

Six frozen inputs/oracles are reused: flower500, padded HD, flower2000,
padded 4K, flower3200x2160, and keong3839x2159. The larger flower/Keong
fixtures are exact nearest-replicated photographic inputs, not independent
native-resolution photographs. Settings are distance 1.2, effort 7,
automatic CPU threads, fully resident CUDA, and final score disabled.
Input file reading, output verification, telemetry, and harness logging
remain outside the measured encode interval. Each output is byte-identical
to its frozen oracle and summaries/storage widths are stable per process.

The ordered campaign runs 24 release/host-ASAN preflights (three encodes
each), two CUDA memcheck/initcheck processes (three each), 24 ordinary
timing processes (four warmups plus twelve measured encodes), and 24 system
traces (four warmups plus five measured encodes). The shuffled case/mode
process order is reversed in the second repetition. Wide and compact are
separate executables/processes; this is not a balanced within-process
comparison and does not establish a causal compact-versus-wide speedup.

Nsight Systems 2023.2.3 records CUDA and NVTX with CPU sampling/context-switch
collection disabled, full-process capture, graph-node detail, and SQLite
export. Trace and ordinary timings stay separate. The analysis retains full
kernel template names, per-launch geometry/resources, transfer counts/bytes,
CUDA API durations, interval unions, and device-idle gaps. CPU worker-work
counters are nested/possibly overlapping and must not be summed as serial
wall time. API waits overlap GPU execution and are not additive costs.

The environment is Windows, CUDA 11.8, MSVC 14.37, and an RTX 3060 Laptop
GPU (sm86). Enforced power limits are observed read-only around every encode;
endpoint agreement does not establish a fixed clock throughout a kernel.
No clock, power, thermal, priority, affinity, firewall, or security setting
is changed. Historical frozen scripts, binaries, and reports are preserved.

## Qualification

All 77 recorded jobs are terminal and accepted, including the 74 serial
GPU-facing jobs. All 678 oracle encodes pass, as do both CUDA sanitizer
jobs; memcheck also reports zero leaked bytes. The analyzer checks every
trace's nine whole-encode ranges, all successful CUDA API returns, stable
measured kernel geometry/counts and copy payloads, and exact host sample
counts. There are 120 measured trace encodes. All 1,356 enforced-limit
endpoints report 40,000 mW. No observation is discarded or retried, and no
admin/firewall blocker appears.

## Fresh whole-encode and GPU ranking

The table gives ranges of the two ordinary process medians, in milliseconds.
These columns are separate distributions, not an additive decomposition of
one representative encode. AC tokenization is nested inside codestream
encoding; it is not an additional independent cost.

| Case / storage | Whole encode | Input preparation | Quantization pipeline | Codestream encoding | AC tokenization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padded 4K / wide | 268.87-273.65 | 17.14-17.18 | 197.68-199.68 | 45.37-48.18 | 9.57-9.94 |
| Padded 4K / compact | 251.84-269.24 | 17.24-17.40 | 188.12-192.46 | 46.03-59.58 | 9.95-12.36 |
| Keong 4K / wide | 295.09-295.20 | 17.13-17.30 | 197.31-202.44 | 69.76-73.79 | 17.64-17.82 |
| Keong 4K / compact | 284.37-300.85 | 17.17-17.35 | 188.68-192.15 | 75.66-79.97 | 18.11-19.06 |

Across the four process medians per case (two storage modes, two repeats),
the system traces give the following kernel-family sums. Families are
summed within each encode before taking medians; medians of individual
specializations are not added together.

| GPU work (ms) | Padded 4K | Keong 4K |
| --- | ---: | ---: |
| All kernels, interval union | 136.31-143.78 | 131.47-138.44 |
| Paired Malta responses | 20.08-21.72 | 17.97-18.21 |
| Seven fused AC evaluators | 16.41-19.15 | 16.10-18.69 |
| Erosion/L2 finalization | 8.88-9.38 | 8.05-8.33 |
| Mirrored opsin convolution | 8.49-9.14 | 7.45-8.43 |
| All low/medium vertical bodies | 8.26-8.50 | 7.78-8.11 |
| Gaborish/EPF fusion | 5.86-6.88 | 4.57-7.17 |

The changed convolution stage is no longer the natural sole focus. Malta
and fused AC remain larger GPU families, but neither dominates the whole
encoder. Their historical layout, preload, occupancy, and clock-context
experiments remain relevant; these ranks do not justify reinstating rejected
variants or claiming a new causal comparison with older captures.

Measured device-union gaps between the first and last device activity have
process medians of 21.83-27.31 ms for padded 4K and 30.53-34.51 ms for
Keong. The post-device tail is 58.87-67.55 ms and 92.26-114.53 ms,
respectively. The latter includes more than just serialization; it is not
all attributed to AC tokenization. These gaps are not automatically removable
launch overhead, and API synchronization durations overlap device work.

Variation remains material. One reverse-order compact Keong trace encode
takes 411.59 ms, including 203.13 ms after the final device-to-host copy.
It is retained in the distribution. The short traced outer timings are not
substituted for ordinary timings, and no cause is assigned to that outlier.

## Exact coefficient-handoff attribution

`handoff_s153.py` independently rebuilds `BuildAcReadbackLayout` from each
input geometry and observed coefficient width. It identifies the unique
ordered device-to-host copy sequence comprising those AC runs followed by
raw quant, DC, and the two CfL maps. The sequence must end at the final
device-to-host activity in every measured encode. It also identifies the
three original-RGB uploads. This avoids mistaking unrelated cost/metadata
copies for the final coefficient handoff.

| Case / storage | Active AC payload (bytes) | AC copy time (ms) | All D2H payload (bytes) | All D2H time (ms) |
| --- | ---: | ---: | ---: | ---: |
| Padded 4K / wide | 99,532,800 | 16.31-16.58 | 103,723,588 | 16.98-17.26 |
| Padded 4K / compact int8 | 24,883,200 | 4.33-4.62 | 29,073,992 | 5.01-5.30 |
| Keong 4K / wide | 99,532,800 | 18.25-21.56 | 103,723,588 | 18.92-22.23 |
| Keong 4K / compact int16 | 49,766,400 | 8.56-8.59 | 53,957,192 | 9.23-9.26 |

Times are ranges of process medians from separate instrumented processes,
not paired speedup estimates. Each of these AC handoffs consists of two
copies, followed by 2,077,680 bytes of frame metadata. Padding in the
authoritative host owner makes its capacity larger than the active transfer:
106,168,320 bytes wide, 26,542,080 bytes for padded-4K int8, and 53,084,160
bytes for Keong int16.

The original RGB payload is 99,460,812 bytes in both 4K cases and takes
15.23-16.36 ms across these process medians. All H2D copies, including later
descriptors/matrices, total 112,566,740 bytes for padded 4K and 113,506,780
for Keong, taking 17.34-18.52 ms. These measurements do not establish a
PCIe bandwidth ceiling or prove that pinned staging would improve the full
operation after its own allocation/copy/lifetime costs.

## Selected next investigation

Investigate a native sparse coefficient handoff and direct consumer before
further sub-millisecond low/medium tuning. Source inspection establishes
the current boundary: `CudaPreparedResidentAqEvaluation::AssembleTypedFrame`
downloads dense typed AC runs, clears unused group tails, and hands an
authoritative dense owner to the serializer. `TokenizeSimpleAcGroupDirectValidated`
then counts nonzeros excluding LLF and scans each transform's coefficient
order until the final nonzero. Compact int8/int16 ownership removes widening,
but not that dense materialization/consumption boundary.

This is a representation experiment, not permission to omit required work.
The [S80 sparse study](cuda-optimization-s1.md#lossless-sparse-ac-transfer-investigation-s80)
already showed that fresh dense zero/scatter reconstruction and subsequent
dense consumption can erase transfer gains; its rejected general zero-backed
allocator is not revived. Its historical sparsity numbers are not treated
as measurements of these current six inputs.

First capture/census current final coefficients, including compact int8,
int16, and high-range int32 fallback cases. Measure sparse headers, values,
nonzero counts, ordered scan lengths, and actual token counts. Test direct
ordered consumption against the existing dense tokenizer, including natural
and custom orders, LLF exclusions, predictor-map updates, all supported
strategies, and group/tail boundaries. A candidate must preserve exact
tokens/contexts and complete codestream bytes while accounting for fresh
allocation, packing, count synchronization, transfer, consumption, and
destruction. Dense/high-nonzero cases need an explicit measured policy;
reused replay buffers alone are not sufficient evidence.

Moving token preparation onto the GPU is another possible way to avoid the
dense host boundary, but is not implemented or qualified here. The existing
nonzero predictor reads prior top/left block counts; a parallel design must
first establish those counts and preserve token ordering/context semantics.
Neither byte savings nor the sum of independently measured stage medians
is a promised whole-encoder saving. Keep S152 production and compact's opt-in
default unchanged until a new candidate passes complete-workflow gates.
The backend is not established to be maxed out.

## Evidence

Local evidence: `build-cuda-ninja/profiles/s153-artifacts/`. `before.json`
pins the retained runtime and six inputs; `native.json` verifies all four
executable module sets; `protocol.json` pins source/tool/input identities and
all 74 ordered GPU-facing jobs. Original logs, complete `.nsys-rep`/SQLite
traces, extracted stdout, `campaign.json`, `analysis.json`, `report.json`,
and `handoff.json` retain the
observations. The final source snapshot and SHA-256 inventory are created
after qualification, analysis, and this report are complete.

```powershell
python -X utf8 build-cuda-ninja/profiles/verify_s153.py --frozen
```

`freeze_s153.py` is an exclusive one-time finalizer, not a command to rerun
over an existing frozen evidence directory.
