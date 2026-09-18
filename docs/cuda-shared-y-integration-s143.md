# CUDA selective shared-Y integration (S143)

## Change

Integrate the S142-selected deferred-store evaluator for 16×16 and 32×32
into the normal fused AC-strategy kernel. Other shapes retain their packed
shared Y copy. There is no runtime selector, fallback, or compatibility layer
in production. Coefficient storage width, APIs, allocation sizes, candidate
ordering, arithmetic and the 192-thread candidate-major launch stay unchanged.

The selected squares read original Y directly from its padded channel tile.
Each lane holds its residual coefficients in compile-time-indexed registers
until every channel has finished reading Y. An unconditional block barrier
then permits residual tile stores; the existing following barrier protects
the inverse transform. Inactive tail groups reach both barriers. Magnitude
and nonzero reductions retain their existing arithmetic and reduction tree,
including Y's zero-factor subtraction.

The normal test adds eight square/tail block geometries: 2×2, 4×4, 4×5, 5×4,
4×6, 6×4, 4×9 and 9×4. These include single-candidate and odd-tail batches.
All 23 geometries exercise every applicable transform and both host and
device signed CfL, with guarded strided buffers and a separate two-kernel
forward/residual-inverse reference. Device-derived quant norms are additionally
covered by the diagnostic comparison fixture.

## Fresh build and native code

A separate Release Ninja build uses CUDA 11.8, MSVC 14.37 and SM86, with CUDA
and tests enabled, Metal and benchmarks disabled, and compact AC disabled by
default. All 82 CTests pass. The expanded normal fused test reports 250 cases,
31,118 descriptors, 250 reference batches and 750 fused batches.

The normal encoder has 221 native bodies. Both selected production bodies
are instruction/control/local-memory-identical to the frozen S141 deferred
prototypes used in S142. The five other fused bodies are exact S138 matches.
Relative to S138, 218 bodies are unchanged; the two squares and a
source-unchanged quant-norm compiler variant account for the other three.

| Shape | Registers | Static shared bytes | Native comparison |
| --- | ---: | ---: | --- |
| 32×32 | 76 | 25,344 | Exact S141 deferred |
| 16×16 | 52 | 13,056 | Exact S141 deferred |
| 16×32 | 88 | 17,152 | Exact S138 |
| 32×16 | 56 | 16,768 | Exact S138 |
| 8×16 | 56 | 8,960 | Exact S138 |
| 16×8 | 39 | 8,576 | Exact S138 |
| 8×8 | 40 | 8,960 | Exact S138 |

The selected squares save 8,192 shared bytes per 32×32 block
(33,536 → 25,344) and 4,096 per 16×16 block (17,152 → 13,056).
These are allocation savings, not a claim of achieved occupancy.

All seven report zero stack/local allocation and zero static local loads or
stores. The normal build has three existing unused-variable warnings in the
unchanged exact-AQ source; edited CUDA source and additional diagnostic/host
builds are warning-free.

## Qualification design

Fresh production-library callers cover wide and compact resident encoding,
including freshly compiled address-sanitized host validators. The isolated
comparison GPU translation unit includes normal production source and clones
only the previous residual source, fused kernel and launcher from the pinned
pre-edit archive. It does not reuse or overwrite a historical GPU object.

Three diagnostic families use duplicate labels: S138 baseline (0/3), new
32×32 only (1/4), and normal production integration for all seven shapes (2/5).
The intermediate family uses baseline kernels for its six other shapes.
Thus the complete-integration family exercises normal entry points even for
the five unchanged shapes. Ordinary and event-instrumented builds remain
separate. The event protocol retains S142's seven paired stream intervals
and endpoint NVML observations; endpoints are not continuous kernel clocks.

Broad production qualification completes: 80 successful configurations
cover quality, effort, stress, concurrency and wide/compact coefficient modes.
Fourteen known invalid configurations preserve their original error and exit
status. Production callers pass 1,296 frozen-oracle encodes across normal,
scoped host-ASAN and CUDA memory/initialization checks.

Both exhaustive geometry sweeps pass normally and under host ASAN. The normal
reference fixture covers 4,402 cases and 269,618 descriptors per exhaustive
run; the comparison fixture covers 8,804 cases and 539,236 descriptors,
including both quant-norm sources. Both focused fixtures pass all four CUDA
sanitizers, including the deferred barrier and inactive tails. Eight further
production encoding memory/initialization checks bring the total to sixteen
zero-error CUDA sanitizer jobs. Race checks report zero hazards; memory checks
report no leaks.

Native/module auditing covers 29 executables in three groups: eleven normal
production, twelve ordinary comparison and six instrumented comparison.
Every executable contains ten GPU modules. All modules match within each
group. Ordinary comparison has all 221 production bodies unchanged plus seven
instruction-identical S138 baseline bodies. Instrumented comparison differs
only in the previously observed, source-unchanged quant-norm compiler variant.
Production/comparison groups share eight byte-identical modules; ordinary and
instrumented comparisons share nine. All additional host/GPU builds are fresh.

## Repeated performance

Both timed campaigns and preflights complete: 3,456 measured encodes across
32 timed processes, plus warmup/reference calls. With ten trace processes,
the performance callers check 4,934 frozen-oracle encodes. Combined with broad
production qualification, 6,230 encodes pass, including 334 scoped
host-ASAN encodes. No recorded build or diagnostic job overlaps a timed process.

The primary statistic is median within-round mean candidate-label time minus
mean baseline-label time. Negative is faster. Quantization is nested inside
whole-call time; per-stage medians and phase deltas are not additive. Duplicate
labels and all cross-label comparisons remain in the evidence, including
unfavorable observations. Enforced power and endpoint telemetry are observed
without changing device policy.

### Ordinary whole-call results

#### 32×32 only

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | +0.168 / +0.024 | +0.030 / +0.009 |
| Flower 500 / compact | −0.341 / +0.060 | −0.077 / +0.076 |
| HD / wide | +1.104 / +0.339 | +0.419 / +0.547 |
| HD / compact | +0.039 / +1.489 | +0.542 / +0.491 |
| 4K / wide | −0.587 / −5.037 | +0.565 / −2.691 |
| 4K / compact | −0.150 / −4.901 | +1.328 / +0.674 |
| Flower 2000 / wide | +0.427 / −4.403 | −1.521 / −3.339 |
| Flower 2000 / compact | −0.368 / +3.768 | +0.575 / +0.277 |

Favorable primary comparisons: 7/16 whole-call and 4/16
quantization; cross-label pairs: 34/64 and 31/64 respectively.

#### Production: 32×32 plus 16×16

| Case / width | Whole-call delta r0 / r1 (ms) | Quantization delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.025 / −0.081 | −0.023 / −0.015 |
| Flower 500 / compact | −0.131 / +0.172 | −0.103 / +0.038 |
| HD / wide | −0.657 / +0.214 | −0.407 / +0.251 |
| HD / compact | +0.737 / +1.017 | +0.286 / −0.015 |
| 4K / wide | +7.577 / −10.666 | +5.904 / +0.614 |
| 4K / compact | −4.152 / −5.956 | −1.161 / −2.747 |
| Flower 2000 / wide | −1.783 / −3.118 | −0.576 / −3.358 |
| Flower 2000 / compact | +0.302 / +1.705 | −0.560 / +0.049 |

Favorable primary comparisons: 9/16 whole-call and 10/16
quantization; cross-label pairs: 36/64 and 36/64 respectively.

Largest absolute duplicate-label whole-call control: +12.711 ms
(4K / compact, r1, labels 3 minus 0).

### GPU-stage intervals

#### 32×32 only

| Case / width | 32×32 delta r0 / r1 (ms) | Sum of seven stages delta r0 / r1 (ms) |
| --- | ---: | ---: |
| Flower 500 / wide | −0.025 / −0.021 | −0.021 / −0.020 |
| Flower 500 / compact | −0.021 / −0.025 | −0.028 / −0.029 |
| HD / wide | −0.133 / −0.140 | −0.129 / −0.139 |
| HD / compact | −0.141 / −0.150 | −0.157 / −0.140 |
| 4K / wide | −0.931 / −0.651 | −0.352 / +1.137 |
| 4K / compact | −1.260 / −1.265 | +0.607 / +0.138 |
| Flower 2000 / wide | −0.334 / −0.297 | −0.600 / −0.446 |
| Flower 2000 / compact | −0.258 / −0.356 | −0.133 / −0.481 |

32×32 is favorable in 16/16 primary and 64/64 cross-label
comparisons; summed intervals in 13/16 and 55/64.

#### Production: 32×32 plus 16×16

| Case / width | 16×16 delta r0 / r1 (ms) | 32×32 delta r0 / r1 (ms) | Sum of seven stages delta r0 / r1 (ms) |
| --- | ---: | ---: | ---: |
| Flower 500 / wide | −0.003 / −0.005 | −0.026 / −0.021 | −0.026 / −0.022 |
| Flower 500 / compact | −0.005 / −0.005 | −0.021 / −0.025 | −0.028 / −0.027 |
| HD / wide | −0.034 / −0.034 | −0.135 / −0.141 | −0.167 / −0.172 |
| HD / compact | −0.037 / −0.037 | −0.137 / −0.147 | −0.149 / −0.188 |
| 4K / wide | −0.466 / +0.538 | −0.832 / −0.436 | −1.242 / +3.513 |
| 4K / compact | +0.238 / −0.473 | −0.794 / −1.412 | −0.124 / −2.154 |
| Flower 2000 / wide | −0.152 / −0.101 | −0.428 / −0.280 | −0.903 / −0.433 |
| Flower 2000 / compact | −0.068 / −0.102 | −0.312 / −0.398 | −0.371 / −0.578 |

32×32 is favorable in 16/16 primary and 64/64 cross-label
comparisons; summed intervals in 15/16 and 54/64.

The added 16×16 stage is favorable in 14/16 primary comparisons and
52/64 cross-label pairs. Variation in unchanged stages is not credited
as an algorithmic improvement. Event intervals may include stream idle gaps
and are not pure kernel-instruction durations. Successful queries establish
readiness when queried after the encode, not at the exact encode-return instant.

### Telemetry and traces

All 3,456 measured instrumented endpoints succeed for all six telemetry APIs.

| Reported field | Measured-endpoint range |
| --- | ---: |
| SM clock | 210–1,545 MHz |
| Memory clock | 5,500–5,500 MHz |
| Power draw | 22,992–66,872 mW |
| GPU temperature | 72–75 °C |
| Performance state | 3–3  |
| Throttle mask | 36–36  |

Reported power draw and enforced limits are separate fields. Encode endpoints
can occur in host-side gaps and can reflect sensor update intervals. They do
not establish clocks during an evaluator or prove a cause for a timing delta.

All 9,868 recorded enforced-limit endpoints remain at 40,000 mW. This does
not imply constant GPU clocks. The measured `0x24` throttle mask contains
the software-power-cap and software-thermal-slowdown flags in the pinned CUDA
11.8 NVML header. Neither this mask nor encode-endpoint clocks identify the
cause of a specific kernel or whole-call timing difference.

Eight ordinary Nsight Systems captures verify exact family dispatch, unchanged
192-thread grids, seven preparation/evaluation/finalization launches each, and
unchanged unrelated kernels and memcpy payloads. Summed evaluator durations
average the duplicate labels:

| Case / width | Baseline / 32×32 only / production (ms) |
| --- | ---: |
| HD / wide | 3.416 / 3.345 / 3.246 |
| HD / compact | 3.553 / 3.398 / 3.739 |
| 4K / wide | 26.605 / 23.959 / 17.691 |
| 4K / compact | 23.563 / 16.384 / 25.508 |
| Flower 2000 / wide | 6.407 / 6.049 / 6.332 |
| Flower 2000 / compact | 6.988 / 6.594 / 7.074 |
| Flower 500 / wide | 0.485 / 0.461 / 0.457 |
| Flower 500 / compact | 0.485 / 0.461 / 0.455 |

Two additional instrumented 4K captures have identical kernels, dimensions,
resource counts and memcpy payloads. Each labeled encode adds exactly fourteen
event-record API calls and no other in-window CUDA API count change. All
in-window calls succeed; the caller checks query, elapsed-time and destruction
status outside the windows.

## Disposition and next bottleneck

Retain the two-shape integration. It preserves the qualified native evaluator
bodies, reduces shared allocation without spills, passes broad normal-path
qualification, and repeats S142's 16/16 favorable 32×32 and 14/16 favorable
16×16 primary stage comparisons. The 32×32-only alternative remains diagnostic.
Do not generalize reuse to the five other shapes: their earlier register and
synchronization tradeoffs remain unfavorable, and their production instructions
are deliberately unchanged here.

This is a local evaluator improvement, not a proven universal encoder speedup.
The two-shape whole-call result is favorable in only 9/16 comparisons, with
a +7.577 ms first-wide-4K reversal and a −10.666 ms second-wide-4K result.
Three of four 4K whole-call comparisons are favorable, but the +12.711 ms
duplicate control prevents treating those deltas as a precise guaranteed gain.
The 16×16 stage has two unfavorable 4K observations, and the summed evaluator
interval is unfavorable in the second wide-4K instrumented process. Those
observations are retained, as are the three unfavorable ordinary trace sums.

Fresh production-label 4K traces identify substantial remaining GPU work:

| Work in production-label snapshots | Wide / compact mean kernel duration (ms) |
| --- | ---: |
| Paired Malta scale response, 64-thread true/false variants combined | 21.430 / 20.708 |
| Low/medium vertical convolution | 10.261 / 9.211 |
| Erosion/L2 finalization | 9.550 / 8.701 |
| All seven fused AC evaluators | 17.691 / 25.508 |
| All kernels | 144.222 / 141.619 |

These average labels 2 and 5 in each trace; the full ranked list and kernel
interval union are preserved in `bottlenecks.json`. They are workload snapshots,
not a controlled compact-versus-wide comparison. Their kernel interval unions
equal the sums here, but they do not explain all host-side or transfer time.
Next investigate the Malta/convolution data reuse and scheduling costs, with
GPU-active-window observations to distinguish operating-state variation from
kernel changes. The encoder is not established to be maxed out.

All 227 recorded jobs are terminal: 213 accepted, plus fourteen matched expected
rejections. The forty retained runtime files keep their hashes. No prior
artifact was deleted or overwritten, and no privilege/firewall blocker occurred.

Evidence root: `build-cuda-ninja/profiles/s143-artifacts/`; build, comparison,
qualification and analysis drivers: `build-cuda-ninja/profiles/s143_*`.
Late report, verification and archival drivers use `*_s143.py`, including
`report_s143.py`, `bottlenecks_s143.py`, `verify_s143.py` and `freeze_s143.py`.
Core records include `before.json`, `production_inputs.json`, `inputs.json`,
`trace_inputs.json`, `native_scan.json`, `diagnostic_native.json`, campaign
reports, stage/whole/trace analyses, `timing_summary.json`, `bottlenecks.json`,
`final_summary.json`, `source_snapshot_index.json` and `artifact_hashes.json`.
Historical binaries, inputs and oracles remain untouched. No device policy,
power, clock, thermal, firewall, affinity or priority settings were changed.
