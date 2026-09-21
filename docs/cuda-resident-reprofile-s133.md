# Complete resident critical-path refresh (S133)

After S131 filter fusion and S132 reconstruction-storage reuse, fresh
complete-encode traces identify substantial host preparation and strategy
selection gaps as well as GPU work. No runtime change is made in this study.
The next bounded experiment is to remove unnecessary host initial-field and
mask storage from encoding-only resident preparation, then measure the whole
encode with duplicate controls. The backend is not established to be maxed out.

## Scope and method

Starting revision: `c4d54e9`, branch `feat/cuda`, September 8, 2026.
Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37, Nsight Systems
2023.2.3. All measured power-limit endpoints are 40 W. No power, clock,
firewall or privilege setting was changed. There was no firewall/elevation
blocker. Nsight's local help states that CPU stack sampling requires
administrator privilege; this study uses `--sample=none --cpuctxsw=none`,
CUDA/NVTX timelines and existing host phase counters instead.

The four inputs are Flower 500 by 500, padded HD from a 1919 by 1079 source,
padded 4K from a 3839 by 2159 source, and Flower 2000 by 2000. The last is
fourfold nearest-neighbor replication of the 500-square crop, **not an
independent native-resolution photograph**. Distance 1.2, effort 7 and fully
resident encoding are fixed. Wide and opt-in compact coefficient owners are
tested separately; production compact defaults remain unchanged.

Four phase executables link frozen S132 libraries: Release and scoped host
ASAN, each wide/compact. Two additional diagnostic executables replace three
host translation units with source-identical copies plus scoped NVTX labels.
Removing those labels reproduces the original sources exactly. All ten
linked GPU modules in all six executables match S132; all 214 kernel bodies
are unchanged. Host ASAN covers the diagnostic caller and resident owner,
not every linked library. No new CTest or CUDA-sanitizer qualification is
claimed for this profiling-only study.

Uninstrumented timing uses four warmups plus one reference encode and sixteen
measured encodes per process, repeated with reversed image/width order. These
are separate-process width comparisons, not a balanced within-process A/B
experiment, so differences do not establish a causal compact speedup.
Each encode checks frozen codestream bytes, within-process summary equality,
and stable coefficient width and storage size.

Sixteen complete-encode CUDA traces provide three measured NVTX windows each.
Eight additional scoped-host traces cover Flower 500 and padded 4K, both
widths and two reversed repeats, again with three measured windows each.
Warmups are outside the captured range. Every qualified capture includes
terminal correctness and power records. Trace timings include instrumentation
overhead and are used for attribution, not uninstrumented latency claims.

## Uninstrumented wall phases

Ranges below span the two process medians, each over sixteen measured encodes.
The broad phases are wall time; nested parallel `*_work` counters must not be
summed as elapsed time. Medians of component times need not sum to the median
of the whole call.

| Input / storage | Complete call ms | Input preparation ms | Quantization pipeline ms | Codestream serialization ms |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 / wide | 18.57–19.41 | 0.94–0.97 | 9.07–9.58 | 8.19–8.47 |
| Flower 500 / compact | 18.02–19.44 | 0.95–0.98 | 8.91–9.18 | 7.88–8.88 |
| HD / wide | 71.73–76.71 | 5.72–6.05 | 42.65–43.90 | 21.18–24.11 |
| HD / compact | 66.29–68.51 | 5.74–5.78 | 38.74–39.39 | 20.93–22.01 |
| 4K / wide | 284.45–292.33 | 22.55–22.96 | 207.29–212.75 | 48.86–49.46 |
| 4K / compact | 274.76–285.12 | 22.78–23.64 | 196.52–199.28 | 47.64–55.03 |
| Flower 2000 / wide | 132.36–138.18 | 11.05–11.38 | 84.02–84.16 | 33.89–37.94 |
| Flower 2000 / compact | 121.92–123.27 | 11.05–11.23 | 74.13–75.41 | 33.35–34.20 |

Compact uses int16 for Flower 500 and int8 for the other three inputs. At
padded 4K its AC host owner is 26,542,080 rather than 106,168,320 bytes.
The traced total D2H volume is 29,073,992 rather than 103,723,588 bytes; the
74,649,596-byte difference includes the extra four-byte compact-width flag.
H2D volume is unchanged at 112,566,740 bytes. D2H activity medians are
4.57–5.04 ms compact versus 16.64–22.11 ms wide. These figures include all
copies in the measured encode, not just the largest contiguous AC run.
Transfer-time differences are not a substitute for controlled whole-call
comparisons, nor evidence to change compact's default by themselves.

## GPU work and idle intervals

The plain 4K traces contain 301 wide or 302 compact kernel launches per
encode. Across their four process medians, summed kernel time is
147.64–161.43 ms. Internal gaps between recorded GPU activities total
26.25–38.62 ms. The portion before the following CUDA API begins is
22.89–34.71 ms; gaps after that API returns contribute only 1.53–1.67 ms.
Those timing categories alone are not proof of a driver, scheduler or host
function cause. Synchronization API durations overlap useful GPU execution
and must not be counted as additional avoidable idle time.

Supplemental host labels resolve major gaps. The following entries are
inclusive host durations, and each also lies entirely inside GPU-idle time
in these captures. Candidate construction sums calls across strategy shapes.

| Padded 4K process | Resident pipeline preparation ms | Candidate construction ms | CPU cost merge ms | Resident metadata construction ms |
| --- | ---: | ---: | ---: | ---: |
| Wide / repeat 0 | 7.020 | 4.776 | 9.004 | 2.315 |
| Wide / repeat 1 | 6.914 | 4.552 | 11.761 | 2.590 |
| Compact / repeat 0 | 6.439 | 4.891 | 8.997 | 2.550 |
| Compact / repeat 1 | 6.935 | 4.749 | 16.518 | 2.552 |

The scope analyzer partitions each GPU gap at host-range boundaries and
assigns each segment to its innermost label. Parent and child idle durations
are therefore not double-counted. Unlabelled time remains unlabelled. Other
time in the search owner includes preparation, cost readback/scattering and
API activity; it is not all assigned to merge arithmetic.

Butteraugli remains a major GPU cluster. Template-aware 4K kernel medians
include 13.28–13.88 ms for the sixteen paired LF Malta launches, 8.76–9.33 ms
for the eight other paired Malta launches, 10.03–10.63 ms for six low/medium
row convolutions, and 8.74–9.80 ms for six fused mirrored-opsin launches.
These are ranges of per-process medians, not simultaneous worst-case sums.

Short kernel names can be misleading. `InverseDctFactoredKernel` accounts
for much more than AQ reconstruction: the residual/loss transforms in
strategy search are separate template instantiations. For example, 4K
32-by-32 strategy residual inverse DCT totals 2.47–2.97 ms per encode, while
the 32-by-32 AQ image-output specialization totals about 1.60–1.61 ms.
Full template names and per-launch geometry/resources are retained in the
analysis. Future transform experiments should target the actual specialization
and consumer, not the aggregated short name.

## Selected next investigation

`PrepareResidentQuantizationPipeline` allocates and zero-initializes host
`initial_quant`, `strategy_mask` and `pixel_mask` vectors. The pixel mask
alone contains 33,177,600 bytes at padded 4K. Yet the encoding-only resident
initial-quantization interface requests no host outputs, and
`RunPreparedQuantizationPipelineWithProviders` passes empty host field/mask
views when `resident_initial_quantization` is true. Strategy search consumes
the evaluator's device-resident inputs instead.

This source evidence makes eliminating that unused materialization the first
bounded candidate. The 6.4–7.0 ms preparation scope is an opportunity bound,
**not a measured saving attributable solely to the vectors**. The experiment
must distinguish encoding-only execution from genuine host-output requests,
maximum-error routing and prepared reuse. It must preserve transactional
failure behavior and compare complete encodes with duplicate controls before
promotion. No compatibility shim or general persistent host cache is needed.

CPU candidate construction and cost merging are subsequent targets. S112
already demonstrated a faster local parallel merge without a dependable
whole-workflow gain; these new traces do not justify restoring that rejected
scheduler. Consider reducing candidate preparation/scattering or changing
the representation before retrying thread scheduling. Malta/convolution
specializations and compact transfer policy remain independent GPU/transfer
targets. No single remaining filter kernel dominates the entire encoder.

## Audit and reproduction

Evidence is retained in `U:/gjxl-cuda-diagnostics/s133`, with reproduction
drivers under `build-cuda-ninja/profiles/s133_*`. Run into a new artifact root
and retain the frozen S132 libraries and input/oracle hashes. Main outputs
are `phase_analysis.json`, `trace_analysis.json`, `scopes_analysis.json` and
`ranking.json`, plus source snapshots and the artifact SHA-256 manifest.

The validator checks 71 recorded jobs with successful process exits. One
initial trace is explicitly excluded: `--capture-range-end=stop` ended the
capture before terminal stdout/power records were exported, so the collector
rejected its completeness despite the profiler's zero exit code. That report
is preserved. Corrected captures use `--capture-range-end=none` and continue
to process exit; there is no silent retry or overwritten evidence.

The qualified population comprises 64 encode processes, 576 frozen-oracle
encodes (sixteen with scoped host ASAN), 256 uninstrumented measured encodes,
and 72 measured windows across 24 complete traces. All 1,152 power-limit
endpoints are 40 W. Endpoints do not prove constant clocks or absence of
transient changes. The validator also checks source isolation, linked GPU
modules, exact timeline decompositions, lack of overlap between measured
jobs and other recorded jobs, and forty retained runtime hashes. Historical
oracles, runtime binaries and the three protected scratch files are untouched.
