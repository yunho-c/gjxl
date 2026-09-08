# Fused device generation of strategy candidates (S136)

**Disposition: not promoted.** Generating the regular candidate descriptors
inside the existing quant-norm pass removes host construction and seven
uploads. At 4K the candidate saves 12,530,880 host-to-device bytes and
6.7–7.8 ms of measured preparation, without adding a kernel launch. Complete
encoding nevertheless improves in only eleven of sixteen primary comparisons;
three of four 4K comparisons are unfavorable. The tested implementation and
regression fixture are archived, and runtime source is restored to S134 through
`cdfe007`. No experiment selector, capability addition or compatibility layer
is retained in production. This does not establish that the encoder is maxed out.

## Hypothesis and implementation

S135 identified a larger removable representation than packed-cost scattering:
522,120 regular descriptors, each 24 bytes, are constructed and uploaded at
padded 4K. On the normal resident route their image-dependent quantization and
CfL values are already supplied through device views. Coordinates follow the
fixed search policy; the remaining descriptor values are policy constants.

The candidate adds a writable generated-descriptor output to the checked
batch contract. The CUDA frontend reports support explicitly; Metal reports
no support and rejects requests. Explicit descriptor evaluation remains a real
operation for nonresident and other callers, not a version-compatibility shim.
The experimental API is replaced directly and all consumers are rebuilt.

For the generated route, the host counts fitting anchors from the policy's
seven footprints and steps, accounting for partial color tiles. It does not
construct or upload the descriptor vector. A fused device kernel decodes each
linear index to color-tile y/x and local-anchor y/x, writes the exact descriptor,
and computes the existing quant norm into the temporary cost output. Ordered
forward, residual/inverse and final-cost consumers then read that descriptor.
The ordinary explicit-descriptor preparation kernel is kept separately.

The device descriptor arena is still allocated: this is **not a VRAM reduction
or fully implicit descriptor representation**. The common transform parameter
struct is unchanged; a separate 28-byte grid parameter is passed only to the
fused preparation kernel. Dense host costs, CPU hierarchy/tie-breaking, merge
order, coefficient-width policy and tile scheduling are unchanged. The host
recovers regular coordinates directly when scattering returned costs, so no
descriptor readback is introduced. The rejected S135 packed consumer is not
combined with this experiment.

| Input / padded blocks | Descriptors | Cold host descriptor bytes removed | Measured H2D bytes removed | Launches, wide / compact |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 / 63 × 63 | 15,879 | 381,096 | 381,096 | 366 / 367 |
| HD / 240 × 135 | 130,380 | 3,129,120 | 3,129,120 | 327 / 328 |
| 4K / 480 × 270 | 522,120 | 12,530,880 | 12,530,880 | 301 / 302 |
| Flower 2000 / 250 × 250 | 250,923 | 6,022,152 | 6,022,152 | 366 / 367 |

Launch counts are identical for baseline and candidate. The within-process
diagnostic verifies zero candidate host-vector capacity for each cold normal
resident encode and unchanged logical device descriptor bytes. A prepared
owner previously used for explicit descriptors may retain its vector capacity
after `clear()`; this experiment does not claim to release that retained
capacity. These figures are not RSS, pool reserve or measured PCIe bus traffic.
H2D figures are CUDA memcpy payload bytes from traces, not kernel-argument
traffic. The archived savings are not active in the restored checkout.

## Validation and native-code isolation

Starting revision: `cdfe007`, branch `feat/cuda`, September 8, 2026.
Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37, Release; scoped host
ASAN uses clang-cl. A clean CMake/Ninja CUDA build passes **82/82 CTests**, with
none skipped. The added test is part of the archived candidate, not the
restored production test suite. Compact and ASAN resident-owner objects are
freshly compiled because the batch layout and capability interface changed;
no frozen S132/S134 resident objects are linked into this experiment.

The focused GPU fixture covers 72 nonempty geometry/strategy cases and
14,938 descriptors per execution. Geometries include thin images, full and
partial tiles, and long axes through 257 blocks. Strided RGB, mask, quant and
signed CfL inputs share a guarded arena with disjoint matrices, descriptors,
scratch and costs. An independent copy of the original nested host enumeration
produces the descriptor oracle. Explicit GPU evaluation supplies final-cost
bits; two generated evaluations overwrite poisoned descriptors and must match
both descriptors and costs exactly while preserving every input/guard byte.

Across all seven stages at 33 × 17 blocks, 175 invalid requests per execution
check mixed explicit/generated pointers, wrong counts, missing resident fields,
misalignment, insufficient range, input/output overlap, partial overlap and
foreign ownership. They allocate/submit nothing and leave caller storage
unchanged. Empty generated batches remain no-work, and a valid retry overwrites
all descriptors before consumption. The fixture passes standalone, CTest,
scoped ASAN, CUDA memcheck and CUDA initcheck: five executions, 875 invalid
requests in total. This is rejection/retry coverage, not new injection of
actual asynchronous submission/completion failures.

An exhaustive diagnostic additionally covers every block width/height pair
from 1 through 19: 2,201 nonempty geometry/strategy cases and 134,809 descriptors,
each in Release and scoped ASAN. It checks the same exact descriptors, final
costs and unchanged input/guard bytes. Across focused and exhaustive runs there
are 9,559 generated and 4,762 explicit GPU evaluations; repeated checks are not
claimed as independent images.

Whole-encode validation includes 80 successful production configurations and
14 writer rejections matching the frozen S108 errors, sixteen production ASAN
configurations, four production memchecks and four production initchecks.
Quality, rate search/prepared reuse, stress and one-/two-request batch routes
cover both coefficient widths. Production checks total 1,296 frozen-oracle
encodes. Within-process preflights/timing add 1,376 and traces add forty:
**2,712 frozen-oracle encodes, including 262 scoped host-ASAN encodes**, plus
the fourteen matching rejections. Ten CUDA sanitizer jobs qualify, including
the two focused grid jobs; memcheck reports zero errors and zero leaked bytes,
and initcheck reports zero errors. ASAN covers modified host search/validation,
the caller/resident owner and S134 pipeline units, not every library. There is
no new independent decoder, Metal or Linux qualification claim.

All fifteen audited GPU executables contain the same ten current modules.
Nine modules are byte-identical to S134. The AC-strategy module adds the
generated preparation kernel, for 215 kernels in total: 213 S134 bodies are
unchanged, one source-unchanged explicit quant-norm body has a historical
code-generation variant, and one kernel is new. The explicit norm's instruction
and control bytes match frozen S132/S133 exactly. Its difference from S134 is
the already observed zero-initialization/move-scheduling variant, not a change
to its source formula. No compiler root cause is established here.

Both preparation kernels use 35 registers and no stack, local or shared
storage; generated preparation uses 476 constant bytes versus 448 for explicit
preparation. The unchanged final-cost kernel uses 22 registers. Transform and
perceptual-analysis modules are byte-identical to S134.

## Balanced complete-encode timing

Inputs match S133–S135: Flower 500, HD from 1919 × 1079, 4K from 3839 × 2159,
and Flower 2000. The last is fourfold nearest-neighbor replication of the
500-square crop, not a native 2000-square photograph. Distance is 1.2, effort
7, fully resident, with wide and opt-in compact coefficients.

One instrumented frontend uses the common candidate owner and production
implementation, with a diagnostic-only condition selecting explicit versus
generated preparation. Labels 0/2 are duplicate baselines; 1/3 are duplicate
candidates. Every four-round Williams block balances label positions and
ordered predecessors. Each process performs one reference encode, four warm
rounds and sixteen measured rounds. Two process passes reverse case/width
order: sixteen timed processes, 1,024 measured encodes. No recorded task build,
qualification or trace overlaps these timed processes. Output bytes, summary,
coefficient width/storage and branch/allocation counters are checked per encode.

The primary statistic is the median, across rounds, of the mean candidate-label
time minus the mean baseline-label time. It is not a difference of independent
medians. Negative means faster. Preparation ends immediately before candidate
evaluation; readback/scatter and CPU merging are timed separately. Whole-call
time includes input preparation, quantization and serialization.

| Case / width | Whole-call delta r0 / r1 (ms) | Search delta r0 / r1 (ms) | Preparation delta r0 / r1 (ms) |
|---|---:|---:|---:|
| Flower 500 / wide | −0.600 / +0.028 | −0.315 / −0.298 | −0.300 / −0.281 |
| Flower 500 / compact | −0.484 / −0.660 | −0.294 / −0.279 | −0.288 / −0.288 |
| HD / wide | −0.728 / −1.804 | −1.798 / −2.076 | −1.735 / −2.139 |
| HD / compact | −2.632 / −3.436 | −1.857 / −2.084 | −1.914 / −1.928 |
| 4K / wide | +3.625 / −0.357 | −2.080 / −8.339 | −6.650 / −7.761 |
| 4K / compact | +11.110 / +0.043 | −2.730 / +0.934 | −7.621 / −6.875 |
| Flower 2000 / wide | −1.921 / +2.092 | −5.741 / −4.599 | −4.196 / −3.747 |
| Flower 2000 / compact | −0.995 / −3.391 | −5.283 / −4.697 | −4.133 / −4.116 |

Preparation and readback/scatter favor the candidate in all sixteen primary
comparisons and all 64 candidate/baseline label pairs. Enclosing search favors
it in 15/16 primary comparisons and 61/64 pairs. Whole-call results are 11/16
and 44/64; quantization is 10/16 and 44/64. All four 4K quantization primary
deltas are unfavorable, despite the preparation saving. Duplicate controls
remain substantial: the first compact 4K pass has +9.834 and +5.963 ms
whole-call differences between equivalent labels. These results do not justify
a universal speedup claim or an image-size threshold chosen after the fact.

## Transfer and GPU-work crosscheck

Eight complete-process Nsight Systems captures contain 32 labeled encode
windows plus eight reference encodes. They are correctness/transfer evidence,
not replacements for unprofiled timing. CPU sampling/context-switch collection
is disabled; collection continues through terminal stdout and all power
records. Each window has matching kernel/API counts, successful CUDA calls,
seven preparation-kernel substitutions and seven fewer H2D copies. D2H and
other copy counts/bytes are unchanged. The 4K H2D reduction is exactly
12,530,880 bytes in every candidate/baseline label comparison.

In the two 4K traces, summed preparation-kernel time increases by only
0.039–0.044 ms. Internal GPU-idle time falls by 7.77–7.89 ms and H2D copy time
by 2.07–2.08 ms, while total kernel time grows by 9.66–13.96 ms. The growth is
distributed across unchanged residual-transform and perceptual-analysis
kernels, not explained by the new preparation kernel alone. These short,
instrumented windows do **not** establish a power, clock, cache, driver,
allocation-placement or RDP cause. Constant enforced power limits do not
prove constant clocks or interference-free execution.

All 2,832 recorded power endpoints (2,752 within-process and eighty trace)
report 40 W. No power/clock/priority/affinity/firewall policy was changed, and
no firewall or elevation blocker was encountered. Reducing a host interval
cannot by itself establish a public-encode improvement on this system.

## Disposition and evidence

The candidate's source and test are preserved, but all eleven modified
production/test/build files are restored exactly in normalized content to
starting revision `cdfe007`; the new test is removed from the working tree
and remains recoverable in the archive. Forty retained runtime files keep
their recorded hashes. The commit contains documentation only.

Next investigate active GPU work, particularly candidate residual transforms
and perceptual analysis, with complete-encode controls and better attribution
of the observed GPU-time variation. A fully implicit or narrower device
candidate representation remains a possible separate experiment; the present
one only removes host construction/upload. Do not re-enable the rejected S112
CPU scheduler or S135 packed consumer on the strength of these local savings.

Evidence root: `U:/gjxl-cuda-diagnostics/s136/`. Principal records are
`candidate_index.json`, `within_inputs.json`, `production_inputs.json`,
`trace_inputs.json`, `within_analysis.json`, `timing_summary.json`,
`trace_analysis.json`, `trace_cost_analysis.json`, the four
`production_*.json` manifests, `native_probe.json`, `native_summary.json`,
`native_crosscheck.json`, `linked.json`, `trace_linked.json`, job logs,
`source_snapshot_index.json`, `final_summary.json` and `artifact_hashes.json`.
Drivers are the archived `build-cuda-ninja/profiles/s136_*` files.

There are 173 ordinary terminal job records: 159 successful and fourteen
expected writer rejections. Separately, the first native-probe wrapper failed
to write its metadata because its report name collided with the completed
child analysis `native_probe.json`. The child analysis/stdout are preserved;
`native_probe_wrapper_collision.json` records the bookkeeping failure without
inventing a finish timestamp or PID. The distinct `native_audit_v2` and
`native_crosscheck_job` records verify those results. This collision is not
counted as a GPU correctness failure or a successful wrapper job. No live
process is left waiting behind it.
