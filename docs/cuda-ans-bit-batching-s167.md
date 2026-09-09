# S167: batched ANS bit emission (retained, with mixed 4K results)

## Scope and hypothesis

S165's diagnostic decomposition identified bit emission as a major dense-input
cost. S166 removed repeated AC model validation but did not establish a stable
dense-input benefit. This experiment tests coalescing the ANS writer's adjacent
reverse chunks before calling `BitWriter::WriteBits`.

The predecessor is S166, commit `7ee6dcd5b9d54558dfde2fa17961910dd446f31f`.
Only `src/codestream/ans.cpp` changes runtime behavior; the promotion phase
also adds a permanent regression test and its CMake registration. The change
is retained for repeatable dense-input and majority-corpus benefits, with the
4K limitations below explicitly accepted rather than declared resolved. Aligned append,
reverse-chunk storage, ANS recurrence, model construction, coefficient layouts,
GPU kernels, and scheduling remain unchanged.

## Packing and safety

The old emission loop writes the 32-bit final state, then calls `WriteBits`
once per reverse chunk. The prototype begins a `uint64_t` accumulator with
that state and appends chunks in exactly the same order. Before adding a chunk
that would exceed `BitWriter::kMaxBitsPerWrite` (56), it flushes the pending
bits and clears the accumulator. It flushes the remaining bits at the end.

Generated chunks have at most 31 HybridUint extra bits or 16 renormalization
bits. Capacity is checked before shifting; the accumulated value never exceeds
56 logical bits. Empty token streams still emit their 32-bit state. Nothing
adds padding or changes logical output length. The temporary writer and its
final checked append remain in place, as do all model/token/state checks.
Chunk values are generated internally from checked HybridUint configurations
or a masked 16-bit ANS state; this is not a new API accepting arbitrary
caller-provided chunk widths or values. The independent-reference width
matrix below checks that invariant across the supported configurations.
No unchecked word stores, endian-dependent reinterpretation, new API,
compatibility layer, or allocation-reservation change is introduced.

## Initial host qualification

The standalone `s167_writer_test.cpp` constructs valid flat 256-symbol ANS
models and uses an independent reference recurrence with ordinary division
and modulo, checked HybridUint conversion, and unbatched writes. It does not
reuse the candidate's recurrence or batching loop to produce expected bytes.
Candidate public and validated-model internal writers both match that oracle
and the count-only path's exact logical length.

Each baseline-normal, candidate-normal, and candidate-ASAN execution passes:

- All 816 valid HybridUint configurations, mapped/unmapped contexts, and
  interleaved/offset-split token storage. Values that cannot fit the ANS
  alphabet are omitted from this valid-stream cross product.
- Every initial bit alignment, 0 through 7, and all eight output tail widths.
- Short-stream lengths 0 through 257 with deterministic full-width values.
- 30,240 comparisons and 30,240 insufficient-allotment failures preserving
  the destination's original bits. Exact-size allotments also succeed.
- All extra-bit widths 0 through 31 and exact 56-bit accumulator boundaries.

The reference observes 1,651,280 original state/chunk writes. An independent
simulation predicts 808,688 batched writes over these fixtures. The latter
is a prediction from the reference chunks, not an instrumented count of
production calls or an end-to-end performance claim. These flat fixtures are
not exhaustive ANS model coverage; the existing entropy suite additionally
exercises optimized models, ordinary/ANS policy paths, malformed inputs, and
40 late ANS failures.

Candidate normal and ASAN builds also pass the existing entropy and private
AC-section validation suites. The latter covers 192 valid batches, 1,408
invalid models, 96 late token failures, and 64 shared-model concurrent checks
per run. In total seven host test processes pass their test assertions:
one baseline writer, three candidate normal, and three candidate ASAN tests.
The pilot did not include full CTest, a new allocation-fault campaign, or
ThreadSanitizer. The full-suite follow-up is recorded separately below.

The initial launcher incorrectly expected `All entropy tests passed.` instead
of the actual `All entropy primitive tests passed.`. Its entropy process
returned zero and passed its assertions, but its journal was rejected.
The original log and journal remain unchanged; a separate recovery helper
validates the actual marker/hash and runs the remaining tests. No codec
source correction or failed-test retry was required.

## Whole/retained and CUDA qualification

Forty-four uninstrumented comparisons cross all six established real images
and five synthetic inputs with whole/retained mode and automatic/eight
participants. Each compares four encodes and a baseline oracle. Bytes and
public summaries match exactly; real baselines additionally match frozen
codestream files. Retained frames check native fingerprints and serialization
against a captured whole encode, with no further GPU allocation/submission.

CUDA memcheck with full leak checking and initcheck both pass whole 4K/eight
comparisons: zero errors, zero memcheck leaks, four byte checks plus an oracle
per job. All 11 raw CUDA modules in each candidate DLL match frozen S162/S166
modules. The DLLs reuse frozen S166 bridge objects and libraries, overridden
only by the newly compiled ANS object. ASAN executables use the new ANS object
with the frozen S166 encoder overlay and S162 sanitized libraries where needed.
No frozen predecessor file is rebuilt or overwritten.

## Timing pilot protocol

The predeclared exploratory pilot has 96 jobs. Primary inputs are synthetic
65 `p2`, sparse 513 `p0`, dense 513 `p2`, dense 1025 `p2`, flower 500, and 4K.
Synthetic height is width plus six. They cross whole/retained mode,
automatic/eight participants, and both DLL/backend creation orders.

Both-build, same-DLL independent-state controls cover the exact dense 513,
dense 1025, and 4K inputs in both modes/budgets/orders. One seeded shuffle
interleaves all 48 primary and 48 control jobs. There are four warmup and eight
measured rotating duplicate-ABBA rounds per job: 3,072 measured calls,
1,536 warmups, 96 oracle calls, and 96 untimed whole captures for retained
inputs if the complete schedule succeeds.

Primary statistics use medians of within-round duplicate-label mean
differences. Four cross-label medians are sensitivity checks, not confidence
intervals. Controls are not subtracted. Setup, previous-output cleanup, result
queries, byte checks, retained capture, and NVML observations are outside
outer timing. Worker-profile sums are not additive wall time. No build or
sanitizer overlaps timing. No power, clock, thermal, affinity, priority,
firewall, or security setting is changed.

## Pilot results

The complete schedule ran on 2026-09-09 from 22:06:36 to 22:13:58 UTC, with
all 96 jobs accepted and no retries or dropped samples. Counts match the
protocol above. All 9,216 timing NVML observations report a 40,000 mW enforced
limit; this does not imply constant operating power, clocks, temperature, or
exclusive use of the machine. Small read-only checks and documentation edits
occurred during the campaign, but no build or sanitizer overlapped it.

Whole-encode outer time improves in 23/24 primary trials and 80/96 cross-label
comparisons. Retained outer time improves in 20/24 and 79/96 respectively.
Section wall improves in 23/24 whole and 21/24 retained trials. Token-writing
worker sums improve in 24/24 whole and 23/24 retained trials. These counts do
not establish a universal speedup.

Dense 513 and 1025 show the clearest signal: all eight primary trials per
mode and all 32 cross-label comparisons per mode improve in outer, section,
and token-worker time. Dense whole outer gains span 5.98–15.77%, retained
7.95–23.08%; section wall improves 21.06–36.60% whole and 19.94–38.40% retained.
Those section/worker percentages must not be presented as whole-encode gains.

All primary outer-time percent results follow. Each cell is creation order
`O0, O1`; negative means faster. Millisecond and percentage deltas are separate
medians and may disagree in sign near zero.

| Case | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| 65 `p2` | −6.032, −2.807 | −5.202, −3.205 | −5.949, −5.503 | −1.807, −4.332 |
| 513 `p0` | −0.154, −1.402 | −2.293, −0.905 | +1.752, −0.950 | −4.737, −5.550 |
| 513 `p2` | −11.971, −10.427 | −9.835, −5.983 | −7.953, −14.264 | −11.525, −23.076 |
| 1025 `p2` | −15.767, −14.898 | −12.654, −15.487 | −20.454, −15.763 | −14.625, −16.745 |
| flower 500 | −2.546, −1.802 | −1.708, −1.803 | −5.212, −1.784 | +0.168, −5.496 |
| 4K | −0.187, −0.671 | +1.281, −1.434 | −2.951, −2.041 | +3.461, +4.273 |

Both tested real images have faster token-writing worker sums in all eight
trials per mode, but whole outer gains are small and retained 4K/eight is
slower in both creation orders. Its O0 section wall is +5.408% (+0.443 ms)
while token worker time is −0.289% (−0.081 ms). O1 section wall is −2.243%
(−0.246 ms) and token worker time −8.171% (−2.203 ms), despite outer time
+4.273% (+2.497 ms). These overlapping/separately aggregated phases are not
additive, and the prototype cannot yet be called harmless on this workload.

The exact-input control results are retained without subtraction:

| Control case/build | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| 513 `p2` baseline | −3.214, −5.195 | +1.941, +1.335 | +1.684, +2.245 | +4.846, −0.005 |
| 513 `p2` candidate | −2.509, +6.514 | −2.151, −0.294 | +2.617, −0.799 | +3.851, −0.192 |
| 1025 `p2` baseline | −0.636, +0.404 | −1.162, +4.708 | +1.091, +1.223 | −0.472, +1.216 |
| 1025 `p2` candidate | +0.360, +2.196 | −1.314, −3.571 | +2.779, +3.399 | +1.350, +3.068 |
| 4K baseline | +0.223, +0.011 | −1.575, −4.142 | −0.038, +2.346 | −2.808, +0.401 |
| 4K candidate | +3.605, −1.021 | −0.227, +2.311 | +1.768, +5.574 | +1.694, +2.855 |

Dense primary effects are consistently favorable and generally larger than
the controls, but controls are not error bars or an explanation that removes
a regression. In particular, retained 4K/eight needs a repeat and broader
real-image coverage. No image-specific dispatch threshold is inferred.

## Pilot gate and phase evidence

The pilot justified a broader promotion run, not an immediate commit. The
standard-build follow-up below integrates the test, broadens qualification,
and completes repeated full-corpus timing. The focused 4K follow-up and final
retention decision are recorded below. Aligned append remains separate so
it cannot obscure the effect of batching.

`verify_s167_pilot.py` successfully recomputes the paired analysis, checks all
150 journals (149 accepted plus the preserved success-marker rejection),
source/library hashes, native modules, test markers, qualification metadata,
and timing schedule/non-overlap. `s167_record_pilot.py` preserves the phase's
source/helper inputs, protocol/analysis hashes, and verification summary
before later test or build integration. This is a phase snapshot, not the
final stage freeze or production retention decision.

## Standard-build follow-up

Before changing test registration, preparation revalidated and froze all
1,401 pilot artifact files. The follow-up writes to
`U:/gjxl-cuda-diagnostics/s167-promotion`, separately from that frozen pilot.
The runtime ANS source remains byte-identical to the pilot source. The new
`tests/ans_bit_emission_test.cpp` is the pilot test with only its printed
success-label name changed; CMake registers it as `ans_bit_emission`.

The existing mutable Ninja build recompiles CPU code and relinks consumers;
no CUDA source is compiled. The integrated writer test passes directly, and
the complete configured CTest suite passes 92/92 tests in 293.82 seconds.
All 92 unique passed names are checked against CTest's JSON test inventory.
The normal codestream library and eight relevant normal test executables
are archived on U:. Optional unavailable conformance tools are not claimed
as newly tested.

Seven newly linked ASAN executables pass: bit emission, entropy, AC section
validation, codestream encoder, public codestream workflow, compact frame,
and sparse frame. The writer test retains its 30,240 exact comparisons and
30,240 allotment failures. Compact and sparse frame counts match the existing
S166 suite markers. Those representation tests are not an independent full
entropy-policy cross product. The sanitized ANS object is the frozen pilot
object, tied to the unchanged runtime source; the encoder overlay and other
sanitized libraries are frozen S166/S162 inputs.

New whole/retained DLLs link the standard CMake-built codestream library and
frozen S166 bridge objects/unchanged dependency libraries. All 11 CUDA modules
per DLL match the frozen predecessor. All 44 whole/retained, auto/eight,
eleven-case byte-oracle comparisons pass. The new DLLs also pass 4K/eight
CUDA memcheck with zero errors and zero leaks, and initcheck with zero errors.
The qualification verifier independently recomputes the frozen pilot analysis,
checks the new test-only integration changes, and validates the normal/ASAN,
byte-oracle, native-module, relocation, and input-hash evidence.

The predeclared follow-up timing plan has 336 jobs: 176 primary comparisons
and 160 both-build independent-state controls. It covers all eleven inputs,
both modes/budgets, and two shuffled repeats each crossing both creation
orders. Controls cover tiny 65, sparse 513, dense 513, dense 1025, and 4K.
The four-warm/eight-measured duplicate-ABBA protocol is unchanged. It produced
10,752 measured calls, 5,376 warmups, 336 oracle calls, and 336
retained-input whole captures. It is not pooled with the exploratory pilot.
Qualification verification passed before the campaign started at
2026-09-09 22:37:10 UTC. It completed at 23:00:49 UTC with all 336 jobs
accepted, no retries, and no dropped samples. Counts match the plan above.
All 32,256 timing NVML observations report a 40,000 mW enforced limit, not
constant operating power, clocks, temperature, or exclusive machine use.

### Full-corpus results

Whole outer time improves in 69/88 primary trials (245/352 cross-label
comparisons), retained outer time in 67/88 (262/352). Whole section wall
improves in 81/88 trials and retained section wall in 80/88; token-writing
worker sums improve in 82/88 in each mode.

Both dense inputs improve in every one of the 16 primary trials per mode
and all 64 cross-label comparisons per mode, for outer, section wall, and
token-worker time. Whole outer gains span 4.84–16.48%, retained 8.62–20.53%.
Section gains span 19.22–35.73% whole and 21.26–32.62% retained; worker-sum
gains span 24.34–39.34% and 24.73–35.32% respectively. Only the outer figures
are end-to-end changes.

Real-image results are less uniform: whole outer improves in 34/48 trials,
with a range from 7.02% faster to 3.34% slower; retained outer improves in
31/48, from 8.60% faster to 5.32% slower. Section wall improves in 44/48 in
each mode; token worker time improves in 47/48 whole and 46/48 retained.
The whole/retained real outer cross-label comparisons are 115/192 and
118/192 faster. These are mixed measurements, not a universal speedup.

All full-corpus primary outer percentages follow in
`R0/O0, R0/O1, R1/O0, R1/O1` order; negative means faster.

| Case | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| 65 `p2` | −2.872, −2.344, −2.277, −3.088 | −3.670, −4.632, −1.039, −5.476 | −4.303, −5.029, −3.060, −4.648 | −3.662, −4.805, −5.249, −4.414 |
| 513 `p0` | −0.357, +2.102, −1.270, −0.403 | +1.701, −0.815, −0.941, +0.311 | +0.310, −1.012, +1.611, +0.177 | −1.291, +1.987, −1.141, −1.233 |
| 513 `p1` | −3.486, −7.674, +0.324, −0.957 | +1.930, −3.586, −2.044, −1.909 | −3.766, −2.682, −0.696, −2.642 | −1.089, −4.430, −5.349, −1.939 |
| 513 `p2` | −11.193, −7.966, −4.838, −7.633 | −8.735, −7.475, −6.894, −10.519 | −12.041, −10.208, −14.850, −12.888 | −9.014, −13.446, −13.294, −17.235 |
| 1025 `p2` | −13.397, −9.642, −12.436, −16.036 | −15.111, −16.481, −15.210, −15.298 | −17.317, −20.530, −14.292, −14.334 | −14.476, −18.215, −17.063, −8.616 |
| flower 500 | −2.530, +0.174, −2.819, −1.990 | −0.858, −2.203, −0.462, −2.753 | +0.992, −3.460, −1.778, +0.266 | −3.345, +2.413, −1.003, +1.239 |
| 1080p | −2.807, −1.100, −2.753, −0.741 | +0.146, +0.415, −2.069, −4.528 | −0.073, +0.747, −1.206, −0.362 | −1.404, +0.367, +1.044, +1.347 |
| flower 2000 | −2.611, −1.277, −2.125, +0.051 | −5.991, −1.450, −1.111, −0.916 | −1.841, −5.204, −3.245, −1.870 | −0.516, +0.337, −3.640, −2.776 |
| flower 3200×2160 | −0.814, +3.337, −0.920, −0.821 | +1.546, −0.828, +0.620, −1.934 | −5.802, +1.772, −1.770, +5.322 | +4.942, −4.102, −0.435, −2.674 |
| keong 3839×2159 | −2.134, −0.825, −0.464, −4.190 | +2.361, −0.821, −2.544, −7.021 | −4.499, −8.598, −1.181, −1.347 | −0.216, +3.565, +1.417, −1.755 |
| 4K | +0.344, +2.415, −2.916, +1.358 | +1.345, −0.767, +0.917, +2.343 | +2.087, −2.814, +0.782, −2.525 | +1.035, −6.369, −1.589, −1.461 |

All full-corpus control outer percentages are preserved in the same order:

| Control case/build | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| 65 `p2` baseline | +4.116, +5.195, −0.489, +0.293 | −0.298, +3.594, −2.563, +1.847 | +1.203, +0.041, −0.563, −1.749 | +0.191, −0.899, +4.320, +0.038 |
| 65 `p2` candidate | +0.452, +2.223, −0.508, +0.486 | +0.935, −0.282, −0.239, −2.275 | −0.364, +4.086, +5.762, −0.256 | +2.536, +1.915, +3.725, −3.409 |
| 513 `p0` baseline | −2.118, −0.538, +0.545, +0.705 | +2.414, +0.808, −0.356, +1.990 | −3.381, −0.625, +1.869, −0.257 | +0.746, −1.273, +0.887, +1.150 |
| 513 `p0` candidate | −0.890, +1.020, −0.796, +0.414 | +2.753, +0.409, +0.032, +0.825 | +6.462, +2.165, −1.026, −4.280 | +0.810, +0.321, +1.437, −1.580 |
| 513 `p2` baseline | −5.667, +0.845, +8.017, +1.504 | −2.309, +3.735, +1.002, −0.573 | −0.730, −3.427, −0.014, +2.264 | −1.255, −3.320, −4.097, −2.539 |
| 513 `p2` candidate | +0.731, −4.348, +0.330, +1.514 | −5.256, −4.436, +4.061, +4.742 | −0.371, −4.585, +2.255, −2.975 | +3.806, +4.633, −1.544, +8.107 |
| 1025 `p2` baseline | +5.045, −3.284, −1.479, +3.789 | +2.358, +3.272, +4.179, −3.192 | −3.004, +2.272, −2.026, +2.219 | −3.124, −3.511, −1.604, +4.664 |
| 1025 `p2` candidate | −0.353, +0.612, +5.486, +0.775 | +8.679, −1.034, +0.298, +1.360 | −3.233, −0.134, +0.385, +8.937 | +4.303, +1.367, +1.903, −0.271 |
| 4K baseline | −0.460, −1.616, +2.112, −1.564 | +1.251, +0.600, +0.240, −1.233 | −2.490, +1.242, −3.955, −0.207 | +3.213, −1.665, +0.195, −3.108 |
| 4K candidate | −0.611, −0.247, +1.574, −2.023 | −4.141, −0.882, −2.180, +2.610 | −0.954, −3.209, +1.930, −0.778 | +1.635, +3.188, −2.822, −0.841 |

The original pilot's retained 4K/eight slowdown is not consistent in this
campaign: three of four primary trials improve. Retained 4K section wall and
token-worker time improve in all eight auto/eight primary trials. However,
whole 4K is slower in six of eight, up to +2.415%, and its outer cross-label
comparisons are only 12/32 faster. That result requires explicit attention;
it is not erased by favorable dense cases or nonzero controls.

For whole 4K/auto/R0/O1, outer time is +6.302 ms while quantization-pipeline
time is +9.368 ms, section wall +0.229 ms, and token-worker time −0.732 ms.
For whole 4K/eight/R1/O1, outer is +6.582 ms despite codestream −0.218 ms
and token worker −1.230 ms; its quantization paired median is −1.709 ms
while all four cross-label quantization medians are slower. Different
median aggregations can disagree, and overlapping phases do not add.
These measurements do not establish the cause of the whole-encode slowdown.

The worst real retained trial, flower 3200/auto/R1/O1, is +5.322% (+2.700 ms)
outer, with entropy optimization +2.963 ms despite section −0.043 ms and
token worker −0.624 ms. All four outer cross-label comparisons are slower.
This limitation is retained in the report, not attributed away as noise.

### Focused 4K follow-up

Because full-corpus whole 4K regressed in six of eight trials, a separate
24-job exploratory campaign tests the unchanged standard-build binaries on
that exact input. It crosses whole/retained, auto/eight, both creation orders,
baseline/candidate and both same-build controls, in one seeded shuffle. It
uses four warmup and sixteen measured duplicate-ABBA rounds per job—twice
the measured rounds of the original campaigns. Its protocol pins the original
analysis hash and does not pool results, subtract controls, or introduce a
case-specific runtime policy. No build or sanitizer overlaps it.

All 24 jobs completed successfully from 2026-09-09 23:03:30 to 23:09:42 UTC,
without retries or dropped jobs: 1,536 measured calls, 384 warmups, 24 oracle
calls, 24 untimed whole captures, and 3,840 NVML observations of a 40 W limit.
The limit does not establish constant clocks or exclusive machine use.

All primary paired percentages follow; negative means faster. These are the
same within-round paired-median definitions as above, not confidence intervals.

| Mode/budget/order | Outer | Codestream wall | Entropy optimization wall | AC section wall | Token-writing worker sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole auto O0 | +0.868 | −2.552 | −1.432 | −6.014 | −7.266 |
| Whole auto O1 | +0.924 | −0.993 | −0.633 | +1.553 | −3.440 |
| Whole 8 O0 | −1.134 | −1.746 | +3.704 | +1.339 | −2.617 |
| Whole 8 O1 | +1.246 | +5.396 | +5.786 | +2.751 | −0.115 |
| Retained auto O0 | +3.344 | +3.365 | +3.655 | −3.573 | −4.926 |
| Retained auto O1 | +1.726 | +0.737 | +3.099 | −2.266 | −4.394 |
| Retained 8 O0 | −0.742 | −0.970 | +2.216 | −1.666 | −4.234 |
| Retained 8 O1 | +1.314 | +1.104 | +2.533 | −0.500 | −4.998 |

Both-build control outer percentages, in O0/O1 order:

| Build | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| Baseline | +0.484, +1.650 | +0.076, +2.867 | −0.909, +1.520 | −1.964, −1.099 |
| Candidate | +2.750, +0.562 | −1.278, +0.914 | −0.216, −0.615 | −2.100, +0.908 |

Whole and retained outer improve in only one of four trials each; their
cross-label comparisons improve in 6/16 and 5/16 respectively. All four
cross-label outer comparisons regress for whole/eight/O1 and retained/auto/O1.
The follow-up therefore does not resolve the 4K end-to-end concern. Token
worker sums improve in all eight trials (14/16 whole and 16/16 retained
cross-label comparisons), but section wall improves in only one of four
whole trials versus all four retained trials. These local improvements do
not establish a whole-encode gain.

For whole/eight/O1, outer is +3.610 ms, codestream +3.082 ms, entropy
optimization +1.075 ms, section +0.274 ms, and token worker −0.032 ms. For
retained/auto/O0, outer is +1.768 ms, entropy optimization +0.727 ms, section
−0.420 ms, and token worker −1.378 ms. The latter has three of four outer
cross-label comparisons slower. These separately aggregated, overlapping
phases are not additive; they do not establish a causal explanation for the
regressions. Controls remain sensitivity evidence, not a correction.

### Retention decision and remaining work

Retain the bounded ANS-emission batching change and its permanent regression
test. The full-corpus dense trials are consistently faster in both modes,
both budgets, both repeats, and every cross-label comparison; most real-image
outer trials also improve. Exact output, logical bit counts, failure atomicity,
normal tests, ASAN suites, and CUDA checks pass. The decision accepts the
documented mixed real-image results, including the recurring whole-4K concern;
it is not a claim that every input improves or that 4K is regression-free.
No image-dependent dispatch, compatibility path, or new public option is added.

`verify_s167_final.py` successfully recomputes both standard-build timing
analyses and the frozen pilot analysis. It verifies the pilot's 150 journals
(one preserved launcher-marker rejection), all 417 accepted promotion journals
with no promotion rejections, 92 CTests, seven ASAN suites, 44 qualification
comparisons, two CUDA sanitizer jobs, unchanged native modules, source/binary
pins, timing schedules, and the recoverable relocation. The final freeze
archives the decision, report, source/helper inputs, logs, and recomputed
analyses; `--frozen` additionally checks the complete artifact inventory and
every recorded SHA-256. No failed timing job was retried or removed.

The next separate candidate is ordinary `BitWriter::Append` when the destination
is byte-aligned: preserve its checks, exact logical length, partial-byte zero
padding, and transactional behavior while avoiding per-byte bit writes. It
needs its own direct bitwise-oracle alignment/allotment matrix and paired
whole/retained evaluation. It is not implemented in S167. Compact coefficient
consumption and fused composition/reduction remain as retained by earlier
stages; the rejected S112 tile schedule is not reintroduced. The optimization
goal remains open.

### Storage constraint and recoverable relocation

C: had about 277 MB free when follow-up began and fell to about 2 MB by the
end of ASAN qualification. New stage artifacts, compiler temporaries, and
logs are on U:. No broad directory cleanup or deletion of frozen evidence
was authorized or performed. The cause of the overall C: space loss has not
been established.

The just-completed install-consumer CTest generated a disposable
`build-cuda-ninja/installed-consumer-test` tree. Its CMake driver explicitly
recreates that tree on each execution. After completion, that exact tree was
relocated to `s167-promotion/installed_consumer_outputs` on U:, retaining all
196 files (57,440,367 bytes) and verifying every SHA-256 before and after.
The source and destination were resolved and checked, and reparse points
were excluded before moving. The relocation report records both paths and
all file hashes; the output remains recoverable. Source files, active test
executables, and frozen predecessors were not moved. C: then had about 62 MB
free. This relocation occurred during correctness qualification, before any
follow-up timing campaign.

Pilot artifacts and exclusive journals are frozen under
`build-cuda-ninja/profiles/s167-artifacts`; standard-build qualification and
both later timing campaigns are under `U:/gjxl-cuda-diagnostics/s167-promotion`.
Preparation verified all 2,451 frozen S166 artifact files and archived 341
predecessor production inputs. The
protected untracked Markdown files remain unread and excluded. No privilege
or firewall blocker has been observed. The overall optimization goal is open.
