# CUDA parameter-weight integration (S148)

## Scope and retained initial policy

Starting commit: `bb3b26a`. This integration carries the S147 rolling
low/medium vertical convolution into the normal prepared Butteraugli path.
The explicit ownership mechanism and initial large-plane mixed policy are
retained after correctness and integrated stage qualification. Complete-encoder
timing remains noisy; the area boundary is not claimed to be optimal.

The prepared object owns 33 float taps (132 bytes), copied from the existing
host-generated Gaussian kernel before its normal device upload. The low/medium
plan also owns these values. Horizontal convolution and the separate-pass
oracle retain their device weight pointer; rolling vertical launches receive
the owned values by value. The internal contract explicitly requires the two
copies to match. There is no host-pointer lifetime dependency, global cache,
fixed Gaussian table, device-to-host fetch, extra device allocation, or
compatibility adapter.

The new bodies preserve the S147 64-row, three-plane shared ring and
24-output-row chunks. Tap FMAs consume constant kernel parameters; thread 0
computes the original ordered normalization on the GPU and publishes one
scalar in an unused ring slot through the existing barriers. Input addressing,
tap order, divisions, low/medium reconstruction, and output contracts remain
unchanged.

The initial geometry policy is deliberately explicit and independently
testable. Width below 32, height below 96, or area below 2,000,000 selects the
retained plain 48-row body. Remaining planes below 4,000,000 pixels use rolling
48; larger planes use rolling 96. Area multiplication is 64-bit. Tests can
force each of the three bodies without changing the production policy.

## Native code and correctness

The clean CUDA 11.8 / MSVC 14.37 / SM86 production build contains 223 GPU
bodies: all 221 S143 bodies are instruction-exact, and the two new bodies are
instruction-exact matches for the qualified S147 prototype. Each new body has
46 registers, 24,576 shared bytes, 1,984 static instructions, and zero local
memory, stack, or spills. The S147 occupancy and retired-work measurements
describe these same bodies; this integration does not repeat those counters.

Both the normal within-process harness and the event-instrumented harness
match all 223 production kernel bodies. Full encoders link ten CUDA modules;
the two release standalone tests link three, all belonging to that same set.
Explicit ASAN support objects pull the full ten modules. Each diagnostic
family changes only one module's identity and preserves the other nine.

All 84 CTests passed without skips. The permanent low/medium grid now runs
1,840 fixtures: 46 geometries, packed/padded layouts, five patterns, and four
schedules (production, forced plain, rolling 48, rolling 96). Both original
oracles, bitwise comparisons of sixteen arrays, padding guards, and three
reuse stages remain. Eight actual allocation-and-dispatch fixtures cover the
2M/4M boundaries and narrow/short exclusions. Two tall cases exercise flattened
launch geometry, including more than 65,535 horizontal tile rows.

A permanent ownership test captures two independent streams using local plan
copies, poisons both local and owned payloads before launch, checks all sixteen
arrays, then changes the device/owned taps and captures a second generation.
The full horizontal-plus-vertical graphs have eight nodes each. Every process
checks 72 graph executions, including differing and signed tap sets.

Release and ASAN checks preserve existing frozen JXL and summary oracles for
wide and compact coefficients, multiple distances/efforts, searches, and both
batch entry styles. ASAN instruments the changed prepared Butteraugli host
implementation, in addition to the resident/pipeline/search ownership code.
This step concerns convolution taps, not a new compact-AC storage format.

The full production campaigns check 1,296 frozen-oracle encodes: 960 release,
222 host-ASAN, 48 device-memcheck, and 66 device-initcheck. Fourteen expected
invalid-input rejections match the original error lines. Release and ASAN
each pass the 1,840 guarded fixtures, eight policy-boundary cases, two tall
cases, and 72 ownership graph executions. Sixteen CUDA sanitizer jobs pass:
all four tools on the 120-fixture scope and ownership suite, plus four
full-encoder memcheck and four initcheck configurations. Racecheck reports
zero errors and warnings; scoped racecheck takes 392.786 seconds and ownership
racecheck 55.556 seconds. No timing sample is taken concurrently with a
sanitizer job.

## Uninstrumented complete encodes

Four S145/current-S143 cases are measured in wide and compact modes, with two
reversed process orders. Each process has six balanced Williams labels:
plain (0/3), rolling 48 for all eligible planes (1/4), and the candidate mixed
policy (2/5). Six warm rounds precede eighteen measured rounds. Every encode
checks the frozen JXL oracle and exact summary/storage contracts. All labels
include the new host ownership, so these comparisons isolate scheduling but
cannot establish the cost of introducing ownership itself.

There are 16 release/ASAN preflights and 16 measured processes, 2,432 exact
encodes in total, and 4,864 enforced-power-limit endpoints, all 40 W. Endpoint
equality is not continuous monitoring or proof of a stable operating state.
No observation is discarded.

The table reports paired-round median percentage changes versus the two
plain labels; each cell contains the forward/reverse process-order results.
Negative is faster. “All 48” retains plain tiles on ineligible planes.

| Case | AC mode | All 48: outer % | Mixed: outer % | Mixed: quantization % |
| --- | --- | ---: | ---: | ---: |
| flower_500 | wide | +0.81 / +1.13 | +0.13 / -1.21 | +0.06 / -0.57 |
| flower_500 | compact | -0.54 / +0.43 | -0.72 / -1.61 | -0.49 / -1.30 |
| 1080p | wide | +3.26 / +2.86 | +4.69 / +1.53 | +3.04 / +0.84 |
| 1080p | compact | -0.55 / -1.42 | -1.40 / -3.10 | +0.17 / -0.85 |
| 4k | wide | -3.35 / -2.66 | -0.00 / +2.17 | +0.01 / -0.82 |
| 4k | compact | +1.14 / -2.65 | -1.97 / -1.36 | -2.15 / -1.64 |
| flower_2000 | wide | +2.72 / -1.34 | +4.03 / -2.06 | +1.53 / -1.30 |
| flower_2000 | compact | +0.50 / -0.41 | +0.50 / +0.39 | +1.18 / +0.89 |

All-48 wins 8/16 primary outer-time comparisons and 34/64 cross-label
comparisons; mixed wins 9/16 and 29/64. Median primary outer changes are
+0.008% and -0.360%, respectively. The largest within-process duplicate
outer difference is -15.824 ms (4K compact, first process order); a mixed
duplicate quantization difference reaches -11.327 ms (4K wide, first order).
Unchanged serialization also varies. These are not robust universal
whole-encoder gains, and identical-body flower_500 labels remain negative
controls rather than optimization successes.

## Separate frozen-versus-current executables

To include the ownership change, the frozen S143 wide/compact `s108_encode`
harnesses are compared with the same harness source linked to S148. Each
process checks a single frozen quality-oracle item: distance 1.2, effort 7,
final score enabled. That differs from the within-process campaign's
final-score-disabled workload; timings are not pooled between campaigns.

There are sixteen preflights and sixty-four timed processes. Each timed
process has one reference encode, two warm encodes, and eighteen measured
encodes. Four interleaved process labels provide two copies per executable
family, and the second campaign reverses case/mode order. All 1,392 encodes
match their frozen bytes and summaries. The 160 process-level power endpoints
are all 40 W; these endpoints do not observe every measured encode.

The table compares the two current process medians with the two frozen
process medians. It is a separate-process comparison, not paired-round data.

| Case | AC mode | S148 outer % | S148 quantization % |
| --- | --- | ---: | ---: |
| flower_500 | wide | +0.43 / -5.65 | +1.07 / -0.84 |
| flower_500 | compact | -12.22 / +10.67 | -10.27 / +6.73 |
| 1080p | wide | -1.59 / -2.75 | -0.71 / -0.64 |
| 1080p | compact | -1.35 / +1.15 | +0.30 / +1.20 |
| 4k | wide | -3.00 / -1.70 | -1.38 / -1.18 |
| 4k | compact | +0.60 / -0.81 | -0.14 / -0.66 |
| flower_2000 | wide | -1.13 / -1.04 | -1.93 / -2.00 |
| flower_2000 | compact | +0.13 / -1.69 | -1.55 / +0.04 |

S148 wins 11/16 primary outer and quantization comparisons, with 40/64 outer
and 41/64 quantization cross-label wins. Median primary changes are -1.239%
outer and -0.686% quantization. Large duplicate controls remain: 12.713 ms
outer (4K wide, first order), -7.021 ms quantization (4K compact, second
order), and -13.217 ms serialization (4K compact, first order). This suggests
a possible small gain without resolving a reliable end-to-end magnitude.

## Integrated stage events and retention

The separate event campaign checks another 2,432 exact encodes, records
29,184 GPU intervals (horizontal and vertical for each of six calls per
encode), 4,864 detailed telemetry records, and 4,864 enforced-power-limit
endpoints. Every telemetry query succeeds; every enforced-limit endpoint is
40 W. These observations are endpoints, not continuous monitoring.

Both eligible-plane policies improve the full-resolution vertical total,
all vertical intervals combined, and the horizontal-plus-vertical total in
all twelve large-case/mode/process-order comparisons. All 48 cross-label
comparisons for each of those three measures are favorable for each policy.
The unchanged horizontal total has only 24/48 favorable crosses for all-48
and 28/48 for mixed. This supports a local convolution improvement while
leaving the complete-encoder magnitude unresolved.

Each cell below gives the forward/reverse results. The full-vertical total
contains the three full-resolution vertical invocations in each encode.
The horizontal column is the unchanged six-invocation control, not a claimed
optimization. Negative percentages are faster.

| Case | AC mode | All 48: full vertical % | Mixed: full vertical % | Mixed: all vertical % | Mixed: horizontal control % |
| --- | --- | ---: | ---: | ---: | ---: |
| 1080p | wide | -8.62 / -8.80 | -8.65 / -8.88 | -6.94 / -7.06 | -1.73 / +0.14 |
| 1080p | compact | -9.39 / -9.10 | -9.54 / -9.03 | -7.59 / -6.97 | -1.55 / +0.86 |
| 4k | wide | -10.52 / -8.62 | -7.37 / -8.32 | -8.45 / -9.50 | +0.77 / -0.04 |
| 4k | compact | -8.57 / -9.57 | -9.70 / -9.83 | -8.68 / -9.97 | -0.76 / -0.08 |
| flower_2000 | wide | -11.24 / -11.09 | -8.43 / -8.94 | -6.46 / -6.83 | +1.48 / +2.15 |
| flower_2000 | compact | -13.26 / -11.68 | -10.21 / -8.93 | -7.72 / -6.91 | -2.23 / +1.48 |

The largest vertical duplicate difference is 0.259 ms, and the largest
horizontal-plus-vertical duplicate difference is 0.506 ms (both 4K compact,
first order). The unchanged 500-square control also varies: its half-size
vertical primary changes span roughly -1.84% to +12.19%. Those small-plane
results are controls, not gains attributable to this integration. Events
can include stream idle gaps and perturb scheduling; instrumented outer
timings are preserved separately and are not pooled with normal throughput.

Retain the explicit ownership mechanism and the fully qualified initial
large-plane mixed policy as a bounded improvement over S143. The selected
full-resolution vertical stages improve by 7.37–10.21% across the measured
large cases. Do not claim a universal whole-encoder speedup or an optimal
area boundary from these data.

The 4M switch is a concrete follow-up: on the 2000-square input, rolling 48
beats rolling 96 in all sixteen cross-label full-vertical comparisons,
by 0.035–0.123 ms per encode. The 4K preference varies by AC mode/order and
has larger duplicate/control shifts. A separately qualified policy that
keeps rolling 48 at 4M is the next scheduling experiment. This integration
does not conclude that CUDA VarDCT is maximally optimized.


## Reproduction and preserved failures

Evidence is under `build-cuda-ninja/profiles/s148-artifacts`. The clean build,
auxiliary builds, initial policy, input/expected-output hashes, normal and
instrumented sources, module extractions, process metadata, every timing
observation, and sanitizer logs are retained. Original S143 and S147 evidence
and the forty historical retained runtime hashes remain untouched.

The first linked-module audit incorrectly assumed three modules for every
executable; its failure is preserved. The corrected audit passed but its
outer runner then collided with the child's same-named JSON report. The
successful child report/log are preserved, with a separately named recovery
job verifying all hashes and explaining the missing original runner PID.
Neither failure was a product correctness failure. No administrator or
firewall blocker was observed, and no clock, power, thermal, affinity,
priority, or machine security setting was changed.

Final verification accounts for 290 terminal recorded jobs (275 accepted,
14 expected rejections, and the preserved first audit failure), plus the
separately documented runner-report collision. All 276 GPU jobs are serial.
The production, ordinary-within, external, and event campaigns together
check 7,552 frozen-oracle encodes, in addition to the permanent guard tests.
Run `python -X utf8 build-cuda-ninja/profiles/verify_s148.py --frozen` to
recheck the frozen evidence, source snapshots, native code, all timing
derivations, and the forty retained-runtime hashes.
