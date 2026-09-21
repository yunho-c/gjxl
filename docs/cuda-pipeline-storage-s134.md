# Deferred resident host-field storage (S134)

Encoding-only resident preparation no longer allocates three unused host
initial-quantization arrays. This removes 34,214,400 bytes (32.63 MiB) of live
vector capacity at padded 4K, three allocations and their zero-initialization.
Controlled input-preparation savings are 5.9–6.6 ms at 4K. All sixteen
whole-call paired medians favor the candidate, but duplicate controls and
later-phase variation do not establish a precise universal throughput gain.
The change is retained for the exact storage reduction and repeatable
preparation improvement. The backend is not established to be maxed out.

## Implementation and ownership

Starting revision: `96459a0`, branch `feat/cuda`, September 8, 2026.
[S133](cuda-resident-reprofile-s133.md) traced 6.4–7.0 ms of padded 4K GPU-idle
time to resident pipeline preparation. Source inspection showed that
`initial_quant`, `strategy_mask` and `pixel_mask` were allocated and zeroed
even when downstream strategy search consumed the evaluator's device fields.

`PrepareResidentQuantizationPipeline` now leaves those vectors empty. It
still validates pixel/block area overflow and prepares EPF sharpness.
`PrepareResidentAcStrategyInputs` checks the existing encoding-initialization
capability before constructing host output views. The normal Butteraugli
encoding-only route needs no host arrays. A genuine host-output request,
maximum-error initialization, or evaluator without that capability calls the
new internal `PrepareHostInitialStorage` method before requesting readback.

The method validates geometry and retains correctly sized storage without
allocating or changing its contents. Otherwise, it constructs three local
zero-initialized vectors and swaps all three into the owner only after every
allocation succeeds. Allocation failure preserves the previous pointers and
contents. Prepared reuse, including encoding-only runs followed by host
materialization and subsequent reuse, remains supported. CPU/host-prepared
paths remain eager. There is no compatibility layer, runtime experiment
selector, persistent host cache, or change to compact's default.

For a fresh encoding-only resident preparation, removed storage is
`sizeof(float) * (padded_pixels + 2 * base_blocks)`:

| Input | Padded extent | Removed host vector bytes |
| --- | --- | ---: |
| Flower 500 | 504 × 504 | 1,047,816 |
| HD | 1920 × 1080 | 8,553,600 |
| 4K | 3840 × 2160 | 34,214,400 |
| Flower 2000 | 2000 × 2000 | 16,500,000 |

These are live vector-capacity savings, not measurements of process RSS,
allocator-reserved memory, or VRAM. Host materialization still incurs the
required allocations on demand. Device allocations, transfers, numerical
operations and coefficient ownership are unchanged by the source patch.

## Controlled complete-encode experiment

Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37, Release. A clean
production build disables compact by default; a diagnostic resident-owner
object also exercises the existing compact route. The inputs and frozen
oracles are those of S133: Flower 500, HD from 1919 × 1079, 4K from
3839 × 2159, and Flower 2000. Flower 2000 is fourfold nearest-neighbor
replication of the 500-square crop, not an independent native photograph.
Distance 1.2, effort 7 and fully resident encoding are fixed.

A diagnostic translation unit contains separately named, exact copies of
the original and candidate preparation functions. A small selector chooses
between them; all other code is shared, including the new on-demand host
materialization branch. That branch is not taken in these encoding-only
Butteraugli measurements. The functions have distinct linked addresses.
There are no differing class definitions across translation units.

Labels 0/2 are duplicate baselines; 1/3 are duplicate candidates. Randomized
four-label Williams sequences balance positions and immediate predecessors
within each four-round block. Each process performs one reference encode,
four warmup rounds and sixteen measured rounds: 81 checked encodes, of which
64 are measured. Four inputs, two widths and two repeats give sixteen timed
processes and 1,024 measured encodes. The second repeat reverses image/width
order. Sixteen Release/scoped-ASAN preflights add eighty checks. Each encode
checks frozen codestream bytes, summary equality, coefficient width, selector
call counts and the exact baseline/candidate host capacity.

Below, each delta is the median across rounds of the mean of labels 1/3
minus the mean of labels 0/2. It is not a difference between aggregate
medians. Negative means faster. Duplicate deltas are baseline 2−0 and
candidate 3−1, using the same per-round median rule.

| Input / width / repeat | Baseline whole ms | Whole delta ms | Input delta ms | Whole duplicate deltas ms |
| --- | ---: | ---: | ---: | ---: |
| Flower 500 / wide / 0 | 21.675 | −0.298 | −0.171 | +0.413 / −0.580 |
| Flower 500 / wide / 1 | 21.254 | −0.437 | −0.189 | +0.009 / −0.040 |
| Flower 500 / compact / 0 | 21.060 | −0.239 | −0.121 | −0.380 / +0.176 |
| Flower 500 / compact / 1 | 20.894 | −0.162 | −0.155 | −0.275 / +0.149 |
| HD / wide / 0 | 79.222 | −2.198 | −1.245 | +1.413 / +1.080 |
| HD / wide / 1 | 83.843 | −2.788 | −1.343 | +0.803 / +1.062 |
| HD / compact / 0 | 72.181 | −2.104 | −1.167 | +2.468 / −2.229 |
| HD / compact / 1 | 84.006 | −4.944 | −1.543 | +1.967 / +0.378 |
| 4K / wide / 0 | 325.083 | −1.234 | −5.885 | +4.296 / −4.096 |
| 4K / wide / 1 | 349.985 | −9.969 | −6.436 | −5.491 / −19.724 |
| 4K / compact / 0 | 322.741 | −7.379 | −6.178 | −4.928 / −9.210 |
| 4K / compact / 1 | 334.272 | −22.560 | −6.594 | +5.448 / +3.485 |
| Flower 2000 / wide / 0 | 147.881 | −3.550 | −2.702 | +3.554 / −5.133 |
| Flower 2000 / wide / 1 | 147.692 | −4.301 | −2.595 | +6.096 / −0.843 |
| Flower 2000 / compact / 0 | 145.028 | −4.805 | −2.645 | +0.974 / +1.037 |
| Flower 2000 / compact / 1 | 140.992 | −1.374 | −2.765 | −0.302 / +2.007 |

All 64 candidate/baseline label-pair input medians favor the candidate, and
input duplicate deltas are substantially smaller than the preparation gain.
Whole-call direction is favorable in all sixteen primary comparisons and
58 of 64 label-pair medians. Its magnitude is less certain: for example,
4K wide repeat 0 has a +8.03 ms quantization-phase delta offsetting much of
the preparation gain; 4K compact repeat 1 gains another 8.25 ms in
quantization and 6.86 ms in serialization, beyond the removed preparation
work. Phase medians need not sum to whole-call medians. Do not attribute
those downstream differences to the removed allocations, or promote the
largest observed whole-call percentage as a dependable speedup.

All 2,752 recorded power-limit endpoints are 40 W. Endpoints do not prove
constant clocks or absence of transient changes. No other recorded job
overlaps a measured process; lightweight read-only analysis/tool activity
did occur during some measurements, so this is not a claim of an otherwise
idle machine. No power, clock, firewall or privilege setting was changed,
and no firewall/elevation blocker was encountered.

## Correctness and native-code audit

The clean CUDA build passes all 81 CTests without skips. The production
campaign has eighty successful qualification configurations and fourteen
writer rejections that exactly match the prior frozen expectations. Sixteen
scoped host-ASAN and four CUDA memcheck production jobs also pass. Together
with the within-process experiment, this is 2,606 frozen-oracle encodes,
including 262 scoped host-ASAN encodes. These check historical codestream
oracles; this study does not claim a new independent decoder run.

The new portable `quantization_storage` test covers three geometries with
empty and previously populated owners, each of the three allocation failure
points, atomic preservation, successful retry, zero initialization,
allocation-free ready reuse, invalid geometry and area overflow. CTest and
standalone host ASAN exercise 36 injected allocation failures in total.

The permanent CUDA AQ test additionally compares lazy and explicit eager
storage with smooth and noisy 257 × 17 sources padded to 264 × 24. It covers
Butteraugli and maximum-error control, changing targets, repeated encoding,
transition to host outputs and subsequent pointer-stable reuse. It checks
serialized frames, score histories, maximum-error results, padded host
field/mask outputs, and source-extent reconstructed RGB. Across CTest, full
AQ ASAN, isolated Release/ASAN, and isolated CUDA memcheck/initcheck, 108
lazy/eager pairs (216 serialized frames) pass. This is separate from the
frozen-oracle population. Host ASAN instruments the changed codec/GPU
pipeline translation units, diagnostic caller and resident owner, not every
linked library. Six qualified CUDA sanitizer jobs report zero errors; the
five memcheck jobs also report zero leaked allocations.

No CUDA source changed, but the clean rebuild did not reproduce every
historical native body bit-for-bit. Nine of ten GPU modules match S133.
In the remaining AC-strategy module, `FinalizeCostKernel` is identical;
`PrepareQuantNormsKernel` differs. The entire PTX difference is moving one
zero initialization before a branch; its dependent arithmetic is unchanged.
The corresponding SASS scheduling changes retain the same resources
(35 registers, no local/stack/shared storage). Two further fresh compiles
reproduce the new module hash. All fourteen audited current executables
share the same ten current GPU modules, so both timing labels use identical
GPU code. The accurate historical comparison is **213 bit-identical kernel
bodies and one source-unchanged code-generation variant**, not 214 unchanged
bodies. The underlying reason for the historical code-generation difference
is not established.

## Evidence and continuation

Evidence is retained in `U:/gjxl-cuda-diagnostics/s134`; reproduction drivers
are `build-cuda-ninja/profiles/s134_*`. Reproduction requires a new artifact
root, the pinned inputs/oracles and the recorded toolchain. The unchanged
compact/scoped-ASAN resident-owner objects come from frozen S132. Main
results are `within_analysis.json`, `timing_summary.json`, production
campaign JSONs, `linked_v4.json`, `codegen_v3.json`,
`codegen_resources.json`, source snapshots and the artifact SHA-256 manifest.

The validator accounts for 164 recorded jobs: 146 qualified successes,
fourteen matching writer rejections, two excluded native-audit attempts,
and two incomplete sanitizer captures. The first strict native audit exposed
the historical module difference; the second failed a map parser that
incorrectly required MSVC's function flag in clang-linked maps. Neither is
a production correctness failure. The first two isolated sanitizer logs
reported zero errors but omitted buffered test completion output. They are
preserved and excluded from the qualified pair/sanitizer counts. A new
diagnostic copy differing only by explicit stdout flushing was rebuilt,
audited and run under both tools; both full completion records pass. Earlier
PTX-inspection assertions are also retained; they are inspection attempts,
not qualified encode jobs. No failed or incomplete evidence was overwritten.

Validation checks input/log/executable hashes, exact old/new function
isolation, label balancing, lack of measured-job overlap, frozen output
counts, native modules and forty retained runtime hashes. Historical
oracles, runtime binaries and the three protected scratch files are
untouched.

Next investigate resident AC strategy candidate construction and cost
scattering/representation, guided by S133's host-idle attribution. S112's
parallel merge improved a local stage without a dependable whole-workflow
gain; it remains rejected. Compact consumption, further composition/reduction
fusion and template-specific GPU work remain separate candidates, requiring
whole-encode qualification rather than local-kernel timing alone.
