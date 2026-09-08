# Resident filter storage lifetimes (S132)

The resident CUDA evaluator now needs at most one dedicated filter image.
It reuses the inverse-DCT image after its first filter read instead of
retaining two scratch images. The default padded 4K workflow removes
99,532,800 bytes (94.92 MiB) from its live persistent arena. Kernel code,
filter arithmetic, coefficient-storage defaults and public APIs are unchanged.

## Lifetime plan

Preparation counts materialized XYB outputs, not logical filters. Fused
Gaborish / EPF pass 1 counts as one write, and a final perceptual EPF writes
RGB directly without an XYB materialization. `FuseGaborishEpf` supplies the
same eligibility decision to preparation and dispatch.

The first XYB output goes to the one scratch image. Further XYB writes
alternate between the inverse-DCT image and scratch. Input and output never
alias within a filter launch. All work stays on the existing stream, so
the preceding reader finishes before that storage is overwritten. Final
maximum-error XYB ownership follows the parity of physical XYB writes;
zero writes retain the inverse-DCT image.

| Metric | Gaborish | EPF iterations | Dedicated scratch images |
| --- | --- | --- | --- |
| Either | Off | 0 | 0 |
| Perceptual | Off | 1 | 0 |
| Perceptual | Off | 2-3 | 1 |
| Maximum error | Off | 1-3 | 1 |
| Either | On | 0-3 | 1 |

The previous allocation was `min(2, Gaborish + EPF iterations)`. Every
previous two-image profile now needs one, and the single perceptual EPF
without Gaborish needs none. Other profiles have unchanged storage.
Both arena planning and view allocation use the same physical-write count.

The inverse-DCT image already had a temporary lifetime: resident frontend
inverse Gaborish and forward transforms use it before reconstruction.
Every evaluated reconstruction rewrites the complete coding image before
filtering. Original/coding inputs, forward coefficients and final AC/DC
storage are separate and remain untouched by the new aliasing. Scale
reconfiguration cannot change filter options, so the prepared storage plan
remains valid. Coefficient-only finalization and optional RGB readback retain
their existing paths.

This reduces live arena capacity, not the number of device allocation
calls. It does not claim that the CUDA pool immediately releases the same
number of reserved bytes; the private pool can retain freed allocations.
The change adds no lazy allocation or steady-state allocation.

## Verification

A clean CUDA 11.8 / MSVC 14.37 `sm_86` Release build passes all 80 CTests.
The extended permanent `cuda_aq` test checks all sixteen combinations of
metric, Gaborish and zero through three EPF iterations. For each profile it
checks the exact persistent-byte difference from its unfiltered reference,
evaluates changed quant fields in A/B/A order without device allocations,
and requires the repeated RGB and block-map bits, score, quantizer and
maximum-error reduction to agree. An injected completion failure must
preserve those caller outputs and invalidate the evaluator without a second
submission.

The same checks pass in the ASAN AQ test and in a separate noisy-input
fixture under Release, ASAN, CUDA memcheck, racecheck, initcheck and
synccheck. Together these cover 128 profile executions, 384 successful
evaluations and 128 injected completion failures. All four CUDA sanitizer
modes are clean; memcheck reports zero leaked allocations.

Release and ASAN 128-profile record streams match the frozen S131/S127
reference byte-for-byte. Each includes eight successful serialized profiles
and 120 matching writer rejections, not 128 successful codestreams. The
reference SHA-256 remains
`ad4680547ccd84a66668ddd2ea64727037adf86b19d3fd2b4c863e66c885315a`.
These records cover smooth/noisy images, custom filter weights and scales,
both metrics, every filter count and zero/two AQ updates.

The complete-workflow campaign covers 94 Release cases (including fourteen
matching expected writer rejections), sixteen host-ASAN cases and four CUDA
memcheck cases. Wide and opt-in compact outputs match frozen bytes and
summaries through photographic, padded HD/4K, target search, extreme-input,
serial-batch and concurrent-batch workflows. These populations provide
1,230 exact encodes, 222 under host ASAN. All 214 GPU kernel bodies remain
unchanged: all ten linked GPU modules in twelve qualified executables match
S131 exactly. The two excluded preliminary timing executables are also
retained and GPU-module audited, but supply no performance evidence.

## Complete-encode timing

The final harness compiles the original S131 owner and current owner as
distinct nested classes inside the existing friend class. Their bodies are
source-identical apart from type names. The backend header stays unchanged;
link maps verify distinct preparation symbols and virtual tables. A factory
selector chooses an owner before preparation and records its memory stats.
No selector, extra class, or measurement hook is in production.

Two baseline and two candidate labels use a randomized four-label Williams
schedule. Four warm rounds precede sixteen measured rounds, balancing label
positions and within-round predecessor pairs. Two process repeats reverse
image order. Every encode checks frozen bytes, summary, coefficient width
and ownership, exactly one selected resident preparation, the expected
persistent-byte saving, and unchanged staging/scratch statistics.

| Input | Baseline persistent bytes | Current persistent bytes | Saved bytes |
| --- | ---: | ---: | ---: |
| Flower 500 by 500 | 12,361,280 | 9,313,088 | 3,048,192 |
| Padded HD (1919 by 1079 source) | 100,909,566 | 76,026,366 | 24,883,200 |
| Padded 4K (3839 by 2159 source) | 403,563,256 | 304,030,456 | 99,532,800 |

The following are paired medians across sixteen rounds, averaging duplicate
labels within each round. Negative changes favor the smaller storage plan.

| Input / repeat | Baseline encode ms | Change ms | Change % | Duplicate baseline / candidate differences ms |
| --- | ---: | ---: | ---: | ---: |
| Flower / 0 | 19.298 | -0.425 | -2.20 | +0.081 / -0.031 |
| Flower / 1 | 18.789 | -0.240 | -1.27 | +0.327 / -0.061 |
| HD / 0 | 71.995 | -1.250 | -1.76 | +0.198 / +0.828 |
| HD / 1 | 78.343 | +0.143 | +0.18 | +1.896 / -3.337 |
| 4K / 0 | 289.201 | -2.497 | -0.88 | -6.188 / +7.529 |
| 4K / 1 | 298.609 | +0.516 | +0.17 | -4.476 / -10.226 |

HD and 4K change sign between repeats, with differences smaller than control
variation. Even Flower's total-time changes exceed its quantization-phase
changes (-0.084 and -0.107 ms). These runs do not establish a dependable
whole-encode speedup or a repeatable regression. Retain the change for its
guaranteed live-storage reduction, without attributing noisy total-time
differences to memory layout or cache behavior.

Six measured jobs supply 486 checked encodes. Separate Release/ASAN
preflights supply thirty more, fifteen under ASAN. Measured jobs overlap no
other recorded job. All 1,032 power-limit endpoints are 40 W without an
endpoint change; this is not proof of constant clocks or unchanged limits
between endpoints.

The first timing preflight failed its memory assertion because separately
compiled old/new implementations shared a class name: the linker could
coalesce incompatible methods. That harness and its partial output are
preserved and excluded. A simple distinct-class rename then failed to
compile because it lost friend access. The final nested-class harness
resolves both issues without production header changes and passes all
preflights. The failed preflight, its parent campaign, and the failed
compile are the three excluded harness jobs. No production test or
correctness campaign failed, and no firewall/elevation blocker occurred.

## Audit and next work

Evidence lives in `U:/gjxl-cuda-diagnostics/s132`; reproduction drivers are
under `build-cuda-ninja/profiles/s132_*`. Reproduce into a new artifact root
without overwriting historical binaries or frozen references. Source
snapshots and a SHA-256 manifest retain the implementation and evidence.
No power, clock, firewall or privilege setting was changed.

The final validator checks 159 recorded jobs: 142 accepted, fourteen
matching expected writer rejections and the three excluded harness jobs.
Totals are 80 CTests, 1,746 frozen-oracle encodes (237 under host ASAN),
256 profile-record comparisons, 384 successful lifetime evaluations, 128
injected atomic failures and eight clean CUDA sanitizer jobs. It also checks
the original and corrected linked modules, source isolation, input hashes,
memory sizes, balanced schedules, power records and forty retained runtime
hashes. No historical oracle or runtime binary is overwritten.

The backend is not established to be maxed out. The next investigation is
a fresh complete-resident critical-path profile after the filter fusion
and storage changes, to rank remaining GPU work against host preparation
and serialization costs.
