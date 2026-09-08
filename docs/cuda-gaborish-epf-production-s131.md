# Production Gaborish / first-EPF fusion (S131)

The qualified all-channel reuse path is enabled in production. The clean
integration improves the measured first-filter stage by 11.0-17.7%, with
all frozen-output and sanitizer gates passing. Complete-encode timing is
reported separately below; it is not represented as an equally large
end-to-end percentage gain.

## Implementation

This integrates [S130's all-channel shared-storage reuse](cuda-gaborish-epf-reuse-s130.md)
into the resident CUDA owner. Gaborish and EPF pass 1 produce filtered XYB
in one launch without a global Gaborish intermediate. The raw 38-by-38
three-channel shared window becomes a 36-by-36 filtered halo after every
raw reader finishes. Eighteen thread-local values bridge those lifetimes;
the kernel uses three block barriers, 46 registers per thread and 17,328
shared bytes, with zero reported stack or local-memory allocation.

`EpfFromTile` shares the existing EPF arithmetic between ordinary tiled
EPF, final EPF / RGB conversion and the new fusion. `GaborishValue`
evaluates the shared halo with the standalone Gaborish kernel's addition
order and nonfinite handling. The standalone global kernel retains its
original source body and native code. There is no second EPF arithmetic
implementation, compatibility wrapper, diagnostic switch, geometry
threshold or new public option in production.

The new internal `LaunchCudaAqGaborishEpf` accepts only EPF pass 1 and
matching extents. It validates active pitches, required pointers and the
one-dimensional launch-grid bound; matching empty extents are no-ops.
Gaborish output stride and EPF input stride are unused because their
intermediate is not materialized. Input, sigma and output must not alias,
and callers remain responsible for allocation sizes.

| Resident profile | First filter boundary |
| --- | --- |
| Gaborish + two EPF iterations, either metric | Fused Gaborish / pass 1 to XYB |
| Gaborish + one EPF iteration, maximum-error metric | Fused Gaborish / pass 1 to XYB |
| Gaborish + one EPF iteration, perceptual metric | Existing Gaborish then final EPF / RGB |
| Gaborish + three EPF iterations | Existing Gaborish then pass 0; no new fusion |
| No Gaborish or no EPF | Existing dispatch |

The fused result occupies scratch image 1, the same image as two separate
logical filter stages. The owner advances to logical stage 2 / next pass 2.
This preserves final XYB selection for maximum-error scoring, including
one-EPF evaluation. Existing final EPF / RGB fusion remains in place where
the metric only consumes RGB. Allocations, scratch counts, compact
coefficient defaults and the exact-coefficient path are unchanged.

## Native-code qualification

A clean Ninja Release build uses the pinned CUDA 11.8 / MSVC 14.37 toolchain
for `sm_86`, without fast math. The final filter module contains thirteen
kernels. All twelve prior S127 kernel bodies are byte-for-byte identical
after normalizing only the translation-unit symbol. Factoring EPF into a
device helper therefore does not change the existing EPF / color kernels.

The new fused body matches S130's winning pass-1-to-XYB body across all
2,744 instruction slots, registers, predicates and scheduling/control
words. The only differences are 62 constant-parameter relocations caused
by removing the unused 20-byte color parameter. The audit checks each
disassembled operand and the exact corresponding encoded-address delta,
not just opcode counts. Resource usage remains 46 registers, 17,328 shared
bytes, zero stack and zero local bytes; constant parameter space shrinks
from 536 to 516 bytes.

Twenty linked executables are audited. The two captured-input replay
executables contain only the new thirteen-kernel module. The baseline
matrix executable retains S127's ten original GPU modules. The other
seventeen current executables contain the new module and the other nine
S127 GPU modules unchanged: 214 kernel bodies in total. This includes the
actual clean-build AQ / EPF tests, wide and compact complete encoders,
ASAN variants and the separate timing executables.

The initial refactor also factored standalone Gaborish through the helper;
its compiler register allocation changed. That source/object and native
probe are preserved. The standalone body was restored before final
qualification, leaving only the intended new kernel addition. An initial
incremental test-build command also omitted the Visual Studio environment
and failed to locate the standard `array` header. Rebuilding through the
pinned `vcvars64.bat` environment fixed the invocation; no elevation or
firewall change was needed. Neither preliminary attempt was used for
timing or accepted as the final runtime.

## Permanent test and correctness

The new `cuda_gaborish_epf` CTest compares two-stage Gaborish / EPF against
the fused launcher on 24 geometries, two unequal-pitch layouts, sixteen
patterns, two channel-weight sets and three changed-input reuses. Two
paths per case give 9,216 pipeline executions. Guarded planes have
different per-channel prefixes and suffixes. The test checks exact output
bits, immutable inputs/sigma, error flags, all guards and the untouched
global intermediate on the fused path. It deliberately zeros the unused
intermediate pitches in fused calls.

Patterns cover tiny images, tile edges, very wide short images, signed
zeros, subnormals, infinities, NaNs, extreme magnitudes, mixed sigma bypass
rows, the exact bypass threshold and adjacent representable values.
Independent assertions require Gaborish's nonfinite flag rather than an
EPF or color flag for the designated nonfinite input cases. Eighteen
invalid-argument variants and matching empty extents must submit no
output writes and preserve the error seed.

All 80 clean-build CTests pass. The new test and existing EPF / color and AQ
tests also pass host ASAN. The 128-profile record matrix covers smooth and
noisy inputs, zero through three EPF iterations, Gaborish on/off, custom
filter parameters, perceptual / maximum-error evaluation and zero/two AQ
updates. Baseline, current Release and current ASAN records have identical
SHA-256 `ad4680547ccd84a66668ddd2ea64727037adf86b19d3fd2b4c863e66c885315a`,
also matching the previously frozen S127 record stream. This includes
eight successful serialized profiles and 120 matching writer rejections,
not 128 successfully encoded codestreams.

Six preserved S129 captures are replayed through the actual production
launcher, with both aligned and 19-float-offset layouts and repeated
buffer reuse. Release, ASAN and CUDA memcheck pass all 72 executions each.
The new permanent test's smaller 1-by-1, 33-by-33 and 65-by-17 population
passes CUDA memcheck, racecheck, initcheck and synccheck: 1,152 executions
per tool, no hazards/errors, and no memcheck leaks.

## Complete-workflow qualification and performance

The preserved S108 complete-workflow population passes: 94 Release cases
including fourteen matching expected writer rejections, sixteen additional
host-ASAN cases and four CUDA-memcheck cases. Successful outputs match
frozen bytes, summaries and coefficient ownership/width checks. This
covers wide and opt-in compact AC storage, photographic inputs through
2000 pixels, padded HD/4K, quality and target search, extreme inputs and
serial/concurrent batch workflows. These populations provide 1,230 exact
encodes, including 222 under ASAN. The existing EPF/color regression test
also passes all four CUDA sanitizer modes after the shared-helper refactor.

Timing uses one diagnostic resident owner with baseline and fused branches
only at the qualified boundary. The rest of that owner is exactly the
production source. Both branches use the audited clean-build GPU module;
the baseline Gaborish and EPF bodies are unchanged from S127. Every encode
checks frozen bytes, summary, coefficient storage and the expected two
selected filter-boundary calls. No diagnostic selector or event calls are
included in production.

Each process has two baseline and two candidate labels in a randomized,
four-label Williams schedule. Four warm rounds precede sixteen measured
rounds, balancing positions and distinct within-round predecessor pairs
over each four-round block. Two process replicates reverse image order.
A reference plus twenty four-label rounds gives 81 checked encodes per
measured process. Separate Release/ASAN preflights use five encodes each.
There are 1,032 frozen-output encodes in the two timing populations, of
which thirty are ASAN preflight encodes. Timed jobs overlap no other
recorded job.

The event-instrumented population measures the sum of the two first-filter
boundaries. Each row averages the duplicate labels within a round and
reports paired medians across the sixteen measured rounds. Negative
changes favor fusion.

| Image / replicate | Baseline stage ms | Fused change ms | Change % | Duplicate baseline / candidate differences ms |
| --- | ---: | ---: | ---: | ---: |
| Flower / 0 | 0.163328 | -0.028160 | -17.53 | +0.001024 / +0.001024 |
| Flower / 1 | 0.164864 | -0.021248 | -13.13 | -0.001024 / +0.005632 |
| HD / 0 | 0.996608 | -0.177152 | -17.67 | +0.008704 / +0.011776 |
| HD / 1 | 1.018112 | -0.161536 | -16.02 | -0.016384 / -0.014848 |
| 4K / 0 | 12.108032 | -1.440768 | -11.94 | +0.452608 / -0.090624 |
| 4K / 1 | 12.641024 | -1.385728 | -10.97 | -0.636928 / +0.402432 |

All 24 individual candidate-versus-baseline label-pair median differences
are negative too. This independently reproduces the local gain from S130
with the production kernel and only baseline/candidate branches.

The separate uninstrumented complete-encode population is:

| Image / replicate | Baseline ms | Fused change ms | Change % | Duplicate baseline / candidate differences ms |
| --- | ---: | ---: | ---: | ---: |
| Flower / 0 | 19.893550 | -0.157875 | -0.80 | +0.229800 / -0.372550 |
| Flower / 1 | 21.271400 | -0.129025 | -0.64 | +0.289700 / -0.087500 |
| HD / 0 | 79.780575 | -0.490925 | -0.64 | +0.822150 / +0.761150 |
| HD / 1 | 82.172450 | +1.107975 | +1.37 | +3.844550 / +0.580800 |
| 4K / 0 | 343.890975 | -13.013450 | -3.88 | +6.909150 / -5.151850 |
| 4K / 1 | 345.187800 | -11.398675 | -3.24 | +1.589750 / +4.929150 |

Both 4K repeats favor fusion, but the total-time changes substantially
exceed the roughly 1.4 ms filter-stage saving and there is sizable control
variation. Instrumented 4K total-time differences are mixed (+1.918 and
-3.113 ms), and HD's uninstrumented result changes sign. Do not attribute
the full 3-4% 4K difference exclusively to this kernel or claim a stable
whole-encode percentage from these short populations. The acceptance
evidence is the repeated local saving, preserved native code elsewhere,
frozen outputs and absence of a demonstrated repeatable overall regression.

All 2,064 power-limit endpoints, including preflights and warmups, are
40 W with no before/after endpoint changes. This does not prove constant
SM clocks or absence of intermediate limit changes. No clock, power,
firewall or privilege setting was changed, and no firewall/elevation
blocker was encountered.

## Decision, audit and next work

Retain the production pass-1-to-XYB fusion. Do not enable the rejected
channel-wise schedule, old raw-shared fusion, pass-0 fusion or the unmeasured
pass-1-to-RGB fusion. This is a measured resident-pipeline optimization;
compact coefficient storage remains independently opt-in.

The next independent allocation experiment is filter-scratch lifetime
planning. The default two-EPF perceptual path writes its fused XYB to
scratch image 1 and then writes RGB directly, leaving scratch image 0
unused. It is still allocated here to preserve stage/owner policy while
qualifying the kernel. Removing that allocation needs a separate audit of
maximum-error final XYB ownership, one/three-EPF profiles, failure behavior
and repeated prepared evaluations. The backend is not established to be
maxed out.

Evidence is in `U:/gjxl-cuda-diagnostics/s131`, with reproduction drivers
under `build-cuda-ninja/profiles/s131_*`. The final validator checks 177
recorded jobs: 162 accepted, fourteen matching writer rejections and the
corrected compiler-environment invocation. It verifies native operands and
encoded relocations, all twenty linked executable/module hashes, exact
source-change scope, input/oracle hashes, profile records, guarded replays,
sanitizer summaries, balanced timing schedules, paired medians, power
records, nonoverlapping measured jobs and forty retained runtime hashes.

Totals are 80 passing CTests, 2,262 frozen-oracle encodes (252 under host
ASAN), 384 AQ profile records / 256 comparisons, 24,192 new guarded filter
pipeline executions, 46,080 existing EPF/color pipeline executions, 216
captured-input replay executions and thirteen clean CUDA sanitizer jobs.
The AQ test supplies additional coverage without being converted into an
invented encode count. Frozen source snapshots and a SHA-256 artifact
manifest preserve the final runtime and evidence. Reproduce in a new
artifact root; do not overwrite these binaries or historical oracles.
