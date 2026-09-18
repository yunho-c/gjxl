# Native sparse coefficient consumption in production (S160)

## Scope and implementation

S160 brings the qualified S156–S159 native sparse representation into the main
source tree, without the experimental mode selectors, fault hooks, or diagnostic
compatibility layers. The direct comparison baseline is the actual S152
production source, archived and rebuilt before editing the main tree. The
starting commit is `c66932d` (S159's report; production code was still S152).
All 2,379 frozen S159 artifacts were verified before preparation and are checked
again by the final verifier. No predecessor source, binary, helper, or report
is changed or rebuilt.

Frames can privately own dense or sparse int8/int16/int32 AC coefficients.
Sparse storage consists of one 64-bit mask and one 32-bit payload offset per
64 logical coefficients, plus ranked nonzero values. Payload intervals may
arrive in any GPU tile order. Assembly validates shape, tail masks, bounds,
non-overlap, complete coverage, and values before publishing the owner.
Sparse group offsets describe only active coefficients, without dense edge
group padding. Native views support logical indexing, iteration, subranges,
and mask-based nonzero counting; there is no dense expansion cache or implicit
contiguous `data()` operation.

Quantization/reconstruction, coefficient-order consumers, and the AC serializer
consume the native representation. `GetAcGroup` remains the explicit dense
int32 accessor; callers that accept all storage types use `GetNativeAcGroup`.
The Metal consumer receives the corresponding native-view adaptation, but this
Windows stage does not build or test Metal.

CUDA selection requires at least `3 * 256 * 256` active coefficients, a total
fitting `uint32_t`, and a supported packer grid. Only eligible final evaluations
move existing full coefficient-order population readback into the earlier
quantizer/policy batch. The first 5,952 full-population zero bins yield an exact
nonzero count; the 192 sampled duplicate bins are excluded. No extra counting
kernel, allocation, or submission is introduced. Readiness and count reset on
each evaluation, including calls without a final frame.

Sparse output is selected at no more than one nonzero per eight active slots.
The packer reuses existing coefficient workspace and lazily allocates only the
mask/offset/count device header. A single sparse pack submission is explicitly
completed before one fixed eight-descriptor readback batch copies headers,
exactly sized values, and frame metadata. A zero-length value copy is skipped
by the backend. The packed count must equal the independently available full
population count. The existing production backend already drains queued copies
on an enqueue error; no S157 diagnostic fault code is copied into it.

There is no speculative packing on dense inputs, no post-pack density decision,
no second payload readback, and no production policy selector. At the accepted
density, even one-byte values need at most 31.25% of the corresponding active
dense coefficient bytes, so the old diagnostic post-pack 80% gate is redundant.
Small wide inputs preserve their original up-front host allocation. Sparse
header bytes are published atomically for concurrent memory-stat queries, and
remain accounted for after the prepared object returns to dense output.

Sparse consumption is enabled in the production candidate. The existing
`GJXL_CUDA_COMPACT_AC` compile-time width option remains OFF by default; ordinary
final evaluation still uses int32. Compact ON receives separate correctness
qualification, not a default-width change. No tile-scheduling or thread-budget
default changes are bundled here. S152's qualified scheduling and earlier fused
composition/reduction work remain intact; S159 did not establish a separate
fusion-only or CPU-budget win.

## Qualification

All 130 recorded job journals are accounted for: 126 accepted and the four
retained unsuccessful attempts described below. No qualification failure is
silently reclassified as a successful run.

The final normal build passes all 89 CTests, including the install consumer.
Five host-ASAN fixtures cover sparse coefficient views, frame assembly,
allocation failure, resident CUDA evaluation, and the public workflow. All
C/C++ translation units in that build are host-ASAN instrumented; CUDA units
and their launch wrappers are compiled by nvcc with MSVC, not ASAN.

The sparse coefficient fixture covers 405 cases and 335,205 subranges, rejects
3,849 invalid inputs, and includes nine near-`UINT32_MAX` empty-shape guards.
The final ceiling-division arithmetic avoids `n + 63` overflow on 32-bit
`size_t`; no 32-bit build is run here. Sparse frame coverage comprises 384
cases, 1,536 exact codestream comparisons, 1,488 atomic assembly failures, and
384 bitwise reconstructions. Allocation testing injects 168 failures across
12 cases and checks 48 allocation-free queries.

The production resident fixture uses 24 cases: five shapes from 1x1 through
513x519 and a mixed-transform frame, each at four amplitudes. Two prepared
contexts per case traverse quantization fields 0.8, 0.0001, 16, and 0.8 while
alternating scored policy, unscored policy, ordinary final evaluation, and
no-final calls. It makes 672 evaluated-output comparisons against 384 control
evaluations per run. Checks include bitwise output arrays, exact codestreams,
native coefficients, independently recounted full populations, cached versus
uncached coefficient orders, dense materialization in test code only, poisoned
outputs, lazy header reuse, and a concurrent memory-stat reader. Counters are
150 sparse calls, 426 dense calls, and 24 changes between sparse and dense output.

Both wide and compact resident fixtures pass all-kernel CUDA memcheck with full
leak checking and initcheck, four jobs. The compact build additionally passes
both CUDA AC-strategy reference tests, 18 direct whole-encoding correctness
comparisons (six real images and three synthetic patterns under automatic/eight
CPU budgets), and whole-GPU-workflow memcheck/initcheck on the 500x500 real
flower image. Those two sanitizer jobs execute five encodes each and verify
the frozen real-image codestream. All six accepted CUDA checks report zero
errors; all three memchecks report zero bytes leaked in zero allocations.
The 18 unsanitized compact comparisons execute 90 encodes, including their
oracles, and are correctness evidence only.

The portable S157 Compute Sanitizer 2025.2.1 is reused without installation,
suppressions, blocking-launch flags, or system changes. This stage does not
repeat S158's packer race/synchronization fault checks: those remain frozen
predecessor evidence, not additional S160 runs.

## Native-code audit and retained unsuccessful attempts

Native extraction covers the baseline DLL (10 CUDA modules), final production
DLL, normal resident executable, compact resident/workflow executables, and
ASAN resident/workflow executables (11 modules each). The extra compact DLL
has exactly the compact resident executable's 11 raw module hashes. All 13
pre-existing CUDA source files are unchanged from S152. The new sparse packer
source text is unchanged from S159.

Every module's resources match S159 after normalizing only the two source-derived
eight-hex anonymous-namespace IDs. All production and ASAN instructions also
match. The compact rebuild has one instruction-level difference in
`PrepareQuantNormsKernel`, despite unchanged source; its resources remain
35 registers, no stack, and no shared memory. This is a genuine instruction
difference, not merely a symbol change, and its exact cause is not established.
It motivated the extra compact strategy and full-GPU-workflow qualification.
The sparse packer's instructions and resources match in every audited build.

Four rejected job journals are preserved rather than overwritten:

- The first baseline preflight expected an explicit compact-OFF cache entry,
  absent from the old cache. It stopped before compilation. The corrected
  configure/build pinned OFF and rebuilt the unchanged S152 source.
- The first comparison-harness build applied `/TP` to the link input as well,
  treating `nvml.lib` as C++ source. Separate compilation/linking succeeded.
- The initial CTest checker expected 87 tests; all 88 then-existing tests
  actually passed with exit zero. A separate validator checked all 88 rows.
  The added resident test brings both subsequent full runs to 89 passing tests.
- The first extra compact workflow memcheck used a generic CPU-only fixture.
  Its assertions passed, but Compute Sanitizer rejected it because no CUDA API
  was called. The replacement uses the actual full GPU encoding harness and
  passes both memcheck and initcheck. This is not a firewall/admin failure.

The first strict native audit stopped on the compact instruction difference;
its script and partial extraction are retained alongside the completed audit.
Initial, V2, V3, and final source snapshots and prior DLL/library versions are
preserved. The first two source-index JSON files contain duplicate descriptors
for some absolute/relative paths; their unique archives are identical. The
authoritative final snapshot contains 339 unique inputs with no duplicates.

## Predeclared direct-production timing method

The baseline and candidate are independently linked DLLs loaded in one process,
using the same MSVC 14.37 runtime and identical pinned public summary header.
Private frames/backends/results are allocated and destroyed within their own
module. The diagnostic bridge is not an installed or production API.

Every call executes the whole fully resident effort-7/no-final-score workflow
with a fresh prepared context, retaining one backend per implementation. Input
creation, result destruction, oracle checks, and NVML queries are outside the
timed `EncodeFrame` call. One baseline oracle runs before measurement; every
real-image oracle must also match its frozen S153 codestream. Every warmup and
measured call compares complete bytes and the public encoding summary.

There are 36 specifications: six real images under automatic/eight CPU budgets,
65/513/1025-wide synthetic images at three amplitude/target settings under both
budgets, and baseline-versus-baseline controls for real 4K and small/1025 dense
synthetics. Two shuffled/reversed campaigns give 72 jobs, each with four warmup
and eight measured rotating duplicate-ABBA rounds. The second campaign also
reverses DLL loading and backend creation order; that change is confounded with
repeat, not an independent factorial test. Null controls use distinct backend
states in the same baseline DLL.

The primary statistic is the median of eight within-round differences between
duplicate-label means. Percentage changes use the corresponding round means.
Four cross-label median comparisons are sensitivity checks, not confidence
intervals. Negative is faster. Total and phase medians are computed separately
and need not add; near zero, percentage and millisecond medians can differ in
sign. Primary timing uses compact OFF in both implementations; all logged
coefficient widths must be four bytes.

Sources, binaries, helpers, matrix, order, and analyzer are pinned before the
first sample. No build or sanitizer job overlaps this campaign. The enforced
GPU power limit is recorded before/after each interval; it does not establish
constant actual power, clocks, temperature, or load. No power, clock, thermal,
affinity, priority, security, or firewall settings are changed. RDP is not
assumed to explain variation.

## Results and production decision

The campaign ran from 17:16:25 through 17:23:16 UTC on 2026-09-09. All 3,456
warmup/measured calls matched their 72 baseline oracle evaluations, and all
6,912 enforced-limit observations were 40,000 mW. The machine is the RTX 3060
Laptop GPU, compiled for sm86 with CUDA 11.8 and MSVC 14.37. The primary
analyzer and final verifier both pass.

### Real-image whole encodes

Each cell gives the two repeats' primary total-time changes versus production
S152 (percent; negative is faster):

| Real input | Automatic CPU budget | Eight-thread budget |
| --- | ---: | ---: |
| Flower 500x500 | -5.880, -5.726 | -4.516, -3.654 |
| Padded 1080p | -13.501, -13.043 | -11.987, -14.774 |
| Flower 2000x2000 | -10.267, -12.043 | -8.561, -9.964 |
| Padded 4K | -10.130, -11.113 | -8.307, -10.027 |
| Flower 3200x2160 | -13.805, -12.275 | -10.373, -13.143 |
| Keong 3839x2159 | -4.291, -10.761 | -5.884, -12.367 |

All 24 primary comparisons and all 96 cross-label comparisons favor the
candidate. The real-image improvement range is 3.654–14.774%, not a universal
speedup claim or a confidence interval. Quantization-pipeline time improves
in every real-image primary comparison. For padded 4K, whole-encode savings
are 26.05–34.68 ms and quantization-pipeline savings are 21.88–28.57 ms.
Native AC ownership falls from 106,168,320 to 5,779,732 bytes, about 94.6% less;
the additional GPU header is 4,665,604 bytes. These figures are AC ownership,
not total process or GPU memory.

Serialization is not uniformly faster. Flower 500 serialization is slower in
all four comparisons (+1.503% to +6.482%, about 0.13–0.57 ms), and flower 2000
at eight threads is +4.489%/+6.361%. Whole encoding still improves in each case.
The direct result supports reduced transfer/allocation work as useful, but does
not establish every native consumer as individually optimal.

### Synthetic density controls and small inputs

Synthetic p0 uses amplitude 0.01 and target 1.2; p1 uses amplitude 1 and target
1.2; p2 uses amplitude 1 and target 0.01, with the same fixed seed. The 513x519
and 1025x1031 p0 cases are extremely sparse and improve 15.638–18.514% and
13.626–15.572%, respectively, across the two budgets/repeats. All their
cross-label comparisons also improve. Their native owners shrink from
7,077,888 to 152,192 bytes and from 19,660,800 to 599,240 bytes.

Both larger p1/p2 inputs retain identical dense widths and owner sizes in the
baseline and candidate. They expose a cost that the real-image results do not:

| Dense synthetic | Automatic total change | Eight-thread total change |
| --- | ---: | ---: |
| 513x519 p1 | -1.744, +2.059 | +1.485, +3.730 |
| 513x519 p2 | -0.547, +0.916 | +4.226, +5.945 |
| 1025x1031 p1 | +1.139, -0.614 | +0.619, +1.468 |
| 1025x1031 p2 | +0.469, -2.425 | +3.187, +1.397 |

In particular, 513x519 p2 at eight threads is slower in both repeats and all
eight cross-label comparisons. Its whole-encode cost is 1.83/2.60 ms;
serialization is +5.060%/+7.309% (1.52/2.27 ms), while quantization-pipeline
changes are -0.004/+0.209 ms. Across all four larger dense/eight-thread cases,
serialization is slower in both repeats. This is evidence of a dense-consumer
performance concern, not grounds to declare the population movement free or
attribute the difference solely to GPU work, RDP, or noise.

The 65x71 inputs do not select sparse storage. Their primary changes span
-1.049% to +3.615%. The low-amplitude case is slower in all four primary
comparisons (+1.515% to +3.326%, 0.13–0.21 ms), although cross-label signs vary.
The small-input eligibility gate prevents sparse work but does not demonstrate
zero total overhead for the broader native-representation change.

### Null controls and interpretation

These compare two baseline states in the same baseline DLL, not different
implementations:

| Null control | Automatic total change | Eight-thread total change |
| --- | ---: | ---: |
| 65x71 p2 | +0.727, -2.665 | -1.392, +0.516 |
| 1025x1031 p2 | +1.500, -1.602 | -1.297, -8.442 |
| Real padded 4K | -0.750, -0.919 | -1.933, +1.459 |

Substantial variation therefore remains even without an implementation change;
it is especially visible in the second 1025/eight-thread null. Several
within-round statistics and cross-label medians disagree in sign. The
predeclared within-round statistic remains primary; neither an alternative
summary nor a subtraction of the null result is used to improve the candidate's
reported result. The null controls do not exonerate the repeatable dense
513/eight-thread regression.

The native sparse implementation is promoted as a real-workload throughput and
AC-memory improvement, with the dense/small-input costs explicitly retained as
a tradeoff. This is justified by all six real inputs improving under both
budgets and repeats, including every cross-label check, plus exact output and
memory-safety qualification. It is not a claim of a Pareto improvement across
all inputs. Compact width and CPU-budget defaults remain unchanged.

The next investigation should isolate dense serialization against the frozen
S152 DLL and this candidate, retaining the adverse 513/eight-thread fixture.
The expanded native dispatch and its generated CPU code are hypotheses to
test, not established causes. Separate serializer-only measurements and
code-generation/phase evidence are needed before changing dispatch or tile
scheduling. No additional production change is made after this campaign's
source pin.

## Reproduction

Local evidence is under `build-cuda-ninja/profiles/s160-artifacts`.
`s160_prepare.py`, the `s160_build*` helpers, and the versioned qualification
helpers preserve source/build pins and exclusive job journals. `s160_native.py`
records native-code differences. `s160_timing.py` writes its protocol before
execution; `analyze_s160.py` derives paired statistics from raw logs.
`verify_s160.py` checks predecessor hashes, archived/final sources, job outcomes,
qualification markers, native evidence, exact derived analysis, and timing
non-overlap. `freeze_s160.py` archives sources/helpers and inventories local
artifacts; `verify_s160.py --frozen` checks that inventory again.

The three protected untracked Markdown files are never inspected, inventoried,
edited, or staged. No remote push is made. This stage does not establish that
CUDA VarDCT encoding is maxed out.
