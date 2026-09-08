# Composition / AQ reduction fusion and DCT8 scheduling (S111)

Date: 2026-09-08. Starting revision: `bfc2e6f`, branch `feat/cuda`.
Windows 11, RTX 3060 Laptop (sm86), CUDA 11.8, MSVC 14.37, Release.

## Mechanism and boundaries

S110 fused composition with the first global-maximum pass but still wrote
the pixel map needed by adaptive quantization. S111 eliminates that
intermediate entirely inside the fully-resident Butteraugli policy.
Each transform's kernel composes its pixels, computes the original L16
block distance, and emits an anchor maximum. The existing NaN-aware maximum
reduction then reduces the much smaller anchor array before policy update.
The public prepared-map operation and non-policy evaluations still return
the complete pixel map. Tiny, expanded or single-scale comparisons have no
multiscale composition to eliminate and retain their original reductions.

The generic kernel preserves the prior 256-thread L16 accumulation and
tree. A DCT8 specialization instead schedules four independent warps per
128-thread CTA, one warp per transform. Only 64 pixels exist per full DCT8
anchor, so the prior eight-warp CTA had six warps without source pixels.
The specialization folds lanes i and i+32 before shuffle steps 16/8/4/2/1.
Explicit rounded multiplication and addition prevent contraction of the
last square and fold, preserving the old rounded sum. There is no fast-math,
size-dependent tuning threshold or runtime selector.

Inputs and anchors remain immutable. Invalid pixels set error bit 16, become
zero only for L16 accumulation, and propagate NaN into the score. Invalid
L16 results set bit 32 without writing the block result. Existing error bits
are preserved. The caller validates that anchor ranges partition the anchor
array and their nonoverlapping rectangles cover every logical source pixel.
The kernel defensively flags invalid anchor origins with bit 8, but this is
not a general validator for malformed transform metadata.

The internal reduction sink distinguishes host batch descriptors from device
anchors and requires disjoint input, output, error and maximum storage.
Anchor maxima reuse the reconstruction-coefficient buffer after all inverse
transforms have consumed it on the same stream. The next evaluation rewrites
coefficients before another inverse transform. Final AC group packing
overwrites this scratch only after policy evaluation, and compact conversion
consumes that newly packed data. Quantized source coefficients are untouched.
No allocation, capacity increase, readback or synchronization is added.

Ignoring smaller anchor/reduction traffic, logical per-pixel traffic falls
from 20N to 8N bytes: two composition reads remain, while a map write and its
two reduction reads disappear. At 3839x2159 that saves 99,460,812 logical
bytes per comparison, about 198.9 MB across the two ordinary d1.2/e7
comparisons. These are logical operations, not measured DRAM transactions.
The pixel-distance allocation remains available as main-scale scratch;
this change does not claim a memory-capacity reduction.

## Test-harness ordering correction

The first two prototype checks failed at different cases: 1x1/strategy2/
pattern9 score, then 7x3/strategy1/pattern3 flags. Their logs and v1 source
are preserved and are not successful qualification runs. The shared S110
test helper uploaded pageable memory on the default stream, then launched
work on a non-blocking stream without explicitly finishing the upload.
CUDA documents that pageable H2D copies may return after staging before DMA
finishes. The helper now synchronizes the default stream after upload,
outside all timed regions. See NVIDIA's
[CUDA 11.8 synchronization rules](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/api-sync-behavior.html).

No kernel math changed for this repair. The corrected prototype passed
3,840 differential comparisons twice, and the production-linked probe
passed another 3,840. Shared test utilities now contain the independent
old composition/maximum oracle and the corrected upload helper; S110's
permanent test is requalified as well.

## Correctness and compact interoperability

The new permanent `cuda_aq_compose_reduction` test passes 2,148 comparisons.
Its reference materializes composition/maximum with independent old kernels
and calls the original production AQ reducer; its candidate calls the new
production kernel. It checks bitwise full block maps and finite scores,
NaN classification, exact error flags, immutable main/sub/anchor data and
allocation/row guards. Two candidate runs verify reuse and preservation of
a seeded error word. An empty batch with null pointers is a no-op.

Coverage includes six small/odd geometries, all seven supported transform
shapes and mixed nonoverlapping tilings, two layouts with distinct padded
strides, random/exponent-range values, mixed and all-negative signed zero,
NaN, infinity, negative pixels, composition overflow, L16 overflow and
subnormal inputs. Finite input is restored after exceptional values on the
same allocation. Additional release cases cover 500-square, odd HD and 4K.
The 2,112 focused comparisons pass all four Compute Sanitizer tools.
The corrected composition/maximum test passes 480 release comparisons and
384 comparisons under each sanitizer. All sanitizer runs report zero errors;
racecheck reports zero hazards/warnings and memcheck reports zero leaks.
The diagnostic three-schedule prototype also passes all four tools.

Dense qualification repeats S108's frozen byte and complete-summary oracles:
480 exact encode checks across 40 successful jobs, including effort/distance/
final-score settings, byte/error searches, high-range inputs and independent/
public-driver batches. Seven extreme inputs are rejected with identical
errors. A compact-enabled resident object linked against the same libraries
repeats all 480 checks and seven rejections. This is a scoped diagnostic
compile with `GJXL_CUDA_COMPACT_AC=1`, not a second full compact CMake build;
that definition is consumed only in the resident implementation.
Compact storage remains default OFF, and its native consumers and two-pass
producer are unchanged from S107–S109.

Four dense integrated memchecks cover seven prepared-reference cases,
independent/public-driver quality batches and rate search, contributing
36 more exact encode checks. Four additional compact memchecks pass a
public-driver quality batch and all three actual int32 fallback fixtures
(4096 checker/ramp and 16777216 ramp), contributing 21 exact checks with
zero errors/leaks. This specifically exercises the reused reconstruction
scratch before narrow conversion and full-width overflow fallback.
These byte/summary checks establish deterministic parity with the frozen
CUDA oracle, not a new independent decoder or cross-device quality campaign.

## Isolated schedule experiment

Thirty-six serial processes cover three geometries, DCT8/DCT32-preferred/
mixed tilings, two padded layouts and two repetitions. Each uses five warm
and 30 measured position-balanced randomized rounds, five labels and eight
complete chains per event sample. Labels 0/4 duplicate the separate path;
1 is fused256, 2 fused four-warps, 3 fused eight-warps. All 5,400 measured
rows and 900 warm rows are retained, including noisy duplicate controls.
Error reset is included equally. These are synthetic maps/tilings, and event
spans can include host submission gaps.

At odd 4K, representative per-comparison medians are:

| Tiling | Separate ms | Fused four-warps ms |
|---|---:|---:|
| DCT8 | 2.968–3.093 | 0.428–0.447 |
| DCT32 preferred | 1.640–1.722 | 0.547–0.565 |
| Mixed | 1.983–2.007 | 1.177–1.181 |

Every candidate paired median against both baseline labels is favorable.
Four-warps is approximately 1.7–3.2% faster than eight-warps on the HD/4K
DCT8 fixtures. Other tilings use the same generic kernel. The selected
four-warp kernel uses 27 registers, no shared memory, stack or spills in the
diagnostic sm86 build; generic256 uses 28 registers and 2,048 shared bytes,
with no stack/spills. This supports a schedule choice on this device, not
a universal occupancy or complete-encode speedup claim.

## In-encode stage versus complete-call timing

The uninstrumented within-process sweep uses labels 0/2 for separate and
1/3 for fused, with diagnostic-only branch counters. Five inputs run twice,
reverse case order, one reference plus four warm and 24 measured shuffled
Williams-design rounds. This provides 1,130 exact checks and 960 measured
encodes. Every encode verifies the expected two comparison branches.
Distance is 1.2, effort 7, CPU threads automatic and final score OFF.
Complete-call candidate paired medians have mixed signs on every input.
Quantization timing is consistently favorable for Flower and Keong 2000,
but not every other input. No samples or outliers are discarded.

A second diagnostic records CUDA events immediately after subscale
difference and after composition/global maximum/AQ reduction, before policy
update. Both routes share that scope. Event allocation is outside the
campaign loops, durations are read after encoding, and exactly two event
pairs are required per encode. Another 1,130 exact checks / 960 measured
encodes confirm the saving on actual hot encoder maps and transform layouts.
The following are ranges across eight correlated paired medians per input;
stage time is the sum of both comparisons, not one comparison:

| Input | Stage saving per encode | Stage percentage change |
|---|---:|---:|
| Flower 500 | 0.034–0.064 ms | -28.4% to -17.8% |
| Keong 500 | 0.020–0.040 ms | -18.1% to -9.8% |
| Keong 2000 | 1.50–1.67 ms | -70.1% to -66.4% |
| Odd HD | 0.32–0.34 ms | -59.6% to -57.6% |
| Odd 4K | 3.43–3.78 ms | -67.8% to -65.2% |

All 40 candidate stage paired medians are favorable. At 4K the separate
stage medians are 5.30–5.60 ms versus 1.77–1.86 ms fused. This is strong
evidence of a local stage reduction. It does not convert the mixed
uninstrumented whole-call results into an established general speedup.

## Production-linked complete calls

Both executables use the same S108 harness object; the baseline links S110's
final restored libraries, while the candidate links S111's integrated
libraries. Six repetitions alternate executable order and reverse case
order. Five single-image cases use two warm/eight measured loops; HD/4K
two-image batches use two warm/four measured loops, with two CPU threads
per request. Independent backends and the public driver are both included.
All 108 jobs pass, with 1,332 exact encode checks and 864 measured encodes.

The table shows medians of process medians; paired percentages are separately
computed medians of six pairwise percentages, not ratios of table entries.

| Input / mode | Separate ms | Fused ms | Paired change | Faster pairs / 6 |
|---|---:|---:|---:|---:|
| Flower 500 | 20.512 | 19.913 | -2.02% | 4 |
| Keong 500 | 21.045 | 20.174 | -5.34% | 5 |
| Keong 2000 | 148.152 | 146.998 | -1.29% | 4 |
| Odd HD | 77.577 | 76.666 | +0.67% | 3 |
| Odd 4K | 324.642 | 327.334 | +1.92% | 1 |
| HD x2, independent | 134.739 | 140.816 | +8.16% | 1 |
| HD x2, public driver | 132.601 | 130.929 | -1.08% | 4 |
| 4K x2, independent | 759.925 | 751.164 | -0.70% | 3 |
| 4K x2, public driver | 756.357 | 745.004 | -0.51% | 4 |

The 4K single and HD independent cohorts are slower in five of six pairs.
These are important counterevidence, not observations discarded as noise.
Flower also has a +55.7% repetition. Whole-encode variability remains
larger than many expected stage savings; its cause is not established here.
A targeted same-process concurrent comparison follows below.

## Targeted same-process batches

To investigate the HD independent regression, a single diagnostic executable
uses the same two-request lifecycle as the production harness with both
routes linked in. Each job retains its backends or public driver across
labels. Atomic branch counters verify four comparisons per batch; the mode
is set only between fully joined calls. No CUDA events are added.
Two repetitions reverse case order, with four warm rounds and 12 measured
HD or eight measured 4K rounds of four position-balanced shuffled labels.
The eight jobs pass 912 exact checks and contain 640 measured encodes.

| Two-image batch | Rep 0 paired range | Rep 1 paired range |
|---|---:|---:|
| HD, independent | -5.04% to -0.80% | -3.89% to +1.82% |
| HD, public driver | -4.46% to -1.93% | -1.51% to +2.00% |
| 4K, independent | -3.94% to -0.73% | +0.13% to +1.01% |
| 4K, public driver | -3.51% to +0.36% | -4.35% to -0.56% |

The earlier +8.16% HD independent aggregate does not reproduce as a
consistent same-process regression. This does not establish its cause or
prove absence of regressions. All four second-repetition 4K independent
comparisons are slower, and duplicate controls range as widely as +5.49%.
The complete-call conclusion remains mixed; the added sweep is not grounds
for claiming a broad throughput improvement or deleting the earlier data.

## Disposition

Retain the composition/L16/maximum fusion in the resident policy, with the
four-warp DCT8 schedule. Unlike the smaller S110 experiment, the actual
in-encode event scope now establishes a substantial local improvement:
all 40 candidate paired medians are favorable, including 3.43–3.78 ms
saved per 4K encode. This disposition is explicitly based on the qualified
GPU-stage result, not an established general whole-encode speedup.
The complete-call counterevidence above remains a qualification limit.
No new runtime toggle or compatibility layer is introduced.

The full study contains 5,521 exact encode checks, including 57 under
integrated memcheck, and 3,424 measured encodes. Fourteen matching extreme
rejections and the two failed v1 prototype checks are separate from that
success count. Compact consumption is qualified alongside fusion but remains
opt-in; direct narrow packing and CPU tile scheduling retain their earlier
dispositions. The new scheduling change here is GPU warp-per-DCT8 work
assignment, not a claim that S106's CPU tile scheduler has been promoted.

The next useful scheduling work is S106's CPU strategy-merge candidate:
qualify concurrent-request thread budgets and separate merge-phase savings
from complete-call effects. For further GPU work, prioritize larger live
dataflow costs identified in S105 rather than repeating S93's rejected
erosion-tiling experiments. The encoder is not established to be maxed out.

## Evidence and reproduction

Evidence root: `U:/gjxl-cuda-diagnostics/s111`; diagnostic sources, build/
campaign scripts and analyzers: `build-cuda-ninja/profiles/s111_*`.
Named output files are exclusive; use a new root for a fresh campaign.
The retained 40-file production runtime is not rebuilt or overwritten.
No firewall/admin prompt or permission blocker has appeared, and no security,
driver, clock, power or priority settings were changed.

Timing windows (UTC, September 8): isolated 05:17:59.857–05:18:48.040;
within-process 05:30:33.586–05:32:52.908; production-linked
05:34:07.453–05:39:00.380; in-encode scope 05:43:36.685–05:45:59.564.
Targeted same-process batches ran 05:52:48.366–05:56:13.291.
No task build, sanitizer or other GPU job overlaps those timing windows.
Only intended concurrent requests within each batch overlap.
Light editing and ordinary shared-machine state are limitations.

The validator recomputes every analysis, verifies input/oracle/executable/
log hashes and full summary records, retains failed-probe provenance, checks
the serial task-GPU timeline and build exclusion from timing windows, and
checks the unchanged 40-file retained runtime plus S110 baseline binaries.
The manifest freezes sources, tests, scripts, v1/v2 snapshots, linked probes,
libraries, test binaries, input/spec/oracle files and all reports.

```powershell
python build-cuda-ninja/profiles/s111_validate.py --frozen
```

The final integrated Release build passes **78/78 CTest tests** in
**173.21 seconds**, including both permanent fusion tests, prepared
Butteraugli and the install-consumer check. All 294 task GPU jobs are
serial except their intentional in-batch requests. The frozen validator
checks the same final source and artifacts; user scratch files were neither
read nor staged.
