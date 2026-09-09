# S166: reuse validated AC entropy models within a section batch

## Retention decision

Retain the private validation-reuse change. It removes demonstrated duplicate
work without weakening public validation, changes no output bytes, and improves
section wall time in every larger-real-image primary trial in both modes.
Most real-image whole encodes improve, but not all. Dense synthetic results
remain mixed, including the focused follow-up; this is not a Pareto improvement
or a claim that encoder optimization is complete. No image-specific threshold
is introduced to select favorable benchmark cases.

## Change and safety boundary

S165 isolated repeated full entropy-model validation as a substantial cost for
the 4K workload. S166 changes only the private AC section-writing call site:
after `WriteSimpleAcGlobal` succeeds, ANS groups call the existing internal
ANS writer instead of the public wrapper that validates the whole model again.
Prefix groups continue through the public wrapper.

The global-header path calls `WriteEntropyCode(ac_code)`, including full model
validation, before any group worker starts. The model belongs to the local
encoding candidate, is referenced read-only throughout the batch, and cannot
be changed by those workers. All workers join before the candidate can be
changed or destroyed. This is validation reuse within one known lifetime, not
a cache keyed by address or a persistent validation flag on mutable data.

The internal writer still validates its output pointer and token view, checks
HybridUint configs and every token context, verifies symbol/frequency/lookup
bounds, and checks ANS state transitions. It still constructs a temporary
writer and appends only on success. The containing section batch is local and
is published only after all groups succeed. Public `WriteTokenStream` remains
unchanged and continues to validate caller-supplied mutable models.

The internal declaration now documents its already-validated-model precondition.
There is no new public API, ownership field, compatibility path, cache, GPU
kernel, coefficient layout, tile policy, or scheduler. The default sparse
coefficient policy and wide coefficient-width policy are unchanged.

## Focused regression coverage

The new `ac_section_validation` CTest compiles the private implementation into
its test translation unit instead of exposing a production test hook. It
compares the private batch with a reference assembled through the fully
validating public header and token writers, including exact logical bit counts
and padded section bytes.

Each execution covers:

- 192 valid combinations: prefix, balanced ANS, high-density ANS; interleaved
  and offset-split tokens; empty/populated streams; custom/natural order;
  mapped/unmapped contexts; automatic/one/two/eight participant budgets.
- 1,408 malformed-model cases, including dimensions, context mapping, configs,
  ANS alphabet width, missing histograms, reverse-map dimensions, reciprocal
  dimensions, and histogram method.
- 96 late invalid-context failures after earlier populated and empty groups.
  A failed batch must preserve the destination's original three bits and
  sentinel token count, while matching the public oracle's error category.
- 64 concurrent comparisons from four callers sharing a read-only model, with
  independent output and participant tracking. This is not a ThreadSanitizer
  run or exhaustive scheduling proof.

Every comparison also requires the entropy model to remain unchanged and an
explicit participant budget not to be exceeded. The direct normal run, full
CTest run, and ASAN run all pass this test.

The full configured suite passes 91/91 tests in 289.37 seconds. Optional system
`djxl`/`jxlinfo` conformance smoke and ImageMagick wrapper tests are unavailable
in this environment; no new decoder-conformance run is claimed. Frozen real
codestreams are used as byte oracles below.

Six host AddressSanitizer suites pass: AC section validation, entropy,
codestream encoder, public codestream workflow, compact frame, and sparse
frame. The compact suite reports 96 cases, 384 exact-stream comparisons,
480 atomic failures, and 96 bitwise reconstructions. The sparse suite reports
384 cases, 1,536 exact-stream comparisons, 1,488 atomic failures, and 384
bitwise reconstructions. These existing frame suites compare representations
under full and effort-7 sampled coefficient-order behavior; they are not an
independent full entropy-policy cross product. Encoder/entropy tests cover
their respective additional behavior paths.

## Audit the work actually removed

Separate instrumented DLLs reuse the frozen S165 CPU overlays with S166's
current serializer. They are used only for qualification, never for the
performance campaign. Twenty-eight comparisons cross seven cases with
whole/retained encoding and automatic/eight participants.

| Case | AC groups | ANS writes, unchanged | Token-stream full validations, before → after |
| --- | ---: | ---: | ---: |
| 65 `p2` | 1 | 3 | 3 → 2 |
| 513 `p0` | 9 | 11 | 11 → 2 |
| 513 `p1` | 9 | 12 | 12 → 3 |
| 513 `p2` | 9 | 12 | 12 → 3 |
| 1025 `p2` | 25 | 28 | 28 → 3 |
| flower 500 | 4 | 7 | 7 → 3 |
| 4K | 135 | 144 | 144 → 9 |

Every sample requires exactly one fewer full validation per AC group and no
change in ANS call count, token count, chunk count, reservation capacity,
emitted bits, aligned-append counts/bits, or prefix call/validation counts.
ANS failure counts remain zero. The remaining full validations are outside
the optimized AC group calls. Model validation during header writing remains
in place and is outside this token-stream audit counter.

Forty-four uninstrumented qualification comparisons additionally cover all
six established real images and five synthetic cases, in both modes/budgets.
Every encode must match the frozen S162 baseline's exact bytes and public
summary. Real-image baselines must also match their frozen codestream oracle.
Retained states compare logical coefficient fingerprints, counts and nonzero
counts, and their serializer output must match the captured whole encode,
without further GPU allocations or submissions.

CUDA memcheck with full leak checking and initcheck each pass whole flower
500/eight and 4K/eight comparisons: four sanitizer jobs, zero errors, and zero
memcheck leaks. All 11 raw CUDA modules in each of the four new DLLs match
frozen S162 modules byte-for-byte. Only CPU objects are rebuilt.

## Performance protocol

The predeclared campaign has 304 jobs: 176 baseline/candidate comparisons and
128 same-build, independent-state controls. The baseline is frozen S162;
both production candidates are uninstrumented. Whole and retained modes each
have 152 jobs. Primary cases are six real images plus synthetic 65 `p2`,
513 `p0`/`p1`/`p2`, and 1025 `p2`; synthetic heights are width plus six.
Controls cover 65 `p2`, sparse 513 `p0`, dense 1025 `p2`, and 4K for both builds.

Each case crosses automatic/eight participants, two repeats, and both
DLL/backend creation orders. Each repeat shuffles all jobs together. Four
warmup and eight measured rotating duplicate-ABBA rounds give two samples per
implementation per round. Primary statistics are medians of within-round
duplicate-label mean differences. Four cross-label median comparisons per
trial are sensitivity checks, not confidence intervals. Controls are reported
without subtraction. Small mixed effects are not treated as proven speedups.

Input setup, clearing/destruction of prior output, result queries, byte checks,
and NVML observations are outside outer timing. Retained capture/deep copy is
also outside timing; retained serialization makes no GPU submissions. Worker
profile sums overlap and are not additive section or encode wall time.
No build or sanitizer is permitted to overlap the timing campaign.

## Original campaign results

Across all primary trials, whole encode is faster in 62/88 comparisons
(240/352 cross-label comparisons); retained encode is faster in 65/88
(236/352 cross-label comparisons). Real-image whole encode is faster in 36/48,
with outcomes ranging from 5.24% faster to 4.37% slower. Real-image retained
encode is faster in 40/48, ranging from 10.98% faster to 3.79% slower.
This is not a universal end-to-end speedup.

The targeted worker-aggregate token-writing time improves in all 48 real-image
trials in each mode, including all 192 cross-label comparisons per mode. For
the five larger real images, section wall time improves in all 40 trials per
mode: 156/160 whole cross-label comparisons and 160/160 retained comparisons.
Their section-wall improvement ranges are:

| Case | Whole section wall faster | Retained section wall faster |
| --- | ---: | ---: |
| 1080p | 5.52–12.60% | 5.75–8.52% |
| flower 2000 | 5.26–19.28% | 6.10–11.14% |
| flower 3200×2160 | 8.16–23.48% | 10.14–16.96% |
| keong 3839×2159 | 12.90–21.88% | 10.09–21.60% |
| 4K | 19.49–31.13% | 16.52–24.74% |

For these larger images, token-writing worker sums improve 29.30–55.39% in
whole encodes and 26.23–50.04% in retained encodes. Those reductions must not
be presented as end-to-end speedups. Whole 4K/eight is faster in all four
primary trials; whole 4K/automatic is mixed despite faster section writing
in every trial. Small flower 500 effects are less distinct: section wall is
faster in 4/8 whole and 7/8 retained trials, while token worker sums improve
in all eight in each mode.

All primary outer-time results follow, in percent. Each cell lists
`R0/O0, R0/O1, R1/O0, R1/O1`; negative means faster. Ratios, millisecond
differences, and cross-label comparisons use distinct median aggregations and
can disagree in sign near zero or with uneven samples.

| Case | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| 65 `p2` | +2.770, −4.023, −5.033, +0.819 | −1.323, −3.465, −2.628, +0.708 | −2.627, −0.784, +0.549, +0.514 | +1.260, −2.275, −0.138, −4.430 |
| 513 `p0` | −1.582, −0.056, −0.005, +1.484 | −4.271, +1.064, +0.417, +1.492 | −0.682, −0.010, −0.484, −0.000 | −3.763, −0.425, +4.200, −1.349 |
| 513 `p1` | −0.189, −0.344, +0.367, −0.495 | −1.727, −0.561, −5.123, +1.084 | −2.292, +0.570, +3.245, −1.802 | −3.950, −5.586, −1.056, +1.087 |
| 513 `p2` | +0.244, −2.319, −3.738, −1.875 | +1.879, +0.094, +2.370, −4.175 | −3.035, +2.223, +0.404, −0.354 | −6.924, +1.324, −0.667, +8.805 |
| 1025 `p2` | −3.882, −4.862, −3.142, +3.354 | −1.121, −5.932, −2.838, −3.191 | +2.158, +0.124, −1.683, −4.470 | +1.653, +1.398, −3.668, −4.665 |
| flower 500 | −1.817, +3.418, −1.165, −1.359 | −1.562, −0.621, +0.648, +1.293 | −6.391, +0.466, −1.264, −1.137 | +0.922, +1.479, −0.973, −1.445 |
| 1080p | −2.456, −1.330, +1.185, −0.951 | −0.096, −1.138, −3.425, +0.433 | −1.266, −1.660, −2.068, −3.660 | −2.460, −1.104, −1.899, −1.314 |
| flower 2000 | +0.649, −2.088, −2.137, +4.367 | −2.381, +0.399, −4.122, −0.495 | −2.827, +0.790, −0.507, −0.804 | −3.820, −2.588, −2.701, −2.162 |
| flower 3200×2160 | −2.529, −2.167, +1.591, −2.346 | −0.312, −2.172, −0.986, +1.479 | +2.868, −5.108, −10.976, −3.156 | −0.759, +0.085, −2.084, −7.918 |
| keong 3839×2159 | −1.597, −1.004, −0.615, −1.626 | −3.768, −0.558, −2.289, −0.713 | −8.670, +3.789, −3.872, −6.793 | −4.694, −6.711, −3.875, −9.259 |
| 4K | +1.401, −0.390, −4.322, +1.899 | −2.026, −5.242, −0.259, −2.889 | −8.858, −6.819, −6.972, +0.721 | −4.788, −5.748, −1.768, −3.257 |

The controls are nonzero. Original 4K whole controls range −2.83% to +1.85%
outer, and retained controls −5.05% to +4.94%. Dense 1025/eight candidate
retained control reaches +11.04% outer and +13.88% section wall. These are
observed control outcomes, not error bars, and a different input's control
does not explain away the dense 513 outlier below.

For example, the worst whole real-image regression, flower 2000/auto/R1/O1,
is +5.286 ms outer while section wall improves 0.740 ms and token worker sums
improve 11.362 ms. Quantization is +2.560 ms and entropy optimization +2.534 ms.
Retained keong/auto/R0/O1 is +2.817 ms outer despite section wall −2.179 ms;
its DC tokenization, AC tokenization, and entropy phases are all slower.
Separate medians do not add. These observations locate the measured variation
but do not establish its cause or prove the candidate caused it. No image-
specific dispatch threshold is inferred from them.

## Dense outlier follow-up

The original retained 513 `p2`/eight/R1/O1 trial is +8.805% outer and +12.662%
section wall, with all four cross-label outer comparisons slower. Because the
original controls did not cover this exact dense input, an additional,
separately predeclared 48-job exploratory campaign tests the unchanged binaries
on 513 `p2`: both modes/budgets, baseline/candidate and both same-build controls,
two repeats and both creation orders. It uses four warmup and twelve measured
duplicate-ABBA rounds. Its results are not pooled with the original campaign.

The follow-up completes with all byte checks passing. Outer-time results,
again in `R0/O0, R0/O1, R1/O0, R1/O1` order, are:

| Mode / comparison | Automatic % | Eight participants % |
| --- | --- | --- |
| Whole / baseline → candidate | +2.417, +0.972, +0.583, +1.163 | +2.101, −3.352, +0.120, −2.446 |
| Whole / baseline control | −1.783, −3.238, −4.571, +1.981 | +0.682, −3.697, −0.911, +1.447 |
| Whole / candidate control | −0.192, −1.970, +3.251, −2.294 | +1.752, −4.362, −2.361, −1.961 |
| Retained / baseline → candidate | −4.490, −1.191, +2.224, +0.163 | +6.029, +1.177, +0.369, −4.277 |
| Retained / baseline control | −2.922, −2.157, +2.859, −0.947 | +3.238, −3.870, −2.438, −2.093 |
| Retained / candidate control | −6.704, +5.066, −4.677, +11.672 | +0.038, −0.756, +3.333, +2.422 |

Only 2/8 follow-up whole primary trials and 3/8 retained primary trials are
faster; their cross-label comparisons are 12/32 and 15/32 faster respectively.
The original +8.805% does not recur at that magnitude, but retained/eight still
regresses in three of four trials, up to +6.029%. Candidate same-build controls
also vary materially, including +11.672% retained/automatic; that is a different
budget and is not an explanation or correction for the eight-participant
regression. The exact dense input does not show a stable benefit, and this
follow-up does not prove that the candidate is harmless for it.

The retention decision rests on the validated lifetime boundary, demonstrated
work removal, consistent larger-real-image section gains, and majority real
whole-encode gains, with this unresolved dense performance limitation retained
as evidence. S165 found much greater dense cost in bit emission than model
validation. The next optimization should investigate that path directly,
including fewer per-chunk writer calls and aligned append, while continuing
to measure dense cases and controls. It should not infer a dispatch threshold
from these noisy comparisons.

## Reproducibility and build recovery

Artifacts are under `build-cuda-ninja/profiles/s166-artifacts`. Preparation
verifies all 1,327 frozen S165 artifact files and archives the 340 predecessor
production inputs. The candidate adds one test source. Current normal libraries,
seven normal test executables, ASAN executables, four bridges, source snapshots,
protocols, journals, and raw logs are retained. Predecessor binaries and frozen
sources are never rebuilt or overwritten.

Three rejected journals are preserved, all attributable to diagnostic/build
helpers rather than a failed codec test:

- CTest returned zero with all 91 tests passing, but the initial acceptance
  string expected a different summary format.
- The first separate CTest validator did not allow the extra padding before
  single-digit test numbers. Its replacement checks all 91 unique passed names
  against CTest's JSON inventory and the successful raw summary.
- The initial bridge helper built `whole.dll`, then concatenated two source
  names because PowerShell had unwrapped a one-element array to a string.
  A typed-array recovery builds only the missing bridges, preserving the
  already-built whole bridge and failed artifacts.

No production-source correction was needed after the first successful build.
The exact encoder edit, source/library hashes, qualification counters, native
modules, journals, and timing statistics are checked independently before
freezing and committing. No system power, clock, thermal, affinity, priority,
firewall, or security setting is changed, and no privilege blocker was found.
The protected untracked Markdown files remain unread and excluded.

The original timing campaign completed on 2026-09-09 from 21:14:35 to 21:37:26
UTC: 9,728 measured calls, 4,864 warmups, 304 oracle calls, and 304 untimed
whole captures for retained inputs. The separate dense follow-up ran
21:40:34–21:42:58 UTC, adding 2,304 measured calls, 768 warmups, 48 oracle calls,
and 48 retained-input whole captures. Qualification adds 304 comparison calls,
76 oracle calls, and 72 retained-input captures. All 35,936 NVML observations
report a 40,000 mW enforced limit, not constant operating power, clocks,
temperature, or exclusive machine activity.

The final verifier checks all 442 journals (439 accepted, three preserved
helper rejections), both predeclared schedules, byte-oracle/audit metadata,
source/binary hashes, unchanged native modules, no build/sanitizer overlap,
and exact recomputation of both analyses. Original and exploratory results
remain separate. The optimization goal remains open.
