# S193: integrated packed ANS and worker-local AC output

Date: 2026-09-10. Parent: `016b572b5d832238abe3537cb163e77c02b7c3bd`.
This stage integrates the combination qualified by
[S192](cuda-local-direct-ans-whole-s192.md), then qualifies the actual standard
build. The complete fixed campaign passes all seven prospective performance
gates, and the independent integrated qualification passes. Promote the five
production/test files; keep the broader resident optimization goal active.

## Production integration

The new private `codestream/ans_reverse_bits_internal.h` contains AnsReverseBits.
It stores full 56-bit reverse words plus a pending tail of fewer than 56 bits,
using the growable vector policy qualified in S189/S192. A validated chunk has
at most 31 bits. Combining it with a tail can require 86 conceptual bits, but
the split expressions remain within uint64_t; a flush leaves at most 30 bits.

WriteAnsTokenStream reserves `floor(47 * token_count / 56)` words. The 47-bit
bound is at most 31 hybrid-uint extra bits plus one 16-bit renormalization chunk
per token. A size_t multiplication guard precedes reservation. Existing token,
model and state-transition checks and the ANS recurrence are unchanged. The
unused ReverseBitChunk declaration and old reverse-chunk/temporary-output path
are removed, without a compatibility selector or image-specific fallback.

Append emits the 32-bit terminal state, pending tail and reversed full words
directly through existing checked BitWriter writes inside WithMaxBits. State
and tail combine when the tail is at most 24 bits. The transactional reservation,
nested allotment limits, exception rollback and exact unpadded output length
remain in the existing BitWriter implementation. The pointer-sized callback
target does not imply that every standard library avoids allocation; enclosing
allocation catches remain. No BitWriter implementation, layout or ABI changes.

WriteAcSections now emits each ANS or prefix group into a worker-local writer
and moves successful ownership into its ordered destination slot. Construction,
emission and move remain inside the existing token-work timer. Moved-from local
destruction occurs after that timer, inside section-wall/whole timing. The
writer's transactional callback has returned before the move, so no active
stack allotment is transferred. Failed groups do not publish their local data;
the caller's whole section vector is still published only after batch success.

Dispatch, task count, joining, model lifetime and thread budgets are unchanged.
There is no new pool or padding. The private helper's `_internal.h` suffix uses
the existing installation exclusion. No coefficient representation, GPU kernel,
composition/reduction or tile-scheduling change is mixed into this stage.

The S189 independent reverse-bit oracle is now the regular
`ans_reverse_bits` CTest target. The generator, checks and coverage are retained;
only includes, private names, messages, comments and formatting change. It
checks all 32 chunk widths, 56 pending positions, eight destination alignments,
short and long streams, zero-width chunks, exact/split flushes, repeated append,
insufficient nested limits, post-append limit failure and outer exceptions.
The oracle writes one bit at a time independently of BitWriter and the packer.

Per run it covers 102,552 cases, 6,081,088 chunks and 98,515,702 payload bits,
including 113,670 exact and 1,596,524 split flushes. There are two exact appends
and three rollback checks per case, plus a null-output check. The long fixtures
include generic 62-bit/two-chunk token pairs, beyond ANS's 47-bit bound, while
retaining representable total payloads. Allocation-fault injection is not claimed.

## Source and native fidelity

The source audit reuses 343 original production-source archives. Three existing
files change (ANS, encoder and CMake), two files are added (private helper and
oracle test), and the other 340 sources remain hash-identical. The encoder is
text-identical to S192 after line-ending/terminal-blank normalization.

The lexer-based ANS/helper/test comparisons preserve literals and every
noncomment token. Explicit transformations cover the private include/type,
namespace, three production diagnostic strings, the obsolete struct, equivalent
single-return braces and test names/messages/includes. CMake differs only by
the new test target. These are bounded source comparisons, not a substitute
for native or runtime qualification.

The actual standard-build encoder object matches all 2,512 S192 complete code
COMDATs: 2,511 common entries and one equal-code/relocation opaque symbol-name
pair. In the ANS object, 2,462 common entries match S189 exactly. The writer and
its cleanup have identical instruction bytes, with only their explicitly
renamed append/destructor targets differing. Twenty-six private/helper symbols
are renamed; nine selected pairs are compared, not silently all declared equal.

The actual recurrence (1,140 bytes), recurrence callback (170), packing push
(169), reserve (87), payload getter (22), and append emission lambda (582)
retain complete normalized bytes and typed relocations. The writer remains
812 bytes. The append wrapper grows from 401 to 418 bytes with the production
diagnostic strings and cold-path code/layout changes; it is not native-equivalent.
Eight of nine selected helper pairs match after explicit edge renaming.

Thirty-two complete linked entries in the standard whole/retained bridges are
checked against their actual objects outside typed linker fields. Actual calls
to ANS, prefix writing, vector move, checked append and checked WriteBits are
resolved. The append wrapper's callback-table address is resolved, all six
table entries are checked, and invoke slot two reaches the audited emission
lambda. WithMaxBits's decoded indirect call at byte offset 16 matches that slot
in the pinned MSVC functional header. This checks the indirect path as well as
its caller; no hardware execution-counter claim is made.

Each linked AC worker is 184 instructions/702 bytes, append wrapper 93/418 and
emission lambda 162/582. These counts cover complete selected sections, including
local labels, not every transitive callee or handler. Full normal/ASAN ANS and
encoder object disassemblies are retained. ASAN is behavioral qualification,
not native equivalence. All eleven CUDA modules in each of three new bridges
match current production exactly.

## Build and correctness qualification

The standard build configured 94 CTests and completed 37 build steps without
recompiling CUDA objects. The current codestream library and actual ANS/encoder
objects are archived. Six GPU/support libraries remain identical to S166.
The normal whole and retained bridges link this standard codestream library,
not an individually substituted ANS object. Fresh ASAN ANS and encoder objects
provide an independently instrumented path.

All 93 non-install CTests passed sequentially. The remaining installed-consumer
test passed in a fresh artifact-local directory, using the original install,
configure, build and three consumer executions (codec, codestream and C).
Its diagnostic driver refuses an existing scratch root rather than deleting
one, and additionally checks that the private ANS header is not installed.

Eleven ASAN fixtures passed: reverse-bit oracle, ANS bit emission, entropy,
AC-section validation, BitWriter, BitWriter append, codestream encoder, public
codestream workflow, compact frame, sparse frame and CUDA sparse-resident.
AC-section validation reports 192 valid cases, 1,408 invalid models, 96 invalid
token cases and 64 concurrent cases. BitWriter append reports 18,936 cases,
18,864 insufficient-limit cases and 37,872 rollbacks across all eight offsets
and tails. The reverse-bit oracle reports the complete counts above in both
the regular and ASAN runs. No ASAN errors are reported.

Whole preflight runs all eleven inputs with automatic/eight threads under
normal/ASAN builds, each against original/integrated policies: 88 processes
and 176 whole calls. Two more integrated whole processes run CUDA memcheck
(4K/eight) and initcheck (Keong/eight), for 180 whole preflight calls total.
Both report zero errors; memcheck reports zero leaked bytes/allocations.
Every whole call is byte-checked against the independently loaded baseline.

Twenty-two retained-frame processes separately verify GPU frame hashes,
coefficient counts, nonzeros, ownership bytes and exact codestream output.
Each loads both bridges, captures one frame in each and performs one baseline
oracle plus four compared CPU encodes. Thus these are 44 frame captures and
110 retained encodes: 66 baseline (including the oracles), 44 integrated.
The four compared calls per process are not all candidate calls. All six
photos and 513/p0 use native sparse ownership; the other four inputs are dense.

Preflight therefore covers 290 encodes, with 356 NVML endpoints (180 whole and
176 retained), all reporting the unchanged 40,000 mW enforced limit. Retained
timings are correctness diagnostics, not evidence for the performance gate.
The independent preflight verifier reconciles all 127 terminal-zero child jobs,
the fixed command/order records, source/native audits and complete output logs.

## Prospective performance protocol

The protocol was pinned before source integration and builds. Eleven inputs,
two thread budgets and two repetitions give 44 groups. Each group has four
fresh processes: slots 0/1 are duplicate production controls, 2/3 are duplicate
integrated candidates. Raw harness labels are only slot modulo two; policy and
binary paths/hashes are explicit in the outer records.

The 22 cells are shuffled with seed 19320260910. Four Williams orders are
`0,1,3,2`, `1,2,0,3`, `2,3,1,0`, `3,0,2,1`. The second repetition reverses cell
order and shifts the order index by two. Each slot occupies each position
exactly eleven times. Each process performs one unloaded-reference encode,
eight warmups and 32 measured calls. The fixed budget is 176 processes,
7,216 whole encodes, 5,632 measured calls and 14,080 timing NVML endpoints.

Each metric first takes the median of a process's 32 measured samples, then
the mean of two duplicate medians per policy. All 41 profile phases, all four
candidate/control cross-pairs and both same-policy duplicate differences are
retained. Input aggregates use the geometric mean of four whole-call ratios;
photo/dense aggregates weight groups equally. Separately reduced phase medians
are not added, and process ordinals are not treated as paired observations.

All seven prospective conditions are necessary for promotion:

| Condition | Required |
| --- | ---: |
| Lower ANS/prefix token-worker work | At least 35/44 groups |
| Lower whole-call time | At least 28/44 groups |
| Lower section-wall time | At least 30/44 groups |
| Each of six photo whole-call ratios | At most 1.01 |
| Aggregate photo whole-call ratio | At most 1.005 |
| Strong dense 513/p2 + 1025/p2 whole-call ratio | At most 0.98 |
| Tiny 65/p2 whole-call ratio | At most 1.02 |

The first six repeat S192's primary requirements. The seventh is a new,
prospectively pinned guard for S192's observed 6.114% aggregate tiny-image
regression. No fixed campaign is rerun, filtered or stopped based on performance.
There are no concurrent builds, sanitizers, profilers or heavy verification
during timing. The prelaunch free-C guard is three billion bytes; no clock,
power, cooling, affinity, priority, security or driver setting is changed.

## Complete performance result

All 176 timing processes completed successfully, with every sample retained.
The seven results are worker wins 43/44 (required 35), whole wins 29/44 (28),
section-wall wins 41/44 (30), worst photo ratio 1.000479793080 (at most 1.01),
photo ratio 0.996379752494 (1.005), strong dense ratio 0.945794200551 (0.98),
and tiny 65/p2 ratio 0.934976438081 (1.02). All pass without altering any gate.

The all-input whole-call geometric ratio is 0.981488430661 (-1.851%). Strong
dense whole calls improve 5.421%, photos 0.362%, and tiny 65/p2 6.502%. Photo
whole wins are only 13/24: 1080p (+0.001752%) and 4K (+0.047979%) are essentially
flat, not demonstrated whole-call improvements. Strong dense wins all eight
worker, section-wall and whole groups. The positive 513/p0 aggregate regression
(+0.981%) is retained; the prospective non-regression guard was specifically
65/p2, not a post-hoc requirement invented or discarded for this input.

Negative percentages below mean less time. Worker work aggregates parallel
participants and is not elapsed latency. Each input has four groups.

| Input | Whole change | Worker change | Section change | Whole wins |
| --- | ---: | ---: | ---: | ---: |
| flower_500 | -0.679% | -10.420% | -6.809% | 2/4 |
| 1080p | +0.002% | -6.985% | -5.207% | 1/4 |
| flower_2000 | -0.360% | -9.133% | -10.685% | 3/4 |
| 4k | +0.048% | -5.442% | -3.113% | 2/4 |
| flower_3200x2160 | -0.509% | -12.617% | -12.322% | 3/4 |
| keong_3839x2159 | -0.670% | -7.767% | -9.477% | 2/4 |
| 65_p2 | -6.502% | -28.370% | -13.780% | 3/4 |
| 513_p0 | +0.981% | -3.736% | +0.148% | 2/4 |
| 513_p1 | -1.473% | -7.245% | -6.734% | 3/4 |
| 513_p2 | -6.249% | -26.906% | -22.658% | 4/4 |
| 1025_p2 | -4.585% | -23.900% | -15.943% | 4/4 |

| Pass / input / threads | Baseline ms | Integrated ms | Whole change | Worker change | Section change | Whole pairs won |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / flower_3200x2160 / auto | 200.728900 | 198.090875 | -1.314% | -14.608% | -14.065% | 2/4 |
| 0 / 1025_p2 / auto | 83.834575 | 82.798425 | -1.236% | -20.038% | -13.458% | 2/4 |
| 0 / flower_2000 / auto | 113.177950 | 112.840275 | -0.298% | -8.865% | -10.726% | 2/4 |
| 0 / 513_p2 / 8 | 34.832825 | 32.201400 | -7.554% | -27.919% | -23.651% | 3/4 |
| 0 / 4k / auto | 261.030750 | 261.557625 | +0.202% | -4.000% | -1.797% | 2/4 |
| 0 / 513_p0 / 8 | 12.795175 | 13.572375 | +6.074% | +5.387% | +8.400% | 0/4 |
| 0 / 1080p / auto | 63.919850 | 64.022250 | +0.160% | -8.393% | -5.177% | 1/4 |
| 0 / 65_p2 / 8 | 10.395475 | 9.655975 | -7.114% | -28.874% | -13.547% | 3/4 |
| 0 / flower_3200x2160 / 8 | 205.768750 | 209.054200 | +1.597% | -12.765% | -12.446% | 0/4 |
| 0 / 513_p1 / 8 | 29.093925 | 28.600300 | -1.697% | -7.172% | -6.351% | 2/4 |
| 0 / 1025_p2 / 8 | 89.996300 | 86.294275 | -4.114% | -26.971% | -13.708% | 2/4 |
| 0 / keong_3839x2159 / 8 | 293.640425 | 283.994975 | -3.285% | -11.422% | -13.563% | 4/4 |
| 0 / 513_p0 / auto | 13.971600 | 13.524275 | -3.202% | -7.366% | -2.773% | 3/4 |
| 0 / flower_500 / auto | 17.808200 | 18.616000 | +4.536% | -7.111% | -1.821% | 1/4 |
| 0 / flower_2000 / 8 | 118.328100 | 116.609925 | -1.452% | -11.156% | -10.407% | 2/4 |
| 0 / 513_p1 / auto | 29.652875 | 29.162525 | -1.654% | -9.327% | -8.216% | 3/4 |
| 0 / keong_3839x2159 / auto | 293.576600 | 295.353400 | +0.605% | -7.723% | -6.019% | 2/4 |
| 0 / 65_p2 / auto | 11.158175 | 9.666900 | -13.365% | -34.828% | -20.197% | 4/4 |
| 0 / 513_p2 / auto | 34.568875 | 32.768250 | -5.209% | -26.708% | -21.506% | 4/4 |
| 0 / 1080p / 8 | 63.710750 | 64.678550 | +1.519% | -2.883% | -3.856% | 1/4 |
| 0 / flower_500 / 8 | 19.102400 | 17.530000 | -8.231% | -16.709% | -14.181% | 4/4 |
| 0 / 4k / 8 | 272.182225 | 276.418325 | +1.556% | -2.907% | -0.782% | 1/4 |
| 1 / 4k / 8 | 275.159400 | 274.537225 | -0.226% | -8.520% | -5.236% | 2/4 |
| 1 / flower_500 / 8 | 18.200950 | 19.296650 | +6.020% | -4.466% | -1.560% | 0/4 |
| 1 / 1080p / 8 | 66.391625 | 65.204150 | -1.789% | -9.822% | -7.763% | 3/4 |
| 1 / 513_p2 / auto | 36.731600 | 32.823400 | -10.640% | -29.947% | -27.176% | 4/4 |
| 1 / 65_p2 / auto | 10.349025 | 11.303775 | +9.226% | -17.294% | +0.559% | 0/4 |
| 1 / keong_3839x2159 / auto | 323.197500 | 329.257675 | +1.875% | -3.567% | -4.364% | 0/4 |
| 1 / 513_p1 / auto | 33.267075 | 33.609800 | +1.030% | -4.029% | -3.443% | 2/4 |
| 1 / flower_2000 / 8 | 119.570525 | 120.453350 | +0.738% | -7.169% | -9.805% | 1/4 |
| 1 / flower_500 / auto | 19.479800 | 18.637750 | -4.323% | -12.877% | -9.068% | 4/4 |
| 1 / 513_p0 / auto | 14.711875 | 15.488750 | +5.281% | -0.444% | +6.097% | 1/4 |
| 1 / keong_3839x2159 / 8 | 313.842875 | 308.209400 | -1.795% | -8.187% | -13.568% | 2/4 |
| 1 / 1025_p2 / 8 | 98.295100 | 89.772875 | -8.670% | -27.396% | -20.301% | 4/4 |
| 1 / 513_p1 / 8 | 30.735925 | 29.654400 | -3.519% | -8.367% | -8.831% | 3/4 |
| 1 / flower_3200x2160 / 8 | 220.832800 | 218.617625 | -1.003% | -13.754% | -14.160% | 4/4 |
| 1 / 65_p2 / 8 | 11.317550 | 9.839775 | -13.057% | -31.330% | -20.344% | 4/4 |
| 1 / 1080p / auto | 65.626175 | 65.720800 | +0.144% | -6.698% | -3.977% | 2/4 |
| 1 / 513_p0 / 8 | 14.583900 | 14.028200 | -3.810% | -11.645% | -10.041% | 3/4 |
| 1 / 4k / auto | 283.079775 | 279.346000 | -1.319% | -6.244% | -4.567% | 3/4 |
| 1 / 513_p2 / 8 | 36.865175 | 36.368575 | -1.347% | -22.868% | -18.012% | 2/4 |
| 1 / flower_2000 / auto | 120.248250 | 119.747650 | -0.416% | -9.295% | -11.792% | 3/4 |
| 1 / 1025_p2 / auto | 97.627050 | 93.554800 | -4.171% | -20.897% | -16.123% | 2/4 |
| 1 / flower_3200x2160 / auto | 222.841175 | 219.971575 | -1.288% | -9.247% | -8.500% | 4/4 |

### Duplicate controls and regressions

Across the 176 whole-call cross-pairs, integrated wins 101; worker work wins
163 and section wall 155. Photos win only 50/96 whole pairs, versus 90/96 worker
and 87/96 section pairs. Strong dense wins 23/32 whole pairs, despite all eight
mean-of-duplicate groups winning; its worker and section pairs both win 32/32.
This is weaker whole-pair consistency than S192's 32/32 dense result. The four
cross-pairs share samples and are descriptive checks, not independent trials.

The largest baseline whole duplicate spread is Keong/eight/pass one:
329.463000 versus 298.222750 ms, a 31.240250 ms spread (9.954% of their mean).
Its two integrated medians, 306.456650 and 309.962150 ms, beat the slower control
and lose to the faster one. The group mean improves 1.795%, but only two of its
four whole cross-pairs win. The largest integrated whole duplicate spread is
1025/p2/eight/pass zero: 91.420900 versus 81.167650 ms, 10.253250 ms (11.882%).
Both dense control medians lie between those candidate medians. These samples
are not removed or attributed to RDP, thermals, clocks or cache effects.

Tiny 65/p2 passes its aggregate guard but is not uniformly faster. Pass-one
automatic threads regresses from 10.349025 to 11.303775 ms (+9.226%), losing all
four whole pairs. Its worker work falls 17.294%, while section wall rises 0.559%.
The baseline medians are 10.325700/10.372350, integrated 11.931900/10.675650 ms.
The other three tiny groups win, but pass-one/eight has a large baseline
duplicate spread of 2.856600 ms (25.240% of its policy mean). This campaign's
aggregate improvement neither erases S192's regression nor proves it resolved
under every scheduling condition.

513/p0/eight/pass zero is the only worker-work loss (+5.387%) and also loses
all four whole pairs: 12.795175 to 13.572375 ms (+6.074%). Its section wall rises
8.400%; another 513/p0 group also loses whole time, leaving its aggregate +0.981%.
No input-specific fallback is introduced. All other worker groups improve.

4K whole changes are +0.202%, +1.556%, -0.226% and -1.319% in the table order;
all four groups have mixed whole cross-pairs. Worker and section group means
improve throughout. In the largest whole loss (pass-zero/eight), independently
reduced quantization, codestream, section-wall and worker differences are
+0.922600, +3.173575, -0.070625 and -0.640075 ms. They are not additive causes
of the +4.236100 ms whole difference. Unchanged CUDA modules do not imply
constant GPU execution time, and this CPU integration does not demonstrate
a 4K throughput gain.

### Promotion and next bottleneck

Promote the qualified packed reverse-bit emitter, worker-local output and
independent regular test without a compatibility layer. The strongest result
is consistent reduction in dense token-worker and section work, supported by
all dense cross-pairs and the full integrated correctness checks. Whole-call
gains are smaller and noisier. Do not claim a universal improvement or that the
backend is maxed out.

The complete audit reconciles 303 terminal-zero child jobs: 302 raw accepted
records plus the explicitly adjudicated CTest summary mismatch. Whole/retained
campaigns total 7,506 encodes (7,396 whole and 110 retained), 5,632 measured whole
calls, 44 separately counted GPU frame captures and 14,436 NVML endpoints, all
at the unchanged 40,000 mW enforced limit. Regular/ASAN fixture operations are
additional and are not silently counted as whole encodes.

The next bounded step is to reprofile this integrated production path. Large
photo throughput remains dominated elsewhere; S193's token-work win alone
does not justify another ANS micro-optimization, a GPU scheduling change, or
reusing the old S169 phase balance as if production had not changed. Preserve
the small-input losses and duplicate variation when selecting the next test.

## Execution records and bookkeeping corrections

The standard build ran 14:37:40.136282–14:38:01.720312 UTC. The sequential
CTest run was 14:40:08.654291–14:44:35.357730 (266.41 seconds reported by CTest),
and the installed consumer ran 14:45:44.879003–14:46:00.401466. The supplemental
bridge/ASAN build ran 14:47:19.182943–14:49:07.772629, producing 35 archived
outputs totaling 63,896,719 bytes. ASAN fixtures ran 14:52:01.109381–14:54:29.317408.
Whole/retained/CUDA preflight ran 14:58:01.450127–15:01:25.484746. The fixed
timing campaign ran 15:07:14.850681–15:22:29.761131 (15 minutes 14.910 seconds).
These are journaled child-process times,
not estimates of compiler/kernel-only execution.

Several bookkeeping corrections are preserved without changing binaries or
rerunning workloads:

- The first preparation script failed its schedule-balance assertion before
  creating the stage root or launching a build. Version two pins the balanced
  order described above before any source/build action.
- Installed CTest prints `100% tests passed out of 93`, while the initial wrapper
  expected an additional `0 tests failed` phrase. Its raw job record therefore
  retains `accepted:false`, despite terminal exit zero. A separate adjudication
  checks all 93 unique expected Passed lines, the exact summary, PID, terminal
  and log hashes. Only the still-missing installed consumer was then run.
- The successful supplemental build's summary initially collided with its
  existing `extra_build.json` job record. No file was overwritten. A new reader
  verifies all 35 outputs and writes `extra_build_summary.json`. Read-only CUDA
  extraction confirmation uses new directories, preserving the initial extracts.
- The initial native audit compared in-memory relocation tuples with serialized
  JSON lists and stopped before creating its output directory. Version two
  compares their exact JSON representation. This is not a native-code fix.
- The source reader's later versions compare against the pinned baseline commit
  and account for the two new files becoming tracked. The final verifier selects
  that reader so staging/committing unchanged source does not invalidate a
  working-tree-state assumption. Earlier readers and their successful checks
  remain preserved; no generated source or workload is changed.
- The retained-frame accounting is clarified by independently parsing all four
  labels and the original harness source: 66 baseline and 44 integrated encodes,
  not four candidate encodes per process. The separately pinned whole endpoint
  count of 180 remains correct; retained adds 176 for 356 preflight endpoints.

Every job has exclusive intent, launch, terminal, result and log records. The
raw CTest marker mismatch remains distinguishable from an actual child failure.
There was no admin/firewall intervention or observed security blocker.

## Evidence and reproduction

The artifact root is `build-cuda-ninja/profiles/s193-artifacts`. Its records
include `before.json`, the prospective `protocol.json`, `main_inputs.json`,
`source.json`, `main_build.json`, `normal_checks.json`, `ctest_adjudication.json`,
`extra_build_summary.json`, `asan_checks.json`, `probe.json`, `native.json`,
`qualification_plan.json`, `preflight.json`, `retained_validation.json`, and
`timing_result.json`. Raw logs and exclusive process journals are retained.
`analysis.json` holds all samples' derived per-group metrics, duplicate medians,
four cross-pairs, aggregate ratios and seven gate decisions. It does not replace
the raw samples in `timing_result.json` and the independently reparsed logs.

The final source archive reuses verified pre-build production archives, retains
all versioned S193 helpers and this report, and preserves the unchanged S192
2,198-file frozen inventory. Final source hashes, decision and summary records
are included in the stage's artifact inventory. The read-only audit command is:

```powershell
python build-cuda-ninja/profiles/s193_verify_v2.py --frozen
```

This is verification, not a benchmark restart or rebuild. The source reader
compares against the pinned parent rather than requiring an uncommitted tree.
The independent bit-at-a-time oracle is now a normal repository test, so future
builds do not need the diagnostic artifact harness to exercise it.

The output of `s193_report.py` includes all 44 groups and selected complete
phase/duplicate details. No significance, measured peak-memory reduction,
cache-line contention proof, fixed-clock/thermal causal explanation or
cross-platform performance claim follows from this campaign. The broad
resident-encoding optimization goal remains active.
