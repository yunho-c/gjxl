# S186: fixed-policy qualification of packed ANS staging

The fixed-policy packed writer passes correctness but misses its prospective
performance gate: 26/44 whole wins versus the required 28. Token writing
improves in 39/44 quartets; strong dense controls improve consistently, while
photo whole times remain mixed. The candidate stays unpromoted and
production remains unchanged. No other optimization is combined with it.

## Scope and controls

Parent: `364be8abc5a7f30d09423f49ee62e599538cff9e`, branch `feat/cuda`.
S185 passed its follow-up gate but showed consistent dense-input benefits
and mixed photo whole times. Its runtime selector, renamed/noinline writer
helpers, and common atomic audit could not establish production performance.

S186 compiles current production `src/codestream/ans.cpp` for the original
DLL and the exact frozen `s183_ans.cpp` plus `s183_reverse_bits.h` for the
packed DLL. The candidate changes only the reverse-staging writer and its
header include. ANS recurrence, models, configuration and token validation,
count-only paths, and other production sources are unchanged. The packing
mechanism and bounds remain those in [S183](cuda-reverse-bit-packing-s183.md).
Smaller reservation is not a measured peak-memory or performance claim.

There is no selector, counter, diagnostic writer rename, noinline wrapper,
or per-call atomic audit in these DLLs. Labels 0/1 use the same original
binary and 2/3 the same packed binary. Labels appear only in harness output
and NVML records. Separate processes execute one fixed policy throughout.
No simultaneous original/candidate backend residency is introduced.

Normal MSVC and clang AddressSanitizer builds reuse frozen support objects
and libraries; the new ANS objects override their archive definitions.
Compile and link are separate. All four DLLs contain the same eleven CUDA
module hashes as production; no GPU code is compiled or changed.
The new normal original writer, its recurrence specialization, and callback
all match production instruction bytes and normalized relocations. Only
anonymous-namespace hashes are normalized. Candidate disassembly and both
full ASAN objects are retained. This verifies the original control's
unlinked functions, not linked placement or clock/thermal equivalence.

The harness mechanically removes S181 configuration and audit code from the
S182 fixed-policy loop. It still loads and encodes the frozen S168 reference
first, checks any frozen external oracle, then destroys and unloads that
reference before opening the measured DLL. Each measured Encode prepares a
fresh workflow while retaining the backend. Exact bytes, the complete
`VarDctEncodingSummary`, four-byte coefficients, native-owner storage, and
the S185 frozen output-byte count are checked every time. ANS call/token
counters are intentionally absent.

The workload remains fully resident effort 7, distance 1.2 (pattern 2 uses
0.01), final score disabled. Setup/reference unloading, clearing results,
validation, and read-only NVML capture are outside the outer timer. All
Encode work is inside. All 41 phase fields are retained; worker work is
nested, not additive wall time.

## Qualification and prospective timing protocol

Six normal/ASAN candidate host fixtures cover independent bit emission,
entropy primitives, and private AC sections. The exact same fixture coverage
as S185 is rerun against the fixed candidate, including 30,240 byte
comparisons and 30,240 atomic failures per bit-emission build, 816 HybridUint
configurations, mapped/unmapped and split layouts, invalid models/tokens,
and concurrent AC-section cases. The fixture's predicted batch-write count
still describes the S167 baseline calculation, not new packer writes.

Whole preflight covers eleven inputs, automatic/eight threads, original/
packed, normal/ASAN: 88 processes. Each performs one S168 reference and one
measured encode. Separate candidate memcheck on 4K/eight and initcheck on
Keong/eight bring this to 90 processes and 180 whole calls. Sanitizer times
are not performance evidence.

The fixed timing campaign has 44 quartets: eleven inputs, automatic/eight,
two passes. Twenty-two case/thread cells are shuffled once. Pass 0 alternates
label orders 0/2/3/1 and 2/0/1/3 (ABBA and BAAB). Pass 1 reverses cell order
and reverse-complements each label order, balancing both path placement and
duplicate labels. Each label occupies every quartet position eleven times.
Eight warm and 32 measured encodes plus one unloaded reference per process
give 176 timing processes, 7,216 whole calls, and 5,632 measured calls.

The primary statistic first takes each process's median, then subtracts the
mean of original duplicate medians from the mean of packed duplicate
medians. Samples at the same ordinal position in separate processes are
not treated as contemporaneous pairs. All four candidate/original pairs,
both same-path duplicate differences, and all phases are retained. Geometric
whole-time ratios give equal weight to each case/thread/pass quartet; every
input has four quartets. No timing sample is filtered or silently rerun.

The prospective gate requires every criterion below to be met before the
candidate becomes eligible for production implementation and full
qualification. This practical tolerance gate is not statistical significance
and does not itself promote the change.

| Criterion | Requirement |
|---|---:|
| Token-write worker wins, stage 33 | At least 35/44 |
| Whole-encoder wins | At least 28/44 |
| AC-section wall wins, stage 31 | At least 30/44 |
| Each photo's geometric whole-time ratio | At most 1.01 |
| All photo quartets' geometric whole-time ratio | At most 1.005 |
| Strong dense controls' geometric whole-time ratio | At most 0.98 |

The six photos are flower/500, 1080p, flower/2000, 4K, flower/3200x2160,
and Keong. Strong dense controls are 513/p2 and 1025/p2 (eight quartets).
Other controls are 65/p2 and 513/p0/p1. The labels 1080p and 4K mean
1919x1079 and 3839x2159; some photo inputs are derived, not independent
native-resolution captures. Pattern dimensions are 65x71, 513x519, 1025x1031.

All 14,080 timing NVML endpoints must report a configured 40,000 mW limit;
this does not lock clocks, thermals, or actual power. No build, sanitizer,
profiler, or heavy artifact verification overlaps timing. Routine per-job
log hashing remains. Each child requires at least 3 GB free on C: before
launch and has separate intent, PID launch, terminal exit, and final report
journals. A resource abort is not a license to restart the fixed campaign.
S182's earlier storage error is not reclassified as explained or resumed.

## Results and decision

All 273 journaled jobs are accepted: one build, six host fixtures,
90 whole preflight/sanitizer processes, and 176 fixed-policy timing
processes. All 7,396 whole calls pass, including 5,632 measured calls.
Memcheck reports zero errors and zero leaked bytes/allocations; initcheck
reports zero errors. All 14,260 observed preflight/warm/measured NVML
endpoints are 40,000 mW. No encoder rerun, filtered sample, resource abort,
or privilege/firewall intervention was needed.

On 2026-09-10 UTC, the build ran 10:53:08.191–10:55:04.790, producing
34 files totaling 58,132,233 bytes. Whole preflight ran
10:57:01.407–10:59:57.143. The fixed campaign ran
11:01:24.427–11:16:12.960. Source/native audits and host tests preceded
whole preflight; no compilation or sanitizer overlapped the campaign.

**The candidate fails the prospective gate and remains unpromoted.**
It has 26 whole wins, below the required 28. Other criteria pass; neither
the aggregate improvement nor proximity to the threshold overrides that
failure. No criteria are relaxed and this campaign is not rerun.

| Criterion | Actual | Result |
|---|---:|---|
| Token-write worker wins | 39/44 | Pass |
| Whole-encoder wins | 26/44 | Fail (requires 28) |
| AC-section wall wins | 30/44 | Pass |
| Largest photo geometric ratio | 1.006841 (4K) | Pass |
| All photo geometric ratio | 0.996766 | Pass |
| Strong dense geometric ratio | 0.940735 | Pass |

The all-input equal-weight geometric ratio is 0.979596, a 2.040% reduction.
Photos aggregate to -0.323%; strong dense controls to -5.927%. These are
defined cohort summaries, not universal speedups or significance estimates.

Negative values below favor packing. Each percentage is the ratio of
candidate and original duplicate means of process medians, minus one.

| Input | Threads | Whole R0 (%) | Whole R1 (%) | Token work R0 (%) | Token work R1 (%) | Section R0 (%) | Section R1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flower_500 | auto | -3.116 | -3.174 | -9.998 | -7.238 | -4.585 | -5.004 |
| flower_500 | 8 | +3.171 | -2.844 | -2.130 | -7.900 | +0.886 | -5.648 |
| 1080p | auto | -0.017 | +0.671 | -2.065 | -1.228 | +0.563 | +1.233 |
| 1080p | 8 | -1.045 | -0.270 | -4.625 | -3.059 | -2.533 | -1.357 |
| flower_2000 | auto | -2.674 | +2.577 | -12.337 | -0.091 | -11.965 | +4.317 |
| flower_2000 | 8 | -1.892 | +0.090 | -9.839 | -5.329 | -8.785 | -8.651 |
| 4k | auto | +0.645 | +0.584 | -2.178 | -3.600 | +3.468 | +0.855 |
| 4k | 8 | +0.731 | +0.776 | -4.577 | +1.101 | -3.414 | +3.311 |
| flower_3200x2160 | auto | -2.547 | +1.625 | -3.977 | +1.681 | -3.403 | +2.603 |
| flower_3200x2160 | 8 | -0.326 | -0.908 | -0.849 | -5.408 | +3.753 | -6.199 |
| keong_3839x2159 | auto | +0.478 | +0.461 | -1.112 | -1.540 | +0.116 | +0.456 |
| keong_3839x2159 | 8 | -1.167 | +0.756 | -6.136 | -4.201 | -5.932 | -1.574 |
| 65_p2 | auto | -2.070 | +8.216 | -22.810 | -22.649 | -11.272 | -7.105 |
| 65_p2 | 8 | -13.734 | -10.142 | -34.121 | -26.914 | -21.745 | -15.231 |
| 513_p0 | auto | +2.382 | +0.451 | +2.525 | +0.108 | +3.694 | +0.582 |
| 513_p0 | 8 | +1.451 | -13.136 | +4.787 | -21.423 | +4.975 | -20.841 |
| 513_p1 | auto | -3.244 | -2.275 | -8.263 | -4.563 | -8.647 | -4.208 |
| 513_p1 | 8 | +0.586 | +0.459 | -6.483 | -6.609 | -3.460 | -5.326 |
| 513_p2 | auto | -0.609 | -6.693 | -21.802 | -28.029 | -16.760 | -24.663 |
| 513_p2 | 8 | -8.037 | -12.179 | -26.882 | -31.674 | -23.101 | -29.109 |
| 1025_p2 | auto | -2.650 | -8.287 | -21.242 | -26.248 | -12.585 | -21.129 |
| 1025_p2 | 8 | -5.299 | -3.139 | -26.039 | -19.284 | -16.865 | -13.919 |

The four-quartet input aggregates help distinguish repeat consistency from
one strong or weak cell. In particular, sparse 513/p0's negative aggregate
does not mean it consistently improves.

| Input | Geometric whole change (%) | Whole wins / 4 |
|---|---:|---:|
| flower_500 | -1.527 | 3 |
| 1080p | -0.167 | 3 |
| flower_2000 | -0.496 | 2 |
| 4k | +0.684 | 0 |
| flower_3200x2160 | -0.550 | 3 |
| keong_3839x2159 | +0.129 | 1 |
| 65_p2 | -4.797 | 3 |
| 513_p0 | -2.429 | 1 |
| 513_p1 | -1.133 | 2 |
| 513_p2 | -6.971 | 4 |
| 1025_p2 | -4.870 | 4 |

Dense 513/p2 and 1025/p2 improve whole time in all eight quartets. Their
whole reductions range from 0.609% to 12.179%, and token-write work falls
19.284–31.674%. These support a real optimization opportunity in bit staging,
but do not establish this exact implementation as the new default.

Photo whole times improve in only 12/24 quartets. All four 4K comparisons
are slower: +1.5810 to +2.1136 ms (+0.584% to +0.776%), despite token-write
work improving in three. Their quantization deltas have both signs:
+0.9384, +1.3065, -0.7288, and -1.2155 ms in campaign order. This cannot
be explained merely by pointing to one unchanged phase. Separate phase
medians are not additive, and this experiment does not identify a causal
GPU cadence, thermal, scheduling, or linked-layout mechanism.

Keong improves in one whole quartet even though all four token-write
worker comparisons improve. Tiny 65/p2 and sparse 513/p0 remain variable:
65/auto/pass 1 is +8.216% whole with -22.649% token work, while sparse
513/eight/pass 1 is -13.136% whole but the other three sparse-513 whole
comparisons are slower. The latter's aggregate -2.429% is therefore
insufficient grounds for a sparse-specific policy.

Same-path duplicate differences remain in the analysis. For 4K/auto/pass 1,
the original and packed whole duplicate deltas are +8.9718 and -2.1859 ms,
versus the primary +1.5810 ms. For Keong/auto/pass 1, they are -12.5248 and
+6.9700 ms versus +1.3513 ms. At 65/auto/pass 1, they are +1.5911 and
-3.9651 ms versus +0.8558 ms. These controls are not subtracted, filtered,
or used to causally excuse a regression.

## Next bounded investigation

The result supports improving the mechanism, not promoting this version or
selecting a photo/synthetic threshold from the observed cohort. Native
disassembly gives a concrete source lead: the original chunk callback is
16 instructions/59 bytes; the packed callback is 51 instructions/170 bytes.
The latter performs packing and still contains vector-growth handling even
though the 47-bits-per-token reservation already bounds all full words.
Static instruction counts are not executed counts or a timing attribution.

A separate primitive experiment can replace the growable word vector with
fixed-capacity, uninitialized integer-array ownership allocated once under
the same proven bound. Keep the used-word count separate, initialize every
word before reading it, retain allocation failure handling at construction,
and make the chunk callback allocation-free. This can remove a redundant
growth path without zero-initializing the entire worst-case capacity or
adding input-specific selection. It must preserve the exact bit-order,
tail, arithmetic bounds, empty-stream behavior, and transactional append
proofs and pass independent normal/ASAN fixtures before whole-encoder work.
Any compiler inlining change needs its own native evidence; no speedup is
predicted here.

Read-only scheduling review also confirmed that eager and on-demand
per-encode pools were already tested in
[S163](cuda-per-encode-worker-pool-s163.md) and
[S164](cuda-on-demand-worker-pool-s164.md), with neither promoted.
Those are not new untested remedies. The packed-buffer follow-up should
remain separate from pool, tile, composition/reduction, and coefficient
ownership changes.

## Evidence and preservation

Artifacts are under `build-cuda-ninja/profiles/s186-artifacts`, with versioned
`s186_*` helpers beside them. The verifier reparses every whole log,
reconstructs the fixed schedule and commands, checks all separate process
journals and nonoverlapping intervals, recomputes the full analysis, and
checks source/support/native hashes. All 495 frozen S185 artifacts remain
unchanged. The 343 production-source archives are reused from S182 rather
than recopied. Raw samples, every phase and duplicate comparison, complete
build/native records, protocol, decision, and final source/hash inventories
are retained. At the post-campaign check this stage held 123,746,308 logical
non-temp artifact bytes before the final documentation/snapshot writes;
C: had 4,112,199,680 bytes free. U: remained at 30,883,840 bytes free.

No production or machine settings were changed, no old artifact or helper
was overwritten, no cleanup was performed, and no recovery volume was used.
There was no source/build correction or failed journaled child in S186.
The protected untracked notes were not read, edited, or staged. Nothing is
pushed. Full CTest, installed-consumer and cross-platform qualification,
new decoded-quality cohorts, ThreadSanitizer, allocation-fault injection,
and measured peak-memory reduction are not claimed by this diagnostic stage.
The broader resident-encoder objective remains active and is not maxed out.
