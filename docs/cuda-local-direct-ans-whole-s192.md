# S192: worker-local direct-packed ANS in the resident encoder

Date: 2026-09-10. Parent: `9c2388adcaf3203da78ba2acc721da5e8c68c7da`.
This resident experiment follows the CPU ownership screen in
[S191](cuda-local-ans-output-concurrency-s191.md). Production remains unchanged.
All nine prospective criteria pass: local direct output wins 42/44 token-work
and 29/44 whole-time groups against production. Geometric whole time improves
8.965% on strong-dense cases and 0.754% across photos. This qualifies the
combination for separate production integration and qualification, not automatic
promotion. The tiny 65x71 case regresses 6.114% overall and remains a follow-up
concern rather than being filtered away.

## Question and isolated change

[S189](cuda-direct-packed-ans-append-s189.md) qualified a checked direct append
from growable packed reverse storage. Its single-threaded actual-ANS screen
passed, but [S190](cuda-direct-packed-ans-whole-s190.md) rejected the whole
encoder candidate, including repeated dense-input token-writing regressions.
S191 then found a strong concurrency interaction when changing adjacent
destination metadata to worker-local metadata followed by one move publication.

S192 tests that ownership change at the real AC section boundary. Three
fixed-policy binaries provide the following comparisons:

| Policy | ANS implementation | AC group destination |
|---|---|---|
| original | current production | ordered candidate vector slot |
| adjacent_direct | exact S189 direct-packed object | ordered candidate vector slot |
| local_direct | exact S189 direct-packed object | worker-local writer, then successful move |

Local versus adjacent holds the ANS object fixed. Local versus original
tests the complete proposed combination. Adjacent versus original is a
contemporaneous control, not a rerun or reclassification of S190. All three
contrasts share the same six processes in each group; they are not three
independent campaigns.

The private encoder overlay changes only the WriteAcSections group lambda.
Both the ANS and prefix branches emit to a default-constructed local BitWriter.
On success its ownership moves into `candidate[1 + index]`. The existing
WorkEnd remains after that move, so construction, emission and publication
remain inside token-writing work timing. No byte-buffer copy is added.

The real AC destinations are initially empty. S191's synthetic nonzero
prefixes, repeated identical sections and diagnostic barrier pool are not
introduced here. Its 37.814% large parallel direct-output gain is not a
prediction of a real-image gain. In particular, the original/growable S191
controls often used bytewise unaligned Append; that does not describe these
initially aligned real AC destinations.

Models remain immutable until workers join. Task count, dispatch, thread
budget, joining, ordering, token/model validation and final output publication
are unchanged. Failed local output is destroyed without publishing its slot.
The enclosing candidate vector is still published to the caller only after
the entire batch succeeds, preserving caller-visible failure atomicity.
BitWriter's default move is noexcept; its layout and public ABI are unchanged.
The direct writer's WithMaxBits operation returns and restores its enclosing
allotment pointer before the local writer moves. No live callback/allotment
pointer is transferred from that stack frame. The moved-from local writer's
destruction occurs after WorkEnd, but remains inside section-wall and whole
timing; the moved byte buffer remains owned by the candidate vector.
No padding, compatibility layer, new pool, image-specific policy, coefficient
ownership, GPU composition/reduction or tile-scheduling change is included.

## Source, build and native evidence

The source verifier reconstructs the encoder from current production plus
one exact block replacement. The private AC section fixture differs from its
production test only by including that overlay. Terminal blank-line
normalization is explicit; no substantive source differences are ignored.
The S190 source proof and frozen S189/S186 provenance are also revalidated.

Normal MSVC 14.37 and clang-cl 22 ASAN builds compile the new encoder and its
private fixture. A new normal production-source encoder object is compiled
only as a compiler-control audit, not linked or timed as another candidate.
Compile/link remain separate. The exact S189 normal/ASAN ANS objects are reused.
The new ASAN encoder replaces, rather than supplements, the old encoder object.
Pinned support libraries and existing bridge objects supply the rest.

The four original/adjacent normal/ASAN whole DLLs from S190 and both sealed
S186 whole harnesses are reused without recompilation or relinking. No ANS,
BitWriter, harness or GPU source is compiled. The successful build produces
fourteen files totaling 34,550,078 bytes. All eleven embedded CUDA module
hashes match current production in each new DLL and in the reused controls.

The normal compiler-control object matches all 2,509 common complete production
code COMDATs, including typed relocation targets and original addends, with
no added or removed entries. Comparing that control with the local encoder
finds 2,508 common entries: 2,506 unchanged, two changed, four added and one
removed. Only the AC worker and its `dtor$0` cleanup change among common entries.

The additions are BitWriter/std move helpers, an additional AC cleanup and an
opaque symbol. The added/removed opaque pair has identical complete normalized
116-byte code and relocations; only its symbol name changes. No other common
code change is hidden by that pairing.

Thirty-five complete selected linked entries are checked against their object
code outside explicit typed linker fields. These include the actual AC worker
and dispatcher, the selected ANS writer/recurrence/callback/direct append,
selected BitWriter methods, and the local cleanup/move path. Actual listed
REL32 call targets are resolved, including both ANS/prefix calls and the local
worker's existing byte-vector move assignment. BitWriter/std move wrappers are
inlined; their unused out-of-line copies do not survive linking.

The original and adjacent AC workers each contain 147 instructions/597 bytes;
the local worker contains 184 instructions/702 bytes. These are complete
selected sections, not dynamic instruction counts or all transitive callees.
Normal/ASAN encoder object disassemblies are retained, but ASAN behavioral
qualification is not presented as native equivalence.

The broader encoder audit requires four-byte ADDR32NB RVA and SECREL TLS fields
as well as the REL32 family. Types and widths are checked against the pinned
Windows SDK winnt.h definitions. Object comparisons retain typed targets and
addends, including local self-relative targets. Individually resolved TLS layout
and every linked data section are not claimed.

## Correctness before timing

The normal and ASAN private AC fixtures both report:

```text
AC section validation PASS valid=192 invalid_models=1408 invalid_tokens=96 concurrent=64
```

They compare with the fully validating legacy writer across prefix and ANS
models, layouts, mapped/unmapped contexts, empty/populated/custom sections,
thread budgets 0/1/2/8, errors and concurrent callers. Complete bytes and logical
bits, immutable models and caller-output atomicity are checked. This is not
allocation-fault injection, thread-failure injection, TSAN or full CTest.

Whole preflight uses eleven inputs, auto/eight threads, normal/ASAN builds and
all three policies: 132 processes. Each checks one current-production reference
encode and one candidate encode using the sealed S186 harness; the reference
DLL is unloaded before the measured DLL is loaded.
Exact output bytes, complete summaries, native coefficient-owner storage and
the frozen output census must agree. Raw harness labels 0/1 identify duplicate
slots only; actual policy is recorded by DLL path/hash and process metadata.

Two additional local-direct processes run CUDA memcheck on 4K/eight and
initcheck on Keong/eight. Both report zero errors; memcheck also reports zero
leaked bytes/allocations. All 134 preflight processes pass, covering 268 whole
calls and 268 NVML endpoints. The independent preflight verifier accepts all
137 journaled jobs including the build and two host fixtures.

## Prospectively fixed timing protocol

The input matrix is inherited from S186/S190: flower500, 1919x1079, flower2000,
3839x2159 flower, flower3200x2160, 3839x2159 Keong, 65x71 pattern 2,
513x519 patterns 0/1/2, and 1025x1031 pattern 2. The six photo cases are nearest
replicated inputs, not independent native-resolution captures. Strong-dense
cases are 513 pattern 2 and 1025 pattern 2.

The protocol is pinned before build. Preflight order uses seed 19220260910;
timing cell order uses 19220260911. Eleven inputs times auto/eight threads times
two passes give 44 six-process groups. Slots 0/1 are original, 2/3 adjacent and
4/5 local. The first-pass order rotates `[0,1,5,2,4,3]` by cell index modulo six.
Pass one reverses cell order and uses the reversed slot order plus one modulo
six. Every slot occupies each process position seven or eight times.

Each process performs eight warm and 32 measured candidate calls plus one
reference call whose DLL is unloaded before measurement: 264 timing processes,
10,824 whole calls, 8,448 measured
calls and 21,120 NVML endpoints. Combined with preflight the planned totals are
11,092 whole calls and 21,388 endpoints. Every endpoint must report the existing
40,000 mW enforced limit; this is not a fixed-clock or thermal-state claim.

Each process has durable intent, launch/PID, terminal exit and final records,
an exclusive log and a 3 GB C-space prelaunch guard. There are no sample filters,
performance stopping rules or campaign reruns. No build, sanitizer, profiler or
heavy artifact verification overlaps timing. Live handles are polled until a
durable terminal exit; an observation timeout is not treated as a failed run.

For every one of 41 existing phases and outer whole time, analysis takes each
process's median, then the mean of the two duplicate medians per policy.
Each contrast retains four cross-pairs and both same-path duplicate differences.
There is no ordinal sample pairing across separate processes. Geometric ratios
weight groups equally, giving four groups per input. Separately reduced phase
medians are not added together or treated as causal attribution.

All six primary requirements compare local_direct with original. They are
unchanged from S190. Three additional ownership requirements compare local
with adjacent direct output. All nine must pass before separate production
integration and qualification is justified; passing is not automatic promotion.

| Contrast / criterion | Requirement |
|---|---:|
| local/original token-writing work wins | at least 35/44 |
| local/original whole-time wins | at least 28/44 |
| local/original section-wall wins | at least 30/44 |
| local/original each photo geometric whole ratio | at most 1.01 |
| local/original all-photo geometric whole ratio | at most 1.005 |
| local/original strong-dense geometric whole ratio | at most 0.98 |
| local/adjacent strong-dense token-work wins | at least 6/8 |
| local/adjacent strong-dense section-wall wins | at least 6/8 |
| local/adjacent strong-dense geometric whole ratio | at most 0.99 |

These are prospective practical acceptance criteria, not statistical
significance, universal image gains, hardware-counter causality or completion
of the broad optimization objective.

## Results and decision

All 264 timing processes complete successfully, preserving 10,824 whole calls
and 8,448 measured calls. The full independent verifier accepts all 401 jobs,
11,092 combined preflight/timing whole calls and 21,388 NVML endpoints. Every
endpoint reports the unchanged 40,000 mW enforced limit. There is no campaign
restart, dropped group, sample filter or modified acceptance requirement.

**All nine prospective criteria pass.** Production is not promoted in this
diagnostic stage; the exact integrated implementation still needs qualification.

| Criterion | Observed | Result |
|---|---:|---|
| local/original token-work wins | 42/44 | pass |
| local/original whole wins | 29/44 | pass |
| local/original section-wall wins | 37/44 | pass |
| local/original largest per-photo geometric whole ratio | 0.9953310283 | pass |
| local/original all-photo geometric whole ratio | 0.9924558787 | pass |
| local/original strong-dense geometric whole ratio | 0.9103459672 | pass |
| local/adjacent strong-dense token-work wins | 8/8 | pass |
| local/adjacent strong-dense section-wall wins | 8/8 | pass |
| local/adjacent strong-dense geometric whole ratio | 0.8742809605 | pass |

The complete combination improves whole time by **1.987%** over the full
equal-weight matrix, **0.754%** over photos and **8.965%** over strong-dense
cases. It wins all eight dense whole/worker/section groups and 14/24 photo
whole groups. Each photo's four-group geometric whole ratio is below one;
this does not mean that every individual photo/thread/pass group wins.

The ownership-only comparison improves strong-dense whole time **12.572%**
against adjacent direct output, with eight wins in each of whole, token work
and section wall. It wins 24/44 whole groups overall. Its photo aggregate is
**0.445% slower** than adjacent direct output; the ownership benefit is not
universal across image types or stages.

The contemporaneous adjacent-direct control is 4.125% slower on strong-dense
cases than production, with zero dense section-wall wins, despite a 1.194%
photo aggregate improvement. This is consistent with investigating an
ownership-sensitive dense cost, but does not establish cache-line invalidations
as its cause. It does not replace, pool with or reclassify S190's failed campaign.

All per-input whole-time changes follow. Negative favors the numerator policy;
each row represents four groups. The three columns share the same processes.

| Input | Local / original (%) | Local / adjacent (%) | Adjacent / original (%) |
|---|---:|---:|---:|
| flower_500 | -0.629 | +3.138 | -3.652 |
| 1080p | -0.669 | -0.229 | -0.441 |
| flower_2000 | -0.467 | +1.436 | -1.876 |
| 4k | -0.992 | -0.988 | -0.004 |
| flower_3200x2160 | -1.238 | -0.632 | -0.610 |
| keong_3839x2159 | -0.530 | +0.005 | -0.535 |
| 65_p2 | +6.114 | +7.346 | -1.148 |
| 513_p0 | -1.884 | +3.560 | -5.257 |
| 513_p1 | -2.738 | -0.781 | -1.973 |
| 513_p2 | -9.611 | -14.108 | +5.236 |
| 1025_p2 | -8.316 | -11.009 | +3.026 |

Every group is retained below. Work/section/whole columns compare local with
original; the last column isolates local versus adjacent whole time. All values
are percentage changes. Full 41-phase results, four cross-pairs and both
same-path duplicate controls per contrast remain in analysis.json.

| Input | Threads | Pass | Token work (%) | Section wall (%) | Whole (%) | Whole local/adjacent (%) |
|---|---|---:|---:|---:|---:|---:|
| 1025_p2 | auto | 0 | -22.878 | -15.880 | -4.725 | -10.582 |
| 1025_p2 | auto | 1 | -21.767 | -14.191 | -2.124 | -13.085 |
| 1025_p2 | 8 | 0 | -33.929 | -25.095 | -12.993 | -8.983 |
| 1025_p2 | 8 | 1 | -33.337 | -24.236 | -12.908 | -11.335 |
| 1080p | auto | 0 | -5.333 | -1.091 | +0.052 | +1.147 |
| 1080p | auto | 1 | -9.034 | -5.038 | -0.369 | +0.179 |
| 1080p | 8 | 0 | -13.351 | -8.649 | -3.451 | -0.556 |
| 1080p | 8 | 1 | -3.332 | -1.968 | +1.153 | -1.665 |
| 4k | auto | 0 | -2.269 | -4.517 | -1.026 | -0.148 |
| 4k | auto | 1 | -8.393 | -8.017 | -1.912 | -2.258 |
| 4k | 8 | 0 | -10.909 | -11.045 | -2.316 | -4.079 |
| 4k | 8 | 1 | +0.081 | +4.004 | +1.325 | +2.659 |
| 513_p0 | auto | 0 | +2.691 | +8.434 | +7.657 | +0.963 |
| 513_p0 | auto | 1 | -19.820 | -16.494 | -10.450 | -1.987 |
| 513_p0 | 8 | 0 | -8.467 | -6.647 | -7.212 | +7.249 |
| 513_p0 | 8 | 1 | -1.498 | +5.659 | +3.599 | +8.375 |
| 513_p1 | auto | 0 | -2.363 | +0.857 | +3.363 | -4.333 |
| 513_p1 | auto | 1 | -16.988 | -15.022 | -7.660 | +1.385 |
| 513_p1 | 8 | 0 | -5.933 | -3.994 | -0.661 | -2.353 |
| 513_p1 | 8 | 1 | -15.557 | -12.473 | -5.617 | +2.326 |
| 513_p2 | auto | 0 | -32.232 | -28.588 | -10.860 | -12.728 |
| 513_p2 | auto | 1 | -28.216 | -25.037 | -6.665 | -14.836 |
| 513_p2 | 8 | 0 | -35.184 | -32.005 | -16.110 | -11.471 |
| 513_p2 | 8 | 1 | -25.601 | -21.111 | -4.359 | -17.282 |
| 65_p2 | auto | 0 | -14.020 | +2.963 | +10.388 | +9.796 |
| 65_p2 | auto | 1 | -28.067 | -12.351 | -0.686 | -7.840 |
| 65_p2 | 8 | 0 | -12.821 | +5.039 | +20.292 | +20.725 |
| 65_p2 | 8 | 1 | -25.267 | -9.188 | -3.857 | +8.696 |
| flower_2000 | auto | 0 | -2.691 | -3.681 | +2.219 | +2.806 |
| flower_2000 | auto | 1 | -6.732 | -4.253 | +0.221 | -0.291 |
| flower_2000 | 8 | 0 | -14.963 | -16.894 | -4.120 | +2.857 |
| flower_2000 | 8 | 1 | -9.502 | -13.684 | -0.080 | +0.410 |
| flower_3200x2160 | auto | 0 | -18.845 | -21.760 | -6.099 | -0.213 |
| flower_3200x2160 | auto | 1 | -1.346 | +3.480 | +1.682 | -0.794 |
| flower_3200x2160 | 8 | 0 | -9.704 | -9.488 | +0.623 | +0.448 |
| flower_3200x2160 | 8 | 1 | -10.811 | -12.171 | -0.975 | -1.951 |
| flower_500 | auto | 0 | -13.289 | -7.429 | -2.876 | +2.564 |
| flower_500 | auto | 1 | -17.542 | -12.874 | -6.311 | +0.351 |
| flower_500 | 8 | 0 | -9.162 | -3.263 | +2.045 | +1.284 |
| flower_500 | 8 | 1 | -7.560 | -2.076 | +5.011 | +8.547 |
| keong_3839x2159 | auto | 0 | -7.304 | -7.771 | -0.627 | -0.374 |
| keong_3839x2159 | auto | 1 | -4.609 | -3.498 | -1.610 | +0.946 |
| keong_3839x2159 | 8 | 0 | -6.012 | -3.948 | +1.810 | -0.155 |
| keong_3839x2159 | 8 | 1 | -13.756 | -12.857 | -1.653 | -0.389 |

## Interpretation, regressions and next qualification

The dense ownership effect is visible in the intended CPU boundary. Against
adjacent direct output, token-writing work improves 32.690-46.834% across the
eight strong-dense groups. Against production it improves 21.767-35.184%, with
14.191-32.005% section-wall gains. All 32 dense whole-time cross-pairs against
each control are negative, not just the duplicate means. This supports the
bounded local-output mechanism under actual resident token/model diversity.
It does not isolate allocator behavior, stack layout or coherence traffic.

Dense whole-time gains versus production range from 2.124% to 16.110%.
Duplicate variation remains material: 1025/eight/pass 0 has original/local
whole duplicate deltas +13.014/+7.172 ms versus a primary -12.216 ms delta.
At 1025/auto/pass 0 the local duplicate difference is +4.984 ms versus a
primary -4.427 ms delta. These controls are neither subtracted nor discarded.

Photo-level gains are less uniform. The 4K aggregate improves 0.992%, but
eight threads/pass 1 loses 1.325% (+3.683 ms), with token work +0.081% and
section wall +4.004%. All four whole cross-pairs for that group lose. In
auto/pass 0, one of four whole cross-pairs loses even though the primary mean
wins. Keong has four token-work/section wins but only three whole wins; its
eight-thread/pass 0 loss is +1.810% (+4.893 ms). None is omitted because the
four-group photo aggregate passes.

For 4K, quantization median-mean deltas in auto R0/R1 then eight R0/R1 are
-2.888/-3.157/-4.314/+2.720 ms. Whole deltas are
-2.844/-5.334/-6.095/+3.683 ms. The CUDA modules are unchanged, but these
observations do not establish GPU clock, thermal, scheduling or causal
quantization effects. Phase medians are separately reduced and not additive;
the CPU ownership change cannot be credited with every observed whole delta.
The 4K/eight/R0 local duplicate difference of +9.747 ms is also larger than
its -6.095 ms primary whole delta and remains in the report.

The tiny 65x71 pattern-2 input is a real qualification concern. Its aggregate
whole time is 6.114% slower than production. Auto/eight in pass zero lose
10.388%/20.292%, while pass one wins 0.686%/3.857%. The largest loss is
10.100075 to 12.149625 ms (+2.049550 ms); all four cross-pairs lose in that
group. Token work improves in all four groups, which does not justify ignoring
the whole regression or attributing it to a specific unmeasured cause. This
case was not a separate veto in S192's prospective gate; no new criterion is
invented retrospectively to change its recorded pass/fail outcome.

The next step is production-quality integration of the private packed owner
and local AC output, with no experiment-specific names or compatibility layers.
The exact integrated implementation needs independent packed-bit/ANS checks,
public rollback and limit contracts, normal/ASAN fixtures and whole checks,
CUDA sanitizers, build/consumer qualification and a newly pinned resident
performance protocol. Preserve and investigate the tiny-input regression,
mixed photo groups and duplicate controls in that qualification. S192 passes
the 28-whole-win minimum by one group; its practical gate is not a confidence
interval or assurance that a new build will reproduce every gain.

S192 therefore records `eligible_for_production_qualification=true` and
`promote=false`. No production source changes are committed in this stage.
It does not add a pool, pad BitWriter, specialize by image or reclassify the
earlier failed owner, direct-append or scheduling campaigns.

## Execution and audit-reader corrections

| Job | UTC interval on 2026-09-10 | Outcome |
|---|---|---|
| Local DLL/fixture/compiler-control build | 13:39:53.265-13:40:47.818 | exit 0 |
| Normal AC section fixture | 13:45:40.486-13:45:41.077 | passed |
| ASAN AC section fixture | 13:45:41.103-13:45:42.162 | passed |
| Whole preflight, including CUDA sanitizers | 13:50:55.030-13:54:37.706 | 134 passed |
| Fixed six-process timing campaign | 14:00:16.865-14:23:11.296 | 264 passed |

There is one successful build child and no rebuild. The initial encoder probe
extracts the production encoder object successfully, then stops before writing
its comparison because its inherited normalizer assumes REL32 only. Version
two rechecks the existing extraction and explicitly handles the observed
ADDR32NB/SECREL fields as described above; it does not rewrite that object.

The first linked native auditor assumes added out-of-line move helpers remain
linked. Version two records their actual omissions, but then stops at a
mangled-name lookup: a JavaScript replacement string reduced a double-dollar
sequence to one dollar. Version three corrects that lookup with literal
replacement. Neither failed reader writes native result files or changes code,
objects or DLLs. The original readers remain available, and the final native
audit completes before whole preflight starts.

The first complete preflight verifier also stops on audit representation, not
encoder behavior: Python relocation tuples compare unequal to their serialized
JSON lists. Version two compares the exact JSON representation. Every field
then matches, all 35 native entries pass, and all 137 preflight-stage job records
reconcile. This correction occurs before timing and reruns no workload.

The source helper's newline literal was corrected before the source/protocol
pinning and build. There is no C++ correction after pinning, failed build,
encoder failure, changed benchmark policy or timing restart associated with
any of these reader/bookkeeping corrections.

## Reproduction and preservation

Artifacts are under `build-cuda-ninja/profiles/s192-artifacts`; versioned
`s192_*` helpers are beside them. The verifier reconstructs source overlays,
native comparisons, schedules, raw log checks, policy/duplicate labels, all
phase statistics, criteria and process journals. Commands, intent, launch PID,
terminal exit, log hashes and nonoverlapping job intervals must reconcile.

S191's 134, S190's 1,468 and S189's 190 frozen artifact files are checked
unchanged. The 343 production-source archives are reused, not recopied. Both
new and reused objects/DLLs, the unchanged harnesses, support/SDK inputs,
oracle files and output/storage census are hash-pinned. All failed reader
versions, final reader versions, raw samples, pair comparisons, duplicate
controls, source archives and final inventory are retained.

No machine power, clocks, cooling, affinity, priority, driver, firewall or
security settings are changed. No elevation/admin intervention, profiler,
recovery-volume access, cleanup or push is involved. Protected untracked notes
are not read, edited or staged. This experiment does not claim a new coefficient
format, measured peak allocation/DRAM traffic, coherence-counter evidence,
cross-platform qualification or completion of the broad resident-path goal.

C had 3,794,337,792 bytes free at the postmeasurement check; final free space
and inventory/source-record counts are retained in final_summary.json. The
complete source and artifact inventory is frozen after this report and decision
are written. No existing frozen artifact is deleted, moved or overwritten.

After freezing, verify read-only with:

```powershell
python build-cuda-ninja/profiles/s192_verify_v2.py --frozen
```

The broad resident-path optimization objective remains active. Passing this
diagnostic stage is progress toward integration, not evidence that the encoder
has been maxed out.
