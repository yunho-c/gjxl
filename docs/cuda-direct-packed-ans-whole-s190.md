# S190: whole-encoder qualification of direct packed ANS

Date: 2026-09-10. Parent: `262a22f6dc79363500b1ab06af4ed0315d4d93d1`.
This stage compares current production against the growable packed owner with
direct checked append from [S189](cuda-direct-packed-ans-append-s189.md).
Correctness passes, but the candidate fails four of six prospective performance
criteria: 21/44 whole wins and a 4.738% dense-control regression. It remains
unpromoted; production is unchanged. Concurrent adjacent writer metadata is
the next bounded investigation, not an established explanation.

## Isolated candidate and controls

S189 passed its synthetic actual-ANS screen, with 48/48 primary wins and a
0.805166 large-stream geometric ratio versus S183's packed temporary-writer
path. That result was not a whole-image speedup. S190 instead compares the
combined packed representation and direct append against current production.
It does not combine the fixed-capacity owner rejected in S187/S188.

The candidate packs reverse-generated chunks into 56-bit words, retaining a
pending tail below 56 bits. It reserves the existing proven 47-bits-per-token
upper bound. The final state, tail and reversed words are emitted directly
inside the destination's existing `WithMaxBits` transaction, through checked
`WriteBits` calls. No temporary BitWriter or final Append copy is needed.
The arithmetic, ordering, lifetime and rollback proof is in S183/S189.
There is no BitWriter header, implementation, layout or ABI change, raw store,
fixed owner, public API, compatibility layer, selector or counter.

No C++ is compiled in this stage. Fresh normal/ASAN original/direct whole DLLs
link the exact frozen S186 original and S189 candidate ANS objects. The sealed
S186 whole harnesses and the same frozen bridge/support libraries are reused.
Normal DLLs use the current S168 BitWriter archive definition, not an override.
All four DLLs have the same eleven CUDA module hashes as production.
No coefficient ownership, composition/reduction or GPU tile policy changes.

Labels 0/1 use the same original DLL and 2/3 the same direct-packed DLL.
Each process uses one fixed policy. The old harness's S186 log tags are kept;
they identify the reused harness, not an older candidate. It first loads and
encodes the frozen S168 reference, checks any external oracle, then destroys
and unloads that reference before loading the measured DLL. There is no
simultaneous original/candidate backend residency.

Each checked Encode prepares a fresh workflow while retaining the backend.
The fully resident workload uses effort 7, distance 1.2 (pattern 2: 0.01),
and no final score. Every call checks exact output bytes, the complete
`VarDctEncodingSummary`, four-byte coefficients, native-owner storage bytes,
and the S185 census output size. Reference/setup/unloading, result clearing,
validation and read-only NVML capture are outside the outer timer; all Encode
work is inside. All 41 phase fields are retained. Nested worker work is not
additive wall time.

## Complete native audit

The normal current-original writer, recurrence and callback match the actual
current production archive's complete code COMDATs after masking only REL32
relocation fields and normalizing anonymous-namespace symbol hashes. The
direct writer, recurrence, callback and owner Append match the frozen tested
S189 object. Linked PE bytes and actual call destinations are checked, not
just disassembler labels or instruction counts.

| Selected normal native entry section | Original instructions / bytes | Direct instructions / bytes |
|---|---:|---:|
| Writer | 362 / 1552 | 192 / 812 |
| Recurrence | 299 / 1175 | 289 / 1140 |
| Chunk callback | 16 / 59 | 51 / 170 |
| Packed owner Append | not applicable | 90 / 401 |

These are complete selected entry sections, not all callees or separately
outlined exception handlers and not executed instruction counts. In particular,
the direct owner Append delegates emission to its transaction lambda.
The original writer count corrects S186's partial 324/1387 report; no old
artifact is rewritten. The complete-COMDAT method follows S188's corrected
auditor, avoiding symbol-only disassembly truncation at local labels.

The initial S190 audit stopped at an incorrect expectation of two outlined
original callback calls. The actual original recurrence has one, while the
direct recurrence has two. A new versioned auditor uses that observed boundary
and checks the actual resolved targets. It byte-revalidates the already
extracted production ANS and BitWriter objects without replacing them.
There was no build, source or workload correction, and no campaign rerun.

The current production BitWriter's complete PrepareWrite, WriteBitsUnchecked,
WriteBits, Append and WithMaxBits sections and normalized relocations also match
the BitWriter object used by S189's CPU fixtures. Both full ASAN ANS object
disassemblies are retained and hash-pinned; native equivalence is claimed for
the normal audit, not ASAN instrumentation or execution conditions.

## Correctness and prospective protocol

S189's frozen host qualification is reused, not rerun or recounted as new
S190 jobs. It includes 410,208 normal/ASAN exact primitive comparisons,
615,312 rollback checks, eight host fixtures and 1,152 sealed DLL checks.
Source reconstruction, exact object hashes and the BitWriter native audit
connect those tests to the new whole DLLs.

New whole preflight covers eleven inputs, automatic/eight threads,
original/direct, normal/ASAN: 88 processes. Each performs one reference and
one checked encode. Candidate CUDA memcheck on 4K/eight and initcheck on
Keong/eight bring this to 90 processes and 180 whole calls. Sanitizer times
are not performance evidence.

The fixed timing matrix is specified before preflight: eleven inputs,
automatic/eight threads, two passes, or 44 quartets and 176 processes.
Twenty-two case/thread cells are shuffled once with seed 19020260911.
Pass 0 alternates label orders 0/2/3/1 and 2/0/1/3 (ABBA/BAAB); pass 1
reverses the cell order and reverse-complements each label order. Each label
occupies every quartet position eleven times. The independent preflight
shuffle seed is 19020260910.

Eight warm and 32 measured encodes plus one unloaded reference per process
give 7,216 timing whole calls, including 5,632 measured calls. For every
field, first take each process's median, then subtract the mean of original
duplicate medians from the mean of direct duplicate medians. Samples with
the same ordinal number in different processes are not contemporaneous pairs.
All four cross-pair comparisons and both same-path duplicate differences
are retained. Whole-time geometric ratios weight quartets equally; every
input has four quartets. No samples are filtered or silently rerun.

The prospective gate is unchanged from S186 and requires all six criteria:

| Criterion | Requirement |
|---|---:|
| Token-write worker wins, stage 33 | at least 35/44 |
| Whole-encoder wins | at least 28/44 |
| AC-section wall wins, stage 31 | at least 30/44 |
| Each photo's geometric whole-time ratio | at most 1.01 |
| All photo quartets' geometric whole-time ratio | at most 1.005 |
| Strong dense controls' geometric whole-time ratio | at most 0.98 |

This is a practical qualification gate, not statistical significance or
automatic promotion. The six photos are flower/500, 1080p, flower/2000, 4K,
flower/3200x2160 and Keong. Photo resizes use nearest replication; these are
not six independent native-resolution captures. 1080p/4K mean 1919x1079 and
3839x2159. Strong dense controls are 513/p2 and 1025/p2 (eight quartets);
other controls are 65/p2 and 513/p0/p1. Control dimensions are 65x71,
513x519 and 1025x1031.

All 14,080 timing NVML endpoints must report an enforced 40,000 mW limit.
This does not lock clocks, thermals or actual power. A read-only NVML check
verified the enforced limit even though the nvidia-smi power.limit query
reported N/A. No setting was changed. No build, sanitizer, profiler or heavy
artifact verification overlaps timing; routine per-job log hashing remains.
Every launch requires at least 3 GB free on C and has separate durable intent,
PID launch, terminal exit and final report journals. A resource abort is not
permission to restart the fixed campaign.

## Results and decision

All 267 journaled jobs are accepted: one link build, 90 whole preflight/
sanitizer processes and 176 timing processes. All 7,396 whole calls pass,
including 5,632 measured calls. Memcheck reports zero errors and zero leaked
bytes/allocations; initcheck reports zero errors. All 14,260 preflight/warm/
measured NVML endpoints report 40,000 mW. No workload rerun, filtered sample,
resource abort or privilege/firewall intervention was needed.

**The candidate fails four of six prospective criteria and stays unpromoted.**
The photo aggregate passes, but it cannot override the worker, whole, section
and strong-dense failures. Neither S186's earlier gate nor this gate is relaxed.

| Criterion | Actual | Result |
|---|---:|---|
| Token-write worker wins | 31/44 | fail; requires 35 |
| Whole-encoder wins | 21/44 | fail; requires 28 |
| AC-section wall wins | 25/44 | fail; requires 30 |
| Largest photo geometric ratio | 1.006305 (flower/3200x2160) | pass |
| All photo geometric ratio | 0.993956 | pass |
| Strong dense geometric ratio | 1.047382 | fail; requires at most 0.98 |

The all-input geometric ratio is **1.012356588**, a **1.236% regression**.
Photos aggregate to -0.604%, with 14/24 whole wins. Strong dense controls
aggregate to +4.738%, with only 1/8 whole wins. These are equal-weight cohort
summaries, not universal effects or significance claims.

Negative values below favor direct packing. Each percentage compares the
candidate and original duplicate means of process medians.

| Input | Threads | Whole R0 (%) | Whole R1 (%) | Token work R0 (%) | Token work R1 (%) | Section R0 (%) | Section R1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flower_500 | auto | -4.152 | -8.337 | -4.911 | -15.005 | -3.964 | -13.045 |
| flower_500 | 8 | -2.997 | +2.224 | -9.547 | -5.474 | -7.353 | -3.136 |
| 1080p | auto | -1.081 | -0.769 | -4.853 | -10.322 | -3.475 | -8.578 |
| 1080p | 8 | -2.576 | +3.028 | -13.658 | -0.183 | -10.253 | +6.929 |
| flower_2000 | auto | -3.376 | +1.694 | -12.870 | -5.349 | -16.669 | -4.528 |
| flower_2000 | 8 | +0.900 | -0.180 | -3.665 | -9.586 | -5.889 | -12.367 |
| 4k | auto | -0.784 | +0.011 | -5.905 | -2.894 | -5.164 | +0.282 |
| 4k | 8 | -0.304 | +1.223 | -4.332 | -1.736 | -1.178 | +0.621 |
| flower_3200x2160 | auto | -0.958 | -1.742 | -8.896 | -6.254 | -11.995 | -6.419 |
| flower_3200x2160 | 8 | +1.000 | +4.329 | -2.440 | +0.080 | -1.548 | +3.990 |
| keong_3839x2159 | auto | -0.617 | -1.068 | -2.226 | -9.405 | -2.169 | -10.729 |
| keong_3839x2159 | 8 | +0.569 | +0.240 | -4.362 | -5.007 | -10.198 | -10.607 |
| 65_p2 | auto | +6.448 | +12.694 | -20.430 | -15.116 | -2.145 | +1.680 |
| 65_p2 | 8 | +16.009 | -4.135 | -9.771 | -23.886 | +8.652 | -13.243 |
| 513_p0 | auto | +5.257 | +8.533 | +2.350 | +15.004 | +8.086 | +16.522 |
| 513_p0 | 8 | -3.406 | -0.497 | -6.419 | -3.223 | -1.088 | -1.858 |
| 513_p1 | auto | -0.691 | +0.679 | +3.003 | +5.417 | +1.488 | +2.052 |
| 513_p1 | 8 | -1.122 | -5.333 | +3.271 | -2.316 | +0.402 | -0.704 |
| 513_p2 | auto | +8.544 | -4.751 | +15.182 | -5.298 | +31.327 | +11.680 |
| 513_p2 | 8 | +5.688 | +0.855 | +13.866 | +1.474 | +23.752 | +5.073 |
| 1025_p2 | auto | +2.984 | +7.028 | +12.837 | +14.321 | +14.734 | +23.771 |
| 1025_p2 | 8 | +8.806 | +9.581 | +18.526 | +20.072 | +26.138 | +26.141 |

| Input | Geometric whole change (%) | Whole wins / 4 |
|---|---:|---:|
| flower_500 | -3.388 | 3 |
| 1080p | -0.371 | 3 |
| flower_2000 | -0.259 | 2 |
| 4k | +0.034 | 2 |
| flower_3200x2160 | +0.630 | 2 |
| keong_3839x2159 | -0.221 | 2 |
| 65_p2 | +7.473 | 1 |
| 513_p0 | +2.364 | 2 |
| 513_p1 | -1.642 | 3 |
| 513_p2 | +2.458 | 1 |
| 1025_p2 | +7.069 | 0 |

The largest dense control, 1025/p2, regresses in all four whole comparisons:
+2.8483 to +8.9728 ms (+2.984% to +9.581%). Its token-write work increases
12.837-20.072% and section wall time increases 14.734-26.141%. This repeated
CPU-stage regression matters despite the synthetic CPU win in S189.
513/p2 has three whole losses and one win; all four section-wall comparisons
are slower, including the whole-winning auto/pass-1 cell.

Four 4K token-work comparisons improve 1.736-5.905%, but whole time improves
only in pass 0. In campaign order, whole deltas are -2.0352, -0.7892,
+3.3193 and +0.0308 ms; quantization deltas are +1.0184, +2.6922,
-0.3492 and -0.5074 ms. These separately reduced medians cannot be added or
subtracted to infer a missing cost or a causal GPU cadence explanation.
Keong similarly has four token-work and section-wall wins but two whole wins.

Same-path duplicate differences remain visible. For 1025/eight/pass 1,
original/direct whole duplicate deltas are -3.8218/-1.4028 ms versus a
+8.9728 ms primary regression; token-work duplicates are -2.3000/-9.2819 ms
versus +30.0703 ms. For 1025/auto/pass 0, whole duplicates are
-7.1172/-2.1038 ms versus +2.8483 ms. For 4K/auto/pass 1 they are
+2.3560/+4.9202 ms versus +0.0308 ms. These controls are not discarded,
subtracted from the results or used to excuse the failed gate. No causal
claim about clocks, temperature, RDP or other machine activity is established.

## New bounded concurrency lead

A postmeasurement source review identifies an important difference from the
single-threaded S188/S189 actual-ANS screen. `WriteAcSections` creates a
contiguous `std::vector<BitWriter>` and passes `&candidate[1 + index]` to
concurrent group writers. Production first emits into a worker-local temporary
and publishes with Append. Direct packed output instead invokes checked writes
on those adjacent destination objects throughout emission. PrepareWrite changes
the vector metadata, and WriteBitsUnchecked updates the writer's bit counter
inside its bytewise loop. BitWriter has no cache-line isolation.

Repeated writes to nearby writer metadata therefore present a plausible
false-sharing mechanism. This is a source-derived hypothesis, **not measured
cache-line contention or a proven cause of S190's regressions**. The synthetic
CPU screen lacks concurrent adjacent destinations and cannot settle it.
This lead also avoids assuming that the small synthetic models faithfully
represent the resident encoder's token/model distributions.

The next bounded experiment should compare adjacent destination writers with
worker-local output followed by one move publication, keeping the ANS object,
input/model data, output bytes and scheduling protocol fixed. Test one and
multiple workers with exact oracle and normal/ASAN error checks. Retain
original/growable controls to distinguish a general concurrency cost from an
interaction with direct append. A move of completed ownership is not a byte
copy and needs no BitWriter ABI change or blanket cache-line padding.

Such a test must preserve section ordering, failure publication, lifetime,
join/error behavior and allocation handling. Instrumented/captured workloads
would be diagnostic only, not production performance evidence. If this
mechanism is not reproduced, capture and replay actual resident ANS groups
before further output-loop changes. Do not substitute a photo-specific policy,
rerun S190 or promote based on its passing photo aggregate.

## Execution and preservation

| Job group | UTC interval on 2026-09-10 | Outcome |
|---|---|---|
| Four-DLL link build | 12:34:24.222-12:34:36.247 | exit 0 |
| Whole normal/ASAN preflight | 12:40:04.457-12:41:28.553 | all accepted |
| CUDA memcheck | 12:41:28.572-12:42:20.615 | zero errors/leaks |
| CUDA initcheck | 12:42:20.639-12:42:59.181 | zero errors |
| Fixed timing matrix | 12:45:01.658-13:00:09.237 | all 176 accepted |

The link build produced 14 files totaling 43,319,098 bytes. The completed
source/native audit and host-qualification reuse checks preceded preflight.
The native-auditor expectation correction was before any whole preflight;
no C++ source, object or DLL was overwritten or recompiled. An oversized
postmeasurement report exceeded the tool response budget; compact read-only
extraction recovered the needed reporting fields from the intact analysis.
It did not rerun any workload or replace an artifact.

Artifacts are under `build-cuda-ninja/profiles/s190-artifacts`, with versioned
`s190_*` helpers beside them. The verifier reparses every whole log, rebuilds
the fixed schedule/commands and analysis, reconciles all durable process
journals and nonoverlapping intervals, checks complete native code and CUDA
module hashes, and verifies source/support/SDK/input integrity. S189's frozen
190-file inventory is unchanged. The 343 production-source archives from
S182 are reused and rehashed. All raw samples, phases, four cross-pairs,
duplicates, initial/corrected native helpers and final inventories are kept.

At the postmeasurement check C had 3,945,267,200 bytes free and U had
30,883,840. Final space and artifact/source counts are recorded in the frozen
summary and inventory. No old data was deleted, moved or overwritten; no
recovery volume was used. Power, clocks, cooling, affinity, priority, driver,
security and firewall settings were not changed. No elevation, profiler or
push was used. Protected untracked notes were not read, edited or staged.

After freezing, verify read-only with:

```powershell
python build-cuda-ninja/profiles/s190_verify.py --frozen
```

This stage does not claim full CTest, installed-consumer or cross-platform
qualification, decoded-quality coverage beyond exact frozen outputs,
allocation-fault injection, TSAN/MSAN, measured peak-memory savings or causal
hardware-counter attribution. Production remains unchanged and the broader
resident-encoder optimization goal remains active.
