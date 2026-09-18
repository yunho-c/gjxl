# S185: whole-encoder screen of packed reverse ANS bits

The private S183 packed writer passes whole-encoder correctness and its
prospective screen gate: 19/20 token-write worker wins and 13/20 whole wins.
Dense controls improve consistently; photo whole times remain mixed.
Broader qualification is warranted, but the writer is not promoted.
Production, GPU kernels, and scheduling remain unchanged.

## Scope and mechanism

Parent: `7db4ad47e9addb47844ea432556f55832c8ee666`, branch `feat/cuda`.
[S184](cuda-scratch-recovery-s184.md) recovered storage headroom by archiving
non-inventoried generated ETLs. This is a new campaign, not a resumption of
the interrupted S182 schedule. The underlying S182 `ENOSPC` cause remains
unresolved. Each S185 child requires at least 3,000,000,000 bytes free on C:
before launch. Intent, PID launch, and terminal exit are journaled separately;
the exit is also printed before log hashing and the final report write.

The [S183 prototype](cuda-reverse-bit-packing-s183.md) packs incoming reverse
ANS chunks into 56-bit words rather than retaining eight-byte records with
individual widths. The callback handles at most 31 extra bits and one
16-bit renormalization chunk per token. The final traversal writes reversed
full words plus the pending tail and final ANS state, retaining the temporary
BitWriter and atomic append. No raw stores or unchecked BitWriter writes
are added. Packing moves work into the recurrence callback; it does not
remove the recurrence, final traversal, output copy, or coefficient scan.

The initial staging reservation falls from 16 bytes per token to
`8 * floor(47 * tokens / 56)` bytes. For the conservative maximum AC group
of 199,680 tokens, this is 1,340,704 instead of 3,194,880 requested bytes,
about 58% less. This is a reservation bound, not measured peak memory or
evidence of a speedup. Allocator overhead and output buffers are excluded.

The diagnostic ANS source contains exact copies of the current original
writer body and the frozen S183 packed body, renamed and marked noinline.
Labels 0/1 call the same original function; 2/3 call the same packed function.
One common atomic slot allocation per writer call records success, token
count, and dispatch. There is no per-token diagnostic counter. Configuration
and reading the counters occur only outside joined encodes. Default host
fixtures select the packed path with recording disabled. Production files,
compatibility layers, libraries, and machine settings are not modified.

The source audit reconstructs this overlay exactly; all other ANS code is
unchanged, including model/configuration validation, HybridUint, recurrence,
and count-only paths. Its first pre-preparation invocation rejected one
missing blank line in the expected wrapper text. That expectation was fixed
before artifact creation or compilation; the C++ source was unchanged.

## Build, native audit, and correctness

New normal MSVC and clang AddressSanitizer objects/DLLs/executables link the
frozen support inputs. The ANS override replaces the old archive definition;
the ASAN link does not also include the S167 ANS override. Compile and link
are separate. Both DLLs contain the same 11 CUDA modules as the S169 frozen
production profile. No CUDA compilation is performed.

Unlinked normal original-writer instruction bytes match production.
24/25 normalized relocations match; the remaining reference names the
renamed callback's `ProcessAnsTokenStream` specialization. That specialization
also has identical instruction bytes (299 instructions, 1,175 bytes), with
15/16 normalized relocations matching and one renamed lambda reference.
The limited normalization does not erase mangled namespace backreferences;
both comparison reports preserve `normalized_relocations_equal=false`.
This is not a claim of complete linked-code identity, placement equivalence,
or neutral diagnostic overhead. Full ASAN object disassembly is retained.

The six normal/ASAN host runs use the existing independent ANS bit-emission,
entropy, and AC-section fixtures. Each bit-emission run checks 816 HybridUint
configurations, 30,240 exact comparisons, and 30,240 failure-atomicity cases.
The entropy fixture checks 443,904 value encodings, mapped/unmapped and
interleaved/offset-split layouts, and 40 late section failures. Each private
AC-section fixture checks 192 valid cases, 1,408 invalid models, 96 invalid
tokens, and 64 concurrent cases. The bit-emission fixture's reported
`predicted_batch_writes` remains the S167 reference calculation; it is not a
measurement of the new packer's write count.

Whole preflight covers all 11 established inputs, automatic/eight threads,
normal/ASAN: 44 processes. Each runs one S168 reference followed by four
diagnostic labels. Separate memcheck on 4K/eight and initcheck on Keong/eight
bring this to 46 processes and 230 whole encodes. Every diagnostic encode
checks exact bytes, the full `VarDctEncodingSummary`, four-byte coefficients,
native-owner storage, and identical ANS call/token/output-byte census across
labels. The parser additionally requires that census to agree across build
and thread configurations. Sanitizer times are not performance evidence.

## Prospective screen protocol

The frozen protocol selects 1080p, 4K, Keong, 513/pattern 2, and 1025/pattern 2,
each at automatic/eight threads. Ten cells are shuffled once; the second pass
reverses that order and the internal Williams order. The four orders are
0132, 1203, 2310, and 3021. Each process has eight warm and 16 measured
four-label rounds, plus one separate S168 reference: 20 processes, 1,940
whole calls, 1,280 measured calls, and 3,840 timing NVML endpoints.

The fully resident workload remains effort 7, distance 1.2 (pattern 2 uses
0.01), with final score disabled. Every Encode prepares a fresh workflow
while retaining the backend. Reference setup/unload, configuration, result
clearing, validation, and NVML capture are outside the outer timer; all
encoder work is inside. No build, sanitizer, profiler, or heavy verification
hashing overlaps timing. All samples are retained, with no reruns, filtering,
or optional performance stopping. All warm/measured configured-power
endpoints must be 40,000 mW. This does not fix instantaneous GPU clocks,
temperature, actual power draw, or operating-system activity.

Primary comparison: median within-round difference between candidate-label
mean and original-label mean. All 41 phase fields, all four individual
candidate/control pairs, and both same-path duplicate differences are
retained. Worker durations are nested work, not additive wall time.
The prospective broader-follow-up gate requires at least 16/20 token-write
worker wins (stage 33) and 12/20 whole-encoder wins. The screen itself cannot
authorize promotion. Before timing, an addendum pins `s185_analyze_v2.py`,
correcting stale S181 variable names in the unused initial analyzer without
changing the protocol or thresholds. The original helper is retained.

The photo labels 1080p and 4K mean 1919x1079 and 3839x2159. Some photos are
derived inputs, not independent native-resolution captures. Pattern controls
are 513x519 and 1025x1031.

## Results and decision

The fixed campaign completed with 73 accepted journaled jobs: one build,
six host fixtures, 46 whole preflight/sanitizer processes, and 20 timing
processes. All 2,170 whole calls passed, including the 1,280 measured calls.
All 4,208 warm/preflight/measured NVML endpoints reported 40,000 mW.
Memcheck reported zero errors and zero leaked bytes/allocations; initcheck
reported zero errors. No failed child, resource abort, campaign retry,
filtered sample, privilege prompt, or firewall intervention occurred.

On 2026-09-10 (UTC), the build ran 10:32:01.182–10:33:27.756 and produced
34 files totaling 34,324,288 bytes. Whole preflight ran
10:35:29.219–10:40:13.543. The fixed screen ran
10:40:41.009–10:45:48.410. The initial source-audit text rejection happened
before the journaled build and is not counted as a child-process failure.

Negative values below favor packing. Percentages are medians of the
within-round candidate/control ratios, not ratios of the displayed median
durations; deltas are medians of within-round differences.

| Input | Threads | Pass | Whole delta (ms) | Whole (%) | Token-write work (%) | Section wall (%) |
|---|---:|---:|---:|---:|---:|---:|
| 1080p | auto | 0 | +0.7162 | +0.903 | -3.392 | -2.550 |
| 1080p | auto | 1 | +0.1164 | +0.152 | -3.572 | -2.283 |
| 1080p | 8 | 0 | +0.5252 | +0.716 | -2.968 | +1.753 |
| 1080p | 8 | 1 | -0.1123 | -0.182 | -1.428 | -2.114 |
| 4k | auto | 0 | -0.4360 | -0.180 | -3.535 | -4.403 |
| 4k | auto | 1 | +0.0606 | +0.015 | -4.565 | -4.669 |
| 4k | 8 | 0 | +6.4766 | +2.453 | +2.987 | +9.984 |
| 4k | 8 | 1 | -0.5357 | -0.204 | -8.759 | -9.085 |
| keong_3839x2159 | auto | 0 | -11.2057 | -3.738 | -4.741 | -3.207 |
| keong_3839x2159 | auto | 1 | +2.5294 | +0.875 | -5.546 | -0.695 |
| keong_3839x2159 | 8 | 0 | -7.6230 | -2.494 | -1.304 | +2.780 |
| keong_3839x2159 | 8 | 1 | +3.1813 | +1.094 | -0.125 | -1.624 |
| 513_p2 | auto | 0 | -1.5750 | -4.896 | -24.960 | -21.069 |
| 513_p2 | auto | 1 | -1.2195 | -3.656 | -21.258 | -17.756 |
| 513_p2 | 8 | 0 | -3.0302 | -8.459 | -26.793 | -21.610 |
| 513_p2 | 8 | 1 | -2.3782 | -6.726 | -24.609 | -21.653 |
| 1025_p2 | auto | 0 | -4.7521 | -4.870 | -16.849 | -16.178 |
| 1025_p2 | auto | 1 | -2.9171 | -3.190 | -14.697 | -12.079 |
| 1025_p2 | 8 | 0 | -1.6409 | -1.716 | -17.840 | -16.341 |
| 1025_p2 | 8 | 1 | -2.2449 | -2.310 | -16.750 | -15.451 |

The precommitted gate **passes**: 19/20 token-write worker wins and 13/20
whole wins, versus required 16 and 12. Section wall improves in 17/20.
The outcome is permission for a separately precommitted broader
qualification, **not production promotion**.

The dense controls account for eight whole wins out of eight. On 513/p2,
token-write work falls 21.26–26.79%, section wall 17.76–21.65%, and whole
time 3.66–8.46%. On 1025/p2 the corresponding reductions are
14.70–17.84%, 12.08–16.34%, and 1.72–4.87%. These are per-cell screen
results, not a pooled universal speedup.

Photos are materially less conclusive: only 5/12 whole cells improve,
although token-write work improves in 11/12. All four 1080p worker deltas
are negative, but only one whole delta is negative. 4K/eight/pass 0 is
slower by 6.4766 ms (2.453%); its unchanged quantization phase is slower
by 8.0426 ms, while token-write worker work also rises 2.987%.
Both second-pass Keong whole cells are slower, reversing their first-pass
direction. These observations do not prove either an intrinsic downstream
regression or a GPU cadence/thermal explanation.

Same-path duplicate noise is retained rather than subtracted or filtered.
For example, Keong/eight/pass 0's original and candidate whole duplicate
deltas are +13.4873 and -9.2894 ms, respectively, alongside the primary
-7.6230 ms comparison. 4K/eight/pass 1's duplicates are +9.4034 and
+3.9762 ms versus a primary -0.5357 ms. Small photo whole deltas should not
be treated as established improvements.

The call/token census is identical across all labels, builds, and thread
settings. Screen inputs have 43/338,430 calls/tokens at 1080p;
144/1,131,618 at 4K; 144/2,102,398 at Keong; 12/836,402 at 513/p2;
and 28/3,294,564 at 1025/p2. These include other ANS streams, not just
coefficient groups. The eleven-case census and exact output-byte counts
are retained in `preflight.json`.

## Preservation and next step

Evidence is under `build-cuda-ninja/profiles/s185-artifacts`; versioned
`s185_*` helpers sit beside it. Raw logs, separate process journals,
frozen source/support inputs, disassemblies, CUDA module hashes, exact
parsers, all-phase analyses, and the decision are retained. The verifier
reparses every whole log, checks schedule and command identity, reconciles
all 73 process records, recomputes analysis, and rehashes the prior S183
91-file and S184 70-file inventories. The 343 production-source archives
are reused from S182. The final inventory and source-record counts are
recorded by the freezer rather than guessed in advance.

No production code, libraries, GPU kernels, power/clock/cooling/priority/
affinity settings, firewall/security settings, or drivers were changed.
No artifact was deleted, no recovery volume used, and nothing was pushed.
The protected untracked notes were not read, edited, or staged.

Next: separately qualify fixed-policy builds without this diagnostic
selector/counter scaffolding, cover all eleven inputs at both thread
settings, and retain focused photo comparisons alongside dense controls.
Do not combine another optimization, reuse this screen as promotion
evidence without broader checks, or silently rerun its cells. Full CTest,
ThreadSanitizer, allocation-fault injection, measured peak-memory reduction,
and a causal explanation for the photo timing variation remain unclaimed.
The resident-encoder optimization goal remains active; it is not maxed out.
