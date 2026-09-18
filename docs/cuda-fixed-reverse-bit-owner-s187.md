# Fixed-capacity reverse-bit owner: S187

Date: 2026-09-10. Parent: `d842bda7a70838841c89a840bbbe0a2c5d82fa92`.
Production is unchanged. This is a private CPU-only experiment, not a promotion.

## Decision

The fixed-capacity owner passed the primitive and integrated host correctness
checks and removed the growable owner's native ANS callback growth path.
Its raw packing-plus-append screen won **13/60 cells**, below the prospectively
specified **48/60** threshold. The gate failed; do not launch a whole-encoder
campaign or promote this version on this evidence.

A postmeasurement native audit identified an important scope distinction:
the raw benchmark inlines packing into its loop, whereas integrated ANS invokes
a separately emitted callback. The screen therefore does not directly measure
the observed 51-to-33-instruction callback boundary. This does not overturn the
failed gate, prove an integrated ANS regression, or explain the timings.

## Source hypothesis and change

[S186](cuda-packed-ans-fixed-policy-s186.md) qualified the growable
[S183](cuda-reverse-bit-packing-s183.md) packer but missed its whole-encoder
promotion gate. Its native ANS callback retained a vector-growth relocation even
though the enclosing writer reserves a sufficient worst-case capacity.
That is a code-generation opportunity, not evidence of observed reallocations.

The S187 overlay changes only the owner and its callback exception declaration:

- Allocate `floor(upper_bits / 56)` uninitialized `uint64_t` elements once,
  when capacity is nonzero; keep capacity, owned pointer, and used count separate.
- Replace `vector::push_back` by an indexed store followed by incrementing the
  used count. `PushValidated` is allocation-free and `noexcept`.
- Keep the same 56-bit packing arithmetic, pending bits, ANS state placement,
  and checked temporary-`BitWriter` append. Reverse traversal reads only the
  initialized used prefix.
- Disallow copy and move: this is a one-shot private owner, and no moved-from
  object with a null pointer and stale nonzero count can be consumed.
- Retain the writer's `N <= SIZE_MAX / 47` guard, outer allocation/error catches,
  ANS recurrence, model checks and validation. No force-inline annotations,
  selectors, counters, compatibility layer or GPU changes are added.

The constructor intentionally omits value initialization of the integer array;
unused worst-case slack is not zeroed and must never be read. This replaces
storage ownership, not the packed representation or its capacity estimate.
S183 and S187 request the same number of payload elements. There is no additional
claimed peak-memory saving over S183 and no allocation-failure injection.

The whole ANS source is reconstructed exactly from S183 with just the include,
owner construction, and callback `noexcept` substitutions. A separate post-timing
header audit normalizes comments/whitespace and reconstructs the exact push and
append bodies after the indexed-store/loop and error-label substitutions.

## Bounds and qualification

Let a running payload prefix be `P`, incoming chunk width `b <= 31`, and the
constructor bound be `U >= P + b`. The invariant is:

`used = floor(P / 56)`, `pending_bits = P % 56`,
`capacity = floor(U / 56)`.

A push emits at most one word. If it emits, then
`used + 1 = floor((P + b) / 56) <= capacity`, so the preincrement write index
is strictly below capacity. The store initializes the element before the used
count increments; append reads only indices below that count. These are
single-owner sequencing statements, not a concurrent publication protocol.
With cumulative payload bounded by `SIZE_MAX`, payload bookkeeping cannot wrap.
The 47-bit-per-token ANS bound is unchanged. Invalid internal callers that exceed
the constructor bound are outside this unchecked private primitive's contract.

The arithmetic model checked 28,672 operand-pattern combinations across all
56 pending widths and 32 input widths, plus 655,520 capacity transitions
(prefixes 0..4096, widths 0..31, and slack 0/1/55/56/4096).
It also checked 32- and 64-bit `size_t` limits for the reservation multiplication
and array byte count. This is not exhaustive enumeration of all 86-bit combined
values and is not 32-bit binary qualification.

The independent bit-at-a-time primitive fixture repeats the same S183 seed and
cases with reservation slack 0, 55, 56 and 4096 bits. Each normal/ASAN build passed:

| Check | Per build |
|---|---:|
| Exact byte/length comparisons | 410,208 |
| Insufficient-limit rollback checks | 410,208 |
| Pushed chunks | 24,324,352 |
| Emitted full words | 6,840,776 |
| Payload bits | 394,062,808 |
| Exact-boundary flushes | 454,680 |
| Split flushes | 6,386,096 |

All 32 chunk widths, 56 pending widths, eight destination bit tails, exact limits,
null destination handling and four reservation bounds are covered. The fixture
checks used size against capacity after every push. Combined normal/ASAN totals
are 820,416 exact comparisons and 820,416 rollback checks. ASAN does not replace
an uninitialized-read detector; the initialized-prefix source proof matters.

All three existing integrated host fixtures passed in both normal and ASAN builds:

- ANS bit emission: 816 configurations, 30,240 byte comparisons and 30,240
  atomic-failure checks per build. Its logged `predicted_batch_writes` is the
  existing reference calculation, not a measured S187 write count.
- Entropy: 443,904 value encodings over 816 configurations, mapped/unmapped
  and interleaved/offset-split scans, plus 40 late section failures.
- AC section validation: 192 valid cases, 1,408 invalid models, 96 invalid-token
  cases and 64 concurrent cases per build.

MSVC 14.37 supplied normal builds; clang-cl/LLVM 22 supplied ASAN builds.
Own newly compiled ANS and current BitWriter objects override the frozen support
libraries. Normal uses S168 codestream and S166 support; ASAN uses S162 support
with the S166 encoder object except for the private AC fixture that includes
encoder source itself. Support files and SDK inputs are hash-verified.
No allocation-fault, MSAN, TSAN, CUDA execution or whole-image qualification
was performed for S187.

## Native audit

Counts below are static instructions and encoded instruction bytes, not
executed counts or cycle predictions.

| Integrated ANS function | S186 growable instructions / bytes | S187 fixed instructions / bytes |
|---|---:|---:|
| Writer | 154 / 645 | 128 / 530 |
| Recurrence | 289 / 1140 | 289 / 1140 |
| Callback | 51 / 170 | 33 / 108 |

The growable callback has one `_Emplace_reallocate` relocation. The fixed callback
has no growth relocation and no call instruction. The recurrence's instruction
bytes and all 16 relocation triples match after normalizing only the
anonymous-namespace identifier; this is stronger than matching instruction counts.
The changed callbacks are not claimed equivalent native code.

Separately, the actual raw benchmark's `VectorPack` is 118 instructions / 445 bytes
and `FixedPack` is 89 / 336. Both inline the packing operation. The former retains
vector reserve/growth targets; the latter retains array allocation/deallocation.
Neither has a `PushValidated` target or the integrated ANS callback target.
The benchmark audit was performed after measurement without changing or rerunning
the benchmark. Native instruction reduction alone did not predict its result.

## Fixed CPU screen

The protocol was frozen before the benchmark build and run:

- 30 datasets: 1,024 / 65,536 / 262,144 raw chunks, five width patterns
  (1, 16, 31, alternating 31/16, deterministic random 1..31), and two reservation
  bounds (exact payload or `47 * chunk_count`).
- Two passes through all datasets, forward then reverse: 60 cells total.
  The 47C synthetic bound is safe but is not measured real-ANS token metadata.
- Four labels: 0/1 share the same growable function pointer; 2/3 share the same
  fixed function pointer. Four warm and 16 measured rounds per cell.
  Williams orders 0132 / 1203 / 2310 / 3021 cycle, with the order index reversed
  in pass one. No sample filtering, rerun or performance-based early stopping.
- Each call is independently checked against bit-at-a-time expected bytes and
  exact bit length, including a destination prefix. All 4,800 calls passed;
  3,840 were measured.
- Timing includes staging allocation, packing, checked append and staging-owner
  destruction. Input generation, oracle construction, output setup/validation,
  sample logging and final output destruction are outside the timed interval.
  No ANS recurrence/model work, GPU work, NVML collection or clock locking.
- Primary delta is the median across rounds of the fixed duplicate mean minus
  the growable duplicate mean. Percent change is the median within-round ratio
  minus one. These are not differences/ratios of independently computed medians.
  All four cross-pair comparisons and both same-path duplicate deltas are retained.

The prospective gate was at least 48 primary wins among 60 cells, permitting only
separately qualified whole-encoder follow-up, not promotion. Actual wins were
5/30 in pass zero and 8/30 in pass one: **13/60, gate failed**.
This is a screening rule, not a statistical significance test; correlated cases
and nanosecond-scale deltas must not be treated as independent efficacy evidence.

Complete cell table follows. Negative means fixed is faster; `C` is chunk count.
Percent and ns columns use their separately defined within-round statistics.

| Case | C | Width pattern | Bound | R0 change % | R0 delta ns | R1 change % | R1 delta ns |
|---:|---:|---|---|---:|---:|---:|---:|
| 0 | 1024 | 1 | exact | +2.5641 | 50 | +10.8889 | 425 |
| 1 | 1024 | 1 | 47C | -2.0408 | -50 | +11.8298 | 600 |
| 2 | 1024 | 16 | exact | +0.3236 | 50 | +0.2679 | 75 |
| 3 | 1024 | 16 | 47C | +0.1618 | 25 | +0.6255 | 200 |
| 4 | 1024 | 31 | exact | +0.5425 | 150 | -0.5513 | -300 |
| 5 | 1024 | 31 | 47C | +0.6336 | 175 | +0.7402 | 375 |
| 6 | 1024 | 31/16 | exact | +0.4603 | 100 | +2.0729 | 750 |
| 7 | 1024 | 31/16 | 47C | +0.6803 | 150 | +0.3132 | 75 |
| 8 | 1024 | random | exact | +1.9171 | 175 | +2.2539 | 225 |
| 9 | 1024 | random | 47C | +0.8982 | 150 | +0.5548 | 100 |
| 10 | 65536 | 1 | exact | +0.7670 | 825 | +1.1599 | 1300 |
| 11 | 65536 | 1 | 47C | +0.3663 | 450 | +0.6390 | 725 |
| 12 | 65536 | 16 | exact | +0.8255 | 9825 | +1.2015 | 12200 |
| 13 | 65536 | 16 | 47C | +0.4342 | 5550 | -1.2010 | -12450 |
| 14 | 65536 | 31 | exact | +0.2664 | 6075 | -5.0095 | -129775 |
| 15 | 65536 | 31 | 47C | +0.0041 | 75 | +0.7661 | 18400 |
| 16 | 65536 | 31/16 | exact | +0.5485 | 5675 | +0.8591 | 5350 |
| 17 | 65536 | 31/16 | 47C | -0.1461 | -2600 | -1.7852 | -38325 |
| 18 | 65536 | random | exact | +1.1823 | 15275 | -0.0313 | -500 |
| 19 | 65536 | random | 47C | +1.1837 | 15925 | -1.1401 | -15125 |
| 20 | 262144 | 1 | exact | +0.1983 | 975 | +0.6745 | 3275 |
| 21 | 262144 | 1 | 47C | +1.5490 | 8125 | +1.8201 | 8900 |
| 22 | 262144 | 16 | exact | -0.6476 | -31375 | +4.5348 | 192950 |
| 23 | 262144 | 16 | 47C | +1.1532 | 49100 | +1.2136 | 64400 |
| 24 | 262144 | 31 | exact | +1.0432 | 46900 | -2.9371 | -165350 |
| 25 | 262144 | 31 | 47C | +0.2103 | 19275 | +1.4010 | 191025 |
| 26 | 262144 | 31/16 | exact | -1.4160 | -84300 | +1.3147 | 135625 |
| 27 | 262144 | 31/16 | 47C | +0.2507 | 23975 | -3.8537 | -283775 |
| 28 | 262144 | random | exact | +0.6105 | 33050 | +3.1350 | 232950 |
| 29 | 262144 | random | 47C | -0.3119 | -15025 | +1.7395 | 96225 |

Small cells include deltas of only 25–200 ns. Large-cell duplicate discrepancies
also matter: R1 case 25 has growable/fixed duplicate deltas +334,700 / -37,550 ns
versus primary +191,025 ns; R1 case 26 has -250,700 / +314,400 ns versus primary
+135,625 ns. These samples remain included. No allocator, CPU scheduling,
thermal, RDP or clock cause has been established for any timing difference.

## Execution, preservation and reproducibility

All 12 primary jobs have reconciled intent, launch/PID, terminal exit and final
result journals, exit code zero, accepted markers and hashed logs:

| Job group | UTC interval | Outputs |
|---|---|---:|
| Primitive build | 11:27:20.321–11:27:42.852 | 6 files / 564,412 B |
| Normal primitive | 11:27:43.106–11:27:44.882 | passing fixture |
| ASAN primitive | 11:27:44.906–11:28:01.474 | passing fixture |
| Integrated build | 11:30:23.597–11:31:35.662 | 14 files / 12,055,752 B |
| Six integrated fixtures | 11:31:36.338–11:32:01.210 | all passed |
| Benchmark build | 11:34:17.242–11:34:25.263 | 2 files / 186,821 B |
| CPU benchmark | 11:34:25.314–11:34:36.858 | 4,800 exact calls |

Native inspections are read-only completed subprocesses, separate from those
12 workload/build jobs. All build and sanitizer work preceded timing; no heavy
hash verification overlapped it. Every workload launch required at least 3 GB
free on C. Initial free space was 4,110,307,328 B. Final free space is recorded
in `final_summary.json`; nothing was deleted or moved in this stage.
No admin/firewall blocker occurred. No power, clock, cooling, priority, affinity,
security, firewall, driver or production-setting change was made; no push.

Artifacts are under `build-cuda-ninja/profiles/s187-artifacts`. Helpers, source
inputs, logs, proofs, objects, disassemblies and complete sample analysis are
preserved locally. The S186 1,512-file frozen inventory is unchanged, and 343
production-source archives are reused and rehashed from S182.

An initial postmeasurement verifier confused a native row's encoded instruction
byte count with the disassembly text-file size. It failed a read-only assertion.
The retained `s187_verify_v2.py` checks native byte counts against parsed
instructions and file integrity against SHA separately. No C++ source, build or
benchmark correction/rerun resulted. Both verifier versions are retained.
The corrected verifier additionally reconstructs every dataset's payload count,
including the unsigned xorshift random sequence, and recomputes the full schedule,
statistics, proof, source audit and journal reconciliation.

After freezing, verification is read-only:

```powershell
python build-cuda-ninja/profiles/s187_verify_v2.py --frozen
```

Do not rerun the fixed benchmark or rebuild into this artifact root. The final
inventory excludes only itself and the designated temporary directory.
The tracked change for this stage is this report; experimental C++ remains private.

## Next bounded investigation

If pursuing this owner further, specify a new **CPU-only integrated ANS writer
experiment** that uses and audits the actual recurrence/callback boundary.
The raw-loop inlining distinction is an explicit measurement gap, not evidence
that the integrated version must be faster. Preserve the S187 failed result and
gate; any new experiment needs its own protocol, inputs, controls and decision.
Do not advance to another whole-encoder campaign from S187.

More broadly, compact consumption, composition/reduction fusion and tile
scheduling remain separate mechanisms. This stage changes only CPU reverse-word
ownership; it does not qualify another GPU fusion or scheduling change.
