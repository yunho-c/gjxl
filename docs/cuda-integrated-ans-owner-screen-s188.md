# Integrated ANS owner screen: S188

Date: 2026-09-10. Parent: `16fe21cd0c6cfb5557edfa8ac5c9a7e508791dac`.
Production is unchanged. This is a new CPU-only diagnostic, not an S187 rerun.

## Decision

The actual integrated ANS writer comparison passed its correctness and native
call-boundary checks, but failed all three prospectively specified performance
thresholds:

| Diagnostic condition | Required | Observed |
|---|---:|---:|
| Primary wins, all 48 cells | at least 36 | 32 |
| Primary wins, 16 large-stream cells | at least 12 | 10 |
| Large-stream geometric-mean fixed/growable ratio | at most 0.99 | 0.9954080123 |

The aggregate large-stream reduction was only 0.4592%. Do not promote the
fixed-capacity owner or launch another whole-encoder campaign from this result.
The [S187](cuda-fixed-reverse-bit-owner-s187.md) raw-loop gate remains failed.

The next distinct source-level opportunity is to remove the packed owner's
temporary BitWriter and final copy through bounded bulk emission. That is a
hypothesis requiring new proof and measurement, not an established speedup.

## Why this diagnostic differs from S187

S187's raw chunk loop inlined packing. It did not time the separately emitted ANS
callback whose code shrank from 51 to 33 instructions. S188 closes that measurement
gap using the actual existing ANS implementations, with no source recompilation
of the writers, recurrence or callbacks:

- Growable: exact normal/ASAN `s186-artifacts/build/*_packed.obj`.
- Fixed: exact normal/ASAN `s187-artifacts/integrated/*_ans.obj`.
- BitWriter: exact current normal/ASAN objects from S187.
- Harness model construction, production-reference emission and count checks:
  exact normal/ASAN `s186-artifacts/build/*_original.obj`.

Each packed object is relinked as a small DLL whose `S188Write` export aliases
the actual `WriteAnsTokenStream(EntropyTokenStreamView,...)` entry directly.
There is no forwarding shim, instrumentation selector, renamed helper,
counter or forced-inline change. Duplicate labels 0/1 share the growable export
pointer and labels 2/3 share the fixed export pointer. The loaded modules stay
resident until process exit, after all output objects have been destroyed.

MSVC 14.37 normal and LLVM 22 clang-cl ASAN builds use the shared release CRT.
Only the new harness is compiled. Existing normal S168 / ASAN S162 codestream
archives resolve supporting routines; explicit ANS and BitWriter objects
override their archived counterparts. Inputs and support files are hash-pinned.
No CUDA compilation, GPU execution, NVML sampling or image-encoding campaign
is part of S188.

## Complete native-code audit and correction

The new audit reads raw COFF code sections and relocation records, maps them to
the linked PE DLL, and verifies each complete sole-function code COMDAT section.
It masks only four-byte REL32-family relocation fields when comparing linked
bytes to object bytes. The export RVA is the mapped writer itself. Actual
resolved destinations independently verify one writer-to-recurrence edge and
two recurrence-to-callback edges for each policy.

The complete recurrence bytes and normalized relocation triples also match
between growable and fixed, normalizing only anonymous-namespace identifiers.
The normal linked code is therefore the code intended for this diagnostic.
ASAN DLL-boundary behavior is checked by fixtures, not by a claimed normal/ASAN
native-code equality.

| Complete normal function | Growable instructions / bytes | Fixed instructions / bytes |
|---|---:|---:|
| Writer | 192 / 812 | 164 / 690 |
| Recurrence | 289 / 1140 | 289 / 1140 |
| Callback | 51 / 170 | 33 / 108 |

These are static counts, not executed instruction counts or cycle predictions.

This audit corrects two limitations in the earlier static-count tooling:

1. A space-only byte parser omitted the last byte of some ten-byte instructions
   when LLVM printed a tab immediately afterward.
2. Symbol-selected disassembly stopped at a local label in the writer, omitting
   trailing error/return code. S187's writer counts 154/645 and 128/530 describe
   incomplete parsed prefixes, not the complete function.

The recurrence and callback counts quoted in S187 remain correct. Full-object
disassembly checked against raw COFF sections also corrects S187's raw benchmark
sizes: VectorPack is 118 instructions / **447** bytes, not 445; FixedPack is
89 / **337**, not 336. Its inlined-packing observation remains valid.
These corrections change no binaries, measurements, or gate decisions.
Frozen S187 artifacts and report remain untouched; this report is the correction.

During development, the first S188 audit inherited the incomplete parser.
A second attempt used the first PE unwind range, but the recurrence has multiple
chained-unwind regions. The final `s188_native_v3.py` uses the complete
sole-function COFF COMDAT extent and handles REL32_1 as well as REL32.
All audit corrections preceded timing; older helper versions are retained.

## Synthetic inputs and qualification

There are 24 datasets: three token counts (1,024 / 16,384 / 199,680), four
distributions, and interleaved or offset-split storage. Each token stream cycles
through 458 contexts, with a supplied initial eight-histogram partition.
The balanced direct-ANS optimizer runs outside timing and can merge that
partition: the observed models have one, two or three clusters, not eight fixed
clusters. Alphabet log sizes are five or six.

The source-defined deterministic xorshift generator produces:

- Zero-heavy: 95% zero, otherwise values 1..15.
- Dense-low: values 0..511.
- Mixed: 80% values 0..7, 15% 0..255, 5% 0..65535.
- Wide: full-range unsigned 32-bit values.

These are synthetic distribution rules, not captured coefficient populations.
The labels do not establish image representativeness. Initial HybridUint
configuration is the default except wide values start with {0,0,0}; the actual
validated optimized model drives both implementations and the oracle.

Each layout pair uses identical token values, contexts and serialized models.
Destination prefix is `case % 8`, so layout and prefix are not independently
crossed. Per-cell policy comparisons have identical inputs, but differences
between layout rows must not be attributed solely to token layout.

A reference recurrence uses ordinary integer division and then independently
emits each bit of the resulting reversed chunks, including state and prefix.
It shares the public HybridUint conversion and model tables, not the optimized
reciprocal recurrence or packed output routine. Production writer and count-only
results are also checked before either qualification or timing.

Each normal/ASAN qualification process checks all 24 datasets and four labels:

| DLL-boundary check | Per process |
|---|---:|
| Ordinary successful emission | 96 |
| Exact output allotment | 96 |
| Insufficient allotment and exact rollback | 96 |
| Early/late invalid contexts and untouched destination | 192 |
| Null destination rejection | 96 |
| Total calls | 576 |

Both passed: 1,152 DLL qualification calls. Across normal check, ASAN check and
benchmark setup, 72 production-reference outputs and 72 count-only checks also
passed. All 6,912 benchmark calls matched exact expected bytes and bit length.
FNV-1a metadata fingerprints for tokens, models and outputs are retained and
agree between the three processes; they are consistency checks, not cryptographic
collision proofs. The verifier independently reconstructs token fingerprints.
Artifact integrity uses SHA-256.

S187's larger primitive and integrated-host qualification remains prerequisite
evidence for the unchanged objects. S188 does not add allocator-failure injection,
MSAN, TSAN, GPU memory checking or whole-image qualification.

## Fixed timing protocol and results

One normal benchmark process preconstructs all 24 models and oracles, then visits
all datasets forward and reverse: 48 cells. Each cell has four warm and 32 measured
rounds with all four labels. Williams orders 0132 / 1203 / 2310 / 3021 cycle;
the order index is reversed in pass one. Totals are 6,912 calls, including 6,144
measured. No sample filtering, rerun or performance-based early stop was used.

Timing includes token-view construction, the indirect call to the actual writer,
ANS recurrence/callback, staging allocation and destruction, packing, and checked
append. Model construction, oracle generation, output/prefix setup, result/byte
checking, logging and final output destruction are outside the timed interval.
No builds, sanitizers or heavy hash verification overlapped timing. CPU clocks,
priority, affinity, cooling and other system controls were not changed.

Primary delta is the median within-round fixed duplicate mean minus growable
duplicate mean. Percent is the median within-round ratio minus one, not a ratio
of separately computed medians. The large-stream geometric mean uses those
per-cell ratios. All four cross-pair results and both same-path duplicate
differences are retained. The gate is a screening rule, not a significance test.

There were 18/24 primary wins in pass zero and 14/24 in pass one: 32/48.
Complete cell results follow. Negative means fixed is faster.

| Case | Tokens | Distribution | Layout | Prefix bits | R0 change % | R0 delta ns | R1 change % | R1 delta ns |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 1024 | zero-heavy | interleaved | 0 | -0.4350 | -50 | +0.4706 | 50 |
| 1 | 1024 | zero-heavy | split | 1 | +0.0000 | 0 | +1.7982 | 200 |
| 2 | 1024 | dense-low | interleaved | 2 | +0.3402 | 75 | +0.6928 | 150 |
| 3 | 1024 | dense-low | split | 3 | -3.4557 | -800 | +4.2076 | 925 |
| 4 | 1024 | mixed | interleaved | 4 | -0.4310 | -75 | +0.5970 | 100 |
| 5 | 1024 | mixed | split | 5 | -0.2849 | -50 | +3.8128 | 750 |
| 6 | 1024 | wide | interleaved | 6 | -0.1591 | -75 | -0.1835 | -75 |
| 7 | 1024 | wide | split | 7 | -1.7670 | -850 | -2.2195 | -900 |
| 8 | 16384 | zero-heavy | interleaved | 0 | -0.2554 | -450 | -0.0335 | -50 |
| 9 | 16384 | zero-heavy | split | 1 | +0.0438 | 75 | -0.1460 | -225 |
| 10 | 16384 | dense-low | interleaved | 2 | -0.1076 | -375 | -0.2413 | -775 |
| 11 | 16384 | dense-low | split | 3 | -2.9512 | -11175 | -3.4168 | -11525 |
| 12 | 16384 | mixed | interleaved | 4 | -0.8992 | -2400 | -0.3626 | -900 |
| 13 | 16384 | mixed | split | 5 | -0.9447 | -2625 | -0.8248 | -2125 |
| 14 | 16384 | wide | interleaved | 6 | +0.7484 | 8800 | -0.6559 | -4275 |
| 15 | 16384 | wide | split | 7 | -2.4108 | -19775 | -1.8588 | -12450 |
| 16 | 199680 | zero-heavy | interleaved | 0 | +0.1241 | 2500 | +0.3467 | 6250 |
| 17 | 199680 | zero-heavy | split | 1 | -0.0436 | -850 | -0.2480 | -4925 |
| 18 | 199680 | dense-low | interleaved | 2 | -0.5523 | -24450 | +1.3619 | 57975 |
| 19 | 199680 | dense-low | split | 3 | -3.5993 | -164875 | -3.2655 | -177850 |
| 20 | 199680 | mixed | interleaved | 4 | -1.0494 | -37075 | +0.2224 | 8375 |
| 21 | 199680 | mixed | split | 5 | -0.3228 | -13350 | -1.8069 | -64450 |
| 22 | 199680 | wide | interleaved | 6 | -0.9581 | -99750 | +2.3496 | 196450 |
| 23 | 199680 | wide | split | 7 | +0.4833 | 54950 | -0.2175 | -8200 |

The large dense-low split case wins both passes (-3.60% / -3.27%), but this
subset cannot override the failed aggregate gate. Small-cell deltas often lie
within tens or hundreds of nanoseconds. Large wide-value controls also show
substantial variation: case 22 R0 growable/fixed duplicate deltas are
+63,650 / +471,550 ns versus primary -99,750 ns; R1 duplicates are
+450,050 / -319,500 ns versus primary +196,450 ns. These samples remain included.
No clock, thermal, scheduler, RDP or allocator cause is inferred.

The independent oracle derives these counts for each 199,680-token stream:

| Distribution | Extra-bit chunks | Renormalization chunks | Payload bits | Full 56-bit words |
|---|---:|---:|---:|---:|
| Zero-heavy | 0 | 6,057 | 96,912 | 1,730 |
| Dense-low | 193,475 | 49,867 | 1,797,117 | 32,091 |
| Mixed | 38,124 | 50,457 | 1,047,147 | 18,699 |
| Wide | 199,680 | 24,849 | 6,389,964 | 114,106 |

These are derived from the reference recurrence, not counters added to timing.
They describe work volume but do not establish phase timing or causation.

## Execution and recording

All four primary build/workload jobs have reconciled intent, launch/PID,
terminal exit and accepted result journals, with exit code zero:

| Job | UTC interval |
|---|---|
| Build | 11:54:16.126–11:54:44.162 |
| Normal qualification | 11:59:56.147–11:59:57.186 |
| ASAN qualification | 11:59:57.209–11:59:59.266 |
| Fixed benchmark | 12:01:42.083–12:01:56.605 |

The build produced 20 files totaling 2,649,457 bytes. No C++ correction, rebuild,
or workload rerun occurred. A prelaunch metadata check initially assumed eight
final clusters; it was corrected to allow optimizer merging before the timing
protocol was written or any timing child launched.

The benchmark wrapper subsequently exited with a recording error: after its job
had durably created `benchmark.json`, it attempted to create the same filename
for an enclosing summary. The child had already exited zero; exit journal,
passing log marker, log hash and accepted job record were all durable.
`s188_finish_benchmark.py` reconciled those existing records into a new
`benchmark_result.json`. Nothing was overwritten or relaunched. Do not describe
the wrapper itself as a clean exit-zero process.

Every primary launch required at least 3 GB free on C. Initial free space was
4,077,424,640 bytes; final space is recorded in `final_summary.json`.
No files were removed or moved, no recovery volume was used, and no admin or
firewall intervention was needed. No security, driver, power, clock, cooling,
priority, affinity or production setting changed. No push was made.

Artifacts live under `build-cuda-ninja/profiles/s188-artifacts`: input archives,
direct-export DLLs, maps, complete disassemblies, raw logs, all samples, analyses,
journals and corrected audit helpers. The S187 frozen 180-file inventory remains
unchanged; 343 production-source archives are reused and rehashed from S182.
After the root is frozen, verify it without executing another workload:

```powershell
python build-cuda-ninja/profiles/s188_verify.py --frozen
```

## Next mechanism, not another callback screen

Source inspection of the unchanged packed helper shows one checked
`temporary.WriteBits(56,...)` call per full word, followed by
`destination->Append(temporary)`. Current BitWriter prepares each write and
emits it byte by byte; final append is a memcpy for aligned destinations and
another bytewise path otherwise.

A bounded bulk-emission primitive could consume the reverse words directly,
reserve/prepare the final destination once, and avoid the temporary stream and
copy. It must preserve exact state/tail order, all bit alignments, overflow and
allotment errors, initialized storage, failure atomicity and public-model
validation. That requires a new private proof/fixture first, then an actual-ANS
comparison with audited native boundaries. The existing owner-only results
do not qualify it, and no unmeasured speedup is claimed.

This stage advances compact CPU coefficient consumption diagnostics. It does not
qualify another GPU composition/reduction fusion or tile-scheduling change.
