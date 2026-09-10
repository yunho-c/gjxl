# S180: native POPCNT for sparse coefficient ranking

The candidate is correct and removes the intended compiled call sites, but
misses the precommitted performance gate: four of six worker-work wins and
three of six tokenization-wall wins, short of six and four. It is not
promoted. Production and its CPU requirements remain unchanged; the full
optimization objective remains active.

## Question and scope

S179's transform-local expansion passed correctness but missed its fixed
performance gate. It cleared seven to fifteen logical values for every
emitted coefficient token in the screened images. This stage investigates
the compiled cost of the original sparse point read instead of adding an
expansion buffer, payload copy, or another reservation policy.

The parent is `203c8ff73566232f12e19f6e21aa5020ddd0add9` on `feat/cuda`.
Production remains the qualified S168 implementation. Evidence is under
`build-cuda-ninja/profiles/s180-artifacts`; versioned `s180_*` helpers are
beside it. All 343 production source files remain unchanged.

The frozen production codestream library's int32 sparse direct-token template
contains a call to an out-of-line `std::popcount` helper for each nonzero
coefficient's payload rank. Zero coefficients already skip ranking and
payload loading. Thus avoiding zero lookups alone is not a new improvement,
and counting fewer source expressions is not a speedup measurement.

The candidate replaces only the coefficient read expression in a private
clone of the direct-token template. Its force-inline sparse accessor retains
the same absolute word, mask, bit test, offset, and lower-bit rank expression,
but uses `__popcnt64` for that rank. The original nonzero-count operation,
LLF exclusion, reservations, predictors, coefficient orders, contexts,
populations, and transactional output publication remain exact. There is no
new buffer, clearing, payload enumeration, or copying.

The intrinsic requires CPU POPCNT support. The diagnostic checks CPUID leaf
1, ECX bit 23 before use, outside Encode; unsupported hardware fails before
running the candidate. This matches Microsoft's documented support test.
No fallback layer, `/arch` flag, or project-wide minimum-ISA change is added.
[Microsoft POPCNT intrinsic documentation](https://learn.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64?view=msvc-170).

## Controls and qualification protocol

Modes 0/1 invoke the same original template. Modes 2/3 invoke the same native
rank candidate for sparse groups only; dense whole routes always invoke the
original. The common diagnostic selector is configured between joined
encodes, and common per-group audit rows record dispatch, tokens, anchors,
and capacities. Their overhead is inside Encode. Setup, reference encoding,
prior-result clearing, validation, and read-only NVML endpoints are outside.
Concurrent diagnostic reconfiguration is unsupported.

Source auditing reconstructs the full translation unit and checks that only
the private clone's name/read expression and the diagnostic dispatch differ.
Native auditing examines the frozen production library plus both diagnostic
objects, rather than inferring production code generation from a prototype.
Normal and ASAN whole DLLs retain eleven unchanged CUDA modules; no GPU
source, launch, device recorder, scheduling pause, or machine setting changes.

The fixed screen retains S179's comparison design: 1080p, 4K, and Keong;
eight CPU threads; two reversed process-order passes; eight warm-up and
sixteen measured rounds per process. The four Williams orders are `0132`,
`1203`, `2310`, and `3021`. The primary statistic is the median within-round
difference of the candidate duplicate mean and original duplicate mean.
Individual comparisons and same-path duplicate disagreements are retained.
Worker work is nested and cannot be added to wall time.

The screen cannot authorize promotion. Its precommitted broader-test gate
requires lower coefficient-token worker work in all six cells and lower
AC-tokenize wall time in at least four. Every sample stays; no selection,
filtering, or rerun is allowed to rescue a failed gate. No build, sanitizer,
profiler, compression, or heavy hashing runs during timing. Ordinary shared-
machine activity and light report editing remain limitations.

## Build recovery

The initial normal compile stopped with MSVC 14.37 error C3495 on the
diagnostic's nested-lambda use of a local `constexpr` predicate. It produced
no output file. Version 2 repeats the dependent `requires` expression
directly in the inner lambda, using the prior qualified dispatch pattern.
This changes no candidate algorithm or expression. The empty output
directory is checked before recovery; all original pins, sources, command,
error output, and failed-job metadata are preserved. The corrected source,
fixture include, build script, source auditor, and recovery runner have new
versioned names. No frozen helper or build output is overwritten.

## Compiled-code evidence

The completed native audit verifies the following call sites in both the
frozen production library and the corresponding original diagnostic
templates. All three normal candidate instantiations contain exactly one
inline POPCNT instruction and no sparse point/rank helper call site.

| Native width | Original emission call site | Candidate emission |
|---|---|---|
| int8 | Sparse point accessor | Inline mask test, POPCNT rank, payload load |
| int16 | Sparse point accessor | Inline mask test, POPCNT rank, payload load |
| int32 | `std::popcount` rank helper | Inline mask test, POPCNT rank, payload load |

All whole inputs in this stage use int32 coefficients. The original standard
rank helper reads `__isa_available`, branches between software and hardware
population count, and returns. The relevant helper definitions in the
production library and normal diagnostic are retained. Instruction presence
establishes the intended mechanism, not its performance benefit; branch
prediction, register allocation, surrounding code, and unchanged histogram
work still matter.

The audit itself needed three corrected assumptions. Version 1 assumed the
narrow templates called the rank helper directly, but they call the sparse
accessor. Version 2 then expected standalone ASAN template symbols, whereas
clang inlines both paths into the variant visitor. Version 3 expected one
standard-helper definition per archive, but the frozen library has COMDAT
definitions in both `coefficient_order` and `ac_group`. Version 4 distinguishes
the call paths, retains the full ASAN object disassembly, and checks both
production helper definitions. Every partial audit and version is preserved;
these are audit failures, not candidate correctness failures. No build or
fixture is repeated for them, and no performance sample yet existed.

Normal machine-code assertions cover each separate sparse template. The ASAN
object is archived without asserting identical code or standalone template
instruction counts. Source auditing verifies that counting/reservation and
context/population logic are unchanged, but does not claim identical machine
code for separately instantiated whole functions.

An additional control audit compares the complete original sparse templates
against the frozen production library. All instruction bytes and relocation
offsets/types/targets match for all three widths after normalizing only the
translation-unit anonymous-namespace identifier. This is an unlinked-object
comparison; it does not prove whole-DLL identity or performance neutrality
of the surrounding diagnostic scaffolding.

## Host correctness

Normal and ASAN fixtures each pass 97,920 checks across 128 frames / 320
groups, plus 1,296 focused sparse-range cases. Combined coverage is 195,840
checks and 2,592 ranges. Every value, context, and population field matches
the original dense and sparse oracles, and vector capacities are unchanged.
The dense candidate forwarding overload is also checked, although whole
dense routes continue to call the original template.

Coverage includes all seven strategies and mixed layouts, int8/int16/int32
extrema, eight coefficient patterns, natural/custom orders, three context
maps, populations on/off, unaligned starts, reversed payload-word order,
LLF/suffix extrema, and partial-word ranges from zero through 1,024 values.
Range tests also check all-zero input and that source storage is unchanged.
Short spans and null scratch/output are rejected without modifying the
sentinel output. Both executables confirm CPU POPCNT support before use.

Whole preflight then passes all eleven inputs, automatic/eight requested CPU
threads, and normal/ASAN builds: 44 processes and 220 encodes. The 22 ASAN
processes include 88 instrumented Encode calls and 22 normal reference calls.
CUDA memcheck on 4K and initcheck on Keong add ten encodes, for 230 total.
Both tools report zero errors; memcheck reports zero leaked bytes/allocations.
These slow instrumented runs show advancing samples and are not performance
evidence. No administrator or firewall blocker is encountered.

Every whole call matches the frozen S168 reference's bytes and complete
summary. Image references also match their frozen codestreams. Native owner,
coefficient width, group dispatch, token count, anchors, and original vector
capacities are checked independently against the retained S169 transfer data
and S172 per-group census. The corpus includes derived photograph sizes and
synthetic controls; it is not a survey of independent native-resolution
images or other machines.

## Performance result and decision

The six screen processes complete 582 whole encodes, including 384 measured
calls. All 1,152 warm/measured NVML endpoints report the unchanged 40,000 mW
limit. Negative deltas are faster. These are medians of within-round paired
differences, not differences between independent medians or additive stage
decompositions.

| Case / pass | Whole ms (%) | AC-tokenize wall ms | Coefficient-token worker ms (%) | Earlier quantization ms |
|---|---:|---:|---:|---:|
| 1080p / 0 | -0.650 (-1.09%) | +0.045 | -0.265 (-3.34%) | -0.226 |
| 4K / 0 | -0.175 (-0.05%) | -0.243 | -1.410 (-6.05%) | +2.702 |
| Keong / 0 | -1.610 (-0.58%) | -0.820 | -3.952 (-6.11%) | +2.516 |
| Keong / 1 | +11.655 (+4.23%) | +0.745 | +1.453 (+2.56%) | +2.649 |
| 4K / 1 | -7.759 (-2.88%) | -0.427 | -0.959 (-3.48%) | -0.029 |
| 1080p / 1 | +0.353 (+0.55%) | +0.153 | +0.074 (+0.59%) | +0.772 |

The 4K worker reduction repeats, but its whole-encode result ranges from
nearly unchanged to substantially faster. Keong and 1080p change sign in
both worker work and whole time. The unchanged earlier quantization stage
also varies; those shifts are not work removed by the later sparse reader.
Neither the intended instruction change nor the favorable 4K subset proves
a robust general whole-encode improvement.

Same-path duplicate disagreements are material. The following within-round
whole deltas compare mode 1 minus 0 and mode 3 minus 2, in milliseconds.
Full individual-pair and all 41 phase comparisons remain in `analysis.json`.

| Case / pass | Original duplicate | Candidate duplicate |
|---|---:|---:|
| 1080p / 0 | +0.783 | -0.754 |
| 4K / 0 | +4.465 | -6.069 |
| Keong / 0 | +14.323 | -0.309 |
| Keong / 1 | +9.455 | +4.904 |
| 4K / 1 | +1.481 | +1.517 |
| 1080p / 1 | -1.903 | +2.680 |

The second Keong pass has worker duplicate deltas of +6.320 / +3.165 ms,
larger than its +1.453 ms candidate delta. The data therefore do not establish
that the intrinsic itself causes that slowdown, or that every small effect
has been resolved. They do establish that the precommitted broader-test gate
was not met. No broader campaign, filtering, rescue rerun, subset promotion,
project-wide ISA change, or production modification follows this screen.

The next actionable question is the larger append/population path, not
another isolated rank or blanket-expansion sweep on this cohort. S43 already
removed per-token string-owning success results. Further changes should
first prove which count/context/slot bounds follow from a validated AC group,
then independently qualify any reduction in repeated checks or calls,
including exact outputs and failure atomicity. This is a new investigation,
not a claim that removing checks is already safe or fast. GPU perceptual work
and resident handoff integration remain open parts of the full objective.

## Timing and retained evidence

All times below are UTC on 2026-09-10:

| Activity | Start | Finish |
|---|---|---|
| Failed initial compile | 08:16:34.836 | 08:16:39.361 |
| Corrected normal/ASAN build | 08:17:50.479 | 08:18:51.039 |
| Normal fixture | 08:19:06.134 | 08:19:46.698 |
| ASAN fixture | 08:19:46.710 | 08:21:16.300 |
| Normal/ASAN whole preflight | 08:22:42.958 | 08:23:42.063 |
| CUDA memcheck | 08:23:42.075 | 08:25:53.178 |
| CUDA initcheck | 08:25:53.182 | 08:27:30.761 |
| Fixed screen | 08:28:11.949 | 08:30:17.671 |

The 56 journaled subprocesses do not overlap: 55 are accepted and one is the
preserved initial compile failure. The three partial native audits described
above are separate diagnostic-audit failures, not additional failed builds
or correctness tests. Combined whole coverage is 812 encodes: 230 preflight
and 582 screen calls. All commands, journaled-job logs/status, binary and
module hashes, source/protocol snapshots, native disassembly, control
identity checks, counters, and unfiltered results are retained.

`s180_verify.py` checks S179's immutable 556-file inventory; nineteen inherited
support inputs and SDK pins; original and corrected build-input records;
eleven CUDA modules per build; input/oracle hashes; source reconstruction;
native call-site assertions and original-control identity; fixture counts;
all whole logs, schedules, journal ordering, analysis, and decision.
`s180_freeze.py` archives the report and versioned helpers and writes
`final_sources.json`, `final_summary.json`, and `artifact_hashes.json`.
Frozen verification checks the exact artifact set and every recorded hash.

From the repository root:

```powershell
python build-cuda-ninja/profiles/s180_verify.py --frozen
```

Do not edit or rebuild this frozen evidence in place. Any follow-up requires
new versioned files. No material file is deleted, no predecessor/helper or
build output is overwritten, and no power, clock, cooling, affinity,
priority, driver, firewall, or security setting changes. No permission
blocker is encountered. Only this report is committed; no maxed-out claim
is made.
