# S168: byte-aligned ordinary bit-writer append (retained, with mixed image results)

## Scope and hypothesis

The predecessor is S167, commit
`cffd4bf87dc1182c8c05e2567d832ee6deb58d93`. That stage retained ANS chunk
batching with consistent dense-input gains but mixed real-image results,
including recurring whole-4K slowdowns. This stage does not assume that those
slowdowns are resolved or caused by append.

S165's earlier instrumented ANS decomposition found append to be a material
dense-input cost (about 15.2% and 9.9% of sampled ANS worker time on dense 513
and 1025), while its 4K share was much smaller (about 4.1%). Those measurements
predate S166/S167; they motivate this candidate, not a current speedup estimate.
Current source inspection confirms that ordinary `BitWriter::Append` still
routes every complete source byte through `WriteBitsUnchecked`, including
when the destination is already byte-aligned.

Only `src/codestream/bit_writer.cpp` changes runtime behavior. ANS batching,
reverse-chunk allocation/storage, ANS recurrence, coefficient representation,
GPU composition/reduction, tile scheduling, and public options remain unchanged.
The append test is registered in CMake. The change is retained for the
consistent dense-input benefit, with the mixed image results below explicitly
accepted rather than declared resolved.

## Implementation and invariants

After the existing self-append rejection and `PrepareWrite` gate, a byte-aligned
destination copies the source's padded byte span with `memcpy`. It advances by
the exact source logical bit count, not the padded byte count. The existing
unaligned byte/bit loop is unchanged. The distinct `AppendByteAligned` API,
which intentionally concatenates padded sections, is not substituted for
ordinary append and is not changed.

The existing gate checks arithmetic overflow, byte-size limits, active
allotments, and allocation failure before copying. For an aligned destination,
the allocated extent is exactly the destination's existing complete bytes plus
the source's padded bytes. Source and destination are distinct writer objects
with independent owned vectors. The public padded-byte invariant guarantees
zero unused high bits in the final byte; copying that byte preserves the
invariant and does not insert logical padding. Empty spans are guarded before
forming a destination data pointer or calling `memcpy`.

There is no unchecked word store beyond vector size, new API, image-dependent
dispatch threshold, compatibility layer, change to error handling, or change
to rollback/allotment policy.

## Direct and host qualification

The new `tests/bit_writer_append_test.cpp` constructs source bits through
one-bit writes, then independently packs the expected output directly from
individual bits. Expected bytes do not use production append or the bit writer.
It checks logical length, all byte values including unused high tail bits,
and source immutability.

Each S167-baseline normal, candidate normal, and candidate ASAN execution passes:

- 18,936 cases: prefix lengths 0–23, source lengths 0–257 plus 512, 1023,
  1024, 4097, and 65539 bits, crossed with zero, one, and deterministic random
  source patterns. This covers all eight initial alignments and tail widths.
- Ordinary append and exact-size allotments, including empty sources and
  empty destinations; 18,864 one-bit-insufficient allotments fail atomically.
- Nested exact allotments followed by a one-bit suffix, repeated appends
  interleaved with bit writes, and source immutability after all operations.
- 37,872 rollbacks after successful append: a late invalid write and an
  intentional exception each restore original destination bytes and length.
- Self-append rejection, including empty writers, without modifying the writer.

These are explicit deterministic contract fixtures, not exhaustive allocation
failure injection or proof over malformed private writer state.

All fourteen host test executions pass: the baseline append matrix; candidate
normal append, existing bit-writer, ANS-emission, and entropy suites; and nine
candidate ASAN suites (append, bit writer, ANS emission, entropy, AC section
validation, encoder, public workflow, compact frame, and sparse frame).
The ASAN frame/model tests retain their existing coverage; they are not a new
independent full entropy-policy cross product. Standard-build integration and
the configured test coverage are recorded separately below.

### Preserved build-harness rejection

The first build launcher named both the implementation object and existing
bit-writer test object `normal_bit_writer.obj`. Compiling the test overwrote
that attempt's implementation object; linking the next test failed with
duplicate `main` symbols (`LNK2005`/`LNK1169`). No host tests or timing campaign
ran from that failed attempt. Its original scripts, input snapshots, partial
outputs, log, and rejected journal remain untouched.

A separate `build-v2` directory and separately named launcher use
`normal_bit_writer_impl.obj` / `asan_bit_writer_impl.obj`. The new build
passes, with no linker warnings. Whole/retained link maps identify the new
implementation object as the sole provider of ordinary `BitWriter::Append`.
Runtime source and test source were not changed for this harness correction.

The first qualification verifier also counted MSVC `$unwind$` / `$chain$`
metadata containing the append symbol name as additional function definitions.
Its original source is archived as `verify_s168_initial.py`; the corrected
map parser matches the actual function-symbol field. Both DLLs have exactly
one append definition, supplied by the new implementation object. No build,
test, or comparison was repeated for this verifier correction.

## Whole/retained qualification and timing plan

The whole/retained DLLs use frozen S166 bridge objects, S167's standard-build
codestream library, unchanged dependency libraries, and the new bit-writer
object. ASAN reuses source-pinned S167 ANS/test objects, the S166 encoder
object, and S162 dependency libraries alongside the new sanitized bit writer.
No frozen predecessor is rebuilt or edited.

All eleven extracted CUDA modules per DLL match the frozen S162 module hashes.
Forty-four whole/retained comparisons across the established six real images
and five synthetic cases, automatic/eight participants, plus whole-4K/eight
CUDA memcheck/full leaks and initcheck all pass. They check
exact output/public summaries, real codestream oracles, and retained native
fingerprints/captured-whole oracles with no further GPU work during serialization.

The qualification comparisons ran from 2026-09-09 23:25:15 UTC; memcheck
completed at 23:28:27 with zero errors and zero leaked bytes, and initcheck
completed at 23:30:11 with zero errors. All 61 accepted prototype journals
and the one preserved failed-build journal are retained. Before any timing,
the workflow moves to standard-build integration in a separate artifact root.
The prototype timing script was prepared but never launched; there are no
prototype timing results or discarded performance trials.

## Standard-build integration

Prototype qualification was independently verified and frozen into 1,595
artifact files and 365 source/helper snapshots before CMake registration.
Standard-build artifacts are separate, under
`U:/gjxl-cuda-diagnostics/s168-promotion`. Preparation checked every frozen
prototype file and archived 343 current production inputs. The runtime
bit-writer source and test body remain byte-identical to the qualified prototype;
only the four-line CMake test registration is added.

The normal Ninja build completed 34 CPU compilation/link actions from
2026-09-09 23:33:55 to 23:34:09 UTC; no CUDA source was compiled. The integrated
append matrix passed directly. All 92 CTest entries selected by excluding
only `codec_install_consumer` passed in 287.56 seconds, finishing at 23:39:01.
The complete configured inventory contains 93 unique tests, including the new
append test; the 92 passed names are checked against that inventory.

The remaining install-consumer check passed separately at 23:39:15. To avoid
exhausting C:, its driver is copied with only the scratch-directory assignment
changed to an exact, previously nonexistent U: directory and a final success
marker added. Install/configure/build and all four downstream executable runs
are unchanged, as are the original CTest command's other arguments. The original
driver remains untouched. The adapted driver is compared against that exact
two-change transformation, and its source hash, original/adapted commands, and
scratch root are recorded. This is 92/92 CTests plus the separate remaining
install-consumer check, not a claim of 93 CTest executions. No existing directory
or user data was deleted to perform the check.

The standard codestream library and ten normal test executables are archived.
New whole/retained DLLs link that standard library and frozen bridge/dependency
objects, without a bit-writer overlay. Their 44 exact-output comparisons and
two CUDA sanitizer checks all pass. The comparisons began at 23:40:24 UTC;
memcheck completed at 23:43:34 with zero errors and zero leaks, and initcheck
completed at 23:45:19 with zero errors. All eleven CUDA modules per standard
DLL match the frozen predecessor. The nine prototype ASAN
suites are tied to the unchanged runtime/test sources; they are not relabeled
as nine additional standard-build ASAN executions.

`verify_s168_promote.py --qualification` passes before timing. It verifies the
frozen prototype, archived-versus-current input differences, exact CMake
registration, all test/driver evidence, standard link-map provenance, comparison
rows/oracles, unchanged CUDA modules, and the 51 accepted standard-build
qualification journals. The source and test bodies match the prototype.

## Timing protocol

The planned standard-build timing evaluation is one seeded shuffled campaign of 168
jobs: 88 primary comparisons (all eleven cases, both modes, both thread budgets,
both creation orders) and 80 both-build independent-state controls on tiny 65,
sparse 513, dense 513/1025, and 4K. Each has four warmup and eight measured
rotating duplicate-ABBA rounds: 5,376 measured calls, 2,688 warmups, 168 oracle
calls, and 168 untimed whole captures if completed. Analysis uses medians of
within-round duplicate-label means and reports all four cross-label median
comparisons as sensitivity evidence, not confidence intervals. Controls are
not subtracted and results are not pooled with S167. Setup/cleanup, result
queries, byte checks, capture, and NVML observations are outside timing.

The campaign ran from 2026-09-09 23:45:55 to 23:58:00 UTC. All 168 jobs
completed, with all planned counts above and 16,128 NVML observations of a
40 W limit. No failed timing job was retried, omitted, or replaced. The limit
does not establish constant clocks or exclusive machine use. Light source
review/documentation work occurred during the run, but no build or sanitizer.

### Preserved analyzer correction

The pinned initial analyzer interpreted every JSON object with a `command`
field as a job journal. The separate install-consumer protocol legitimately
contains a command but no job name, causing a `KeyError` after the timing
campaign finished. It wrote no partial analysis. The subsequent report command
also found no analysis file. No measurement or codec behavior failed.

The pinned analyzer remains unchanged. `analyze_s168_final.py` recognizes that
one specific protocol record, verifies that it predates timing, and retains
the interval/non-overlap checks for every actual job. All sample-count,
byte/metadata, source/binary hash, schedule, and power checks remain in force.
It recomputes the complete campaign successfully; `verify_s168_final.py` then
verifies all 219 original promotion journals and the frozen prototype.
Its verified pre-follow-up source is additionally archived as
`verify_s168_before_focus.py` before extending verification for the new campaign.

## Full-corpus timing results

Negative percentages mean faster. Each table cell lists creation orders
O0/O1; these paired medians and the separate cross-label checks are not
confidence intervals. All primary outer-time results are included:

| Case | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| 65 `p2` | +0.129, −2.239 | −0.000, −2.852 | −0.264, −0.920 | −0.566, −1.792 |
| 513 `p0` | +0.186, +0.701 | +0.113, +1.240 | −0.546, +1.865 | −0.460, +1.927 |
| 513 `p1` | −0.235, −0.590 | +3.280, −0.268 | −2.880, −3.905 | −4.911, −4.208 |
| 513 `p2` | −9.889, −2.944 | −4.257, −5.565 | −9.757, −17.991 | −11.208, −13.469 |
| 1025 `p2` | −9.521, −8.187 | −5.851, −4.458 | −11.797, −9.996 | −11.454, −7.193 |
| flower 500 | −2.229, −0.671 | −1.770, −0.994 | −1.227, −1.499 | +1.389, −0.676 |
| 1080p | +0.297, −1.224 | +1.349, −0.906 | +2.088, −1.323 | +0.211, −0.182 |
| flower 2000 | +0.214, −3.292 | +1.342, −0.447 | −1.343, −1.433 | −1.151, −1.954 |
| flower 3200×2160 | +3.261, +4.890 | +1.557, +1.238 | −1.209, −0.986 | −0.110, +0.907 |
| keong 3839×2159 | +1.822, −2.998 | −1.571, +1.028 | −1.551, −4.614 | −1.180, −5.159 |
| 4K | −1.075, −0.487 | −0.126, −0.352 | +4.790, +1.328 | −2.686, −1.056 |

Whole outer time improves in 28/44 trials (103/176 cross-label comparisons),
and retained outer in 36/44 (126/176). All eight dense 513/1025 trials per mode
and all 32 cross-label comparisons per mode improve in outer time, codestream
wall, AC-section wall, and token-worker sums. Dense whole outer gains span
2.94–9.89%, retained 7.19–17.99%. Dense section-wall gains span 12.15–27.25%
whole and 16.00–31.14% retained; token-worker reductions span 9.26–21.95% and
17.68–31.50%. These worker/section reductions are not whole-encode percentages.

The six established image fixtures are less uniform: whole outer improves in
14/24 trials (50/96 cross-label comparisons), from 3.29% faster to 4.89% slower.
Retained outer improves in 18/24 (60/96), from 5.16% faster to 4.79% slower.
Whole/retained section wall improves in 16/24 and 18/24; token worker sums in
17/24 and 20/24. These fixtures do not establish universal improvement or broad
real-world coverage; the large flower/Keong fixtures are enlarged photographic
inputs, not independent native-resolution photographs.

Whole 4K improves in all four paired outer medians, but by only 0.126–1.075%
and with just 9/16 cross-label comparisons faster. This is not strong evidence
that S167's 4K concern is resolved. Retained 4K/auto regresses in both orders,
by +4.790% and +1.328%; both eight-thread paired medians improve.

Whole flower3200 regresses in all four primary trials. Its auto/O1 outer
change is +10.751 ms (+4.890%), with all four cross-label comparisons slower;
codestream is +8.412 ms, entropy optimization +2.720 ms, section +0.752 ms,
and token worker +1.501 ms. Auto/O0 is +7.232 ms (+3.261%) by within-round
pairing even though all four cross-label outer medians are faster. Different
aggregations can disagree. Retained 4K/auto/O0 is +3.244 ms (+4.790%), with
all four outer cross-label comparisons slower, despite token worker −0.673 ms.
These separately aggregated/overlapping phases are not additive and do not
establish the cause of the regressions.

All same-build control outer results follow in O0/O1 order. They are sensitivity
checks, not corrections or an explanation that erases unfavorable trials.

| Control case/build | Whole auto | Whole 8 | Retained auto | Retained 8 |
| --- | --- | --- | --- | --- |
| 65 `p2` baseline | −1.264, +0.181 | −2.305, +2.553 | +0.513, −0.640 | +1.108, +0.771 |
| 65 `p2` candidate | +1.796, +1.940 | +0.500, +6.926 | +0.947, +2.387 | −0.643, −0.591 |
| 513 `p0` baseline | −0.401, +0.735 | +1.188, +0.235 | −0.576, −0.427 | +0.737, +0.559 |
| 513 `p0` candidate | −0.402, −0.713 | −0.094, −0.783 | −0.069, +0.478 | +1.025, +0.415 |
| 513 `p2` baseline | +2.451, +3.683 | −0.273, +0.632 | −1.429, −5.145 | +0.566, +0.422 |
| 513 `p2` candidate | +1.428, −0.026 | −1.655, +0.425 | −4.044, −2.931 | +4.444, −3.054 |
| 1025 `p2` baseline | −2.672, −1.863 | −1.115, +2.814 | −1.377, −2.334 | +4.160, −0.116 |
| 1025 `p2` candidate | +0.134, +0.432 | +3.742, +1.612 | −1.196, −5.438 | +4.260, −1.647 |
| 4K baseline | −0.802, −1.337 | −1.525, +0.700 | −1.303, +0.559 | −2.925, −1.082 |
| 4K candidate | −1.608, −0.724 | −0.307, +0.609 | −1.971, −1.995 | −0.719, −0.206 |

## Targeted follow-up

The recurring whole-flower3200 and retained-4K/auto regressions motivate a
separate 24-job follow-up on exactly those two case/mode pairs. Both budgets,
creation orders, and both-build controls are included, with four warmup and
sixteen measured duplicate-ABBA rounds. The predeclared totals are 1,536 measured
calls, 384 warmups, 24 oracle calls, and 24 untimed whole captures. The unchanged
standard binaries and original analysis hash are pinned before starting at
2026-09-10 00:01:05 UTC. This is not a second full-corpus campaign, and its
results will not be pooled with the original or used for image-specific dispatch.
All 24 jobs completed successfully at 2026-09-10 00:06:19 UTC, with every
predeclared count above plus 3,840 NVML observations of a 40 W limit. No
timing retry, source/binary change, build, sanitizer, or result exclusion occurred.

All follow-up primary paired percentages are below; negative means faster.

| Case/mode/budget/order | Outer | Codestream wall | Entropy optimization wall | Section wall | Token-worker sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flower3200 whole auto O0 | +0.523 | +2.195 | +3.794 | −1.324 | −0.632 |
| Flower3200 whole auto O1 | +0.296 | −5.641 | −6.857 | −10.454 | −8.945 |
| Flower3200 whole 8 O0 | −2.967 | −1.744 | +1.506 | −3.075 | −2.774 |
| Flower3200 whole 8 O1 | −1.704 | −3.307 | −6.817 | −9.241 | −5.563 |
| 4K retained auto O0 | −2.430 | −1.936 | −3.355 | −3.000 | −2.912 |
| 4K retained auto O1 | +2.045 | +3.053 | −0.005 | −0.405 | −5.085 |
| 4K retained 8 O0 | +0.681 | +0.402 | +1.579 | −4.711 | −2.431 |
| 4K retained 8 O1 | −2.381 | −0.184 | −0.825 | −2.153 | −2.687 |

All follow-up control outer percentages, in O0/O1 order:

| Case/mode/build | Auto | Eight |
| --- | --- | --- |
| Flower3200 whole baseline | +0.198, −0.613 | +2.988, −2.557 |
| Flower3200 whole candidate | +0.753, −1.533 | −3.454, +0.140 |
| 4K retained baseline | +0.719, −2.533 | −0.605, −1.352 |
| 4K retained candidate | −0.623, +2.948 | −3.360, +0.246 |

Whole flower3200 outer time improves in two of four trials and 13/16
cross-label comparisons. Both eight-thread trials improve, including all
eight cross-label comparisons, while automatic-thread paired medians remain
slower by +0.296–0.523%. Its section and token-worker medians improve in all
four trials (13/16 and 14/16 cross-label comparisons respectively). The original
four larger whole regressions are not reproduced uniformly, but the follow-up
does not prove that this input is regression-free.

Retained 4K outer improves in two of four trials and 9/16 cross-label
comparisons. The automatic-thread O0 paired result flips from +4.790% to
−2.430%, but three of its four cross-label comparisons remain slower. O1 is
+2.045% paired despite all four cross-label comparisons being faster. Section
and token-worker medians improve in all four trials, with 11/16 and 12/16
cross-label comparisons faster. These aggregations are retained separately;
they do not justify averaging away the unfavorable results or assigning a
causal explanation to the changes. Controls are not subtracted.

## Retention and next investigation

Retain the ten-line aligned-copy path and its permanent contract test. The
full-corpus dense 513/1025 trials improve in every whole/retained, budget,
creation-order, and cross-label comparison for outer time and the changed
worker/section phases. Most overall primary trials improve, while exact bytes,
logical lengths, atomic failures, normal tests, ASAN suites, and CUDA checks pass.
This decision accepts the documented mixed image-fixture measurements; it is
not a universal throughput claim or a claim that S167/S168's 4K concerns are
resolved. No image-specific policy, compatibility path, or public knob is added.

The final verifier recomputes both timing campaigns independently and checks
the frozen prototype plus all 243 accepted promotion journals, with no promotion
rejections. It preserves the prototype's one failed-build journal and both
analyzer/verifier correction histories. Its frozen mode additionally verifies
the complete artifact inventory and every source/helper/archive hash. Prototype
ASAN runs, 92 CTests, the separate install-consumer check, 44 standard comparisons,
and two standard CUDA checks remain explicitly distinguished.

Next, refresh fully resident whole-pipeline/GPU attribution against the actual
retained runtime, including native sparse handoff and the newer CPU tail.
S153's pre-sparse GPU ranking and S165's pre-batching writer decomposition are
historical guidance, not current bottleneck measurements. The internal GPU
timestamp profile entry is Metal-specific; the established CUDA route is the
separate CUDA/NVTX system-trace workflow. Trace and ordinary timings must remain
separate, with no clock/power/security changes. Do not reinstate rejected tile
schedules or select another writer tweak without fresh whole-path evidence.
The optimization goal remains open.

No build or sanitizer may overlap timing. No OS power, clocks, affinity,
priority, security, or firewall configuration is changed. Whole fully resident
encoding remains the end-to-end criterion; a faster append/worker phase alone
does not establish a whole-encode gain. The earlier 4K limitation remains an
explicit evaluation concern.

## Evidence and storage

Preparation verified all 2,524 frozen S167 artifact files and archived 342
production inputs. New artifacts and compiler temporaries are under
`U:/gjxl-cuda-diagnostics/s168-artifacts`; C: has only about 79 MB free during
qualification. No data cleanup, deletion, or relocation is performed in this
stage. Protected untracked Markdown files remain unread and excluded.
No privilege/firewall blocker has been observed. The optimization goal is open.
