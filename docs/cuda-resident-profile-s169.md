# Fully resident profile after sparse handoff and ANS writer changes (S169)

## Scope and qualification

Starting production revision is S168, `eede5222b7aad276baf9769522a17073a3f1881a`.
This checkpoint refreshes attribution; it does not change production code,
coefficient-width defaults, thread policy, public APIs, or compatibility behavior.
It is not an S153-versus-S168 or candidate-versus-baseline speedup experiment.
The fully resident backend is not established to be maxed out.

The normal harness loads the frozen, standard-build S168 `whole.dll` directly.
The separate host-ASAN bridge combines the qualified S168 bit writer, S167 ANS
writer, S166 encoder, and S162 instrumented support libraries. The unchanged
PFM reader library is not ASAN-instrumented. Both DLLs contain exactly the same
eleven native CUDA module hashes as qualified S168. There is no diagnostic GPU
override, new event instrumentation, or alternative runtime implementation.
The prior S168 manifest and 343 production-source files are checked and archived.

The eleven established cases comprise six frozen image fixtures and five
deterministic synthetic inputs: 65x71 pattern 2, 513x519 patterns 0/1/2,
and 1025x1031 pattern 2. Larger Flower and Keong fixtures are nearest-replicated
photographs, not independent native-resolution photographs. Settings are
fully resident CUDA, effort 7, final Butteraugli score disabled, and distance
1.2 except synthetic pattern 2 at 0.01. Automatic and eight requested CPU
threads are separate processes. The harness does not expose a measured peak
participant count; this is not a fresh thread-budget implementation test.
Default native sparse handoff is enabled and compact-width packing is off.
An observed four-byte coefficient width does not imply dense storage.

Every process first produces an untimed reference using frozen normal S168.
The six image references must also equal their frozen codestream files.
Every subsequent warmup and measured output must match reference bytes,
the complete public summary, coefficient width, and native storage size.
All 41 host phase durations must be finite and nonnegative. Input file reading,
reference encoding, backend setup, result queries, result clearing, verification,
NVML calls, and logging are outside the measured encode interval.

Qualification consists of 44 normal/ASAN processes covering every case/budget,
then full-leak CUDA memcheck on padded 4K and initcheck on Keong/eight threads.
Each has one warmup, two subsequent checks, and one untimed reference.
These are correctness checks, not timing observations. Production sources are
unchanged, so no new CTest run, downstream install test, decoded-quality cohort,
or cross-platform qualification is claimed in S169; S168 retains those gates.

## Measurement protocol

One separate normal 4K/automatic smoke trace validates NVTX range selection,
stdout extraction, SQLite schema, and work identity. The broad campaign has
44 ordinary processes followed by 44 trace processes: eleven cases, two thread
budgets, and two repeats. A single shuffle uses seed `16920260910`; repeat two
reverses it. Every ordinary process has four warmups and twelve measured calls;
every trace process has four warmups and five measured calls. The smoke is not
pooled with the broad traces. Per-process medians and their ranges are retained;
they are neither confidence intervals nor causal thread-count comparisons.

Nsight Systems 2023.2.3 records CUDA/NVTX with CPU sampling and context-switch
collection disabled, graph-node tracing enabled, and full-process capture.
Analysis selects only `S169 encode index=...` ranges, excluding the untimed
reference and excluding warmups from timing summaries. It checks successful
CUDA API returns, range/host-timer agreement, complete activity containment,
and stable measured kernel geometry/resources and ordered copy sizes within
each process. Full names and launch resources remain in the artifact record.

CPU worker-work counters can overlap; they are not additive wall phases.
CUDA API waits overlap GPU execution. Device-span gaps are not automatically
removable launch overhead, and the post-device tail is not all serialization.
No claim of a bandwidth ceiling follows from these traces.

The Windows RTX 3060 Laptop environment uses CUDA 11.8, MSVC 14.37 and
Clang 22 host ASAN. Read-only NVML enforced-limit endpoints bracket the measured
and warmup calls, not the untimed reference. Equal endpoints do not prove
constant clocks, thermals, cache state, or absence of preemption. No power,
clock, thermal, affinity, priority, firewall or security setting is changed.
No privilege/firewall blocker appeared in this checkpoint.

## Preserved tooling failures and storage recovery

The first harness build failed under `/WX`: renaming the old comparator's
`main` exposed its implicit-return warning. It ran no encoder. A separate v2
source reuses only the DLL-loading helper and builds successfully; the original
source, build inputs, log and rejected journal remain unchanged.

The smoke left three generated Windows ETL scratch files totaling 587,202,560
bytes, despite exported Nsight/SQLite reports totaling only 1,446,899 bytes.
The original campaign completed all ordinary jobs, then stopped at its 250 MB
free-space guard before the first broad trace. The first archival helper failed
before copying/deleting because child Windows PowerShell could not resolve
`Get-FileHash`; the corrected launcher uses the observed PowerShell 7.6.5.

Each completed capture's exact three ETL files is ZIP-archived, decompressed and
SHA-256 checked against the originals before only those individually resolved
scratch files are removed. They remain recoverable from `scratch_archives/`.
No frozen predecessor or unrelated data is deleted, moved, or overwritten.
Lossless NTFS compression is used on new archived sources, verified scratch
ZIPs, completed reports, and later unused build artifacts/extracted cubins;
all logical byte hashes and lengths remain unchanged. The normal measured
EXE and frozen production DLL are excluded.

Source/archive compression overlapped `trace_r0_4k_t8`; that diagnostic trace
is retained with this caveat. Ordinary timing had already finished. Subsequent
report/build compression occurred while the trace driver was stopped.

The first `trace_r1_1080p_t8.sqlite` export was zero-length although Nsight
returned zero and printed its export completion. Tight disk space makes
exhaustion plausible, but no explicit ENOSPC diagnostic was captured.
That empty file, original journal, and original `.nsys-rep` are preserved.
After reclaiming scratch space, the unchanged report is re-exported to the
distinct `trace_r1_1080p_t8_reexport.sqlite`; all nine original encode ranges
and exact-output checks are recovered. No timestamp shifting or encoder rerun
is used. A separate v2 analyzer accepts this explicit file mapping and keeps
the original analyzer unchanged.

Two later 600 MB guard pauses occurred before further captures. Unused new
artifacts and completed reports were compressed while stopped. For the final
two captures, the guard was set to 595,202,560 bytes: the observed 587,202,560-byte
ETL triple plus eight million bytes for exports, ZIP creation and margin.
The original order resumed. These pauses and helper versions are recorded
explicitly; no timing sample is discarded, substituted, or repeated to improve
a result.

## Results

All 135 encoder processes pass: 1,247 warmup/subsequent exact-output checks
and 135 untimed references, totaling 1,382 encodes. Both CUDA sanitizer runs
report zero errors and memcheck reports zero leaked bytes. All 2,494 observed
power-limit endpoints are 40,000 mW. The broad cohorts contain 528 measured
ordinary calls and 220 measured traced calls; their warmups and the smoke,
preflight and sanitizer results are not pooled into those distributions.
Across all four broad traces per case, kernel work/geometry, ordered transfer
sizes, and storage sizes agree. All 190 job journals are terminal: 188 accepted
and the two preserved tooling failures above. No encoder process is repeated.

Ordinary jobs run from 00:31:15.745235 to 00:33:10.563097 UTC on September 10,
2026. Broad traces span 00:36:22.257229 to 00:55:04.071964 UTC, including the
explicit storage pauses; ETL archival finishes afterward. These are separate
attribution cohorts, not a paired comparison with an earlier runtime.

Padded 4K whole-call medians span 265.394–270.708 ms; Keong spans
294.998–300.645 ms. At 4K, quantization still takes 192.785–198.480 ms,
while codestream encoding takes 49.313–52.263 ms. Dense 1025/p2 instead
spends 56.118–67.162 ms in codestream encoding out of 88.619–102.961 ms
whole calls. A single optimization priority does not fit all these inputs.

The six image fixtures and 513/p0 use sparse int32 ownership; the other four
synthetic cases remain dense. Padded 4K stores/transfers 5,779,732 native AC
bytes, versus 106,168,320 bytes in a padded dense owner and 99,532,800 active
dense coefficient bytes. Its 4,665,600 header bytes accompany 1,114,132 value
bytes (278,533 nonzeros); the four-byte count and frame metadata are separate.
Keong has the same header size and 2,597,496 value bytes (649,374 nonzeros).
The sparse packer's 2.483–3.130 ms at padded 4K exceeds its 0.926–1.081 ms
native AC copy duration. That supports inspecting the packer, not claiming
that an unmeasured scheduling change will improve the whole encoder.

The original three RGB uploads still move 99,460,812 bytes and take
15.293–15.807 ms across the two 4K cases' trace-process medians. This does not
prove pinned staging or an input cache would pay for its setup and lifetime.
Composition/anchor reduction is already only 1.280–1.785 ms across those
cases, versus 16.909–23.966 ms for Malta and 14.836–20.856 ms for fused AC.
The packer is a bounded next experiment, not the largest remaining family.

## Ordinary host timings

Ranges of two process medians (ms), separately for each requested thread budget. Nested phases and worker-work sums are not additive wall time.

| Case / threads | Whole | Input | Quantization | Codestream | AC tokenize | Entropy | Section write |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flower_500 / auto | 20.893–26.770 | 0.866–0.953 | 9.341–11.063 | 10.341–14.661 | 2.965–3.909 | 3.784–5.727 | 2.621–3.159 |
| flower_500 / 8 | 20.959–21.834 | 0.887–0.921 | 9.291–9.409 | 10.439–11.424 | 2.723–3.017 | 3.916–4.440 | 2.678–2.714 |
| 1080p / auto | 64.295–65.446 | 4.614–4.632 | 35.452–35.904 | 23.506–24.203 | 5.644–5.910 | 8.503–8.782 | 5.142–5.233 |
| 1080p / 8 | 64.973–67.742 | 4.612–4.718 | 35.765–36.954 | 24.170–25.194 | 5.334–5.666 | 9.009–9.554 | 5.309–5.418 |
| flower_2000 / auto | 115.053–124.589 | 8.719–8.862 | 69.157–70.502 | 36.709–43.365 | 8.227–9.365 | 11.636–14.194 | 8.878–10.375 |
| flower_2000 / 8 | 113.922–118.861 | 8.601–8.672 | 68.550–70.327 | 35.396–38.662 | 7.506–8.013 | 11.764–12.547 | 8.574–9.355 |
| 4k / auto | 265.500–266.499 | 17.226–17.243 | 192.785–193.563 | 51.152–51.174 | 10.542–10.655 | 16.096–16.461 | 9.198–9.240 |
| 4k / 8 | 265.394–270.708 | 17.223–17.361 | 194.820–198.480 | 49.313–52.263 | 9.977–10.406 | 16.196–16.551 | 8.841–9.544 |
| flower_3200x2160 / auto | 211.718–218.451 | 14.386–14.508 | 145.116–145.341 | 50.680–55.605 | 11.151–12.428 | 14.390–17.719 | 10.908–12.101 |
| flower_3200x2160 / 8 | 213.673–216.968 | 14.416–14.452 | 146.363–147.987 | 47.396–50.674 | 10.720–10.790 | 13.927–15.293 | 10.403–10.593 |
| keong_3839x2159 / auto | 296.952–300.157 | 17.269–17.291 | 189.016–194.511 | 80.077–86.559 | 20.824–20.905 | 24.706–26.285 | 15.133–15.879 |
| keong_3839x2159 / 8 | 294.998–300.645 | 17.207–17.321 | 193.107–194.741 | 77.291–90.534 | 19.721–21.800 | 25.756–30.525 | 14.269–16.467 |
| 65_p2 / auto | 10.324–16.614 | 0.201–0.333 | 4.036–6.655 | 6.030–9.274 | 1.410–2.323 | 2.926–4.531 | 1.337–1.796 |
| 65_p2 / 8 | 9.620–10.107 | 0.195–0.205 | 3.915–4.047 | 5.487–5.781 | 1.225–1.298 | 2.797–2.941 | 1.215–1.325 |
| 513_p0 / auto | 15.427–16.221 | 1.128–1.175 | 8.412–8.823 | 5.830–5.972 | 2.137–2.228 | 1.254–1.319 | 1.672–1.695 |
| 513_p0 / 8 | 14.101–15.013 | 1.076–1.084 | 7.843–8.672 | 5.000–5.028 | 1.736–1.793 | 1.014–1.062 | 1.464–1.573 |
| 513_p1 / auto | 30.002–40.398 | 1.080–1.317 | 10.848–14.064 | 17.264–24.998 | 4.618–7.552 | 6.144–8.184 | 4.996–7.550 |
| 513_p1 / 8 | 30.352–30.712 | 1.104–1.111 | 11.151–11.470 | 17.596–17.622 | 4.656–4.674 | 6.233–6.290 | 5.164–5.186 |
| 513_p2 / auto | 38.512–39.470 | 1.125–1.207 | 11.223–11.290 | 25.392–26.428 | 5.235–5.599 | 6.854–7.012 | 11.038–11.232 |
| 513_p2 / 8 | 33.910–37.772 | 1.084–1.137 | 9.975–11.298 | 22.250–25.156 | 4.550–5.210 | 5.967–6.607 | 9.479–11.024 |
| 1025_p2 / auto | 90.653–98.017 | 2.778–2.780 | 27.902–29.320 | 58.737–63.556 | 13.723–15.112 | 11.596–12.466 | 25.826–27.541 |
| 1025_p2 / 8 | 88.619–102.961 | 2.780–2.842 | 27.710–30.504 | 56.118–67.162 | 12.962–15.609 | 10.839–13.033 | 24.676–30.250 |

## Trace GPU attribution

Ranges of four process medians (two requested budgets, two repeats), ms. Each family is summed within an encode before taking its process median. Separate diagnostic traces, not ordinary performance measurements.

| GPU interval/family | 4k | keong_3839x2159 | flower_3200x2160 | 1025_p2 |
| --- | ---: | ---: | ---: | ---: |
| All kernels (union) | 137.502–148.193 | 133.681–140.812 | 99.436–107.478 | 13.821–13.849 |
| All device activity (union) | 156.388–167.600 | 153.141–160.094 | 115.420–123.252 | 18.458–18.724 |
| Device-span gaps | 29.560–33.441 | 29.044–40.872 | 25.734–29.959 | 10.444–14.147 |
| After final device activity | 53.345–67.350 | 82.523–101.524 | 57.250–74.866 | 59.194–70.205 |
| Malta | 19.274–23.966 | 16.909–18.473 | 14.101–14.807 | 1.557–1.562 |
| Fused AC evaluators | 16.220–19.717 | 14.836–20.856 | 11.711–17.691 | 1.724–1.728 |
| Erosion/L2/final | 8.731–10.749 | 7.915–8.658 | 5.995–6.322 | 0.913–0.913 |
| Mirrored opsin | 7.703–8.320 | 7.452–8.195 | 5.047–5.882 | 0.635–0.637 |
| Low/medium vertical | 7.947–8.411 | 7.731–7.944 | 6.020–6.310 | 0.937–0.938 |
| Gaborish/EPF | 4.958–7.330 | 4.883–6.923 | 3.584–4.640 | 0.313–0.315 |
| Composition/anchor reduction | 1.280–1.517 | 1.646–1.785 | 0.928–1.337 | 0.100–0.101 |
| Generic maximum reduction | 0.026–0.032 | 0.028–0.030 | 0.020–0.029 | 0.012–0.012 |
| Sparse AC packing | 2.483–3.130 | 2.357–2.447 | 1.780–2.026 | 0.000–0.000 |
| Resident coefficient encoding | 5.550–6.588 | 6.722–7.706 | 4.231–5.324 | 0.694–0.696 |

## Coefficient handoff

Bytes are exact, not timed estimates. AC transfer excludes the four-byte sparse counter; metadata is separate. Dense owner includes group padding. Timing ranges are four trace-process medians (ms).

| Case | Native kind | Native owner / dense owner bytes | AC transfer bytes | AC copy ms | Sparse pack ms | Original RGB upload ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| flower_500 | sparse32 | 302,468 / 3,145,728 | 302,468 | 0.049–0.049 | 0.024–0.024 | 0.462–0.464 |
| 1080p | sparse32 | 1,450,028 / 31,457,280 | 1,450,028 | 0.224–0.224 | 0.180–0.191 | 3.805–3.809 |
| flower_2000 | sparse32 | 2,893,140 / 50,331,648 | 2,893,140 | 0.447–0.447 | 0.489–0.724 | 7.331–7.483 |
| 4k | sparse32 | 5,779,732 / 106,168,320 | 5,779,732 | 0.926–1.081 | 2.483–3.130 | 15.293–15.776 |
| flower_3200x2160 | sparse32 | 5,041,160 / 92,012,544 | 5,041,160 | 0.923–0.994 | 1.780–2.026 | 12.645–12.810 |
| keong_3839x2159 | sparse32 | 7,263,096 / 106,168,320 | 7,263,096 | 1.285–1.690 | 2.357–2.447 | 15.377–15.807 |
| 65_p2 | dense32 | 786,432 / 786,432 | 62,208 | 0.010–0.010 | 0.000–0.000 | 0.016–0.017 |
| 513_p0 | sparse32 | 152,192 / 7,077,888 | 152,192 | 0.028–0.028 | 0.023–0.023 | 0.504–0.504 |
| 513_p1 | dense32 | 7,077,888 / 7,077,888 | 3,244,800 | 0.505–0.505 | 0.000–0.000 | 0.503–0.504 |
| 513_p2 | dense32 | 7,077,888 / 7,077,888 | 3,244,800 | 0.505–0.506 | 0.000–0.000 | 0.504–0.505 |
| 1025_p2 | dense32 | 19,660,800 / 19,660,800 | 12,780,288 | 2.160–2.431 | 0.000–0.000 | 1.959–1.963 |

## Next bounded investigation

Investigate the newer native sparse packer's metadata writes and block
scheduling without changing the sparse consumer or introducing a compatibility
layer. `SparseAc<T>` currently handles 256 coefficients per block, uses two
block barriers and a thread-zero prefix loop, reserves a block payload with
one atomic when nonempty, and emits four masks/four offsets from thread zero.
At padded 4K this schedules 97,200 blocks over 24,883,200 coefficients.

First inspect generated native stores and census active/empty blocks on the
current retained frames. Test distributing contiguous header stores across
lanes, and separately test bounded multi-item block coarsening to reduce
block/atomic overhead. These are hypotheses, not proof that atomics, stores,
or occupancy currently limit the kernel. Preserve 64-bit mask semantics,
per-word offsets, exact values, tail handling, and nonzero-count validation.
Do not assume physical payload order is fixed: the existing atomic reservation
already permits a different allocation order while offsets define semantics.

Use native-identical duplicate controls, exact reconstructed coefficients and
codestreams, all storage widths and partial/empty/dense cases, host ASAN and
appropriate CUDA memory/race/synchronization checks. A local kernel gain must
then survive fresh fully resident whole encodes, including packing, allocation,
readback, native consumption and destruction. Keep the current dense policy
unless a separately qualified experiment supports changing it.

The broader alternatives remain constrained by prior evidence: composition and
anchor reduction are already fused in [S111](cuda-compose-aq-s111.md);
[S112](cuda-tile-scheduling-s112.md) and [S140](cuda-ac-scheduling-s140.md)
do not justify restoring rejected schedules based only on local improvements.
[S67](cuda-optimization-s1.md#reuse-dead-prepared-butteraugli-planes-s67)
rejected eroded/nonlinear reference-mask caches even with preparation excluded.
The repeated arithmetic visible in source is therefore not sufficient reason
to reinstate those caches. This study selects the next test, not a promised
whole-encoder saving or a declaration that other avenues are exhausted.

## Evidence and replay

Evidence root: `U:/gjxl-cuda-diagnostics/s169-artifacts/`. The original and
corrected builds, protocols, serial job journals, all Nsight/SQLite exports,
recovered export, extracted stdout, scratch ZIPs and removal records, compression
records, `analysis.json`, `handoff.json`, `report.json`, and `tables.md` are retained.
`handoff.json` independently reconstructs dense geometry or sparse header/value
sizes, identifies the unique ordered final readback sequence, requires it to
end at the final D2H copy, and separately identifies the three original RGB
uploads. It does not label unrelated metadata/cost copies as AC payload.

`s169_verify_final.py` recomputes the analyses, checks every job and artifact hash,
the eleven-module identities, recoverable ZIP identities, and the unchanged
S168 predecessor inventory. `s169_freeze_final.py` is an exclusive one-time finalizer;
after it finishes, verify with:

```powershell
python -B build-cuda-ninja/profiles/s169_verify_final.py --frozen
```

Only this document is committed. No external push is performed. The three
protected untracked Markdown files are not read, edited or staged.
