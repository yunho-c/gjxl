# S194: resident attribution after packed ANS and local AC output

Date: 2026-09-10. Production is S193, `3567e8b`.
This stage refreshes attribution after the qualified CPU-side integration.
It does not change production, coefficient policy, GPU kernels, thread
scheduling, compatibility behavior or machine settings. The backend is not
established to be maxed out.

## Ordinary evidence and focused traces

All 88 integrated-policy processes from the frozen S193 campaign are reused
for ordinary host attribution: eleven inputs, automatic/eight requested threads,
two repeats and two duplicate processes, each with 32 measured encodes.
The 2,816 measured calls are not rerun. First reduce within each process,
then report the range of four process medians per input/budget. All 41 phases
remain available. Phase shares divide each duration by its own whole call
before reduction; nested phases and parallel worker work are not additive.

Inputs and resident encoding settings are the established S193 fixtures and
bridge configuration, not regenerated images. The large Flower/Keong images
include nearest-replicated photos, not independent native-resolution photos.
Distance is 1.2 except synthetic pattern two at 0.01; no quality target changes.

Fresh GPU attribution focuses on 4K, Keong, dense 1025/p2 and sparse 513/p0.
The photos cover large resident GPU work, 1025/p2 covers CPU-heavy dense output,
and 513/p0 retains a small-input regression seen in S193. Both automatic and
eight requested threads are covered. These are diagnostic cohorts, not a
new old-versus-new speed comparison or a causal thread-budget experiment.

The frozen normal/ASAN S169 NVTX executables load the exact S193 qualified
whole DLLs. Their unloaded reference is normal S168, with frozen image oracles
also checked. No harness, production library or CUDA module is rebuilt.
The 345 current production/test source records and S193's 2,031-file inventory
are verified. S193 retains the standard-build, CTest, installed-consumer,
ASAN fixture, CUDA sanitizer and native-link qualification; this stage does
not claim a fresh run of those unchanged-code suites.

Sixteen normal/ASAN preflight processes cover the four inputs and both budgets,
each with one warmup, one subsequent encode and one reference. A separate
4K/automatic smoke trace has four warmups and three measured windows. The
eight focused cells are shuffled with seed 19420260910, then reversed for a
second repetition. Each of the sixteen broad traces has four warmups, five
measured calls and one reference. No performance-dependent stopping, filtering,
sample substitution or encoder rerun is allowed.

Every warmup and measured output checks exact bytes, the full public summary,
four-byte coefficient width and native storage size. All 41 host phases must
be finite/nonnegative, whole time positive, and output sizes match the qualified
census. NVML endpoints check the unchanged 40,000 mW enforced limit. Input
setup, reference, output queries/checks/clearing, logging and NVML sampling are
outside the measured encode interval. Default native sparse ownership is used;
compact-width packing is off. Four-byte values do not imply a dense owner.

Nsight Systems 2023.2.3 captures CUDA/NVTX and graph-node activity, with CPU
sampling and context-switch collection disabled. Analysis retains the sealed
`S169 encode index=...` range names; only nonnegative measured indices enter
trace summaries. It checks range/host-timer agreement, successful CUDA returns,
complete containment including activity crossing either range boundary, kernel
geometry/resources, ordered copy kinds/bytes and memset sizes. The reference,
warmups and separate smoke are not pooled into broad trace timing summaries.

## Storage pause and exact continuation

The original protocol required 3.3 GB free before an encoder/profiler launch,
with 1 GB before a preservation utility. Every capture has its own TEMP directory.
After terminal completion, its exact three ETLs were NTFS-compressed and
hash/length checked, with no overlapping encoder or heavy diagnostic work.

After the smoke and six broad traces, free C reached 3,285,860,352 bytes and
the driver stopped at the original guard before creating the next job's intent
or launching a child. All sixteen preflights and seven captures were already
terminal and successful. This was not an observation timeout or an encoder,
profiler, firewall or privilege failure.

NTFS compression stored one 587,202,560-byte ETL triple in 36,868,096 physical
bytes in the inspected 4K/eight record. That remaining allocation accumulated
across captures. A new, separately pinned resume protocol changes only scratch
preservation: write a ZIP, decompress/hash every entry against the original
record, recheck the original file, record verification, then remove exactly the
three individually resolved task-owned ETLs. ZIPs retain every original byte.
Reports, SQLite exports, stdout, journals, source archives and frozen predecessor
artifacts are not removed or overwritten. Reparse ancestors/targets are rejected.

Seven completed captures are archived while the driver is stopped. The ten
remaining trace specs then retain their original order, settings, warmups and
sample counts. Each still performs the original NTFS-compression step, then
verified ZIP archival. The 3.3 GB launch guard is unchanged. Original scripts,
protocol and compression records are preserved; the revised verifier checks
recoverable ZIP contents instead of requiring removed scratch paths to exist.
No power, clock, cooling, affinity, priority, firewall or security setting changes.

## Completed qualification and attribution

All 33 encoder processes exit zero: sixteen normal/ASAN preflights, one smoke
and sixteen broad traces. The 183 warmup/subsequent checks plus 33 references
total 216 new encodes; broad traces contribute 80 measured windows and the
separate smoke three. All 366 NVML endpoints report 40,000 mW. The reused
ordinary cohort remains 88 processes/2,816 measured calls, not new operations.
All 67 journaled child jobs are terminal zero, including seventeen NTFS-compression and
seventeen ZIP-preservation jobs. The launcher-level storage guard is separately
preserved; it did not produce a failed encoder job or launch the next case.

The first preflight started 15:33:26.445949 UTC. The final trace exited
15:43:18.359771, and its archival finished 15:43:24.297899. This span includes
the explicit storage pause and recovery, not uninterrupted GPU execution.
All 51 scratch ETLs, totaling 9,982,443,520 logical bytes, are preserved in
seventeen byte-verified ZIPs totaling 11,006,593 bytes. Only their resolved task-owned scratch originals
were removed; they are recoverable. Exported reports and SQLite files remain.

Across the four focused cases, all four current traces and all four frozen S169
traces have identical complete kernel names/counts, launch geometry, registers,
shared/local-memory records, ordered copy kinds/bytes and memset sizes. Native
owner and output sizes also agree. This is structural identity, not a temporal
speed comparison or proof of identical host code, clocks or thermals.

Current ordinary 4K quantization medians range 192.744–207.602 ms out of
259.957–281.841 ms whole calls. Dense 1025/p2 instead spends 47.672–58.926 ms in
codestream work out of 79.062–94.168 ms whole calls. These are descriptive ranges
of process medians, not simultaneous worst cases or additive phase components.

The GPU refresh does not make the already fused composition/reduction the
largest opportunity: at 4K composition/anchor reduction is 1.238–1.562 ms and
generic maximum reduction 0.025–0.032 ms, versus Malta 19.711–21.623 ms and fused
AC evaluators 15.197–23.045 ms. Sparse AC transfer is 0.992–1.167 ms for exactly
5,779,732 bytes, while sparse packing takes 2.484–3.182 ms. Original RGB upload
still transfers 99,460,812 bytes in 15.318–17.470 ms. These are trace-process
median ranges, not a predicted whole-encode saving from removing any component.

## Ordinary host timings

Ranges of four integrated S193 process medians (ms), separately for each requested thread budget. Nested phases and worker-work sums are not additive wall time.

| Case / threads | Whole | Input | Quantization | Codestream | AC tokenize | Entropy | Section write |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flower_500 / auto | 18.017–19.215 | 0.830–0.837 | 8.481–8.809 | 8.543–9.527 | 2.557–2.848 | 3.145–3.535 | 2.057–2.258 |
| flower_500 / 8 | 17.228–19.720 | 0.819–0.864 | 8.308–9.206 | 8.044–9.497 | 2.229–2.618 | 3.026–3.564 | 2.037–2.347 |
| 1080p / auto | 63.587–65.892 | 4.635–4.658 | 35.513–36.468 | 22.358–23.896 | 5.646–5.948 | 8.175–8.810 | 4.554–5.021 |
| 1080p / 8 | 63.537–65.981 | 4.630–4.666 | 35.810–36.549 | 21.871–24.072 | 5.050–5.493 | 8.310–9.090 | 4.656–5.159 |
| flower_2000 / auto | 111.223–120.144 | 8.621–8.706 | 67.744–73.478 | 32.062–38.210 | 7.507–8.352 | 10.448–12.761 | 7.028–8.386 |
| flower_2000 / 8 | 114.707–121.079 | 8.645–8.724 | 69.177–71.659 | 35.379–38.121 | 7.647–8.241 | 11.522–12.804 | 7.863–8.645 |
| 4k / auto | 259.957–281.841 | 17.226–17.370 | 192.744–207.602 | 45.648–52.872 | 9.753–11.058 | 14.845–16.667 | 7.935–9.049 |
| 4k / 8 | 272.213–279.519 | 17.303–17.354 | 200.157–205.198 | 46.558–52.581 | 9.722–10.458 | 14.948–17.060 | 8.218–8.992 |
| flower_3200x2160 / auto | 198.030–220.806 | 14.383–14.459 | 135.226–152.730 | 42.366–53.899 | 10.081–12.133 | 13.236–16.445 | 8.977–11.041 |
| flower_3200x2160 / 8 | 208.079–219.345 | 14.430–14.565 | 143.553–152.208 | 45.477–49.563 | 10.325–10.897 | 13.904–15.783 | 9.158–9.931 |
| keong_3839x2159 / auto | 290.888–330.977 | 17.332–17.516 | 197.826–210.462 | 71.271–96.677 | 18.556–23.301 | 21.546–31.851 | 13.603–17.051 |
| keong_3839x2159 / 8 | 283.907–309.962 | 17.361–17.496 | 191.275–203.446 | 68.927–86.082 | 18.249–22.500 | 22.398–28.398 | 12.490–14.696 |
| 65_p2 / auto | 9.642–11.932 | 0.207–0.237 | 3.921–4.735 | 5.408–6.958 | 1.260–1.655 | 2.734–3.660 | 1.113–1.389 |
| 65_p2 / 8 | 9.336–9.976 | 0.200–0.224 | 3.944–4.108 | 5.106–5.570 | 1.148–1.273 | 2.642–2.884 | 1.073–1.197 |
| 513_p0 / auto | 13.217–15.963 | 1.022–1.162 | 7.422–8.614 | 4.766–6.110 | 1.720–2.285 | 1.018–1.361 | 1.366–1.759 |
| 513_p0 / 8 | 13.454–14.602 | 1.034–1.087 | 7.564–8.288 | 4.557–5.069 | 1.635–1.811 | 0.908–1.022 | 1.372–1.482 |
| 513_p1 / auto | 28.426–34.276 | 1.067–1.204 | 10.677–11.978 | 15.988–20.715 | 4.542–5.825 | 5.503–7.192 | 4.494–5.879 |
| 513_p1 / 8 | 28.239–30.265 | 1.083–1.106 | 10.721–11.125 | 15.713–17.583 | 4.329–4.754 | 5.494–6.186 | 4.518–4.997 |
| 513_p2 / auto | 32.405–33.131 | 1.090–1.125 | 10.654–10.876 | 20.020–20.672 | 4.794–4.837 | 5.825–6.015 | 7.295–7.521 |
| 513_p2 / 8 | 30.357–37.408 | 1.096–1.208 | 10.291–11.573 | 18.243–23.266 | 4.172–5.270 | 5.345–6.979 | 6.763–9.000 |
| 1025_p2 / auto | 79.062–94.168 | 2.764–2.831 | 27.578–30.244 | 47.672–58.926 | 11.757–14.681 | 10.250–12.440 | 18.737–23.526 |
| 1025_p2 / 8 | 81.168–92.944 | 2.784–2.806 | 28.343–29.786 | 48.694–58.039 | 11.926–13.605 | 10.800–12.370 | 20.113–23.248 |

## Trace GPU attribution

Ranges of four process medians (two requested budgets, two repeats), ms. Each family is summed within an encode before taking its process median. Separate diagnostic traces, not ordinary performance measurements.

| GPU interval/family | 4k | keong_3839x2159 | 1025_p2 | 513_p0 |
| --- | ---: | ---: | ---: | ---: |
| All kernels (union) | 138.496–145.802 | 132.686–140.775 | 13.815–13.823 | 4.681–4.699 |
| All device activity (union) | 157.674–166.211 | 152.026–160.065 | 18.502–18.788 | 5.369–5.391 |
| Device-span gaps | 21.870–30.146 | 30.658–35.336 | 9.689–13.156 | 3.667–4.368 |
| After final device activity | 47.877–58.531 | 73.717–81.943 | 45.682–68.408 | 4.609–6.089 |
| Malta | 19.711–21.623 | 17.309–18.713 | 1.558–1.559 | 0.482–0.484 |
| Fused AC evaluators | 15.197–23.045 | 15.099–19.750 | 1.722–1.724 | 0.485–0.488 |
| Erosion/L2/final | 8.491–9.820 | 8.092–8.467 | 0.913–0.913 | 0.246–0.247 |
| Mirrored opsin | 8.383–9.387 | 7.207–7.542 | 0.634–0.636 | 0.196–0.198 |
| Low/medium vertical | 8.184–8.579 | 7.795–7.841 | 0.936–0.938 | 0.271–0.272 |
| Gaborish/EPF | 4.760–7.763 | 4.980–6.179 | 0.313–0.314 | 0.136–0.137 |
| Composition/anchor reduction | 1.238–1.562 | 1.611–1.961 | 0.100–0.101 | 0.049–0.050 |
| Generic maximum reduction | 0.025–0.032 | 0.027–0.033 | 0.012–0.012 | 0.010–0.010 |
| Sparse AC packing | 2.484–3.182 | 2.357–2.448 | 0.000–0.000 | 0.023–0.023 |
| Resident coefficient encoding | 5.437–6.796 | 6.992–7.124 | 0.693–0.694 | 0.194–0.195 |

## Coefficient handoff

Bytes are exact, not timed estimates. AC transfer excludes the four-byte sparse counter; metadata is separate. Dense owner includes group padding. Timing ranges are four trace-process medians (ms).

| Case | Native kind | Native owner / dense owner bytes | AC transfer bytes | AC copy ms | Sparse pack ms | Original RGB upload ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4k | sparse32 | 5,779,732 / 106,168,320 | 5,779,732 | 0.992–1.167 | 2.484–3.182 | 15.318–17.470 |
| keong_3839x2159 | sparse32 | 7,263,096 / 106,168,320 | 7,263,096 | 1.240–1.388 | 2.357–2.448 | 15.227–15.721 |
| 1025_p2 | dense32 | 19,660,800 / 19,660,800 | 12,780,288 | 2.200–2.433 | 0.000–0.000 | 1.960–2.077 |
| 513_p0 | sparse32 | 152,192 / 7,077,888 | 152,192 | 0.028–0.029 | 0.023–0.023 | 0.504–0.505 |

## Separating device gaps from runtime API duration

`s194_gaps.py` intersects each measured window's complete runtime events with
the gaps in the union of all kernel/copy/memset activity. The all-API union
partitions each gap into covered and uncovered time. Per-name overlaps may
overlap one another and must not be summed. A blocking API's full duration
also includes active GPU time; interval coverage is not causation or removable
overhead. No CPU sampling or scheduler/preemption attribution is collected.

Ranges below are four process medians, ms. Columns are reduced separately,
so the displayed median ranges should not be added or subtracted.

| Case | Device gaps | Inside runtime API union | Outside runtime API union | Before first candidate upload | After last cost readback |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4k | 21.870–30.146 | 4.794–6.122 | 16.936–24.024 | 5.043–7.549 | 9.203–12.717 |
| Keong | 30.658–35.336 | 5.459–6.129 | 25.198–29.208 | 5.217–6.451 | 17.059–21.535 |
| 1025/p2 | 9.689–13.156 | 4.078–5.242 | 5.856–7.914 | 0.683–1.121 | 3.475–4.755 |
| 513/p0 | 3.667–4.368 | 2.971–3.564 | 0.700–0.826 | 0.165–0.194 | 0.335–0.394 |

4K has 295 kernel-launch API calls, 49 linear async copies, three 2D copies,
34 stream synchronizations, seven event synchronizations and six pooled
allocations per measured encode. Keong has 360 launch calls; dense 1025/p2
has 281 and small 513/p0 has 321. Launch count alone is not launch-boundness.
For 513/p0 the launch API's gap overlap spans 1.126–1.899 ms, whereas at 4K
it is 0.406–0.517 ms. Graph capture would still have setup/lifetime costs and
must not be inferred to win from these intervals alone.

At 4K, pooled-allocation API totals are only 0.049–0.059 ms, and their gap
overlap 0.043–0.051 ms. Keong totals are 0.059–0.080 ms. This does not support
another allocation-pool change as the first explanation for tens of milliseconds
of photo gaps. Conversely, 4K event-synchronization totals of 135.407–141.757 ms
mostly overlap GPU execution; only 1.145–1.227 ms intersects device gaps.
Calling the entire wait removable synchronization overhead would be incorrect.

### Source-and-transfer identification of the host boundaries

`s194_boundaries.py` independently counts the seven candidate families from
their footprints, two-block steps for DCT32 families, and partial 8x8 color
tiles. Every measured trace must contain exactly one ordered sequence of
fourteen descriptor/matrix uploads and seven packed-cost readbacks with those
exact sizes. It also checks the next metadata upload is anchor-aligned and
followed by exactly one EPF byte per base block. SQLite proves no device kernel,
copy or memset intersects either selected open boundary, across all 80 windows.

At 4K and Keong the source still constructs 522,120 descriptors, uploads
12,530,880 descriptor bytes and reads 2,088,480 packed cost bytes. The last
DCT32x32 cost readback is exactly 72,720 bytes. The first subsequent metadata
upload is 65,280 bytes at 4K and 253,288 at Keong, followed by 129,600 EPF bytes.
These match the source's first AQ upload, a vector of two-uint32 coordinate
records. This source/sequence match is not an instrumented host-function timer.

The first boundary follows the initial four-byte readback and precedes the
first candidate upload. Source preparation includes descriptor construction,
matrix packing, dense-cost initialization and resource planning/validation.
The second follows the final packed-cost readback and precedes AQ metadata
upload. It includes the final scatter, serial hierarchical search, grid export,
caller transitions and AQ metadata construction/validation. It is not all
`SearchTile`, not all scattering, and not all allocation.

The current `CandidateCostTableView` is explicitly dense, indexed by global
base-block position. `ac_strategy_search.cpp` initializes seven image-wide
cost vectors and scatters packed device results. At 4K those duplicate dense
vectors contain 3,628,800 bytes. The codec then searches color tiles serially.
`SearchGrid::Export` allocates another byte-cell owner, validates coverage,
reconstructs anchors through checked setters and checks completeness. AQ
`BuildMetadata` subsequently copies/visits the selected grid and constructs
grouped and row-major anchor metadata before upload.

Importantly, [S135](cuda-packed-strategy-costs-s135.md) did **not** promote its
packed-cost consumer: whole-call results were mixed despite local gains.
[S136](cuda-generated-candidates-s136.md) did not promote fused device descriptor
generation, and [S112](cuda-tile-scheduling-s112.md) did not promote the parallel
tile merge. Current source agrees with those recorded dispositions. These
rejected prototypes must not be mistaken for installed production optimizations
or reintroduced solely because the same source redundancy remains visible.

## Decision and next experiment

Keep production S193 unchanged. The next bounded investigation is host-boundary
decomposition: measure final scatter, tile search, validated grid export, and
AQ metadata construction separately inside the complete resident call, with
normal/ASAN exact-output checks and duplicate native-identical controls. That
will distinguish the potentially redundant representations from the search
arithmetic before choosing an implementation change.

A direct ownership transfer for the already-complete private search grid is
one source-grounded candidate, but its saving is unmeasured. It must retain
full coverage validation, malformed-grid rejection and transactional publication;
an unchecked bypass is not the proposal. Fusing selected-grid consumption with
AQ metadata construction is another possible representation change, contingent
on the decomposition and preserving prepared/reconfigured ownership contracts.
Do not equate the full 9–22 ms boundary with either candidate's opportunity.

Large photo GPU arithmetic (especially Malta and fused AC) remains important,
and small-input launch overhead is a separate pattern. This checkpoint does
not justify reviving a rejected tile schedule or graph policy, nor establish
that compact consumption, composition/reduction, scheduling or the broader
fully resident backend have been maxed out.

## Evidence and verification

Root: `build-cuda-ninja/profiles/s194-artifacts`. The pinned original and resume
protocols, all exclusive process journals, full Nsight/SQLite/stdout artifacts,
seventeen recoverable scratch ZIPs, `host.json`, `analysis.json`, `handoff.json`,
`report.json`, `work_identity.json`, `gaps.json` and `boundaries.json` are retained.
The original verifier remains unchanged. Version two validates recovered ZIPs
and exact continuation; version three also recomputes structural identity,
runtime-gap intersections and the source-derived transfer boundaries.

The final inventory includes source snapshots, versioned helpers and this
report. Read-only verification (no encoder or profiler rerun) is:

```powershell
python build-cuda-ninja/profiles/s194_verify_v3.py --frozen
```

The audit preserves the entire S193 2,031-file frozen inventory and all 345
current production/test sources. Only this report is committed; no external
push is performed. The three protected untracked Markdown files are not read,
edited or staged. Shared-machine timing variation remains a limitation; no
significance, fixed-clock, cache-contention or measured peak-memory claim is made.
