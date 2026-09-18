# S170: sparse-packer header stores and block coarsening

## Scope and status

This experiment starts from `a317005abf8b380c7c45e39b870f6a3593be731d`
(S169), with S168's aligned bit-writer append and all preceding production
changes retained. It investigates the current fully resident sparse **int32**
handoff, not exact-coefficient mode, compact-width enablement, AC strategy tile
merging, or a new sparse representation. Production source is unchanged.

The diagnostic archive is
`U:/gjxl-cuda-diagnostics/s170-artifacts`; helpers are under
`build-cuda-ninja/profiles/s170_*`. All predecessor artifacts are immutable.
The final source archive includes the diagnostic implementation and protocols;
the ignored helpers are not a new supported API or compatibility layer.

Decision: **do not promote**. Block coarsening is faster in isolated replay,
but the complete-encoder cohort does not establish a reproducible benefit
beyond the identical-control disagreement. Keep the current production packer.

## Why this boundary

S169 identified sparse packing as a remaining native handoff cost. Inspection
of the current `sm_86` machine code found that one thread performs four 64-bit
mask stores and four 32-bit offset stores before the second block barrier.
The production block covers 256 coefficients with 256 threads. Warp ballots
produce populations; thread zero scans eight populations and reserves one
payload interval atomically when the block is nonempty.

Three candidates keep the masks/offsets/payload representation and exact
coefficient values unchanged:

| Mode | Coefficients/block | Threads/block | Change |
| --- | ---: | ---: | --- |
| 0 | 256 | 256 | Exact production source/native instructions |
| 1 | 256 | 256 | Independently named, native-identical control |
| 2 | 256 | 256 | Consecutive lanes store consecutive headers after publication |
| 3 | 512 | 256 | Two input items/thread, coalesced headers |
| 4 | 1,024 | 256 | Four input items/thread, coalesced headers |

All candidates retain two barriers. Larger blocks reduce grid size and the
number of nonempty-block atomic reservations, but lengthen the serial prefix
scan and increase registers/shared memory. No physical payload-order guarantee
is added: concurrent atomic reservations may reorder blocks, and offsets define
the logical representation.

The extracted production baseline and duplicate match both instruction text
and both 64-bit native encoding words per instruction, for all three widths.
The other ten CUDA modules are byte-identical to S169. The same diagnostic
15-kernel module is used by the fixtures, replays and whole-encoder DLLs.

| int32 mode | Registers/thread | Static shared bytes | Static instructions | Global-store instructions |
| --- | ---: | ---: | ---: | ---: |
| 0 / 1 | 28 | 68 | 136 | 9 |
| 2 | 20 | 68 | 120 | 3 |
| 3 | 29 | 132 | 168 | 4 |
| 4 | 40 | 260 | 256 | 6 |

These are static code counts, including padding instructions, not dynamically
executed instruction totals. There are no stack/local allocations. CUDA's
occupancy query permits six 256-thread blocks/SM for every int32 variant on the
30-SM RTX 3060 Laptop GPU (compute capability 8.6). This is theoretical
occupancy, not measured achieved occupancy; reduced registers do not increase
that limit here.

## Exact current-input captures and sparsity census

The diagnostic hook captures the actual source pointer passed to the packer,
before packing. It writes sorted nonzero `(uint32 index, int32 value)` pairs,
count/width/population and a logical int32 little-endian FNV-1a hash. The replay
independently reconstructs all zero and nonzero values and checks that hash.
Normal and ASAN captures match byte-for-byte for all seven currently sparse
cases. Capture encodes also match the frozen normal reference's codestream,
full summary, coefficient width and native owner size.

| Input | Coefficients | Nonzeros | Active blocks, 256 | Active blocks, 512 | Active blocks, 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 513 p0 | 811,200 | 23 | 22 | 21 | 17 |
| Flower 500 | 762,048 | 39,896 | 1,986 | 1,222 | 721 |
| 1080p | 6,220,800 | 70,907 | 8,100 | 4,050 | 2,025 |
| Flower 2000 | 12,000,000 | 160,785 | 13,351 | 8,089 | 5,264 |
| Flower 3200×2160 | 20,736,000 | 288,290 | 20,824 | 12,370 | 8,877 |
| 4K | 24,883,200 | 278,533 | 32,400 | 16,200 | 8,100 |
| Keong 3839×2159 | 24,883,200 | 649,374 | 29,768 | 18,845 | 12,137 |

Total grids are `ceil(count/span)` blocks. For example, 4K's 97,200 original
blocks include 64,800 empty blocks. `census.json` retains full nonzero-count
histograms at all three block spans, not just these active-block totals.

## Local replay

Each process uses the exact captured int32 input, dense-sized input/payload
allocations and one contiguous `masks | offsets | total` header, with aligned
guards. Allocation, copies and checking are outside the timing window. Every
inner repetition resets the total counter before launching the packer.
CUDA-event time includes that required reset and device queue gaps; separate
host batch wall time includes enqueue/synchronization overhead.

Five modes run in ten balanced Williams orders. Each process has ten warmup
rounds and forty measured rounds, with four pack operations per window.
There are two process repeats for each of seven cases; the second reverses
case and order traversal. All 3,500 windows / 14,000 launches are retained,
including 2,800 measured windows. Header, logical payload, population and
adjacent guards are checked after every window; full input immutability and
the untouched payload tail are checked at process end. Physical packed-byte
ordering is deliberately not compared.

The table shows ranges across the two processes of the median, round-paired
latency change versus mode 0. Negative means faster. It is not a whole-encoder
speedup table.

| Input | Coalesced 256 | Coalesced 512 | Coalesced 1,024 |
| --- | ---: | ---: | ---: |
| 513 p0 | −3.99% to 0.00% | −13.13% to −9.76% | −15.66% to −6.94% |
| Flower 500 | −1.12% to 0.00% | −14.85% to −13.89% | −21.68% to −19.44% |
| 1080p | −1.93% to −1.58% | −18.48% to −17.04% | −21.58% to −18.70% |
| Flower 2000 | −1.01% to −0.97% | −18.47% to −13.63% | −22.75% to −16.42% |
| Flower 3200×2160 | −0.80% to −0.58% | −17.60% to −14.98% | −21.44% to −18.14% |
| 4K | −1.93% to −1.84% | −20.21% to −20.10% | −24.35% to −24.28% |
| Keong 3839×2159 | −0.94% to −0.94% | −18.87% to −18.53% | −23.80% to −23.72% |

The duplicate-control paired changes range from −0.17% to +0.92% over all
fourteen processes. 4K's baseline median is 0.635–0.637 ms versus 0.482–0.483 ms
for mode 4. Local coarsening wins are reproducible on these inputs; simply
coalescing header stores has much less effect. The larger block is therefore
the candidate taken to whole-workflow testing.

An initial normal replay preflight completed its correctness checks at an
enforced limit of 68,971 mW. The external Python assertion requiring 40,000 mW
then rejected it for timing eligibility. Its process, log and original protocol
are retained; it was not silently rerun or counted as a measured sample.
A separately pinned protocol allowed correctness qualification independently
of power and explicitly recorded the current envelope for the new timing
cohort. In fact, all 7,000 endpoints of the measured replay cohort were
40,000 mW. No GPU, clock, power, thermal, priority, affinity, firewall or security
settings were changed. Historical profiled kernel durations are not treated
as comparable unprofiled replay timings, even at the same power limit.

## Whole-workflow protocol

The strongest local candidate (mode 4), production-native baseline (mode 0),
and native-identical duplicate (mode 1) run inside the same diagnostic DLL.
Capture is disabled and its hook is audited on every encode. All eleven S169
cases are included at automatic and eight-thread CPU budgets, including the
four dense fallback cases with zero sparse-packer calls.

Each process creates one frozen S168 normal reference, then runs six warmup
and eighteen measured three-mode rounds. The six orders balance mode position
and directed predecessors. Two repeats reverse the case/thread schedule and
order traversal. This is 44 processes, 3,168 diagnostic encodes plus 44
references, including 2,376 measured encodes. Every output must match reference
bytes, the full summary, width and native owner size. All 41 host phases and
untimed NVML endpoints are retained.

The timed bridge call includes the complete `Encode` workflow and its internal
frame/buffer lifetimes. Image/backend creation and clearing the prior/final
codestream result are outside the timer, consistently for all three modes.
Paired results are reported within each process; baseline-control disagreement
and dense fallback behavior are essential checks on small apparent gains.

## Qualification and diagnostic corrections

The guarded kernel fixtures cover five modes, all three widths, seven patterns,
three changed-input reuse states, signed extrema, tails, empty/dense/sparse
populations, and invalid dispatch arguments. Full normal/ASAN runs each pass
9,765 checks over 31 sizes. Four CUDA sanitizers each pass the 1,890-check
six-size subset: 27,090 fixture checks total, zero CUDA errors/hazards and zero
memcheck leaks. Sixteen replay preflight processes add 160 checked windows;
normal/ASAN cover all seven captures, and memcheck covers two representative
current inputs. These preflight timings are not performance evidence.

The first host build failed ASAN compilation because the diagnostic audit
struct's default member initializers triggered a C-linkage return-type warning
under `/WX`. A separate POD-header version built cleanly without suppression.
The original build, sources and successful CUDA object were retained.

The first whole-encoder ASAN preflight then reported a global-buffer-overflow
in `S170Configure`, on a one-byte read of its empty string literal when disabling
capture. The shadow byte was poisoned despite the address being at the start
of that one-byte global. The underlying global registration/poisoning mechanism
was not established; this is neither a demonstrated production encoder defect
nor a proved sanitizer false positive. The report is retained in
`whole_preflight_513_p0_t8_asan.log`. It occurred after the frozen reference and
before any diagnostic encode in that process.

The new hook uses `capture_path.clear()` for a null capture path, removing that
unnecessary literal read. Both normal and ASAN hooks were rebuilt separately;
all eleven GPU modules remain identical. All 44 normal/ASAN whole preflight
processes were repeated with the corrected hook, followed by full-encoder CUDA
memcheck (4K) and initcheck (Keong): 46 processes / 184 encodes, exact output
checks, zero reported CUDA errors, zero memcheck leaks. The earlier successful
normal preflight and rejected ASAN run are preserved. No sanitizer diagnostic
was suppressed, and no measured whole-encoder run preceded this correction.

This diagnostic-only investigation does not claim a new CTest, installation,
downstream-consumer, decoder-quality, or production ASAN coverage cohort.

## Whole-workflow results and decision

All 44 processes / 3,212 encodes completed with exact byte/full-summary/storage
checks. Every one of the 6,336 warmup/measured NVML endpoints was 40,000 mW.
No measured sample was removed or rerun. The ordinary campaign ran from
02:05:07 to 02:11:55 UTC on September 10, 2026; no concurrent builds,
sanitizers, profiler captures or archive compression ran during it.

Ranges below cover the four process cells per input (two thread budgets ×
two repeats). Each cell uses the median of within-round latency ratios;
negative means faster. The final column counts cells where mode 4 beats both
controls on that statistic, not statistically significant wins. Pairwise
medians are not algebraically transitive and should not be subtracted from
one another. `whole_report.json` retains every process and all 41 phase fields.

| Input | Mode 4 vs baseline | Mode 4 vs duplicate | Identical control vs baseline | Faster than both / 4 |
| --- | ---: | ---: | ---: | ---: |
| 513 p0 | −2.85% to +0.58% | −1.85% to +0.51% | −0.72% to +1.09% | 1 |
| Flower 500 | −0.45% to +2.44% | −0.14% to +1.77% | −0.65% to +2.07% | 0 |
| 1080p | −1.78% to +2.45% | −0.81% to +1.74% | −1.24% to +2.43% | 1 |
| Flower 2000 | −1.62% to +2.54% | −2.45% to +0.00% | −0.76% to +3.27% | 1 |
| Flower 3200×2160 | −1.94% to +1.02% | −1.10% to +2.94% | −1.74% to +1.89% | 1 |
| 4K | −0.10% to +0.42% | −3.56% to +2.53% | −1.89% to +0.86% | 0 |
| Keong 3839×2159 | −6.62% to +1.10% | −3.58% to +3.16% | −4.02% to +4.55% | 2 |
| 65 p2 (dense) | −0.53% to +1.38% | −2.80% to +0.02% | −1.12% to +2.50% | 0 |
| 513 p1 (dense) | −4.32% to +1.67% | −0.20% to +1.12% | −1.79% to +0.36% | 1 |
| 513 p2 (dense) | +1.37% to +5.27% | −0.81% to +1.90% | −0.43% to +3.08% | 0 |
| 1025 p2 (dense) | −3.20% to +2.35% | −3.54% to −2.17% | +0.85% to +5.66% | 2 |

No input beats both controls in all four cells. On 4K, the candidate's
−0.10% to +0.42% paired result is smaller than control disagreement; its
whole medians range from 271.91 to 286.89 ms. Keong's attractive first-repeat
automatic-thread result does not reproduce at eight threads in repeat two.
Dense fallback changes despite zero packer calls further demonstrate that
the cohort cannot attribute every percent-level movement to sparse scheduling.

These observations do not prove that the candidate intrinsically regresses,
that every coarsening design is bad, or that the backend is maxed out. They
do reject promotion based on this evidence. The native storage capacity and
readback byte counts are unchanged; the isolated 4K saving is only about
0.15 ms. Its end-to-end value is unestablished here. Modes 2 and 3 were fully
fixture/replay-qualified but were not given separate whole-performance
cohorts; the conclusion must not be extended to unmeasured designs.

Retain the exact current-input captures and native evidence for future work.
Prioritize higher-cost resident computation or native coefficient consumption
before repeating header-store-only scheduling. Fusion with an upstream producer
would be a distinct experiment, not a result of S170. Do not revive previously
rejected AC tile schedulers or reference caches merely because this local
packer experiment won.

## Preservation and verification

`s170_verify.py` checks archived/live source pins, the complete 1,964-file S169
predecessor inventory, 147 job journals/log hashes, built binaries/native
modules, capture pairs, replay/whole sample ordering and power endpoints,
and independently recomputes the reported whole paired outer-time comparisons.
There are 145 accepted jobs and two rejected jobs: the original host build and
the original ASAN whole preflight. The separate 40 W external analysis gate
failure is preserved too; its successful executable is not mislabeled a
crashed process.

Accepted encoder jobs account for 3,428 completed encodes. One additional
frozen reference completed in the rejected ASAN process; no candidate encode
ran there. Kernel fixtures account for 27,090 checks. Local replay accounts
for 3,660 checked windows / 14,160 launches including qualification.

`final_sources.json`, `final_summary.json` and `artifact_hashes.json` freeze
the sources, decision and complete non-temporary artifact inventory.
`python -B build-cuda-ninja/profiles/s170_verify.py --frozen` verifies that
checkpoint without requiring future production sources to remain unchanged.
No artifacts were deleted or overwritten, no predecessor was modified, and
the three protected user scratch files were not opened, edited or staged.
Only this report is committed; the broader optimization goal remains active.
