# Device-side strategy tile selection (S175)

September 10, 2026. Starting revision: `715fc00`, branch `feat/cuda`.
Windows, RTX 3060 Laptop (`sm_86`), CUDA 11.8, MSVC 14.37 Release;
scoped host ASAN uses clang-cl 22.

## Outcome

The GPU selection prototype is not promoted. It passes the correctness and
sanitizer gates and removes the intended host cost consumption boundary.
The complete strategy frontend is faster in 40 of 44 primary process cells,
but complete encodes are faster in only 29. All four 4K and all four
Flower 3200 × 2160 primary cells are slower. Runtime source stays unchanged;
no compatibility layer, diagnostic selector or size/content cutoff is added.

This follows S174's rejected private host-cell representations. It is distinct
from S106/S112's CPU tile scheduling and S135's packed host cost consumer:
the selection itself now consumes the existing packed costs on the GPU.
The promising local result warrants attribution of the whole-call loss,
not promotion based on the frontend timer alone.

## Mechanism and ownership

The original resident path evaluates seven policy cost batches on the GPU,
waits, reads seven packed scalar cost arrays, scatters them into seven dense
host tables, selects strategies serially across color tiles, and validates
the exported grid. The candidate appends one selection kernel after those
same cost kernels, inside the same submission and before the same completion
event. It retains one submission wait, then performs one readback containing
the selected grid and per-tile error flags. Host export retains the original
validation before committing the caller's grid.

Each CUDA block has one warp and handles one independent color tile, at most
8 × 8 base blocks. Lanes cooperate in staging seven tile-local cost tables,
initializing shared state and writing the resulting cells. Only lane zero
executes the serial hierarchical policy within the tile. Original operation
order, strict comparisons, priorities, tie behavior, boundary traversal and
strategy/anchor byte encoding are preserved. This does not reuse S174's
rejected direct edge predicates or its alternate geometry encoding.

Compilation uses `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.
The native compiler reports 60 registers, 2,176 shared bytes, a 32-byte stack
frame, and zero spill stores/loads. Shared storage consists of 64 cell bytes,
64 priority bytes, 64 current-cost floats and 7 × 64 candidate-cost floats.
These are compiler resource counts, not a measured kernel duration.

The packed address calculation follows the existing descriptor enumeration:
strategy, tile row, tile column, local row, local column. Partial final tiles
use their actual candidate count. No descriptor generation, upload, matrix,
cost kernel or existing kernel launch geometry changes. Both whole DLLs
contain the eleven byte-identical standard S169 native modules plus the one
qualified selection module.

The candidate releases both host packed-cost and dense-cost vector owners;
every encode checks zero logical size and zero retained capacity for them.
Device costs remain resident. Grid/error output is an additional aligned
range after existing scratch in the resource arena, not a separate device
allocation. A `uint32_t` host vector provides aligned error storage followed
by grid bytes, rounded to whole words.

For 3840 × 2160 pixels (480 × 270 base blocks, 2,040 tiles):

| Quantity | Original | Candidate |
| --- | ---: | ---: |
| Packed cost readback | 2,088,480 bytes | 0 |
| Selected cells plus tile flags readback | 0 | 137,760 bytes |
| Packed plus dense host cost logical storage | 5,717,280 bytes | 0 |
| Retained host cost capacity | At least logical size | 0 |

The candidate readback is 129,600 cell bytes plus 8,160 error bytes. The
original host total is 2,088,480 packed plus 3,628,800 dense bytes. This is
an ownership/transfer census, not an RSS or VRAM reduction measurement:
the output range is added to the arena before its allocation rounding.

The experiment compiles source-copy overlays, without changing private class
headers, vtables or runtime sources. A scoped thread-local diagnostic request
connects frontend and backend. The existing submission callback executes
synchronously while the local context is alive; CUDA receives parameters by
value, and the owning arena survives the wait. The request is restored on
every exit. This diagnostic hook is not a proposed production API.

Backend checks cover geometry, target, policy order/counts, device ownership,
output alignment/size, grid limits, and nonoverlap with cost, descriptor,
matrix, scratch and resident input ranges. Error flags are checked in tile
order before the exact original host export. Failures do not replace the
caller's preexisting grid. Inconsistent intermediate grids from extreme
finite costs are preserved, including the S174 counterexample.

## Correctness and recorded corrections

Normal and scoped-ASAN standalone fixtures each check 373 geometries,
nine cost patterns, and targets 1.0 and 1.2. Geometries include every width
and height from 1 through 19 blocks, plus twelve thin, partial and large
cases through 4K. Patterns cover zero/uniform costs, random values and
exponents, ties, neighboring floats, finite extremes, signed zero/subnormals,
and random nonnegative finite bit patterns. Six invalid DCT8-leaf cases per
geometry exercise negative, NaN and infinite first/last costs.

Each fixture passes 8,953 checks: 6,664 successes and 2,289 matching errors,
including 2,238 injected invalid leaves. Combined: 17,906 checks. Raw GPU
cells and every tile-error word match the CPU port, even on failing cases;
exported status, message and full grids match the separately compiled
original implementation. A nonempty sentinel survives failures. Input bytes
and 64-byte output guards are checked. The host port fills tile costs from
independent dense tables, separately testing the GPU packed-address formula.
The S174 11 × 13 extreme-cost counterexample is included explicitly.

The source audit mechanically reconstructs the original search grid and
decision body after the documented host/device type adaptations, verifies
the unchanged original host `Get`/`Export`, and checks the exact versioned
corrections. It does not treat matching whole codestreams alone as a proof
of all intermediate selection decisions.

Compute Sanitizer memcheck, racecheck, synccheck and initcheck each pass 169
representative fixture cases: 120 successes and 49 matching errors, including
42 injected invalid leaves. Total: 676 CUDA-sanitizer cases, with zero errors,
zero reported race hazards and zero memcheck leaks. Integrated normal and
ASAN contract fixtures each pass 20 boundary rejections, nine successful
routes, two injected failure-atomicity checks and prepared-object reuse
through original/candidate modes. The latter checks reacquisition of host
cost owners when returning to the original path.

Three rejected jobs remain in the evidence, with immutable input versions:

- The first CUDA build rejects a copied C++20 designated aggregate initializer
  under CUDA 11.8's C++17 parser. Policy version 2 changes only that initializer
  to positional syntax; the CUDA and fixture version changes only select it.
- The first integrated contract fixture exposes an all-empty batch bypass of
  the original backend early return. Backend version 2 changes that one guard:
  an active selection request must reach selection validation. Empty work
  without a selection request keeps its original behavior.
- The next fixture reaches a public resident call with neither a host nor
  device CFL map. Fixture version 3 supplies valid device CFL inputs and adds
  two overlap checks. No backend, frontend or kernel change accompanies it.

The corrected whole preflight covers all eleven standard inputs with
automatic/eight-thread budgets, normal/ASAN builds, plus whole memcheck at 4K
and initcheck at Keong. These 46 processes check 230 encodes. Three earlier
valid whole smoke jobs check 15 encodes. The 22 ASAN preflight processes
execute 88 instrumented candidate/control calls and 22 normal reference-DLL
calls; ASAN does not instrument every library or the CUDA driver. Every whole
call checks exact reference bytes, full summaries, coefficient width, native
storage and branch/transfer/host-capacity counters.

This stage does not newly qualify concurrent independent encoder contexts,
the batch encoder, CTest, install consumers, an independent decoder, Metal or
Linux. Such qualification and a clean runtime API would be required before
promotion. Frozen predecessor tests are not counted as fresh stage tests.

## Balanced unprofiled campaign

All eleven cases run with automatic/eight-thread budgets twice, reversing
case and label-order schedules on the second pass: 44 processes. Four labels
share one DLL: 0/1 are identical original CPU selection, 2/3 identical GPU
selection. Each process checks one reference, eight warm and sixteen measured
four-label rounds using even Williams orders `0132`, `1203`, `2310`, `3021`.
The campaign checks 4,268 encodes, including 2,816 measured calls. With smoke
and preflight, the stage checks 4,513 whole encodes.

The complete `Encode` call is timed. Input I/O, backend creation, reference
generation, result clearing/checking and NVML queries are outside the timer;
results remain alive until it ends. The harness reuses its backend, but each
whole encode creates its own encoding/preparation state. All 41 existing
phase timers and common frontend/wait/readback/merge-export scopes remain.

Primary deltas are within-round means of labels 2/3 minus means of 0/1,
followed by the median of sixteen rounds. Percentages use corresponding
within-round ratios. Each range spans four process cells, not a confidence
interval; negative is faster. All four individual candidate/control pair
medians and both duplicate-control comparisons remain in the report.

| Input | Frontend delta (ms) | Whole-call change (%) | Faster primary cells: frontend / whole |
| --- | ---: | ---: | ---: |
| 65, pattern 2 | +0.069 to +0.093 | +0.47 to +4.87 | 0/4 / 0/4 |
| 513, pattern 0 | −0.389 to −0.295 | −3.78 to −1.94 | 4/4 / 4/4 |
| 513, pattern 1 | −0.527 to −0.369 | −0.97 to +1.55 | 4/4 / 3/4 |
| 513, pattern 2 | −0.579 to −0.512 | −3.52 to −0.82 | 4/4 / 4/4 |
| 1025, pattern 2 | −3.211 to −2.685 | −3.79 to −0.46 | 4/4 / 4/4 |
| Flower 500 | −0.439 to −0.277 | −4.17 to −1.36 | 4/4 / 4/4 |
| 1080p | −4.252 to −2.625 | −7.87 to −4.26 | 4/4 / 4/4 |
| Flower 2000 | −6.641 to −5.097 | −2.91 to −1.10 | 4/4 / 4/4 |
| Flower 3200 × 2160 | −11.694 to −8.812 | +1.58 to +3.51 | 4/4 / 0/4 |
| 4K | −15.536 to −9.480 | +0.38 to +5.35 | 4/4 / 0/4 |
| Keong 3839 × 2159 | −15.910 to −11.954 | −1.00 to +3.05 | 4/4 / 2/4 |

The local mechanism succeeds: 4K frontend time drops 20.1–32.0%, and the
merge/export scope drops 8.333–10.947 ms (93.7–94.6%). Nevertheless, whole
calls increase 1.010–14.343 ms. Quantization minus the complete strategy
frontend, computed within each encode before pairing, increases
17.246–20.028 ms. The analogous residual increases 11.299–15.006 ms for
Flower 3200 × 2160 and 9.671–17.881 ms for Keong. This post-measurement
decomposition locates lost time outside the optimized boundary; it is not
a measurement of any specific kernel, allocation or scheduling mechanism.

Identical labels vary materially: original 1-versus-0 whole deltas span
−3.192 to +9.163 ms at 4K, and −26.637 to +8.434 ms at Keong. Candidate
3-versus-2 deltas span −5.484 to +3.066 ms at 4K. Individual 4K candidate
versus original deltas span −3.020 to +12.985 ms, with only two of sixteen
favorable pair medians. Across all inputs 106 of 176 individual pair medians
are favorable. These observations limit generalization; they do not negate
the clear local boundary removal or establish performance equivalence.

The extra submission-wait time is not the selection kernel's duration: the
wait includes queued cost work and host enqueue/completion behavior. No fresh
Nsight or CUDA-event kernel measurement is made here. Standard module hash
identity establishes unchanged machine code, not unchanged execution time.
No thermal, clock or host/device-gap cause is assigned from coarse timers.

### Driver interruption, preserved samples and timing conditions

After all 22 first-pass encoder jobs succeeded, the original driver exited
with `KeyError: 'pilot_schedule'` while selecting the second schedule. The
precommitted protocol correctly contains `schedule`; the driver mistakenly
uses the obsolete key only on pass two. No second-pass encoder had started.
This is a driver control failure, not an encoder timeout or privilege block.

The original driver and all first-pass logs remain unchanged. A versioned
resume helper verifies and reparses those 22 completed jobs, then runs only
the 22 missing jobs in the originally specified reversed order, with the
same binaries, inputs, warmups, measured rounds and checks. No sample is
filtered and no completed encode job is rerun. The last first-pass worker
finishes at 05:44:46.433 UTC; driver failure is observed by 05:45:05, and the
reversed pass starts at 05:46:22.307. This approximately 96-second inter-pass
gap is an explicit campaign limitation. An exact driver failure timestamp
is not fabricated from its last worker's journal.

All 8,448 warm/measured NVML endpoints report a 40,000 mW enforced limit.
This does not mean constant clocks or thermals. No recorded builds,
sanitizers, captures or heavy source/archive/hash sweeps overlap timing.
Light source reads, editing and ordinary shared-machine activity remain
limitations. No power, clock, thermal, priority, affinity, firewall, driver
or security setting is changed, and no admin/firewall blocker is observed.

## Evidence and next investigation

Root: `U:/gjxl-cuda-diagnostics/s175-artifacts`. Sources/helpers:
`build-cuda-ninja/profiles/s175_*`. There are 109 journaled jobs: 106 accepted
and three rejected, plus the separately documented driver interruption.
First and corrected source/build versions, all raw logs, source audit,
contracts, sanitizer results, protocol, recovery record, parsed measurements,
paired report, residual decomposition and decision are retained. The frozen
S174 inventory of 709 artifacts is reverified; predecessor binaries and
pinned helpers are neither rebuilt nor overwritten. No material file is
deleted. Compiler/runtime `temp` scratch is excluded from immutable inventory.

| Phase (UTC, September 10) | Start | Finish |
| --- | --- | --- |
| Corrected kernel build and full normal/ASAN fixtures | 05:09:48.906 | 05:10:29.993 |
| Four CUDA sanitizer fixture tools | 05:12:46.366 | 05:13:23.481 |
| Initial integrated whole build | 05:23:38.795 | 05:24:26.345 |
| Corrected backend whole build | 05:30:24.483 | 05:30:54.028 |
| Corrected contract build and normal/ASAN tests | 05:32:39.083 | 05:32:57.218 |
| Whole preflight including CUDA tools | 05:35:34.167 | 05:40:22.767 |
| Unprofiled first pass | 05:40:42.696 | 05:44:46.433 |
| Resumed, reversed second pass | 05:46:22.307 | 05:50:30.149 |

Verification checks sources, inputs/oracles, support binaries, all native
modules, every journal outcome/nonoverlap, exact schedules and counters,
raw-log reconstruction of both timing reports, and the final inventory:

```powershell
python build-cuda-ninja/profiles/s175_verify.py --frozen
```

The next investigation should trace original and candidate complete calls
to distinguish unchanged downstream kernel execution, host/device gaps,
allocation effects and the new selection kernel. It should use a fresh root
with sufficient capture scratch space: the U drive is nearly full, and prior
Windows Nsight captures need substantial transient ETL space. No additional
fusion or tile-layout sweep is justified merely by the saved frontend time.
The encoder is not established to be maxed out; the optimization goal stays
active.
