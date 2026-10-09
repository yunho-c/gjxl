# Host metadata handoff

Proposal #3 in the tokenization scheduling investigation, layered on earlier
population/entropy readiness and production-default earlier DC preparation.
This is enabled by default. `GJXL_HOST_METADATA=0` opts out to the prior host
handoff; unset, 1, and other values select the default. Set the environment
before encoding, without concurrent environment mutation. No public API or
codestream policy changes. Earlier entropy and DC scheduling retain their
independent controls and eligibility rules.

## Changes and contracts

The Metal resident strategy selector already produces encoded ownership cells and
GPU strategy, anchor and color-transform records. Its completed cells now enter
`AcStrategyGrid` through a checked import. Each anchor must name a known strategy,
fit the grid, cover previously unoccupied cells, and match every ownership byte.
Exact cover is required. Output publication is atomic; malformed GPU maps remain
device errors, and allocation failures retain their status.

Resident reconfiguration retains the host grid, row-major anchors, family counts,
parameters and final-transform layouts needed for output and evaluator reuse.
It skips upload-only strategy/grouped-anchor/color records and the self-copy of
sharpness. Supported strategies, color-tile boundaries, sharpness, complete cover
and shader integer bounds remain checked. Public host reconfiguration keeps the
existing path. Storage admission remains conservative; no extra managed buffers
or GPU submissions are introduced.

After final coefficients and metadata are complete, the Metal completed-frame
lease performs the full structural/value validation once. A private validation
bit travels with a copied internal view. The checked publication function
establishes it; constructing a raw view never imports it. The serializer still
checks its codec/profile support gates. This requires immutable, stable backing
until every reader returns; the completed lease owns that backing independently
of backend/evaluator lifetime. Owned/raw public frame paths retain full validation.

## CUDA assembly validation reuse

CUDA already checks geometry, strategies, raw quantization, EPF sharpness, DC
reconstruction, group coverage and native AC storage while assembling its owned
frame. The assembler now records those guarantees only after all checks succeed.
The CUDA completed-frame lease uses `BorrowFrameWithAssemblyValidation` to carry
that proof into the serializer. This avoids rescanning the finished metadata; it
does not add a full validation pass at publication. `GJXL_HOST_METADATA=0` restores
ordinary borrowed validation for same-binary comparisons.

The proof belongs to the private immutable frame. Copies preserve it with their
independent storage; moves transfer it and clear the source, including self-moves.
Failure-atomic copy assignment and assembly preserve the prior output and proof.
Assigning an invalid frame or a frame from an unproven producer clears the proof.
The new borrowing helper falls back to ordinary validation for unproven owners.
Moving, assigning or destroying an owner ends all previous borrows.

`VarDctEncoderFrame::valid()`, ordinary `BorrowFrame`, and raw views still perform
their existing validation. `ValidateFrameViewForPublication` still revalidates
its input. Serializer codec/profile support and encodability checks always run.
No coefficient representation, readback, device ownership, tokenization,
scheduling, storage admission or image policy changes accompany this handoff.
CUDA host metadata construction and reconfiguration are otherwise unchanged;
the Metal upload-record optimization above is backend-specific.

## Scope

This reduces existing host work. It does not move context modeling or entropy
coding to the GPU, remove the host strategy grid, or implement the broader
GPU-native context/anchor dependency chain. Earlier stage estimates must not be
read as an achieved speedup. Complete-call AC qualification and retained evidence
are in `reports/host-metadata-20260927` in the original checkout.

## Metal qualification

The initial opt-in study covered 105 inputs, 491 processes and 4,417 public
calls on AC power (M4 Pro, 14 CPU cores, 48 GiB, normal Metal Release, distance
1.9, automatic CPU). Byte/summary parity, CPU and managed-memory bounds,
cleanup and independent decode coverage of 329 distinct bitstreams passed.
Both default-off and enabled suites passed 165/165 tests; the immutable view
and encoded-grid tests also passed ThreadSanitizer.

Incremental complete-call gains against earlier entropy plus earlier DC were
about 3.72 ms / 1.21% on alpine24MP at effort 7 and 4.85 ms / 0.61% on
forest48MP at effort 7. Alpine24MP effort 4 saved 1.17 ms / 1.43%. Smaller
inputs were mixed. A CLIC effort-4 regression signal did not reproduce in a
longer repeat. These are input/device-specific results, not a universal gain.

The original evidence remains in `reports/host-metadata-20260927`; default
publication assertions, opt-out checks and normal-build promotion captures
are recorded separately in `reports/host-metadata-default-20260927`.

## CUDA qualification

The CUDA change is based on main `fb5fd36` and was tested on Windows with an
RTX 3060 Laptop GPU (6 GiB), Ryzen 9 5900HS, CUDA 11.8, MSVC 14.37 and driver
577.00. The Release SM86 build uses the existing default CUDA token provider
and earlier entropy/DC scheduling.

All 172 native checks passed. Two compile-at-test-time checks initially lacked
the MSVC/Windows SDK environment; both passed after correcting the runner.
Seven focused checks also passed with `GJXL_HOST_METADATA=0`. CUDA memcheck and
initcheck passed for the coefficient-owner, token-provider and public-workflow
fixtures (six runs, each requiring its completion marker and zero errors).

Coverage includes all six dense/sparse coefficient representations, ordinary
versus proven views, exact serialized bytes, unsupported-profile rejection,
invalid metadata and edge tails, 72 copy/move combinations, 1,044 injected copy
allocation failures, and completed leases surviving producer reuse/destruction
and concurrent readers. Frame and view sizes remain 720 and 544 bytes on this
build; the provenance state occupies existing padding.

Detailed profiles measure validation alone; profiling changes entropy-scheduling
eligibility and is not used for the complete-call comparison. Median validation
time fell from 2.561 ms to 0.001 ms for alpine12MP E1, from 2.652 ms to 0.001 ms
for alpine12MP E8, and from 12.163 ms to 0.001 ms for campus48MP E1.

The campus48MP E8 diagnostic without reuse occupied about 5,975 MiB of device
memory and took a median 33.3 seconds per call (30.9 seconds in quantization).
Its reuse-on diagnostic was stopped. That case was excluded before collecting
the repeated timing cohort; no paired performance or byte-parity result is
claimed for it. E8 remains covered through 24 MP, and E1-E5 through 48 MP.

The ordinary complete-call survey uses two Kodak images, one CLIC image,
alpine12MP, forest24MP and campus48MP at distance 1.9 and CPU limit 8. E1-E5
run on all six inputs; E8 runs on the first five. CLIC and alpine12MP also run
E1/E4/E5/E8 at distance 0.55. Four rounds reverse the order of reuse-on/off
processes. Each process has one validation encode, two warmups and five timed
encodes. This gives 344 A/B processes and 1,720 timed calls. Another 43
unchanged-main processes establish exact byte parity; all 43 representative
outputs decode independently with the pinned libjxl decoder.

Initial distance-1.9 results below are geometric means across inputs of the
ratio of per-mode medians over four process medians. Negative latency changes
mean faster. These are small qualification samples, not a new paper table.

| Effort | Complete-call latency change | Throughput change |
| --- | ---: | ---: |
| E1 | -2.73% | +2.81% |
| E2 | -5.17% | +5.46% |
| E3 | -2.84% | +2.92% |
| E4 | -0.95% | +0.96% |
| E5 | +2.10% | -2.06% |
| E8 | -0.28% | +0.28% |

Individual timings vary substantially. In particular, forest24MP E2 appeared
16.96% faster, much more than the removed validation work accounts for, while
campus48MP E5 appeared 12.41% slower with paired changes ranging from -18.60%
to +39.86%. Alpine12MP E1 was 3.62% slower and slower in all four pairs. These
three cases received a separate eight-pair repeat with five warmups and nine
samples per process. All 432 additional timed encodes matched the original
baseline bytes. Its results are retained separately from the initial table:

| Repeated case | Latency without reuse | Latency with reuse | Change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Alpine12MP E1 | 151.041 ms | 149.222 ms | -1.20% | 5/8 |
| Forest24MP E2 | 380.246 ms | 365.288 ms | -3.93% | 5/8 |
| Campus48MP E5 | 897.500 ms | 895.111 ms | -0.27% | 4/8 |

Neither slowdown reproduced. The large positive outlier also shrank, and pair
variation remains substantial. The focused reuse is enabled by default as a
modest reduction in verified host work; these data do not establish a universal
complete-call gain. The existing opt-out remains available. This qualification
does not replace the fixed-quality or score-calibrated paper sweeps.

The isolated branch is `perf/cuda-host-metadata`. Local evidence is retained in
`build/cuda-host-metadata-20261009` in the original checkout: `REPORT.txt`,
`manifest.json`, `analysis.json`, `timing/plan.json`, per-process JSON/JXL files,
independent decode hashes, `profiles/`, `recheck/` and `verification/`. The
baseline source and build are preserved separately from the candidate.
