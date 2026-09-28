# Host metadata handoff

Proposal #3 in the tokenization scheduling investigation, layered on earlier
population/entropy readiness and production-default earlier DC preparation.
This is enabled by default. `GJXL_HOST_METADATA=0` opts out to the prior host
handoff; unset, 1, and other values select the default. Set the environment
before encoding, without concurrent environment mutation. No public API or
codestream policy changes. Earlier entropy and DC scheduling retain their
independent controls and eligibility rules.

## Changes and contracts

The resident strategy selector already produces encoded ownership cells and
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
bit travels with a copied internal view. Only the checked publication function
can set it; constructing a raw view never imports it. The serializer still
checks its codec/profile support gates. This requires immutable, stable backing
until every reader returns; the completed lease owns that backing independently
of backend/evaluator lifetime. CUDA and owned/raw public frame paths are unchanged.

## Scope

This reduces existing host work. It does not move context modeling or entropy
coding to the GPU, remove the host strategy grid, or implement the broader
GPU-native context/anchor dependency chain. Earlier stage estimates must not be
read as an achieved speedup. Complete-call AC qualification and retained evidence
are in `reports/host-metadata-20260927` in the original checkout.

## Qualification

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
