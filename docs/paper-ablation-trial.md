# Ablation infrastructure trial — 2026-09-20

The bounded trial completed successfully on an Apple M4 Pro, macOS 15.6,
while on battery. It validates execution controls, correctness checks, and
collection plumbing. It does **not** qualify performance or corpus-wide parity.

The implementation is in the isolated `experiment/paper-ablation` worktree.
The trial used `b1a7373beff4c0370494c2f9584bcb1510491357` plus the experiment
changes, captured before commit in the run's source archive and hashes. The original checkout
was left untouched. See [the protocol](paper-ablation.md) for comparisons,
build instructions, resume behavior, and full-collection commands.

## Retained evidence

The final run is locally retained under `ablation-runs/trial-validated/`:

- [Manifest](../ablation-runs/trial-validated/manifest.json), source archive,
  CMake cache, frozen encoder, and source patch.
- [Trial report](../ablation-runs/trial-validated/REPORT.md) and
  [machine-readable summary](../ablation-runs/trial-validated/summary.json).
- Per-case codestreams, decoded linear-sRGB PFM files, SSIMULACRA2 values,
  raw timings, path audits, power records, and completion records.
- [Ordinary-build parity](../ablation-runs/trial-validated/ordinary-build-parity.json),
  [harness checks](../ablation-runs/trial-validated/harness-checks.json),
  [core CTest log](../ablation-runs/trial-validated/ctest-core.log), and
  [runner guard tests](../ablation-runs/trial-validated/runner-tests.log).

These generated artifacts are ignored by Git; the documentation and source
changes are available for review. Earlier development trials are also retained
but are superseded by `trial-validated`.

Frozen encoder SHA-256:
`ff8d532aa4f61f99f81469e4d1375c42799f847b3503649f0cec1518924de7bb`.
The decoder is the existing pinned libjxl `djxl` reporting revision `e8ff0976`;
SSIMULACRA2 is from the installed jpeg-xl 0.12.0 tools. Their exact paths and
binary hashes are in the manifest.

## Outcomes

- **36 configurations completed:** nine arms × two generated images × efforts
  5 and 8, at distance 1 and two CPU threads. Each process performed an initial
  preparation encode, one untimed warm audit encode, and one timed encode.
- **28/28 intended pairwise comparisons were byte-identical**, including the
  four scalar/SIMD DCT comparisons. Decoded pairwise error and SSIMULACRA2
  differences were zero. All repeated calls within each process also matched.
- All configurations retained the expected one or three AQ evaluations. Actual
  kernel names confirmed the selected DCT arithmetic and fusion/dataflow paths.
- A second build of the **same modified source with experimental controls OFF**
  matched the production arm on all four image/effort cases. An invalid ablation
  environment variable was ignored by that ordinary build. This is a build-mode
  parity check, not a comparison with a separately rebuilt pristine parent.
- The existing DCT numerical-reference test passed. The AQ evaluation test
  passed with both ordinary and segmented scheduling, including final scoring,
  zero-update materialization, ownership, and injected failures.
- Eleven runner tests passed, including inactive-control rejection, unknown
  kernel rejection, changed-artifact rejection, the fresh-directory requirement,
  and refusal to start a full study on battery.
- Resuming the completed run left the ledger unchanged and performed no new
  encodes. Changing the decoder refused resume. Unknown variant names failed.

Two existing AQ test assertions assumed exactly one policy submission. They
now check the explicit expected segmented count when that experiment is active;
the numerical and ownership assertions remain in place.

## Observed control activation

For the 257x193 edge image at effort 8, the untimed warm audit recorded:

| Arm | Submissions | Completion waits | Encoded dispatches | AQ evaluations |
|---|---:|---:|---:|---:|
| `production` | 4 | 4 | 422 | 3 |
| `aq-sync` | 8 | 8 | 422 | 3 |
| `ac-handoff` | 5 | 5 | 326 | 3 |
| `split-malta` | 4 | 4 | 458 | 3 |
| `split-epf` | 4 | 4 | 425 | 3 |
| `host-fused` | 5 | 5 | 325 | 3 |
| `host-split-ac` | 5 | 5 | 346 | 3 |
| `packed-simd` | 5 | 5 | 362 | 3 |
| `packed-scalar` | 5 | 5 | 362 | 3 |

The AQ synchronization pair had identical per-kernel invocation counts and
four additional submissions/waits, as intended. AC handoff also changes
metadata preparation and indirect dispatch, so its dispatch-count difference
does not measure a reduction in useful GPU work. Indirect commands may encode
zero-work grids. No latency ratios from this trial are used as paper results.

## Remaining full-study work

Choose the final image and size cohorts, distances, and effort coverage; freeze
the reviewed implementation; then run the explicit AC-powered corpus protocol.
Review any future DCT numerical differences before making equal-quality timing
claims. The infrastructure is ready for that collection; the full study has
not been run.
