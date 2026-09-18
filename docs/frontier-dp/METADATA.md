# GPU strategy metadata construction

The Metal metadata constructor reproduces the existing CPU AQ and completed-frame
builders from the native selector's compact byte map. It is a tested foundation;
ordinary encodes do not use it yet, and the ACS-to-AQ wait remains.

Seven dependent kernels produce stable grouped anchors, per-family counts and
coefficient offsets, the expanded strategy map, tile-ordered CfL records and
32×32-block AC-group coefficient destinations. Families follow AQ's order
`[0, 4, 5, 6, 7, 10, 11]`, which differs from candidate scoring. Stable chunk
prefixes preserve global row-major order within each family. Coefficient storage
remains three image planes. Invalid maps or selector error flags disable all
families and publish a safe DCT8 map; the control error must be consumed by the
future AQ integration.

The checked allocation plan adds prefix/rank scratch, approximately 1.49 MiB at
24MP. It borrows the final metadata output planes. All bindings are range,
backend, type and overlap checked. The caller owns their lifetime through GPU
completion. Production admission and prepared-evaluation integration remain part
of the next step.

## Correctness

`metal_aq_strategy_metadata` compares every output word with the actual CPU
reconfiguration and completed-frame builders. All 292 cases pass, in both normal
and Metal API/shader-validation runs: all 64 partial-tile shapes after full tiles,
pure DCT8 and mixed covers, narrow images, AC-group boundaries and reused
preparations. Guards and unused output tails remain intact. Six malformed-map
cases, descriptor bounds/alias errors, storage overflow, injected submission and
completion failure, and successful subsequent reuse are covered.

The validation build initially published an incorrect family count while its
aggregate count was correct. Keeping the shared validity flag in the same uint
array as family totals, instead of a separate shared bool, resolved the observed
discrepancy. Both configurations pass with the retained source. This is an
observed toolchain sensitivity; the underlying compiler issue is not isolated.

## Standalone timing

Retained artifact: `build/frontier/results/metadata-builder-20260918/`. It freezes
the test executable, external metallib, source snapshot and parent revision
`d32625944a5a5093e8e9f2fe050bc252fb6b0b44`. Portable hashes, commands, summaries and
raw timing logs are in `evidence/metadata-builder-20260918/`.

Apple M4 Pro 20-core GPU; three sequential processes with rotated shape order,
three warmups and 21 GPU timestamp samples per shape/process. A deterministic
mixed cover is resident before measurement. Times include only the metadata
kernels, excluding selection, upload, AQ and CPU work.

| Pixels | Process medians (ms) | Median (ms) | Prefix scratch (bytes) |
|---:|---|---:|---:|
| 3,008,000 | 0.573, 0.396, 0.385 | 0.396 | 196,880 |
| 8,294,400 | 0.545, 0.780, 0.532 | 0.545 | 541,200 |
| 24,000,000 | 1.028, 1.024, 1.015 | 1.024 | 1,565,456 |

Small-image process variation is substantial. These timings establish the cost
of this standalone builder, not a complete-encoder speedup. Device-generated
parameter records, AQ consumers, deferred host publication and ownership/error
handling are still required; see [HANDOFF.md](HANDOFF.md).

The existing AQ evaluation, complete quantization pipeline, completed-frame,
Metal storage, submission storage, cache admission and workflow admission tests
also pass with the new pipeline table (seven tests).
