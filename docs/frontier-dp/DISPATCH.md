# AQ consumers driven by GPU strategy counts

AQ's ordinary transform and reduction kernels can now consume device-generated
family parameters and indirect dispatch grids. This is enabled through an
internal qualification seam; ordinary encodes still use the host handoff.
The test provides the GPU metadata builder's family table while preserving the
CPU-prepared anchors and other metadata as the consumer oracle. The separate
292-case metadata comparison checks the device equivalents of those buffers.
Connecting all device metadata, ownership and deferred publication is still
required for a resident ACS-to-AQ path.

A shared host/Metal record ABI holds the existing parameter payloads and seven
indirect grids per family. A small kernel patches every selection-dependent
count/offset from the validated family table. Seven records require 3,808 bytes.
Existing reconstruction arithmetic is unchanged. Forward/inverse DCT, quantizer
adjustment, reconstruction, DC low frequencies, initial quant-field adjustment,
coefficient-population counting and block-distance reduction bind these records.
The direct-image SIMD-matrix DCT implementations are required; other transform
implementations retain their established path.

The scalar/parallel adjusted-quantization choice depends on the actual count.
Both variants are encoded with mutually exclusive indirect grids, preserving
the established switch at 256 anchors. Empty families have zero-sized grids.
Completed-frame population counting derives the pure-DCT8 sampling decision from
the device family table.

Resident Butteraugli patches its own per-family payloads into the same records.
It clears the full base-block-capacity score-partial plane to zero, then reduces
the maximum after active anchors finish. This preserves the existing scalar
score without a host read of the total anchor count. Small-image complete-map
reduction also uses device parameters. Indirect dispatch profiling is explicitly
unavailable in this qualification mode because the current profile format
requires host-known grids.

## Evidence

Frozen executable, external metallib, changed sources, parent revision
`407cd05` and commands are retained in
`build/frontier/results/indirect-aq-dispatch-20260918/`. The portable manifest,
including normal/validation results, is in
[evidence/indirect-aq-dispatch-20260918.json](evidence/indirect-aq-dispatch-20260918.json).

- All 292 CPU-builder/device-metadata cases continue to pass.
- 27 paired AQ cases consume the actual GPU-generated family tables. They cover
  partial tiles, 69×69-block mixed/large grids, pure DCT8, empty families,
  255/256/257 anchors for each larger family, and 8-pixel-wide image fallbacks.
  Fixture assertions confirm that each requested count boundary is reached.
- Quant fields, score histories and quantized DC/AC coefficients match exactly.
  Owned and completed-frame outputs are exercised; completed-frame population
  counts and masks also match. Each policy call uses one submission. Borrowed
  parameter guards remain intact.
- Invalid parameter capacity/aliasing and profiling requests are rejected.
  Upload, submission, completion, numeric and readback failures preserve output
  atomicity and invalidate the preparation.
- Normal and Metal API/shader-validation runs pass. Nine existing Butteraugli,
  AQ, full quantization pipeline, completed-frame, host/device storage, cache
  admission and workflow admission regressions pass.

These are correctness results, not a complete-encode or latency qualification of
the future handoff. Production storage admission must include the prefix and
parameter planes. Selection/submission lifetime, error import, geometry-only
completed-output preparation and post-AQ host publication remain open, as listed
in [HANDOFF.md](HANDOFF.md).

The subsequent complete device-metadata consumer qualification is in
[RESIDENT-AQ-METADATA.md](RESIDENT-AQ-METADATA.md).
