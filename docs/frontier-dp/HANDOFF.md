# ACS-to-AQ handoff design and qualification

Historical design audit after native GPU selection and fused AQ initialization, based on
`7ab1db3` plus the retained AQ-initialization patch. This is an implementation
plan. The implemented combined path and qualification are in COMBINED-HANDOFF.md.

The selected map is still read after `FindAcStrategyGridGpuImpl` waits for its
scoring/selection submission. `MetalPreparedAqEvaluation::Reconfigure` derives
the metadata below on the CPU, uploads it, and prepares host dispatch parameters.
Consequently, removing `Wait()` alone would be incorrect.

| Consumer | Selection-dependent data | Required ordering |
|---|---|---|
| Forward/inverse DCT and quantization | Seven family counts, anchor offsets, coefficient offsets, grouped anchors | Global row-major order within each family; channel-major coefficients within each family |
| Quant-field adjustment | Family anchors/counts and covered dimensions | Each selected rectangle owns its blocks |
| Final CfL | Six-word transform records, tile offsets, coefficient offsets/strides | Tile-major records, then row-major anchors inside each tile |
| EPF/postprocessing | Two-word strategy/anchor map | Global row-major base blocks |
| Resident Butteraugli sinks | Family descriptors and score partials | Same family/anchor mapping as the DCT passes |
| Completed frame | AC-group destinations, population-family mask, optional DCT8 samples | Row-major anchors within each 32×32-block AC group |
| CPU frame assembly | Selected strategy grid and transform layouts | Materialize after final AQ completion |

## Device metadata construction

A bounded stable prefix/scatter pass can reproduce the host metadata. Divide
the global row-major base-block array into fixed-size chunks. Count anchors by
family in each chunk, scan the small chunk-count arrays, and scatter anchors
using the chunk prefix plus the within-chunk rank. The seven total counts give
both anchor offsets and `3 * count * coefficient_count` coefficient offsets.
Total coefficient capacity stays at three full image planes, independent of the
selected cover. Allocating a maximum-sized coefficient bank for every family is
unnecessary and would invalidate existing memory admission assumptions.

CfL tile counts require a separate prefix over tiles. A per-tile pass can then
emit the six-word records in the existing order, using each anchor's global
family rank. Invalid selector flags must propagate to the eventual AQ error
readback; any metadata emitted on a failed tile must remain safe to consume.

Completed-frame destinations need a prefix of rectangle areas over anchors
within each AC group. Each group's total used coefficient count is actually
selection-independent: it is 64 times the number of base blocks in the group.
This permits allocation and tail clearing before selection. Allocate destination
and sampling capacity for the maximum anchor count, then derive the actual
population-family mask on device. Pure-DCT8's existing deterministic sampling
order must be preserved; it is distinct from the ordinary mixed-family path.

## Dispatch and output integration

The ordinary direct-image DCT pipelines already take compact parameter records.
Bind device-generated records and use indirect dispatch, or bounded dispatch
with a uniform count guard, instead of host `setBytes` counts. The initial path
can require these image pipelines and keep the established fallback for other
transform implementations. Quantization, LLF/DC reconstruction, adjustment and
resident Butteraugli batch descriptors need the same treatment.

Resident Butteraugli's final scalar reduction is a **maximum**, not a sum.
Zeroing unused score-partial slots and reducing the maximum anchor capacity can
therefore preserve the result without changing an FP32 addition tree. The
per-transform distance calculation and its active anchors must still match.

The provider interface currently promises a completed `AcStrategyGrid` before
AQ starts. Introduce an explicit deferred strategy result and completion phase;
do not pretend the provisional DCT8 preparation is the selected grid. The search
owner must retain the outstanding submission and rate scratch until the metadata
consumer finishes. Failure, reset and destruction must drain outstanding work
before releasing/reusing borrowed buffers. Queue ordering and error propagation
must be part of the Metal handoff contract.

`PrepareCompletedFrame` currently reads selected host anchors before submitting
AQ. Split geometry-only allocation from selection-dependent device packing and
post-completion host metadata publication. Other diagnostic frame outputs can
materialize the same authoritative map after completion. A failed final readback
must leave caller outputs unchanged.

## Qualified foundations

The metadata builder (`407cd05`), indirect consumers (`7182d06`) and complete
AQ device-map boundary (`197a20b`) are qualified in METADATA.md, DISPATCH.md and
RESIDENT-AQ-METADATA.md. The ordinary frontend now composes the producer and AQ
in one command buffer; see COMBINED-HANDOFF.md for ownership/admission, fallback
scope, exact encode comparisons and the remaining full-call timing gate.

## Acceptance gates

1. Compare every device metadata field and coefficient destination with the
   existing host builder for all partial tiles, mixed families, pure DCT8 and
   AC-group boundaries, including reused preparations.
2. Prove device/host memory plans cover new prefix scratch, parameter buffers,
   maximum-capacity output metadata and error paths.
3. Require ordinary and Metal-validation runs, invalid-cost/metadata tests, and
   submission/completion/readback failure atomicity and lifetime tests.
4. Require byte-identical complete encodes against the frozen CPU-policy control
   before timing. Preserve efforts 5–8, zero-update AQ and iterative AQ; retain
   dense-bank fallback.
5. Verify the ACS-to-AQ host wait is absent in the actual executing path, then
   measure complete-call latency. Retain profiler fallbacks explicitly until
   their stage graphs and admission plans describe the new path.
