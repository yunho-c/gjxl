# AQ consumes the complete device-selected map

Prepared Metal AQ can now accept a packed device strategy map without waiting,
submitting work, or reading that map on the host. The next unprofiled resident
Butteraugli policy builds all strategy metadata and indirect dispatch records
inside its own submission, then runs adjustment, CfL, reconstruction, metric
updates and final coefficient coding. The selected host grid is materialized
only after that submission completes. Caller outputs are published together.

The preparation explicitly reserves prefix scratch and parameter records;
backends without the required direct-image transform pipelines report the
capability unavailable. Ordinary production frontend calls do not activate this
entry point yet. Scoring/selection composition and workflow admission remain the
next integration step, so this milestone does not remove the production ACS wait.

## Ownership, errors and frame output

The caller borrows the selection buffer through the next policy call, including
its failure path. Its producer must precede AQ on the same backend queue.
Reconfiguration validates geometry, storage, ownership and sharpness, then
uploads only sharpness. The metadata kernels write the evaluator's existing
strategy, anchor and CfL planes. Completed-frame allocation uses geometry and
maximum anchor capacity; its destinations are filled on device. Group used
coefficient counts depend only on base-block geometry. DCT8 sample decisions keep
the established AC-group-major random sequence.

Metadata errors are imported into the AQ error word before adjustment and
survive every reconstruction reset, including zero-update policies. Failed
maps yield safe bounded metadata but never a successful output. After the
completion check, the host materializes and verifies the compact map, rebuilds
host metadata without uploading it, and snapshots the final frame. This late
host rebuild is still an optimization opportunity.

The device storage plan includes all five prefix planes and 3,808 parameter
bytes. The host plan includes overlap during map materialization and atomic grid
publication. Completed-frame storage remains independently accounted.

## Retained evidence

Frozen executable, external metallib, changed sources and commands are in
`build/frontier/results/resident-aq-metadata-20260918/`, based on `7182d06`.
The [portable manifest](evidence/resident-aq-metadata-20260918.json) retains hashes
and normal/Metal API-plus-shader-validation results. Both runs pass:

- 292 existing CPU-builder/device-metadata cases and 27 family-dispatch cases.
- 27 complete device-metadata AQ cases, starting from a different provisional
  DCT8 host grid. Quant fields, scores, selected maps, sharpness, quantizers, CfL,
  float/integer DC, AC coefficients and completed populations match exactly.
- 12 malformed-map cases through actual zero-update and iterative AQ policies,
  including completed output. Every failure preserves caller outputs.
- 42 injected managed allocation failures, including 37 after GPU completion,
  preserve atomic publication. Three complete lifetimes fit the declared memory
  admission and release all resources afterward.
- Binding adds no device allocation, submission or wait. Each AQ policy uses one
  submission. Existing upload/submission/completion/numeric/readback failure and
  indirect-parameter guard checks continue to pass.

Ten existing Butteraugli, AQ, quantization-pipeline, completed-frame,
host/device/submission storage, cache and workflow admission regressions pass.
These results establish correctness of the consumer boundary. They do not
establish complete-encode speedup for the forthcoming combined path.
