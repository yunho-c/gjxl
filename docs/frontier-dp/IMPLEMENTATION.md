# Resident AC strategy implementation ledger

Objective: implement, qualify, document and commit GPU-resident AC selection
for ordinary efforts 5–8, including removal of the ACS-to-AQ CPU handoff where
feasible. Compare exact and cheaper policies against the existing selector using
complete encodes and measured quality. Effort 10 may retain the existing path.
After a satisfactory committed implementation, start fresh fixed and calibrated
GJXL study sessions using frozen clean binaries for the paper.

The initial original-bank Metal prototype is complete; see REPORT.md. It proves
solver correctness and standalone timing, not production integration or rate
improvement. This ledger does not redefine the objective around the prototype.

Remaining work and evidence gates:

1. Diagnostic full-encode policy comparison: existing greedy, full frontier,
   and recursive rectangle DP with exact-cost greedy fallback. Freeze binaries,
   retain codestreams and decoded hashes, score SSIMULACRA2 and Butteraugli,
   compare matched-quality curves, and inspect outliers.
2. Production device selection and data ownership. Read the existing family
   cost buffers directly; retain preparation reuse, numeric error propagation,
   partial tiles, and budget accounting. Select policy from measured evidence.
3. Resident AQ handoff. Produce strategy cells, deterministic grouped anchors,
   coefficient offsets, CfL metadata, and dispatch parameters on device.
   Defer strategy host materialization until final frame completion. Move
   strategy-aware quant-field adjustment and its AQ-bound reduction into the
   dependent device sequence. Eliminate the ACS wait/readback and the following
   adjustment wait instead of relocating them.
4. Correctness and performance qualification across ordinary efforts 5–8,
   small/partial/large images, prepared reuse, failure/atomicity, budgets and
   supported fallbacks. Compare identical selected maps through old/new AQ;
   measure complete-call latency and rate-quality rather than GPU duration alone.
5. Document final scope, controls, outcomes and limitations; make scoped commits
   without Python test files. Preserve the primary dirty checkout. Do not push
   or merge unless separately authorized.
6. Start new fixed-sweep and calibrated-run sessions from a clean frozen checkout
   of the committed implementation, with independent lifetime, bounded/resumable
   ledgers and no overlapping benchmark jobs. Report exact revision and coverage.

Current source dependencies: ordinary FindAcStrategyGridGpuImpl now scores and
selects on device but waits and reads the selected map.
MetalPreparedAqEvaluation::Reconfigure still builds grouped anchors, batch
offsets and CfL records on the CPU. Unprofiled resident Butteraugli AQ now adjusts
the quant field and computes bounds inside the AQ submission; profiled and
maximum-error paths retain the separate adjustment path.
GPU reconstruction, inverse transforms, block reduction and Butteraugli sinks
currently receive host batch counts, so resident metadata needs device parameter
bindings and indirect dispatch or a justified bounded dispatch implementation.

The diagnostic policy pilot and comparative kernels are complete; see
SELECTORS.md and the frozen rate-pilot/selector-comparison artifacts. Across six
images and efforts 5/8, rate-quality gains are small and include regressions.
GPU greedy matches the production CPU maps on 20,606 captured tiles and the
expanded 2,816-case synthetic suite, and is substantially cheaper than full DP.
Start production integration with GPU greedy; retain rectangle + greedy as an
experimental alternative and full frontier as a reference. This decision
preserves the existing policy while testing the residency benefit.

Native policy-preserving GPU selection is committed in `7ab1db3`; its complete
encode parity and timing are in NATIVE-SELECTION.md. Fused AQ initialization is
qualified separately in AQ-INITIALIZATION.md. The remaining metadata handoff and
its consumers are detailed in HANDOFF.md.

Progress: active. Device metadata, deferred completion, profiling the executing
new paths, final combined qualification and fresh paper studies remain open.
