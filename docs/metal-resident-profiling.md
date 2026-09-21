# Profiling the production resident path

Stage profiling uses the same resident computation as ordinary Metal encoding:
GPU candidate scoring and greedy selection, device strategy metadata, indirect
family dispatch, and initial quant-field adjustment and policy bounds. Eligible
search and AQ execute in one command buffer with one completion wait. Fixed-DCT8
efforts still omit search; dense search and backends without direct image
transforms retain the same fallback boundaries as their ordinary execution.

The profile includes candidate-family stages (`frontend.ac_strategy.dct*`),
`frontend.ac_strategy.select`, `frontend.ac_strategy.metadata`,
`aq.strategy_dispatch`, `frontend.quant_adjustment`, and `aq.policy_bounds`,
followed by the existing reconstruction, filtering, Butteraugli and final-frame
stages. In the combined path these belong to the `resident.aq` submission.
Stage names describe computation, not separate submissions or CPU handoffs.

The encoding profile retains production's bounded output materialization. It
does not request full quant fields, distance maps, or reconstructed RGB merely
to collect timestamps. Resident input preparation still has no GPU stage graph,
so the sum of recorded GPU intervals is not complete encoder wall time.

## Indirect dispatch and empty stages

Raw profiles retain schema version 4 and add
`execution_path: "production-aligned-resident-v1"`. Dispatch `kind` can now be
`indirect_threadgroups`. Its grid is read from the retained shared argument
buffer **after the existing command-buffer completion wait**. Encoding uses the
original indirect dispatch: no argument readback, extra transfer, or host
selection occurs between GPU stages.

Some transform families have zero selected blocks. Metal may leave timestamp
slots unwritten for encoders containing only empty indirect grids. Such stages
are verified from completed arguments, retain zero duration, and have
`timestamp_valid: false`. This is known zero work, not a measured zero-length
interval. Nonempty stages with invalid counters remain errors. Dispatch
timestamps are valid only in dispatch mode; stage mode records dispatch
identities and dimensions without per-dispatch timing. Published graphs own no
Metal argument buffers.

## Remaining measurement overhead

The profiler still creates one compute encoder per stage inside the command
buffer. It allocates timestamp buffers and records stage/kernel metadata, then
resolves counters and shared indirect arguments after completion. Dispatch mode
also inserts counter-sampling barriers and requires device support. This change
does not optimize or quantify those costs.

Measure them before redesigning the profiler. First compare alternating ordinary
and stage-profiled complete encodes using the same build, device, image, effort,
distance, CPU threads, warmups and output checks. Use multiple image sizes and
efforts, reporting paired wall-time ratios and absolute milliseconds. Do not
infer host overhead by subtracting summed GPU stages from ordinary wall time.

If total overhead matters, separate it with controlled variants: ordinary
single encoder; identical kernels split at the profiling boundaries with no
counter collection; then full profiling. The second-minus-first comparison
estimates splitting cost. Full-minus-split includes counters and recording;
separate host timers around recording and post-wait resolution can then identify
which part merits optimization. These variants are a proposed follow-up, not
new production modes implemented here.

Historical profiles predating this alignment retain their original execution
path. Recollect data before using GPU stage proportions to discuss production
bottlenecks; the new marker does not retroactively qualify older notebook data.

## Validation

The focused tests compare ordinary and profiled bytes, decisions, submission
counts, and output materialization. Whole-workflow storage tests cover efforts
1–10, tiny/odd/multiscale inputs, final-score modes, DC policy overrides and
repeated calls under precomputed resource budgets. Combined-path tests also
exercise prepared reuse and upload, submission, completion, readback, numeric
and staging failures in both ordinary and profiled execution.

Build with `GJXL_ENABLE_METAL_PROFILING=OFF` and
`GJXL_BUILD_FRONTIER_EXPERIMENT=OFF` for production shader settings. The former
controls shader debug information, not timestamp profiling. No corpus sweep or
production overhead qualification is implied by these correctness tests.

On the development Metal device, eight focused CTest suites passed, along with
the three stage-profile/kernel-selection/unsupported-dispatch CLI tests. The
full encoding CLI suite passed 11 of 13 tests. Its effort-nine entropy-label and
eliminated-host-phase assertions also fail in a fresh, unchanged `b1a7373` build;
those unrelated expectations were left unchanged. Local logs are in
`build/profile-alignment-tests.log`, `build/profile-alignment-cli.log`, and
`build/baseline-cli.log`.
