# GPU tokenization default rollout

22 September 2026, branch `perf/gpu-tokenization`, following the qualified implementation at `59d9761` and evidence checkpoint at `7a53e1a`.

Resident Metal encoding now selects the measured V7 GPU tokenization configuration in ordinary builds, with compact guarded allocation, CPU/DC overlap, one histogram shard, 128-thread groups, and scalar emission for pure-DCT8 frames. Whole-group DCT8 fusion remains experimental. `GJXL_GPU_TOKENIZATION=0` selects CPU AC tokenization without changing the Metal frontend; `1` explicitly enables the default GPU policy.

The rollout also scopes GPU token arena admission to resident Metal serialization. CPU/CUDA, compatibility, and exhaustive CPU serialization do not reserve token device arenas. Default selection has no size, effort, batch, or memory threshold yet. The known 24 MP/e1 batch regression is accepted provisionally while the CPU override is available; broader measurements will refine the policy.

[Usage and policy](../../docs/gpu-tokenization.md) describe supported routes, the process-wide override, and development controls. [The earlier measurements](../tokenization-20260922/REPORT.md) remain the performance evidence; their historical opt-in checkpoint is preserved. No new throughput or latency claim is made here.

## Validation

| Configuration | Result | Retained log |
| --- | --- | --- |
| Production Release, experiment controls OFF | 159/159 passed | [Full suite](production-full.log) |
| Production default, oracle and admission checks | 6/6 passed | [Focused suite](production-focused.log) |
| Production with CPU tokenizer override, resident/public admission | 2/2 passed | [CPU override](cpu-override-tests.log) |
| Metal disabled, serializer and CPU workflow storage plans | 2/2 passed | [CPU-only build](cpu-tests.log) |
| Experiment controls ON, default/oracle/failure recovery | 3/3 passed | [Experimental checks](experiment-tests.log) |

The new integration test compares encoded bytes and summaries for CPU override, default GPU, and explicit GPU across efforts 1/4/7/8/9/10. It verifies additional GPU submissions, separate token metadata/output pool admission, unchanged CPU/compatibility plans, and exhaustive serialization remaining CPU-tokenized. The independent oracle checks 2,560 GPU submissions and 10,240 groups; production uses the default configuration, while the experimental build retains shard and failure-injection coverage.

The existing finite-domain workflow and batch tests cover resource bounds, cache behavior, rejection and recovery. The full production suite includes public APIs, CLI, and install-consumer checks. The inherited intermittent profiler registry limitation recorded in the earlier study was not changed by this rollout; all tests in this run passed.

Screening scripts now explicitly select each named CPU/GPU mode, set their ablation parameters, record the stable override, and capture the source commit as well as the diff. Their Python syntax was checked. Existing captures remain tied to their frozen scripts and binaries.

[Validation metadata](validation.json) records build configuration, log hashes, and source hashes. Build logs and binaries are generated under the worktree's ignored `build/` directory. Test durations are correctness-run durations, not performance measurements. No push or merge was performed.

## DC test portability follow-up

The pre-merge review reproduced a failure in the new DC fault-injection test when the hardware-concurrency query returned 1, 2, or 3. The dispatcher correctly caps worker launches at the hardware count and uses the caller as one participant; the test incorrectly expected every injection point to be reachable.

The test now skips only unreachable launch-failure points, matching the existing worker-failure test convention. Its eight stream-equality, finite-storage, and CPU-cap cases still run at every hardware count. The output reports how many launch-failure cases actually ran.

[Before/after validation](dc-portability-validation.json), using a [diagnostic link-time shim](dc-portability-hardware-shim.cpp), passed simulated hardware counts 0/1/2/3/4/8. Zero and one ran eight ordinary cases and no launch-failure cases; two ran 12 total/four failure cases; three ran 16/eight; four and eight ran all 20/12. Zero exercises the standard's unknown-concurrency result and the dispatcher's one-participant fallback. The normal Metal-enabled and Metal-disabled Release test targets also passed. This is simulated scheduler-query coverage, not validation on six physical machines.

Only the test and this evidence changed. The production encoder retains its earlier 159/159 full-suite qualification; the full suite was not repeated for this test-only adjustment.
