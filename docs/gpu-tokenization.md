# GPU tokenization default

Resident Metal encoding uses GPU AC tokenization by default. No experimental CMake option or environment variable is needed. The default includes guarded compact token allocation, one histogram shard, a 128-thread emitter, scalar emission for pure-DCT8 frames, and overlap with parallel CPU DC groups. This is the V7 configuration measured in the [qualification report](../reports/tokenization-20260922/REPORT.md). Optional whole-group DCT8 fusion remains experimental and disabled by default.

To use CPU AC tokenization while keeping the rest of the Metal encoder:

```sh
GJXL_GPU_TOKENIZATION=0 <encoder command>
```

`GJXL_GPU_TOKENIZATION=1` explicitly enables the default GPU policy. An unset or unrecognized value uses the default policy. The setting is process-wide: configure it before starting encodes and keep it fixed while jobs, batches, or their admission plans are active. It takes precedence over the legacy experimental GPU selector.

GPU tokenization applies to resident Metal fully-resident and throughput workflows with a completed resident coefficient buffer, including eligible target-size searches. CPU/CUDA backends, compatibility routes, maximum-error workflows, and exhaustive maximum-compression serialization keep CPU tokenization. The CPU still chooses coefficient orders, contexts and entropy models and writes the bitstream. The switch preserves encoded bytes; it changes execution and resource use.

The resident Metal admission plan includes token metadata/output arenas and their idle pools. CPU-only, compatibility, and exhaustive serializer plans do not reserve these arenas. Selection does not currently adapt to image size, effort, batch concurrency, or memory budget. Resource errors retain the existing error/publication behavior; they do not silently trigger an unplanned CPU retry.

## Performance policy

Default-on is an explicit rollout choice based on the measured benefits, with the CPU override available for regressions. On the qualification M4 Pro, large-image single-call latency improved at efforts 1, 7, and 8. The 24 MP/e1 four-image batch had a repeatable throughput regression against the improved CPU tokenizer. Other measured e1 batch cases improved, so effort and requested batch size alone do not identify the regression.

Broader measurement of image content/size, quality, actual concurrency, cold/warm state, memory budgets, and devices will refine automatic selection. The historical report describes the opt-in checkpoint at `59d9761`; this default change does not turn that limited cohort into a universal performance guarantee or provide new timing results.

## Development controls and validation

`GJXL_BUILD_TOKENIZATION_EXPERIMENT=OFF` is the normal build configuration. The option now gates diagnostic ablations, including alternate layouts, overlap, shards, emitters, forced capacity retries, whole-group DCT8 fusion, and parallel CPU token cleanup. It is not required for production GPU tokenization. Experimental builds accept `GJXL_EXPERIMENT_GPU_TOKENS=0` as a legacy CPU selector unless the stable override is set.

The ordinary build includes the independent GPU-token oracle and `gpu_tokenization_default` integration test. The latter compares default GPU, explicit GPU, and CPU-override bytes/summaries, verifies actual GPU submissions across efforts 1/4/7/8/9/10, and checks CPU/compatibility/exhaustive admission isolation. The existing resident and public-batch admission tests cover bounded resources and repeated calls. Failure-injection and tuning-specific tests remain available in experimental builds.

The screen runners explicitly select CPU or GPU tokenization for each named mode, record the stable override, and retain the source commit beside their diff. Use a fresh directory when collecting with the updated runners; historical runs remain bound to their frozen runner hashes. Rebuild standalone capture/batch binaries before collecting from changed source.
