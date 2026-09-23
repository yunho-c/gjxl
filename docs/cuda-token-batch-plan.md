# CUDA token batch qualification

This candidate reproduces the 1,274 source files in the reviewed profiling
snapshot, `gjxl-cuda-token-profiling/build/profile-native-r1/source.zip`, then
adds only a batch fixture, its build target, and this protocol. Production
tokenization, kernels, ownership, memory planning and scheduling are unchanged.

The public persistent batch driver encodes four different geometries, including
partial groups and multiple groups, using efforts 1/4/5/8, fully resident and
throughput AQ, scoring and maximum-compression CPU fallback. A fifth invalid
request verifies per-image error isolation. CPU-token single-image encodes are
the exact byte/full-summary oracle. A propagated worker observer measures real
thread-local GPU-provider calls for every request, without changing scheduling.

For each worker count 1/2/4, test both the full planned finite budget and the
minimum budget which forces one in-flight image and cache trimming. Each case
runs three enabled batches on the same driver, a stable CPU-override batch, an
injected serializer allocation failure, and recovery. Published results survive
shutdown. A one-byte-short minimum budget rejects the whole batch before any
worker enters and leaves caller output unchanged. Check resource and CPU limits,
default-domain escape, cleanup, result order and queue/service timing.

Full coverage is six policies and 36 mixed-size batches per compact variant;
the sanitizer smoke fixture uses two configured workers with the minimum budget
(one admitted work slot), six batches, failure and rejection checks. Native
qualification uses compact AC off/on and all four CUDA sanitizers. This is a
correctness study, with no throughput or default-enablement claim.
