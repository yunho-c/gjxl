# Opt-in CUDA resident token provider

Base 715f033; preserve the independently qualified owner and kernel worktrees.
This worktree combines their reviewed source with a real Begin/Finish provider.
It is built with GJXL_BUILD_CUDA_RESIDENT_TOKEN_EXPERIMENT and selected only by
GJXL_EXPERIMENT_CUDA_GPU_TOKENS=1. The stable GJXL_GPU_TOKENIZATION=0 override
takes precedence. No production default is changed.

The completed owner copies packed int32 coefficients device-to-device before
sparse assembly can overwrite compact-width aliases. Native host sparse/compact
coefficients remain available to shared order/context selection. The provider
uses the actual packed stride, validates scan/index/capacity bounds, uploads
managed metadata and starts GPU work in Begin. CPU DC groups can then run before
Finish waits, reads counts and performs a guarded output-capacity retry if
required. Count readback is intentionally in Finish: pageable CUDA D2H can block
the calling thread, so enqueueing it in Begin would not establish useful overlap.
Split values/contexts and integer histograms are read back for the shared CPU
serializer. No CPU token count sizes device output. A backend-local capacity
hint comes only from prior device counts.

All metadata, token arrays, histogram/population owners and device allocations
are managed. A geometry/context-policy storage recipe bounds exact host reserves
and twice the complete device envelope to cover replacement/cache overlap. The
workflow adds the independent completed coefficient allocation and token storage
to its existing admission bound. This is conservative capacity, not measured
peak. Allocation/submit/completion failures preserve public outputs; provider
destruction drains its submission before freeing buffers.

Eligible routes are fully resident/throughput CUDA with a completed frame and
non-exhaustive entropy. Existing maximum-error, exact-coefficient, maximum-
throughput and exhaustive routes retain CPU tokenization. Detailed host/GPU
profiles currently retain the CPU path too; this initial integration has no new
semantic profiling support. Target-size attempts each create their own owner
and provider. Complete memory/batch qualification and profiling parity remain
requirements before any promotion.

Native qualification: compact OFF and ON; unchanged 2,048-case kernel oracle;
512 direct provider cases/2,048 exact groups (eight strategies/mixed, four
patterns, natural/custom orders, four maps and populations on/off); owner and
existing sparse suites; eighteen whole-workflow policies including stable CPU
override, finite repeated admission/rejection and target search; existing CUDA
workflow admission suite. Four sanitizer modes cover provider and whole-workflow
smoke fixtures in both compact builds. Freeze source/archive before building.
Any failed attempt remains intact; no timing claim follows from these checks.

After independent native review, collect complete-call first/warm matched-main
timings on retained natural and large inputs, plus scored/fallback/search and
finite batch policies. Retain CPU/CPU controls, exact output/decoder checks and
all observations. Diagnostic warm stage gains are not a public encode forecast.
