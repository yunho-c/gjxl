# Default DC/AC section-writing overlap

Eligible non-batch encodes overlap selected DC/common and AC section writing by
default. This extends the production #1–#4 host-handoff work at
`e3b267cf0a7a2ae833d632e8af1faac2a9763954`. Entropy models, token order,
representation selection, and final codestream assembly are unchanged.

## Schedule and admission

The encoder writes and validates both DC and AC global headers, then submits
DC group jobs followed by AC group jobs to one dynamic queue. Free workers can
start AC jobs while other workers finish DC groups. The queue uses the existing
CPU-domain admission and maximum of eight section participants.

Each job owns a separate destination writer. Models and token backing remain
immutable until every worker joins. Task failures are selected in DC-first
index order, and launch/allocation failures use the existing joined worker
implementation. Both output arrays remain local until success; public output
publication remains atomic. Both global headers are checked before group jobs,
so internal header-versus-group failure sequencing differs from the old schedule.

Overlap requires CPU participation, more than one desired section participant,
no explicit nested-parallel scope, no batch marker, and no detailed serializer
profile. Explicit CPU=1, batch, profiled, and unadmitted calls keep the previous
common-then-AC schedule. The batch marker is propagated through high-effort
representation workers. A larger request receiving only one participant under
contention can execute the combined queue serially without exceeding the CPU
limit. The policy favors serial use of the single-image API; it does not promise
better throughput for concurrent independent callers. CPU-token and eligible
GPU-token single-image routes can both use it.

## Runtime opt-out

Set `GJXL_SECTION_WRITE_OVERLAP=0` to retain the previous section-writing
schedule. Unset the variable to use the production default. Configure this
process-wide control before encoding, not while calls are active.

Diagnostic builds configured with `GJXL_BUILD_TOKENIZATION_EXPERIMENT=ON` also
accept `GJXL_EXPERIMENT_SECTION_WRITE=0` to disable overlap. The stable opt-out
always takes precedence. Ordinary builds ignore the diagnostic variable. The
qualification-only `GJXL_SECTION_WRITE_CANDIDATE` macro has been removed.

## Storage

The storage plan bounds every possible mixture of simultaneously active DC and
AC scratch within the same worker ceiling, plus the combined task-status array.
It conservatively covers both schedules regardless of runtime controls,
including batch/profile fallbacks. Consequently, some configurations have a
larger minimum admitted memory budget even when overlap is inactive. The opt-out
does not lower that budget.

In the original sampled controls, the maximum added budget was 906,722 bytes
(0.865 MiB); sampled batch admission remained unchanged. Admission at every
possible memory threshold is not guaranteed to match older builds.

## Qualification and rollout decision

The original AC-only study is retained at
`/Users/yunhocho/GitHub/gjxl/reports/section-writing-20260927/REPORT.md`.
It found consistent complete-call gains in 11 of 15 primary settings. A longer
Kodak effort-4 repeat improved in all five rounds; the 48 MP and 24 MP
high-effort cases were roughly flat. The batch-of-four Kodak control remained
mixed with a 5.48% median slowdown despite overlap being inactive there. That
candidate-build observation remains unresolved; it is not evidence that the
batch scheduler now overlaps these writers. The user explicitly approved
production adoption for non-batch encoding with those results available.

That study verified exact output across 105 images at efforts 1/4/7 plus
high-effort and resource controls, independent decoder evidence keyed by
compressed hash, full suites, and focused ThreadSanitizer coverage. Diagnostic
stage spans are kept separate from ordinary public-call latency.

The production-default integration checks, including stable opt-out and batch
fallback, are recorded at
`/Users/yunhocho/GitHub/gjxl/reports/section-writing-default-20260927/REPORT.md`.
The original reports and their frozen artifacts are retained at those paths.
High-effort entropy preparation and ANS cost evaluation remain separate
optimization targets.

## Current-main integration (2026-10-10)

Merge `911015c9c0e7aaa9ecd31e0b207aaa36ec7f5d98` brings upstream main
`86301a11098b6b1808504ae44f975de81f6935b4` into the feature branch without
conflicts. The original overlap implementation remains in place alongside
current low-effort policies, CUDA tokenization/early-entropy scheduling, and
the opt-in EPF sharpness search. The section-writing contract test now uses
the shared cross-platform environment helper instead of POSIX-only calls.

The early-DC dispatcher joins its workers and releases protected CPU capacity
before selected section writing. CUDA admission composes the shared serializer
storage plan, including its mixed DC/AC scratch bound. These are source-level
integration checks; they do not establish CUDA hardware execution or timing.

This update retains the September performance qualification at the user's
request. It does not repeat performance captures or establish a current-main
speedup. In particular, the earlier batch-of-four timing caveat remains open.
Fresh validation is limited to build and correctness tests. Local commands,
logs, JUnit results, source hashes, and primary-checkout preservation checks are
retained under `build/section-writing-modernization-20261010/`.

The fresh ordinary Release build passed all **174/174 CPU/Metal tests**, with
no skips. This includes 222 earlier-DC contract cases, 288 entropy-readiness
contract cases, the opt-in Metal EPF search tests, and the installed consumer.
The diagnostic CPU build also passed its section-writing test. Both ordinary
and diagnostic section-writing runs verified **88 exact-byte encodes each**,
including planned memory, CPU limits, profile/batch fallbacks, shared callers,
and worker-launch failure cleanup. Both builds used C++20, with pinned libjxl
reference tests disabled. CUDA compilation/execution and Windows execution
were not performed on this Mac.
