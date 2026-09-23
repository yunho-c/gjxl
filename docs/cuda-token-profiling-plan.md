# CUDA token profiling parity

Historical development protocol. See the [current checkpoint](cuda-metal-catchup.md)
for integrated behavior, selection flags, completed qualification and limitations.

This isolated candidate starts from all 1,272 source files in the qualified
`gjxl-cuda-token-provider/build/provider-native-r4/source.zip` snapshot.
The resident token experiment stays build-time and runtime opt-in. Its existing
performance studies do not justify enabling it by default.

Host-only, GPU-stage and GPU-dispatch workflow profiling now preserve the same
tokenizer selection as the plain encode. The stable CPU-token override and
unsupported-policy exclusions retain precedence. This matches Metal's workflow
route behavior; neither backend's public workflow graph is a complete token
kernel trace. Separate token-kernel diagnostics are outside this change.

The independent device coefficient copy receives a named timed stage,
`aq.completed_token_copy`, within the resident policy capture. It has no kernel
dispatches. Admission accounts for the additional submission and capture entry.
Unprofiled copies do not create profile events or diagnostic owners.

Qualification uses Release SM86 with compact AC both off and on. The dedicated
43-case fixture covers host/stage/dispatch profiling, repeated finite-domain
encodes, two geometries spanning group boundaries, efforts 1/5/8, resident and
throughput policies, scoring, maximum-compression CPU fallback, and host-profiled
target-size retries. It compares full bytes and summaries against a CPU-token
oracle, observes actual provider calls, validates copy timelines and graph
bounds, exercises the stable override, and rejects a one-byte-short reservation
without publishing any output or submitting GPU work. The existing full workflow
profile fixture adds 160 profile encodes and a diagnostic allocation-failure
sweep per build. Existing owner/provider/kernel/workflow/admission tests remain
in the regression set. Both builds run all four CUDA sanitizers on provider,
workflow and six-case profile smoke fixtures.

Passing correctness checks establishes profiling compatibility only. It does
not establish complete-encode speed, batch qualification or default suitability.
