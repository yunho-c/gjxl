# Tokenization host handoff

The `perf/tokenization-host-handoff` branch combines the four production-default
changes from the tokenization scheduling investigation:

| Change | Behavior and scope | Details |
|---|---|---|
| #1: Earlier population and entropy readiness | Eligible single-image resident Metal calls use capacity-aware DC/AC overlap. | [Scheduling and controls](entropy-readiness-capacity-policy.md) |
| #2: Earlier DC preparation | The admitted branch split precedes AC context/order preparation. | [DC preparation](earlier-dc-preparation.md) |
| #3: Host metadata handoff | Reuses validated strategy metadata and completed-frame validation. | [Metadata contracts](host-metadata-handoff.md) |
| #4: CPU population reduction | Uses the specialized reducer whenever the Metal provider produces populations, including effort 4. | [Reducer and qualification](ac-population-reduction.md) |

#1–#3 are inherited from `a4fce07600e2d20f2f0fcfb3bd350c9902f4f6ce`
(`perf/host-metadata-handoff`). #4 adds the qualified reducer and its independent
arithmetic test. This branch does not include the separate High-X/Y filtering
optimization on `perf/early-entropy-schedule`.

Each change retains its own eligibility and fallback rules. In particular,
batch calls retain their existing scheduling even though their Metal provider
can use the new CPU population reducer. The existing controls
`GJXL_EARLY_ENTROPY=0`, `GJXL_EARLY_DC=0`, `GJXL_HOST_METADATA=0`, and
`GJXL_GPU_TOKENIZATION=0` retain their documented meanings. Configure these
process-wide settings before encoding, without concurrent environment mutation.

The improvements have different effort/input profiles; their separately
measured percentages should not be added together. The #4 default decision
accepts the recorded small, uncertain effort-4 penalty in exchange for broader
measured benefits. See the linked qualification notes for the exact boundaries.
