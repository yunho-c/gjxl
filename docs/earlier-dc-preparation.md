# Earlier DC preparation

Eligible single-image Metal calls start DC preparation earlier by default in
normal builds. This extends the existing capacity-aware earlier entropy
schedule. `GJXL_EARLY_DC=0` restores its previous DC start boundary while retaining
earlier entropy readiness. Unset, `1`, and unrecognized values use the production
default. Configure this process-wide control before encoding; do not change it
while any encode or batch is active. `GJXL_EARLY_ENTROPY=0` takes precedence.
Experimental entropy modes other than capacity-aware mode 3 also take precedence.

The earlier fork follows frame validation and DC context-layout selection:

```
DC branch: DC tokens and metadata -> DC entropy model
AC branch: contexts/orders/signaling -> tokenizer Begin
           -> order model -> tokenizer Finish/populations -> AC entropy model
join -> existing selection, section writing and assembly
```

DC reads validated quantized DC, strategy, raw quantization, EPF sharpness,
color correlation and geometry. It does not read AC context/order choices.
The independent DC model helper also avoids reading the AC candidate vector
or order flags while the other branch constructs them. The AC provider is
captured on the caller before dispatch, because the provider scope is local
to that caller's thread. It has one sequential Begin/Finish owner.

Admission reserves the full desired DC participant count plus one AC participant
before starting either branch. It uses the existing per-image/shared-domain
budgets and protected reservation split. Partial grants are released before
fallback. The existing early-entropy path remains available after fallback.
This is the same two-branch dispatcher moved earlier, not another nested fork.
CPU-only, batch (including one in flight), rate-optimized, detailed-profile,
unadmitted, explicitly nested and insufficient-capacity cases do not start DC
earlier. The public timing-only API follows the ordinary scheduling policy.

Both branches join before selection or publication, including errors in AC
metadata, Begin, Finish, model construction and worker launch. Provider/token
backing and frame data outlive their readers. DC and AC failure statuses use
the dispatcher's deterministic branch-index order; public output stays unchanged
on failure. Shared single-image calls remain supported and resource-bounded.

The serializer storage plan already sums complete context/order preparation,
DC/AC tokenization and retained entropy-model bounds, plus simultaneously active
entropy scratch. Earlier launch extends those lifetimes within that summed
bound. The existing two Status entries and one worker-thread entry cover the
outer dispatcher; no second dispatcher or queue is introduced. Admission remains
independent of the runtime switch. Exact-bound workflow tests cover both starts.

AC qualification on 105 inputs found the strongest incremental benefit at
effort 4, including approximately 0.73 ms on campus12MP and 1.54 ms on alpine24MP.
Efforts 1/7 and smaller images were mixed; one effort-1 cross-binary comparison
remained adverse in longer repeats. Default enablement is an explicit rollout
choice, not a universal gain, batch throughput improvement, or a speedup on
other hardware. Earlier DC
alone versus the old all-token barrier is a different comparison from the
increment beyond the existing early-entropy default.

Implementation tests and AC-only results are retained in the local
[qualification archive](../reports/earlier-dc-20260927/REPORT.md). The subsequent
[default-promotion report](../reports/earlier-dc-default-20260927/REPORT.md)
records the production default and opt-out checks without rewriting that
historical evidence.
