# Scheduling experiment and production work remaining

The central change is to move the first entropy tasks across the all-tokens
barrier, keeping their existing algorithms and subsequent selection/writing.
This is implemented as two synchronous branches; it does not require a new
public asynchronous API.

After the existing tokenizer Begin:

```
branch A: DC preparation -> DC entropy model
branch B: order entropy model -> AC Finish/reduction -> AC entropy model
join -> existing selection, writing, and assembly
```

Mode 1 uses the same two branches for DC preparation and AC Finish only, then
retains the original entropy phase. This distinguishes the benefit of moving
population reduction from the additional benefit of releasing entropy models.

The order model is ready before submission, but this prototype starts it on
branch B after Begin. Its short duration can overlap the GPU command. It does
not add another worker or move coefficient-order signaling before submission.
DC retains its original launch point. Thus measured gains do not include the
separate earlier-DC or resident-metadata proposals.

## CPU participation

The dispatcher first attempts a nonblocking reservation for one additional
participant. If unavailable, the original schedule executes. Otherwise the
branch ceilings are `workers - 1` for DC and one for AC completion/modeling.
`EncodeScope` resets only local nesting depth inside the already admitted
outer group, retaining the shared domain, per-image budget, and resource owner.
This avoids the existing helper's rule that serializes explicitly nested loops.

This remains a static split. A worker waiting on the GPU suspends active CPU
participation but keeps protected capacity for safe resumption. It cannot lend
that reserved slot to DC. In particular, CPU=2 leaves only one DC participant;
on an image with several DC groups this can erase the entire headroom. A
production policy could retain the old path when the split would reduce DC
group capacity, or use a shared ready-task queue to defer reservation until
completion work can run. Neither policy is evaluated by this prototype.

## Ownership and errors

Each branch writes separate outputs: DC groups/streams and its entropy code,
versus AC views/populations/code and the order code. The frame and selected
orders/context map remain immutable. Entropy code/cost storage survives until
the existing section writers finish; provider-owned token backing survives
all entropy tasks and writers. The top-level representation is never read by
the writer before both branches join.

The existing `RunParallelWork` catches worker exceptions and joins all started
threads before status propagation. Branch-index order retains deterministic
error selection. Public output is published only on success. The Metal provider
destructor still waits for outstanding submission completion before releasing
its arenas, including failures before Finish consumes the tokens.

The dedicated contract harness uses a deterministic provider to exercise
completion and launch errors, budget fallback, and output publication. The
full suite additionally retains real Metal allocation/submission/completion
failure tests. Captures verify actual GPU output, forced capacity retry, byte
and summary identity, shared CPU limits, and zero post-call reservations.

## Profiling and high-effort scope

Native additive stage profiles retain the old schedule. Their old DC/AC/entropy
phase partition cannot describe these overlaps just by moving timers; a
production patch needs explicit readiness events and a documented elapsed
partition. The study's independent trace sink records the overlapping spans,
and ordinary public-call comparisons have tracing disabled.

Mode 2 retains the old schedule for rate-optimized effort 8–10. That path
constructs balanced and expanded complete representations concurrently under
separate CPU ceilings. Moving both representations' DC/order/AC tasks across
the barrier needs a broader task graph and must retain the complete-file
comparison and fallback semantics. Mode 1 measures the earlier-Finish part
without restructuring either entropy representation.

## Review artifacts

- [Measured source patch](measured-source.patch): trace hooks plus the prototype.
- [Measured source patch](measured-source.patch): exact changes from the pinned base, including trace hooks and toolchain support.
- [Clean review patch](prototype-clean.patch): same scheduling experiment with
  trace hooks removed, relative to the pinned encoder source. This is a review
  artifact; full measurement results belong to the hashed instrumentable binary.
- [Contract tests](contract_test.cpp) and [protocol](PROTOCOL.md).

Before production integration: settle scheduling/admission policy, account for
native profiling and both high-effort representations, write the live-dispatcher
storage-bound argument, broaden corpus/power-condition validation, and validate
a production build without measurement hooks. No default switch is made here.
