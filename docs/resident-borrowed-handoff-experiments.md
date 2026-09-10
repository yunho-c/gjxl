# Borrowed AC scores and adjusted quant fields

These experiments found concrete reductions in copies and intermediate
allocations, but did not establish a repeatable whole-encode latency or
throughput benefit. Both remain unintegrated candidates that can be deferred
without treating their memory-efficiency benefit as zero. This note preserves
the findings and revisit criteria separately from the local experiment files.

The prototypes were tested against
`32cd437b709707c6de7debf93e0407ae1afabebf`. They are not part of the packed-input
integration in `eebbf8e`; that change is documented in
[packed resident input](packed-resident-input.md). The results below do not
requalify either the current branch or the prototypes on top of packed input.

## What each prototype changes

| Candidate | Removed work | Retained work |
| --- | --- | --- |
| Borrow completed AC scores | The CPU `resource.costs` vectors and the GPU-score-to-vector copy for each candidate family. CPU scatter reads the completed shared score allocation directly. | GPU scoring, completion synchronization, spatial `cost_storage` allocation/population, and deterministic CPU placement. |
| Borrow the adjusted quant field | The temporary `adjusted` and caller `adjusted_initial` buffers, both copies into those buffers, and the matching upload back into `resident_quant_field_`. CPU setup reads the completed shared field. | The initial field upload, GPU adjustment, completion wait, finite/positive validation, and CPU policy scans and bound construction. |

Source locations for reimplementation:

- AC preparation and handoff in
  [ac_strategy_search.cpp](../src/gpu/ops/ac_strategy_search.cpp), specifically
  `resource.costs`, `device_costs`, and the scatter into `cost_storage`.
- Quant orchestration: `adjusted_initial` and `policy_initial` in
  [adaptive_quantization.cpp](../src/gpu/ops/adaptive_quantization.cpp).
- Quant adjustment/readback: `AdjustQuantFieldResidentImpl` in
  [metal_aq_reconstruction.cpp](../src/gpu/metal/metal_aq_reconstruction.cpp).
- Matching quant re-upload: `MetalPreparedAqEvaluation::UploadInput` in
  [metal_aq_evaluation.cpp](../src/gpu/metal/metal_aq_evaluation.cpp).

Metal's copy helpers perform `memcpy` on shared storage. Borrowing therefore
removes explicit reads/writes despite unified memory. This is not a measurement
of DRAM traffic, bandwidth utilization, cache misses, or energy consumption;
those hardware counters were not collected.

For the 3839 x 2159 input, the padded 480 x 270 block field contains 518,400
bytes. Eliminating three field copies removes 1,555,200 bytes of copy payload
(approximately 1.48 MiB), plus two field buffers totaling approximately 0.99 MiB.
Counting each eliminated copy's source read and destination write gives twice
the payload; that is logical copy work, not measured DRAM bytes. AC removes
four bytes of host score backing and one four-byte copy per candidate. Both
prototypes kept production admission estimates conservative and unchanged;
smaller backing does not itself grant additional admitted jobs.

## Recorded performance and its limits

Positive percentages mean lower latency or higher throughput. Percentages are
medians of paired process ratios, not ratios of separately aggregated times.

| Same-binary comparison | Borrowed AC scores | Borrowed quant field |
| --- | ---: | ---: |
| Packed C API, padded 4K, latency reduction; five pairs | +1.18%; 4/5 pairs faster | -1.00%; 2/5 pairs faster |
| Two-caller padded-4K cohort throughput gain; five pairs | +0.61%; range -2.01% to +7.87% | -2.35%; range -4.65% to +1.84% |
| Original-float C++ 4K latency reduction; three pairs | +0.31%; range -2.19% to +1.29% | -0.42%; range -4.33% to +0.06% |
| Separate targeted stage probe; two pairs | Readback/scatter: 0.4381 to 0.4122 ms | Adjustment: 0.8309 to 0.7620 ms |

The stage differences, approximately 26 and 69 microseconds, are indicative
only. Their spans include retained work, the quant span excludes the later
re-upload, and combined-path attribution did not reproduce the savings
cleanly. They cannot be added into a guaranteed whole-encode saving. The C++
float path with both prototypes changed latency by only +0.10% median.

Calibration found the modified control slower than pristine by 1.37% in
padded-4K latency and 1.71% in two-caller throughput. The individual AC/quant
percentages above compare flags within the modified binary; they are not
qualified gains over pristine. Combined results that also include packed-input
conversion do not establish either candidate's independent contribution.

Configuration: Apple M4 Pro, 48 GB RAM, macOS 15.6, Release, forced Metal,
fully resident AQ, effort 7, distance 1.2, automatic CPU budget. Each process
ran 15 calls/cohorts, discarded three warmups, and alternated original/changed
images. Process pairs alternated execution order. Timings include the complete
synchronous API call, publication, and internal teardown, excluding backend
creation, caller-side input generation, hashing, and output freeing. Throughput
is two-request synchronous cohort throughput, not a replenished queue.
Known competing build/test/encoder activity caused whole-pair rejection;
ordinary desktop-load variation remained.

The combined prototype passed 404 byte-identical encodes across 98 cases,
four selected pinned-decoder checks, and 30 ASan encodes. These cover the
experimental implementation, not a future production rewrite. ASan did not
instrument GPU accesses and used `detect_leaks=0`. The adapted AC storage test
checked the smaller physical backing; reservations were not reduced.

## When to revisit

- AC borrowing is a candidate for a small ownership cleanup or a measured
  memory-pressure workload. Recheck actual backing and whole-call behavior;
  GPU production of the spatial consumer layout is a separate, untested way
  to remove the remaining scatter.
- Quant borrowing can remove the documented copies, but its copy-only version
  has no established latency/throughput case. Eliminating the GPU-to-CPU wait
  requires moving or restructuring CPU validation/policy setup and deserves
  a separate experiment with preserved numeric and rate-control decisions.
- For either production implementation, make completed-buffer ownership,
  generation validity, and mutation boundaries explicit. Readers must finish
  before reuse. Skip the quant upload only when the already resident field is
  the exact current input; the prototype's pointer check is not a general
  cross-call validity contract. Retain byte parity, failure/recovery, resource
  drainage, concurrency, and borrowed-lifetime coverage.
- Rerun isolated variants against the then-current pristine parent using
  matching headers/libraries, representative inputs, warmups, and alternating
  independent process pairs. Measure allocation/copy counts separately from
  latency and throughput. Collect hardware counters if claiming DRAM or energy
  savings. A decision to integrate for memory simplicity should state that
  objective rather than imply an established speedup.

## Local evidence and reproduction

The full artifacts remain local, outside this branch's tracked files, under
`/Users/yunhocho/GitHub/gjxl-resident-handoff-experiments/experiments/handoff/`:

- `REPORT.md`, `RESULTS.md`, `PROTOCOL.md`: conclusions, all paired results,
  exact timing boundaries, settings, and limitations.
- `prototype-complete.patch`: complete prototype including fixtures/helper;
  apply only in an isolated checkout of the recorded parent. It includes the
  packed-input prototype and experimental interfaces, not just these two
  candidates. SHA-256:
  `a16e617e5a74e668416125193395f3f0afb1a651c81138385a9687001d5a4fc7`.
- `driver.cpp`, `build_drivers.py`, `run_monitored.py`, `run_pristine.py`,
  `analyze.py`: build, execution, and analysis. Use fresh suite directories
  and the recorded manifest options, not the historical `patch.py` helper.
- `clean-main/`, `clean-core/`, `clean-attribution/`, `clean-calibration/`:
  relevant manifests, raw attempts, accepted-pair references, and summaries.
- `verification.json`, `build-seal.json`, `artifacts.sha256.json`,
  `validation/summary.json`, `asan-summary.json`: identity and correctness
  evidence. The unmonitored `screen/` results are excluded.

The prototype feature flags are `GJXL_HANDOFF_EXPERIMENT=0` (control), `2`
(borrowed quant), `4` (borrowed AC), and `6` (both). Flag `1` enables packed
input and `7` enables all three. The recorded original-float combined run
used flag `7`, where packed conversion does not apply. Match each baseline
and candidate's libraries to its own headers because experimental virtual
interfaces differ. Preserve the local artifact directory separately if full
raw-data reproduction is required after the experimental worktree is removed.
