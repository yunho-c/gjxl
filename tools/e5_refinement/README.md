# Effort 5 refinement qualification

The candidate changes ordinary e5 from one Butteraugli-guided quantization-field
update to zero. Initial spatial quantization, Gaborish, mixed-transform search,
and the writer remain enabled. E6 retains one update. High-density and
maximum-error overrides retain their existing policies.

`qualify.py` explicitly separates collection from saved-data analysis. The
65-image corpus and pinned external decoder/scorer come from the supplied
metadata file; there are no automatic downloads. It checks input, source,
binary, shader, scorer, and decoder-library hashes on every invocation.
Each quality probe checks deterministic bytes across two complete encodes,
decodes through pinned libjxl, and scores finite linear-sRGB pixels.
Decoded temporary files are deleted only after retaining their hashes and
reproducible codestreams. The incidental wall sample is not a timing qualification.

Prepare a fresh result directory with `baseline-source`, `candidate-source`,
and separate `baseline-build`/`candidate-build` CMake Release directories.
Build `gjxl_quality_benchmark` and `gjxl_encode` in both, with Metal enabled,
profiling disabled, and the same pinned metal-cpp headers. The candidate
snapshot must include the proposed diff and these scripts. Put the full patch
in `source.patch` and the exact revisions in `source-identity.json`.
Never reuse a CMake build tree from another source tree.

```sh
python3 tools/e5_refinement/qualify.py freeze --root "$RESULTS" --metadata "$METADATA"
python3 tools/e5_refinement/qualify.py rate --root "$RESULTS"
python3 tools/e5_refinement/qualify.py cross-metric --root "$RESULTS"
python3 tools/e5_refinement/qualify.py analyze --root "$RESULTS"
python3 tools/e5_refinement/outliers.py --root "$RESULTS"
python3 tools/e5_refinement/controls.py --root "$RESULTS"
python3 tools/e5_refinement/near_target.py --root "$RESULTS"
```

Run these phases serially. Collection resumes from hashed ledgers, and
calibration has a persistent fourteen-probe budget per image/arm. Unmatched
targets stay unresolved. Rate analysis uses the five predeclared distances,
requires four or more strictly monotone points and full 75–85 coverage, and
never repairs curves or extrapolates. It reports PCHIP and Akima results.
The three worst rate regressions are supplementary post-selection diagnostics;
they are not folded into the twelve preselected matched-quality averages.

The October 7 run resolved 8/12 strict score-80 pairs. `near_target.py` preserves
the four original failures and separately selects closely matched saved probes
within 80 ±0.1, with pair difference ≤0.01, without additional encodes. Its
selection and Butteraugli scores have separate ledgers. Do not present that
secondary matching as complete original target coverage.

`report.py` generates the report and plot from saved evidence. It requires
Matplotlib in addition to the collector's NumPy/SciPy dependencies. The local
run used Python 3.14 for collection and Python 3.13 for report rendering.

Also run the full native CTest suite, focused Metal API/shader validation,
and finite-budget resident batch checks. `controls.py` checks unchanged effort
and override codestreams, e5 scored/unscored parity, and target-byte/BPP retry
parity on CPU and Metal.

## AC-powered timing protocol

Timing was initially deferred on battery, then explicitly authorized after
AC power was available. `timing.py` freezes the eight strict and four secondary
pairs from `near-target-matches.jsonl`; the original strict calibration failures
remain visible. It verifies all frozen source, input, binary, shader and output
hashes, using the actual independently measured distances for both arms.

There are eight A/B process pairs and two interleaved unchanged-baseline A/A
pairs per image. Image order rotates, and arm order is balanced within both
four-round blocks. Each process performs one validation encode, two warmups
and five timed complete synchronous public calls. Initialization, input loading,
hashing and file writing are outside the timing boundary. Every output must
match its retained quality-study codestream. The primary statistic is the
equal-image geometric mean of median paired process ratios; uncertainty
resamples pairs within the fixed images, not a population of unseen images.

The one-second monitor requires AC power, clear thermal status, no swap growth,
no competing codec/build, no other process above 80% CPU, and aggregate other
CPU at most 200% (100% means one core). A contaminated pair is excluded in full
and a twenty-second quiet interval is required before retry. Every raw sample,
command and exclusion is retained. Frozen timing gates require at least 15%
aggregate time reduction, conditional 95% interval below no change, no image
median slowdown above 5%, A/A aggregate within 3%, and A/A image medians within
10%. Timing never uses incidental quality-probe wall times.

The October 7 run exhausted its original three-attempt budget after 33 accepted
and seven environment-excluded pairs. `timing_continuation.py` applies the
recorded uniform twelve-attempt amendment, preserving the original controller,
manifest and attempt ledger. Thresholds, samples, ordering and timing gates
are unchanged. Explicit user approval allowed a temporary `mediaanalysisd`
suspension with a watchdog that resumes it on collector exit or after thirty
minutes. Service-control events and watchdog sources accompany the evidence.

```sh
python3 tools/e5_refinement/timing.py freeze --root "$TIMING" --reference "$RESULTS"
python3 tools/e5_refinement/timing.py collect --root "$TIMING" --reference "$RESULTS"
# For the existing October 7 amended manifest only:
python3 tools/e5_refinement/timing_continuation.py collect --root "$TIMING" --reference "$RESULTS"
python3 tools/e5_refinement/timing_continuation.py report --root "$TIMING" --reference "$RESULTS"
python3.13 tools/e5_refinement/timing_report.py --root "$TIMING" --destination "$REPORT/timing"
```

Never rewrite a frozen manifest silently or reset retry counts. The report
audits complete coverage, retained samples, bytes, exclusions and restoration
events. Review the baseline-reproduced shader-validation assertion and unresolved
API-validation timeout separately. Batch throughput, cold starts, CUDA, and the
paper's historical throughput/comparison rows require separate measurements.

The first complete timing run failed the Kodak 08 A/A stability gate. Its two
unchanged-binary pairs differed by +27.257% and +0.056%, producing a +13.657%
median against the predeclared 10% limit. A separate `confirmation-plan.json`
declared exactly one fresh complete run, with the authorized media-analysis
pause active before measurement. All other settings use the amended protocol.
The confirmation must pass every original timing gate and its aggregate ratio
must be within 0.05 of the first run. Both runs are reported independently;
the first failure is never erased, pooled away, or relabeled as passing.
`timing_report.py` exports the confirmation decision separately, and `report.py`
links the completed timing evidence when regenerating the quality report.

The complete confirmation passed: aggregate warm-call time changed by -42.472%
(conditional 95% interval -42.807% to -42.045%), versus -42.482% in the first
run. A/A aggregate variation was +1.471%; all image controls were within 10%.
It retained 120 accepted pairs and ten environment-excluded attempts. The first
run's failed control remains reported separately. The measured speed benefit
supports the ordinary e5 policy change, with the documented +1.028% mean BD-rate
cost and larger image-specific perceptual regressions; e6 retains one update.

During confirmation, the user also approved temporarily pausing Spotlight's
updater and requested autonomous continuation. The pause was extended to the
remaining Spotlight indexer under the same restoration watchdog. One separately
logged 70-encode conditioning burst tested whether background work quieted under
foreground load; it did not enter the timing sample. All three paused processes
were restored and verified running afterward. The quality report links the two
timing reports, raw provenance, confirmation plan, service-control records and
preserved failure evidence under `reports/e5-zero-refinement-20261007/`.
