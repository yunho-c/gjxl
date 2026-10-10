# Earlier population reduction and entropy readiness

This package records the completed 2026-09-26 scheduling investigation on
Apple M4 Pro. Start with the [findings](FINDINGS.md), [full measured report](REPORT.md),
[design](DESIGN.md), and [protocol](PROTOCOL.md).

The prototype helps some single-image workloads, but its static reservation
can reduce DC throughput and regress low-CPU or batch encoding. It is **not
integrated into the encoder**. Before default enablement, validate a policy
that preserves intended DC concurrency and shared-budget admission, address
profiling/storage bounds, and resolve the observed battery regressions.
The first policy can conservatively keep batches on the existing schedule;
this policy is proposed, not measured by this study.

## Retained artifacts

- [Clean review patch](prototype-clean.patch): scheduling changes without trace
  hooks. Application and syntax checks passed; production-build qualification
  of this clean patch remains outstanding.
- [Measured source patch](measured-source.patch): complete text changes from
  `d71eeb034f1f1d7d8b1aee2775ed9fe18bb24e98`, including the audited Xcode 27
  compatibility update, independent trace hooks, and prototype. Apply to a
  separate checkout of that base. Do not apply both patches.
- [Capture harness](capture.cpp), [contract harness](contract_test.cpp),
  [matrix](matrix.json), and [source/binary identities](manifest.json).
- [Paired summaries](analysis/summary.csv), selected stage aggregates, figures,
  and [validation](analysis/validation.json). Battery, AC and transition
  processes remain separate; traced calls do not support ordinary wall-time
  claims. All 160 Release tests and 84 contract cases passed.
- [Provenance](provenance.json): input/decoded-output hashes, original script
  hashes, binary identities, and the complete local archive manifest hash.

The complete working archive remains at `reports/eager-entropy-20260926/`
relative to the repository. It contains the source snapshot, isolated build,
all raw calls/events, process/power records, codestreams, original reproduction
scripts, and `artifact-manifest.json`. Those large machine-local artifacts are
not added to Git. This compact package supports review and source recovery;
replaying the measurements also requires the input corpus and that archive's
runner/configuration. The earlier boundary investigation is retained separately
at `~/Documents/Codex/gjxl-boundary-latency-20260925/`.

`analysis/audit.json` is the historical audit at study closure, before this
commit, rather than a claim that the current working tree has no changes.
`package-manifest.json` hashes the files in this committed package.
