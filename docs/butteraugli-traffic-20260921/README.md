# Butteraugli traffic study: committed evidence

The study reached the user-approved practical stopping point. [REPORT.md](REPORT.md) gives the final results and limitations: directly measured warm ordinary encoding improves **12.697% at 24 MP/e7** and **10.715% at 48 MP/e10** on the tested M4 Pro, with identical tested codestreams. This establishes achievable headroom, not a theoretical hardware optimum.

## Source and revision boundary

| Role | Commit |
|---|---|
| Measured baseline, including production-aligned profiling | `4f3e4147c8bde16b0b49523adbf10ddd8043585f` |
| Main measured optimization, `short-bundle` | `db8a4d67d554bcb04b651243c3660c42ec245bb8` |
| Separate small refinement, `short-final` | `ef5f29003c52ec788d339f00b6f2e50a6f1818e7` |
| Primary `main` observed when preserving the study | `3b0de6d9d47846f6aa13f33ad6cca7e24e52ada6` |

Both source commits exactly preserve the independently tested snapshots, including generated kernel implementations. They are study checkpoints on `perf/butteraugli-traffic-20260921`, based on the measured revision. They have not been rebased onto or merged into the newer `main`. The new `main` is a descendant of the baseline; the two Butteraugli implementation files remain unchanged between those revisions, while the internal header has unrelated additions. This does not establish whole-encoder qualification on the new revision. Source cleanup, portability/default-policy review and integration are separate work.

The final refinement has direct-parent medians of -1.220% / -0.296% on the two flagship settings, with smaller and more variable broader effects. It is intentionally a separate commit so its incremental evidence is visible. Do not multiply results from different cohorts into a new original-baseline speedup.

## What Git preserves

- Final report, hypothesis audit, traffic/lifetime reasoning, historical phase notes and full saved-summary tables.
- All 23 cohorts from the final independent artifact audit: identities, protocols, paired metrics and compact per-call samples/commands/execution records. `case-records.json` preserves the original JSON text for hash verification without creating hundreds of separate case folders.
- Both candidates' exact-byte/decoder summaries, focused test logs, build commands, source patches and frozen binary/shader hashes.
- Earlier positive/negative screen summaries, variant inventories and the append-only experiment ledger.
- The [original hardware investigation](../performance-headroom-20260921/REPORT.md), compact Instruments results, plots, capture commands and raw-artifact hashes.

Raw image corpora, decoded pixels, executable binaries, shader libraries and large Instruments/GPU traces remain in the original external artifacts. They are not needed to recompute the committed ordinary medians and paired aggregates. Binary/input/output hashes identify them but are not substitutes for independently rerunning those correctness or timing measurements. Absolute paths in historical JSON, logs and scripts deliberately retain the original capture provenance. Local `.gitattributes` preserve the evidence bytes and original line endings; whitespace in patch context and the generated SVG is retained for their recorded hashes.

The external roots are `/Users/yunhocho/GitHub/gjxl/reports/butteraugli-traffic-20260921` and `/Users/yunhocho/GitHub/gjxl/reports/performance-headroom-20260921`. They were preserved, not moved or deleted. `archive-copy-provenance.json` records original source paths/hashes; final narrative status updates are reflected in `evidence-manifest.json`. `study.json` and older phase/identity files are historical records and may describe the then-uncommitted state. The current source mapping is [git-provenance.json](git-provenance.json).

## Verify without measurement

From any checkout that includes these commits and their ancestors:

```sh
python3 docs/butteraugli-traffic-20260921/verify_evidence.py
```

This standard-library-only verifier checks package hashes, committed source hashes, original per-call JSON hashes, sample medians, paired changes, summary aggregates, recorded codestream/decoded equality and relative document links. It does not run encoders or GPU jobs. The earlier `short-artifact-audit.json` records the larger 3,356-file audit against the original external artifacts; `audit_short_artifacts.py` is its historical script and requires those full artifacts. Scripts other than `verify_evidence.py` are archived protocols, not turnkey portable launchers: review and adapt their absolute input/build paths and create fresh run directories before use.

## Reproduce the measured boundary

Use separate clean checkouts and build directories for the baseline and candidate commits. The study used Release/Ninja, tests and benchmarks enabled, libjxl reference support disabled, eight CPU participants, and fully resident Metal. [Build commands](integrated/short-final/build-command.json), the [probe compile command](integrated/short-final/build-probe-command.json), [probe source](encode_probe.cpp), and each cohort's `protocol.py` preserve the actual boundary and options. The corresponding CMake options were:

```sh
cmake -S . -B build/traffic -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF \
  -DGJXL_ENABLE_METAL_PROFILING=OFF
```

That build-time Metal profiling option is distinct from the runtime stage-timing mode used by the recorded profiling cohorts. The compiler command, flags and original library paths are in the build/probe records. Restore the corpus files by their recorded SHA-256, compile a probe for each revision, and adapt the frozen protocol to fresh output locations. Preserve the declared ordinary/profiled separation and each cohort's parent. Use three warmups/seven samples across seven alternating independent-process pairs for confirmation; broader three-pair cohorts use two warmups/three samples. Loading and backend construction stay outside the timed public call; reference preparation and CPU serialization stay inside.

Correctness was independently gated for both candidates by seven focused Metal/resident tests under API/shader validation, 56 exact-byte comparisons and three identical finite decoded pairs. The restored final candidate re-passed the seven tests and reproduced its frozen shader byte for byte. Those validations are retained; committing byte-identical sources does not introduce a new measurement campaign. Existing full-suite/C API build failures remain disclosed in the study, and numerical tolerances were not widened.
