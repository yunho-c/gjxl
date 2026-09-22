# Tokenization checkpoint evidence

Implementation commit: `59d976152937a5a2e171c87be83254569273a3c9`. Measurements were collected before this commit; the final qualified source snapshot matches it byte for byte. [checkpoint.json](checkpoint.json) records the check and retained test results.

## Tracked evidence

The report, work log, plots, aggregate CSV/JSON data, all four final process-result collections and run configurations, binary hash manifests, input manifest, selected test logs, build configurations, and analysis/collection scripts are committed. [The report](REPORT.md) states timing boundaries and the known batch regression. [The input manifest](input-manifest.json) binds the six inputs to paths and hashes; input images are not redistributed.

## Local raw evidence

The [SHA-256 inventory](artifact-manifest.json) lists 7,223 local files totaling 2.46 GiB that are deliberately excluded from Git. They remain at `/Users/yunhocho/GitHub/gjxl-gpu-tokenization-20260922/reports/tokenization-20260922`. This includes frozen executables, source diffs and historical source copies, individual paired/stage captures, codestreams, VM/admission records, and exploratory screens. No remote archive has been created. A fresh Git checkout contains the summaries, not these raw captures or the input corpus. Report links to raw directories require the local archive.

The four qualified run roots are `final-profile`, `final-batch`, `group-final-profile`, and `group-final-batch`. Their tracked `results.json` files contain 360, 108, 90, and 54 process records respectively. Earlier screen and pilot directories are diagnostic evidence, not additional qualified comparisons.

## Reproduction boundaries

`plot.py` and `write_report.py` consume tracked summary files. `analyze.py`, `analyze_group.py`, and `decode.py` also require local raw captures and/or the external corpus configuration. Collection scripts are retained study utilities with machine-local paths; inspect and adapt those paths before a new collection. Saved compiler command JSON files show how the standalone capture and batch executables were linked, and CMake caches record the experiment-enabled build.

Use a fresh run directory for new measurements. Existing manifests bind runs to their original runner hashes and binaries; historical runner/source copies are in the local archive. For a new run after this checkpoint, also retain `git rev-parse HEAD` beside `source.diff`: the original runners captured a dirty diff plus then-untracked files, so a diff alone no longer identifies committed source. Keep the original frozen binaries when reproducing the V7 comparison; the current checkpoint includes the later optional group-fused kernel.

The code checkpoint preserves opt-in GPU tokenization. The proposed default-on rollout and automatic selection refinement are follow-up work, not claims that the existing cohort covers every production workload.
