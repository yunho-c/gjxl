# Butteraugli production qualification

[REPORT.md](REPORT.md) describes the prepared M4 Pro implementation, fresh current-main timings, exact-output validation, fallback coverage and the resolved stale profiling test.

Verify the compact saved evidence without running measurements:

```sh
python3 docs/butteraugli-production-qualification/verify_evidence.py
```

See [git-provenance.json](git-provenance.json) for the exact source commits and [hardware.json](hardware.json) for the measured device/toolchain. Large original artifacts remain outside Git; archived protocols retain their original absolute paths.
