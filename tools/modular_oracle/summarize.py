"""Summarize repeated measurements; timing results never update oracle hashes."""
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    source, output = map(Path, sys.argv[1:])
    groups = defaultdict(list)
    for row in csv.DictReader(source.open()):
        key = tuple(row[k] for k in ("fixture", "encoder", "entropy", "search", "threads"))
        groups[key].append(row)
    result = []
    for key, rows in groups.items():
        # libjxl is repeated once per native entropy mode; retain all samples.
        warm = [r for r in rows if int(r["run"]) != 0]
        assert warm
        seconds = [float(r["seconds"]) for r in warm]
        median = statistics.median(seconds)
        assert len({r["bytes"] for r in rows}) == 1
        result.append(dict(zip(("fixture", "encoder", "entropy", "search", "threads"), key)) | {
            "warm_runs": len(warm), "first_calls_seconds": [float(r["seconds"]) for r in rows if r["run"] == "0"],
            "warm_seconds": seconds,
            "median_seconds": median, "min_seconds": min(seconds), "max_seconds": max(seconds),
            "mpixels_per_second": int(warm[0]["width"]) * int(warm[0]["height"]) / median / 1e6,
            "encoded_bytes": int(warm[0]["bytes"]),
            "managed_peak_bytes": max(int(r["managed_peak"]) for r in warm),
            "planned_capacity_bytes": int(warm[0]["planned_capacity"]),
            "encoded_candidates": int(warm[0].get("candidates", 0)),
            "stage_median_seconds": {stage: statistics.median(float(r[stage]) for r in warm)
                for stage in ("input", "transforms", "training", "tokens", "model", "emission", "assembly")}})
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for pattern in ("*.bin", "*.jxl") for p in source.parent.glob(pattern)}
    for name, digest in files.items():
        if name.endswith("-1.jxl") or name.endswith("-1-resolved.bin"):
            parallel = name.replace("-1.jxl", "-4.jxl").replace("-1-resolved.bin", "-4-resolved.bin")
            assert files[parallel] == digest, f"Serial/parallel mismatch: {name}"
    output.write_text(json.dumps({"schema": 1, "source": source.as_posix(),
        "artifact_sha256": files, "measurements": result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
