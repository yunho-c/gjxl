#!/usr/bin/env python3
"""Alternate process-cold backend setup and first public-call diagnostics.

The driver separates backend setup from the first public call. These intervals
exclude input creation/driver bookkeeping and do not measure executable launch.
Metal driver/disk caches are not reset. Run after resource qualification.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from compare import quiet, save, sha

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--drivers", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pairs", type=int, default=7)
args = parser.parse_args()
if args.pairs < 1:
    parser.error("Positive pairs required")
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
drivers = {
    k: (args.drivers / (k + "-driver")).resolve() for k in ["baseline", "candidate"]
}
save(
    out / "identity.json",
    {
        "drivers": {str(p): sha(p) for p in drivers.values()},
        "protocol": sha(Path(__file__)),
        "pairs": args.pairs,
    },
)
rows = []
for name, extent in [("small", "17x9"), ("4k", "3839x2159")]:
    for pair in range(args.pairs):
        values = {}
        for side in (
            ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        ):
            quiet()
            cmd = [str(drivers[side]), "--synthetic", extent, "--count", "1"]
            log = out / f"{name}-{pair}-{side}.log"
            with log.open("x") as f:
                subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=600
                )
            data = [
                json.loads(line)
                for line in log.read_text().splitlines()
                if line.startswith("{")
            ]
            if [x["type"] for x in data] != ["setup", "sample", "trim"]:
                raise RuntimeError("Incomplete setup sample")
            values[side] = {
                "setup_ms": data[0]["setup_ns"] / 1e6,
                "first_call_ms": data[1]["calls"][0]["wall_ns"] / 1e6,
                "command": cmd,
                "log_sha256": sha(log),
                "peak_footprint": data[2]["peak_footprint"],
                "post_trim_footprint": data[2]["post_trim_footprint"],
                "image": data[1]["calls"][0]["images"][0],
            }
        if (
            values["baseline"]["image"]["fnv64"]
            != values["candidate"]["image"]["fnv64"]
        ):
            raise RuntimeError("First call bytes differ")
        rows.append({"name": name, "pair": pair, "values": values})
        save(out / "pairs.json", rows)
    summary = []
    for name in {r["name"] for r in rows}:
        chosen = [r for r in rows if r["name"] == name]
        summary.append(
            {
                "name": name,
                "pairs": len(chosen),
                "metrics": {
                    k: {
                        "baseline": statistics.median(
                            r["values"]["baseline"][k] for r in chosen
                        ),
                        "candidate": statistics.median(
                            r["values"]["candidate"][k] for r in chosen
                        ),
                        "paired_changes_percent": [
                            100
                            * (
                                r["values"]["candidate"][k] / r["values"]["baseline"][k]
                                - 1
                            )
                            for r in chosen
                        ],
                    }
                    for k in [
                        "setup_ms",
                        "first_call_ms",
                        "peak_footprint",
                        "post_trim_footprint",
                    ]
                },
            }
        )
    save(out / "summary.json", summary)
    print(name + " setup comparisons done", flush=True)
