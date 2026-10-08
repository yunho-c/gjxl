"""Run alternating paired captures and report the predeclared extraction gates."""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("photo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases", default="")
    args = parser.parse_args()
    if args.rounds < 7:
        parser.error("Qualification requires at least seven paired rounds")
    if args.repeats < 1:
        parser.error("At least one timed repetition is required")
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {name: getattr(args, name).resolve() for name in ("baseline", "candidate")}
    binary_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}
    samples = {}
    for round_index in range(-1, args.rounds):
        # First pair is warmup. Each capture also prepares/encodes the case once
        # before its timed ordinary execution; fixtures and writes are untimed.
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for version in order:
            directory = args.output / f"round-{round_index + 1}-{version}"
            argv = [str(binaries[version]), str(directory), str(args.repeats), str(args.photo.resolve())]
            if args.cases:
                argv.append(args.cases)
            with (args.output / f"round-{round_index + 1}-{version}.log").open("w") as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
            if round_index >= 0:
                with (directory / "timings.csv").open(newline="") as data:
                    for row in csv.DictReader(data):
                        key = (row["case"], row["stage"])
                        samples.setdefault(key, {"baseline": [], "candidate": []})[version].append(int(row["nanoseconds"]))
            print(f"Completed {version} {'warmup' if round_index < 0 else 'round ' + str(round_index + 1)}", flush=True)
    results = []
    for (case, stage), values in sorted(samples.items()):
        if any(len(v) != args.rounds * args.repeats for v in values.values()):
            raise RuntimeError(f"Incomplete samples for {case}/{stage}")
        medians = {name: statistics.median(v) for name, v in values.items()}
        delta = medians["candidate"] - medians["baseline"]
        relative = delta / medians["baseline"]
        relative_limit, absolute_limit = (0.03, 200_000) if stage == "workflow" else (0.05, 50_000)
        results.append({"case": case, "stage": stage, "nanoseconds": values,
                        "medians_ns": medians, "relative_change": relative,
                        "absolute_change_ns": delta,
                        "median_absolute_deviation_ns": {name: statistics.median(abs(v - medians[name]) for v in data)
                                                         for name, data in values.items()},
                        "flagged": delta > absolute_limit and relative > relative_limit})
    record = {"schema": 1, "rounds": args.rounds, "repeats_per_round": args.repeats, "warmup_pairs": 1,
              "binaries": {name: str(path) for name, path in binaries.items()},
              "binary_sha256": binary_hashes,
              "criteria": {"dc_serializer": {"relative": 0.05, "absolute_ns": 50_000},
                           "workflow": {"relative": 0.03, "absolute_ns": 200_000}},
              "results": results, "flagged": [r["case"] + "/" + r["stage"] for r in results if r["flagged"]]}
    (args.output / "timings.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"cases_and_stages": len(results), "flagged": record["flagged"]}), flush=True)
    return bool(record["flagged"])


if __name__ == "__main__":
    raise SystemExit(main())
