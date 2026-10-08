"""Hash deterministic captures or compare two capture directories."""
import argparse
import hashlib
import json
from pathlib import Path


def artifacts(directory):
    return {
        path.name: {"bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(directory.iterdir())
        if path.suffix in (".jxl", ".snapshot")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = artifacts(args.baseline)
    if not reference:
        parser.error("No deterministic capture artifacts")
    result = {"schema": 1, "baseline": str(args.baseline), "artifacts": reference}
    failures = []
    if args.candidate:
        candidate = artifacts(args.candidate)
        failures = [name for name in sorted(reference.keys() | candidate.keys())
                    if reference.get(name) != candidate.get(name)]
        result.update(candidate=str(args.candidate), compared=len(reference),
                      mismatches=failures, passed=not failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(reference)} deterministic artifacts; mismatches: {failures}")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
