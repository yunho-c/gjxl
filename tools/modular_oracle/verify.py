"""Replay the frozen CPU oracle. This command never updates expected hashes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def artifacts(root):
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "timings.csv"}


def main():
    executable, corpus, output = map(Path, sys.argv[1:])
    expected = json.loads((corpus / "oracle-v1.json").read_text())
    photo = corpus / "photo.ppm"
    assert hashlib.sha256(photo.read_bytes()).hexdigest() == expected["photo_sha256"]
    subprocess.run([str(executable), "freeze", str(photo), str(output)], check=True)
    actual = artifacts(output)
    mismatches = [name for name in sorted(actual.keys() | expected["artifacts"].keys())
                  if actual.get(name) != expected["artifacts"].get(name)]
    if mismatches:
        raise SystemExit("Frozen oracle mismatch (evidence retained):\n" + "\n".join(mismatches))
    print(f"{len(actual)} frozen stage/file artifacts match")


if __name__ == "__main__":
    main()
