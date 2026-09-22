#!/usr/bin/env python3
"""Compare frozen encoders on the canonical corpus and policy cases.

Kernel changes deliberately permit different shader hashes. Encoder, input,
command, output and optional pinned-decoder identities remain explicit.
"""

import argparse
import json
from pathlib import Path
import subprocess
from compare import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--decoder", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "tools/resident_qualification/corpus.json").read_text()
    )
    corpus = args.corpus.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = manifest["inputs"]
    names = {
        "kodak-kodim17.pfm",
        "imazen26-1029-planter-4k.pfm",
        "padded-stress-4k.pfm",
    }
    if args.screen:
        entries = [e for e in entries if e["canonical_path"] in names]
    default = [
        "--backend",
        "metal",
        "--metal-aq",
        "fully-resident",
        "--distance",
        "1.2",
        "--effort",
        "7",
    ]
    tasks = []
    for entry in entries:
        source = corpus / entry["canonical_path"]
        if sha(source) != entry["canonical_sha256"]:
            raise RuntimeError("Noncanonical input: " + str(source))
        tasks.append((source.stem, source, default))
    if not args.screen:
        source = corpus / "kodak-kodim17.pfm"
        tasks += [
            (f"effort-{e}", source, default[:-1] + [str(e)]) for e in range(1, 11)
        ]
        tasks += [
            ("high-density", source, default[:-2] + ["--high-density"]),
            ("maximum-compression", source, default + ["--maximum-compression"]),
            ("final-score", source, default + ["--collect-final-score"]),
        ]
        tasks += [
            (
                mode,
                source,
                [
                    "--backend",
                    "metal",
                    "--metal-aq",
                    mode,
                    "--distance",
                    "1.2",
                    "--effort",
                    "7",
                ],
            )
            for mode in ("exact-coefficients", "throughput", "maximum-throughput")
        ]
        tasks += [
            (
                "target",
                root / "testdata/codestream_sample.pfm",
                [
                    "--backend",
                    "metal",
                    "--target-bytes",
                    "280",
                    "--size-tolerance",
                    "0.1",
                    "--max-attempts",
                    "8",
                ],
            ),
            (
                "maximum-error",
                source,
                ["--backend", "cpu", "--maximum-error", "0.05", "0.05", "0.05"],
            ),
        ]
    builds = {
        "baseline": args.baseline_build.resolve(),
        "candidate": args.candidate_build.resolve(),
    }
    identity = {
        "encoders": {
            str(b / "gjxl_encode"): sha(b / "gjxl_encode") for b in builds.values()
        },
        "inputs": {str(p): sha(p) for _, p, _ in tasks},
        "tasks": [(n, str(p), a) for n, p, a in tasks],
        "protocol": sha(Path(__file__)),
        "manifest": sha(root / "tools/resident_qualification/corpus.json"),
        "decoder": {str(args.decoder.resolve()): sha(args.decoder)}
        if args.decoder
        else None,
    }
    identity = json.loads(json.dumps(identity))
    if (out / "identity.json").exists():
        if json.loads((out / "identity.json").read_text()) != identity:
            raise RuntimeError("Identity changed; use a new directory")
    else:
        save(out / "identity.json", identity)
    rows = []
    for name, source, options in tasks:
        if sha(source) != identity["inputs"][str(source)]:
            raise RuntimeError("Input changed during qualification")
        record = out / (name + ".json")
        if record.exists():
            row = json.loads(record.read_text())
            if row["name"] != name or set(row["outputs"]) != set(builds):
                raise RuntimeError("Recorded case inventory changed")
            for label, item in row["outputs"].items():
                expected_command = [
                    str(builds[label] / "gjxl_encode"),
                    *options,
                    str(source),
                    str(out / (name + "-" + label + ".jxl")),
                ]
                if item["command"] != expected_command:
                    raise RuntimeError("Recorded encode command changed")
                if (
                    sha(item["file"]) != item["sha256"]
                    or sha(item["log"]) != item["log_sha256"]
                ):
                    raise RuntimeError("Artifact changed")
        else:
            row = {"name": name, "outputs": {}}
            for label, build in builds.items():
                dest = out / (name + "-" + label + ".jxl")
                log = dest.with_suffix(".log")
                if dest.exists() or log.exists():
                    raise RuntimeError("Incomplete attempt retained: " + str(dest))
                executable = build / "gjxl_encode"
                if sha(executable) != identity["encoders"][str(executable)]:
                    raise RuntimeError("Encoder changed")
                command = [str(executable), *options, str(source), str(dest)]
                with log.open("w") as stream:
                    subprocess.run(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                row["outputs"][label] = {
                    "file": str(dest),
                    "sha256": sha(dest),
                    "bytes": dest.stat().st_size,
                    "command": command,
                    "log": str(log),
                    "log_sha256": sha(log),
                }
            if (
                row["outputs"]["baseline"]["sha256"]
                != row["outputs"]["candidate"]["sha256"]
            ):
                save(out / (name + "-mismatch.json"), row)
                raise RuntimeError("Codestream mismatch: " + name)
            save(record, row)
        if (
            row["outputs"]["baseline"]["sha256"]
            != row["outputs"]["candidate"]["sha256"]
        ):
            raise RuntimeError("Recorded codestream mismatch: " + name)
        rows.append(row)
        print(name + " bytes match", flush=True)
    decodes = []
    if args.decoder:
        version = subprocess.check_output(
            [str(args.decoder), "--version"], stderr=subprocess.STDOUT, text=True
        )
        if "e8ff0976" not in version:
            raise RuntimeError("Decoder is not pinned libjxl e8ff0976")
        for row in rows:
            if row["name"] + ".pfm" not in names:
                continue
            outputs = []
            for label, item in row["outputs"].items():
                dest = Path(item["file"]).with_suffix(".decoded.pfm")
                log = dest.with_suffix(".log")
                command = [
                    str(args.decoder),
                    "--quiet",
                    "--num_threads=0",
                    item["file"],
                    str(dest),
                ]
                sidecar = dest.with_suffix(".json")
                if (
                    sha(args.decoder)
                    != identity["decoder"][str(args.decoder.resolve())]
                ):
                    raise RuntimeError("Decoder changed")
                if sidecar.exists():
                    decoded = json.loads(sidecar.read_text())
                    if (
                        decoded["command"] != command
                        or decoded["sha256"] != sha(dest)
                        or decoded["log_sha256"] != sha(log)
                    ):
                        raise RuntimeError("Decoded artifact changed")
                else:
                    if dest.exists() or log.exists():
                        raise RuntimeError("Incomplete decode retained: " + str(dest))
                    with log.open("w") as stream:
                        subprocess.run(
                            command,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=600,
                        )
                    decoded = {
                        "label": label,
                        "file": str(dest),
                        "sha256": sha(dest),
                        "command": command,
                        "log_sha256": sha(log),
                    }
                    save(sidecar, decoded)
                outputs.append(decoded)
            if outputs[0]["sha256"] != outputs[1]["sha256"]:
                raise RuntimeError("Decoded pixels differ")
            decodes.append({"name": row["name"], "outputs": outputs})
    save(out / "summary.json", {"cases": len(rows), "rows": rows, "decodes": decodes})


if __name__ == "__main__":
    main()
