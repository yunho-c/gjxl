#!/usr/bin/env python3
"""Reuse the resident qualification driver with revision-matched libraries.

Verify changed-image output parity, scheduling/admission invariants, shutdown
and trimming. Footprint and setup observations are diagnostics, not paired
performance estimates. Preserve every command and retained codestream.
"""

import argparse
import json
from pathlib import Path
import subprocess

from compare import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-build",
        "candidate-build",
        "baseline-source",
        "candidate-source",
        "corpus",
        "output",
        "decoder",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    driver = root / "tools/resident_qualification/driver.cpp"
    builds = {
        "baseline": args.baseline_build.resolve(),
        "candidate": args.candidate_build.resolve(),
    }
    sources = {
        "baseline": args.baseline_source.resolve(),
        "candidate": args.candidate_source.resolve(),
    }
    corpus = args.corpus.resolve()
    natural = ["--input", str(corpus / "kodak-kodim17.pfm")]
    large = ["--synthetic", "3839x2159"]
    mixed = [
        "--synthetic",
        "17x9",
        *natural,
        "--synthetic",
        "1919x1079",
        *large,
        "--batch",
        "4",
        "--in-flight",
        "4",
    ]
    cohort = [*natural, "--batch", "4", "--in-flight", "4", "--callers", "2"]
    tasks = [
        ("small", ["--synthetic", "17x9"], False),
        ("4k", large, False),
        ("mixed", mixed, True),
        ("cohort", cohort, True),
        ("mixed-tight", [*mixed, "--limit", "tight"], False),
        ("mixed-full", [*mixed, "--limit", "full"], False),
        ("cohort-tight", [*cohort, "--limit", "tight", "--cpu-limit", "1"], False),
        ("cohort-full", [*cohort, "--limit", "full"], False),
    ]
    commands = []

    def run(name, command):
        command = list(map(str, command))
        log = out / (name + ".log")
        with log.open("x") as stream:
            process = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, timeout=900
            )
        commands.append(
            {
                "name": name,
                "command": command,
                "exit": process.returncode,
                "log": str(log),
                "log_sha256": sha(log),
            }
        )
        save(out / "commands.json", commands)
        if process.returncode:
            raise RuntimeError("Command failed: " + name)
        return log.read_text()

    if "e8ff0976" not in run("decoder-version", [args.decoder, "--version"]):
        raise RuntimeError("Unpinned decoder")
    identity = {
        str(driver): sha(driver),
        str(args.decoder): sha(args.decoder),
        str(corpus / "kodak-kodim17.pfm"): sha(corpus / "kodak-kodim17.pfm"),
    }
    libs = [
        "codestream",
        "pfm_io",
        "metal",
        "gpu_butteraugli",
        "gpu_ops",
        "codec",
        "gpu",
    ]
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    for label, build in builds.items():
        libraries = [build / ("libgjxl_" + name + ".a") for name in libs]
        identity.update({str(p): sha(p) for p in libraries})
        headers = [
            p
            for base in ("include", "src")
            for p in (sources[label] / base).rglob("*.h")
        ]
        identity.update({str(p): sha(p) for p in headers})
        command = [
            "/usr/bin/clang++",
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-DGJXL_FINAL_SCHEDULER",
            "-isysroot",
            sdk,
            "-I" + str(sources[label] / "include"),
            "-I" + str(sources[label] / "src"),
            driver,
            *libraries,
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-framework",
            "CoreGraphics",
            "-o",
            out / (label + "-driver"),
        ]
        run(label + "-compile", command)
        identity[str(out / (label + "-driver"))] = sha(out / (label + "-driver"))
    save(
        out / "identity.json", {"artifacts": identity, "protocol": sha(Path(__file__))}
    )
    rows, retained = [], []
    for name, options, retain in tasks:
        records = {}
        for label in builds:
            command = [out / (label + "-driver"), *options, "--count", "2"]
            if retain:
                directory = out / (name + "-" + label)
                directory.mkdir()
                command += ["--retain", directory]
            text = run(name + "-" + label, command)
            data = [
                json.loads(line) for line in text.splitlines() if line.startswith("{")
            ]
            if [item["type"] for item in data] != [
                "setup",
                "sample",
                "sample",
                "trim",
            ] or [item["index"] for item in data[1:-1]] != [0, 1]:
                raise RuntimeError("Incomplete driver samples")
            records[label] = data

        def signature(data):
            return [
                [
                    [
                        (i["source"], i["phase"], i["bytes"], i["fnv64"])
                        for i in call["images"]
                    ]
                    for call in sample["calls"]
                ]
                for sample in data[1:-1]
            ]

        if records["baseline"][0]["sources"] != records["candidate"][0][
            "sources"
        ] or signature(records["baseline"]) != signature(records["candidate"]):
            raise RuntimeError("Changed-image signatures differ: " + name)
        if retain:
            files = sorted((out / (name + "-baseline")).glob("*.jxl"))
            expected = (
                2 * records["baseline"][0]["batch"] * records["baseline"][0]["callers"]
            )
            if len(files) != expected:
                raise RuntimeError("Incomplete retained outputs")
            for file in files:
                pair = []
                for label in builds:
                    encoded = out / (name + "-" + label) / file.name
                    decoded = encoded.with_suffix(".decoded.pfm")
                    run(
                        name + "-" + label + "-" + file.stem + "-decode",
                        [args.decoder, "--quiet", "--num_threads=0", encoded, decoded],
                    )
                    pair.append(
                        {
                            "file": str(encoded),
                            "sha256": sha(encoded),
                            "decoded_sha256": sha(decoded),
                        }
                    )
                if (
                    pair[0]["sha256"] != pair[1]["sha256"]
                    or pair[0]["decoded_sha256"] != pair[1]["decoded_sha256"]
                ):
                    raise RuntimeError("Retained codestream or decoded pixels differ")
                retained.append(pair)
        rows.append({"name": name, "records": records})
        save(
            out / "summary.json",
            {"rows": rows, "retained": retained, "complete": False},
        )
        print(name + " passed", flush=True)
    if any(sha(p) != digest for p, digest in identity.items()):
        raise RuntimeError("Resource qualification input changed")
    save(out / "summary.json", {"rows": rows, "retained": retained, "complete": True})


if __name__ == "__main__":
    main()
