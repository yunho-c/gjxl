#!/usr/bin/env python3
"""Fresh serial Release suites, guarded kernel probes and pinned conformance.

Use a new output directory for every attempt. Failures and partial logs are
retained. Corpus/policy and retained-resource gates are separate commands.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from compare import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Revalidate and reuse completed commands with identical runtime artifacts",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    builds = {
        "baseline": args.baseline_build.resolve(),
        "candidate": args.candidate_build.resolve(),
    }
    artifacts = [args.decoder, args.info]
    for build in builds.values():
        artifacts += [
            p for p in build.glob("gjxl_*") if p.is_file() and os.access(p, os.X_OK)
        ]
        artifacts.append(build / "metal/gjxl.metallib")
    identity = {str(p): sha(p) for p in artifacts}
    identity_file = out / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text())["artifacts"] != identity:
            raise RuntimeError("Qualification runtime changed")
        save(
            out / ("resume-protocol-" + str(time.time_ns()) + ".json"),
            {"protocol": sha(Path(__file__)), "identity_sha256": sha(identity_file)},
        )
    else:
        save(identity_file, {"artifacts": identity, "protocol": sha(Path(__file__))})
    commands_file = out / "commands.json"
    commands = json.loads(commands_file.read_text()) if commands_file.exists() else []

    def run(name, command, accepted=(0,), environment=None):
        command = list(map(str, command))
        log = out / (name + ".log")
        previous = next((row for row in commands if row["name"] == name), None)
        if previous is not None:
            if (
                previous["command"] != command
                or previous["environment"] != environment
                or previous["exit"] not in accepted
                or previous["sha256"] != sha(log)
            ):
                raise RuntimeError("Changed recorded command: " + name)
            return log.read_text()
        started = time.time()
        with log.open("x") as stream:
            process = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, **(environment or {})},
                timeout=3600,
            )
        commands.append(
            {
                "name": name,
                "command": command,
                "environment": environment,
                "start_unix": started,
                "end_unix": time.time(),
                "exit": process.returncode,
                "log": str(log),
                "sha256": sha(log),
            }
        )
        save(out / "commands.json", commands)
        if process.returncode not in accepted:
            raise RuntimeError("Command failed: " + name)
        print(name + " finished", flush=True)
        return log.read_text()

    version = run("decoder-version", [args.decoder, "--version"])
    if "e8ff0976" not in version:
        raise RuntimeError("Decoder is not pinned to e8ff0976")
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    tests = {}
    for label, build in builds.items():
        inventory = json.loads(
            run(
                label + "-inventory",
                ["ctest", "--test-dir", build, "--show-only=json-v1"],
            )
        )
        expected = sorted(t["name"] for t in inventory["tests"])
        xml = out / (label + "-tests.xml")
        run(
            label + "-tests",
            [
                "ctest",
                "--test-dir",
                build,
                "--output-on-failure",
                "--parallel",
                "1",
                "--output-junit",
                xml,
            ],
            accepted=(0, 8),
            environment={"SDKROOT": sdk},
        )
        cases = ET.parse(xml).getroot().findall(".//testcase")
        if (
            not expected
            or sorted(c.get("name") for c in cases) != expected
            or any(c.find("skipped") is not None for c in cases)
        ):
            raise RuntimeError("Incomplete or skipped CTest inventory")
        failed = [c for c in cases if c.find("failure") is not None]
        if failed and not (
            len(failed) == 1
            and failed[0].get("name") == "quantization_pipeline"
            and "actual=0.24919039011001587 expected=0.24914586544036865"
            in "".join(failed[0].itertext())
        ):
            raise RuntimeError("New Release test failure")
        tests[label] = {
            "count": len(cases),
            "failures": [c.get("name") for c in failed],
            "xml_sha256": sha(xml),
        }
    if tests["candidate"]["failures"] != tests["baseline"]["failures"]:
        raise RuntimeError("Baseline does not reproduce candidate failure")
    libraries = [b / "metal/gjxl.metallib" for b in builds.values()]
    candidate = builds["candidate"]
    for mode in ("full", "scored", "final"):
        result = run(
            "coefficients-" + mode,
            [candidate / "gjxl_metal_coefficient_probe", *libraries, "--mode", mode],
        )
        if json.loads(result)["bitwise_cases"] != 448:
            raise RuntimeError("Incomplete coefficient probe")
    for name, count in (("dct_image", 56), ("epf", 432)):
        result = run(name, [candidate / ("gjxl_metal_" + name + "_probe"), *libraries])
        if (
            name == "dct_image"
            and "56 forward/inverse image pairs agree bitwise" not in result
        ) or (name == "epf" and json.loads(result)["bitwise_cases"] != count):
            raise RuntimeError("Incomplete guarded probe")
    for label, build in builds.items():
        result = run(
            label + "-conformance",
            [
                build / "gjxl_codestream_conformance_test",
                "--decoder",
                args.decoder,
                "--info",
                args.info,
                "--encoder",
                build / "gjxl_encode",
                "--sample",
                root / "testdata/codestream_sample.pfm",
                "--artifacts",
                out / (label + "-conformance"),
            ],
        )
        if "All 22 pinned codestream conformance fixtures passed." not in result:
            raise RuntimeError("Incomplete conformance fixtures")
    if any(sha(p) != digest for p, digest in identity.items()):
        raise RuntimeError("Qualification runtime changed")
    save(
        out / "summary.json",
        {
            "tests": tests,
            "coefficient_cases": 1344,
            "dct_cases": 56,
            "epf_cases": 432,
            "conformance_per_revision": 22,
        },
    )


if __name__ == "__main__":
    main()
