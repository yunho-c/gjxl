#!/usr/bin/env python3
"""Validate completed qualification records and export a compact result ledger."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from compare import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()

    def read(name):
        return json.loads((evidence / name).read_text())

    qualification = read("qualified/summary.json")
    parity = read("final-parity/summary.json")
    resources = read("final-resources/summary.json")
    if (
        parity["cases"] != 56
        or len(parity["decodes"]) != 3
        or not resources["complete"]
        or len(resources["rows"]) != 8
        or len(resources["retained"]) != 24
    ):
        raise RuntimeError("Incomplete parity/resource gates")
    for row in parity["rows"]:
        for item in row["outputs"].values():
            if (
                sha(item["file"]) != item["sha256"]
                or sha(item["log"]) != item["log_sha256"]
            ):
                raise RuntimeError("Parity artifact changed")
    for label, tests in qualification["tests"].items():
        if (
            tests["count"] != 122
            or tests["failures"] != ["quantization_pipeline"]
            or sha(evidence / "qualified" / (label + "-tests.xml"))
            != tests["xml_sha256"]
        ):
            raise RuntimeError("Release gate changed")
    sanitizers = {}
    for name, count in (("asan-cpu", 11), ("asan-metal", 13)):
        cases = ET.parse(evidence / (name + ".xml")).getroot().findall(".//testcase")
        if len(cases) != count or any(
            c.find("failure") is not None or c.find("skipped") is not None
            for c in cases
        ):
            raise RuntimeError("Incomplete sanitizer gate")
        sanitizers[name] = {
            "passed": count,
            "xml_sha256": sha(evidence / (name + ".xml")),
        }
    metal = read("metal-validation-commands.json")
    if len(metal) != 5 or any(
        r["exit"] != 0 or sha(r["log"]) != r["log_sha256"] for r in metal
    ):
        raise RuntimeError("Metal validation changed")
    for row in metal:
        text = Path(row["log"]).read_text()
        if (
            "Metal API Validation Enabled" not in text
            or "Metal GPU Validation Enabled" not in text
        ):
            raise RuntimeError("Metal validation was not enabled")
    workflow = read("combined-wall/summary.json")
    stages = read("combined-stage/summary.json")
    for row in workflow:
        if len(row["metrics"]["total"]["paired_changes_percent"]) != 7:
            raise RuntimeError("Incomplete workflow cohort")
        row["metrics"] = {
            k: v
            for k, v in row["metrics"].items()
            if k
            in (
                "total",
                "quantization_pipeline",
                "codestream_encoding",
                "input_preparation",
            )
        }
    for row in stages:
        row["metrics"] = {
            k: v for k, v in row["metrics"].items() if k.startswith("group.")
        }
        if any(len(v["paired_changes_percent"]) != 3 for v in row["metrics"].values()):
            raise RuntimeError("Incomplete stage cohort")
    if len(workflow) != 5 or len(stages) != 2:
        raise RuntimeError("Missing final workload")
    setup = read("setup-comparison/summary.json")
    if len(setup) != 2 or any(r["pairs"] != 7 for r in setup):
        raise RuntimeError("Incomplete setup cohort")
    # Retain a complete local manifest; the compact tracked report anchors its
    # digest instead of embedding thousands of raw records and decoded PFMs.
    manifest = {
        str(p.relative_to(evidence)): sha(p)
        for p in sorted(evidence.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    save(evidence / "manifest.json", manifest)
    files = [
        "qualified/identity.json",
        "qualified/summary.json",
        "final-parity/identity.json",
        "final-parity/summary.json",
        "final-resources/identity.json",
        "final-resources/summary.json",
        "combined-wall/identity.json",
        "combined-wall/pairs.json",
        "combined-stage/identity.json",
        "combined-stage/pairs.json",
        "setup-comparison/identity.json",
        "setup-comparison/pairs.json",
        "asan-commands.json",
        "metal-validation-commands.json",
        "environment-final.json",
    ]
    save(
        args.output,
        {
            "baseline_revision": "1f16769bef87b42094fcd6a3f6dc50722662e20b",
            "scope": "Apple M4 Pro, Release, distance 1.2, effort 7, fully resident; same codestream bytes",
            "qualification": qualification,
            "parity": {
                "codestream_pairs": 56,
                "decoded_pairs": 3,
                "retained_changed_image_pairs": 24,
                "resource_scenarios": 8,
            },
            "sanitizers": {
                "results": sanitizers,
                "objcxx_instrumented": False,
                "leak_detection": False,
                "metal_ubsan_suppression": "tools/resident_qualification/metal-ubsan.supp",
            },
            "metal_api_and_shader_validation": {
                "coefficient_cases": 1344,
                "dct_pairs": 56,
                "epf_cases": 432,
            },
            "workflow": workflow,
            "gpu_stages": stages,
            "setup_and_footprint": setup,
            "artifacts": {name: sha(evidence / name) for name in files},
            "local_manifest": {
                "path": "build/dataflow/evidence/manifest.json",
                "sha256": sha(evidence / "manifest.json"),
                "files": len(manifest),
            },
        },
    )


if __name__ == "__main__":
    main()
