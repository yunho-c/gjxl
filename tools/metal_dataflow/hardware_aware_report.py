#!/usr/bin/env python3
"""Verify the hardware-aware follow-up gates and export an evidence ledger."""
import argparse
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
from compare import extract, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pending-timing", action="store_true")
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    files = {}

    def read(name):
        path = evidence / name
        files[name] = sha(path)
        return json.loads(path.read_text())

    def verify_hashes(artifacts):
        for path, expected in artifacts.items():
            if sha(path) != expected:
                raise RuntimeError("Recorded artifact changed: " + path)

    if (evidence / "commands.json").exists():
        for command in read("commands.json"):
            if command["exit"] != 0 or sha(command["log"]) != command["sha256"]:
                raise RuntimeError("Final command failed or its log changed")
    qualification = read("final-qualification2/summary.json")
    for label, tests in qualification["tests"].items():
        if (tests["count"] != 122 or tests["failures"] != ["quantization_pipeline"] or
            sha(evidence / "final-qualification2" / (label + "-tests.xml")) != tests["xml_sha256"]):
            raise RuntimeError("Release gate changed")
    for key, count in {"coefficient_cases": 1344, "dct_cases": 56, "epf_cases": 432,
                       "hardware_aware_cases": 1578, "hardware_aware_validation_cases": 1578,
                       "conformance_per_revision": 22}.items():
        if qualification[key] != count:
            raise RuntimeError("Incomplete guarded/conformance gate: " + key)
    for command in read("final-qualification2/commands.json"):
        accepted = (0, 8) if command["name"] in ("baseline-tests", "candidate-tests") else (0,)
        if sha(command["log"]) != command["sha256"] or command["exit"] not in accepted:
            raise RuntimeError("Qualification command changed")
        if command["name"].endswith("-metal-validation"):
            text = Path(command["log"]).read_text()
            if "Metal API Validation Enabled" not in text or "Metal GPU Validation Enabled" not in text:
                raise RuntimeError("Metal validation was not enabled")
    details = read("final-qualification2/hardware-aware-metal-validation-details.json")
    detail_path = evidence / "final-qualification2/hardware-aware-metal-validation-details.log"
    if sha(detail_path) != details["sha256"]:
        raise RuntimeError("Detailed Metal validation output changed")
    detail_text = detail_path.read_text()
    if detail_text.count("Metal GPU Validation Enabled") < 5:
        raise RuntimeError("Missing detailed Metal validation output")
    parity = read("final-parity/summary.json")
    resources = read("final-resources/summary.json")
    if (parity["cases"] != 56 or len(parity["decodes"]) != 3 or not resources["complete"] or
        len(resources["rows"]) != 8 or len(resources["retained"]) != 24):
        raise RuntimeError("Incomplete corpus or retained-resource gate")
    for row in parity["rows"]:
        if (len(row["outputs"]) != 2 or
            len({item["sha256"] for item in row["outputs"].values()}) != 1):
            raise RuntimeError("Corpus codestream pair differs")
        for item in row["outputs"].values():
            if sha(item["file"]) != item["sha256"] or sha(item["log"]) != item["log_sha256"]:
                raise RuntimeError("Corpus artifact changed")
    for row in parity["decodes"]:
        if len(row["outputs"]) != 2 or len({item["sha256"] for item in row["outputs"]}) != 1:
            raise RuntimeError("Decoded corpus pair differs")
        for item in row["outputs"]:
            decoded = Path(item["file"])
            if sha(decoded) != item["sha256"] or sha(decoded.with_suffix(".log")) != item["log_sha256"]:
                raise RuntimeError("Decoded corpus artifact changed")
    for pair in resources["retained"]:
        if len(pair) != 2 or any(len({item[key] for item in pair}) != 1
                                 for key in ("sha256", "decoded_sha256")):
            raise RuntimeError("Retained pair differs")
        for item in pair:
            encoded = Path(item["file"])
            if sha(encoded) != item["sha256"] or sha(encoded.with_suffix(".decoded.pfm")) != item["decoded_sha256"]:
                raise RuntimeError("Retained artifact changed")
    for command in read("final-resources/commands.json"):
        if command["exit"] != 0 or sha(command["log"]) != command["log_sha256"]:
            raise RuntimeError("Resource command changed")
    asan_xml = evidence / "asan-tests.xml"
    cases = ET.parse(asan_xml).getroot().findall(".//testcase")
    if len(cases) != 7 or any(c.find("failure") is not None or c.find("skipped") is not None for c in cases):
        raise RuntimeError("Incomplete sanitizer gate")
    files["asan-tests.xml"] = sha(asan_xml)
    for command in read("asan-commands.json"):
        if command["exit"] != 0 or sha(command["log"]) != command["sha256"]:
            raise RuntimeError("Sanitizer command changed")
    verify_hashes(read("asan-identity.json"))
    verify_hashes(read("final-qualification2/identity.json")["artifacts"])
    verify_hashes(read("final-resources/identity.json")["artifacts"])
    parity_identity = read("final-parity/identity.json")
    for group in ("decoder", "encoders", "inputs"):
        verify_hashes(parity_identity[group])
    source_identity = read("final-source-identity.json")
    root = Path(__file__).resolve().parents[2]
    verify_hashes({str(root / name): digest for name, digest in source_identity["files"].items()})
    verify_hashes(source_identity["runtime"])
    if sha(evidence / "final-source-changes.tar.gz") != source_identity["source_archive_sha256"]:
        raise RuntimeError("Qualified source archive changed")
    isolated = None
    isolated_path = evidence / "epf-linear-final-timing.log"
    if isolated_path.exists() or not args.allow_pending_timing:
        files[isolated_path.name] = sha(isolated_path)
        samples = [json.loads(line) for line in isolated_path.read_text().splitlines()
                   if line.startswith("{")]
        by_key = {(r["pass"], r["variant"], r["width"], r["height"], r["pair"],
                   r["candidate"], r["sample"]): r["gpu_ns"] for r in samples}
        groups = [(p, variant, w, h) for p in (1, 2) for variant in ("direct", "tile32x4_p2")
                  for w, h in ((512, 512), (3840, 2160))]
        expected = {(*group, pair, side, sample) for group in groups for pair in range(5)
                    for side in (False, True) for sample in range(7)}
        if (len(samples) != len(expected) or by_key.keys() != expected or
            any(not math.isfinite(v) or v <= 0 for v in by_key.values())):
            raise RuntimeError("Incomplete isolated EPF timing")
        isolated = []
        for group in groups:
            medians = {side: [statistics.median(by_key[(*group, pair, side, sample)]
                                for sample in range(7)) / 1e6 for pair in range(5)]
                       for side in (False, True)}
            changes = [100 * (b / a - 1) for a, b in zip(medians[False], medians[True])]
            isolated.append(dict(pass_index=group[0], variant=group[1], width=group[2], height=group[3],
                baseline_ms=statistics.median(medians[False]), candidate_ms=statistics.median(medians[True]),
                paired_changes_percent=changes, median_change_percent=statistics.median(changes)))
    timing = {}
    for name, workloads, pairs, metric in [("combined-wall", 5, 7, "total"),
            ("combined-stage", 2, 3, "group.all_stages"),
            ("epf-integrated-stage-final", 3, 3, "group.epf_linear")]:
        path = evidence / name / "summary.json"
        if not path.exists() and args.allow_pending_timing:
            timing[name] = None
            continue
        rows = read(name + "/summary.json")
        if len(rows) != workloads or any(len(row["metrics"][metric]["paired_changes_percent"]) != pairs for row in rows):
            raise RuntimeError("Incomplete timing cohort: " + name)
        timing[name] = rows
        identity = read(name + "/identity.json")
        if (identity["pairs"] != pairs or identity["profile"] != (name != "combined-wall") or
            len(identity["jobs"]) != workloads):
            raise RuntimeError("Timing protocol changed: " + name)
        for artifacts in identity["runtime"].values():
            verify_hashes(artifacts)
        verify_hashes(identity["inputs"])
        pair_rows = read(name + "/pairs.json")
        expected = {(job[0], pair) for job in identity["jobs"] for pair in range(pairs)}
        if len(pair_rows) != len(expected) or {(r["name"], r["pair"]) for r in pair_rows} != expected:
            raise RuntimeError("Incomplete raw timing pairs: " + name)
        for pair in pair_rows:
            records = {}
            for side in ("baseline", "candidate"):
                stem = f'{name}/{pair["name"]}-pair{pair["pair"]}-{side}'
                record = read(stem + ".json")
                records[side] = record
                raw = evidence / (stem + ".raw.json")
                if sha(raw) != record["raw_sha256"] or sha(evidence / (stem + ".log")) != record["log_sha256"]:
                    raise RuntimeError("Timing raw output changed: " + stem)
                medians = extract(raw, identity["profile"], identity["samples"])
                if any(medians[k] != v for k, v in pair["process_medians_ms"][side].items()):
                    raise RuntimeError("Timing medians disagree with raw samples: " + stem)
            first, second = ("baseline", "candidate") if pair["pair"] % 2 == 0 else ("candidate", "baseline")
            if records[first]["end_unix"] > records[second]["start_unix"]:
                raise RuntimeError("Timing pair order changed")
        for row in rows:
            selected = [p for p in pair_rows if p["name"] == row["name"]]
            if len(selected) != pairs:
                raise RuntimeError("Timing summary workload changed")
            for key, values in row["metrics"].items():
                changes = [100 * (p["process_medians_ms"]["candidate"][key] /
                           p["process_medians_ms"]["baseline"][key] - 1) for p in selected]
                if (changes != values["paired_changes_percent"] or
                    statistics.median(changes) != values["median_change_percent"] or
                    any(statistics.median(p["process_medians_ms"][side][key] for p in selected) !=
                        values[side + "_ms"] for side in ("baseline", "candidate"))):
                    raise RuntimeError("Timing summary disagrees with pairs")
    setup_path = evidence / "setup-comparison/summary.json"
    setup = None
    if setup_path.exists() or not args.allow_pending_timing:
        setup = read("setup-comparison/summary.json")
        if len(setup) != 2 or any(row["pairs"] != 7 for row in setup):
            raise RuntimeError("Incomplete setup cohort")
        setup_identity = read("setup-comparison/identity.json")
        verify_hashes(setup_identity["drivers"])
        setup_pairs = read("setup-comparison/pairs.json")
        expected = {(name, pair) for name in ("small", "4k") for pair in range(7)}
        if len(setup_pairs) != 14 or {(p["name"], p["pair"]) for p in setup_pairs} != expected:
            raise RuntimeError("Incomplete raw setup pairs")
        for pair in setup_pairs:
            for side, values in pair["values"].items():
                log = evidence / f'setup-comparison/{pair["name"]}-{pair["pair"]}-{side}.log'
                if sha(log) != values["log_sha256"]:
                    raise RuntimeError("Setup log changed")
        for row in setup:
            selected = [p for p in setup_pairs if p["name"] == row["name"]]
            for key, values in row["metrics"].items():
                changes = [100 * (p["values"]["candidate"][key] / p["values"]["baseline"][key] - 1)
                           for p in selected]
                if (changes != values["paired_changes_percent"] or
                    any(statistics.median(p["values"][side][key] for p in selected) != values[side]
                        for side in ("baseline", "candidate"))):
                    raise RuntimeError("Setup summary disagrees with pairs")
    save(args.output, {"baseline_revision": "62d617541471087cb29302f8a5abf87c96cd36da",
        "status": "qualified" if isolated is not None and setup is not None and all(timing.values()) else "correctness-qualified-timing-pending",
        "qualification": qualification, "codestream_pairs": 56, "decoded_pairs": 3,
        "resource_scenarios": 8, "retained_decoded_pairs": 24, "sanitizer_tests": 7,
        "sanitizer_scope": "C++ ASan/UBSan, leaks disabled, existing narrow metal-cpp null-call suppression",
        "timing": timing, "isolated_epf": isolated, "setup": setup, "evidence_sha256": files})


if __name__ == "__main__":
    main()
