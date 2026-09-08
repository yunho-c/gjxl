#!/usr/bin/env python3
"""Publish compact evidence for the launch-bound/rectangular-AC/psycho study.

This study uses the 473ad7a baseline. Raw logs stay in the experiment directory;
the output preserves paired changes, fine attribution and artifact identities.
"""

import argparse
import json
from pathlib import Path
import statistics

from compare import extract, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    hashes = {}

    def read(relative):
        path = evidence / relative
        hashes[relative] = sha(path)
        return json.loads(path.read_text())

    commands = read("final-commands.json")
    for row in commands:
        if row["exit"] != 0 or sha(row["log"]) != row["sha256"]:
            raise RuntimeError("Failed or changed final command: " + row["name"])
    expected_commands = {"asan-tests", "final-parity", "final-resources", "final-setup",
                         "final-stage", "final-wall", "profile-overhead-stage",
                         "profile-overhead-wall"}
    if {r["name"] for r in commands} != expected_commands:
        raise RuntimeError("Incomplete final command inventory")
    qualification = read("qualification-final/summary.json")
    if (qualification["ac_candidate_cases"] != 320 or
            qualification["ac_candidate_validation_cases"] != 320 or
            any(t["count"] != 122 or t["failures"] != ["quantization_pipeline"]
                for t in qualification["tests"].values())):
        raise RuntimeError("Unexpected qualification results")
    wall_keys = {"total", "quantization_pipeline", "codestream_encoding",
                 "input_preparation", "codestream_ac_tokenization",
                 "codestream_entropy_optimization", "codestream_section_writing"}
    stage_keys = {"group.all_stages", "group.ac_strategy", "group.epf_linear"}
    trials = {}
    for name in ("bounds-tight", "bounds-loose", "rect", "final", "profile-overhead"):
        for mode in ("stage", "wall"):
            trial = name + "-" + mode
            identity = read(trial + "/identity.json")
            summary = read(trial + "/summary.json")
            pairs = read(trial + "/pairs.json")
            if len(pairs) != len(identity["jobs"]) * identity["pairs"]:
                raise RuntimeError("Incomplete paired trial: " + trial)
            keys = wall_keys if mode == "wall" else stage_keys
            trials[trial] = {
                "identity": identity,
                "summary": [{**row, "metrics": {k: v for k, v in row["metrics"].items()
                    if k in keys or (mode == "stage" and k.startswith(
                        ("frontend.ac_strategy.", "butteraugli.", "frontend.prepare_aq.reference")))}}
                    for row in summary],
            }
    # All phases below are independently timed stages. Accumulate each stage
    # across AQ iterations within a sample, then take process/paired medians.
    fine, combined = {}, {}
    identity = trials["final-stage"]["identity"]
    for name, _ in identity["jobs"]:
        processes, combined_processes = [], []
        for pair in range(identity["pairs"]):
            relative = f"final-stage/{name}-pair{pair}-candidate.raw.json"
            raw = read(relative)
            processes.append(extract(evidence / relative, True, identity["samples"]))
            samples = []
            for sample in raw["workloads"][0]["samples"]:
                totals = {}
                for submission in sample["submissions"]:
                    for stage in submission["stages"]:
                        if stage["stage_id"].startswith(
                                ("butteraugli.psycho.", "frontend.prepare_aq.reference.")):
                            phase = stage["stage_id"].rsplit(".", 1)[1]
                            totals[phase] = totals.get(phase, 0) + stage["gpu_nanoseconds"]
                samples.append(totals)
            combined_processes.append({k: statistics.median(s[k] for s in samples) / 1e6
                                       for k in samples[0]})
        keys = {k for k in processes[0] if k.startswith(
            ("butteraugli.psycho.", "frontend.prepare_aq.reference."))}
        fine[name] = {k: statistics.median(p[k] for p in processes) for k in sorted(keys)}
        combined[name] = {k: statistics.median(p[k] for p in combined_processes)
                          for k in combined_processes[0]}
    # Instrumentation overhead is observed in its own coarse-vs-fine trial,
    # including command-buffer spans and the resident AQ host operation span.
    overhead = []
    identity = trials["profile-overhead-stage"]["identity"]
    for name, _ in identity["jobs"]:
        for pair in range(identity["pairs"]):
            values = {}
            for side in ("baseline", "candidate"):
                raw = read(f"profile-overhead-stage/{name}-pair{pair}-{side}.raw.json")
                samples = raw["workloads"][0]["samples"]
                values[side] = {
                    "resident_command_ms": statistics.median(next(
                        sub["command_buffer_gpu_nanoseconds"] for sub in sample["submissions"]
                        if sub["submission_id"] == "resident.aq") / 1e6 for sample in samples),
                    "resident_operation_ms": statistics.median(next(
                        stage["wall_nanoseconds"] for stage in sample["wall_stages"]
                        if stage["stage_id"] == "resident.aq" and stage["kind"] == "operation")
                        / 1e6 for sample in samples),
                }
            overhead.append({"name": name, "pair": pair, "values": values})
    parity = read("final-parity/summary.json")
    resources = read("final-resources/summary.json")
    if (parity["cases"] != 56 or len(parity["decodes"]) != 3 or
            not resources["complete"] or len(resources["rows"]) != 8 or
            len(resources["retained"]) != 24):
        raise RuntimeError("Incomplete corpus/resource qualification")
    for row in parity["rows"]:
        if row["outputs"]["baseline"]["sha256"] != row["outputs"]["candidate"]["sha256"]:
            raise RuntimeError("Corpus bytes differ")
    output = {
        "baseline_revision": "473ad7a482305d091bad8e01420876139e4c9237",
        "baseline_source": read("baseline-identity.json"),
        "evidence_directory": str(evidence),
        "sources": read("final-source.json"),
        "qualification": qualification,
        "parity": {"cases": parity["cases"], "decoded_pairs": len(parity["decodes"]),
            "outputs": [{"name": row["name"],
                         "bytes": row["outputs"]["candidate"]["bytes"],
                         "sha256": row["outputs"]["candidate"]["sha256"]}
                        for row in parity["rows"]]},
        "resources": {"complete": resources["complete"],
            "retained_decoded_pairs": len(resources["retained"]),
            "scenarios": [{"name": row["name"], "observations": {
                side: next(x for x in records if x["type"] == "trim")
                for side, records in row["records"].items()}} for row in resources["rows"]]},
        "setup": read("final-setup/summary.json"),
        "trials": trials,
        "focused_small": {"identity": read("final-small-focused/identity.json"),
                          "summary": read("final-small-focused/summary.json")},
        "fine_psycho_ms": fine,
        "psycho_across_reference_and_feedback_ms": combined,
        "instrumented_coarse_vs_fine": overhead,
        "evidence_sha256": hashes,
        "report_protocol_sha256": sha(Path(__file__)),
    }
    save(args.output, output)


if __name__ == "__main__":
    main()
