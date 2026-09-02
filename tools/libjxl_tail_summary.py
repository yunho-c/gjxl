#!/usr/bin/env python3
"""Validate and summarize retained libjxl-tail benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


NANOSECONDS_PER_MILLISECOND = 1_000_000.0
PROCESS_SUFFIX = re.compile(r"-process-\d+$")


def median(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise ValueError("cannot summarize an empty sample set")
    return float(statistics.median(collected))


def process_label(path: Path, workload: dict[str, Any]) -> str:
    if workload["name"] == "external_input":
        return PROCESS_SUFFIX.sub("", path.stem)
    return str(workload["name"])


def ns_to_ms(value: int | float) -> float:
    return float(value) / NANOSECONDS_PER_MILLISECOND


def metric(values: Sequence[float], digits: int = 2) -> str:
    center = median(values)
    low = min(values)
    high = max(values)
    return f"{center:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def scalar(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def discover(paths: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(candidate for candidate in path.rglob("*.json"))
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError(f"result path does not exist: {path}")
    if not files:
        raise ValueError("no JSON result files found")
    return sorted(files)


def required(document: dict[str, Any], key: str, path: Path) -> Any:
    if key not in document:
        raise ValueError(f"{path}: missing {key!r}")
    return document[key]


@dataclass(frozen=True)
class Record:
    path: Path
    document: dict[str, Any]
    workload: dict[str, Any]
    label: str


def load_records(paths: Sequence[Path]) -> list[Record]:
    records: list[Record] = []
    for path in discover(paths):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{path}: could not read benchmark JSON: {error}"
            ) from error
        if required(document, "schema_version", path) != 1:
            raise ValueError(f"{path}: unsupported schema version")
        if required(document, "timing", path) != "elapsed-wall-time":
            raise ValueError(f"{path}: result is not elapsed wall time")
        if required(document, "tail_boundary", path) != "warm-context":
            raise ValueError(f"{path}: result is not a warm-context tail run")
        scope = required(document, "scope", path)
        if scope not in {"codestream-tail", "hybrid-workflow"}:
            continue
        workloads = required(document, "workloads", path)
        if not isinstance(workloads, list) or not workloads:
            raise ValueError(f"{path}: no workload records")
        for workload in workloads:
            if workload.get("decoded_validation") != "exact-float-equal":
                raise ValueError(f"{path}: decoded validation is not exact")
            outputs = required(workload, "correctness_outputs", path)
            if set(outputs) != {"gjxl", "libjxl"}:
                raise ValueError(f"{path}: correctness outputs are incomplete")
            for backend in ("gjxl", "libjxl"):
                if (
                    not outputs[backend].get("sha256")
                    or outputs[backend].get("bytes", 0) <= 0
                ):
                    raise ValueError(f"{path}: invalid {backend} correctness output")
            records.append(
                Record(path, document, workload, process_label(path, workload))
            )
    if not records:
        raise ValueError("no supported libjxl-tail records found")
    return records


def group_key(record: Record) -> tuple[Any, ...]:
    document = record.document
    workload = record.workload
    return (
        document["scope"],
        record.label,
        workload.get("source_width"),
        workload.get("source_height"),
        workload.get("frame_fingerprint_sha256"),
        document["frontend"],
        round(float(document["distance"]), 5),
        document.get("frontend_effort", 7),
        document["libjxl_effort"],
        document.get("cpu_threads", "auto"),
        document["libjxl_threads"],
        document["gjxl_revision"],
        document["libjxl_base_revision"],
        document["libjxl_patch_revision"],
    )


def group_records(records: Sequence[Record]) -> list[list[Record]]:
    groups: dict[tuple[Any, ...], list[Record]] = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)
    return [
        groups[key] for key in sorted(groups, key=lambda item: tuple(map(str, item)))
    ]


def validate_group(group: Sequence[Record]) -> None:
    first = group[0]
    expected_outputs = first.workload["correctness_outputs"]
    expected_fingerprint = first.workload.get("frame_fingerprint_sha256")
    seen_paths: set[Path] = set()
    for record in group:
        if record.path in seen_paths:
            raise ValueError(f"duplicate process file: {record.path}")
        seen_paths.add(record.path)
        if record.workload["correctness_outputs"] != expected_outputs:
            raise ValueError(f"{record.path}: correctness hashes changed within group")
        if record.workload.get("frame_fingerprint_sha256") != expected_fingerprint:
            raise ValueError(f"{record.path}: frame fingerprint changed within group")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def tail_process(record: Record) -> dict[str, float]:
    samples = record.workload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{record.path}: no tail samples")
    by_backend: dict[str, list[dict[str, Any]]] = {"gjxl": [], "libjxl": []}
    by_index: dict[int, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        backend = sample.get("backend")
        if backend not in by_backend:
            raise ValueError(f"{record.path}: unknown tail backend {backend!r}")
        if (
            sample.get("codestream_sha256")
            != record.workload["correctness_outputs"][backend]["sha256"]
        ):
            raise ValueError(f"{record.path}: timed {backend} hash changed")
        if sample.get("frame_fingerprint_sha256") != record.workload.get(
            "frame_fingerprint_sha256"
        ):
            raise ValueError(f"{record.path}: timed frame fingerprint changed")
        by_backend[backend].append(sample)
        by_index.setdefault(int(sample["sample_index"]), {})[backend] = sample
    if not by_backend["gjxl"] or not by_backend["libjxl"]:
        raise ValueError(f"{record.path}: both tail backends are required")
    ratios = []
    for index, pair in by_index.items():
        if set(pair) != {"gjxl", "libjxl"}:
            raise ValueError(f"{record.path}: incomplete tail pair {index}")
        ratios.append(
            pair["gjxl"]["wall_nanoseconds"] / pair["libjxl"]["wall_nanoseconds"]
        )
    libjxl_phases = [sample["phase_nanoseconds"] for sample in by_backend["libjxl"]]
    return {
        "gjxl_ms": median(
            ns_to_ms(sample["wall_nanoseconds"]) for sample in by_backend["gjxl"]
        ),
        "libjxl_ms": median(
            ns_to_ms(sample["wall_nanoseconds"]) for sample in by_backend["libjxl"]
        ),
        "adapter_ms": median(
            ns_to_ms(phase["adapter_validation_and_copy"]) for phase in libjxl_phases
        ),
        "internal_ms": median(
            ns_to_ms(phase["libjxl_internal"]) for phase in libjxl_phases
        ),
        "speedup": median(ratios),
    }


def hybrid_process(record: Record) -> dict[str, float]:
    pairs = record.workload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{record.path}: no hybrid pairs")
    for pair in pairs:
        for backend in ("gjxl", "libjxl"):
            timed = pair[backend]
            if (
                timed.get("codestream_sha256")
                != record.workload["correctness_outputs"][backend]["sha256"]
            ):
                raise ValueError(f"{record.path}: timed {backend} hash changed")
    native = [pair["gjxl"] for pair in pairs]
    hybrid = [pair["libjxl"] for pair in pairs]
    native_workflow_phases = [sample["workflow_phase_nanoseconds"] for sample in native]
    hybrid_workflow_phases = [sample["workflow_phase_nanoseconds"] for sample in hybrid]
    hybrid_tail_phases = [sample["tail_phase_nanoseconds"] for sample in hybrid]
    return {
        "frontend_ms": median(
            ns_to_ms(
                phase["input_preparation"]
                + phase["backend_selection"]
                + phase["quantization_pipeline"]
            )
            for phase in native_workflow_phases
        ),
        "native_tail_ms": median(
            ns_to_ms(phase["codestream_encoding"]) for phase in native_workflow_phases
        ),
        "libjxl_tail_ms": median(
            ns_to_ms(phase["codestream_encoding"]) for phase in hybrid_workflow_phases
        ),
        "adapter_ms": median(
            ns_to_ms(phase["adapter_validation_and_copy"])
            for phase in hybrid_tail_phases
        ),
        "internal_ms": median(
            ns_to_ms(phase["libjxl_internal"]) for phase in hybrid_tail_phases
        ),
        "native_ms": median(ns_to_ms(sample["wall_nanoseconds"]) for sample in native),
        "hybrid_ms": median(ns_to_ms(sample["wall_nanoseconds"]) for sample in hybrid),
        "bound": median(float(pair["perfect_tail_bound"]) for pair in pairs),
        "speedup": median(float(pair["measured_outer_speedup"]) for pair in pairs),
    }


def revisions(groups: Sequence[Sequence[Record]]) -> str:
    triplets = {
        (
            group[0].document["gjxl_revision"],
            group[0].document["libjxl_base_revision"],
            group[0].document["libjxl_patch_revision"],
        )
        for group in groups
    }
    lines = ["## Revisions", ""]
    for gjxl_revision, base_revision, patch_revision in sorted(triplets):
        lines.append(
            f"- GJXL `{gjxl_revision}`, libjxl base `{base_revision}`, "
            f"patch `{patch_revision}`"
        )
    return "\n".join(lines)


def render(records: Sequence[Record]) -> str:
    groups = group_records(records)
    for group in groups:
        validate_group(group)
    hybrid_rows: list[list[str]] = []
    tail_rows: list[list[str]] = []
    for group in groups:
        first = group[0]
        document = first.document
        workload = first.workload
        outputs = workload["correctness_outputs"]
        size_delta = workload["size_delta"]
        common = [
            first.label,
            document["frontend"],
            scalar(float(document["distance"]), 2),
            f"{document.get('frontend_effort', 7)}/{document['libjxl_effort']}",
            f"{document.get('cpu_threads', 'auto')}/{document['libjxl_threads']}",
            str(len(group)),
        ]
        if document["scope"] == "hybrid-workflow":
            summaries = [hybrid_process(record) for record in group]
            hybrid_rows.append(
                common
                + [
                    metric([item["frontend_ms"] for item in summaries]),
                    metric([item["native_tail_ms"] for item in summaries]),
                    metric([item["libjxl_tail_ms"] for item in summaries]),
                    metric([item["native_ms"] for item in summaries]),
                    metric([item["hybrid_ms"] for item in summaries]),
                    metric([item["bound"] for item in summaries], 3),
                    metric([item["speedup"] for item in summaries], 3),
                    f"{size_delta['libjxl_minus_gjxl_bytes']:+d} ({size_delta['percent']:+.2f}%)",
                    "exact",
                ]
            )
        else:
            summaries = [tail_process(record) for record in group]
            tail_rows.append(
                common
                + [
                    metric([item["gjxl_ms"] for item in summaries]),
                    metric([item["libjxl_ms"] for item in summaries]),
                    metric([item["adapter_ms"] for item in summaries], 3),
                    metric([item["internal_ms"] for item in summaries]),
                    metric([item["speedup"] for item in summaries], 3),
                    f"{outputs['gjxl']['bytes']}/{outputs['libjxl']['bytes']}",
                    f"{size_delta['percent']:+.2f}%",
                    "exact",
                ]
            )

    sections = [
        "# libjxl-tail retained-result summary",
        "",
        (
            "Each timing cell is `median [minimum, maximum]` across independent "
            "process medians. Times are elapsed wall milliseconds. `F/L` is "
            "frontend/libjxl effort and `CPU/L` is the CPU/libjxl participant policy."
        ),
        "",
        revisions(groups),
    ]
    if hybrid_rows:
        sections.extend(
            [
                "",
                "## Complete workflow",
                "",
                markdown_table(
                    [
                        "Workload",
                        "Frontend",
                        "d",
                        "F/L",
                        "CPU/L",
                        "Proc",
                        "Frontend ms",
                        "Native tail ms",
                        "libjxl tail ms",
                        "Native total ms",
                        "Hybrid total ms",
                        "Zero-tail bound",
                        "Measured speedup",
                        "Size delta",
                        "Decoded",
                    ],
                    hybrid_rows,
                ),
            ]
        )
    if tail_rows:
        sections.extend(
            [
                "",
                "## Same-frame tail",
                "",
                markdown_table(
                    [
                        "Workload",
                        "Frontend",
                        "d",
                        "F/L",
                        "CPU/L",
                        "Proc",
                        "GJXL ms",
                        "libjxl ms",
                        "Adapter ms",
                        "libjxl internal ms",
                        "Tail speedup",
                        "Bytes G/L",
                        "Size delta",
                        "Decoded",
                    ],
                    tail_rows,
                ),
            ]
        )
    return "\n".join(sections) + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="raw JSON file or directory"
    )
    parser.add_argument("--output", type=Path, help="write Markdown to this path")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        result = render(load_records(arguments.paths))
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if arguments.output is None:
        print(result, end="")
    else:
        arguments.output.write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
