#!/usr/bin/env python3
"""Run a matched-quality stock-libjxl versus GJXL-hybrid comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable, Sequence


SCHEMA_VERSION = 1
LINEAR_SRGB_HINT = "RGB_D65_SRG_Rel_Lin"


class ComparisonError(RuntimeError):
    """An expected comparison setup or validation failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".tmp-",
        delete=False,
    ) as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        temporary = Path(output.name)
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"JSON root is not an object: {path}")
    return value


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not normalized or normalized in {".", ".."}:
        raise ComparisonError(f"Unsafe input name: {value!r}")
    return normalized


def inspect_pfm(path: Path) -> tuple[int, int]:
    def content_line(stream: Any) -> bytes:
        while True:
            line = stream.readline()
            if not line:
                raise ComparisonError(f"Incomplete PFM header: {path}")
            line = line.rstrip(b"\r\n")
            if line and not line.startswith(b"#"):
                return line

    try:
        with path.open("rb") as stream:
            if content_line(stream) != b"PF":
                raise ComparisonError(f"Input is not an RGB PFM: {path}")
            dimensions = content_line(stream).split()
            if len(dimensions) != 2:
                raise ComparisonError(f"Invalid PFM dimensions: {path}")
            width, height = (int(item) for item in dimensions)
            scale = float(content_line(stream))
            header_bytes = stream.tell()
        expected_bytes = header_bytes + width * height * 3 * 4
        if (
            width <= 0
            or height <= 0
            or not math.isfinite(scale)
            or scale == 0.0
            or path.stat().st_size != expected_bytes
        ):
            raise ComparisonError(f"Invalid PFM payload: {path}")
    except (OSError, ValueError) as error:
        raise ComparisonError(f"Could not inspect PFM {path}: {error}") from error
    return width, height


def parse_inputs(values: Sequence[str]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ComparisonError("Each --input must have the form NAME=IMAGE.pfm")
        label, raw_path = value.split("=", 1)
        name = safe_name(label)
        if name in names:
            raise ComparisonError(f"Duplicate input name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ComparisonError(f"Input does not exist: {path}")
        width, height = inspect_pfm(path)
        inputs.append(
            {
                "name": name,
                "path": str(path),
                "width": width,
                "height": height,
                "sha256": sha256_file(path),
            }
        )
        names.add(name)
    return inputs


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_manifest() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    libjxl = repository / "third_party/libjxl"
    libjxl_diff = command_output(["git", "-C", str(libjxl), "diff", "--binary"])
    return {
        "repository": str(repository),
        "revision": command_output(["git", "-C", str(repository), "rev-parse", "HEAD"]),
        "status": command_output(
            ["git", "-C", str(repository), "status", "--porcelain"]
        ),
        "libjxl_revision": command_output(
            ["git", "-C", str(libjxl), "rev-parse", "HEAD"]
        ),
        "libjxl_status": command_output(
            ["git", "-C", str(libjxl), "status", "--porcelain"]
        ),
        "libjxl_diff_sha256": (
            hashlib.sha256(libjxl_diff.encode("utf-8")).hexdigest()
            if libjxl_diff is not None
            else None
        ),
    }


def host_manifest() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "macos": command_output(["sw_vers"]),
        "hardware": command_output(["system_profiler", "SPHardwareDataType"]),
        "power": command_output(["pmset", "-g", "batt"]),
        "thermal": command_output(["pmset", "-g", "therm"]),
        "load": command_output(["uptime"]),
    }


def require_executable(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ComparisonError(f"{label} is not executable: {resolved}")
    return resolved


def run_logged(
    command: list[str], name: str, directory: Path, manifest: dict[str, Any]
) -> None:
    begin = time.monotonic_ns()
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        raise ComparisonError(f"Could not execute {command[0]}: {error}") from error
    record = {
        "name": name,
        "argv": command,
        "elapsed_nanoseconds": time.monotonic_ns() - begin,
        "exit_code": result.returncode,
        "stdout": f"{name}.stdout.txt",
        "stderr": f"{name}.stderr.txt",
    }
    (directory / record["stdout"]).write_text(result.stdout, encoding="utf-8")
    (directory / record["stderr"]).write_text(result.stderr, encoding="utf-8")
    manifest["processes"].append(record)
    write_json_atomic(directory / "manifest.json", manifest)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ComparisonError(
            f"Process {name} failed with status {result.returncode}"
            + (f": {detail}" if detail else "")
        )


def hybrid_command(
    args: argparse.Namespace,
    image: Path,
    raw: Path,
    artifacts: Path,
    *,
    warmups: int,
    samples: int,
) -> list[str]:
    return [
        str(args.gjxl_benchmark),
        "--scope",
        "hybrid-workflow",
        "--input",
        str(image),
        "--tail-frontend",
        "metal",
        "--gpu-aq",
        "fully-resident",
        "--implementation",
        "factored",
        "--cpu-threads",
        str(args.threads),
        "--libjxl-threads",
        str(args.threads),
        "--frontend-effort",
        str(args.frontend_effort),
        "--libjxl-effort",
        str(args.libjxl_effort),
        "--distance",
        str(args.gjxl_distance),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--raw-samples",
        str(raw),
        "--artifacts",
        str(artifacts),
    ]


def stock_command(
    args: argparse.Namespace,
    image: Path,
    raw: Path,
    output: Path,
    distance: float,
    *,
    warmups: int,
    samples: int,
) -> list[str]:
    return [
        str(args.stock_benchmark),
        "--input",
        str(image),
        "--raw-samples",
        str(raw),
        "--output",
        str(output),
        "--distance",
        format(distance, ".12g"),
        "--effort",
        str(args.libjxl_effort),
        "--num-threads",
        str(args.threads),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
    ]


def score_codestream(
    args: argparse.Namespace,
    input_path: Path,
    codestream: Path,
    name: str,
    directory: Path,
    manifest: dict[str, Any],
) -> float:
    with tempfile.TemporaryDirectory(prefix="gjxl-stock-hybrid-") as temporary:
        decoded = Path(temporary) / "decoded.pfm"
        run_logged(
            [str(args.djxl), str(codestream), str(decoded), "--quiet"],
            f"{name}-decode",
            directory,
            manifest,
        )
        width, height = inspect_pfm(decoded)
        if (width, height) != inspect_pfm(input_path):
            raise ComparisonError(f"Decoded dimensions changed for {codestream}")
        run_logged(
            [
                str(args.butteraugli),
                str(input_path),
                str(decoded),
                "--colorspace",
                LINEAR_SRGB_HINT,
            ],
            f"{name}-butteraugli",
            directory,
            manifest,
        )
    stdout_path = directory / f"{name}-butteraugli.stdout.txt"
    try:
        value = float(stdout_path.read_text(encoding="utf-8").splitlines()[0])
    except (OSError, ValueError, IndexError) as error:
        raise ComparisonError(
            f"Could not parse Butteraugli score: {stdout_path}"
        ) from error
    if not math.isfinite(value) or value < 0.0:
        raise ComparisonError(f"Invalid Butteraugli score: {value}")
    return value


def select_bracket(
    evaluations: Sequence[dict[str, Any]], target: float
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ordered = sorted(evaluations, key=lambda item: item["distance"])
    candidates = []
    for lower, upper in zip(ordered, ordered[1:]):
        if (lower["butteraugli"] - target) * (upper["butteraugli"] - target) <= 0:
            candidates.append((lower, upper))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda pair: (
            pair[1]["distance"] - pair[0]["distance"],
            pair[0]["absolute_error"] + pair[1]["absolute_error"],
        ),
    )


def calibrate_distance(
    target: float,
    evaluate: Callable[[float], dict[str, Any]],
    *,
    initial: float,
    minimum: float,
    maximum: float,
    tolerance: float,
    maximum_evaluations: int,
    maximum_relative_error: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0.0 < minimum < maximum or not minimum <= initial <= maximum:
        raise ComparisonError("Invalid calibration bounds")
    if maximum_evaluations < 3:
        raise ComparisonError("Calibration requires at least three evaluations")
    evaluations: list[dict[str, Any]] = []
    by_distance: dict[float, dict[str, Any]] = {}

    def evaluate_once(distance: float) -> dict[str, Any]:
        distance = float(distance)
        if distance not in by_distance:
            result = dict(evaluate(distance))
            score = float(result["butteraugli"])
            result.update(
                distance=distance,
                absolute_error=abs(score - target),
                relative_error=abs(score - target) / target,
            )
            evaluations.append(result)
            by_distance[distance] = result
        return by_distance[distance]

    def acceptable(result: dict[str, Any]) -> bool:
        return result["absolute_error"] <= tolerance

    first = evaluate_once(initial)
    if acceptable(first):
        first["match_kind"] = "absolute-tolerance"
        return first, evaluations
    endpoint = evaluate_once(minimum if first["butteraugli"] > target else maximum)
    if acceptable(endpoint):
        endpoint["match_kind"] = "absolute-tolerance"
        return endpoint, evaluations

    bracket = select_bracket(evaluations, target)
    if bracket is None:
        grid = [minimum + index * (maximum - minimum) / 8.0 for index in range(9)]
        grid.sort(key=lambda distance: abs(distance - initial))
        for distance in grid:
            if len(evaluations) >= maximum_evaluations:
                break
            candidate = evaluate_once(distance)
            if acceptable(candidate):
                candidate["match_kind"] = "absolute-tolerance"
                return candidate, evaluations
            bracket = select_bracket(evaluations, target)
            if bracket is not None:
                break

    while bracket is not None and len(evaluations) < maximum_evaluations:
        lower, upper = bracket
        midpoint = (lower["distance"] + upper["distance"]) / 2.0
        if midpoint in by_distance:
            break
        candidate = evaluate_once(midpoint)
        if acceptable(candidate):
            candidate["match_kind"] = "absolute-tolerance"
            return candidate, evaluations
        bracket = select_bracket(evaluations, target)

    best = min(evaluations, key=lambda item: item["absolute_error"])
    if best["relative_error"] <= maximum_relative_error:
        best["match_kind"] = "relative-tolerance"
        return best, evaluations
    raise ComparisonError(
        "Could not match decoded quality: "
        f"target={target:.9g}, best={best['butteraugli']:.9g}, "
        f"absolute_error={best['absolute_error']:.9g}"
    )


def median(values: Sequence[float]) -> float:
    if not values:
        raise ComparisonError("Cannot summarize an empty sample list")
    return float(statistics.median(values))


def numeric_match(value: Any, expected: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and math.isclose(
        numeric, expected, rel_tol=1e-6, abs_tol=1e-6
    )


def summarize_workflow_backend(
    pairs: Sequence[dict[str, Any]],
    outputs: dict[str, Any],
    backend: str,
    path: Path,
) -> dict[str, Any]:
    try:
        samples = [pair[backend] for pair in pairs]
        output = outputs[backend]
        phases = [item["workflow_phase_nanoseconds"] for item in samples]
    except (KeyError, TypeError) as error:
        raise ComparisonError(
            f"Hybrid result is missing the {backend} backend: {path}"
        ) from error
    if any(
        sample.get("backend") != backend
        or sample.get("encoded_bytes") != output.get("bytes")
        or sample.get("codestream_sha256") != output.get("sha256")
        for sample in samples
    ):
        raise ComparisonError(f"Hybrid {backend} output changed across samples: {path}")
    return {
        "median_nanoseconds": median([item["wall_nanoseconds"] for item in samples]),
        "sample_nanoseconds": [item["wall_nanoseconds"] for item in samples],
        "input_preparation_median_nanoseconds": median(
            [item["input_preparation"] for item in phases]
        ),
        "quantization_pipeline_median_nanoseconds": median(
            [item["quantization_pipeline"] for item in phases]
        ),
        "tail_median_nanoseconds": median(
            [item["codestream_encoding"] for item in phases]
        ),
        "output_bytes": output["bytes"],
        "output_sha256": output["sha256"],
    }


def parse_hybrid(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    document = load_json(path)
    if (
        document.get("schema_version") != 1
        or document.get("scope") != "hybrid-workflow"
        or document.get("timing") != "elapsed-wall-time"
        or document.get("tail_boundary") != "warm-context"
        or document.get("frontend") != "metal"
        or document.get("frontend_effort") != args.frontend_effort
        or document.get("libjxl_effort") != args.libjxl_effort
        or document.get("gpu_aq") != "fully-resident"
        or document.get("cpu_threads") != args.threads
        or document.get("libjxl_threads") != args.threads
        or document.get("warmups") != args.warmups
        or document.get("paired_sample_count") != args.samples
        or not numeric_match(document.get("distance"), args.gjxl_distance)
    ):
        raise ComparisonError(f"Unexpected hybrid benchmark schema: {path}")
    workloads = document.get("workloads")
    if not isinstance(workloads, list) or len(workloads) != 1:
        raise ComparisonError(f"Hybrid result must contain one workload: {path}")
    workload = workloads[0]
    if workload.get("decoded_validation") != "exact-float-equal":
        raise ComparisonError(f"Hybrid tail equivalence failed: {path}")
    pairs = workload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != args.samples:
        raise ComparisonError(f"Hybrid sample count changed: {path}")
    outputs = workload.get("correctness_outputs")
    if not isinstance(outputs, dict):
        raise ComparisonError(f"Hybrid correctness outputs are missing: {path}")
    return {
        "document": document,
        "gjxl": summarize_workflow_backend(pairs, outputs, "gjxl", path),
        "hybrid": summarize_workflow_backend(pairs, outputs, "libjxl", path),
    }


def parse_stock(
    path: Path, args: argparse.Namespace, expected_distance: float
) -> dict[str, Any]:
    document = load_json(path)
    if (
        document.get("schema_version") != 1
        or document.get("timing_semantics") != "complete-encode-wall-time"
        or document.get("encoder") != "libjxl"
        or document.get("encoder_path") != "ordinary-public-api"
        or document.get("thread_count") != args.threads
        or document.get("effort") != args.libjxl_effort
        or document.get("validation_encodes") != 1
        or document.get("warmups") != args.warmups
        or document.get("sample_count") != args.samples
        or not numeric_match(document.get("requested_distance"), expected_distance)
    ):
        raise ComparisonError(f"Unexpected stock benchmark schema: {path}")
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) != args.samples:
        raise ComparisonError(f"Stock sample count changed: {path}")
    encoded_sizes = {sample["encoded_bytes"] for sample in samples}
    if len(encoded_sizes) != 1:
        raise ComparisonError(f"Stock output size changed across samples: {path}")
    return {
        "document": document,
        "median_nanoseconds": median([item["elapsed_nanoseconds"] for item in samples]),
        "sample_nanoseconds": [item["elapsed_nanoseconds"] for item in samples],
        "output_bytes": encoded_sizes.pop(),
    }


def process_order(index: int) -> tuple[str, str]:
    return ("hybrid", "stock") if index % 2 == 0 else ("stock", "hybrid")


def run_comparison(args: argparse.Namespace) -> None:
    args.gjxl_benchmark = require_executable(args.gjxl_benchmark, "GJXL benchmark")
    args.stock_benchmark = require_executable(args.stock_benchmark, "stock benchmark")
    args.djxl = require_executable(args.djxl, "djxl")
    args.butteraugli = require_executable(args.butteraugli, "butteraugli")
    inputs = parse_inputs(args.input)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ComparisonError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "stock-libjxl-vs-fully-resident-gjxl-and-hybrid",
        "status": "running",
        "inputs": inputs,
        "parameters": {
            "gjxl_distance": args.gjxl_distance,
            "frontend_effort": args.frontend_effort,
            "libjxl_effort": args.libjxl_effort,
            "threads": args.threads,
            "process_pairs": args.process_pairs,
            "warmups": args.warmups,
            "samples": args.samples,
            "quality_tolerance": args.quality_tolerance,
            "maximum_relative_error": args.maximum_relative_error,
        },
        "binaries": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "gjxl_benchmark": args.gjxl_benchmark,
                "stock_benchmark": args.stock_benchmark,
                "djxl": args.djxl,
                "butteraugli": args.butteraugli,
            }.items()
        },
        "source": source_manifest(),
        "host_start": host_manifest(),
        "processes": [],
        "results": [],
    }
    write_json_atomic(output / "manifest.json", manifest)

    for entry in inputs:
        name = entry["name"]
        image = Path(entry["path"])
        input_dir = output / name
        input_dir.mkdir()
        pilot_dir = input_dir / "hybrid-pilot"
        pilot_artifacts = pilot_dir / "artifacts"
        pilot_artifacts.mkdir(parents=True)
        pilot_raw = pilot_dir / "raw.json"
        run_logged(
            hybrid_command(
                args,
                image,
                pilot_raw,
                pilot_artifacts,
                warmups=0,
                samples=1,
            ),
            f"{name}-hybrid-pilot",
            output,
            manifest,
        )
        hybrid_artifact = pilot_artifacts / "external_input-hybrid-libjxl.jxl"
        hybrid_sha256 = sha256_file(hybrid_artifact)
        hybrid_bytes = hybrid_artifact.stat().st_size
        gjxl_artifact = pilot_artifacts / "external_input-hybrid-gjxl.jxl"
        gjxl_sha256 = sha256_file(gjxl_artifact)
        gjxl_bytes = gjxl_artifact.stat().st_size
        target_score = score_codestream(
            args, image, hybrid_artifact, f"{name}-hybrid-pilot", output, manifest
        )

        calibration_dir = input_dir / "calibration"
        calibration_dir.mkdir()
        calibration_counter = 0

        def evaluate(distance: float) -> dict[str, Any]:
            nonlocal calibration_counter
            index = calibration_counter
            calibration_counter += 1
            stem = f"evaluation-{index:02d}-d{distance:.9g}"
            raw = calibration_dir / f"{stem}.json"
            codestream = calibration_dir / f"{stem}.jxl"
            run_logged(
                stock_command(
                    args,
                    image,
                    raw,
                    codestream,
                    distance,
                    warmups=0,
                    samples=1,
                ),
                f"{name}-calibration-{index:02d}",
                output,
                manifest,
            )
            score = score_codestream(
                args,
                image,
                codestream,
                f"{name}-calibration-{index:02d}",
                output,
                manifest,
            )
            raw_document = load_json(raw)
            return {
                "butteraugli": score,
                "encoded_bytes": codestream.stat().st_size,
                "codestream_sha256": sha256_file(codestream),
                "raw": str(raw.relative_to(output)),
                "codestream": str(codestream.relative_to(output)),
                "reported_revision": raw_document.get("revision"),
            }

        selected, evaluations = calibrate_distance(
            target_score,
            evaluate,
            initial=args.gjxl_distance,
            minimum=args.minimum_distance,
            maximum=args.maximum_distance,
            tolerance=args.quality_tolerance,
            maximum_evaluations=args.maximum_calibration_evaluations,
            maximum_relative_error=args.maximum_relative_error,
        )

        process_rows = []
        for pair_index in range(args.process_pairs):
            pair_dir = input_dir / f"pair-{pair_index}"
            pair_dir.mkdir()
            hybrid_raw = pair_dir / "hybrid.json"
            hybrid_artifacts = pair_dir / "hybrid-artifacts"
            hybrid_artifacts.mkdir()
            stock_raw = pair_dir / "stock.json"
            stock_output = pair_dir / "stock.jxl"
            commands = {
                "hybrid": hybrid_command(
                    args,
                    image,
                    hybrid_raw,
                    hybrid_artifacts,
                    warmups=args.warmups,
                    samples=args.samples,
                ),
                "stock": stock_command(
                    args,
                    image,
                    stock_raw,
                    stock_output,
                    selected["distance"],
                    warmups=args.warmups,
                    samples=args.samples,
                ),
            }
            for encoder in process_order(pair_index):
                run_logged(
                    commands[encoder],
                    f"{name}-pair-{pair_index}-{encoder}",
                    output,
                    manifest,
                )
            hybrid = parse_hybrid(hybrid_raw, args)
            stock = parse_stock(stock_raw, args, selected["distance"])
            retained_hybrid = hybrid_artifacts / "external_input-hybrid-libjxl.jxl"
            retained_hybrid_sha256 = sha256_file(retained_hybrid)
            retained_gjxl = hybrid_artifacts / "external_input-hybrid-gjxl.jxl"
            retained_gjxl_sha256 = sha256_file(retained_gjxl)
            stock_sha256 = sha256_file(stock_output)
            if (
                retained_hybrid_sha256 != hybrid_sha256
                or hybrid["hybrid"]["output_sha256"] != hybrid_sha256
                or hybrid["hybrid"]["output_bytes"] != hybrid_bytes
                or retained_gjxl_sha256 != gjxl_sha256
                or hybrid["gjxl"]["output_sha256"] != gjxl_sha256
                or hybrid["gjxl"]["output_bytes"] != gjxl_bytes
                or stock_sha256 != selected["codestream_sha256"]
                or stock["output_bytes"] != selected["encoded_bytes"]
            ):
                raise ComparisonError(f"Output changed across processes for {name}")
            process_rows.append(
                {
                    "pair_index": pair_index,
                    "first_encoder": process_order(pair_index)[0],
                    "gjxl_median_nanoseconds": hybrid["gjxl"]["median_nanoseconds"],
                    "gjxl_input_preparation_median_nanoseconds": hybrid["gjxl"][
                        "input_preparation_median_nanoseconds"
                    ],
                    "gjxl_quantization_pipeline_median_nanoseconds": hybrid["gjxl"][
                        "quantization_pipeline_median_nanoseconds"
                    ],
                    "gjxl_tail_median_nanoseconds": hybrid["gjxl"][
                        "tail_median_nanoseconds"
                    ],
                    "hybrid_median_nanoseconds": hybrid["hybrid"]["median_nanoseconds"],
                    "hybrid_input_preparation_median_nanoseconds": hybrid["hybrid"][
                        "input_preparation_median_nanoseconds"
                    ],
                    "hybrid_quantization_pipeline_median_nanoseconds": hybrid["hybrid"][
                        "quantization_pipeline_median_nanoseconds"
                    ],
                    "hybrid_tail_median_nanoseconds": hybrid["hybrid"][
                        "tail_median_nanoseconds"
                    ],
                    "stock_median_nanoseconds": stock["median_nanoseconds"],
                    "stock_over_gjxl_speedup": stock["median_nanoseconds"]
                    / hybrid["gjxl"]["median_nanoseconds"],
                    "stock_over_hybrid_speedup": stock["median_nanoseconds"]
                    / hybrid["hybrid"]["median_nanoseconds"],
                    "gjxl_over_hybrid_speedup": hybrid["gjxl"]["median_nanoseconds"]
                    / hybrid["hybrid"]["median_nanoseconds"],
                    "hybrid_raw": str(hybrid_raw.relative_to(output)),
                    "stock_raw": str(stock_raw.relative_to(output)),
                }
            )

        gjxl_medians = [row["gjxl_median_nanoseconds"] for row in process_rows]
        hybrid_medians = [row["hybrid_median_nanoseconds"] for row in process_rows]
        stock_medians = [row["stock_median_nanoseconds"] for row in process_rows]
        stock_over_gjxl = [row["stock_over_gjxl_speedup"] for row in process_rows]
        stock_over_hybrid = [row["stock_over_hybrid_speedup"] for row in process_rows]
        gjxl_over_hybrid = [row["gjxl_over_hybrid_speedup"] for row in process_rows]
        gjxl_tail_medians = [
            row["gjxl_tail_median_nanoseconds"] for row in process_rows
        ]
        hybrid_tail_medians = [
            row["hybrid_tail_median_nanoseconds"] for row in process_rows
        ]
        gjxl_preparation_medians = [
            row["gjxl_input_preparation_median_nanoseconds"] for row in process_rows
        ]
        hybrid_preparation_medians = [
            row["hybrid_input_preparation_median_nanoseconds"] for row in process_rows
        ]
        gjxl_quantization_medians = [
            row["gjxl_quantization_pipeline_median_nanoseconds"] for row in process_rows
        ]
        hybrid_quantization_medians = [
            row["hybrid_quantization_pipeline_median_nanoseconds"]
            for row in process_rows
        ]
        result = {
            "input": name,
            "quality": {
                "hybrid_butteraugli": target_score,
                "stock_butteraugli": selected["butteraugli"],
                "absolute_error": selected["absolute_error"],
                "relative_error": selected["relative_error"],
                "match_kind": selected["match_kind"],
                "stock_distance": selected["distance"],
            },
            "output": {
                "gjxl_bytes": gjxl_bytes,
                "gjxl_sha256": gjxl_sha256,
                "hybrid_bytes": hybrid_bytes,
                "hybrid_sha256": hybrid_sha256,
                "stock_bytes": selected["encoded_bytes"],
                "stock_sha256": selected["codestream_sha256"],
                "gjxl_size_delta_percent": 100.0
                * (gjxl_bytes - selected["encoded_bytes"])
                / selected["encoded_bytes"],
                "hybrid_size_delta_percent": 100.0
                * (hybrid_bytes - selected["encoded_bytes"])
                / selected["encoded_bytes"],
            },
            "timing": {
                "semantics": "median-of-independent-process-medians",
                "gjxl_median_nanoseconds": median(gjxl_medians),
                "gjxl_process_range_nanoseconds": [
                    min(gjxl_medians),
                    max(gjxl_medians),
                ],
                "gjxl_tail_median_nanoseconds": median(gjxl_tail_medians),
                "gjxl_tail_process_range_nanoseconds": [
                    min(gjxl_tail_medians),
                    max(gjxl_tail_medians),
                ],
                "gjxl_input_preparation_median_nanoseconds": median(
                    gjxl_preparation_medians
                ),
                "gjxl_quantization_pipeline_median_nanoseconds": median(
                    gjxl_quantization_medians
                ),
                "hybrid_median_nanoseconds": median(hybrid_medians),
                "hybrid_process_range_nanoseconds": [
                    min(hybrid_medians),
                    max(hybrid_medians),
                ],
                "hybrid_tail_median_nanoseconds": median(hybrid_tail_medians),
                "hybrid_tail_process_range_nanoseconds": [
                    min(hybrid_tail_medians),
                    max(hybrid_tail_medians),
                ],
                "hybrid_input_preparation_median_nanoseconds": median(
                    hybrid_preparation_medians
                ),
                "hybrid_quantization_pipeline_median_nanoseconds": median(
                    hybrid_quantization_medians
                ),
                "stock_median_nanoseconds": median(stock_medians),
                "stock_process_range_nanoseconds": [
                    min(stock_medians),
                    max(stock_medians),
                ],
                "stock_over_gjxl_speedup": median(stock_over_gjxl),
                "stock_over_gjxl_speedup_process_range": [
                    min(stock_over_gjxl),
                    max(stock_over_gjxl),
                ],
                "stock_over_hybrid_speedup": median(stock_over_hybrid),
                "stock_over_hybrid_speedup_process_range": [
                    min(stock_over_hybrid),
                    max(stock_over_hybrid),
                ],
                "gjxl_over_hybrid_speedup": median(gjxl_over_hybrid),
                "gjxl_over_hybrid_speedup_process_range": [
                    min(gjxl_over_hybrid),
                    max(gjxl_over_hybrid),
                ],
            },
            "process_pairs": process_rows,
            "calibration": evaluations,
        }
        manifest["results"].append(result)
        write_json_atomic(output / "manifest.json", manifest)

    manifest["host_end"] = host_manifest()
    manifest["status"] = "complete"
    write_json_atomic(output / "manifest.json", manifest)
    write_json_atomic(
        output / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": manifest["kind"],
            "parameters": manifest["parameters"],
            "results": manifest["results"],
        },
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PFM")
    parser.add_argument("--gjxl-benchmark", type=Path, required=True)
    parser.add_argument("--stock-benchmark", type=Path, required=True)
    parser.add_argument("--djxl", type=Path, required=True)
    parser.add_argument("--butteraugli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gjxl-distance", type=positive_float, default=1.2)
    parser.add_argument("--frontend-effort", type=positive_int, default=7)
    parser.add_argument("--libjxl-effort", type=positive_int, default=7)
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--process-pairs", type=positive_int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=positive_int, default=5)
    parser.add_argument("--quality-tolerance", type=positive_float, default=0.015)
    parser.add_argument("--maximum-relative-error", type=positive_float, default=0.025)
    parser.add_argument("--minimum-distance", type=positive_float, default=0.25)
    parser.add_argument("--maximum-distance", type=positive_float, default=3.0)
    parser.add_argument(
        "--maximum-calibration-evaluations", type=positive_int, default=16
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(argv)
        if args.warmups < 0:
            raise ComparisonError("--warmups must be nonnegative")
        if not 1 <= args.frontend_effort <= 10 or not 1 <= args.libjxl_effort <= 10:
            raise ComparisonError("efforts must be in [1, 10]")
        run_comparison(args)
    except (ComparisonError, OSError, ValueError) as error:
        if args is not None:
            manifest_path = args.output.expanduser().resolve() / "manifest.json"
            try:
                manifest = load_json(manifest_path)
                if manifest.get("status") == "running":
                    manifest["status"] = "failed"
                    manifest["failure"] = str(error)
                    manifest["host_end"] = host_manifest()
                    write_json_atomic(manifest_path, manifest)
            except (ComparisonError, OSError):
                pass
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
