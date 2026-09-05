# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Corpus and calibration helpers extracted from the libjxl comparison harness.

Source: perf/libjxl-comparison c509175. No profiling/build pipeline is imported.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from encode_image import (
    convert_to_pfm,
    find_executable,
    is_pfm,
    require_single_frame,
    run_checked,
    validate_pfm,
)

CORPUS_SCHEMA_VERSION = 1
LINEAR_SRGB_HINT = "RGB_D65_SRG_Rel_Lin"


class ComparisonError(RuntimeError):
    """An expected qualification setup or execution failure."""


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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Unable to read JSON {path}: {exc}") from exc


def run_capture(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ComparisonError(f"Unable to run {command[0]}: {exc}") from exc
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ComparisonError(
            f"Command failed ({result.returncode}): {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return result


def git_output(repo: Path, *arguments: str) -> str:
    return run_capture(["git", "-C", str(repo), *arguments]).stdout.strip()


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not normalized or normalized in {".", ".."}:
        raise ComparisonError(f"Unsafe or empty corpus name: {value!r}")
    return normalized


def _pfm_line(stream: Any) -> bytes:
    while True:
        line = stream.readline()
        if not line:
            raise ComparisonError("PFM header is incomplete")
        line = line.rstrip(b"\r\n")
        if line and not line.startswith(b"#"):
            return line


def inspect_pfm(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as stream:
            if _pfm_line(stream) != b"PF":
                raise ComparisonError(f"Canonical input is not RGB PFM: {path}")
            dimensions = _pfm_line(stream).split()
            if len(dimensions) != 2:
                raise ComparisonError(f"PFM dimensions are invalid: {path}")
            width, height = (int(value) for value in dimensions)
            scale = float(_pfm_line(stream))
            header_bytes = stream.tell()
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"Unable to inspect PFM {path}: {exc}") from exc
    if width <= 0 or height <= 0 or not math.isfinite(scale) or scale == 0:
        raise ComparisonError(f"PFM header is invalid: {path}")
    expected = header_bytes + width * height * 3 * 4
    if path.stat().st_size != expected:
        raise ComparisonError(
            f"PFM payload size mismatch for {path}: "
            f"expected {expected}, found {path.stat().st_size}"
        )
    return width, height


def _manifest_dimension(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComparisonError(f"Corpus transform {field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        relation = "nonnegative" if allow_zero else "positive"
        raise ComparisonError(f"Corpus transform {field} must be {relation}")
    return value


def parse_source_transform(entry: dict[str, Any]) -> dict[str, Any] | None:
    transform: dict[str, Any] = {}
    if "crop" in entry:
        crop = entry["crop"]
        if not isinstance(crop, dict) or set(crop) != {"x", "y", "width", "height"}:
            raise ComparisonError(
                "Corpus crop must contain exactly x, y, width, and height"
            )
        transform["crop"] = {
            "x": _manifest_dimension(crop["x"], "crop.x", allow_zero=True),
            "y": _manifest_dimension(crop["y"], "crop.y", allow_zero=True),
            "width": _manifest_dimension(crop["width"], "crop.width"),
            "height": _manifest_dimension(crop["height"], "crop.height"),
        }
    if "resize" in entry:
        resize = entry["resize"]
        if not isinstance(resize, dict) or set(resize) != {"width", "height"}:
            raise ComparisonError("Corpus resize must contain exactly width and height")
        transform["resize"] = {
            "width": _manifest_dimension(resize["width"], "resize.width"),
            "height": _manifest_dimension(resize["height"], "resize.height"),
            "filter": "Lanczos",
        }
    return transform or None


def validated_source_actions(source_manifest: Path) -> list[dict[str, Any]]:
    document = load_json(source_manifest)
    if document.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ComparisonError("Unsupported source-manifest schema")
    entries = document.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ComparisonError("Source manifest must contain a nonempty inputs list")
    actions: dict[Path, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ComparisonError("Each source-manifest input must be an object")
        path_value = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ComparisonError("Corpus input is missing string field path")
        path = Path(path_value)
        if path.is_absolute() or ".." in path.parts:
            raise ComparisonError(f"Corpus path must be relative and contained: {path}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ComparisonError(f"Corpus source {path} has an invalid SHA-256")
        generator = entry.get("generator")
        if generator is None:
            url = entry.get("source")
            if not isinstance(url, str) or urlparse(url).scheme != "https":
                raise ComparisonError(
                    f"Downloaded source must be an HTTPS URL: {url!r}"
                )
            action = {
                "kind": "download",
                "path": path,
                "url": url,
                "sha256": digest,
            }
        else:
            expected_keys = {"kind", "workload"}
            if not isinstance(generator, dict) or set(generator) != expected_keys:
                raise ComparisonError(
                    "Corpus generator must contain exactly kind and workload"
                )
            if generator["kind"] != "gjxl-encoding-benchmark-source-v1":
                raise ComparisonError(
                    f"Unknown corpus generator: {generator['kind']!r}"
                )
            workload = generator["workload"]
            if not isinstance(workload, str) or not workload:
                raise ComparisonError("Corpus generator workload must be a string")
            action = {
                "kind": "generate",
                "path": path,
                "workload": workload,
                "sha256": digest,
            }
        previous = actions.get(path)
        if previous is not None and previous != action:
            raise ComparisonError(f"Corpus path has conflicting sources: {path}")
        actions[path] = action
    return list(actions.values())


def fetch_corpus_sources(args: argparse.Namespace) -> None:
    source_manifest = args.source_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for action in validated_source_actions(source_manifest):
        relative_path = action["path"]
        expected_sha256 = action["sha256"]
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            actual_sha256 = sha256_file(destination)
            if actual_sha256 != expected_sha256:
                raise ComparisonError(
                    f"Existing corpus source SHA-256 differs for {destination}: "
                    f"actual={actual_sha256} expected={expected_sha256}"
                )
            continue
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=destination.name + ".tmp-",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                if action["kind"] == "download":
                    request = Request(
                        action["url"],
                        headers={"User-Agent": "gjxl-comparison/1"},
                    )
                    with urlopen(request, timeout=args.timeout) as response:
                        shutil.copyfileobj(response, stream)
            if action["kind"] == "generate":
                benchmark = args.gjxl_benchmark.resolve()
                if not benchmark.is_file():
                    raise ComparisonError(
                        f"GJXL benchmark executable is missing: {benchmark}"
                    )
                run_capture(
                    [
                        str(benchmark),
                        "--workload",
                        action["workload"],
                        "--source-output",
                        str(temporary),
                    ]
                )
            actual_sha256 = sha256_file(temporary)
            if actual_sha256 != expected_sha256:
                raise ComparisonError(
                    f"Corpus source SHA-256 differs for {relative_path}: "
                    f"actual={actual_sha256} expected={expected_sha256}"
                )
            temporary.replace(destination)
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
    print(output)


def convert_transformed_source_to_pfm(
    magick_command: str,
    source: Path,
    destination: Path,
    background: str,
    transform: dict[str, Any],
) -> tuple[tuple[int, int], str]:
    magick = find_executable(magick_command, "ImageMagick")
    require_single_frame(magick, source)
    command = [magick, str(source), "-auto-orient"]
    display_command = [magick_command, "INPUT", "-auto-orient"]
    crop = transform.get("crop")
    if crop is not None:
        geometry = f"{crop['width']}x{crop['height']}+{crop['x']}+{crop['y']}"
        command.extend(["-crop", geometry, "+repage"])
        display_command.extend(["-crop", geometry, "+repage"])
    command.extend(["-colorspace", "RGB"])
    display_command.extend(["-colorspace", "RGB"])
    resize = transform.get("resize")
    if resize is not None:
        geometry = f"{resize['width']}x{resize['height']}!"
        command.extend(["-filter", "Lanczos", "-resize", geometry])
        display_command.extend(["-filter", "Lanczos", "-resize", geometry])
    suffix = [
        "-background",
        background,
        "-alpha",
        "remove",
        "-alpha",
        "off",
        "-type",
        "TrueColor",
    ]
    command.extend([*suffix, f"PFM:{destination}"])
    display_command.extend([*suffix, "PFM:OUTPUT"])
    run_checked(command, "ImageMagick transformed corpus conversion")
    return validate_pfm(destination), " ".join(display_command)


def prepare_corpus(args: argparse.Namespace) -> None:
    source_manifest = args.source_manifest.resolve()
    source_root_argument = getattr(args, "source_root", None)
    source_root = (
        source_root_argument.resolve()
        if source_root_argument is not None
        else source_manifest.parent
    )
    document = load_json(source_manifest)
    if document.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ComparisonError("Unsupported source-manifest schema")
    entries = document.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ComparisonError("Source manifest must contain a nonempty inputs list")

    output = args.output.resolve()
    if output.exists():
        raise ComparisonError(f"Corpus output already exists: {output}")
    output.mkdir(parents=True)
    canonical_dir = output / "canonical"
    canonical_dir.mkdir()

    retained: list[dict[str, Any]] = []
    names: set[str] = set()
    magick_version: str | None = None
    try:
        for entry in entries:
            if not isinstance(entry, dict):
                raise ComparisonError("Each source-manifest input must be an object")
            for field in ("name", "path", "source", "license", "source_color"):
                if not isinstance(entry.get(field), str) or not entry[field]:
                    raise ComparisonError(
                        f"Corpus input is missing string field {field}"
                    )
            name = safe_name(entry["name"])
            if name in names:
                raise ComparisonError(f"Duplicate corpus input name: {name}")
            names.add(name)
            source = (source_root / entry["path"]).resolve()
            if not source.is_file():
                raise ComparisonError(f"Corpus source does not exist: {source}")
            source_sha256 = sha256_file(source)
            expected_sha256 = entry.get("sha256")
            if expected_sha256 is not None:
                if not isinstance(expected_sha256, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", expected_sha256
                ):
                    raise ComparisonError(f"Corpus input {name} has an invalid SHA-256")
                if source_sha256 != expected_sha256:
                    raise ComparisonError(
                        f"Corpus source SHA-256 differs for {name}: "
                        f"actual={source_sha256} expected={expected_sha256}"
                    )
            destination = canonical_dir / f"{name}.pfm"
            source_transform = parse_source_transform(entry)
            if is_pfm(source):
                if source_transform is not None:
                    raise ComparisonError(
                        f"Identity PFM input {name} cannot request crop or resize"
                    )
                if entry["source_color"].lower() not in {
                    "linear-srgb",
                    "linear srgb",
                    LINEAR_SRGB_HINT.lower(),
                }:
                    raise ComparisonError(
                        f"Identity PFM input {name} must declare linear-sRGB pixels"
                    )
                shutil.copyfile(source, destination)
                conversion = "identity copy of declared linear-sRGB PFM"
            else:
                if magick_version is None:
                    magick_version = run_capture(
                        [args.magick, "-version"]
                    ).stdout.splitlines()[0]
                if source_transform is None:
                    convert_to_pfm(args.magick, source, destination, args.background)
                    conversion = (
                        f"{args.magick} INPUT -auto-orient -colorspace RGB "
                        f"-background {args.background} -alpha remove -alpha off "
                        "-type TrueColor PFM:OUTPUT"
                    )
                else:
                    _, conversion = convert_transformed_source_to_pfm(
                        args.magick,
                        source,
                        destination,
                        args.background,
                        source_transform,
                    )
            width, height = inspect_pfm(destination)
            retained.append(
                {
                    "name": name,
                    "canonical_path": str(destination.relative_to(output)),
                    "canonical_sha256": sha256_file(destination),
                    "width": width,
                    "height": height,
                    "canonical_color": "linear-sRGB D65 relative",
                    "canonical_format": "PFM RGB float32, file-bottom-up",
                    "conversion": conversion,
                    "source_transform": source_transform,
                    "source_path": str(source),
                    "source_sha256": source_sha256,
                    "source": entry["source"],
                    "license": entry["license"],
                    "source_color": entry["source_color"],
                    "category": entry.get("category", "photographic"),
                }
            )
        write_json_atomic(
            output / "manifest.json",
            {
                "schema_version": CORPUS_SCHEMA_VERSION,
                "source_manifest_sha256": sha256_file(source_manifest),
                "canonical_color": "linear-sRGB D65 relative",
                "alpha_background": args.background,
                "conversion_tool": magick_version,
                "inputs": retained,
            },
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(output / "manifest.json")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ComparisonError(f"{field} must be finite")
    return parsed


def calibrate_distance(
    target_score: float,
    evaluate: Callable[[float], dict[str, Any]],
    *,
    minimum_distance: float,
    maximum_distance: float,
    initial_distance: float,
    tolerance: float,
    maximum_evaluations: int,
    maximum_relative_error: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Find a requested distance whose decoded score matches target_score."""

    if not minimum_distance < maximum_distance:
        raise ComparisonError("Calibration distance bounds are empty")
    if not minimum_distance <= initial_distance <= maximum_distance:
        raise ComparisonError("Initial calibration distance is outside the bounds")
    if maximum_evaluations < 3:
        raise ComparisonError("Calibration requires at least three evaluations")
    evaluations: list[dict[str, Any]] = []
    by_distance: dict[float, dict[str, Any]] = {}

    def select(result: dict[str, Any], match_kind: str) -> dict[str, Any]:
        result["relative_error"] = (
            result["absolute_error"] / target_score if target_score > 0.0 else math.inf
        )
        result["match_kind"] = match_kind
        return result

    def evaluate_once(distance: float) -> dict[str, Any]:
        distance = float(distance)
        if distance not in by_distance:
            result = dict(evaluate(distance))
            result["distance"] = distance
            result["absolute_error"] = abs(
                _finite_number(result.get("butteraugli"), "calibrated score")
                - target_score
            )
            by_distance[distance] = result
            evaluations.append(result)
        return by_distance[distance]

    initial = evaluate_once(initial_distance)
    if initial["absolute_error"] <= tolerance:
        return select(initial, "within-absolute-tolerance"), evaluations

    if initial["butteraugli"] > target_score:
        endpoint = evaluate_once(minimum_distance)
    else:
        endpoint = evaluate_once(maximum_distance)
    if endpoint["absolute_error"] <= tolerance:
        return select(endpoint, "within-absolute-tolerance"), evaluations

    def find_bracket() -> tuple[dict[str, Any], dict[str, Any]] | None:
        ordered = sorted(evaluations, key=lambda item: item["distance"])
        brackets = []
        for lower, upper in zip(ordered, ordered[1:]):
            lower_delta = lower["butteraugli"] - target_score
            upper_delta = upper["butteraugli"] - target_score
            if lower_delta * upper_delta <= 0.0:
                brackets.append((lower, upper))
        if not brackets:
            return None
        return min(
            brackets,
            key=lambda pair: (
                pair[1]["distance"] - pair[0]["distance"],
                pair[0]["absolute_error"] + pair[1]["absolute_error"],
            ),
        )

    bracket = find_bracket()
    if bracket is None:
        coarse_candidates = [
            minimum_distance + index * (maximum_distance - minimum_distance) / 8.0
            for index in range(9)
        ]
        coarse_candidates.sort(key=lambda distance: abs(distance - initial_distance))
        for distance in coarse_candidates:
            if len(evaluations) >= maximum_evaluations:
                break
            candidate = evaluate_once(distance)
            if candidate["absolute_error"] <= tolerance:
                return select(candidate, "within-absolute-tolerance"), evaluations
            bracket = find_bracket()
            if bracket is not None:
                break

    while bracket is not None and len(evaluations) < maximum_evaluations:
        lower, upper = bracket
        midpoint = (lower["distance"] + upper["distance"]) / 2.0
        if midpoint in by_distance:
            break
        candidate = evaluate_once(midpoint)
        if candidate["absolute_error"] <= tolerance:
            return select(candidate, "within-absolute-tolerance"), evaluations
        lower_delta = lower["butteraugli"] - target_score
        candidate_delta = candidate["butteraugli"] - target_score
        if lower_delta * candidate_delta <= 0.0:
            bracket = (lower, candidate)
        else:
            bracket = (candidate, upper)

    best = min(evaluations, key=lambda item: item["absolute_error"])
    if best["absolute_error"] > tolerance:
        if (
            target_score > 0.0
            and best["absolute_error"] / target_score <= maximum_relative_error
        ):
            at_boundary = best["distance"] in {
                minimum_distance,
                maximum_distance,
            }
            return select(
                best,
                (
                    "boundary-limited-relative-tolerance"
                    if at_boundary
                    else "quantized-relative-tolerance"
                ),
            ), evaluations
        raise ComparisonError(
            "Calibration did not converge within tolerance: "
            f"target={target_score:.9g}, best={best['butteraugli']:.9g}, "
            f"error={best['absolute_error']:.9g}, tolerance={tolerance:.9g}"
        )
    return select(best, "within-absolute-tolerance"), evaluations
