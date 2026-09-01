#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho

"""Prepare and run reproducible GJXL/libjxl comparison pilots."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from encode_image import (
    WrapperError,
    convert_to_pfm,
    find_executable,
    is_pfm,
    require_single_frame,
    run_checked,
    validate_pfm,
)


PINNED_LIBJXL_REVISION = "e8ff09762481785938d8e4e01333ed3917571161"
CORPUS_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
CALIBRATION_SCHEMA_VERSION = 1
LINEAR_SRGB_HINT = "RGB_D65_SRG_Rel_Lin"
PROFILE_MINIMUM_RESOLUTION_PERCENT = 95.0


class ComparisonError(RuntimeError):
    """An expected comparison setup or execution failure."""


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


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".tmp-",
        delete=False,
    ) as output:
        output.write(value)
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
            raise ComparisonError(
                "Corpus resize must contain exactly width and height"
            )
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
            raise ComparisonError(
                f"Corpus path must be relative and contained: {path}"
            )
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
        geometry = (
            f"{crop['width']}x{crop['height']}+{crop['x']}+{crop['y']}"
        )
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
                    raise ComparisonError(f"Corpus input is missing string field {field}")
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
                    raise ComparisonError(
                        f"Corpus input {name} has an invalid SHA-256"
                    )
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
                    convert_to_pfm(
                        args.magick, source, destination, args.background
                    )
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


def configure_libjxl(source: Path, build: Path, generator: str) -> None:
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DBUILD_SHARED_LIBS=ON",
        "-DJPEGXL_ENABLE_TOOLS=ON",
        "-DJPEGXL_ENABLE_DEVTOOLS=ON",
        "-DJPEGXL_ENABLE_BENCHMARK=OFF",
        "-DJPEGXL_ENABLE_EXAMPLES=OFF",
        "-DJPEGXL_ENABLE_JNI=OFF",
        "-DJPEGXL_ENABLE_MANPAGES=OFF",
        "-DJPEGXL_ENABLE_DOXYGEN=OFF",
        "-DJPEGXL_ENABLE_SJPEG=OFF",
        "-DJPEGXL_ENABLE_OPENEXR=OFF",
        "-DJPEGXL_ENABLE_VIEWERS=OFF",
        "-DJPEGXL_ENABLE_TRANSCODE_JPEG=OFF",
        "-DHWY_ENABLE_TESTS=OFF",
        "-DHWY_ENABLE_EXAMPLES=OFF",
    ]
    run_capture(command)


def build_libjxl(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    source = (repo / "third_party/libjxl").resolve()
    actual_revision = git_output(source, "rev-parse", "HEAD")
    if actual_revision != PINNED_LIBJXL_REVISION:
        raise ComparisonError(
            f"Pinned libjxl is {actual_revision}, expected {PINNED_LIBJXL_REVISION}"
        )
    root = args.build_root.resolve()
    libjxl_build = root / "libjxl"
    harness_build = root / "harness"
    configure_libjxl(source, libjxl_build, args.generator)
    run_capture(
        [
            "cmake",
            "--build",
            str(libjxl_build),
            "--config",
            "Release",
            "--target",
            "jxl",
            "jxl_threads",
            "djxl",
            "butteraugli_main",
            "-j",
            str(args.jobs),
        ]
    )
    run_capture(
        [
            "cmake",
            "-S",
            str(repo / "benchmarks/libjxl_comparison"),
            "-B",
            str(harness_build),
            "-G",
            args.generator,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DGJXL_LIBJXL_SOURCE={source}",
            f"-DGJXL_LIBJXL_BUILD={libjxl_build}",
            f"-DGJXL_LIBJXL_REVISION={PINNED_LIBJXL_REVISION}",
        ]
    )
    run_capture(
        [
            "cmake",
            "--build",
            str(harness_build),
            "--config",
            "Release",
            "--target",
            "gjxl_libjxl_comparison_benchmark",
            "-j",
            str(args.jobs),
        ]
    )
    binaries = {
        "benchmark": harness_build / "gjxl_libjxl_comparison_benchmark",
        "djxl": libjxl_build / "tools/djxl",
        "butteraugli": libjxl_build / "tools/butteraugli_main",
    }
    for name, path in binaries.items():
        if not path.is_file():
            raise ComparisonError(f"libjxl {name} binary was not produced: {path}")
    cache = {}
    cache_path = libjxl_build / "CMakeCache.txt"
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        if key in {
            "CMAKE_BUILD_TYPE",
            "CMAKE_C_COMPILER",
            "CMAKE_CXX_COMPILER",
            "CMAKE_CXX_FLAGS_RELEASE",
            "CMAKE_GENERATOR",
            "CMAKE_OSX_ARCHITECTURES",
            "CMAKE_SYSTEM_PROCESSOR",
        }:
            cache[key] = value
    write_json_atomic(
        root / "build-manifest.json",
        {
            "schema_version": 1,
            "libjxl_revision": actual_revision,
            "source": str(source),
            "build": str(libjxl_build),
            "harness_build": str(harness_build),
            "cmake": cache,
            "binaries": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in binaries.items()
            },
        },
    )
    print(binaries["benchmark"])


def validate_corpus(manifest_path: Path) -> list[dict[str, Any]]:
    document = load_json(manifest_path)
    if document.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ComparisonError("Unsupported canonical corpus schema")
    entries = document.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ComparisonError("Canonical corpus has no inputs")
    validated = []
    for source_entry in entries:
        entry = dict(source_entry)
        path = (manifest_path.parent / entry["canonical_path"]).resolve()
        if not path.is_file():
            raise ComparisonError(f"Canonical input is missing: {path}")
        width, height = inspect_pfm(path)
        digest = sha256_file(path)
        if (width, height) != (entry.get("width"), entry.get("height")):
            raise ComparisonError(f"Canonical dimensions changed: {path}")
        if digest != entry.get("canonical_sha256"):
            raise ComparisonError(f"Canonical SHA-256 changed: {path}")
        entry["resolved_path"] = str(path)
        requires_padding = width % 8 != 0 or height % 8 != 0
        entry["edge_extension_required"] = {
            "gjxl": requires_padding,
            "libjxl": requires_padding,
        }
        validated.append(entry)
    return validated


def select_corpus_entries(
    entries: list[dict[str, Any]], names: list[str] | None
) -> list[dict[str, Any]]:
    if not names:
        return entries
    if len(names) != len(set(names)):
        raise ComparisonError("Comparison input filter contains duplicates")
    requested = set(names)
    available = {entry["name"] for entry in entries}
    unknown = sorted(requested - available)
    if unknown:
        raise ComparisonError(f"Unknown comparison inputs: {unknown}")
    return [entry for entry in entries if entry["name"] in requested]


def command_output(command: list[str]) -> str | None:
    result = run_capture(command, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def host_manifest() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "macos": command_output(["sw_vers"]),
        "xcode": command_output(["xcodebuild", "-version"]),
        "c_compiler": command_output(["cc", "--version"]),
        "cxx_compiler": command_output(["c++", "--version"]),
        "hardware": command_output(["system_profiler", "SPHardwareDataType"]),
        "power": command_output(["pmset", "-g", "batt"]),
        "thermal": command_output(["pmset", "-g", "therm"]),
        "load": command_output(["uptime"]),
    }


def record_process(
    command: list[str],
    name: str,
    directory: Path,
    manifest: dict[str, Any],
) -> None:
    begin = time.monotonic_ns()
    result = run_capture(command, check=False)
    elapsed = time.monotonic_ns() - begin
    stdout_path = directory / f"{name}.stdout.txt"
    stderr_path = directory / f"{name}.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    record = {
        "name": name,
        "argv": command,
        "exit_code": result.returncode,
        "process_wall_nanoseconds": elapsed,
        "stdout": str(stdout_path.relative_to(directory)),
        "stderr": str(stderr_path.relative_to(directory)),
    }
    manifest["processes"].append(record)
    write_json_atomic(directory / "manifest.json", manifest)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ComparisonError(
            f"Comparison process {name} failed ({result.returncode})"
            + (f": {detail}" if detail else "")
        )


def gjxl_command(
    args: argparse.Namespace,
    image: Path,
    raw: Path,
    worker_limit: int,
    output: Path | None,
    samples: int | None = None,
    distance: float | None = None,
) -> list[str]:
    command = [
        str(args.gjxl_benchmark),
        "--scope",
        "metal-public-workflow",
        "--validation",
        "metal-only",
        "--gpu-aq",
        "fully-resident",
        "--implementation",
        args.gjxl_implementation,
        "--input",
        str(image),
        "--distance",
        str(args.distance if distance is None else distance),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples if samples is None else samples),
        "--serializer-workers",
        str(worker_limit),
        "--raw-samples",
        str(raw),
    ]
    if args.metallib is not None:
        command.extend(("--metallib", str(args.metallib)))
    if output is not None:
        command.extend(("--codestream-output", str(output)))
    return command


def libjxl_command(
    args: argparse.Namespace,
    image: Path,
    raw: Path,
    thread_count: int,
    output: Path | None,
    samples: int | None = None,
    distance: float | None = None,
) -> list[str]:
    command = [
        str(args.libjxl_benchmark),
        "--input",
        str(image),
        "--raw-samples",
        str(raw),
        "--distance",
        str(args.distance if distance is None else distance),
        "--effort",
        str(args.effort),
        "--num-threads",
        str(thread_count),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples if samples is None else samples),
    ]
    if output is not None:
        command.extend(("--output", str(output)))
    return command


def median_process_row(
    encoder: str,
    configuration: str,
    entry: dict[str, Any],
    raw_path: Path,
) -> dict[str, Any]:
    document = load_json(raw_path)
    if encoder == "gjxl":
        if document.get("schema_version") != 10:
            raise ComparisonError(f"Unexpected GJXL raw schema: {raw_path}")
        samples = [
            sample
            for workload in document["workloads"]
            for sample in workload["samples"]
            if sample["backend"] == "metal"
        ]
        elapsed = [sample["phase_nanoseconds"]["total"] for sample in samples]
        codestream = [
            sample["phase_nanoseconds"]["codestream_encoding"]
            for sample in samples
        ]
        encoded = [sample["encoded_bytes"] for sample in samples]
        requested_distance = document["distance"]
        policy = {"serializer_workers": document["serializer_workers"]}
    else:
        if document.get("schema_version") != 1 or document.get("encoder") != "libjxl":
            raise ComparisonError(f"Unexpected libjxl raw schema: {raw_path}")
        if document.get("revision") != PINNED_LIBJXL_REVISION:
            raise ComparisonError(f"libjxl benchmark revision mismatch: {raw_path}")
        samples = document["samples"]
        elapsed = [sample["elapsed_nanoseconds"] for sample in samples]
        codestream = []
        encoded = [sample["encoded_bytes"] for sample in samples]
        requested_distance = document["requested_distance"]
        policy = {"thread_count": document["thread_count"]}
    if not elapsed or len(set(encoded)) != 1:
        raise ComparisonError(f"Invalid or unstable raw samples: {raw_path}")
    pixels = entry["width"] * entry["height"]
    elapsed_median = statistics.median(elapsed)
    return {
        "encoder": encoder,
        "configuration": configuration,
        "input": entry["name"],
        "category": entry.get("category", "unspecified"),
        "raw_path": str(raw_path),
        "sample_count": len(elapsed),
        "median_nanoseconds": elapsed_median,
        "minimum_nanoseconds": min(elapsed),
        "maximum_nanoseconds": max(elapsed),
        "codestream_median_nanoseconds": (
            statistics.median(codestream) if codestream else None
        ),
        "codestream_nanoseconds_per_pixel": (
            statistics.median(codestream) / pixels if codestream else None
        ),
        "encoded_bytes": encoded[0],
        "requested_distance": requested_distance,
        "thread_policy": policy,
        "bits_per_pixel": 8.0 * encoded[0] / pixels,
        # One nanosecond per pixel is numerically one millisecond per
        # megapixel, so no scale factor is required here.
        "nanoseconds_per_pixel": elapsed_median / pixels,
        "milliseconds_per_megapixel": elapsed_median / pixels,
        "milliseconds_per_encoded_megabyte": (
            elapsed_median / 1_000_000.0 / (encoded[0] / 1_000_000.0)
        ),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["input"], row["configuration"], row["encoder"])
        grouped.setdefault(key, []).append(row)
    result = []
    for (input_name, configuration, encoder), group in sorted(grouped.items()):
        medians = [row["median_nanoseconds"] for row in group]
        encoded_sizes = {row["encoded_bytes"] for row in group}
        if len(encoded_sizes) != 1:
            raise ComparisonError(
                f"Encoded size changed across processes for {input_name} {encoder}"
            )
        requested_distances = {row["requested_distance"] for row in group}
        if len(requested_distances) != 1:
            raise ComparisonError(
                f"Requested distance changed across processes for {input_name} "
                f"{encoder}"
            )
        codestream_medians = [
            row["codestream_median_nanoseconds"]
            for row in group
            if row["codestream_median_nanoseconds"] is not None
        ]
        result.append(
            {
                "input": input_name,
                "category": group[0]["category"],
                "configuration": configuration,
                "encoder": encoder,
                "process_count": len(group),
                "median_of_process_medians_nanoseconds": statistics.median(medians),
                "process_median_range_nanoseconds": [min(medians), max(medians)],
                "encoded_bytes": group[0]["encoded_bytes"],
                "requested_distance": requested_distances.pop(),
                "bits_per_pixel": group[0]["bits_per_pixel"],
                "median_milliseconds_per_megapixel": statistics.median(
                    row["milliseconds_per_megapixel"] for row in group
                ),
                "median_milliseconds_per_encoded_megabyte": statistics.median(
                    row["milliseconds_per_encoded_megabyte"] for row in group
                ),
                "codestream_median_of_process_medians_nanoseconds": (
                    statistics.median(codestream_medians)
                    if codestream_medians
                    else None
                ),
                "codestream_median_nanoseconds_per_pixel": (
                    statistics.median(
                        row["codestream_nanoseconds_per_pixel"]
                        for row in group
                        if row["codestream_nanoseconds_per_pixel"] is not None
                    )
                    if codestream_medians
                    else None
                ),
            }
        )
    return result


def encoder_order(pair_index: int) -> tuple[str, str]:
    return ("gjxl", "libjxl") if pair_index % 2 == 0 else ("libjxl", "gjxl")


def validate_output(
    args: argparse.Namespace,
    entry: dict[str, Any],
    encoder: str,
    configuration: str,
    codestream: Path,
    directory: Path,
    manifest: dict[str, Any],
    requested_distance: float | None = None,
) -> dict[str, Any]:
    decoded = codestream.with_suffix(".decoded.pfm")
    decode_name = (
        f"validate-{safe_name(entry['name'])}-{configuration}-{encoder}-decode"
    )
    record_process(
        [str(args.djxl), str(codestream), str(decoded)],
        decode_name,
        directory,
        manifest,
    )
    width, height = inspect_pfm(decoded)
    if (width, height) != (entry["width"], entry["height"]):
        raise ComparisonError(f"Decoded dimensions changed for {codestream}")
    score_name = (
        f"validate-{safe_name(entry['name'])}-{configuration}-{encoder}-butteraugli"
    )
    command = [
        str(args.butteraugli),
        entry["resolved_path"],
        str(decoded),
        "--colorspace",
        LINEAR_SRGB_HINT,
    ]
    record_process(command, score_name, directory, manifest)
    stdout_path = directory / f"{score_name}.stdout.txt"
    try:
        score = float(stdout_path.read_text(encoding="utf-8").splitlines()[0])
    except (OSError, ValueError, IndexError) as exc:
        raise ComparisonError("Unable to parse Butteraugli score") from exc
    validation = {
        "input": entry["name"],
        "encoder": encoder,
        "configuration": configuration,
        "codestream": str(codestream.relative_to(directory)),
        "codestream_sha256": sha256_file(codestream),
        "decoded": str(decoded.relative_to(directory)),
        "decoded_sha256": sha256_file(decoded),
        "butteraugli": score,
    }
    if requested_distance is not None:
        validation["requested_distance"] = requested_distance
    return validation


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ComparisonError(f"{field} must be finite")
    return parsed


def _positive_distance(value: Any, field: str) -> float:
    parsed = _finite_number(value, field)
    if parsed <= 0.0 or parsed > 25.0:
        raise ComparisonError(f"{field} must be in (0, 25]")
    return parsed


def load_nominal_quality_targets(
    nominal_summary_path: Path,
    corpus_manifest_path: Path,
    entries: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    nominal_summary_path = nominal_summary_path.resolve()
    summary = load_json(nominal_summary_path)
    manifest_path = nominal_summary_path.parent / "manifest.json"
    manifest = load_json(manifest_path)
    if summary.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ComparisonError(
            f"Unexpected nominal summary schema: {nominal_summary_path}"
        )
    if manifest.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ComparisonError(f"Unexpected nominal manifest schema: {manifest_path}")
    if manifest.get("corpus_manifest_sha256") != sha256_file(corpus_manifest_path):
        raise ComparisonError(
            "Nominal summary corpus does not match the calibration corpus"
        )
    if manifest.get("libjxl_revision") != PINNED_LIBJXL_REVISION:
        raise ComparisonError("Nominal summary uses a different libjxl revision")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ComparisonError("Nominal manifest has no parameter record")
    target_distance = _positive_distance(
        parameters.get("distance"), "nominal GJXL distance"
    )
    effort = parameters.get("effort")
    if isinstance(effort, bool) or not isinstance(effort, int):
        raise ComparisonError("Nominal libjxl effort is invalid")

    scores: dict[str, list[float]] = {}
    for validation in summary.get("validations", []):
        if validation.get("encoder") != "gjxl":
            continue
        name = validation.get("input")
        if not isinstance(name, str):
            raise ComparisonError("Nominal GJXL validation has no input name")
        scores.setdefault(name, []).append(
            _finite_number(
                validation.get("butteraugli"),
                f"nominal Butteraugli score for {name}",
            )
        )

    expected = {entry["name"] for entry in entries}
    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        extra = sorted(set(scores) - expected)
        raise ComparisonError(
            "Nominal GJXL validation inputs do not match the corpus: "
            f"missing={missing}, extra={extra}"
        )
    targets = {}
    for name, values in scores.items():
        if max(values) - min(values) > 1e-9:
            raise ComparisonError(
                f"Nominal GJXL score differs across configurations for {name}"
            )
        targets[name] = values[0]
    return targets, {
        "summary": str(nominal_summary_path),
        "summary_sha256": sha256_file(nominal_summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gjxl_revision": manifest.get("gjxl_revision"),
        "target_gjxl_distance": target_distance,
        "effort": effort,
    }


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
            result["absolute_error"] / target_score
            if target_score > 0.0
            else math.inf
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
            minimum_distance
            + index * (maximum_distance - minimum_distance) / 8.0
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


def load_quality_calibration(
    path: Path,
    corpus_manifest: Path,
    entries: list[dict[str, Any]],
    *,
    target_distance: float,
    effort: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    path = path.resolve()
    document = load_json(path)
    if (
        document.get("schema_version") != CALIBRATION_SCHEMA_VERSION
        or document.get("kind") != "libjxl-butteraugli-calibration"
    ):
        raise ComparisonError(f"Unexpected quality calibration schema: {path}")
    if document.get("libjxl_revision") != PINNED_LIBJXL_REVISION:
        raise ComparisonError("Quality calibration uses a different libjxl revision")
    if document.get("corpus_manifest_sha256") != sha256_file(corpus_manifest):
        raise ComparisonError("Quality calibration corpus does not match this run")
    parameters = document.get("parameters")
    if not isinstance(parameters, dict):
        raise ComparisonError("Quality calibration has no parameter record")
    calibrated_target = _positive_distance(
        parameters.get("target_gjxl_distance"), "calibrated GJXL distance"
    )
    if not math.isclose(calibrated_target, target_distance, rel_tol=0.0, abs_tol=1e-9):
        raise ComparisonError(
            "Quality calibration GJXL distance does not match --distance"
        )
    if parameters.get("effort") != effort:
        raise ComparisonError("Quality calibration effort does not match --effort")
    tolerance = _finite_number(parameters.get("tolerance"), "calibration tolerance")
    if tolerance <= 0.0:
        raise ComparisonError("Calibration tolerance must be positive")
    maximum_relative_error = _finite_number(
        parameters.get("maximum_relative_error"),
        "maximum calibration relative error",
    )
    if maximum_relative_error < 0.0:
        raise ComparisonError("Maximum calibration relative error must be nonnegative")

    records = document.get("inputs")
    if not isinstance(records, list):
        raise ComparisonError("Quality calibration inputs must be a list")
    expected = {entry["name"] for entry in entries}
    entries_by_name = {entry["name"]: entry for entry in entries}
    distances: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("input"), str):
            raise ComparisonError("Quality calibration input record is invalid")
        name = record["input"]
        if name in distances:
            raise ComparisonError(f"Duplicate quality calibration input: {name}")
        if name not in entries_by_name:
            raise ComparisonError(f"Unexpected quality calibration input: {name}")
        if record.get("canonical_sha256") != entries_by_name[name].get(
            "canonical_sha256"
        ):
            raise ComparisonError(
                f"Quality calibration canonical input changed for {name}"
            )
        record_target_distance = _positive_distance(
            record.get("target_gjxl_distance"),
            f"target GJXL distance for {name}",
        )
        if not math.isclose(
            record_target_distance, target_distance, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ComparisonError(
                f"Quality calibration target distance changed for {name}"
            )
        distance = _positive_distance(
            record.get("libjxl_distance"), f"calibrated libjxl distance for {name}"
        )
        target_score = _finite_number(
            record.get("target_butteraugli"), f"target score for {name}"
        )
        achieved_score = _finite_number(
            record.get("achieved_butteraugli"), f"achieved score for {name}"
        )
        error = _finite_number(
            record.get("absolute_error"), f"calibration error for {name}"
        )
        calculated_error = abs(target_score - achieved_score)
        if not math.isclose(error, calculated_error, rel_tol=0.0, abs_tol=1e-12):
            raise ComparisonError(
                f"Quality calibration error is inconsistent for {name}"
            )
        relative_error = error / target_score if target_score > 0.0 else math.inf
        recorded_relative_error = _finite_number(
            record.get("relative_error"), f"calibration relative error for {name}"
        )
        if not math.isclose(
            relative_error, recorded_relative_error, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ComparisonError(
                f"Quality calibration relative error is inconsistent for {name}"
            )
        match_kind = record.get("match_kind")
        if match_kind not in {
            "within-absolute-tolerance",
            "boundary-limited-relative-tolerance",
            "quantized-relative-tolerance",
        }:
            raise ComparisonError(
                f"Quality calibration match kind is invalid for {name}"
            )
        if (
            match_kind == "within-absolute-tolerance"
            and error > tolerance
        ):
            raise ComparisonError(
                f"Quality calibration absolute match is inconsistent for {name}"
            )
        if (
            match_kind != "within-absolute-tolerance"
            and relative_error > maximum_relative_error
        ):
            raise ComparisonError(
                f"Quality calibration relative match is inconsistent for {name}"
            )
        if error < 0.0 or (
            error > tolerance and relative_error > maximum_relative_error
        ):
            raise ComparisonError(
                f"Quality calibration for {name} exceeds its tolerance"
            )
        distances[name] = distance
    if set(distances) != expected:
        missing = sorted(expected - set(distances))
        extra = sorted(set(distances) - expected)
        raise ComparisonError(
            "Quality calibration inputs do not match the corpus: "
            f"missing={missing}, extra={extra}"
        )
    return distances, {
        "path": str(path),
        "sha256": sha256_file(path),
        "tolerance": tolerance,
        "maximum_relative_error": maximum_relative_error,
        "source_gjxl_revision": document.get("source_gjxl_revision"),
    }


def calibrate_quality(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    corpus_manifest = args.corpus_manifest.resolve()
    entries = validate_corpus(corpus_manifest)
    targets, nominal = load_nominal_quality_targets(
        args.nominal_summary, corpus_manifest, entries
    )
    if nominal["effort"] != args.effort:
        raise ComparisonError(
            "Calibration effort must match the nominal comparison effort"
        )

    binaries = {
        "libjxl_benchmark": args.libjxl_benchmark.resolve(),
        "djxl": args.djxl.resolve(),
        "butteraugli": args.butteraugli.resolve(),
    }
    for name, path in binaries.items():
        if not path.is_file():
            raise ComparisonError(f"Required {name} file does not exist: {path}")
    args.libjxl_benchmark = binaries["libjxl_benchmark"]
    args.djxl = binaries["djxl"]
    args.butteraugli = binaries["butteraugli"]

    gjxl_revision = git_output(repo, "rev-parse", "HEAD")
    libjxl_source = repo / "third_party/libjxl"
    libjxl_revision = git_output(libjxl_source, "rev-parse", "HEAD")
    if libjxl_revision != PINNED_LIBJXL_REVISION:
        raise ComparisonError(
            f"libjxl revision is {libjxl_revision}, expected {PINNED_LIBJXL_REVISION}"
        )

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = (
        args.output_root / f"{timestamp}-{gjxl_revision[:12]}"
    ).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    raw_dir = directory / "raw"
    candidates_dir = directory / "candidates"
    raw_dir.mkdir()
    candidates_dir.mkdir()

    git_diff = run_capture(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"], check=False
    ).stdout.encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "libjxl-butteraugli-calibration-run",
        "created_utc": timestamp,
        "repo": str(repo),
        "gjxl_revision": gjxl_revision,
        "gjxl_dirty": bool(git_output(repo, "status", "--porcelain")),
        "gjxl_worktree_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        "libjxl_revision": libjxl_revision,
        "corpus_manifest": str(corpus_manifest),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "nominal_quality_source": nominal,
        "parameters": {
            "target_gjxl_distance": nominal["target_gjxl_distance"],
            "effort": args.effort,
            "libjxl_thread_count": args.num_threads,
            "warmups": args.warmups,
            "samples": args.samples,
            "minimum_distance": args.minimum_distance,
            "maximum_distance": args.maximum_distance,
            "initial_distance": args.initial_distance,
            "tolerance": args.tolerance,
            "maximum_relative_error": args.maximum_relative_error,
            "maximum_evaluations": args.maximum_evaluations,
        },
        "binaries": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in binaries.items()
        },
        "host": host_manifest(),
        "argv": sys.argv,
        "processes": [],
        "evaluations": [],
        "calibrations": [],
    }
    write_json_atomic(directory / "manifest.json", manifest)

    calibrated_inputs = []
    for entry in entries:
        image = Path(entry["resolved_path"])
        slug = safe_name(entry["name"])
        target = targets[entry["name"]]
        input_candidates = candidates_dir / slug
        input_candidates.mkdir()

        def evaluate(distance: float) -> dict[str, Any]:
            candidate_index = sum(
                item["input"] == entry["name"]
                for item in manifest["evaluations"]
            )
            candidate_name = f"candidate{candidate_index:02d}"
            raw = raw_dir / f"{slug}-{candidate_name}-libjxl.json"
            codestream = input_candidates / f"{candidate_name}.jxl"
            command = libjxl_command(
                args,
                image,
                raw,
                args.num_threads,
                codestream,
                distance=distance,
            )
            process_name = f"{slug}-calibration-{candidate_name}-libjxl"
            record_process(command, process_name, directory, manifest)
            validation = validate_output(
                args,
                entry,
                "libjxl",
                f"calibration-{candidate_name}",
                codestream,
                directory,
                manifest,
            )
            raw_document = load_json(raw)
            encoded_sizes = {
                sample["encoded_bytes"] for sample in raw_document["samples"]
            }
            if len(encoded_sizes) != 1:
                raise ComparisonError(
                    f"Calibration encoded size is unstable for {entry['name']}"
                )
            result = {
                "input": entry["name"],
                "candidate_index": candidate_index,
                "distance": distance,
                "encoded_requested_distance": raw_document["requested_distance"],
                "butteraugli": validation["butteraugli"],
                "absolute_error": abs(validation["butteraugli"] - target),
                "encoded_bytes": encoded_sizes.pop(),
                "raw": str(raw.relative_to(directory)),
                "codestream": validation["codestream"],
                "codestream_sha256": validation["codestream_sha256"],
                "decoded": validation["decoded"],
                "decoded_sha256": validation["decoded_sha256"],
                "retained": True,
            }
            manifest["evaluations"].append(result)
            write_json_atomic(directory / "manifest.json", manifest)
            return result

        selected, evaluations = calibrate_distance(
            target,
            evaluate,
            minimum_distance=args.minimum_distance,
            maximum_distance=args.maximum_distance,
            initial_distance=args.initial_distance,
            tolerance=args.tolerance,
            maximum_evaluations=args.maximum_evaluations,
            maximum_relative_error=args.maximum_relative_error,
        )
        selected_index = selected["candidate_index"]
        for evaluation in evaluations:
            retained = evaluation["candidate_index"] == selected_index
            evaluation["retained"] = retained
            manifest_evaluation = next(
                item
                for item in manifest["evaluations"]
                if item["input"] == entry["name"]
                and item["candidate_index"] == evaluation["candidate_index"]
            )
            manifest_evaluation["retained"] = retained
            if not retained:
                (directory / evaluation["codestream"]).unlink()
                (directory / evaluation["decoded"]).unlink()

        calibration = {
            "input": entry["name"],
            "category": entry.get("category", "unspecified"),
            "canonical_sha256": entry["canonical_sha256"],
            "target_gjxl_distance": nominal["target_gjxl_distance"],
            "target_butteraugli": target,
            "libjxl_distance": selected["distance"],
            "achieved_butteraugli": selected["butteraugli"],
            "absolute_error": selected["absolute_error"],
            "relative_error": selected["relative_error"],
            "match_kind": selected["match_kind"],
            "encoded_bytes": selected["encoded_bytes"],
            "selected_candidate_index": selected_index,
            "evaluation_count": len(evaluations),
        }
        calibrated_inputs.append(calibration)
        manifest["calibrations"].append(calibration)
        write_json_atomic(directory / "manifest.json", manifest)

    calibration_document = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "libjxl-butteraugli-calibration",
        "created_utc": timestamp,
        "source_gjxl_revision": nominal["gjxl_revision"],
        "calibration_tool_revision": gjxl_revision,
        "libjxl_revision": libjxl_revision,
        "corpus_manifest": str(corpus_manifest),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "nominal_quality_source": nominal,
        "parameters": manifest["parameters"],
        "inputs": calibrated_inputs,
    }
    write_json_atomic(directory / "calibration.json", calibration_document)
    manifest["calibration"] = {
        "path": "calibration.json",
        "sha256": sha256_file(directory / "calibration.json"),
    }
    manifest["host_end"] = host_manifest()
    write_json_atomic(directory / "manifest.json", manifest)
    print(directory)


def capture_profile(
    args: argparse.Namespace,
    command: list[str],
    profile: Path,
    name: str,
    directory: Path,
    manifest: dict[str, Any],
) -> None:
    profile_command = [
        str(args.samply),
        "record",
        "--save-only",
        "--unstable-presymbolicate",
        "--rate",
        str(args.samply_rate),
        "--output",
        str(profile),
        *command,
    ]
    record_process(profile_command, name, directory, manifest)


def run_comparison(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    corpus_manifest = args.corpus_manifest.resolve()
    entries = validate_corpus(corpus_manifest)
    binaries = {
        "gjxl_benchmark": args.gjxl_benchmark.resolve(),
        "libjxl_benchmark": args.libjxl_benchmark.resolve(),
        "djxl": args.djxl.resolve(),
        "butteraugli": args.butteraugli.resolve(),
    }
    if args.metallib is not None:
        args.metallib = args.metallib.resolve()
        binaries["metallib"] = args.metallib
    if args.capture_samply:
        args.samply = Path(
            find_executable(str(args.samply), "Samply profiler")
        ).resolve()
        args.samply_analyzer = args.samply_analyzer.resolve()
        binaries["samply"] = args.samply
        binaries["samply_analyzer"] = args.samply_analyzer
    for name, path in binaries.items():
        if not path.is_file():
            raise ComparisonError(f"Required {name} file does not exist: {path}")
    args.gjxl_benchmark = binaries["gjxl_benchmark"]
    args.libjxl_benchmark = binaries["libjxl_benchmark"]
    args.djxl = binaries["djxl"]
    args.butteraugli = binaries["butteraugli"]

    quality_distances = {entry["name"]: args.distance for entry in entries}
    quality_calibration = None
    if args.quality_map is not None:
        quality_distances, quality_calibration = load_quality_calibration(
            args.quality_map,
            corpus_manifest,
            entries,
            target_distance=args.distance,
            effort=args.effort,
        )
    entries = select_corpus_entries(entries, args.inputs)

    gjxl_revision = git_output(repo, "rev-parse", "HEAD")
    libjxl_source = repo / "third_party/libjxl"
    libjxl_revision = git_output(libjxl_source, "rev-parse", "HEAD")
    if libjxl_revision != PINNED_LIBJXL_REVISION:
        raise ComparisonError(
            f"libjxl revision is {libjxl_revision}, expected {PINNED_LIBJXL_REVISION}"
        )
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = (args.output_root / f"{timestamp}-{gjxl_revision[:12]}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    raw_dir = directory / "raw"
    streams_dir = directory / "codestreams"
    profiles_dir = directory / "profiles"
    raw_dir.mkdir()
    streams_dir.mkdir()
    if args.capture_samply:
        profiles_dir.mkdir()

    git_diff = run_capture(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"], check=False
    ).stdout.encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": timestamp,
        "repo": str(repo),
        "gjxl_revision": gjxl_revision,
        "gjxl_dirty": bool(git_output(repo, "status", "--porcelain")),
        "gjxl_worktree_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        "libjxl_revision": libjxl_revision,
        "corpus_manifest": str(corpus_manifest),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "inputs": entries,
        "input_adapters": {
            "gjxl": {
                "layout": "planar linear-sRGB float32",
                "row_stride_bytes": "width * sizeof(float) per plane",
            },
            "libjxl": {
                "layout": "interleaved linear-sRGB float32 RGB",
                "row_stride_bytes": "width * 3 * sizeof(float)",
            },
        },
        "parameters": {
            "distance": args.distance,
            "quality_view": (
                "matched" if quality_calibration is not None else "nominal-distance"
            ),
            "effort": args.effort,
            "warmups": args.warmups,
            "samples": args.samples,
            "pairs": args.pairs,
            "gjxl_implementation": args.gjxl_implementation,
            "configurations": args.configuration,
            "input_filter": args.inputs,
        },
        "binaries": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in binaries.items()
        },
        "host": host_manifest(),
        "argv": sys.argv,
        "processes": [],
        "validations": [],
        "quality_matches": [],
        "profiles": [],
    }
    if quality_calibration is not None:
        manifest["quality_calibration"] = quality_calibration
        manifest["libjxl_distances"] = {
            entry["name"]: quality_distances[entry["name"]]
            for entry in entries
        }
    if args.capture_samply:
        manifest["parameters"]["samply_rate_hz"] = args.samply_rate
        manifest["parameters"]["profile_samples"] = args.profile_samples
        manifest["parameters"]["profile_encode_count"] = (
            1 + args.warmups + args.profile_samples
        )
        manifest["parameters"]["minimum_symbol_resolution_percent"] = (
            PROFILE_MINIMUM_RESOLUTION_PERCENT
        )
    write_json_atomic(directory / "manifest.json", manifest)

    logical_cpus = max(os.cpu_count() or 1, 1)
    production_threads = min(8, logical_cpus)
    configurations = []
    if args.configuration in {"serial", "both"}:
        configurations.append(("serial", 1, 0))
    if args.configuration in {"production", "both"}:
        configurations.append(("production", 0, production_threads))
    manifest["thread_policies"] = [
        {
            "configuration": name,
            "gjxl_serializer_workers": gjxl_workers,
            "libjxl_worker_threads": libjxl_threads,
            "meaning": (
                "one logical serializer participant"
                if name == "serial"
                else "GJXL automatic policy matched to its effective eight-worker cap"
            ),
        }
        for name, gjxl_workers, libjxl_threads in configurations
    ]
    write_json_atomic(directory / "manifest.json", manifest)

    rows = []
    profile_rows = []
    for configuration, gjxl_workers, libjxl_threads in configurations:
        for entry in entries:
            image = Path(entry["resolved_path"])
            slug = safe_name(entry["name"])
            encoder_distances = {
                "gjxl": args.distance,
                "libjxl": quality_distances[entry["name"]],
            }
            output_paths = {
                "gjxl": streams_dir / f"{slug}-{configuration}-gjxl.jxl",
                "libjxl": streams_dir / f"{slug}-{configuration}-libjxl.jxl",
            }
            for pair in range(args.pairs):
                order = encoder_order(pair)
                for encoder in order:
                    raw = raw_dir / f"{slug}-{configuration}-pair{pair}-{encoder}.json"
                    output = output_paths[encoder] if pair == 0 else None
                    if encoder == "gjxl":
                        command = gjxl_command(
                            args,
                            image,
                            raw,
                            gjxl_workers,
                            output,
                            distance=encoder_distances[encoder],
                        )
                    else:
                        command = libjxl_command(
                            args,
                            image,
                            raw,
                            libjxl_threads,
                            output,
                            distance=encoder_distances[encoder],
                        )
                    name = f"{slug}-{configuration}-pair{pair}-{encoder}"
                    record_process(command, name, directory, manifest)
                    rows.append(
                        median_process_row(
                            encoder, configuration, entry, raw
                        )
                    )
                    rows[-1]["raw_path"] = str(raw.relative_to(directory))
            input_validations = {}
            for encoder, path in output_paths.items():
                validation = validate_output(
                    args,
                    entry,
                    encoder,
                    configuration,
                    path,
                    directory,
                    manifest,
                    encoder_distances[encoder],
                )
                input_validations[encoder] = validation
                manifest["validations"].append(validation)
                write_json_atomic(directory / "manifest.json", manifest)

            if quality_calibration is not None:
                score_delta = abs(
                    input_validations["gjxl"]["butteraugli"]
                    - input_validations["libjxl"]["butteraugli"]
                )
                gjxl_score = input_validations["gjxl"]["butteraugli"]
                relative_delta = (
                    score_delta / gjxl_score if gjxl_score > 0.0 else math.inf
                )
                within_absolute = score_delta <= quality_calibration["tolerance"]
                within_relative = (
                    relative_delta
                    <= quality_calibration["maximum_relative_error"]
                )
                quality_match = {
                    "input": entry["name"],
                    "configuration": configuration,
                    "gjxl_butteraugli": gjxl_score,
                    "libjxl_butteraugli": input_validations["libjxl"]["butteraugli"],
                    "absolute_difference": score_delta,
                    "relative_difference": relative_delta,
                    "tolerance": quality_calibration["tolerance"],
                    "maximum_relative_error": quality_calibration[
                        "maximum_relative_error"
                    ],
                    "match_kind": (
                        "within-absolute-tolerance"
                        if within_absolute
                        else "within-relative-tolerance"
                    ),
                    "within_tolerance": within_absolute or within_relative,
                }
                manifest["quality_matches"].append(quality_match)
                write_json_atomic(directory / "manifest.json", manifest)
                if not quality_match["within_tolerance"]:
                    raise ComparisonError(
                        f"Matched-quality validation failed for {entry['name']} "
                        f"({configuration}): difference={score_delta:.9g}, "
                        f"relative={relative_delta:.3%}, "
                        f"absolute_tolerance="
                        f"{quality_calibration['tolerance']:.9g}, "
                        f"relative_tolerance="
                        f"{quality_calibration['maximum_relative_error']:.3%}"
                    )

            if args.capture_samply:
                for encoder in ("gjxl", "libjxl"):
                    raw = raw_dir / f"{slug}-{configuration}-{encoder}-profile.json"
                    profile = profiles_dir / f"{slug}-{configuration}-{encoder}.json.gz"
                    if encoder == "gjxl":
                        command = gjxl_command(
                            args,
                            image,
                            raw,
                            gjxl_workers,
                            None,
                            args.profile_samples,
                            encoder_distances[encoder],
                        )
                    else:
                        command = libjxl_command(
                            args,
                            image,
                            raw,
                            libjxl_threads,
                            None,
                            args.profile_samples,
                            encoder_distances[encoder],
                        )
                    capture_profile(
                        args,
                        command,
                        profile,
                        f"{slug}-{configuration}-{encoder}-samply",
                        directory,
                        manifest,
                    )
                    symbols = Path(
                        str(profile)[: -len(".json.gz")] + ".json.syms.json"
                    )
                    if not profile.is_file() or not symbols.is_file():
                        raise ComparisonError(
                            f"Samply capture or symbol sidecar is missing: {profile}"
                        )
                    profile_record = {
                        "input": entry["name"],
                        "category": entry.get("category", "unspecified"),
                        "configuration": configuration,
                        "encoder": encoder,
                        "profile": str(profile.relative_to(directory)),
                        "profile_sha256": sha256_file(profile),
                        "symbols": str(symbols.relative_to(directory)),
                        "symbols_sha256": sha256_file(symbols),
                    }
                    manifest["profiles"].append(profile_record)
                    write_json_atomic(directory / "manifest.json", manifest)
                    for output_format, suffix in (("json", "json"), ("markdown", "md")):
                        analysis_output = profiles_dir / (
                            f"{slug}-{configuration}-{encoder}-neutral.{suffix}"
                        )
                        record_process(
                            [
                                sys.executable,
                                str(args.samply_analyzer),
                                str(profile),
                                "--format",
                                output_format,
                                "--minimum-resolution-percent",
                                str(PROFILE_MINIMUM_RESOLUTION_PERCENT),
                                "--output",
                                str(analysis_output),
                            ],
                            f"{slug}-{configuration}-{encoder}-analyze-{output_format}",
                            directory,
                            manifest,
                        )
                        if output_format == "json":
                            analysis = load_json(analysis_output)
                            profile_encode_count = (
                                1 + args.warmups + args.profile_samples
                            )
                            source_pixels = entry["width"] * entry["height"]
                            normalized_stages = [
                                {
                                    **stage,
                                    "sampled_cpu_milliseconds_per_encode": (
                                        stage["cpu_delta_us"]
                                        / 1000.0
                                        / profile_encode_count
                                    ),
                                    "sampled_cpu_nanoseconds_per_source_pixel": (
                                        stage["cpu_delta_us"]
                                        * 1000.0
                                        / profile_encode_count
                                        / source_pixels
                                    ),
                                }
                                for stage in analysis["neutral_stages"]
                            ]
                            profile_rows.append(
                                {
                                    **profile_record,
                                    "analysis": str(
                                        analysis_output.relative_to(directory)
                                    ),
                                    "analysis_sha256": sha256_file(analysis_output),
                                    "sample_count": analysis["summary"]["sample_count"],
                                    "sampled_cpu_us": analysis["summary"][
                                        "cpu_delta_us"
                                    ],
                                    "profile_encode_count": profile_encode_count,
                                    "sampled_cpu_milliseconds_per_encode": (
                                        analysis["summary"]["cpu_delta_us"]
                                        / 1000.0
                                        / profile_encode_count
                                    ),
                                    "sampled_cpu_nanoseconds_per_source_pixel": (
                                        analysis["summary"]["cpu_delta_us"]
                                        * 1000.0
                                        / profile_encode_count
                                        / source_pixels
                                    ),
                                    "resolved_leaf_cpu_percent": analysis["summary"][
                                        "resolved_leaf_cpu_percent"
                                    ],
                                    "neutral_stages": normalized_stages,
                                }
                            )

    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "timing_claim": "unprofiled complete-encode wall time",
        "process_rows": rows,
        "aggregate_rows": aggregate_rows(rows),
        "validations": manifest["validations"],
        "quality_matches": manifest["quality_matches"],
        "profile_rows": profile_rows,
    }
    write_json_atomic(directory / "summary.json", summary)
    manifest["host_end"] = host_manifest()
    write_json_atomic(directory / "manifest.json", manifest)
    print(directory)


def _load_result_directory(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = path.resolve()
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    if not root.is_dir() or not manifest_path.is_file() or not summary_path.is_file():
        raise ComparisonError(f"Comparison result is incomplete: {root}")
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    if (
        manifest.get("schema_version") != RESULT_SCHEMA_VERSION
        or summary.get("schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise ComparisonError(f"Unexpected comparison result schema: {root}")
    return root, manifest, summary


def _verify_result_files(
    root: Path, manifest: dict[str, Any], summary: dict[str, Any]
) -> int:
    checked = 0
    for process in manifest.get("processes", []):
        if process.get("exit_code") != 0:
            raise ComparisonError(
                f"Comparison result contains a failed process: {root}"
            )
    for validation in manifest.get("validations", []):
        for field, hash_field in (
            ("codestream", "codestream_sha256"),
            ("decoded", "decoded_sha256"),
        ):
            path = root / validation[field]
            if not path.is_file() or sha256_file(path) != validation[hash_field]:
                raise ComparisonError(f"Validation artifact hash mismatch: {path}")
            checked += 1
    for profile in manifest.get("profiles", []):
        for field, hash_field in (
            ("profile", "profile_sha256"),
            ("symbols", "symbols_sha256"),
        ):
            path = root / profile[field]
            if not path.is_file() or sha256_file(path) != profile[hash_field]:
                raise ComparisonError(f"Profile artifact hash mismatch: {path}")
            checked += 1
    for row in summary.get("profile_rows", []):
        path = root / row["analysis"]
        if not path.is_file() or sha256_file(path) != row["analysis_sha256"]:
            raise ComparisonError(f"Profile analysis hash mismatch: {path}")
        checked += 1
    return checked


def bundle_phase1(args: argparse.Namespace) -> None:
    nominal_root, nominal_manifest, nominal_summary = _load_result_directory(
        args.nominal_result
    )
    matched_root, matched_manifest, matched_summary = _load_result_directory(
        args.matched_result
    )
    profile_root, profile_manifest, profile_summary = _load_result_directory(
        args.profile_result
    )
    calibration_root = args.calibration_result.resolve()
    calibration_path = calibration_root / "calibration.json"
    calibration_manifest_path = calibration_root / "manifest.json"
    if not calibration_path.is_file() or not calibration_manifest_path.is_file():
        raise ComparisonError(
            f"Calibration result is incomplete: {calibration_root}"
        )
    calibration = load_json(calibration_path)
    calibration_manifest = load_json(calibration_manifest_path)
    if (
        calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION
        or calibration.get("kind") != "libjxl-butteraugli-calibration"
    ):
        raise ComparisonError("Unexpected Phase 1 calibration schema")

    results = (
        ("nominal", nominal_root, nominal_manifest, nominal_summary),
        ("matched", matched_root, matched_manifest, matched_summary),
        ("profiles", profile_root, profile_manifest, profile_summary),
    )
    corpus_hashes = {
        manifest.get("corpus_manifest_sha256")
        for _, _, manifest, _ in results
    }
    corpus_hashes.add(calibration.get("corpus_manifest_sha256"))
    if len(corpus_hashes) != 1 or None in corpus_hashes:
        raise ComparisonError("Phase 1 artifacts do not share one corpus manifest")
    revisions = {
        manifest.get("libjxl_revision") for _, _, manifest, _ in results
    }
    revisions.add(calibration.get("libjxl_revision"))
    if revisions != {PINNED_LIBJXL_REVISION}:
        raise ComparisonError("Phase 1 artifacts do not share pinned libjxl")
    nominal_quality_view = nominal_manifest.get("parameters", {}).get(
        "quality_view", "nominal-distance"
    )
    if (
        nominal_quality_view != "nominal-distance"
        or "quality_calibration" in nominal_manifest
    ):
        raise ComparisonError("Nominal result is not a nominal-distance run")
    if matched_manifest.get("parameters", {}).get("quality_view") != "matched":
        raise ComparisonError("Matched result is not a matched-quality run")
    if profile_manifest.get("parameters", {}).get("quality_view") != "matched":
        raise ComparisonError("Profile result is not matched quality")
    if len(nominal_manifest.get("inputs", [])) != 38:
        raise ComparisonError("Nominal result does not cover all 38 inputs")
    if len(matched_manifest.get("inputs", [])) != 38:
        raise ComparisonError("Matched result does not cover all 38 inputs")
    if len(calibration.get("inputs", [])) != 38:
        raise ComparisonError("Calibration does not cover all 38 inputs")

    matches = matched_manifest.get("quality_matches", [])
    if len(matches) != 76 or any(not row.get("within_tolerance") for row in matches):
        raise ComparisonError("Matched result does not contain 76 accepted matches")
    profile_rows = profile_summary.get("profile_rows", [])
    if len(profile_rows) != 20 or len(profile_manifest.get("profiles", [])) != 20:
        raise ComparisonError("Profile result does not contain 20 captures")
    expected_categories = {
        "photographic-4k",
        "photographic-1080p",
        "kodak-continuity",
        "padded-stress-1080p",
        "padded-stress-4k",
    }
    if {row.get("category") for row in profile_rows} != expected_categories:
        raise ComparisonError("Profile result does not cover every workload group")
    if {row.get("configuration") for row in profile_rows} != {
        "serial",
        "production",
    } or {row.get("encoder") for row in profile_rows} != {"gjxl", "libjxl"}:
        raise ComparisonError("Profile result does not cover both policies and encoders")

    stage_names = {
        "coefficient_and_token_preparation",
        "entropy_model_construction",
        "token_and_model_emission",
        "framing_and_assembly",
        "other_encoder_and_runtime",
        "unresolved_or_missing_stack",
    }
    minimum_resolution = 100.0
    for row in profile_rows:
        stages = row.get("neutral_stages", [])
        if {stage.get("stage") for stage in stages} != stage_names:
            raise ComparisonError("Profile row has an incomplete neutral-stage map")
        if sum(stage["cpu_delta_us"] for stage in stages) != row["sampled_cpu_us"]:
            raise ComparisonError("Profile neutral stages are not mutually exhaustive")
        minimum_resolution = min(
            minimum_resolution, row["resolved_leaf_cpu_percent"]
        )
    required_resolution = profile_manifest.get("parameters", {}).get(
        "minimum_symbol_resolution_percent"
    )
    if required_resolution is None or minimum_resolution < required_resolution:
        raise ComparisonError("Profile result fails its symbol-resolution gate")

    checked_files = sum(
        _verify_result_files(root, manifest, summary)
        for _, root, manifest, summary in results
    )
    if any(
        process.get("exit_code") != 0
        for process in calibration_manifest.get("processes", [])
    ):
        raise ComparisonError("Calibration contains a failed process")
    for evaluation in calibration_manifest.get("evaluations", []):
        if not evaluation.get("retained"):
            continue
        for field, hash_field in (
            ("codestream", "codestream_sha256"),
            ("decoded", "decoded_sha256"),
        ):
            path = calibration_root / evaluation[field]
            if not path.is_file() or sha256_file(path) != evaluation[hash_field]:
                raise ComparisonError(f"Calibration artifact hash mismatch: {path}")
            checked_files += 1

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact_sources = {
        "nominal": nominal_root,
        "calibration": calibration_root,
        "matched": matched_root,
        "profiles": profile_root,
    }
    for name, source in artifact_sources.items():
        (output / name).symlink_to(
            os.path.relpath(source, output), target_is_directory=True
        )

    neutral = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "timing_semantics": "sampled-thread-cpu-attribution-not-stage-wall-time",
        "source": str(profile_root / "summary.json"),
        "source_sha256": sha256_file(profile_root / "summary.json"),
        "parameters": {
            key: profile_manifest["parameters"][key]
            for key in (
                "samply_rate_hz",
                "profile_samples",
                "profile_encode_count",
                "minimum_symbol_resolution_percent",
            )
        },
        "profile_rows": profile_rows,
    }
    write_json_atomic(output / "neutral-comparison.json", neutral)

    artifact_index = {}
    for name, source in artifact_sources.items():
        files = {}
        for filename in ("manifest.json", "summary.json", "calibration.json"):
            path = source / filename
            if path.is_file():
                files[filename] = sha256_file(path)
        artifact_index[name] = {
            "path": str(source),
            "link": name,
            "files": files,
        }
    index = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "gjxl-libjxl-phase1-bundle",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "corpus_manifest_sha256": corpus_hashes.pop(),
        "libjxl_revision": PINNED_LIBJXL_REVISION,
        "artifacts": artifact_index,
        "verification": {
            "rehash_count": checked_files,
            "matched_quality_records": len(matches),
            "profile_count": len(profile_rows),
            "minimum_weighted_symbol_resolution_percent": minimum_resolution,
            "classifier": "ordered-mutually-exclusive-neutral-stages",
            "unprofiled_timing_source": "matched/summary.json",
            "sampled_attribution_source": "profiles/summary.json",
        },
        "neutral_comparison": {
            "path": "neutral-comparison.json",
            "sha256": sha256_file(output / "neutral-comparison.json"),
        },
    }
    write_json_atomic(output / "phase1-index.json", index)

    reproduce = shlex.join(str(value) for value in profile_manifest.get("argv", []))
    readme = f"""# GJXL/libjxl Phase 1 comparison artifact

This directory indexes the retained no-libjxl-source-change Phase 1 pilot.
The linked directories are never-overwritten source artifacts; their recorded
manifests, outputs, profiles, sidecars, and analyses were rehashed while this
bundle was created.

## Sources of record

- `nominal/`: full 38-input nominal-distance serial and production timing.
- `calibration/`: per-input matched-quality search and selected outputs.
- `matched/`: full 38-input matched-quality serial and production timing.
- `profiles/`: representative five-group serial and production Samply capture.
- `neutral-comparison.json`: machine-readable mutually exclusive stage table.
- `phase1-index.json`: paths, hashes, and completion evidence.

## Interpretation

Unprofiled complete-encode timing comes from `matched/summary.json`. Sampled
stage values come from `profiles/summary.json` and are aggregate thread CPU,
not stage wall time. Host load was high, so absolute latency remains diagnostic.
Photographs, Kodak continuity images, and padded stress inputs remain separate.

The pilot supports prioritizing GJXL entropy-model construction. It does not
support an exact claim about libjxl stage wall-clock latency; direct libjxl
instrumentation remains optional for that future question.

## Profile reproduction

```sh
{reproduce}
```
"""
    write_text_atomic(output / "README.md", readme)
    print(output)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-corpus")
    fetch.add_argument("--source-manifest", type=Path, required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--timeout", type=positive_int, default=60)
    fetch.add_argument(
        "--gjxl-benchmark",
        type=Path,
        default=repo_default / "build/release/gjxl_encoding_benchmark",
    )
    fetch.set_defaults(function=fetch_corpus_sources)

    prepare = subparsers.add_parser("prepare-corpus")
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument(
        "--source-root",
        type=Path,
        help="resolve manifest input paths under this directory",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--magick", default=os.environ.get("GJXL_MAGICK", "magick"))
    prepare.add_argument("--background", choices=("white", "black"), default="white")
    prepare.set_defaults(function=prepare_corpus)

    build = subparsers.add_parser("build-libjxl")
    build.add_argument("--repo", type=Path, default=repo_default)
    build.add_argument(
        "--build-root",
        type=Path,
        default=repo_default / "build/libjxl-comparison",
    )
    build.add_argument("--generator", default="Ninja")
    build.add_argument("--jobs", type=positive_int, default=8)
    build.set_defaults(function=build_libjxl)

    comparison_build = repo_default / "build/libjxl-comparison"
    calibrate = subparsers.add_parser("calibrate-quality")
    calibrate.add_argument("--repo", type=Path, default=repo_default)
    calibrate.add_argument("--corpus-manifest", type=Path, required=True)
    calibrate.add_argument("--nominal-summary", type=Path, required=True)
    calibrate.add_argument(
        "--libjxl-benchmark",
        type=Path,
        default=comparison_build / "harness/gjxl_libjxl_comparison_benchmark",
    )
    calibrate.add_argument(
        "--djxl", type=Path, default=comparison_build / "libjxl/tools/djxl"
    )
    calibrate.add_argument(
        "--butteraugli",
        type=Path,
        default=comparison_build / "libjxl/tools/butteraugli_main",
    )
    calibrate.add_argument(
        "--output-root",
        type=Path,
        default=repo_default / "logs/libjxl-calibration",
    )
    calibrate.add_argument("--effort", type=int, choices=range(1, 11), default=7)
    calibrate.add_argument("--num-threads", type=positive_int, default=8)
    calibrate.add_argument("--warmups", type=int, default=0)
    calibrate.add_argument("--samples", type=positive_int, default=1)
    calibrate.add_argument("--minimum-distance", type=float, default=0.05)
    calibrate.add_argument("--maximum-distance", type=float, default=2.0)
    calibrate.add_argument("--initial-distance", type=float, default=1.0)
    calibrate.add_argument("--tolerance", type=float, default=0.015)
    calibrate.add_argument("--maximum-relative-error", type=float, default=0.025)
    calibrate.add_argument("--maximum-evaluations", type=positive_int, default=12)
    calibrate.set_defaults(function=calibrate_quality)

    bundle = subparsers.add_parser("bundle-phase1")
    bundle.add_argument("--nominal-result", type=Path, required=True)
    bundle.add_argument("--calibration-result", type=Path, required=True)
    bundle.add_argument("--matched-result", type=Path, required=True)
    bundle.add_argument("--profile-result", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.set_defaults(function=bundle_phase1)

    run = subparsers.add_parser("run")
    run.add_argument("--repo", type=Path, default=repo_default)
    run.add_argument("--corpus-manifest", type=Path, required=True)
    run.add_argument(
        "--gjxl-benchmark",
        type=Path,
        default=repo_default / "build/release/gjxl_encoding_benchmark",
    )
    run.add_argument(
        "--libjxl-benchmark",
        type=Path,
        default=comparison_build / "harness/gjxl_libjxl_comparison_benchmark",
    )
    run.add_argument(
        "--djxl", type=Path, default=comparison_build / "libjxl/tools/djxl"
    )
    run.add_argument(
        "--butteraugli",
        type=Path,
        default=comparison_build / "libjxl/tools/butteraugli_main",
    )
    run.add_argument("--metallib", type=Path)
    run.add_argument(
        "--output-root",
        type=Path,
        default=repo_default / "logs/libjxl-comparison",
    )
    run.add_argument(
        "--configuration",
        choices=("serial", "production", "both"),
        default="both",
    )
    run.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help="run only this corpus input (repeatable; defaults to the full corpus)",
    )
    run.add_argument("--pairs", type=positive_int, default=3)
    run.add_argument("--warmups", type=int, default=2)
    run.add_argument("--samples", type=positive_int, default=5)
    run.add_argument("--distance", type=float, default=1.0)
    run.add_argument(
        "--quality-map",
        type=Path,
        help="per-input libjxl distance calibration for a matched-quality run",
    )
    run.add_argument("--effort", type=int, choices=range(1, 11), default=7)
    run.add_argument(
        "--gjxl-implementation",
        choices=("scalar", "simd", "factored"),
        default="factored",
    )
    run.add_argument("--capture-samply", action="store_true")
    run.add_argument("--samply", type=Path, default=Path("samply"))
    run.add_argument(
        "--samply-analyzer",
        type=Path,
        default=repo_default / "tools/samply_neutral_stages.py",
    )
    run.add_argument("--samply-rate", type=positive_int, default=1000)
    run.add_argument("--profile-samples", type=positive_int, default=20)
    run.set_defaults(function=run_comparison)

    args = parser.parse_args(argv)
    if getattr(args, "warmups", 0) < 0:
        parser.error("--warmups must be nonnegative")
    if not 0.0 < getattr(args, "distance", 1.0) <= 25.0:
        parser.error("--distance must be in (0, 25]")
    if args.command == "calibrate-quality":
        if not 0.0 < args.minimum_distance < args.maximum_distance <= 25.0:
            parser.error(
                "calibration bounds must satisfy 0 < minimum < maximum <= 25"
            )
        if not args.minimum_distance <= args.initial_distance <= args.maximum_distance:
            parser.error("--initial-distance must lie within the calibration bounds")
        if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
            parser.error("--tolerance must be positive and finite")
        if (
            not math.isfinite(args.maximum_relative_error)
            or args.maximum_relative_error < 0.0
        ):
            parser.error("--maximum-relative-error must be finite and nonnegative")
        if args.maximum_evaluations < 3:
            parser.error("--maximum-evaluations must be at least 3")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        args.function(args)
    except (ComparisonError, WrapperError, OSError, ValueError) as exc:
        print(f"libjxl comparison error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
