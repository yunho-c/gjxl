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
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
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
LINEAR_SRGB_HINT = "RGB_D65_SRG_Rel_Lin"


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
        str(args.distance),
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
) -> list[str]:
    command = [
        str(args.libjxl_benchmark),
        "--input",
        str(image),
        "--raw-samples",
        str(raw),
        "--distance",
        str(args.distance),
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
    return {
        "input": entry["name"],
        "encoder": encoder,
        "configuration": configuration,
        "codestream": str(codestream.relative_to(directory)),
        "codestream_sha256": sha256_file(codestream),
        "decoded": str(decoded.relative_to(directory)),
        "decoded_sha256": sha256_file(decoded),
        "butteraugli": score,
    }


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
        args.samply_analyzer = args.samply_analyzer.resolve()
        binaries["samply_analyzer"] = args.samply_analyzer
    for name, path in binaries.items():
        if not path.is_file():
            raise ComparisonError(f"Required {name} file does not exist: {path}")
    args.gjxl_benchmark = binaries["gjxl_benchmark"]
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
            "effort": args.effort,
            "warmups": args.warmups,
            "samples": args.samples,
            "pairs": args.pairs,
            "gjxl_implementation": args.gjxl_implementation,
            "configurations": args.configuration,
        },
        "binaries": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in binaries.items()
        },
        "host": host_manifest(),
        "argv": sys.argv,
        "processes": [],
        "validations": [],
    }
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
    for configuration, gjxl_workers, libjxl_threads in configurations:
        for entry in entries:
            image = Path(entry["resolved_path"])
            slug = safe_name(entry["name"])
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
                            args, image, raw, gjxl_workers, output
                        )
                    else:
                        command = libjxl_command(
                            args, image, raw, libjxl_threads, output
                        )
                    name = f"{slug}-{configuration}-pair{pair}-{encoder}"
                    record_process(command, name, directory, manifest)
                    rows.append(
                        median_process_row(
                            encoder, configuration, entry, raw
                        )
                    )
                    rows[-1]["raw_path"] = str(raw.relative_to(directory))
            for encoder, path in output_paths.items():
                validation = validate_output(
                    args,
                    entry,
                    encoder,
                    configuration,
                    path,
                    directory,
                    manifest,
                )
                manifest["validations"].append(validation)
                write_json_atomic(directory / "manifest.json", manifest)

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
                        )
                    else:
                        command = libjxl_command(
                            args,
                            image,
                            raw,
                            libjxl_threads,
                            None,
                            args.profile_samples,
                        )
                    capture_profile(
                        args,
                        command,
                        profile,
                        f"{slug}-{configuration}-{encoder}-samply",
                        directory,
                        manifest,
                    )
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
                                "--output",
                                str(analysis_output),
                            ],
                            f"{slug}-{configuration}-{encoder}-analyze-{output_format}",
                            directory,
                            manifest,
                        )

    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "timing_claim": "unprofiled complete-encode wall time",
        "process_rows": rows,
        "aggregate_rows": aggregate_rows(rows),
        "validations": manifest["validations"],
    }
    write_json_atomic(directory / "summary.json", summary)
    manifest["host_end"] = host_manifest()
    write_json_atomic(directory / "manifest.json", manifest)
    print(directory)


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

    run = subparsers.add_parser("run")
    run.add_argument("--repo", type=Path, default=repo_default)
    run.add_argument("--corpus-manifest", type=Path, required=True)
    run.add_argument(
        "--gjxl-benchmark",
        type=Path,
        default=repo_default / "build/release/gjxl_encoding_benchmark",
    )
    comparison_build = repo_default / "build/libjxl-comparison"
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
    run.add_argument("--pairs", type=positive_int, default=3)
    run.add_argument("--warmups", type=int, default=2)
    run.add_argument("--samples", type=positive_int, default=5)
    run.add_argument("--distance", type=float, default=1.0)
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
