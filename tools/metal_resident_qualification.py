#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Independent Metal quality qualification; no encoder parity is implied.

Runs are resumable. A candidate report is not an accepted regression baseline.
See docs/metal-resident-qualification.md for the acceptance contract.
"""

from __future__ import annotations

import argparse
from array import array
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

from metal_qualification_support import (
    ComparisonError,
    LINEAR_SRGB_HINT,
    _pfm_line,
    calibrate_distance,
    fetch_corpus_sources,
    git_output,
    load_json,
    prepare_corpus,
    run_capture,
    sha256_file,
    write_json_atomic,
)
from encode_image import WrapperError, convert_to_pfm

ROOT = Path(__file__).resolve().parents[1]
PINNED_REVISION = "e8ff09762481785938d8e4e01333ed3917571161"
SCHEMA = 1
COMPACT_DISTANCES = (0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.499, 2.5, 2.501, 3.0)
FULL_DISTANCES = tuple(
    sorted({i / 10 for i in range(5, 31)} | {0.999, 1.001, 1.199, 1.201, 2.499, 2.501})
)
POLICIES = {f"e{e}": ["--effort", str(e)] for e in (1, 4, 7, 8, 9, 10)}
POLICIES.update(
    {
        f"e{e}-max": ["--effort", str(e), "--maximum-compression"]
        for e in (1, 4, 7, 8, 10)
    }
)
ALIASES = {
    "e1": [["--effort", "2"], ["--effort", "3"]],
    "e4": [["--effort", "5"], ["--effort", "6"]],
    "e10": [["--high-density"]],
    "e8-max": [["--effort", "9", "--maximum-compression"]],
    "e10-max": [["--high-density", "--maximum-compression"]],
}
METRIC = {
    "name": "libjxl-butteraugli",
    "colorspace": LINEAR_SRGB_HINT,
    "intensity_target": 255,
    "absolute_match": 0.015,
    "relative_match": 0.02,
    "calibration_bounds": [0.03, 15.0],
    "maximum_evaluations": 24,
}
LIMITS = {"absolute_quality": 0.10, "relative_quality": 0.10, "relative_size": 0.10}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_pfm(path):
    """Read finite RGB float32 pixels, preserving the file's bottom-up order."""
    with Path(path).open("rb") as stream:
        if _pfm_line(stream) != b"PF":
            raise ComparisonError(f"Not RGB PFM: {path}")
        try:
            width, height = map(int, _pfm_line(stream).split())
            scale = float(_pfm_line(stream))
        except ValueError as exc:
            raise ComparisonError(f"Invalid PFM header: {path}") from exc
        if width <= 0 or height <= 0 or abs(scale) != 1:
            raise ComparisonError(f"Invalid PFM dimensions/scale: {path}")
        pixels = array("f")
        data = stream.read()
        if len(data) != width * height * 12:
            raise ComparisonError(f"Invalid PFM payload: {path}")
        pixels.frombytes(data)
        if (scale < 0) != (sys.byteorder == "little"):
            pixels.byteswap()
        # A finite float32 array cannot overflow a double sum at feasible image
        # sizes. Any NaN or infinity makes this sum non-finite, including +/-inf.
        if not math.isfinite(sum(pixels)):
            raise ComparisonError(f"Non-finite PFM pixels: {path}")
        return width, height, pixels


def write_pfm(path, width, height, top_down):
    with path.open("wb") as stream:
        stream.write(f"PF\n{width} {height}\n-1.0\n".encode())
        for y in reversed(range(height)):
            row = top_down[y * width * 3 : (y + 1) * width * 3]
            stream.write(struct.pack(f"<{len(row)}f", *row))


def compact_sources(destination, magick):
    """Integer-seeded fixtures, plus crops from pinned libjxl testdata."""
    entries = []
    for index, name in enumerate(
        ("flat", "gradient", "edge", "saturated", "texture", "noise")
    ):
        width, height = ((128, 96), (129, 97), (257, 193))[index % 3]
        values = []
        seed = 123456789
        for y in range(height):
            for x in range(width):
                seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
                noise = (seed >> 8) / 16777215
                if name == "flat":
                    rgb = (0.18, 0.18, 0.18)
                elif name == "gradient":
                    rgb = (x / width, y / height, (x + y) / (width + height))
                elif name == "edge":
                    rgb = (float(x > width // 2),) * 3
                elif name == "saturated":
                    rgb = tuple(float((x // 13 + y // 11) % 3 == c) for c in range(3))
                elif name == "texture":
                    rgb = tuple(((x * (c + 3) + y * 7) % 31) / 31 for c in range(3))
                else:
                    rgb = (noise, 0.5 * noise, 0.2 + 0.7 * noise)
                values.extend(rgb)
        path = destination / f"{name}.pfm"
        write_pfm(path, width, height, values)
        entries.append(corpus_entry(path, destination, name, "synthetic"))
    testdata = ROOT / "third_party/libjxl/testdata/jxl"
    flower = destination / "flower-source.pfm"
    convert_to_pfm(magick, testdata / "flower/flower.png", flower, "black")
    for name, source in (
        ("flower", flower),
        ("grayscale", testdata / "blending/grayscale_patches_on_splines.pfm"),
    ):
        w, h, pixels = read_pfm(source)
        cw, ch = 128, 96
        if w < cw or h < ch:
            raise ComparisonError(f"Pinned crop is too small: {source}")
        left, bottom = (w - cw) // 2, (h - ch) // 2
        rows = [
            pixels[((bottom + y) * w + left) * 3 : ((bottom + y) * w + left + cw) * 3]
            for y in reversed(range(ch))
        ]
        path = destination / f"{name}.pfm"
        write_pfm(path, cw, ch, [v for row in rows for v in row])
        entries.append(corpus_entry(path, destination, name, "pinned-crop"))
    flower.unlink()
    return entries


def corpus_entry(path, root, name, category):
    w, h, _ = read_pfm(path)
    return {
        "name": name,
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "width": w,
        "height": h,
        "category": category,
    }


def validate_corpus_entries(entries, suite):
    locked = load_json(ROOT / "tests/metal_qualification/inputs.json")
    expected = {
        e["name"]: e
        for e in locked["inputs"]
        if suite == "full" or e["name"] in locked["compact"]
    }
    if len(entries) != len(expected) or {e["name"] for e in entries} != set(expected):
        raise ComparisonError("Corpus cases differ from the checked-in suite")
    for entry in entries:
        for field in ("sha256", "width", "height"):
            if entry[field] != expected[entry["name"]][field]:
                raise ComparisonError(f"Pinned corpus {field} differs: {entry['name']}")


def prepare(args):
    destination = args.output.resolve()
    if destination.exists():
        raise ComparisonError(
            f"Corpus output exists: {destination}; reuse its manifest"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".prepare-"
    ) as temporary:
        work = argparse.Namespace(**vars(args))
        work.output = Path(temporary) / "corpus"
        _prepare(work)
        if destination.exists():
            raise ComparisonError(
                f"Corpus output appeared during preparation: {destination}"
            )
        work.output.rename(destination)
    print(destination / "manifest.json", flush=True)


def _prepare(args):
    out = args.output.resolve()
    if out.exists():
        raise ComparisonError(f"Corpus output exists: {out}; reuse its manifest")
    if git_output(ROOT / "third_party/libjxl", "rev-parse", "HEAD") != PINNED_REVISION:
        raise ComparisonError("Corpus preparation requires pinned libjxl")
    out.mkdir(parents=True)
    entries = compact_sources(out, args.magick)
    compact = [entry["name"] for entry in entries]
    if args.suite == "full":
        if args.pilot_manifest:
            manifest_path = args.pilot_manifest.resolve()
        else:
            recipe = ROOT / "tests/metal_qualification/phase1-pilot.json"
            sources = out / "sources"
            fetch_corpus_sources(
                argparse.Namespace(
                    source_manifest=recipe,
                    output=sources,
                    timeout=120,
                    gjxl_benchmark=args.gjxl_benchmark,
                )
            )
            prepare_corpus(
                argparse.Namespace(
                    source_manifest=recipe,
                    source_root=sources,
                    output=out / "pilot",
                    magick=args.magick,
                    background="black",
                )
            )
            manifest_path = out / "pilot/manifest.json"
        pilot = load_json(manifest_path)
        recipe_hash = sha256_file(ROOT / "tests/metal_qualification/phase1-pilot.json")
        if (
            pilot.get("source_manifest_sha256") != recipe_hash
            or len(pilot.get("inputs", [])) != 38
        ):
            raise ComparisonError(
                "Pilot corpus recipe/count does not match the pinned 38-image corpus"
            )
        recipe = load_json(ROOT / "tests/metal_qualification/phase1-pilot.json")
        expected_sources = {item["name"]: item["sha256"] for item in recipe["inputs"]}
        if {item["name"] for item in pilot["inputs"]} != set(expected_sources):
            raise ComparisonError("Pilot input names differ from the pinned corpus")
        for item in pilot["inputs"]:
            if item["source_sha256"] != expected_sources[item["name"]]:
                raise ComparisonError(f"Pilot source hash differs: {item['name']}")
            src = manifest_path.parent / item["canonical_path"]
            if sha256_file(src) != item["canonical_sha256"]:
                raise ComparisonError(f"Pilot input changed: {src}")
            # Read-only sharing avoids copying several GB. The manifest binds
            # content, not the storage location; every run rechecks the hash.
            entry = corpus_entry(src, src.parent, item["name"], item["category"])
            try:
                entry["path"] = str(src.resolve().relative_to(out))
            except ValueError:
                entry["path"] = str(src.resolve())
            entries.append(entry)
    validate_corpus_entries(entries, args.suite)
    write_json_atomic(
        out / "manifest.json",
        {
            "schema_version": SCHEMA,
            "kind": "metal-resident-corpus",
            "compact": compact,
            "inputs": entries,
            "libjxl_revision": PINNED_REVISION,
        },
    )


def parse_score(stdout):
    try:
        value = float(stdout.splitlines()[0])
    except (ValueError, IndexError) as exc:
        raise ComparisonError("Invalid Butteraugli output") from exc
    if not math.isfinite(value) or value < 0:
        raise ComparisonError("Butteraugli score must be finite and nonnegative")
    return value


def check_backend(stdout, backend):
    expected = " using Metal fully-resident AQ" if backend == "metal" else " using CPU"
    if expected not in stdout:
        raise ComparisonError(f"Encoder did not report requested {backend} backend")


def quality_limit(reference):
    return reference + max(
        LIMITS["absolute_quality"], LIMITS["relative_quality"] * reference
    )


def acceptance(row, baseline=None):
    failures = []
    metal, cpu = row["metal"], row["cpu"]
    if metal["butteraugli"] > quality_limit(cpu["butteraugli"]):
        failures.append("resident quality exceeds CPU allowance")
    if (
        row.get("cpu_size_ratio") is not None
        and row["cpu_size_ratio"] > 1 + LIMITS["relative_size"]
    ):
        failures.append("resident matched-quality size exceeds CPU allowance")
    if baseline:
        if metal["butteraugli"] > quality_limit(baseline["metal"]["butteraugli"]):
            failures.append("fixed-distance quality regressed from baseline")
        if metal["bytes"] > baseline["metal"]["bytes"] * (1 + LIMITS["relative_size"]):
            failures.append("fixed-distance bytes regressed from baseline")
        if row.get("libjxl_size_ratio") is not None and row[
            "libjxl_size_ratio"
        ] > baseline["libjxl_size_ratio"] * (1 + LIMITS["relative_size"]):
            failures.append("matched-quality libjxl size ratio regressed from baseline")
    return failures


def case_id(name, policy, distance):
    return f"{name}/{policy}/{distance:g}"


def validate_baseline(baseline, contract, expected):
    if (
        baseline.get("kind") != "metal-resident-baseline"
        or baseline.get("schema_version") != SCHEMA
        or baseline.get("contract") != contract
        or not baseline.get("passed")
    ):
        raise ComparisonError(
            "Baseline configuration differs or baseline was not accepted"
        )
    rows = baseline.get("rows", [])
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ComparisonError("Baseline cases are missing, duplicated, or unexpected")
    if any(row.get("failures") or row.get("status") != "pass" for row in rows):
        raise ComparisonError("Baseline contains failing/incomplete cases")
    for row in rows:
        for name in ("cpu", "metal"):
            validate_measurement(row[name])
        for name in ("cpu_size_ratio", "libjxl_size_ratio"):
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ComparisonError("Baseline contains invalid size ratios")
    return by_id


def validate_measurement(result):
    score, size = result.get("butteraugli"), result.get("bytes")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or score < 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ComparisonError("Invalid cached/baseline measurement")


class Evaluator:
    def __init__(self, args, identity):
        self.args, self.identity = args, identity
        self.cache = args.cache.resolve() / digest(identity)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.artifact_lock = Lock()
        self.artifact_bytes = 0

    def command(self, command, directory, label):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=900
            )
        except subprocess.TimeoutExpired as exc:
            write_json_atomic(
                directory / f"{label}.command.json",
                {"argv": command, "timeout_seconds": 900},
            )
            raise ComparisonError(f"{label} timed out; see {directory}") from exc
        (directory / f"{label}.stdout").write_text(result.stdout)
        (directory / f"{label}.stderr").write_text(result.stderr)
        write_json_atomic(
            directory / f"{label}.command.json",
            {"argv": command, "returncode": result.returncode},
        )
        if result.returncode:
            raise ComparisonError(
                f"{label} failed ({result.returncode}); see {directory}: {result.stderr[-1000:]}"
            )
        return result.stdout

    def evaluate(self, entry, backend, policy, distance, extra=None):
        flags = POLICIES[policy] if extra is None else extra
        key = {
            "input": entry["sha256"],
            "backend": backend,
            "flags": flags if backend != "libjxl" else ["effort", 7],
            "distance": distance,
        }
        directory = self.cache / digest(key)
        directory.mkdir(exist_ok=True)
        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._evaluate(entry, backend, flags, distance, directory)

    def _evaluate(self, entry, backend, flags, distance, directory):
        cached = directory / "result.json"
        if cached.exists():
            result = load_json(cached)
            validate_measurement(result)
            return result
        output = directory / "output.jxl"
        if backend == "libjxl":
            command = [
                str(self.args.cjxl),
                entry["resolved"],
                str(output),
                "-d",
                str(distance),
                "-e",
                "7",
                "--num_threads=8",
                "-x",
                f"color_space={LINEAR_SRGB_HINT}",
            ]
        else:
            command = [
                str(self.args.encoder),
                "--backend",
                backend,
                "--metal-aq",
                "fully-resident",
                "--distance",
                str(distance),
                *flags,
                entry["resolved"],
                str(output),
            ]
        stdout = self.command(command, directory, "encode")
        if backend != "libjxl":
            check_backend(stdout, backend)
        if not output.exists() or output.stat().st_size == 0:
            raise ComparisonError("Encoder produced no codestream")
        sha = sha256_file(output)
        if backend == "metal":
            repeat = directory / "repeat.jxl"
            stdout = self.command([*command[:-1], str(repeat)], directory, "repeat")
            check_backend(stdout, backend)
            if sha256_file(repeat) != sha:
                raise ComparisonError("Resident output is not repeatable")
            repeat.unlink()
        decoded = directory / "decoded.pfm"
        self.command(
            [str(self.args.djxl), str(output), str(decoded), "--num_threads=8"],
            directory,
            "decode",
        )
        w, h, _ = read_pfm(decoded)
        if (w, h) != (entry["width"], entry["height"]):
            raise ComparisonError("Decoded dimensions changed")
        decoded_sha = sha256_file(decoded)
        metric_cache = (
            self.cache / f"metric-{digest([entry['sha256'], decoded_sha])}.json"
        )
        metric_command = [
            str(self.args.butteraugli),
            entry["resolved"],
            str(decoded),
            "--colorspace",
            LINEAR_SRGB_HINT,
            "--intensity_target",
            "255",
        ]
        if metric_cache.exists():
            metric = load_json(metric_cache)
            score = parse_score(str(metric["score"]))
            write_json_atomic(directory / "metric-reuse.json", metric)
        else:
            score = parse_score(self.command(metric_command, directory, "metric"))
            write_json_atomic(
                metric_cache,
                {
                    "score": score,
                    "measured_at": str(directory),
                    "source_sha256": entry["sha256"],
                    "decoded_sha256": decoded_sha,
                },
            )
        result = {
            "distance": distance,
            "backend": backend,
            "bytes": output.stat().st_size,
            "butteraugli": score,
            "sha256": sha,
            "decoded_sha256": decoded_sha,
            "artifact": str(directory),
        }
        write_json_atomic(cached, result)
        # Pixels are reproducible from the retained codestream + decode argv.
        # Avoid retaining dozens of TB of calibration PFMs in a broad sweep.
        decoded.unlink()
        output.unlink()
        return result

    def match(self, entry, backend, policy, distance, score):
        # Entropy modes do not define quality identity. A previously successful
        # distance is only a hint: always encode/decode the requested policy
        # and verify its independently measured score before accepting reuse.
        hint_path = (
            self.cache / f"match-{digest([entry['sha256'], backend, score])}.json"
        )
        initial = distance
        if hint_path.exists():
            initial = load_json(hint_path)["distance"]
            hint = self.evaluate(entry, backend, policy, initial)
            if abs(hint["butteraugli"] - score) <= max(
                METRIC["absolute_match"], METRIC["relative_match"] * score
            ):
                return hint, [
                    {k: hint[k] for k in ("distance", "butteraugli", "bytes")}
                ]
        attempted = []

        def evaluate(d):
            result = self.evaluate(entry, backend, policy, d)
            attempted.append(
                {k: result[k] for k in ("distance", "butteraugli", "bytes")}
            )
            return result

        try:
            matched, evaluations = calibrate_distance(
                score,
                evaluate,
                minimum_distance=METRIC["calibration_bounds"][0],
                maximum_distance=METRIC["calibration_bounds"][1],
                initial_distance=initial,
                tolerance=METRIC["absolute_match"],
                maximum_relative_error=METRIC["relative_match"],
                maximum_evaluations=METRIC["maximum_evaluations"],
            )
        except ComparisonError as exc:
            exc.calibration_history = attempted
            raise
        write_json_atomic(hint_path, {"distance": matched["distance"]})
        return matched, [
            {k: row[k] for k in ("distance", "butteraugli", "bytes")}
            for row in evaluations
        ]

    def retain_failure(self, row, destination):
        """Keep bounded failed pixels; every remaining case retains replay argv."""
        retained = []
        for name in ("cpu", "metal", "cpu_matched", "libjxl_matched"):
            result = row.get(name)
            if not result:
                continue
            directory = Path(result["artifact"])
            target = destination / digest([result["sha256"], result["decoded_sha256"]])
            with self.artifact_lock:
                if target.exists():
                    retained.append(str(target))
                    continue
                # Reserve the decoded source extent plus codestream before
                # launching replay. Avoid overshooting on parallel inputs.
                decode_argv = load_json(directory / "decode.command.json")["argv"]
                encode_argv = load_json(directory / "encode.command.json")["argv"]
                source = Path(
                    encode_argv[-2] if name != "libjxl_matched" else encode_argv[1]
                )
                required = source.stat().st_size + result["bytes"]
                if (
                    self.artifact_bytes + required
                    > self.args.artifact_budget_mib * 1024**2
                ):
                    continue
                self.artifact_bytes += required
                target.mkdir(parents=True)
            encoded = target / "output.jxl"
            decoded = target / "decoded.pfm"
            # Rewrite only observed argument positions, never shell text.
            original_encoded = str(directory / "output.jxl")
            encode_argv = [
                str(encoded) if v == original_encoded else v for v in encode_argv
            ]
            decode_argv = [
                str(encoded)
                if v == original_encoded
                else str(decoded)
                if v == str(directory / "decoded.pfm")
                else v
                for v in decode_argv
            ]
            self.command(encode_argv, target, "encode")
            self.command(decode_argv, target, "decode")
            if (
                sha256_file(encoded) != result["sha256"]
                or sha256_file(decoded) != result["decoded_sha256"]
            ):
                raise ComparisonError(
                    "Failed artifact replay differs from the measured output"
                )
            retained.append(str(target))
        row["retained_artifacts"] = retained
        row["replay_note"] = (
            "All evaluation directories retain encode/decode argv; decoded samples use a bounded disk budget."
        )


def check_aliases(evaluator, entry, policy, distance, metal, cpu):
    for flags in ALIASES.get(policy, []):
        for backend, reference in (("metal", metal), ("cpu", cpu)):
            alias = evaluator.evaluate(entry, backend, policy, distance, extra=flags)
            if alias["sha256"] != reference["sha256"]:
                raise ComparisonError(f"Policy alias differs: {backend} {flags}")


def run(args):
    if args.baseline and args.baseline.resolve().is_relative_to(args.output.resolve()):
        raise ComparisonError("Baseline must be outside the report output directory")
    manifest = load_json(args.corpus)
    if (
        manifest.get("kind") != "metal-resident-corpus"
        or manifest.get("schema_version") != SCHEMA
    ):
        raise ComparisonError("Unsupported qualification corpus")
    locked = load_json(ROOT / "tests/metal_qualification/inputs.json")
    if manifest["compact"] != locked["compact"]:
        raise ComparisonError("Compact membership differs from the checked-in corpus")
    entries = manifest["inputs"]
    if args.suite == "compact":
        entries = [e for e in entries if e["name"] in manifest["compact"]]
    validate_corpus_entries(entries, args.suite)
    if len(entries) != (8 if args.suite == "compact" else 46):
        raise ComparisonError("Corpus is missing required suite inputs")
    if len({e["sha256"] for e in entries}) != len(entries):
        raise ComparisonError("Duplicate corpus content hashes")
    if len({e["name"] for e in entries}) != len(entries):
        raise ComparisonError("Duplicate corpus input names")
    for entry in entries:
        path = (args.corpus.resolve().parent / entry["path"]).resolve()
        if sha256_file(path) != entry["sha256"]:
            raise ComparisonError(f"Corpus input changed: {path}")
        w, h, _ = read_pfm(path)
        if (w, h) != (entry["width"], entry["height"]):
            raise ComparisonError("Corpus dimensions differ")
        entry["resolved"] = str(path)
    distances = COMPACT_DISTANCES if args.suite == "compact" else FULL_DISTANCES
    tools = {}
    for name in ("encoder", "cjxl", "djxl", "butteraugli"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise ComparisonError(f"Missing {name}: {path}")
        setattr(args, name, path)
        tools[name] = sha256_file(path)
    reference = load_json(args.reference_manifest)
    if reference.get("revision") != PINNED_REVISION:
        raise ComparisonError("Reference libjxl revision differs")
    for name in ("cjxl", "djxl", "butteraugli"):
        if reference.get("tools", {}).get(name) != tools[name]:
            raise ComparisonError(f"Reference tool hash differs: {name}")
    hardware = run_capture(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip()
    if hardware != "Apple M4 Pro":
        raise ComparisonError(
            f"Default rollout requires qualified Apple M4 Pro, got {hardware}"
        )
    contract = {
        "suite": args.suite,
        "metric": METRIC,
        "limits": LIMITS,
        "reference_revision": PINNED_REVISION,
        "reference_tools": {k: v for k, v in tools.items() if k != "encoder"},
        "hardware": hardware,
        "policies": POLICIES,
        "aliases": ALIASES,
        "distances": distances,
        "inputs": [
            {k: e[k] for k in ("name", "sha256", "width", "height")} for e in entries
        ],
    }
    # Normalize tuples to their JSON representation before comparing manifests.
    contract = json.loads(json.dumps(contract))
    source_diff = run_capture(
        ["git", "-C", str(ROOT), "diff", "HEAD", "--", "src", "tools/gjxl_encode.cpp"]
    ).stdout
    identity = {
        "tools": tools,
        "metric": METRIC,
        "hardware": hardware,
        "os": platform.platform(),
        "evaluation_version": 1,
    }
    expected = [
        case_id(e["name"], p, d) for e in entries for d in distances for p in POLICIES
    ]
    baseline = (
        validate_baseline(load_json(args.baseline), contract, expected)
        if args.baseline
        else {}
    )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run.json"
    run_identity = {
        "contract": contract,
        "identity": identity,
        "baseline": sha256_file(args.baseline) if args.baseline else None,
        "runner_sha256": sha256_file(Path(__file__)),
        "support_sha256": sha256_file(ROOT / "tools/metal_qualification_support.py"),
        "artifact_budget_mib": args.artifact_budget_mib,
    }
    if manifest_path.exists() and load_json(manifest_path)["identity"] != run_identity:
        raise ComparisonError(
            "Resume directory belongs to a different run configuration"
        )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": SCHEMA,
            "identity": run_identity,
            "gjxl_revision": git_output(ROOT, "rev-parse", "HEAD"),
            "encoder_source_diff_sha256": hashlib.sha256(
                source_diff.encode()
            ).hexdigest(),
        },
    )
    evaluator = Evaluator(args, identity)
    rows_dir = out / "cases"
    rows_dir.mkdir(exist_ok=True)
    rows = []
    progress_lock = Lock()
    cancelled = Event()
    artifacts = out / "failure-artifacts"
    evaluator.artifact_bytes = sum(
        p.stat().st_size for p in artifacts.rglob("*") if p.is_file()
    )
    started = time.monotonic()

    def run_input(entry):
        for distance in distances:
            for policy in POLICIES:
                if cancelled.is_set():
                    return
                cid = case_id(entry["name"], policy, distance)
                path = rows_dir / f"{digest(cid)}.json"
                if path.exists():
                    row = load_json(path)
                    if row.get("id") != cid or row.get("status") not in (
                        "pass",
                        "fail",
                        "incomplete",
                    ):
                        raise ComparisonError(f"Invalid resumed case: {path}")
                else:
                    row = {
                        "id": cid,
                        "input": entry["name"],
                        "policy": policy,
                        "distance": distance,
                        "failures": [],
                    }
                    try:
                        row["cpu"] = evaluator.evaluate(entry, "cpu", policy, distance)
                        row["metal"] = evaluator.evaluate(
                            entry, "metal", policy, distance
                        )
                        if entry["name"] in manifest["compact"] and distance == 1.2:
                            check_aliases(
                                evaluator,
                                entry,
                                policy,
                                distance,
                                row["metal"],
                                row["cpu"],
                            )
                        score = row["metal"]["butteraugli"]
                        for reference_backend in ("cpu", "libjxl"):
                            try:
                                match, history = evaluator.match(
                                    entry, reference_backend, policy, distance, score
                                )
                                row[f"{reference_backend}_matched"] = match
                                row[f"{reference_backend}_calibration"] = history
                                row[f"{reference_backend}_size_ratio"] = (
                                    row["metal"]["bytes"] / match["bytes"]
                                )
                            except ComparisonError as exc:
                                row[f"{reference_backend}_calibration"] = getattr(
                                    exc, "calibration_history", []
                                )
                                row["failures"].append(
                                    f"{reference_backend} calibration: {exc}"
                                )
                        row["failures"].extend(acceptance(row, baseline.get(cid)))
                        incomplete = any(
                            f"{b}_matched" not in row for b in ("cpu", "libjxl")
                        )
                        row["status"] = (
                            "incomplete"
                            if incomplete
                            else ("fail" if row["failures"] else "pass")
                        )
                    except (ComparisonError, OSError, ValueError) as exc:
                        row["status"] = "incomplete"
                        row["failures"].append(str(exc))
                    if row["failures"]:
                        try:
                            evaluator.retain_failure(row, artifacts)
                        except (ComparisonError, OSError) as exc:
                            row["failures"].append(f"Artifact replay: {exc}")
                    write_json_atomic(path, row)
                with progress_lock:
                    rows.append(row)
                    if len(rows) % 10 == 0 or len(rows) == len(expected):
                        count = sum(r["status"] != "pass" for r in rows)
                        print(
                            f"{len(rows)}/{len(expected)} cases; {count} failing/incomplete; "
                            f"{time.monotonic() - started:.0f}s; {cid}",
                            flush=True,
                        )

    previous_handler = signal.signal(signal.SIGINT, lambda *_: cancelled.set())
    try:
        # Visit the additional real-image corpus first in the broad run.
        work_entries = entries[8:] + entries[:8] if args.suite == "full" else entries
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            list(executor.map(run_input, work_entries))
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    order = {cid: index for index, cid in enumerate(expected)}
    rows.sort(key=lambda row: order[row["id"]])
    passed = all(row["status"] == "pass" for row in rows) and len(rows) == len(expected)
    report = {
        "schema_version": SCHEMA,
        "kind": "metal-resident-qualification",
        "contract": contract,
        "passed": passed,
        "expected_cases": len(expected),
        "rows": rows,
        "baseline_used": str(args.baseline) if args.baseline else None,
        "cancelled": cancelled.is_set(),
        "qualification_complete": len(rows) == len(expected)
        and not any(r["status"] == "incomplete" for r in rows),
    }
    write_json_atomic(out / "report.json", report)
    with (out / "report.csv").open("w", newline="") as stream:
        fields = [
            "id",
            "status",
            "cpu_score",
            "metal_score",
            "metal_bytes",
            "cpu_size_ratio",
            "libjxl_size_ratio",
            "failures",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "status": row["status"],
                    "cpu_score": row.get("cpu", {}).get("butteraugli"),
                    "metal_score": row.get("metal", {}).get("butteraugli"),
                    "metal_bytes": row.get("metal", {}).get("bytes"),
                    "cpu_size_ratio": row.get("cpu_size_ratio"),
                    "libjxl_size_ratio": row.get("libjxl_size_ratio"),
                    "failures": "; ".join(row["failures"]),
                }
            )
    if args.record_baseline:
        if not passed:
            raise ComparisonError(
                "Failed qualification cannot become an accepted baseline"
            )
        if args.record_baseline.exists():
            raise ComparisonError("Baseline destination exists; never overwrite it")
        write_json_atomic(
            args.record_baseline, {**report, "kind": "metal-resident-baseline"}
        )
    print(f"{'PASS' if passed else 'FAIL'}: {out / 'report.json'}", flush=True)
    return 130 if cancelled.is_set() else (0 if passed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare", help="Prepare pinned corpus (full may download sources)"
    )
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--suite", choices=("compact", "full"), default="compact")
    prep.add_argument(
        "--pilot-manifest",
        type=Path,
        help="Reuse verified existing canonical pilot inputs",
    )
    prep.add_argument(
        "--gjxl-benchmark",
        type=Path,
        default=ROOT / "build/release/gjxl_encoding_benchmark",
    )
    prep.add_argument("--magick", default="magick")
    execute = sub.add_parser(
        "run", help="Run or resume qualification; no baseline is rewritten"
    )
    execute.add_argument("--suite", choices=("compact", "full"), default="compact")
    for name in (
        "corpus",
        "output",
        "cache",
        "encoder",
        "cjxl",
        "djxl",
        "butteraugli",
        "reference-manifest",
    ):
        execute.add_argument(f"--{name}", type=Path, required=True)
    execute.add_argument(
        "--jobs",
        type=int,
        choices=(1, 2),
        default=1,
        help="Independent input workers; default 1 limits GPU memory",
    )
    execute.add_argument(
        "--artifact-budget-mib",
        type=int,
        default=2048,
        help="Maximum retained decoded failure samples; all replay commands remain",
    )
    execute.add_argument("--baseline", type=Path)
    execute.add_argument("--record-baseline", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.artifact_budget_mib < 0:
        parser.error("Artifact budget must be nonnegative")
    try:
        if args.command == "prepare":
            prepare(args)
            return 0
        return run(args)
    except (ComparisonError, WrapperError, OSError, ValueError, KeyError) as exc:
        print(f"Qualification error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
