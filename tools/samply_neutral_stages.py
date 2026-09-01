#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in third_party/libjxl/LICENSE.

"""Classify GJXL and libjxl Samply captures into neutral serializer stages.

This is adapted from libjxl's retained ``cjxl_samply_profile.py`` parser. The
reported values are sampled thread-CPU attribution, never stage wall time.
Every positive sample is assigned to exactly one ordered stage.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import gzip
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROFILE_SUFFIX = ".json.gz"
SYMBOL_SUFFIX = ".json.syms.json"
SCHEMA_VERSION = 1

COEFFICIENT_PREPARATION = "coefficient_and_token_preparation"
ENTROPY_MODEL = "entropy_model_construction"
TOKEN_EMISSION = "token_and_model_emission"
ASSEMBLY = "framing_and_assembly"
OTHER = "other_encoder_and_runtime"
UNRESOLVED = "unresolved_or_missing_stack"

# AC-strategy candidate estimation is deliberately excluded from final
# codestream entropy attribution even when it calls entropy-looking helpers.
PRE_SERIALIZER_PATTERNS = (
    "ProcessRectACS",
    "EstimateEntropy",
    "LossyFrameHeuristics",
)

# Rules are ordered. Model construction precedes token emission so a
# WriteTokens leaf under BuildAndEncodeHistograms remains model construction;
# token emission precedes assembly so AppendByteAligned under a token writer
# remains emission.
STAGE_RULES = (
    (
        COEFFICIENT_PREPARATION,
        (
            "TokenizeSimpleDcGroup",
            "TokenizeSimpleAcMetadata",
            "BuildSimpleAcGroupTokenTemplate",
            "MaterializeSimpleAcGroup",
            "TokenizeSimpleAcGroup",
            "TokenizeSimpleCoefficientOrders",
            "TokenizePermutation",
            "ComputeSimpleCoefficientOrders",
            "ComputeSimpleBlockContextMapCandidates",
            "TokenizeAllCoefficients",
            "TokenizeCoefficients",
            "ComputeCoeffOrder",
            "ComputeAllCoeffOrders",
        ),
    ),
    (
        ENTROPY_MODEL,
        (
            "OptimizeBestEntropyCode",
            "OptimizeEntropyCode",
            "OptimizeAnsEntropyCode",
            "BuildEntropyCodeForPartition",
            "BuildFixedEntropyCodeForPartition",
            "HistogramCountsBitCost",
            "HistogramBitCost",
            "HistogramDistance",
            "ClusterHistograms",
            "NormalizeHistogram",
            "RebalanceHistogram",
            "BuildBestAnsHistogram",
            "EstimateAnsHistogramCost",
            "ChooseUintConfigs",
            "BuildAndStoreEntropy",
            "BuildAndEncodeHistograms",
            "CountTokenStreamBits",
            "CountAnsTokenStreamBits",
            "MeasureCandidateSize",
            "MeasureDcGroupSections",
            "MeasureCommonSections",
            "MeasureAcSections",
        ),
    ),
    (
        TOKEN_EMISSION,
        (
            "WriteAnsEntropyCodeModel",
            "WriteAnsTokenStream",
            "WriteEntropyCode",
            "WriteContextMap",
            "WriteTokenStream",
            "WriteTokens",
            "WriteDcGroupSection",
            "WriteCommonSections",
            "WriteAcSections",
            "EncodeGroupTokenizedCoefficients",
            "EncodeModularChannel",
            "ModularCompress",
        ),
    ),
    (
        ASSEMBLY,
        (
            "AssembleCandidate",
            "PhysicalSectionSizes",
            "WriteSimpleFrameHeader",
            "WriteFrameHeader",
            "WriteTocPermutation",
            "WriteTocSizes",
            "WriteTocAndSections",
            "ConcatenateSections",
            "AppendByteAligned",
            "AppendData",
        ),
    ),
)


class ProfileError(RuntimeError):
    """An expected profile input or symbolization failure."""


@dataclasses.dataclass(frozen=True)
class FrameSymbol:
    name: str
    library: str


@dataclasses.dataclass(frozen=True)
class CaptureSummary:
    path: str
    sample_count: int
    cpu_delta_us: int


@dataclasses.dataclass
class Analysis:
    delta_attribution: str
    captures: list[CaptureSummary]
    sample_count: int = 0
    cpu_delta_us: int = 0
    resolved_leaf_cpu_us: int = 0
    flat: collections.Counter[FrameSymbol] = dataclasses.field(
        default_factory=collections.Counter
    )
    inclusive: collections.Counter[FrameSymbol] = dataclasses.field(
        default_factory=collections.Counter
    )
    stages: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )


def _address_key(address: Any) -> Any:
    if isinstance(address, str):
        try:
            return int(address, 0)
        except ValueError:
            return address
    return address


class SymbolResolver:
    """Resolve Samply frame addresses using its presymbolication sidecar."""

    def __init__(self, profile: dict[str, Any], symbols: dict[str, Any]):
        self.profile = profile
        self.by_code_id: dict[str, dict[Any, str]] = {}
        strings = symbols.get("string_table", [])
        for module in symbols.get("data", []):
            code_id = module.get("code_id") or module.get("debug_id")
            if not code_id:
                continue
            symbol_table = module.get("symbol_table", [])
            address_to_name: dict[Any, str] = {}
            for known in module.get("known_addresses", []):
                if not isinstance(known, list) or len(known) != 2:
                    continue
                address, symbol_index = known
                try:
                    string_index = symbol_table[symbol_index]["symbol"]
                    name = strings[string_index]
                except (IndexError, KeyError, TypeError):
                    continue
                address_to_name[_address_key(address)] = name
            self.by_code_id[str(code_id).upper()] = address_to_name

    def frame(self, thread: dict[str, Any], frame_index: int) -> FrameSymbol:
        frame_table = thread["frameTable"]
        function_table = thread["funcTable"]
        resource_table = thread["resourceTable"]
        strings = thread["stringArray"]
        function_index = frame_table["func"][frame_index]
        fallback = strings[function_table["name"][function_index]]
        resource_index = function_table["resource"][function_index]
        if resource_index is None or resource_index < 0:
            return FrameSymbol(fallback, "<unknown>")
        library_index = resource_table["lib"][resource_index]
        library = self.profile["libs"][library_index]
        code_id = str(library.get("codeId", "")).upper()
        address = _address_key(frame_table["address"][frame_index])
        name = self.by_code_id.get(code_id, {}).get(address, fallback)
        return FrameSymbol(name, library.get("name", "<unknown>"))


def _load_json(path: Path, compressed: bool = False) -> Any:
    try:
        if compressed:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                return json.load(source)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Failed to read {path}: {exc}") from exc


def symbols_path(profile_path: Path) -> Path:
    text = str(profile_path)
    if not text.endswith(PROFILE_SUFFIX):
        raise ProfileError(f"Profile must end in {PROFILE_SUFFIX}: {text}")
    return Path(text[: -len(PROFILE_SUFFIX)] + SYMBOL_SUFFIX)


def expand_profile_paths(arguments: list[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    missing = []
    for argument in arguments:
        candidate = Path(argument).expanduser()
        if candidate.is_dir():
            matches = candidate.glob("*" + PROFILE_SUFFIX)
        else:
            matches = (Path(path) for path in glob.glob(str(candidate)))
        found = False
        for match in matches:
            if match.is_file() and str(match).endswith(PROFILE_SUFFIX):
                found = True
                paths[str(match.resolve())] = match
        if not found:
            missing.append(argument)
    if missing:
        raise ProfileError("No profiles matched: " + ", ".join(missing))
    return [paths[key] for key in sorted(paths)]


def _unwind_stack(
    thread: dict[str, Any], resolver: SymbolResolver, stack_index: int
) -> list[FrameSymbol]:
    stack_table = thread["stackTable"]
    frames = []
    visited = set()
    current: int | None = stack_index
    while current is not None:
        if current in visited:
            raise ProfileError(f"Cycle in Samply stack table at index {current}")
        visited.add(current)
        frames.append(resolver.frame(thread, stack_table["frame"][current]))
        current = stack_table["prefix"][current]
    return frames


def sample_weights(cpu_deltas: list[int], attribution: str) -> list[int]:
    if attribution == "current":
        return list(cpu_deltas)
    if attribution != "previous":
        raise ProfileError(f"Unknown delta attribution: {attribution}")
    if not cpu_deltas:
        return []
    weights = [0] * len(cpu_deltas)
    weights[0] = cpu_deltas[0]
    for index in range(len(cpu_deltas) - 1):
        weights[index] += cpu_deltas[index + 1]
    return weights


def classify_neutral_stage(frames: list[FrameSymbol]) -> str:
    if not frames:
        return UNRESOLVED
    names = tuple(frame.name for frame in frames)
    if any(pattern in name for pattern in PRE_SERIALIZER_PATTERNS for name in names):
        return OTHER
    for stage, patterns in STAGE_RULES:
        if any(pattern in name for pattern in patterns for name in names):
            return stage
    return OTHER


def _is_resolved(name: str) -> bool:
    return bool(name) and not name.startswith("0x")


def analyze_capture(path: Path, attribution: str, analysis: Analysis) -> None:
    sidecar = symbols_path(path)
    if not sidecar.is_file():
        raise ProfileError(f"Missing symbol sidecar for {path}: {sidecar}")
    profile = _load_json(path, compressed=True)
    resolver = SymbolResolver(profile, _load_json(sidecar))
    capture_samples = 0
    capture_cpu = 0
    try:
        for thread in profile["threads"]:
            samples = thread["samples"]
            stacks = samples["stack"]
            deltas = samples["threadCPUDelta"]
            if len(stacks) != len(deltas):
                raise ProfileError(
                    f"{path} has {len(stacks)} stacks but {len(deltas)} deltas"
                )
            if samples.get("length", len(stacks)) != len(stacks):
                raise ProfileError(f"{path} has an inconsistent sample length")
            if any(delta < 0 for delta in deltas):
                raise ProfileError(f"{path} contains a negative CPU delta")
            weights = sample_weights(deltas, attribution)
            capture_samples += len(stacks)
            capture_cpu += sum(deltas)
            for stack_index, weight in zip(stacks, weights):
                if weight <= 0:
                    continue
                if stack_index is None:
                    analysis.stages[UNRESOLVED] += weight
                    continue
                frames = _unwind_stack(thread, resolver, stack_index)
                analysis.stages[classify_neutral_stage(frames)] += weight
                leaf = frames[0]
                analysis.flat[leaf] += weight
                if _is_resolved(leaf.name):
                    analysis.resolved_leaf_cpu_us += weight
                for frame in set(frames):
                    analysis.inclusive[frame] += weight
    except (IndexError, KeyError, TypeError) as exc:
        raise ProfileError(f"Malformed Samply profile {path}: {exc}") from exc
    analysis.sample_count += capture_samples
    analysis.cpu_delta_us += capture_cpu
    analysis.captures.append(
        CaptureSummary(str(path), capture_samples, capture_cpu)
    )


def analyze_profiles(
    paths: list[Path], attribution: str = "current"
) -> Analysis:
    analysis = Analysis(delta_attribution=attribution, captures=[])
    for path in paths:
        analyze_capture(path, attribution, analysis)
    if analysis.cpu_delta_us <= 0:
        raise ProfileError("Profiles contain no positive sampled thread CPU")
    if sum(analysis.stages.values()) != analysis.cpu_delta_us:
        raise ProfileError("Neutral stage attribution is not mutually exclusive")
    return analysis


def _percent(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def _sorted_counter(counter: collections.Counter[Any]) -> list[tuple[Any, int]]:
    return sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            getattr(item[0], "name", str(item[0])),
            getattr(item[0], "library", ""),
        ),
    )


def analysis_as_json_value(analysis: Analysis, top_functions: int) -> dict[str, Any]:
    capture_cpu = [capture.cpu_delta_us for capture in analysis.captures]

    def functions(counter: collections.Counter[FrameSymbol]) -> list[dict[str, Any]]:
        return [
            {
                "function": frame.name,
                "library": frame.library,
                "cpu_delta_us": cpu,
                "sampled_cpu_percent": _percent(cpu, analysis.cpu_delta_us),
            }
            for frame, cpu in _sorted_counter(counter)[:top_functions]
        ]

    stage_order = [
        COEFFICIENT_PREPARATION,
        ENTROPY_MODEL,
        TOKEN_EMISSION,
        ASSEMBLY,
        OTHER,
        UNRESOLVED,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "timing_semantics": "sampled-thread-cpu-attribution",
        "delta_attribution": analysis.delta_attribution,
        "summary": {
            "capture_count": len(analysis.captures),
            "sample_count": analysis.sample_count,
            "cpu_delta_us": analysis.cpu_delta_us,
            "resolved_leaf_cpu_percent": _percent(
                analysis.resolved_leaf_cpu_us, analysis.cpu_delta_us
            ),
            "per_capture_cpu_delta_us": {
                "minimum": min(capture_cpu),
                "maximum": max(capture_cpu),
                "mean": statistics.fmean(capture_cpu),
            },
        },
        "captures": [dataclasses.asdict(item) for item in analysis.captures],
        "neutral_stages": [
            {
                "stage": stage,
                "cpu_delta_us": analysis.stages.get(stage, 0),
                "sampled_cpu_percent": _percent(
                    analysis.stages.get(stage, 0), analysis.cpu_delta_us
                ),
            }
            for stage in stage_order
        ],
        "flat_functions": functions(analysis.flat),
        "inclusive_functions": functions(analysis.inclusive),
    }


def render_markdown(analysis: Analysis, top_functions: int) -> str:
    value = analysis_as_json_value(analysis, top_functions)
    summary = value["summary"]
    lines = [
        "# Neutral serializer Samply analysis",
        "",
        f"- Captures: {summary['capture_count']}",
        f"- Samples: {summary['sample_count']}",
        f"- Sampled thread CPU: {summary['cpu_delta_us'] / 1000.0:.3f} ms",
        "- Timing semantics: sampled thread-CPU attribution, not stage wall time",
        f"- Weighted leaf-symbol resolution: {summary['resolved_leaf_cpu_percent']:.2f}%",
        "",
        "## Mutually exclusive neutral stages",
        "",
        "| Stage | CPU delta | Sampled CPU |",
        "|---|---:|---:|",
    ]
    for row in value["neutral_stages"]:
        lines.append(
            f"| {row['stage']} | {row['cpu_delta_us'] / 1000.0:.3f} ms | "
            f"{row['sampled_cpu_percent']:.2f}% |"
        )
    for title, key in (
        ("Hottest leaf functions", "flat_functions"),
        ("Hottest inclusive functions", "inclusive_functions"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| CPU | CPU delta | Function | Library |",
                "|---:|---:|---|---|",
            ]
        )
        for row in value[key]:
            function = row["function"].replace("|", "\\|")
            library = row["library"].replace("|", "\\|")
            lines.append(
                f"| {row['sampled_cpu_percent']:.2f}% | "
                f"{row['cpu_delta_us'] / 1000.0:.3f} ms | "
                f"`{function}` | `{library}` |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+")
    parser.add_argument(
        "--delta-attribution", choices=("current", "previous"), default="current"
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", default="-")
    parser.add_argument("--top-functions", type=int, default=20)
    parser.add_argument(
        "--minimum-resolution-percent",
        type=float,
        default=95.0,
        help="fail below this weighted leaf-symbol resolution (default: 95)",
    )
    args = parser.parse_args(argv)
    if args.top_functions < 0:
        parser.error("--top-functions must be nonnegative")
    if not 0.0 <= args.minimum_resolution_percent <= 100.0:
        parser.error("--minimum-resolution-percent must be between 0 and 100")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        analysis = analyze_profiles(
            expand_profile_paths(args.profiles), args.delta_attribution
        )
        resolution = _percent(
            analysis.resolved_leaf_cpu_us, analysis.cpu_delta_us
        )
        if args.format == "json":
            content = json.dumps(
                analysis_as_json_value(analysis, args.top_functions),
                indent=2,
                sort_keys=True,
            ) + "\n"
        else:
            content = render_markdown(analysis, args.top_functions)
        if args.output == "-":
            sys.stdout.write(content)
        else:
            Path(args.output).write_text(content, encoding="utf-8")
        if resolution < args.minimum_resolution_percent:
            raise ProfileError(
                f"Weighted symbol resolution {resolution:.2f}% is below "
                f"{args.minimum_resolution_percent:.2f}%"
            )
    except (OSError, ProfileError) as exc:
        print(f"Samply analysis error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
