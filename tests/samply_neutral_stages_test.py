#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in third_party/libjxl/LICENSE.

"""Tests for mutually exclusive neutral Samply stage attribution."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import samply_neutral_stages as stages


def write_capture(root: Path, stacks: list[list[str] | None], deltas: list[int]) -> Path:
    symbols = []
    for stack in stacks:
        for symbol in stack or []:
            if symbol not in symbols:
                symbols.append(symbol)
    addresses = {symbol: 0x1000 + 16 * index for index, symbol in enumerate(symbols)}
    frame_symbols = []
    stack_frames = []
    stack_prefixes = []
    sample_stacks = []
    for stack in stacks:
        if stack is None:
            sample_stacks.append(None)
            continue
        prefix = None
        for symbol in reversed(stack):
            frame_index = len(frame_symbols)
            frame_symbols.append(symbol)
            stack_index = len(stack_frames)
            stack_frames.append(frame_index)
            stack_prefixes.append(prefix)
            prefix = stack_index
        sample_stacks.append(prefix)
    profile = {
        "libs": [{"name": "comparison-test.dylib", "codeId": "TEST-ID"}],
        "threads": [
            {
                "samples": {
                    "length": len(stacks),
                    "stack": sample_stacks,
                    "threadCPUDelta": deltas,
                },
                "stackTable": {"frame": stack_frames, "prefix": stack_prefixes},
                "frameTable": {
                    "func": list(range(len(frame_symbols))),
                    "address": [addresses[symbol] for symbol in frame_symbols],
                },
                "funcTable": {
                    "name": list(range(len(frame_symbols))),
                    "resource": [0] * len(frame_symbols),
                },
                "resourceTable": {"lib": [0]},
                "stringArray": [
                    f"fallback-{index}" for index in range(len(frame_symbols))
                ],
            }
        ],
    }
    sidecar = {
        "string_table": symbols,
        "data": [
            {
                "code_id": "TEST-ID",
                "known_addresses": [
                    [addresses[symbol], index] for index, symbol in enumerate(symbols)
                ],
                "symbol_table": [{"symbol": index} for index in range(len(symbols))],
            }
        ],
    }
    path = root / "capture.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        json.dump(profile, output)
    (root / "capture.json.syms.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return path


class NeutralStageTest(unittest.TestCase):
    def test_rules_are_ordered_and_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gjxl-samply-test-") as temporary:
            path = write_capture(
                Path(temporary),
                [
                    ["TokenizeCoefficients", "EncodeGroups", "main"],
                    ["ComputeAllCoeffOrders", "EncodeFrame", "main"],
                    ["WriteTokens", "BuildAndEncodeHistograms", "main"],
                    ["AppendByteAligned", "WriteTokenStream", "main"],
                    ["AppendData", "EncodeFrame", "main"],
                    ["EstimateEntropy", "ProcessRectACS", "main"],
                    ["UnknownRuntime", "main"],
                    None,
                ],
                [100, 150, 200, 300, 400, 500, 600, 700],
            )
            analysis = stages.analyze_profiles([path])

            self.assertEqual(analysis.stages[stages.COEFFICIENT_PREPARATION], 250)
            self.assertEqual(analysis.stages[stages.ENTROPY_MODEL], 200)
            self.assertEqual(analysis.stages[stages.TOKEN_EMISSION], 300)
            self.assertEqual(analysis.stages[stages.ASSEMBLY], 400)
            self.assertEqual(analysis.stages[stages.OTHER], 1100)
            self.assertEqual(analysis.stages[stages.UNRESOLVED], 700)
            self.assertEqual(sum(analysis.stages.values()), analysis.cpu_delta_us)
            value = stages.analysis_as_json_value(analysis, 5)
            self.assertEqual(
                value["timing_semantics"], "sampled-thread-cpu-attribution"
            )

    def test_previous_delta_attribution_is_retained(self) -> None:
        self.assertEqual(stages.sample_weights([100, 300], "current"), [100, 300])
        self.assertEqual(stages.sample_weights([100, 300], "previous"), [400, 0])


if __name__ == "__main__":
    unittest.main()
