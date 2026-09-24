#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
# Physical-device qualification of the current committed integration checkout.
set -euo pipefail

source_dir="$(git rev-parse --show-toplevel)"
cd "$source_dir"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo 'Use a clean committed checkout for physical Metal qualification.' >&2
  exit 1
fi
build_dir="${1:-$source_dir/build/metal-integration-qualification}"
evidence_dir="$build_dir/evidence/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$evidence_dir"
git rev-parse HEAD > "$evidence_dir/revision.txt"
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
{
  sw_vers
  xcrun clang++ --version
  xcrun --sdk macosx metal --version
  xcrun swift -e '
    import Darwin
    import Metal
    guard let device = MTLCreateSystemDefaultDevice() else {
      fputs("A Metal device is required.\n", stderr); exit(1)
    }
    print("Metal device: \(device.name)")
    if device.name.lowercased().contains("paravirtual") {
      fputs("Physical Metal qualification rejects Apple Paravirtual.\n", stderr); exit(1)
    }
  '
} 2>&1 | tee "$evidence_dir/hardware.log"

cmake -S "$source_dir" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_ENABLE_METAL=ON -DGJXL_ENABLE_CUDA=OFF \
  -DGJXL_BUILD_TESTS=ON -DGJXL_BUILD_BENCHMARKS=ON \
  -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF 2>&1 | tee "$evidence_dir/configure.log"
cmake --build "$build_dir" --parallel 4 2>&1 | tee "$evidence_dir/build.log"

# Require real stage timestamps; physical Apple GPUs need not expose dispatch
# timestamps (the recorded M4 Pro does not). The benchmark also checks ordinary
# versus profiled codestream bytes and complete summaries before publication.
capture_profile() {
  local mode="$1"
  "$build_dir/gjxl_encoding_benchmark" --scope metal-public-workflow \
    --workload synthetic_128x96 --validation metal-only \
    --gpu-aq fully-resident --effort 8 --warmups 1 --samples 2 \
    --gpu-profile "$mode" --gpu-profile-output "$evidence_dir/$mode.json" 2>&1 \
    | tee "$evidence_dir/profile-$mode.log"
}
capture_profile stage
dispatch_supported="$(python3 - "$evidence_dir/stage.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    profile = json.load(stream)
samples = profile["workloads"][0]["samples"]
if len(samples) != 2:
    sys.exit("Expected two physical Metal stage captures")
capabilities = samples[0]["capabilities"]
if not (capabilities["timestamp_counter"] and capabilities["stage_boundary"]):
    sys.exit("Physical Metal qualification requires stage timestamp support")
if any(sample["capabilities"] != capabilities for sample in samples):
    sys.exit("Metal timestamp capabilities changed between samples")
print("true" if capabilities["dispatch_boundary"] else "false")
PY
)"
if [[ "$dispatch_supported" == true ]]; then
  capture_profile dispatch
  echo 'Stage and dispatch timestamp capture supported and exercised.' \
    | tee "$evidence_dir/timestamp-scope.txt"
else
  # The full CLI suite below checks explicit rejection and output preservation.
  # This result does not qualify dispatch capture on hardware that supports it.
  echo 'Stage timestamp capture exercised; dispatch timestamps unsupported.' \
    | tee "$evidence_dir/timestamp-scope.txt"
fi
ctest --test-dir "$build_dir" --output-on-failure --parallel 1 2>&1 \
  | tee "$evidence_dir/ctest.log"
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  ctest --test-dir "$build_dir" --output-on-failure --parallel 1 \
    -R '^(metal_aq_reconstruction|metal_aq_evaluation|metal_quantization_pipeline|metal_dc_processing|metal_submission_storage_plan|metal_ac_strategy_search|metal_aq_strategy_metadata)$' 2>&1 \
  | tee "$evidence_dir/metal-validation.log"

"$build_dir/gjxl_metal_qualification" --synthetic 3839x2159 --count 7 \
  --limit tight --cpu-limit 3 | tee "$evidence_dir/pressure-4k.jsonl"
"$build_dir/gjxl_metal_qualification" --synthetic 129x133 --synthetic 257x263 \
  --batch 4 --in-flight 2 --callers 2 --count 7 --limit full --cpu-limit 3 \
  | tee "$evidence_dir/pressure-concurrent.jsonl"

git diff --exit-code
git diff --cached --exit-code
test "$(git rev-parse HEAD)" = "$(cat "$evidence_dir/revision.txt")"
printf 'PASS physical Metal integration qualification\n' \
  | tee "$evidence_dir/complete.txt"
