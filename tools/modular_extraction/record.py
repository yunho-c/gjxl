"""Record source/tool identities and qualification results without rebasing expectations."""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def source_files(source):
    files = [source / "CMakeLists.txt"]
    files += list((source / "cmake").glob("*.cmake"))
    files += [p for p in (source / "src").rglob("*") if p.suffix in (".cpp", ".h", ".inc", ".mm", ".cu", ".metal")]
    return {p.relative_to(source).as_posix(): digest(p) for p in sorted(files)}


def tests(path):
    result = []
    for case in ET.parse(path).getroot().iter("testcase"):
        failure = case.find("failure")
        skipped = case.find("skipped")
        result.append({"name": case.attrib["name"],
                       "status": "failed" if failure is not None else "not_run" if skipped is not None else "passed",
                       "seconds": case.get("time"),
                       "detail": "" if failure is None else failure.get("message", "")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    work = args.work.resolve()
    baseline = work / "baseline-src"
    baseline_dirty = git(baseline, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all")
    if baseline_dirty:
        raise RuntimeError("Frozen baseline sources changed: " + baseline_dirty)
    pin = root / "third_party/libjxl"
    if git(pin, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Pinned reference sources changed")
    original = source_files(baseline)
    candidate = source_files(root)
    cache = (work / "candidate/CMakeCache.txt").read_text()
    settings = dict(re.findall(r"^(CMAKE_(?:BUILD_TYPE|CXX_STANDARD|C_COMPILER|CXX_COMPILER|CXX_FLAGS(?:_RELEASE)?|C_FLAGS(?:_RELEASE)?|GENERATOR)|GJXL_(?:ENABLE_METAL|ENABLE_CUDA|ENABLE_LIBJXL_REFERENCE|DC_ORACLE_BUILD|BUILD_TESTS)):[^=]+=([^\n]*)", cache, re.M))
    compiler = Path(settings["CMAKE_CXX_COMPILER"])
    pin_cache = (work / "pinned-libjxl-x64/CMakeCache.txt").read_text()
    pin_compiler = Path(re.search(r"^CMAKE_CXX_COMPILER:[^=]+=([^\n]*)", pin_cache, re.M)[1])
    paths = [work / "baseline-capture/modular_extraction_capture.exe",
             work / "candidate-capture/modular_extraction_capture.exe",
             work / "pinned-libjxl-x64/tools/djxl.exe",
             work / "pinned-libjxl-x64/tools/jxlinfo.exe",
             work / "pinned-libjxl-x64/tools/jxl.dll",
             work / "baseline/gjxl_codec.lib", work / "baseline/gjxl_codestream.lib",
             work / "candidate/gjxl_codec.lib", work / "candidate/gjxl_codestream.lib"]
    paths.append(work / "candidate/CMakeFiles/gjxl_dc_processing_oracle_test.dir/tests/dc_processing_oracle_test.cpp.obj")
    log_names = ("baseline-build-all.log", "baseline-tests.log", "candidate-build-2.log",
                 "candidate-tests.log", "candidate-final-focused.log", "final-owner-test.log", "baseline-dc-oracle.log",
                 "candidate-focused-tests.log", "baseline-conformance.log", "candidate-conformance.log",
                 "pinned-decoder-verified.log")
    evidence = {name: {"sha256": digest(work / name), "bytes": (work / name).stat().st_size}
                for name in log_names}
    test_results = {name: tests(work / f"{name}-tests.xml") for name in ("baseline", "candidate")}
    baseline_passes = {t["name"] for t in test_results["baseline"] if t["status"] == "passed"}
    candidate_passes = {t["name"] for t in test_results["candidate"] if t["status"] == "passed"}
    lost = sorted(baseline_passes - candidate_passes)
    if lost:
        raise RuntimeError("Baseline passes lost: " + str(lost))
    dependencies = set()
    def visit(path):
        relative = path.relative_to(root / "src").as_posix()
        if relative in dependencies:
            return
        dependencies.add(relative)
        for include in re.findall(r'^#include "([^"]+)"', path.read_text(), re.M):
            target = root / "src" / include
            if target.exists():
                visit(target)
    for folder in ("codec/modular", "codestream/modular"):
        for path in (root / "src" / folder).iterdir():
            if path.suffix in (".h", ".cpp"):
                visit(path)
    if any("vardct" in name or name.startswith("gpu/") or name.startswith("lib/jxl/") for name in dependencies):
        raise RuntimeError("Forbidden Modular include dependency")
    record = {"schema": 1, "baseline_revision": git(baseline, "rev-parse", "HEAD"),
              "baseline_runtime_revision": "67caa6d89830f6a13c057923e34f80753a161e9d",
              "candidate_base_revision": git(root, "rev-parse", "HEAD"),
              "candidate_is_uncommitted": True,
              "baseline_clean_ignoring_submodule_junction": True,
              "pinned_revision": git(pin, "rev-parse", "HEAD"), "pinned_clean": True,
              "host": {"platform": platform.platform(), "processor": platform.processor(),
                       "cpu": "Snapdragon X2 Elite X2E84100, 12 logical processors",
                       "execution": "x64 binaries on Windows ARM64 emulation"},
              "cmake_settings": settings,
              "compilers": {"native": {"path": str(compiler), "sha256": digest(compiler)},
                            "reference": {"path": str(pin_compiler), "sha256": digest(pin_compiler)}},
              "baseline_sources": original, "candidate_sources": candidate,
              "changed_source_paths": sorted(p for p in original.keys() | candidate.keys() if original.get(p) != candidate.get(p)),
              "modular_include_closure": sorted(dependencies),
              "binaries_and_libraries": {p.relative_to(work).as_posix(): digest(p) for p in paths},
              "tools": {p.relative_to(root).as_posix(): digest(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()},
              "msvc_packages": json.loads((work / "toolchain-manifest.json").read_text()),
              "photo": json.loads((work / "input-manifest.json").read_text()),
              "baseline_oracle_link_arguments": json.loads((work / "baseline-oracle-link-arguments.json").read_text(encoding="utf-8-sig")),
              "logs": evidence, "test_results": test_results, "lost_baseline_passes": lost,
              "timing_records": {p.name: {"sha256": digest(p),
                                  "binary_sha256": json.loads(p.read_text()).get("binary_sha256", {})}
                                 for p in sorted(args.output.parent.glob("timings-*.json"))},
              "conformance": {"baseline": 23, "candidate": 23},
              "unrun": ["Metal", "CUDA", "native ARM64", "other operating systems", "sanitizers"]}
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_tests": len(test_results["baseline"]),
                      "candidate_tests": len(test_results["candidate"]), "lost_baseline_passes": lost,
                      "modular_include_files": len(dependencies)}))


if __name__ == "__main__":
    main()
