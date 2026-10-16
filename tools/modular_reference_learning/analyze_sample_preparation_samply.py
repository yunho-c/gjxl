#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yunho Cho
"""Attribute sampled preparation CPU across all workers, including inline frames.

Requires Samply's presymbolicated sidecar and the frozen executable/dSYM. CPU
delta assignments to current and preceding stacks are both retained. Inclusive
function totals overlap; the nearest preparer source location is disjoint.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import subprocess

from analyze_fitting_samply import resolve, sample_weights
from run_hybrid_qualification import digest, require


def canonical_name(name):
    # Samply and LLVM differ in return types, template arguments and spacing.
    # Avoid reporting two inclusive rows for the same preparer function.
    function = re.search(r"gjxl::modular_internal::(?:HistogramSampleBuilder::|PrepareHistogramSamples\()", name)
    if function:
        name = name[function.start():]
    name = re.sub(r"\b([0-9]+)ul\b", r"\1", re.sub(r"\(unsigned long\)(?=[0-9])", "", name))
    while re.search(r">\s+>", name):
        name = re.sub(r">\s+>", ">>", name)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbolizer", default="llvm-symbolizer")
    args = parser.parse_args()
    with gzip.open(args.profile, "rt") as stream:
        profile = json.load(stream)
    sidecar = Path(str(args.profile).removesuffix(".json.gz") + ".json.syms.json")
    symbols = json.loads(sidecar.read_text())
    require(profile["meta"]["sampleUnits"]["threadCPUDelta"] == "µs", "Unknown CPU units")
    threads, addresses = [], set()
    for thread in profile["threads"]:
        frames = resolve(profile, symbols, thread)
        table = thread["stackTable"]
        stacks, inside = {}, {}
        for index in set(thread["samples"]["stack"]):
            stack, seen, current = [], set(), index
            while current is not None:
                require(current not in seen, "Cyclic stack table")
                seen.add(current)
                stack.append(frames[table["frame"][current]])
                current = table["prefix"][current]
            stacks[index] = stack
            inside[index] = any("gjxl::modular_internal::PrepareHistogramSamples(" in f["name"]
                                for f in stack)
            if inside[index]:
                for frame in stack:
                    lib = frame["lib"]
                    if lib is not None and lib >= 0 and profile["libs"][lib]["name"] == args.binary.name:
                        addresses.add(frame["address"])
        threads.append((thread, stacks, inside))
    addresses = sorted(addresses)
    require(addresses, "No preparation addresses found")
    commands = subprocess.check_output(["otool", "-l", str(args.binary)], text=True)
    match = re.search(r"segname __TEXT\s+vmaddr (0x[0-9a-fA-F]+)", commands)
    require(match is not None, "No Mach-O image base")
    base = int(match.group(1), 16)
    command = [args.symbolizer, "--inlines", "--output-style=JSON",
               "--obj=" + str(args.binary.resolve())]
    output = subprocess.check_output(command, input="".join(f"0x{a+base:x}\n" for a in addresses), text=True)
    resolved = [json.loads(line) for line in output.splitlines()]
    require(len(resolved) == len(addresses), "Symbolizer lost addresses")
    inline = dict(zip(addresses, resolved))
    source = args.source.read_text().splitlines()
    result = dict(scope="Sampled CPU inside PrepareHistogramSamples across all workers. Inclusive function rows overlap; nearest preparer source rows are disjoint. Not caller wall time or GPU execution time.",
                  profile_sha256=digest(args.profile), sidecar_sha256=digest(sidecar),
                  binary_sha256=digest(args.binary), source_sha256=digest(args.source),
                  image_base=base, symbolizer_command=command, inline_symbols=resolved,
                  attribution={})
    for previous in [False, True]:
        flat, inclusive, lines = Counter(), Counter(), Counter()
        cpu = samples = all_cpu = native_cpu = inline_cpu = active_threads = 0
        for thread, stacks, inside in threads:
            weights = sample_weights(thread["samples"]["threadCPUDelta"], previous)
            require(len(weights) == thread["samples"]["length"], "Wrong sample weights")
            seen = False
            for index, weight in zip(thread["samples"]["stack"], weights):
                all_cpu += weight
                if not inside[index] or weight <= 0:
                    continue
                cpu += weight
                samples += 1
                seen = True
                stack = stacks[index]
                names = {canonical_name(f["name"]) for f in stack}
                leaf_info, location = [], None
                for n, frame in enumerate(stack):
                    lib = frame["lib"]
                    native = lib is not None and lib >= 0 and profile["libs"][lib]["name"] == args.binary.name
                    info = inline.get(frame["address"], {}).get("Symbol", []) if native else []
                    if n == 0:
                        leaf_info = info
                        if native:
                            native_cpu += weight
                            if info and info[0].get("FileName") and info[0].get("FunctionName"):
                                inline_cpu += weight
                    for item in info:
                        if item.get("FunctionName"):
                            names.add(canonical_name(item["FunctionName"]))
                        if location is None and item.get("Line", 0) > 0 and item.get("FileName", "").endswith("histogram_sample_preparation.cpp"):
                            location = item.get("Line", 0)
                flat[(leaf_info[0].get("FunctionName") if leaf_info else None) or stack[0]["name"]] += weight
                lines[location or 0] += weight
                for name in names:
                    if "HistogramSampleBuilder::" in name or "PrepareHistogramSamples(" in name:
                        inclusive[name] += weight
            active_threads += seen
        require(cpu and sum(flat.values()) == cpu and sum(lines.values()) == cpu,
                "Preparation sample conservation failed")
        require(native_cpu and inline_cpu / native_cpu >= .95,
                "Insufficient native symbol resolution; check frozen dSYM")
        rows = lambda c: [dict(name=n, cpu_ms=v/1000, percent=100*v/cpu)
                          for n, v in c.most_common(60)]
        result["attribution"]["previous" if previous else "current"] = dict(
            preparation_cpu_ms=cpu/1000, process_sampled_cpu_ms=all_cpu/1000,
            positive_samples=samples, active_threads=active_threads,
            native_symbol_resolution_percent=100*inline_cpu/native_cpu,
            preparer_source_resolution_percent=100*(cpu-lines[0])/cpu,
            flat=rows(flat), inclusive=rows(inclusive),
            nearest_preparer_source=[dict(line=n, text=source[n-1] if 0<n<=len(source) else "unresolved",
                                         cpu_ms=v/1000, percent=100*v/cpu)
                                     for n, v in lines.most_common()])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, a in result["attribution"].items():
        print(name, a["preparation_cpu_ms"], "ms", a["positive_samples"], "samples")
        for row in a["inclusive"][:12]:
            print(f"{row['percent']:.1f}%", row["name"])


if __name__ == "__main__":
    main()
