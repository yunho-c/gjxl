#!/usr/bin/env python3
"""Decode each precision variant independently and score with unchanged CPU math."""
import argparse
import json
import os
from pathlib import Path
import statistics
import numpy as np
from precision import ROOT, OUT, run, save, sha

CORPUS = Path("/Users/yunhocho/GitHub/gjxl-libjxl-comparison/build/libjxl-comparison/corpus-phase1-pilot-pinned/canonical")
DECODER = Path("/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl/tools/djxl")

def inputs():
    folder = OUT / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    w, h = 513, 257
    y, x = np.mgrid[:h, :w].astype(np.float64)
    t = x / (w - 1)
    images = {}
    images["smooth"] = np.stack([t, .5*t+.5*y/(h-1), y/(h-1)], axis=-1)
    dark = np.power(10., -6. + 4.5*t)
    images["dark"] = np.stack([dark, dark*(1.+1.e-3*np.sin(y)), dark*.999], axis=-1)
    neutral = .01 + .98*t
    images["neutral"] = np.stack([neutral+2.e-5*np.sin(y), neutral,
                                  neutral+2.e-5*np.cos(x)], axis=-1)
    v = .4 + .18*np.sin(x*.89)*np.cos(y*.83) + .02*np.sin(x*2.9+y*2.7)
    images["texture"] = np.stack([v, v*.9+.02*np.sin(y), .5+.12*np.cos(x+y)], axis=-1)
    v = np.full((h,w), .25); v[::31, ::29] = 1.; v[15::31, 14::29] = 0.
    images["impulse"] = np.stack([v, v, v], axis=-1)
    v = ((x.astype(int)//17+y.astype(int)//13)%2).astype(float)
    images["contrast"] = np.stack([v, 1.-v, .001+.998*v], axis=-1)
    images["flat"] = np.full((h,w,3), .125)
    images["tiny_dark"] = images["dark"][:7,:5,:]
    manifest = []
    for name, image in images.items():
        p = folder / (name + ".pfm")
        header = f"PF\n{image.shape[1]} {image.shape[0]}\n-1.0\n".encode()
        data = header + image[::-1].astype("<f4").tobytes()
        if p.exists() and p.read_bytes() != data:
            raise RuntimeError(f"Input generator changed: {p}")
        p.write_bytes(data)
        manifest.append(dict(name=name, path=str(p), sha256=sha(p), targets=[.1,.3,1.2,3.0]))
    for name in ("kodak-kodim01", "kodak-kodim17", "kodak-kodim23",
                 "imazen26-1029-planter-1080p", "imazen26-1029-planter-4k"):
        p = CORPUS / (name + ".pfm")
        manifest.append(dict(name=name, path=str(p), sha256=sha(p),
                             targets=[.1,.3,1.2,3.] if name.startswith("kodak") else [1.2]))
    save(OUT / "quality-inputs.json", manifest)
    return manifest

def last_json(record):
    rows = [json.loads(line) for line in Path(record["log"]).read_text().splitlines()
            if line.startswith("{")]
    if len(rows) != 1: raise RuntimeError("Expected exactly one JSON result")
    return rows[0]

def quality(variant, screen=False, corpus=False):
    if corpus:
        manifest=json.loads((ROOT/"tools/resident_qualification/corpus.json").read_text())
        cases=[dict(name=Path(e["canonical_path"]).stem,path=str(CORPUS/e["canonical_path"]),
                    sha256=e["canonical_sha256"],targets=[.1,.3,1.2,3.]) for e in manifest["inputs"]]
    else:
        cases = json.loads((OUT / "quality-inputs.json").read_text())
    quality_root=OUT/("quality-corpus" if corpus else "quality")
    folder = quality_root / variant
    folder.mkdir(parents=True, exist_ok=True)
    library = (OUT / "baseline" if variant == "baseline" else OUT / "variants" / variant) / "metal/gjxl.metallib"
    probe = OUT / "baseline/gjxl_metal_precision_probe"
    identity = dict(library=str(library), library_sha256=sha(library), probe_sha256=sha(probe),
                    decoder=str(DECODER), decoder_sha256=sha(DECODER),
                    metric="unchanged native CPU Butteraugli on decoded linear sRGB; intensity 80", cases=cases)
    old = folder / "identity.json"
    if old.exists() and json.loads(old.read_text()) != identity:
        raise RuntimeError("Quality runtime/input identity changed")
    save(old, identity)
    rows = []
    for case in cases:
        if screen and case["name"] not in ("dark", "neutral", "texture", "flat", "kodak-kodim17"):
            continue
        source = Path(case["path"])
        if sha(source) != case["sha256"]: raise RuntimeError("Input changed")
        for distance in case["targets"]:
            if screen and distance not in (.1, 1.2): continue
            name = case["name"] + "-d" + str(distance)
            stem = variant + ("-corpus-" if corpus else "-") + name
            jxl, decoded = folder / (name + ".jxl"), folder / (name + ".pfm")
            record = folder / (name + ".json")
            if record.exists():
                row = json.loads(record.read_text())
                if row["jxl_sha256"] != sha(jxl) or row["decoded_sha256"] != sha(decoded):
                    raise RuntimeError("Quality output changed")
                rows.append(row)
                continue
            encoded = last_json(run(stem + "-encode", [probe, "encode", library,
                source, jxl, distance, 7]))
            run(stem + "-decode", [DECODER, jxl, decoded, "--color_space=RGB_D65_SRG_Rel_Lin"])
            # Identical codestreams decode identically; reuse the independent
            # baseline metric only after checking both hashes.
            baseline = quality_root / "baseline" / (name + ".json")
            base = json.loads(baseline.read_text()) if baseline.exists() else None
            exact = bool(base and base["jxl_sha256"] == sha(jxl)
                         and base["decoded_sha256"] == sha(decoded))
            same_pixels = bool(base and base["decoded_sha256"] == sha(decoded))
            metric = base["metric"] if same_pixels else last_json(run(stem + "-metric", [probe, "metric", source, decoded]))
            if same_pixels:
                # Decoding has already completed independently. Share immutable
                # byte-identical PFM payloads to bound the study's disk usage.
                alias = decoded.with_suffix(".dedup")
                os.link(baseline.with_suffix(".pfm"), alias)
                os.replace(alias, decoded)
            row = dict(name=name, source_sha256=case["sha256"], distance=distance,
                encoded=encoded, metric=metric, jxl_sha256=sha(jxl), decoded_sha256=sha(decoded),
                baseline_identical=exact, decoded_identical=same_pixels)
            if base:
                delta = metric["butteraugli"] - base["metric"]["butteraugli"]
                budget = max(.002, .01 * base["metric"]["butteraugli"])
                row.update(score_delta=delta, score_budget=budget, quality_pass=delta <= budget,
                           size_ratio=encoded["bytes"] / base["encoded"]["bytes"])
            save(record, row)
            rows.append(row)
    save(folder / ("screen-summary.json" if screen else "summary.json"), dict(
        variant=variant, cases=len(rows), exact=sum(r["baseline_identical"] for r in rows),
        quality_failures=[r["name"] for r in rows if not r.get("quality_pass", True)], rows=rows))
    print(variant, len(rows), "quality failures", [r["name"] for r in rows if not r.get("quality_pass", True)], flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--corpus", action="store_true")
    args = parser.parse_args()
    if args.variant == "inputs": inputs()
    else: quality(args.variant, args.screen, args.corpus)
