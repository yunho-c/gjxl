#!/usr/bin/env python3
"""Render DC visual review artifacts, or report retained measurements only."""
import argparse
import csv
import fcntl
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import zlib

import numpy as np
from PIL import Image, ImageDraw

import effort_qualification as q


def png16(path, pixels):
    """Lossless 16-bit sRGB review image; analysis uses original float arrays."""
    a = np.asarray(np.clip(pixels, 0, 1) * 65535 + .5, dtype=">u2")
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\0" + row.tobytes() for row in a)
    header = struct.pack(">IIBBBBB", a.shape[1], a.shape[0], 16, 2, 0, 0, 0)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"sRGB", b"\0") +
                           chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def diagnostic(reference, decoded):
    error = q.srgb(decoded) - q.srgb(reference)
    # Interior 8x8 means emphasize DC-scale errors; never a perceptual metric.
    h, w = error.shape[:2]
    blocks = error[:h//8*8, :w//8*8].reshape(h//8, 8, w//8, 8, 3).mean(axis=(1, 3))[1:-1, 1:-1]
    rough = np.concatenate([np.diff(blocks, n=2, axis=1).ravel(), np.diff(blocks, n=2, axis=0).ravel()])
    rg = blocks[:, :, 0] - blocks[:, :, 1]
    rg_rough = np.concatenate([np.diff(rg, n=2, axis=1).ravel(), np.diff(rg, n=2, axis=0).ravel()])
    return {"srgb_rmse": float(np.sqrt(np.mean(error**2))),
            "block_error_roughness": float(np.sqrt(np.mean(rough**2))),
            "red_green_error_roughness": float(np.sqrt(np.mean(rg_rough**2)))}


def review(out, m):
    q.verify(m)
    study = q.helpers(out)
    index = []
    gallery = out / "review"
    gallery.mkdir(exist_ok=True)
    for case in m["visual_cases"]:
        source_record = out / "visual" / case["id"] / "complete.json"
        if not source_record.exists():
            continue
        folder = gallery / case["id"]
        folder.mkdir(exist_ok=True)
        if (folder / "review.json").exists():
            index.append(q.read(folder / "review.json")); continue
        reference = q.pfm(case["image"]["path"])
        png16(folder / "reference.png", q.srgb(reference))
        record = q.read(source_record)
        rows = dict(record["rows"])
        selected = out / "matched-size" / case["id"] / "selection.json"
        if selected.exists() and q.read(selected)["complete"]:
            rows["both-matched-size"] = q.read(selected)["both"]
        reviews = {}
        for name, row in rows.items():
            q.check(q.sha(row["artifact"]) == row["sha256"], "Codestream changed")
            path = folder / (name + ".pfm")
            study.run([m["djxl"], "--quiet", "--num_threads=1", "--bits_per_sample=32",
                       "--color_space=RGB_D65_SRG_Rel_Lin", row["artifact"], path], folder, "review-decode-" + name)
            identity = study.pixel_identity(path, case["image"]["width"], case["image"]["height"])
            q.check(identity == row["decoded"], "Independent repeat decode changed pixels")
            pixels = q.pfm(path)
            display = q.srgb(pixels)
            png16(folder / (name + ".png"), display)
            q.picture(folder / (name + "-error40.png"), .5 + 40 * (display - q.srgb(reference)))
            reviews[name] = {**diagnostic(reference, pixels), "bytes": row["encoded_bytes"],
                             "distance": row["distance"], "quality": row["quality"], "decoded": identity,
                             "png_sha256": q.sha(folder / (name + ".png"))}
            path.unlink()  # Only this invocation's regenerable decode scratch.
        item = {"case": case, "modes": reviews, "error_multiplier": 40,
                "scope": "16-bit sRGB review PNGs; error diagnostics are not perceptual metrics"}
        q.save(folder / "review.json", item)
        index.append(item)
        print("REVIEW", case["id"], flush=True)
    q.save(gallery / "index.json", index)
    # Self-contained index data, with local relative PNG links and native-size view.
    data = json.dumps([{"id": r["case"]["id"], "modes": r["modes"]} for r in index])
    document = '''<!doctype html><meta charset="utf-8"><title>GJXL DC visual qualification</title>
<style>body{font:16px system-ui;background:#ddd;color:#111;margin:24px}select,button{font:inherit;margin:4px;padding:6px}img{max-width:100%;display:block}body.native img{max-width:none}#stats{white-space:pre-wrap}#viewer{overflow:auto;max-height:82vh;background:#888}p{max-width:1000px}</style>
<h1>DC visual qualification</h1><p>Independent float decoding, rendered as 16-bit sRGB PNG. Choose a case and alternate modes at the same view position. Same-distance comparisons change bytes; “both-matched-size” meets the declared 1% file-size tolerance. Error ×40 is an amplified diagnostic, not the visible output. Browser scaling and your display can mask or introduce banding.</p>
<select id="cases"></select><select id="modes"></select><button id="toggle">Default ↔ both</button><button id="size">Native size</button><label><input type="checkbox" id="error">Error ×40</label><p id="stats"></p><div id="viewer"><img id="image"></div>
<script>const data=DATA;const cases=document.querySelector('#cases'),modes=document.querySelector('#modes'),err=document.querySelector('#error');
for(const r of data)cases.add(new Option(r.id,r.id));
function row(){return data.find(r=>r.id===cases.value)}
function draw(){const r=row(),m=modes.value;document.querySelector('#image').src=r.id+'/'+m+(err.checked&&m!=='reference'?'-error40':'')+'.png';const v=r.modes[m];document.querySelector('#stats').textContent=v?`${m}: ${v.bytes.toLocaleString()} bytes · distance ${v.distance.toFixed(5)} · SSIMULACRA2 ${v.quality.toFixed(5)}`:'Reference input';}
function change(){const old=modes.value;modes.replaceChildren();for(const n of ['reference',...Object.keys(row().modes)])modes.add(new Option(n,n));modes.value=Object.hasOwn(row().modes,old)?old:'default';draw();}
cases.onchange=change;modes.onchange=draw;err.onchange=draw;document.querySelector('#toggle').onclick=()=>{modes.value=modes.value==='default'?'both':'default';draw()};document.querySelector('#size').onclick=()=>document.body.classList.toggle('native');change();</script>'''.replace("DATA", data)
    (gallery / "index.html").write_text(document)


def paired(record, a="default", b="both"):
    samples = record["encoder"]["samples"]
    rounds = {}
    for s in samples:
        rounds.setdefault(s["round"], {})[s["variant"]] = s["elapsed_nanoseconds"]
    ratios = [r[b] / r[a] for r in rounds.values()]
    return {"default_ms": statistics.median(record["rows"][a]["samples_ns"])/1e6,
            "both_ms": statistics.median(record["rows"][b]["samples_ns"])/1e6,
            "paired_change_pct": 100*(statistics.median(ratios)-1),
            "paired_min_pct": 100*(min(ratios)-1), "paired_max_pct": 100*(max(ratios)-1),
            "slower_rounds": sum(r > 1 for r in ratios), "rounds": len(ratios)}


def csv_file(path, rows):
    if rows:
        with Path(path).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def figures(out, destination, supplement=None):
    """Native-resolution crops; display copies are 8-bit, originals are 16-bit."""
    target = destination / "figures"
    target.mkdir(exist_ok=True)
    specifications = [
        ("alpine-e4-d6-modes", "alpine_lake-visual-e4-d6", ["default", "quantize", "smooth", "both"], 1300, 0, False),
        ("alpine-e4-d6-matched-size", "alpine_lake-visual-e4-d6", ["reference", "default", "both", "both-matched-size"], 1300, 0, False),
        ("alpine-e4-d1_2-modes", "alpine_lake-visual-e4-d1.2", ["default", "quantize", "smooth", "both"], 1300, 0, False),
        ("campus-e4-d3-matched-size", "campus_interior-visual-e4-d3", ["reference", "default", "both", "both-matched-size"], 768, 300, False),
        ("gray-e4-d6-error", "gray-gradient-e4-d6", ["default", "quantize", "smooth", "both"], 768, 300, True),
        ("dark-e4-d6-error", "dark-gradient-e4-d6", ["default", "quantize", "smooth", "both"], 768, 300, True),
        ("sky-e4-d6-error", "sky-gradient-e4-d6", ["default", "quantize", "smooth", "both"], 768, 300, True),
        ("chroma-e4-d6-error", "chroma-gradient-e4-d6", ["default", "quantize", "smooth", "both"], 768, 300, True),
        ("sky-e4-d6-modes", "sky-gradient-e4-d6", ["default", "quantize", "smooth", "both"], 768, 300, False),
        ("chroma-e4-d6-modes", "chroma-gradient-e4-d6", ["default", "quantize", "smooth", "both"], 768, 300, False),
    ]
    records = []
    for title, case, modes, x, y, error in specifications:
        root = supplement if supplement is not None and "gradient" in case else out
        folder = root / "review" / case
        if not (folder / "review.json").exists():
            continue
        suffix = "-error40.png" if error else ".png"
        sheet = Image.new("RGB", (1120, 844), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for j, mode in enumerate(modes):
            path = folder / (mode + suffix)
            im = Image.open(path).convert("RGB").crop((x, y, x+544, y+384))
            left, top = 8+(j%2)*560, 34+(j//2)*416
            sheet.paste(im, (left, top))
            draw.text((left, top-22), mode + (" (error x40)" if error else ""), fill="black")
        path = target / (title + ".png")
        sheet.save(path)
        records.append({"path":str(path),"case":case,"modes":modes,"crop":[x,y,544,384],
                        "error_multiplier":40 if error else None,"sha256":q.sha(path)})
    q.save(target / "manifest.json", records)


def report(out, m, destination, supplement=None):
    destination.mkdir(parents=True, exist_ok=True)
    if supplement is not None:
        extra = q.read(supplement / "manifest.json")
        q.check(extra["revision"] == m["revision"], "Supplement revision differs")
        expected = {c["id"] for c in m["visual_cases"] if c["image"]["kind"] == "synthetic-gradient"}
        q.check({c["id"] for c in extra["visual_cases"]} == expected, "Supplement case matrix differs")
        for key in ("size_tolerance", "max_size_probes", "distance_bounds", "scorer_version", "threads", "backend"):
            q.check(extra[key] == m[key], "Supplement protocol differs: " + key)
    def case_root(case):
        return supplement if supplement is not None and case["image"]["kind"] == "synthetic-gradient" else out
    times, missing, visual, matches, unresolved, controls, transitions = [], [], [], [], [], [], []
    for c in m["timing_cases"]:
        p = out / "timing" / c["id"] / "complete.json"
        if not p.exists():
            missing.append(c["id"]); continue
        r = q.read(p); q.helpers(out).audit(r)
        a,b=r["rows"]["default"],r["rows"]["both"]
        times.append({"case": c["id"], "photo": c["image"]["photo"], "size_class": c["image"]["size_class"],
                      "effort": c["effort"], **paired(r), "default_bytes": a["encoded_bytes"], "both_bytes": b["encoded_bytes"],
                      "size_change_pct": 100*(b["encoded_bytes"]/a["encoded_bytes"]-1),
                      "default_quality": a["quality"], "both_quality": b["quality"], "quality_delta": b["quality"]-a["quality"],
                      "background_cpu_before": r["machine_before"]["cpu_total"], "background_cpu_after": r["machine_after"]["cpu_total"]})
    for p in sorted((out / "controls").glob("*/complete.json")):
        controls.append({"case": str(p.parent.name), **paired(q.read(p), "control_a", "control_b")})
    for c in m["visual_cases"]:
        root = case_root(c)
        p = root / "visual" / c["id"] / "complete.json"
        if p.exists():
            record = q.read(p); q.helpers(out).audit(record)
            for name,r in record["rows"].items():
                visual.append({"case": c["id"], "image": c["image"]["name"], "kind": c["image"]["kind"],
                               "effort": c["effort"], "distance": c["distance"], "mode": name,
                               "bytes": r["encoded_bytes"], "quality": r["quality"]})
        p = root / "matched-size" / c["id"] / "selection.json"
        if p.exists():
            s = q.read(p); a,b=s["default"],s["both"]
            row = {"case": c["id"], "image": c["image"]["name"], "kind": c["image"]["kind"], "effort": c["effort"],
                   "default_distance": c["distance"], "both_distance": b["distance"], "default_bytes": a["encoded_bytes"],
                   "both_bytes": b["encoded_bytes"], "size_error_pct": 100*s["relative_size_error"],
                   "quality_delta": b["quality"]-a["quality"], "complete": s["complete"], "reason": s["reason"], "probes": s["probes"]}
            matches.append(row)
            if not s["complete"]: unresolved.append(row)
    diagnostics = []
    for c in m["visual_cases"]:
        if c["effort"] != 4:
            continue
        root = case_root(c)
        p3 = root / "visual" / f"{c['image']['name']}-e3-d{c['distance']:g}" / "complete.json"
        p4 = root / "visual" / c["id"] / "complete.json"
        if not p3.exists() or not p4.exists():
            continue
        a = q.read(p3)["rows"]["default"]
        fourth = q.read(p4)["rows"]
        b, candidate = fourth["default"], fourth["both"]
        transitions.append({"image":c["image"]["name"],"kind":c["image"]["kind"],"distance":c["distance"],
                            "e3_default_bytes":a["encoded_bytes"],"e4_default_bytes":b["encoded_bytes"],"e4_both_bytes":candidate["encoded_bytes"],
                            "e3_default_quality":a["quality"],"e4_default_quality":b["quality"],"e4_both_quality":candidate["quality"],
                            "existing_transition_quality_delta":b["quality"]-a["quality"],
                            "incremental_dc_quality_delta":candidate["quality"]-b["quality"],
                            "proposed_transition_quality_delta":candidate["quality"]-a["quality"]})
    review_paths = [case_root(c) / "review" / c["id"] / "review.json" for c in m["visual_cases"]]
    gallery_rows = []
    for p in review_paths:
        if not p.exists():
            continue
        record = q.read(p); c=record["case"]
        gallery_rows.append({"id":c["id"],"path":os.path.relpath(p.parent,destination),"modes":record["modes"]})
        for name,r in record["modes"].items():
            diagnostics.append({"case":c["id"],"kind":c["image"]["kind"],"effort":c["effort"],"distance":c["distance"],"mode":name,
                                **{k:r[k] for k in ("srgb_rmse","block_error_roughness","red_green_error_roughness")}})
    for name,rows in [("timing",times),("controls",controls),("visual",visual),("matched-size",matches),("diagnostics",diagnostics),("transition",transitions)]:
        csv_file(destination / (name+".csv"),rows)
    expected_timing = len(m["timing_cases"])
    expected_visual = 4 * len(m["visual_cases"])
    expected_matched = sum(c["effort"] != 3 and c["distance"] != 1.2 for c in m["visual_cases"])
    summary={"revision":m["revision"],"timing_completed":len(times),"timing_expected":expected_timing,"timing_missing":missing,
             "visual_outputs":len(visual),"visual_outputs_expected":expected_visual,"matched_size_completed":sum(r["complete"] for r in matches),
             "matched_size_expected":expected_matched,"matched_size_unresolved":unresolved,"controls":controls,"timing":times,
             "synthetic_supplement":str(supplement) if supplement is not None else None,
             "review_cases":len(gallery_rows)}
    q.save(destination / "summary.json",summary)
    lines=["# e4 DC default qualification: measured results", "", f"Revision `{m['revision']['revision']}`. Defaults unchanged.","",
           f"Completion: {len(times)}/{expected_timing} timing cases, {len(visual)}/{expected_visual} visual outputs, {sum(r['complete'] for r in matches)}/{expected_matched} matched-size targets.","",
           "## Complete-encode timing at unchanged distance 1.2", "", "Three warmups; 20 balanced paired rounds; eight CPU participants; fully resident Metal. Input/output file I/O, validation, and final diagnostic scoring excluded. These are same-setting costs, not matched-quality timing.", "",
           "| Photo / size | Effort | Default ms | Both ms | Paired time change | Size change | SSIMULACRA2 change |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(times,key=lambda r:(r["effort"],int(r["size_class"][:-2]),r["photo"])):
        lines.append(f"| {r['photo']} / {r['size_class']} | {r['effort']} | {r['default_ms']:.2f} | {r['both_ms']:.2f} | {r['paired_change_pct']:+.2f}% | {r['size_change_pct']:+.2f}% | {r['quality_delta']:+.4f} |")
    lines += ["", "Milliseconds are separate variant medians. Paired change is the median of the 20 within-round time ratios, so it need not equal the ratio of those two medians."]
    lines += ["", "## Baseline versus baseline controls", "", "| Invocation | Paired change | Minimum round | Maximum round |", "|---|---:|---:|---:|"]
    for r in controls: lines.append(f"| {r['case']} | {r['paired_change_pct']:+.2f}% | {r['paired_min_pct']:+.2f}% | {r['paired_max_pct']:+.2f}% |")
    lines += ["", "## Four-mode visual measurements", "", "Each cell is actual bytes / SSIMULACRA2 at the same requested distance. Photographic derivatives are 2048 pixels wide; corrected synthetic fixtures are 2048×1024.", "",
              "| Case | Default | Quantize | Smooth | Both |", "|---|---:|---:|---:|---:|"]
    grouped = {}
    for row in visual:
        grouped.setdefault(row["case"], {})[row["mode"]] = row
    for case, rows in grouped.items():
        cells = [f"{rows[n]['bytes']:,} / {rows[n]['quality']:.4f}" if n in rows else "missing" for n in ("default", "quantize", "smooth", "both")]
        lines.append("| " + case + " | " + " | ".join(cells) + " |")
    lines += ["", "## Matched-size visual comparisons", "", "All attempts, including unresolved targets, appear below. Passing means within 1% of default bytes; it does not mean visual acceptance.","",
              "| Case | Size error | SSIMULACRA2 change | Probes | Status |", "|---|---:|---:|---:|---|"]
    for r in matches: lines.append(f"| {r['case']} | {r['size_error_pct']:.3f}% | {r['quality_delta']:+.4f} | {r['probes']} | {r['reason']} |")
    lines += ["", "## Proposed e3/e4 transition on photographs", "", "Same requested distance; bytes can differ. The current e4 baseline separates existing effort behavior from the incremental DC change.", "",
              "| Image | Distance | e3 default score | e4 default score | e4 both score | DC increment at e4 |", "|---|---:|---:|---:|---:|---:|"]
    for r in transitions:
        if r["kind"] == "photographic":
            lines.append(f"| {r['image']} | {r['distance']:g} | {r['e3_default_quality']:.4f} | {r['e4_default_quality']:.4f} | {r['e4_both_quality']:.4f} | {r['incremental_dc_quality_delta']:+.4f} |")
    lines += ["", "## Boundaries", "", "Three natural scenes are reused across sizes. High-resolution inputs retain the earlier Lanczos overshoot. Visual photographic derivatives are separately resized and bounded to [0,1]. Four analytic gradient diagnostics are synthetic and must not be aggregated as photographic compression results.","",
              "One process supplies each timing case; rounds are paired repetitions, not independent sessions. Background CPU snapshots and control spread remain part of interpretation. Diagnostic block-error roughness is not a validated perceptual metric. Agent visual inspection and browser previews do not establish blinded human preference on a calibrated display.","",
              f"Raw artifacts and the 16-bit visual gallery: `{out}`. The explicit `review` command decodes retained streams; `report` reads saved measurements only."]
    if supplement is not None:
        lines += ["", f"All original synthetic measurements are excluded. Corrected synthetic data come from `{supplement}`; see its independent fixture audit. The combined gallery is [review.html](review.html)."]
    (destination/"MEASUREMENTS.md").write_text("\n".join(lines)+"\n")
    figures(out, destination, supplement)
    if (out / "review/index.html").exists():
        document = (out / "review/index.html").read_text()
        document = re.sub(r"const data=.*?;const cases=", lambda _: "const data="+json.dumps(gallery_rows)+";const cases=", document, flags=re.S)
        document = document.replace(".src=r.id+'/'", ".src=r.path+'/'")
        (destination / "review.html").write_text(document)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('timing','controls','matched_size_unresolved')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("review","report"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--destination",type=Path,default=Path("docs/dc-e4-qualification"))
    p.add_argument("--synthetic-supplement",type=Path)
    a=p.parse_args(); out=a.output.resolve(); m=q.read(out/"manifest.json")
    if a.command=="review":
        with (out / "run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            review(out,m)
    else:
        supplement = a.synthetic_supplement
        pointer = out / "synthetic-supplement.json"
        if supplement is None and pointer.exists():
            supplement = Path(q.read(pointer)["path"])
        report(out,m,a.destination,supplement.resolve() if supplement else None)


if __name__=="__main__": main()
