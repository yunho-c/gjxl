#!/usr/bin/env python3
"""Conservative rate comparison at matched or better independently decoded quality."""
import argparse
import json
import math
from pathlib import Path
from precision import OUT, save

def assess(variant, include_refinements=True):
    rows = {}
    for group in ("quality", "quality-corpus"):
        folder = OUT/group/variant
        if not (folder/"summary.json").exists():continue
        for row in json.loads((folder/"summary.json").read_text())["rows"]:
            baseline = json.loads((OUT/group/"baseline"/(row["name"]+".json")).read_text())
            # A tenth of the relative quality budget, with a small absolute
            # allowance. Better-quality points are conservative rate evidence.
            tolerance = max(.0002,.001*baseline["metric"]["butteraugli"])
            delta = row["metric"]["butteraugli"]-baseline["metric"]["butteraugli"]
            key=(row["source_sha256"],row["distance"])
            result=dict(name=row["name"],source_sha256=row["source_sha256"],
                target=row["distance"],baseline_score=baseline["metric"]["butteraugli"],
                candidate_score=row["metric"]["butteraugli"],score_delta=delta,
                matching_tolerance=tolerance,matched_or_better=delta<=tolerance,
                baseline_bytes=baseline["encoded"]["bytes"],candidate_bytes=row["encoded"]["bytes"],
                size_ratio=row["encoded"]["bytes"]/baseline["encoded"]["bytes"])
            if key in rows and rows[key] != result:
                raise RuntimeError("Repeated corpus point differs")
            rows[key]=result
    if not rows:raise RuntimeError("No completed full quality results")
    refinements=OUT/"matched-quality"/(variant+"-refinement.json")
    if include_refinements and refinements.exists():
        for refined in json.loads(refinements.read_text())["selected"]:
            key=(refined["source_sha256"],refined["target"])
            if key not in rows:raise RuntimeError("Refined point has no baseline")
            rows[key].update(refined)
    values=list(rows.values())
    unmatched=[r["name"] for r in values if not r["matched_or_better"]]
    mean=math.exp(sum(math.log(r["size_ratio"]) for r in values)/len(values))
    worst=max(r["size_ratio"] for r in values)
    result=dict(variant=variant,cases=len(values),unmatched=unmatched,
        geometric_mean_size_change_percent=100*(mean-1),worst_size_change_percent=100*(worst-1),
        passes=not unmatched and mean<=1.005 and worst<=1.02, rows=values,
        interpretation="Observed candidate points at matched or better quality; no BD-rate interpolation. Unmatched points require further target sampling.")
    save(OUT/"matched-quality"/(variant+".json"),result)
    print(variant,"cases",len(values),"unmatched",unmatched,"mean size %",100*(mean-1),"worst %",100*(worst-1),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    assess(parser.parse_args().variant)
