#!/usr/bin/env python3
"""Paired metric-bias and guarded strided-memory qualification."""
import argparse
import json
from pathlib import Path
from precision import OUT, run, save, sha
from precision_quality import last_json

VALIDATION = {"MTL_DEBUG_LAYER":"1", "MTL_DEBUG_LAYER_ERROR_MODE":"assert",
              "MTL_SHADER_VALIDATION":"1", "MTL_SHADER_VALIDATION_ABORT_ON_FAULT":"1",
              "MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING":"1",
              "MTL_SHADER_VALIDATION_REPORT_TO_STDERR":"1"}

def evaluate(name, validation=False, full=False):
    mode = ("full-" if full else "screen-") + ("validation" if validation else "normal")
    folder = OUT / "metric-bias" / name / mode
    probe = json.loads((OUT / "probes/metric.json").read_text())
    baseline = OUT / "baseline/metal/gjxl.metallib"
    candidate = OUT / "variants" / name / "metal/gjxl.metallib"
    if sha(probe["binary"]) != probe["sha256"]: raise RuntimeError("Probe changed")
    cases=[]
    for source in json.loads((OUT / "quality-inputs.json").read_text()):
        if not full and source["name"] not in ("dark","neutral","texture","flat","tiny_dark","kodak-kodim17"):
            continue
        for distance in source["targets"]:
            if not full and distance not in (.1,1.2):continue
            case=source["name"]+"-d"+str(distance)
            decoded=OUT/"quality/baseline"/(case+".pfm")
            cases.append(dict(name=case,reference=source["path"],reference_sha256=source["sha256"],
                              decoded=str(decoded),decoded_sha256=sha(decoded)))
        # Identity comparisons detect approximation-induced self distortion.
        cases.append(dict(name=source["name"]+"-identity",reference=source["path"],
                          reference_sha256=source["sha256"],decoded=source["path"],decoded_sha256=source["sha256"]))
    identity=dict(probe=probe, baseline_sha256=sha(baseline), candidate_sha256=sha(candidate),
                  cases=cases, environment=VALIDATION if validation else {})
    record=folder/"identity.json"
    if record.exists() and json.loads(record.read_text())!=identity:
        raise RuntimeError("Metric identity differs; use a separate full-screen directory")
    save(record,identity)
    rows=[]
    for case in cases:
        if sha(case["reference"])!=case["reference_sha256"] or sha(case["decoded"])!=case["decoded_sha256"]:
            raise RuntimeError("Image pair changed")
        result=last_json(run("metric-"+name+"-"+mode+"-"+case["name"],
            [probe["binary"],baseline,candidate,case["reference"],case["decoded"]],
            VALIDATION if validation else None))
        rows.append(dict(name=case["name"],**result))
        save(folder/(case["name"]+".json"),rows[-1])
    summary=dict(variant=name,mode=mode,cases=len(rows),failures=[r["name"] for r in rows if not r["within_budget"]],
                 max_score_error=max(abs(r["score_delta"]) for r in rows),
                 max_underestimation=max(-r["score_delta"] for r in rows),rows=rows)
    save(folder/"summary.json",summary)
    print(name,mode,"metric failures",summary["failures"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants",nargs="+")
    parser.add_argument("--validation",action="store_true")
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    for name in args.variants:evaluate(name,args.validation,args.full)
