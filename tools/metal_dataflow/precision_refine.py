#!/usr/bin/env python3
"""Sample nearby targets where the nominal approximate encode has worse quality."""
import argparse
import json
from pathlib import Path
from precision import ROOT, OUT, run, save, sha
from precision_quality import CORPUS, DECODER, last_json
from precision_matched import assess

def refine(variant):
    # Always reconstruct the nominal points. Otherwise a resumed run would see
    # its own successful refinements and replace them with an empty selection.
    assess(variant, include_refinements=False)
    path=OUT/"matched-quality"/(variant+".json")
    nominal=json.loads(path.read_text())
    probe=OUT/"baseline/gjxl_metal_precision_probe"
    library=OUT/"variants"/variant/"metal/gjxl.metallib"
    folder=OUT/"matched-samples"/variant
    folder.mkdir(parents=True,exist_ok=True)
    sources={row['sha256']:Path(row['path']) for row in
             json.loads((OUT/'quality-inputs.json').read_text())}
    for row in json.loads((ROOT/'tools/resident_qualification/corpus.json').read_text())['inputs']:
        sources[row['canonical_sha256']]=CORPUS/row['canonical_path']
    selected=[]
    samples=[]
    for point in nominal["rows"]:
        if point["matched_or_better"]:continue
        source=sources[point['source_sha256']]
        if sha(source)!=point["source_sha256"]:raise RuntimeError("Input identity mismatch")
        eligible=[]
        for relative in [.0001,.0005,.001,.002,.005,.01,.02]:
            target=point["target"]*(1-relative)
            name=point["name"]+"-minus"+str(relative)
            record=folder/(name+".json")
            jxl,decoded=folder/(name+".jxl"),folder/(name+".pfm")
            if record.exists():
                row=json.loads(record.read_text())
                if (row["library_sha256"]!=sha(library) or row["jxl_sha256"]!=sha(jxl)
                        or row["decoded_sha256"]!=sha(decoded) or row["probe_sha256"]!=sha(probe)
                        or row["decoder_sha256"]!=sha(DECODER) or row["target"]!=target):
                    raise RuntimeError("Refinement artifact changed")
            else:
                encoded=last_json(run("matched-"+variant+"-"+name+"-encode",[probe,"encode",library,source,jxl,target,7]))
                run("matched-"+variant+"-"+name+"-decode",[DECODER,jxl,decoded,"--color_space=RGB_D65_SRG_Rel_Lin"])
                metric=last_json(run("matched-"+variant+"-"+name+"-metric",[probe,"metric",source,decoded]))
                row=dict(name=name,target=target,relative_target_change=-relative,
                    encoded=encoded,metric=metric,library_sha256=sha(library),probe_sha256=sha(probe),
                    decoder_sha256=sha(DECODER),jxl_sha256=sha(jxl),decoded_sha256=sha(decoded))
                save(record,row)
            samples.append(row)
            if row["metric"]["butteraugli"]<=point["baseline_score"]+point["matching_tolerance"]:
                eligible.append(row)
        if eligible:
            best=min(eligible,key=lambda r:r["encoded"]["bytes"])
            selected.append(dict(name=point["name"],source_sha256=point["source_sha256"],target=point["target"],
                candidate_target=best["target"],relative_target_change=best["relative_target_change"],
                candidate_score=best["metric"]["butteraugli"],score_delta=best["metric"]["butteraugli"]-point["baseline_score"],
                candidate_bytes=best["encoded"]["bytes"],size_ratio=best["encoded"]["bytes"]/point["baseline_bytes"],
                matched_or_better=True,refinement_record=best["name"]))
    save(OUT/"matched-quality"/(variant+"-refinement.json"),dict(variant=variant,selected=selected,samples=samples,
        protocol="Seven fixed lower target offsets; choose the smallest observed eligible codestream; no monotonicity assumption or interpolation."))
    assess(variant)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    refine(parser.parse_args().variant)
