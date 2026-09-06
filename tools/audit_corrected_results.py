"""Independently recompute reported errors from frozen CSV predictions."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"outputs/final_expanded_194_ml_2026_08_02"

def load(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))

def family(label):
    if "peak" in label:return "peak"
    if "fixed" in label or "sparse" in label:return "fixed_sparse"
    return "curve"

def nmae(rows,prediction,primary=True):
    selected=[r for r in rows if not primary or r["primary_low_energy"]=="True"]
    return sum(abs(float(r[prediction])-float(r["experimental_sigma_A2"])) for r in selected)/sum(float(r["experimental_sigma_A2"]) for r in selected)

def macro(rows,prediction,primary=True):
    groups=defaultdict(list)
    for r in rows:groups[r["identity_key"]].append(r)
    return sum(nmae(v,prediction,primary) for v in groups.values())/len(groups)

def main():
    rows=load(MODEL/"expanded_194_oof_predictions.csv")
    inp=load(ROOT/"outputs/expanded_194_training_2026_08_02/expanded_194_training_points.csv")
    assert [r["original_csv_line"] for r in rows]==[r["original_csv_line"] for r in inp]
    summary=json.loads((MODEL/"metrics.json").read_text())
    values={}
    for name in ["all","curve","fixed_sparse","peak"]:
        subset=rows if name=="all" else [r for r in rows if family(r["label_type"])==name]
        expected=summary if name=="all" else summary["per_task"][name]
        item={}
        for short,column,metric in [("raw","prediction_raw_beb_A2","raw_metrics"),("constant","prediction_constant_scale_A2","constant_scale_metrics"),("ml","prediction_expanded_ml_A2","expanded_ml_metrics")]:
            value=macro(subset,column)
            recorded=expected[metric if name=="all" else short]["primary_molecule_macro_NMAE"]
            assert math.isclose(value,recorded,abs_tol=1e-12)
            item[short]=value
        item["absolute_improvement_pp"]=100*(item["raw"]-item["ml"])
        item["relative_improvement_percent"]=100*(1-item["ml"]/item["raw"])
        values[name]=item
    groups=defaultdict(list)
    for row in rows:
        if family(row["label_type"])=="curve":groups[row["identity_key"]].append(row)
    cases=[]
    for identity,group in groups.items():
        raw=nmae(group,"prediction_raw_beb_A2")
        ml=nmae(group,"prediction_expanded_ml_A2")
        cases.append({"identity_key":identity,"molecule_name":group[0]["molecule_name"],"formula":group[0]["formula"],"rows":len(group),"primary_rows":sum(r["primary_low_energy"]=="True" for r in group),"cv_fold":group[0]["cv_fold"],"primary_raw_NMAE":raw,"primary_ml_NMAE":ml,"primary_improvement_pp":100*(raw-ml),"whole_raw_NMAE":nmae(group,"prediction_raw_beb_A2",False),"whole_ml_NMAE":nmae(group,"prediction_expanded_ml_A2",False)})
    cases.sort(key=lambda r:-r["primary_improvement_pp"])
    with (ROOT/"audit/per_molecule_corrected_curve_metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    legacy=load(ROOT/"audit/legacy_r2/expanded_194_oof_predictions.csv")
    legacy_by_line={str(n):r for n,r in enumerate(legacy,2)}
    old_survivors=[dict(legacy_by_line[r["original_csv_line"]],primary_low_energy=r["primary_low_energy"]) for r in rows]
    old_curve=[r for r in old_survivors if family(r["label_type"])=="curve"]
    comparison={"old_model_on_same_surviving_all_rows":macro(old_survivors,"prediction_expanded_ml_A2"),"old_model_on_same_surviving_curve_rows":macro(old_curve,"prediction_expanded_ml_A2"),"new_model_on_same_surviving_all_rows":values["all"]["ml"],"new_model_on_same_surviving_curve_rows":values["curve"]["ml"],"note":"Old models were trained on legacy labels and are shown for audit only. These values are not a corrected clean benchmark."}
    result={"verified_from_csv":values,"comparison_with_legacy_predictions":comparison,"observations":len(rows),"molecules":len({r["identity_key"] for r in rows}),"curve_molecules":len(groups),"features":49,"runtime":summary["runtime"],"input_sha256":hashlib.sha256((ROOT/"outputs/expanded_194_training_2026_08_02/expanded_194_training_points.csv").read_bytes()).hexdigest()}
    (ROOT/"audit/independent_metric_recalculation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
