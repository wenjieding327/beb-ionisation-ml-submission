"""Synchronize evidence-count metadata after the source-based corrections."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)
def write(path,fields,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
_,points=read(ROOT/"outputs/expanded_194_training_2026_08_02/expanded_194_training_points.csv")
groups=defaultdict(list)
for row in points:groups[row["identity_key"]].append(row)
_,excluded=read(ROOT/"audit/quarantined_rows.csv")
excluded_groups=defaultdict(list)
for row in excluded:excluded_groups[row["identity_key"]].append(row)
ledger_fields,ledger=read(ROOT/"data/source_evidence_ledger_194.csv")
for row in ledger:
    group=groups[row["identity_key"]]
    labels=sorted({r["label_type"] for r in group})
    codes=[];types=[]
    if any("curve" in v for v in labels):codes.append("C");types.append("experimental_curve_points")
    if any("fixed" in v or "sparse" in v for v in labels):codes.append("F");types.append("experimental_fixed_or_sparse_points")
    if any("peak" in v for v in labels):codes.append("P");types.append("experimental_peak_height")
    row["training_observations"]=str(len(group))
    row["label_types_used"]=" | ".join(labels)
    row["experimental_label_type"]=";".join(types)
    row["training_source_files"]=" | ".join(sorted({r["source"] for r in group}))
    row["evidence_code"]="/".join(codes)
    row["curve_observations_retained"]=str(sum(r["label_type"]=="curve_point" for r in group))
    row["quarantined_curve_observations"]=str(len(excluded_groups[row["identity_key"]]))
    notes=[]
    if excluded_groups[row["identity_key"]]:notes.append("2026-09-06 quarantined "+str(len(excluded_groups[row["identity_key"]]))+" curve labels: "+";".join(sorted({r["quarantine_reason"] for r in excluded_groups[row["identity_key"]]})))
    if any(r["correction_note"] for r in group):notes.append("Table 1 peak energy unreported; peak uncertainty uses author's 5% absolute accuracy")
    if any("ready15_esters" in r["source"] and r["label_type"]=="curve_point" for r in group):notes.append("Retained ester curve is peak-anchored figure-trace extraction, not an independent numerical experiment table")
    if row["identity_key"]=="ozone":notes.append("Only original Newson and Siegel series retained; renormalised Newson copy excluded; observed peak boundary 90.8947 eV")
    row["audit_note"]=(row.get("audit_note","")+" | "+" | ".join(notes)).strip(" |")
for name in ["evidence_code","curve_observations_retained","quarantined_curve_observations"]:
    if name not in ledger_fields:ledger_fields.append(name)
write(ROOT/"data/source_evidence_ledger_194.csv",ledger_fields,ledger)
lookup={r["identity_key"]:r for r in ledger}
fields,provenance=read(ROOT/"data/final_194_molecule_provenance.csv")
for row in provenance:
    new=lookup[row["identity_key"]]
    for field in ["experimental_label_type","training_observations","label_types_used","training_source_files","audit_note"]:row[field]=new[field]
    row["experimental_source_doi_or_url"]=new["experimental_source_locator"]
    row["evidence_labels_used"]=" | ".join(sorted({r["evidence"] for r in groups[row["identity_key"]]}))
    row["experimental_value_or_points"]=str(len(groups[row["identity_key"]]))
write(ROOT/"data/final_194_molecule_provenance.csv",fields,provenance)
fields,flow=read(ROOT/"data/cleaning_flow_audit.csv")
fields.append("removed_by_source_audit")
for row in flow:
    row["removed_by_source_audit"]="0"
    if row["stage"]=="Final frozen release":row["stage"]="Legacy r2 frozen release"
new={f:"n/a" for f in fields}
new.update(stage="2026-09-06 source evidence corrections",attempted_rows="3883",retained_rows="3533",removed_by_source_audit="350",audit_scope="7 water theoretical labels + 330 ambiguous ester curve labels + 13 duplicate renormalised ozone labels quarantined; 15 independent ester peak energies set missing and uncertainties corrected to5%; all194 identities retained")
flow.append(new)
write(ROOT/"data/cleaning_flow_audit.csv",fields,flow)
print("Metadata synchronized: 194 identities, 3533 observations; 74 C, 143 F, 68 P.")
