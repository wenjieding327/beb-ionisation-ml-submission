"""Synchronize current evidence counts and late source notes after replayed fixes."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames), list(reader)


def write(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def family(label):
    if "peak" in label:
        return "P"
    if "fixed" in label or "sparse" in label:
        return "F"
    return "C"


def main():
    _, points = read(ROOT/"outputs/expanded_194_training_2026_08_02/expanded_194_training_points.csv")
    groups = defaultdict(list)
    for point in points:
        groups[point["identity_key"]].append(point)
    late = json.loads((ROOT/"audit/late_source_corrections.json").read_text(encoding="utf-8-sig"))
    patches = late["patches"] if isinstance(late, dict) else late
    notes = defaultdict(list)
    for patch in patches:
        notes[patch["identity_key"]].append(patch["reason"])
    ledger_path = ROOT/"data/source_evidence_ledger_194.csv"
    fields, ledger = read(ledger_path)
    if "late_source_metadata_note" not in fields:
        fields.append("late_source_metadata_note")
    for row in ledger:
        group = groups[row["identity_key"]]
        labels = sorted({point["label_type"] for point in group})
        codes = [code for code in "CFP" if any(family(label) == code for label in labels)]
        row["training_observations"] = str(len(group))
        row["label_types_used"] = " | ".join(labels)
        row["experimental_label_type"] = ";".join({"C": "experimental_curve_points", "F": "experimental_fixed_or_sparse_points", "P": "experimental_peak_height"}[code] for code in codes)
        row["evidence_code"] = "/".join(codes)
        row["curve_observations_retained"] = str(sum(family(point["label_type"]) == "C" for point in group))
        row["late_source_metadata_note"] = " | ".join(notes[row["identity_key"]])
        if row["identity_key"] == "nitromethane":
            row["source_completeness_status"] = "The retained numerical constraint is a matched 60 eV experimental/BEB pair, not a peak-height comparison. Experimental 6.8 A^2 and BEB 5.95 A^2 are unchanged; the observation is routed as fixed energy."
    source_changes = ROOT/"audit/late_source_ledger_overrides.json"
    if source_changes.exists():
        lookup = {row["identity_key"]: row for row in ledger}
        for change in json.loads(source_changes.read_text(encoding="utf-8-sig")):
            for field, value in change["fields"].items():
                assert field not in {"training_observations", "cv_fold", "evidence_code"}
                if field not in fields:
                    fields.append(field)
                lookup[change["identity_key"]][field] = value
    write(ledger_path, fields, ledger)
    lookup = {row["identity_key"]: row for row in ledger}
    path = ROOT/"data/final_194_molecule_provenance.csv"
    fields, provenance = read(path)
    if "late_source_metadata_note" not in fields:
        fields.append("late_source_metadata_note")
    for row in provenance:
        evidence = lookup[row["identity_key"]]
        for field in ("experimental_label_type", "training_observations", "label_types_used", "training_source_files", "audit_note", "late_source_metadata_note"):
            row[field] = evidence[field]
        row["evidence_labels_used"] = " | ".join(sorted({point["evidence"] for point in groups[row["identity_key"]]}))
    write(path, fields, provenance)
    path = ROOT/"data/cleaning_flow_audit.csv"
    fields, flow = read(path)
    flow = [row for row in flow if row["stage"] != "2026-09-06 late source metadata corrections"]
    new = {field: "n/a" for field in fields}
    new.update(stage="2026-09-06 late source metadata corrections", attempted_rows="3533", retained_rows="3533", removed_by_source_audit="0", audit_scope=f"{len(patches)} explicit source patches: 51 common-energy fields cleared, one same-energy pair rerouted, six experimental peak amplitudes replaced by directly verified Bart table values with 3.9% instrumental bounds. All BEB amplitudes unchanged. See audit/late_source_corrections.json.")
    flow.append(new)
    write(path, fields, flow)
    print("Updated v3 source metadata and evidence-class counts without altering numerical input.")


if __name__ == "__main__":
    main()
