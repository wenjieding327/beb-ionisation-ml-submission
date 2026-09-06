"""Apply field-scoped source-audit corrections without changing numerical inputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ledger_path = ROOT / "data/source_evidence_ledger_194.csv"
    with ledger_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames)
        ledger = list(reader)
    by_identity = {row["identity_key"]: row for row in ledger}
    assert len(by_identity) == 194
    before_counts = {key: row["training_observations"] for key, row in by_identity.items()}
    for name in ("retained_source_ledger_overrides.json", "audit_note_overrides_all194.json"):
        changes = json.loads((ROOT / "audit" / name).read_text(encoding="utf-8-sig"))
        for change in changes:
            row = by_identity[change["identity_key"]]
            for key, value in change["fields"].items():
                assert key not in {"training_observations", "cv_fold", "evidence_code"}
                if key not in columns:
                    columns.append(key)
                row[key] = value
    assert before_counts == {key: row["training_observations"] for key, row in by_identity.items()}
    with ledger_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ledger)
    provenance_path = ROOT / "data/final_194_molecule_provenance.csv"
    with provenance_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        provenance_columns = list(reader.fieldnames)
        provenance = list(reader)
    for row in provenance:
        audited = by_identity[row["identity_key"]]
        for key in ("audit_note", "training_source_files"):
            if key in row and key in audited:
                row[key] = audited[key]
    with provenance_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=provenance_columns)
        writer.writeheader()
        writer.writerows(provenance)
    print("Updated source-audit notes for 194 identities; numerical inputs and counts unchanged.")


if __name__ == "__main__":
    main()
