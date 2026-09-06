"""Replay all evidence corrections from the packaged 3,883-row legacy input.

The script uses only the standard library and audit manifests in this release.
It never requires the original research workstation or writes to legacy inputs.
Use --check-only to verify the current corrected CSV without overwriting it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_REL = Path("outputs/expanded_194_training_2026_08_02/expanded_194_training_points.csv")
ORIGINAL_SHA256 = "50f4cf157348bc23bc092533352645b55fec53fa2fd25b73bfa090ba21058e85"
PARENT_V2_COMMIT = "eb80b8107da278a8fb7dfc067613e1ee8f5218c4"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames), list(reader)


def csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def task(label):
    if "peak" in label:
        return "peak"
    if "fixed" in label or "sparse" in label:
        return "fixed_sparse"
    return "curve"


def row_hash(row, fields):
    packed = json.dumps([row.get(key, "") for key in fields], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def serialized(value):
    return "" if value is None else str(value)


def matches(actual, expected):
    expected = serialized(expected)
    if actual == expected:
        return True
    try:
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    except (ValueError, TypeError):
        return False


def build():
    original_path = ROOT / "audit/legacy_r2/expanded_194_training_points.csv"
    original_bytes = original_path.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == ORIGINAL_SHA256
    fields, original = read_csv(original_path)
    assert len(original) == 3883
    _, quarantine = read_csv(ROOT / "audit/quarantined_rows.csv")
    withdrawn = {int(row["original_csv_line"]): row for row in quarantine}
    assert len(withdrawn) == 350
    assert Counter(row["quarantine_reason"] for row in quarantine) == {
        "NIST_exp_5_is_theory_not_experiment": 7,
        "ambiguous_isomer_trace_quarantined": 330,
        "O3_exp_2_renormalized_duplicate_of_Newson": 13,
    }
    kept = []
    for line, source in enumerate(original, 2):
        digest = row_hash(source, fields)
        if line in withdrawn:
            assert withdrawn[line]["original_row_sha256"] == digest
            assert all(source[key] == withdrawn[line][key] for key in fields)
            continue
        kept.append(dict(source, original_csv_line=str(line), original_row_sha256=digest, correction_note=""))
    assert len(kept) == 3533
    by_line = {int(row["original_csv_line"]): row for row in kept}
    changes = []

    # Source-verified Hudson Table 1 heights do not report a peak energy.
    _, peaks = read_csv(ROOT / "audit/peak_metadata_corrections.csv")
    assert len(peaks) == 15
    for correction in peaks:
        row = by_line[int(correction["original_csv_line"])]
        assert row["identity_key"] == correction["identity_key"]
        assert row["energy_eV"] == correction["original_energy_eV"]
        assert matches(row["experimental_sigma_A2"], correction["peak_height_A2"])
        before = {key: row[key] for key in ("energy_eV", "energy_is_reported", "experimental_uncertainty_A2")}
        row["energy_eV"] = ""
        row["energy_is_reported"] = "False"
        row["experimental_uncertainty_A2"] = repr(0.05 * float(row["experimental_sigma_A2"]))
        row["correction_note"] = "Table1_peak_energy_unreported;author_5_percent_absolute_accuracy"
        changes.append({"stage": "Hudson_Table_1_metadata", "original_csv_line": int(row["original_csv_line"]), "identity_key": row["identity_key"], "before": before, "after": {key: row[key] for key in before}, "reason": correction["reason"], "source": correction["source"]})

    late = json.loads((ROOT / "audit/late_source_corrections.json").read_text(encoding="utf-8-sig"))
    patches = late["patches"] if isinstance(late, dict) else late
    assert len(patches) == 52
    patched_lines = set()
    reported_peak_metadata = []
    replacements = []
    for correction in patches:
        line = int(correction["original_csv_line"])
        assert line not in patched_lines, "Combine changes to one original row in one explicit patch"
        patched_lines.add(line)
        row = by_line[line]
        assert row["identity_key"] == correction["identity_key"]
        assert row["original_row_sha256"] == correction["original_row_sha256"]
        for key, expected in correction["before"].items():
            assert matches(row[key], expected), (line, key, row[key], expected)
        all_after = correction["after"]
        assert not set(all_after).intersection({"identity_key", "cv_fold", "original_csv_line", "original_row_sha256", "beb_sigma_A2"}), "Identity, fold or BEB changes are not authorised"
        amplitude_keys = {"experimental_sigma_A2", "experimental_uncertainty_A2"}.intersection(all_after)
        if amplitude_keys:
            assert amplitude_keys == {"experimental_sigma_A2", "experimental_uncertainty_A2"}
            replacements.append({"original_csv_line": line, "original_row_sha256": row["original_row_sha256"], "identity_key": row["identity_key"], "before": {key: row[key] for key in sorted(amplitude_keys)}, "after": {key: all_after[key] for key in sorted(amplitude_keys)}, "reason": correction["reason"], "source": correction["source"]})
        after = {key: value for key, value in all_after.items() if key not in amplitude_keys}
        before = {key: row.get(key, "") for key in after}
        for key, value in after.items():
            if key == "experimental_peak_energy_eV":
                reported_peak_metadata.append({"original_csv_line": line, "identity_key": row["identity_key"], "experimental_peak_energy_eV": value, "model_energy_eV": "", "model_feature_policy": "provenance only; not read by the feature matrix", "source": correction.get("source", ""), "verification_level": correction.get("verification_level", "")})
                continue
            assert key in fields, key
            row[key] = serialized(value)
        reason = correction["reason"]
        row["correction_note"] = ";".join(filter(None, (row["correction_note"], reason)))
        changes.append({"stage": "late_source_metadata", "original_csv_line": line, "identity_key": row["identity_key"], "before": before, "after": {key: serialized(after[key]) if key == "experimental_peak_energy_eV" else row[key] for key in after}, "reason": reason, "source": correction.get("source", "")})

    assert len(replacements) == 6
    replaced_lines = set()
    for replacement in replacements:
        line = int(replacement["original_csv_line"])
        assert line not in replaced_lines
        replaced_lines.add(line)
        row = by_line[line]
        assert row["identity_key"] == replacement["identity_key"]
        assert row["original_row_sha256"] == replacement["original_row_sha256"]
        assert task(row["label_type"]) == "peak", "Only the source-verified peak-amplitude replacements are authorised"
        for key, expected in replacement["before"].items():
            assert matches(row[key], expected), (line, key, row[key], expected)
        after = replacement["after"]
        assert set(after).issubset({"experimental_sigma_A2", "experimental_uncertainty_A2", "source", "evidence"})
        assert "experimental_sigma_A2" in after
        before = {key: row[key] for key in after}
        for key, value in after.items():
            row[key] = serialized(value)
        assert float(row["experimental_sigma_A2"]) > 0
        row["correction_note"] = ";".join(filter(None, (row["correction_note"], replacement["reason"])))
        changes.append({"stage": "verified_experimental_source_replacement", "original_csv_line": line, "identity_key": row["identity_key"], "before": before, "after": {key: row[key] for key in after}, "reason": replacement["reason"], "source": replacement["source"]})

    boundaries = {}
    for identity in ("water", "ozone"):
        subset = [row for row in kept if row["identity_key"] == identity and task(row["label_type"]) == "curve"]
        maximum = max(float(row["experimental_sigma_A2"]) for row in subset)
        peak = min(float(row["energy_eV"]) for row in subset if float(row["experimental_sigma_A2"]) == maximum)
        onset = min(float(row["energy_eV"]) for row in subset)
        changed_flags = []
        for row in subset:
            value = str(onset <= float(row["energy_eV"]) <= peak)
            if value != row["primary_low_energy"]:
                changed_flags.append({"original_csv_line": int(row["original_csv_line"]), "energy_eV": row["energy_eV"], "old_primary": row["primary_low_energy"], "new_primary": value})
            row["primary_low_energy"] = value
        boundaries[identity] = {"onset_eV": onset, "peak_eV": peak, "peak_A2": maximum, "changed_flags": changed_flags}
    assert boundaries["water"]["peak_eV"] == 90.5395
    assert boundaries["ozone"]["peak_eV"] == 90.8947
    counts = Counter(row["identity_key"] for row in kept)
    for row in kept:
        exp = float(row["experimental_sigma_A2"])
        beb = float(row["beb_sigma_A2"])
        relative_uncertainty = float(row["experimental_uncertainty_A2"])/exp if row["experimental_uncertainty_A2"] else 0.12
        molecule_weight = 1.0/counts[row["identity_key"]]
        quality_weight = 1/(1+(relative_uncertainty/0.12)**2)
        row["molecule_weight"] = repr(molecule_weight)
        row["sample_weight"] = repr(molecule_weight*quality_weight*(2.0 if row["primary_low_energy"] == "True" else 0.75))
        row["target_log_ratio"] = repr(math.log(exp/beb))
        assert row["cv_fold"] == original[int(row["original_csv_line"])-2]["cv_fold"]
    false_finite = [row for row in kept if row["energy_is_reported"] == "False" and row["energy_eV"] and math.isfinite(float(row["energy_eV"]))]
    assert not false_finite, "Unreported energies must not enter as measured feature values"
    assert not [row for row in kept if row["energy_is_reported"] == "True" and (not row["energy_eV"] or not math.isfinite(float(row["energy_eV"])))], "Reported-energy flags require an actual finite energy"
    assert all(not row["energy_eV"] and row["energy_is_reported"] == "False" for row in kept if task(row["label_type"]) == "peak"), "Peak-to-peak amplitudes do not share a collision-energy coordinate"
    assert len(reported_peak_metadata) == 27
    family_rows = Counter(task(row["label_type"]) for row in kept)
    family_ids = {family: len({row["identity_key"] for row in kept if task(row["label_type"]) == family}) for family in ("curve", "fixed_sparse", "peak")}
    assert len(counts) == 194 and family_rows["curve"] == 3316 and family_ids["curve"] == 74
    summary = {"release": "evidence_corrected_v3_2026_09_06", "parent_commit": PARENT_V2_COMMIT, "original_input_sha256": ORIGINAL_SHA256, "final_physical_identities": len(counts), "final_observations": len(kept), "primary_observations": sum(row["primary_low_energy"] == "True" for row in kept), "curve_identities": family_ids["curve"], "curve_observations": family_rows["curve"], "fixed_sparse_identities": family_ids["fixed_sparse"], "fixed_sparse_observations": family_rows["fixed_sparse"], "peak_identities": family_ids["peak"], "peak_observations": family_rows["peak"], "fold_row_loads": [sum(int(row["cv_fold"]) == fold for row in kept) for fold in range(5)], "quarantined_rows": len(withdrawn), "quarantine_reasons": dict(Counter(row["quarantine_reason"] for row in quarantine)), "peak_metadata_corrections": len(peaks), "late_source_metadata_corrections": len(patches), "water_primary_boundary": boundaries["water"], "ozone_primary_boundary": boundaries["ozone"], "fold_policy": "Original physical-identity outer cv_fold retained; unchanged deterministic three-fold inner grouping on current training rows.", "model_policy": "Original 49 features and candidate model/search protocol retained; all heads retrained; no selection based on favourable corrected results.", "construction": "Packaged original 3883 rows -> exact 350-row quarantine manifest -> 15 Hudson peak metadata corrections -> explicit late source patches -> retained-data primary boundaries -> recomputed weights and target log ratios."}
    output_fields = fields + ["original_csv_line", "original_row_sha256", "correction_note"]
    summary["energy_field_definition"] = "energy_eV and energy_is_reported specify a shared collision-energy coordinate for an experimental/BEB pair. All peak-to-peak amplitude observations have missing model energy, even when a source reports a separate experimental peak energy. Twenty-seven verified source peak energies are retained only in audit/reported_peak_energy_metadata.csv and are not input features."
    summary["experimental_source_replacements"] = len(replacements)
    summary["construction"] = "Packaged original 3883 rows -> exact 350-row quarantine manifest -> 15 Hudson peak metadata corrections -> 52 explicit late source patches -> six original-table experimental peak replacements -> retained-data primary boundaries -> recomputed weights and target log ratios."
    return output_fields, kept, summary, changes, reported_peak_metadata, replacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    fields, rows, summary, changes, reported_peak_metadata, replacements = build()
    content = csv_bytes(fields, rows)
    summary["corrected_input_sha256"] = hashlib.sha256(content).hexdigest()
    if args.check_only:
        assert (ROOT/INPUT_REL).read_bytes() == content, "Replayed input bytes differ"
        print("PASS: full evidence-correction pipeline reproduces the corrected input byte-for-byte.")
        return
    (ROOT/INPUT_REL).write_bytes(content)
    for target in (ROOT/"audit/correction_scope.json", ROOT/INPUT_REL.parent/"summary.json"):
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT/"audit/all_retained_metadata_corrections.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    (ROOT/"audit/reported_peak_energy_metadata.csv").write_bytes(csv_bytes(list(reported_peak_metadata[0]), reported_peak_metadata))
    (ROOT/"audit/experimental_source_replacements.json").write_text(json.dumps({"derived_from": "audit/late_source_corrections.json", "stage": "six original-table experimental peak replacements, applied after metadata corrections", "patches": replacements}, indent=2), encoding="utf-8")
    folds = []
    for fold in range(5):
        for family in ("curve", "fixed_sparse", "peak"):
            selected = [row for row in rows if int(row["cv_fold"]) == fold and task(row["label_type"]) == family]
            folds.append({"cv_fold": fold, "task_family": family, "rows": len(selected), "molecules": len({row["identity_key"] for row in selected})})
    (ROOT/"audit/corrected_fold_composition.csv").write_bytes(csv_bytes(list(folds[0]), folds))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
