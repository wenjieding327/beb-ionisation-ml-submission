"""Fast integrity and scientific-invariant checks for the frozen release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from audit_corrected_results import macro


ROOT = Path(__file__).resolve().parents[1]
TRAINING = (
    ROOT
    / "outputs"
    / "expanded_194_training_2026_08_02"
    / "expanded_194_training_points.csv"
)
SUMMARY = ROOT / "outputs" / "expanded_194_training_2026_08_02" / "summary.json"
METRICS = ROOT / "outputs" / "final_expanded_194_ml_2026_08_02" / "metrics.json"
VALIDITY = (
    ROOT
    / "outputs"
    / "final_validity_ablation_corrected_2026_08_05"
    / "validity_ablation_results.json"
)
MANIFEST = ROOT / "checksums_sha256.csv"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}

EXPECTED_COLUMNS = {
    "identity_key",
    "molecule_name",
    "formula",
    "energy_eV",
    "experimental_sigma_A2",
    "beb_sigma_A2",
    "label_type",
    "primary_low_energy",
    "cv_fold",
}


def require(condition: bool, message: str) -> None:
    """Raise a concise validation error when *condition* is false."""

    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums() -> int:
    """Verify every file listed in the release manifest."""

    require(MANIFEST.is_file(), "Missing checksums_sha256.csv")
    checked = 0
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["relative_path"]
            require(path.is_file(), f"Manifest file is missing: {row['relative_path']}")
            require(path.stat().st_size == int(row["bytes"]), f"Size mismatch: {path}")
            require(sha256(path) == row["sha256"], f"SHA-256 mismatch: {path}")
            checked += 1
    return checked


def task_family(label: str) -> str:
    """Map a recorded observation label to one of the three model tasks."""

    label = label.lower()
    if "peak" in label:
        return "peak"
    if "fixed" in label or "sparse" in label:
        return "fixed_sparse"
    return "curve"


def verify_training_table() -> dict[str, int]:
    """Check schema, counts, folds and basic physical validity."""

    require(TRAINING.is_file(), "Frozen training table is missing")
    identities: set[str] = set()
    curve_identities: set[str] = set()
    folds: set[int] = set()
    rows = 0
    identity_folds = defaultdict(set)
    task_counts = Counter()

    with TRAINING.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "Training table has no header")
        missing = EXPECTED_COLUMNS.difference(reader.fieldnames)
        require(not missing, f"Training table is missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            identity = row["identity_key"].strip()
            require(identity, f"Row {rows} has no identity_key")
            identities.add(identity)
            folds.add(int(row["cv_fold"]))
            identity_folds[identity].add(int(row["cv_fold"]))
            task_counts[task_family(row["label_type"])] += 1
            if task_family(row["label_type"]) == "curve":
                curve_identities.add(identity)
            for column in ("experimental_sigma_A2", "beb_sigma_A2"):
                value = float(row[column])
                require(math.isfinite(value) and value > 0, f"Invalid {column} in row {rows}")

    require(rows == 3533, f"Expected 3,533 observations, found {rows}")
    require(len(identities) == 194, f"Expected 194 molecules, found {len(identities)}")
    require(len(curve_identities) == 74, f"Expected 74 curve molecules, found {len(curve_identities)}")
    require(folds == {0, 1, 2, 3, 4}, f"Unexpected fold assignments: {sorted(folds)}")
    require(all(len(value) == 1 for value in identity_folds.values()), "A molecule occurs in multiple folds")
    require(task_counts == {"curve": 3316, "fixed_sparse": 149, "peak": 68}, "Task row counts changed")
    return {
        "observations": rows,
        "molecules": len(identities),
        "curve_molecules": len(curve_identities),
    }


def close(actual: float, expected: float, tolerance: float = 5e-10) -> bool:
    """Return True when two recorded floating-point results agree."""

    return abs(actual - expected) <= tolerance


def verify_metrics() -> None:
    """Check the headline values used by the report and README."""

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    validity = json.loads(VALIDITY.read_text(encoding="utf-8"))

    require(summary["final_physical_identities"] == 194, "Summary molecule count changed")
    require(summary["final_observations"] == 3533, "Summary observation count changed")
    require(metrics["physical_molecules"] == 194, "Metrics molecule count changed")
    require(metrics["observations"] == 3533, "Metrics observation count changed")

    curve = metrics["per_task"]["curve"]
    require(
        close(curve["raw"]["primary_molecule_macro_NMAE"], 0.15276432279108756),
        "Raw curve NMAE changed",
    )
    require(
        close(curve["ml"]["primary_molecule_macro_NMAE"], 0.14169094668747828),
        "ML curve NMAE changed",
    )

    source = validity["source_batch_summary"]
    require(source["curve_molecules_in_stress_test"] == 73, "Source stress-test count changed")
    require(
        close(source["weighted_raw_primary_macro_NMAE"], 0.1537811170727106),
        "Raw source-group NMAE changed",
    )
    require(
        close(source["weighted_ml_primary_macro_NMAE"], 0.17381536123812524),
        "ML source-group NMAE changed",
    )

    with (METRICS.parent / "expanded_194_oof_predictions.csv").open(newline="", encoding="utf-8-sig") as handle:
        predictions = list(csv.DictReader(handle))
    for scope in ("all", "curve", "fixed_sparse", "peak"):
        subset = predictions if scope == "all" else [row for row in predictions if task_family(row["label_type"]) == scope]
        expected = metrics if scope == "all" else metrics["per_task"][scope]
        for method, column, all_key in (
            ("raw", "prediction_raw_beb_A2", "raw_metrics"),
            ("constant", "prediction_constant_scale_A2", "constant_scale_metrics"),
            ("ml", "prediction_expanded_ml_A2", "expanded_ml_metrics"),
        ):
            require(close(macro(subset, column), expected[all_key if scope == "all" else method]["primary_molecule_macro_NMAE"]), f"CSV independently recomputed {scope}/{method} metric differs")
    batches = validity["leave_one_source_batch_out"]
    require(sum(row["molecules"] for row in batches) == 73, "Source batch sizes disagree")
    for method in ("raw", "ml"):
        weighted = sum(row["molecules"] * row[f"{method}_primary_macro_NMAE"] for row in batches) / 73
        require(close(weighted, source[f"weighted_{method}_primary_macro_NMAE"]), "Source macro weighting differs")
    interval = validity["curve_primary_result"]["improvement_percentage_points_95_CI"]
    require(interval[0] < 0 < interval[1], "Corrected curve interval should cross zero")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_corrections() -> None:
    """Check every withdrawal, retained fold, and corrected peak metadata field."""
    current = read_csv(TRAINING)
    original = read_csv(ROOT / "audit/legacy_r2/expanded_194_training_points.csv")
    quarantined = read_csv(ROOT / "audit/quarantined_rows.csv")
    peak_changes = read_csv(ROOT / "audit/peak_metadata_corrections.csv")
    require(len(original) == 3883 and len(quarantined) == 350, "Audit counts changed")
    current_by_line = {int(row["original_csv_line"]): row for row in current}
    quarantine_lines = {int(row["original_csv_line"]) for row in quarantined}
    require(not quarantine_lines.intersection(current_by_line), "Quarantined row remains in training")
    require(set(current_by_line) | quarantine_lines == set(range(2, 3885)), "Audit does not partition original observations")
    require(Counter(row["quarantine_reason"] for row in quarantined) == {
        "NIST_exp_5_is_theory_not_experiment": 7,
        "ambiguous_isomer_trace_quarantined": 330,
        "O3_exp_2_renormalized_duplicate_of_Newson": 13,
    }, "Withdrawal reason counts changed")
    require(all(task_family(row["label_type"]) == "curve" for row in quarantined), "Non-curve observation was withdrawn")
    for line, row in current_by_line.items():
        before = original[line - 2]
        for column in ("identity_key", "label_type", "cv_fold", "experimental_sigma_A2", "beb_sigma_A2"):
            require(row[column] == before[column], f"Unexpected retained {column} change on original line {line}")
    require(len(peak_changes) == 15, "Expected 15 published peak metadata corrections")
    for change in peak_changes:
        row = current_by_line[int(change["original_csv_line"])]
        require(task_family(row["label_type"]) == "peak", "Non-peak metadata correction")
        require(row["energy_eV"] == "" and row["energy_is_reported"] == "False", "Hudson peak energy must be unreported")
        require(close(float(row["experimental_uncertainty_A2"]), 0.05 * float(row["experimental_sigma_A2"])), "Hudson peak uncertainty must be 5% of source height")
    ozone = [row for row in current if row["identity_key"] == "ozone" and task_family(row["label_type"]) == "curve"]
    require(len(ozone) == 19, "Expected 19 original ozone observations")
    require(current_by_line[2866]["primary_low_energy"] == "False", "Ozone corrected peak boundary not applied")
    oof = read_csv(METRICS.parent / "expanded_194_oof_predictions.csv")
    require(len(oof) == len(current), "OOF row count differs")
    for observed, predicted in zip(current, oof):
        for column in ("original_csv_line", "original_row_sha256", "identity_key", "label_type", "cv_fold", "primary_low_energy"):
            require(observed[column] == predicted[column], f"OOF alignment differs in {column}")


def verify_no_private_absolute_paths() -> None:
    """Reject personal Windows user paths in published text files."""

    pattern = re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/]+", re.IGNORECASE)
    suffixes = {".py", ".md", ".csv", ".json", ".txt", ".yml", ".yaml"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            require(
                not pattern.search(text),
                f"Private absolute path remains in {relative}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="check scientific invariants after a rerun without requiring byte-identical files",
    )
    args = parser.parse_args()

    checked = 0 if args.skip_checksums else verify_checksums()
    counts = verify_training_table()
    verify_corrections()
    verify_metrics()
    verify_no_private_absolute_paths()
    print(
        "PASS: release is internally consistent "
        f"({counts['molecules']} molecules, {counts['observations']} observations, "
        f"{counts['curve_molecules']} curve molecules, {checked} checksums)."
    )


if __name__ == "__main__":
    main()
