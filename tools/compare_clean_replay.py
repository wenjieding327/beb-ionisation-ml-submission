"""Compare a separate-path numerical replay with the frozen corrected release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(left, right, location="root") -> int:
    """Require agreement, allowing only runtime variation and roundoff."""
    if isinstance(left, dict):
        assert set(left) == set(right), location
        return sum(compare(value, right[key], f"{location}.{key}") for key, value in left.items() if key != "runtime")
    if isinstance(left, list):
        assert len(left) == len(right), location
        return sum(compare(a, b, f"{location}[{number}]") for number, (a, b) in enumerate(zip(left, right)))
    if isinstance(left, (float, int)) and not isinstance(left, bool):
        assert math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-12), location
    else:
        assert left == right, location
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_root", type=Path)
    args = parser.parse_args()
    other = args.comparison_root.resolve()
    assert other != ROOT.resolve(), "A separate directory is required"
    result = {"status": "PASS", "method": "Full nested training and full validity/ablation suite rerun from a copy in a different directory; numerical values compared with 1e-12 absolute/relative tolerance; runtime excluded.", "comparison_directory_role": "separate local working copy (not an independent external laboratory)", "json_comparisons": [], "oof_cells_compared": 0}
    for relative in ("outputs/final_expanded_194_ml_2026_08_02/metrics.json", "outputs/final_validity_ablation_corrected_2026_08_05/validity_ablation_results.json"):
        left = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        right = json.loads((other / relative).read_text(encoding="utf-8"))
        count = compare(left, right)
        result["json_comparisons"].append({"path": relative, "checked_non_runtime_values": count, "frozen_runtime": left["runtime"], "replay_runtime": right["runtime"]})
    relative = "outputs/final_expanded_194_ml_2026_08_02/expanded_194_oof_predictions.csv"
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as stream:
        left_rows = list(csv.DictReader(stream))
    with (other / relative).open(newline="", encoding="utf-8-sig") as stream:
        right_rows = list(csv.DictReader(stream))
    assert len(left_rows) == len(right_rows) == 3533
    maximum_difference = 0.0
    for number, (left, right) in enumerate(zip(left_rows, right_rows), 2):
        assert left.keys() == right.keys()
        for key, value in left.items():
            result["oof_cells_compared"] += 1
            if value == right[key]:
                continue
            try:
                a, b = float(value), float(right[key])
            except ValueError:
                raise AssertionError(f"OOF line {number}, column {key}")
            assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (number, key)
            maximum_difference = max(maximum_difference, abs(a-b))
    result["maximum_numeric_cell_difference"] = maximum_difference
    relative = "outputs/expanded_194_training_2026_08_02/expanded_194_training_points.csv"
    original_bytes = (ROOT / relative).read_bytes()
    assert original_bytes == (other / relative).read_bytes()
    result["identical_input_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    (ROOT / "audit/clean_path_replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
