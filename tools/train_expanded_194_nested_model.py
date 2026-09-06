from __future__ import annotations

import itertools
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("BEB_WORKSPACE_ROOT", Path(__file__).absolute().parents[1]))
sys.path.insert(0, str(ROOT / "tools"))

import joblib
import numpy as np
import pandas as pd

import improve_final_beb_model as v2
from train_final_beb_residual_model import balanced_folds, fit_ridge, ridge_predict, weighted_mean


INPUT = ROOT / "outputs" / "expanded_194_training_2026_08_02" / "expanded_194_training_points.csv"
OUT = ROOT / "outputs" / "final_expanded_194_ml_2026_08_02"
OUT.mkdir(parents=True, exist_ok=True)

ELEMENT_Z = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ti": 22,
    "Ga": 31, "Ge": 32, "As": 33, "Br": 35, "I": 53, "W": 74,
}
ELEMENTS = list(ELEMENT_Z)


def task_family(label: str) -> str:
    text = str(label).lower()
    if "peak" in text:
        return "peak"
    if "fixed" in text or "sparse" in text:
        return "fixed_sparse"
    return "curve"


def formula_counts(formula: str) -> Counter:
    counts = Counter()
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", str(formula)):
        counts[element] += float(number) if number else 1.0
    return counts


def feature_matrix(frame: pd.DataFrame):
    # A missing peak energy remains explicitly missing in the scientific table.
    # The numerical imputation below is only an ML feature convention and is paired
    # with an energy-missing flag so it cannot masquerade as a measurement.
    energy = pd.to_numeric(frame.energy_eV, errors="coerce").to_numpy(float)
    energy_missing = ~np.isfinite(energy)
    safe_energy = np.where(energy_missing, 72.5, np.clip(energy, 1e-3, None))
    beb = np.clip(frame.beb_sigma_A2.to_numpy(float), 1e-8, None)

    formula_rows = []
    for formula in frame.formula:
        c = formula_counts(formula)
        total = sum(c.values()) or 1.0
        heavy = total - c.get("H", 0.0)
        electrons = sum(ELEMENT_Z.get(k, 0) * v for k, v in c.items())
        halogens = sum(c.get(k, 0.0) for k in ["F", "Cl", "Br", "I"])
        formula_rows.append(
            [total, heavy, electrons, halogens, heavy / total]
            + [c.get(e, 0.0) for e in ELEMENTS]
        )
    formula_x = np.asarray(formula_rows, float)

    base = np.column_stack([
        np.log(beb), np.log(safe_energy), 1.0 / np.sqrt(safe_energy),
        energy_missing.astype(float), formula_x,
    ])
    names = ["log_beb", "log_energy", "inv_sqrt_energy", "energy_missing",
             "total_atoms", "heavy_atoms", "total_electrons", "halogen_atoms",
             "heavy_atom_fraction"] + [f"n_{e}" for e in ELEMENTS]

    # Theoretical shape features are derived solely from the BEB inputs, so they
    # remain legitimate for held-out molecules. One-point identities carry an
    # explicit no-shape flag rather than fabricated derivatives.
    shape = np.zeros((len(frame), 9), float)
    for _, idx_values in frame.groupby("identity_key").groups.items():
        idx = np.asarray(list(idx_values), int)
        valid = idx[np.isfinite(energy[idx])]
        if len(valid) == 0:
            shape[idx, -1] = 1.0
            continue
        order = valid[np.argsort(energy[valid])]
        e = energy[order]
        b = beb[order]
        unique_e, inverse = np.unique(e, return_inverse=True)
        unique_b = np.asarray([np.mean(b[inverse == i]) for i in range(len(unique_e))])
        peak_idx = int(np.argmax(unique_b))
        peak_e, peak_b = float(unique_e[peak_idx]), float(unique_b[peak_idx])
        if len(unique_e) >= 3:
            le = np.log(np.clip(unique_e, 1e-6, None))
            lb = np.log(np.clip(unique_b, 1e-8, None))
            slope_u = np.gradient(lb, le)
            curve_u = np.gradient(slope_u, le)
            slope, curvature = slope_u[inverse], curve_u[inverse]
            has_shape = 1.0
        else:
            slope = np.zeros(len(order))
            curvature = np.zeros(len(order))
            has_shape = 0.0
        shape[order] = np.column_stack([
            np.full(len(order), peak_b), np.full(len(order), peak_e),
            b / max(peak_b, 1e-8), e / max(peak_e, 1e-8),
            slope, curvature, np.full(len(order), has_shape),
            np.log(np.clip(e / max(peak_e, 1e-8), 1e-8, None)),
            np.zeros(len(order)),
        ])
        missing_idx = np.setdiff1d(idx, order)
        if len(missing_idx):
            shape[missing_idx, 0] = peak_b
            shape[missing_idx, 1] = peak_e
            shape[missing_idx, 6] = has_shape
            shape[missing_idx, 8] = 1.0
    shape_names = ["beb_shape_peak", "beb_shape_peak_energy", "beb_normalized",
                   "energy_over_beb_peak", "beb_log_slope", "beb_log_curvature",
                   "beb_has_multi_point_shape", "log_energy_from_peak", "shape_at_missing_energy"]

    x = np.column_stack([base, shape])
    # Compact physically interpretable interactions; trees also see raw features.
    interactions = np.column_stack([
        x[:, 0] * x[:, 1], x[:, 0] * x[:, 4], x[:, 0] * x[:, 5],
        x[:, 1] * x[:, 4], x[:, 1] * x[:, 5], x[:, 4] * x[:, 5],
        x[:, 5] * x[:, 6], x[:, 0] ** 2, x[:, 1] ** 2,
        x[:, 4] ** 2, x[:, 5] ** 2, x[:, 6] ** 2,
    ])
    interaction_names = [
        "log_beb*log_energy", "log_beb*total_atoms", "log_beb*heavy_atoms",
        "log_energy*total_atoms", "log_energy*heavy_atoms", "total_atoms*heavy_atoms",
        "heavy_atoms*total_electrons", "log_beb^2", "log_energy^2",
        "total_atoms^2", "heavy_atoms^2", "total_electrons^2",
    ]
    out = np.column_stack([x, interactions])
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite ML features remain after explicit imputation")
    return out, names + shape_names + interaction_names


def normalized_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("identity_key").size()
    molecule = frame.identity_key.map(lambda x: 1.0 / counts[x]).to_numpy(float)
    exp = frame.experimental_sigma_A2.to_numpy(float)
    unc = pd.to_numeric(frame.experimental_uncertainty_A2, errors="coerce").to_numpy(float)
    rel = np.divide(unc, exp, out=np.full_like(exp, 0.12), where=np.isfinite(unc))
    quality = 1.0 / (1.0 + (rel / 0.12) ** 2)
    priority = np.where(frame.primary_low_energy.to_numpy(bool), 2.0, 0.75)
    return molecule * quality * priority


def component_oof(frame: pd.DataFrame, x: np.ndarray, config: dict, seed: int):
    y = frame.target_log_ratio.to_numpy(float)
    weights = normalized_weights(frame)
    groups = frame.identity_key.to_numpy(str)
    folds, _ = balanced_folds(groups, min(3, len(set(groups))))
    tree_r = np.zeros(len(frame))
    ridge_r = np.zeros(len(frame))
    global_r = np.zeros(len(frame))
    for fold in sorted(set(folds)):
        tr, va = folds != fold, folds == fold
        tree = v2.make_tree(config, seed + int(fold), n_estimators=100)
        tree.fit(x[tr], y[tr], sample_weight=weights[tr])
        tree_r[va] = tree.predict(x[va])
        ridge = fit_ridge(x[tr], y[tr], weights[tr], alpha=1.0)
        ridge_r[va] = ridge_predict(ridge, x[va])
        global_r[va] = weighted_mean(y[tr], weights[tr])
    return tree_r, ridge_r, global_r


def simplex_weights(step=0.25):
    grid = np.arange(0.0, 1.0001, step)
    for wt, wr, wg in itertools.product(grid, repeat=3):
        wraw = 1.0 - wt - wr - wg
        if wraw >= -1e-9:
            yield float(wt), float(wr), float(wg), float(max(0.0, wraw))


def select_model(frame: pd.DataFrame, x: np.ndarray, seed: int):
    beb = frame.beb_sigma_A2.to_numpy(float)
    candidates = []
    for ci, config in enumerate(v2.TREE_CONFIGS):
        tree_r, ridge_r, global_r = component_oof(frame, x, config, seed + 1000 * ci)
        best = None
        for weights in simplex_weights():
            wt, wr, wg, _ = weights
            residual = wt * tree_r + wr * ridge_r + wg * global_r
            pred = beb * np.exp(np.clip(residual, math.log(0.25), math.log(4.0)))
            score = v2.score_predictions(frame, pred)
            item = (score, weights)
            if best is None or item < best:
                best = item
        candidates.append((best[0], config, best[1]))
    return min(candidates, key=lambda item: item[0]), candidates


def fit_predict(frame: pd.DataFrame, x_train: np.ndarray, x_test: np.ndarray,
                config: dict, mix: tuple[float, float, float, float], seed: int,
                final=False):
    y = frame.target_log_ratio.to_numpy(float)
    weights = normalized_weights(frame)
    tree = v2.make_tree(config, seed, n_estimators=450 if final else 220)
    tree.fit(x_train, y, sample_weight=weights)
    tree_r = tree.predict(x_test)
    ridge = fit_ridge(x_train, y, weights, alpha=1.0)
    ridge_r = ridge_predict(ridge, x_test)
    global_r = weighted_mean(y, weights)
    wt, wr, wg, _ = mix
    residual = wt * tree_r + wr * ridge_r + wg * global_r
    residual = np.clip(residual, math.log(0.25), math.log(4.0))
    return residual, {
        "tree": tree, "ridge": ridge, "global_residual": global_r,
        "weights": {"tree": wt, "ridge": wr, "global": wg, "raw": mix[3]},
        "config": config,
    }


def constant_scale_oof(frame: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(frame))
    folds = frame.cv_fold.to_numpy(int)
    for fold in sorted(set(folds)):
        for task in sorted(frame.task_family.unique()):
            tr = (folds != fold) & (frame.task_family.to_numpy(str) == task)
            va = (folds == fold) & (frame.task_family.to_numpy(str) == task)
            if not va.any():
                continue
            scale = math.exp(weighted_mean(frame.loc[tr, "target_log_ratio"].to_numpy(float),
                                           normalized_weights(frame.loc[tr].reset_index(drop=True))))
            pred[va] = frame.loc[va, "beb_sigma_A2"].to_numpy(float) * scale
    return pred


def main():
    started = time.perf_counter()
    frame = pd.read_csv(INPUT).reset_index(drop=True)
    assert len(frame) == 3533 and frame.identity_key.nunique() == 194
    assert frame.groupby("identity_key").cv_fold.nunique().eq(1).all()
    frame["task_family"] = frame.label_type.map(task_family)
    frame["target_log_ratio"] = np.log(frame.experimental_sigma_A2 / frame.beb_sigma_A2)
    x, feature_names = feature_matrix(frame)
    folds = frame.cv_fold.to_numpy(int)
    groups = frame.identity_key.to_numpy(str)
    oof_resid = np.zeros(len(frame))
    outer_audit = []

    for fold in sorted(set(folds)):
        for task in sorted(frame.task_family.unique()):
            tr = (folds != fold) & (frame.task_family.to_numpy(str) == task)
            va = (folds == fold) & (frame.task_family.to_numpy(str) == task)
            if not va.any():
                continue
            train = frame.loc[tr].reset_index(drop=True)
            selected, candidates = select_model(train, x[tr], 210000 + fold * 10000)
            score, config, mix = selected
            residual, _ = fit_predict(train, x[tr], x[va], config, mix,
                                      220000 + fold * 100 + len(outer_audit))
            oof_resid[va] = residual
            outer_audit.append({
                "fold": int(fold), "task": task,
                "train_molecules": int(len(set(groups[tr]))),
                "test_molecules": int(len(set(groups[va]))),
                "selected_config": config,
                "selected_weights": {"tree": mix[0], "ridge": mix[1], "global": mix[2], "raw": mix[3]},
                "inner_primary_macro_NMAE": score,
                "candidates": [{"score": s, "config": c,
                                "weights": {"tree": w[0], "ridge": w[1], "global": w[2], "raw": w[3]}}
                               for s, c, w in candidates],
            })
            print(f"fold={fold} task={task} inner={score:.4f} weights={mix}", flush=True)

    frame["prediction_raw_beb_A2"] = frame.beb_sigma_A2
    frame["prediction_constant_scale_A2"] = constant_scale_oof(frame)
    frame["prediction_expanded_ml_A2"] = frame.beb_sigma_A2 * np.exp(oof_resid)

    raw = v2.metrics(frame, frame.prediction_raw_beb_A2.to_numpy(float))
    const = v2.metrics(frame, frame.prediction_constant_scale_A2.to_numpy(float))
    ml = v2.metrics(frame, frame.prediction_expanded_ml_A2.to_numpy(float))
    boot_raw = v2.bootstrap(frame, frame.prediction_raw_beb_A2.to_numpy(float),
                            frame.prediction_expanded_ml_A2.to_numpy(float), repeats=20000)
    boot_const = v2.bootstrap(frame, frame.prediction_constant_scale_A2.to_numpy(float),
                              frame.prediction_expanded_ml_A2.to_numpy(float), repeats=20000)

    final_models, final_selection = {}, {}
    for task in sorted(frame.task_family.unique()):
        mask = frame.task_family.to_numpy(str) == task
        task_frame = frame.loc[mask].reset_index(drop=True)
        selected, candidates = select_model(task_frame, x[mask], 300000)
        score, config, mix = selected
        _, fitted = fit_predict(task_frame, x[mask], x[mask], config, mix, 310000, final=True)
        final_models[task] = fitted
        final_selection[task] = {
            "inner_primary_macro_NMAE": score, "config": config,
            "weights": {"tree": mix[0], "ridge": mix[1], "global": mix[2], "raw": mix[3]},
            "molecules": int(task_frame.identity_key.nunique()), "rows": int(len(task_frame)),
        }

    per_task = {}
    for task, g in frame.groupby("task_family"):
        per_task[task] = {
            "molecules": int(g.identity_key.nunique()), "rows": int(len(g)),
            "raw": v2.metrics(g.reset_index(drop=True), g.prediction_raw_beb_A2.to_numpy(float)),
            "constant": v2.metrics(g.reset_index(drop=True), g.prediction_constant_scale_A2.to_numpy(float)),
            "ml": v2.metrics(g.reset_index(drop=True), g.prediction_expanded_ml_A2.to_numpy(float)),
        }

    package = {
        "version": "evidence_corrected_v3_194_nested_multitask_2026_09_06",
        "models": final_models, "feature_names": feature_names,
        "task_routing": "peak / fixed_sparse / curve",
        "missing_energy_policy": "72.5 eV numerical imputation plus explicit missing flag; never reported as experimental energy",
    }
    joblib.dump(package, OUT / "expanded_194_nested_model.joblib", compress=3)
    frame.to_csv(OUT / "expanded_194_oof_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "physical_molecules": int(frame.identity_key.nunique()),
        "observations": int(len(frame)),
        "curve_molecules": int(frame.loc[frame.task_family == "curve", "identity_key"].nunique()),
        "validation": "Nested five-fold physical-identity-disjoint cross-validation; aliases merged before folds",
        "raw_metrics": raw, "constant_scale_metrics": const, "expanded_ml_metrics": ml,
        "relative_reduction_vs_raw_percent": 100 * (1 - ml["primary_molecule_macro_NMAE"] / raw["primary_molecule_macro_NMAE"]),
        "absolute_reduction_vs_raw_percentage_points": 100 * (raw["primary_molecule_macro_NMAE"] - ml["primary_molecule_macro_NMAE"]),
        "bootstrap_vs_raw": boot_raw, "bootstrap_vs_constant": boot_const,
        "per_task": per_task, "outer_fold_audit": outer_audit,
        "final_selection": final_selection,
        "scientific_note": "All 194 identities remain, including 74 curve-bearing identities. Seven water theory labels, 330 ambiguous ester curve labels and thirteen duplicate renormalised ozone labels are quarantined. All 67 peak-to-peak amplitude observations have missing common-energy features; 27 source-reported experimental peak energies are retained only as audit metadata. Nitromethane is a matched 60 eV fixed-energy pair. The exact late corrections and six original-table experimental peak replacements are replayed from audit/late_source_corrections.json; BEB amplitudes are unchanged. The six replacements use Bart's documented 3.9% instrumental bound, not a standard deviation; fifteen Hudson peak uncertainties use 5% source accuracy. Nine retained ester curves are peak-anchored figure-trace extractions. Original outer folds, 49 features and the model-search protocol are retained without performance-driven retuning.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started},
        "correction_scope": json.loads((ROOT / "audit" / "correction_scope.json").read_text(encoding="utf-8")),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Final expanded 194-molecule BEB residual model\n\n"
        f"- Physical molecules: **{summary['physical_molecules']}**\n"
        f"- Numeric observations: **{summary['observations']}**\n"
        f"- Curve-bearing identities: **{summary['curve_molecules']}**\n"
        f"- Curve task raw BEB / routed ML NMAE: **{per_task['curve']['raw']['primary_molecule_macro_NMAE']*100:.4f}% / {per_task['curve']['ml']['primary_molecule_macro_NMAE']*100:.4f}%**\n"
        f"- Supplementary mixed-evidence raw BEB / ML NMAE: **{raw['primary_molecule_macro_NMAE']*100:.4f}% / {ml['primary_molecule_macro_NMAE']*100:.4f}%**\n\n"
        "The mixed-evidence score combines distinct target types and is not a complete-curve score. Read the full validity results for the conditional uncertainty interval and negative source-batch transfer result.\n\n"
        "All scores are out-of-fold on held-out physical molecular identities. Historical aliases are merged before fold assignment.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "molecules": summary["physical_molecules"], "observations": summary["observations"],
        "raw_primary_NMAE": raw["primary_molecule_macro_NMAE"],
        "constant_primary_NMAE": const["primary_molecule_macro_NMAE"],
        "ml_primary_NMAE": ml["primary_molecule_macro_NMAE"],
        "relative_reduction_vs_raw_percent": summary["relative_reduction_vs_raw_percent"],
        "bootstrap_vs_raw": boot_raw,
    }, indent=2))


if __name__ == "__main__":
    main()
