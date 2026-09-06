from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("BEB_WORKSPACE_ROOT", Path(__file__).absolute().parents[1]))
sys.path.insert(0, str(ROOT / ".ml_packages_2026_07_30"))
sys.path.insert(0, str(ROOT / "tools"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import improve_final_beb_model as v2
import train_expanded_194_nested_model as final194
from train_final_beb_residual_model import fit_ridge, ridge_predict, weighted_mean


INPUT = ROOT / "outputs" / "expanded_194_training_2026_08_02" / "expanded_194_training_points.csv"
MODEL_OUT = ROOT / "outputs" / "final_expanded_194_ml_2026_08_02"
OUT = ROOT / "outputs" / "final_validity_ablation_corrected_2026_08_05"
OUT.mkdir(parents=True, exist_ok=True)


def task_family(label: str) -> str:
    text = str(label).lower()
    if "peak" in text:
        return "peak"
    if "fixed" in text or "sparse" in text:
        return "fixed_sparse"
    return "curve"


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return v2.metrics(frame.reset_index(drop=True), np.asarray(prediction, float))


def curve_bootstrap(frame: pd.DataFrame, old_pred: np.ndarray, new_pred: np.ndarray,
                    repeats: int = 20000) -> dict:
    rows = []
    frame = frame.reset_index(drop=True)
    for _, g in frame.groupby("identity_key"):
        idx = g.index.to_numpy(int)
        primary = g.primary_low_energy.to_numpy(bool)
        if not primary.any():
            primary[:] = True
        idx = idx[primary]
        exp = frame.loc[idx, "experimental_sigma_A2"].to_numpy(float)
        old = old_pred[idx]
        new = new_pred[idx]
        rows.append((np.abs(old - exp).sum() / exp.sum(),
                     np.abs(new - exp).sum() / exp.sum()))
    paired = np.asarray(rows)
    rng = np.random.default_rng(20260802)
    draw = rng.integers(0, len(paired), (repeats, len(paired)))
    diff = (paired[:, 0] - paired[:, 1])[draw].mean(axis=1) * 100
    return {
        "raw_primary_molecule_macro_NMAE": float(paired[:, 0].mean()),
        "ml_primary_molecule_macro_NMAE": float(paired[:, 1].mean()),
        "absolute_improvement_percentage_points": float((paired[:, 0] - paired[:, 1]).mean() * 100),
        "relative_reduction_percent": float(100 * (1 - paired[:, 1].mean() / paired[:, 0].mean())),
        "improvement_percentage_points_95_CI": [float(np.quantile(diff, 0.025)),
                                                  float(np.quantile(diff, 0.975))],
        "probability_improved": float(np.mean(diff > 0)),
        "resamples": int(repeats),
        "molecules": int(len(paired)),
    }


def molecule_primary_errors(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Return one primary-region NMAE value per physical identity."""
    frame = frame.reset_index(drop=True)
    prediction = np.asarray(prediction, float)
    rows = []
    for identity, group in frame.groupby("identity_key"):
        idx = group.index.to_numpy(int)
        primary = group.primary_low_energy.to_numpy(bool)
        if not primary.any():
            primary[:] = True
        idx = idx[primary]
        exp = frame.loc[idx, "experimental_sigma_A2"].to_numpy(float)
        rows.append({
            "identity_key": identity,
            "nmae": float(np.abs(prediction[idx] - exp).sum() / exp.sum()),
        })
    return pd.DataFrame(rows).sort_values("identity_key").reset_index(drop=True)


def paired_method_bootstrap(frame: pd.DataFrame, baseline_pred: np.ndarray,
                            candidate_pred: np.ndarray, comparison: str,
                            scope: str, repeats: int = 20000) -> dict:
    """Bootstrap the molecule-macro NMAE difference: baseline minus candidate."""
    baseline = molecule_primary_errors(frame, baseline_pred)
    candidate = molecule_primary_errors(frame, candidate_pred)
    if not baseline.identity_key.equals(candidate.identity_key):
        raise RuntimeError("Paired ablation identities are not aligned")
    paired = baseline.nmae.to_numpy(float) - candidate.nmae.to_numpy(float)
    rng = np.random.default_rng(20260803 + len(comparison) + len(scope))
    draw = rng.integers(0, len(paired), (repeats, len(paired)))
    sampled = paired[draw].mean(axis=1) * 100
    positive = int(np.sum(sampled > 0))
    return {
        "scope": scope,
        "comparison": comparison,
        "molecules": int(len(paired)),
        "baseline_primary_macro_NMAE": float(baseline.nmae.mean()),
        "candidate_primary_macro_NMAE": float(candidate.nmae.mean()),
        "baseline_minus_candidate_percentage_points": float(paired.mean() * 100),
        "difference_95_CI_low": float(np.quantile(sampled, 0.025)),
        "difference_95_CI_high": float(np.quantile(sampled, 0.975)),
        "positive_resamples": positive,
        "resamples": int(repeats),
    }


def selected_outer_models(summary: dict) -> dict:
    selected = {}
    for item in summary["outer_fold_audit"]:
        mix = item["selected_weights"]
        selected[(int(item["fold"]), item["task"])] = (
            item["selected_config"],
            (float(mix["tree"]), float(mix["ridge"]),
             float(mix["global"]), float(mix["raw"])),
        )
    return selected


def residual_ablation_oof(frame: pd.DataFrame, x: np.ndarray, mask: np.ndarray,
                          selected: dict, seed: int) -> np.ndarray:
    folds = frame.cv_fold.to_numpy(int)
    tasks = frame.task_family.to_numpy(str)
    pred = np.zeros(len(frame))
    for fold in sorted(set(folds)):
        for task in sorted(set(tasks)):
            tr = (folds != fold) & (tasks == task)
            va = (folds == fold) & (tasks == task)
            if not va.any():
                continue
            train = frame.loc[tr].reset_index(drop=True)
            config, mix = selected[(int(fold), task)]
            residual, _ = final194.fit_predict(
                train, x[tr][:, mask], x[va][:, mask], config, mix,
                seed + int(fold) * 100 + len(task),
            )
            pred[va] = frame.loc[va, "beb_sigma_A2"].to_numpy(float) * np.exp(residual)
    return pred


def direct_log_oof(frame: pd.DataFrame, x: np.ndarray, selected: dict, seed: int) -> np.ndarray:
    folds = frame.cv_fold.to_numpy(int)
    tasks = frame.task_family.to_numpy(str)
    pred = np.zeros(len(frame))
    for fold in sorted(set(folds)):
        for task in sorted(set(tasks)):
            tr = (folds != fold) & (tasks == task)
            va = (folds == fold) & (tasks == task)
            if not va.any():
                continue
            train = frame.loc[tr].reset_index(drop=True)
            y = np.log(np.clip(train.experimental_sigma_A2.to_numpy(float), 1e-8, None))
            weights = final194.normalized_weights(train)
            config, mix = selected[(int(fold), task)]
            tree = v2.make_tree(config, seed + int(fold) * 100 + len(task), n_estimators=220)
            tree.fit(x[tr], y, sample_weight=weights)
            tree_log = tree.predict(x[va])
            ridge = fit_ridge(x[tr], y, weights, alpha=1.0)
            ridge_log = ridge_predict(ridge, x[va])
            global_log = weighted_mean(y, weights)
            wt, wr, wg, wraw = mix
            raw_log = np.log(np.clip(frame.loc[va, "beb_sigma_A2"].to_numpy(float), 1e-8, None))
            log_sigma = wt * tree_log + wr * ridge_log + wg * global_log + wraw * raw_log
            pred[va] = np.exp(np.clip(log_sigma, math.log(1e-6), math.log(200.0)))
    return pred


def pooled_oof(frame: pd.DataFrame, x: np.ndarray, seed: int) -> np.ndarray:
    folds = frame.cv_fold.to_numpy(int)
    pred = np.zeros(len(frame))
    for fold in sorted(set(folds)):
        tr, va = folds != fold, folds == fold
        train = frame.loc[tr].reset_index(drop=True)
        _, config, mix = final194.select_model(train, x[tr], seed + int(fold) * 1000)[0]
        residual, _ = final194.fit_predict(train, x[tr], x[va], config, mix,
                                           seed + int(fold) * 1000 + 500)
        pred[va] = frame.loc[va, "beb_sigma_A2"].to_numpy(float) * np.exp(residual)
    return pred


def source_batch(path: str) -> str:
    text = str(path).lower().replace("\\", "/")
    if "ready15_esters" in text:
        return "Hudson ester family"
    if "qec_low_energy_priority_curve_points" in text:
        return "QEC priority batch"
    if "nist_fast12_experiment" in text:
        return "NIST graph-digitised batch"
    if "expanded_measured_total_master" in text:
        return "Expanded measured batch"
    if "aip_fig9_12_13" in text:
        return "Heavy-ECP figure batch"
    if "siclx_experiment" in text:
        return "SiClx figure batch"
    if "nist_strict_usable_numeric_pairs" in text:
        return "NIST curated curve batch"
    return "Other/single source"


def source_batch_membership(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for identity, group in curves.groupby("identity_key"):
        batches = sorted(set(group.source.map(source_batch)))
        if len(batches) != 1:
            raise RuntimeError(f"Curve identity {identity} maps to multiple batches: {batches}")
        rows.append({
            "held_out_batch": batches[0],
            "identity_key": identity,
            "molecule_name": str(group.molecule_name.iloc[0]),
            "formula": str(group.formula.iloc[0]),
            "rows": int(len(group)),
            "source_locator": " | ".join(sorted(set(group.source.astype(str)))),
        })
    membership = pd.DataFrame(rows)
    counts = membership.groupby("held_out_batch").identity_key.transform("nunique")
    membership["batch_molecules"] = counts.astype(int)
    membership["included_in_batch_stress_test"] = counts.ge(3)
    return membership.sort_values(["held_out_batch", "identity_key"]).reset_index(drop=True)


def custom_weights(frame: pd.DataFrame, missing_uncertainty: float = 0.12,
                   uncertainty_scale: float = 0.12,
                   primary_weight: float = 2.0,
                   secondary_weight: float = 0.75) -> np.ndarray:
    counts = frame.groupby("identity_key").size()
    molecule = frame.identity_key.map(lambda key: 1.0 / counts[key]).to_numpy(float)
    exp = frame.experimental_sigma_A2.to_numpy(float)
    unc = pd.to_numeric(frame.experimental_uncertainty_A2, errors="coerce").to_numpy(float)
    rel = np.divide(unc, exp, out=np.full_like(exp, missing_uncertainty), where=np.isfinite(unc))
    quality = 1.0 / (1.0 + (rel / uncertainty_scale) ** 2)
    priority = np.where(frame.primary_low_energy.to_numpy(bool), primary_weight, secondary_weight)
    return molecule * quality * priority


def fit_predict_policy(train: pd.DataFrame, x_train: np.ndarray, x_test: np.ndarray,
                       config: dict, mix: tuple[float, float, float, float], seed: int,
                       clip_low: float = 0.25, clip_high: float = 4.0,
                       missing_uncertainty: float = 0.12,
                       uncertainty_scale: float = 0.12,
                       primary_weight: float = 2.0,
                       secondary_weight: float = 0.75) -> np.ndarray:
    y = train.target_log_ratio.to_numpy(float)
    weights = custom_weights(
        train, missing_uncertainty, uncertainty_scale, primary_weight, secondary_weight
    )
    tree = v2.make_tree(config, seed, n_estimators=220)
    tree.fit(x_train, y, sample_weight=weights)
    tree_r = tree.predict(x_test)
    ridge = fit_ridge(x_train, y, weights, alpha=1.0)
    ridge_r = ridge_predict(ridge, x_test)
    global_r = weighted_mean(y, weights)
    wt, wr, wg, _ = mix
    residual = wt * tree_r + wr * ridge_r + wg * global_r
    return np.clip(residual, math.log(clip_low), math.log(clip_high))


def sensitivity_oof(curves: pd.DataFrame, x_curves: np.ndarray, selected: dict,
                    **policy) -> np.ndarray:
    folds = curves.cv_fold.to_numpy(int)
    pred = np.zeros(len(curves))
    for fold in sorted(set(folds)):
        tr, va = folds != fold, folds == fold
        train = curves.loc[tr].reset_index(drop=True)
        config, mix = selected[(int(fold), "curve")]
        residual = fit_predict_policy(
            train, x_curves[tr], x_curves[va], config, mix,
            220000 + int(fold) * 103, **policy,
        )
        pred[va] = curves.loc[va, "beb_sigma_A2"].to_numpy(float) * np.exp(residual)
    return pred


def leave_one_batch_out(curves: pd.DataFrame, x_curves: np.ndarray, seed: int) -> pd.DataFrame:
    batches = curves.source.map(source_batch).to_numpy(str)
    rows = []
    for index, batch in enumerate(sorted(set(batches))):
        te = batches == batch
        if curves.loc[te, "identity_key"].nunique() < 3:
            continue
        tr = ~te
        train = curves.loc[tr].reset_index(drop=True)
        selected, _ = final194.select_model(train, x_curves[tr], seed + index * 1000)
        inner_score, config, mix = selected
        residual, _ = final194.fit_predict(train, x_curves[tr], x_curves[te],
                                           config, mix, seed + index * 1000 + 500)
        raw = curves.loc[te, "beb_sigma_A2"].to_numpy(float)
        ml = raw * np.exp(residual)
        test = curves.loc[te].reset_index(drop=True)
        raw_m = metrics(test, raw)
        ml_m = metrics(test, ml)
        rows.append({
            "held_out_batch": batch,
            "molecules": int(test.identity_key.nunique()),
            "rows": int(len(test)),
            "raw_primary_macro_NMAE": raw_m["primary_molecule_macro_NMAE"],
            "ml_primary_macro_NMAE": ml_m["primary_molecule_macro_NMAE"],
            "improvement_percentage_points": 100 * (
                raw_m["primary_molecule_macro_NMAE"] - ml_m["primary_molecule_macro_NMAE"]
            ),
            "inner_selected_NMAE": float(inner_score),
            "selected_config": json.dumps(config, sort_keys=True),
            "selected_weights": json.dumps({"tree": mix[0], "ridge": mix[1],
                                             "global": mix[2], "raw": mix[3]}),
        })
    return pd.DataFrame(rows).sort_values("molecules", ascending=False).reset_index(drop=True)


def importance_tables(feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    package = joblib.load(MODEL_OUT / "expanded_194_nested_model.joblib")
    summary = json.loads((MODEL_OUT / "metrics.json").read_text(encoding="utf-8"))
    aggregate = np.zeros(len(feature_names), float)
    total_weight = 0.0
    task_rows = []
    for task, fitted in package["models"].items():
        tree = fitted["tree"]
        ridge = fitted["ridge"]
        wt = float(fitted["weights"]["tree"])
        wr = float(fitted["weights"]["ridge"])
        tree_imp = np.asarray(getattr(tree, "feature_importances_", np.zeros(len(feature_names))), float)
        ridge_imp = np.abs(np.asarray(ridge["coef"][1:], float))
        if tree_imp.sum() > 0:
            tree_imp /= tree_imp.sum()
        if ridge_imp.sum() > 0:
            ridge_imp /= ridge_imp.sum()
        combined = wt * tree_imp + wr * ridge_imp
        if combined.sum() > 0:
            combined /= combined.sum()
        molecule_weight = float(summary["per_task"][task]["molecules"])
        aggregate += molecule_weight * combined
        total_weight += molecule_weight
        for name, value in zip(feature_names, combined):
            task_rows.append({"task": task, "feature": name, "relative_importance": float(value)})
    aggregate /= total_weight

    def group(name: str) -> str:
        shape = {
            "beb_shape_peak", "beb_shape_peak_energy", "beb_normalized",
            "energy_over_beb_peak", "beb_log_slope", "beb_log_curvature",
            "beb_has_multi_point_shape", "log_energy_from_peak", "shape_at_missing_energy",
        }
        composition = {
            "total_atoms", "heavy_atoms", "total_electrons", "halogen_atoms",
            "heavy_atom_fraction",
        }
        if name in shape:
            return "BEB curve-shape descriptors"
        if name in composition or name.startswith("n_"):
            return "Molecular composition"
        if "*" in name or "^2" in name:
            return "Interaction terms"
        if name in {"log_energy", "inv_sqrt_energy", "energy_missing"}:
            return "Collision energy"
        return "BEB magnitude"

    feature = pd.DataFrame({"feature": feature_names, "relative_importance": aggregate})
    feature["feature_group"] = feature.feature.map(group)
    feature = feature.sort_values("relative_importance", ascending=False).reset_index(drop=True)
    grouped = feature.groupby("feature_group", as_index=False).relative_importance.sum()
    grouped = grouped.sort_values("relative_importance", ascending=False).reset_index(drop=True)
    pd.DataFrame(task_rows).to_csv(OUT / "feature_importance_by_task.csv", index=False, encoding="utf-8-sig")
    return feature, grouped


def plot_ablation(ablation: pd.DataFrame, curve_count: int, all_count: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    show = ablation.copy()
    labels = {
        "raw_beb": "Raw BEB",
        "constant_scale": "Constant scale",
        "direct_log_prediction": "Direct log prediction",
        "residual_energy_beb_only": "Residual: BEB + energy only",
        "residual_no_molecular_composition": "Residual: no composition",
        "residual_no_beb_shape": "Residual: no BEB-shape",
        "pooled_no_task_routing": "Residual: pooled tasks",
        "full_routed_residual": "Full routed residual",
    }
    show["label"] = show.method.map(labels)
    y = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(8.1, 4.9))
    w = 0.36
    ax.barh(y + w / 2, show.curve_primary_macro_NMAE * 100, height=w,
            label=f"Curve task ({curve_count} molecules)", color="#176B87")
    ax.barh(y - w / 2, show.all_evidence_primary_macro_NMAE * 100, height=w,
            label=f"All evidence tasks ({all_count} molecules)", color="#D17A22")
    ax.set_yticks(y, show.label)
    ax.invert_yaxis()
    ax.set_xlabel("Primary molecule-macro NMAE (%) - lower is better")
    ax.set_title("Controlled model and feature ablations")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT / "12_validity_ablation_study.png", dpi=220)
    plt.close(fig)


def plot_batches(batch: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    show = batch.sort_values("raw_primary_macro_NMAE", ascending=True).reset_index(drop=True)
    labels = [f"{name} (n={n})" for name, n in zip(show.held_out_batch, show.molecules)]
    y = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(8.1, 4.8))
    w = 0.36
    ax.barh(y + w / 2, show.raw_primary_macro_NMAE * 100, height=w,
            label="Raw BEB", color="#78909C")
    ax.barh(y - w / 2, show.ml_primary_macro_NMAE * 100, height=w,
            label="Residual ML", color="#B04759")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Primary molecule-macro NMAE (%) - lower is better")
    ax.set_title("Leave-one-source-batch-out curve tests")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "13_leave_one_source_batch_out.png", dpi=220)
    plt.close(fig)


def plot_importance(feature: pd.DataFrame, grouped: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    g = grouped.sort_values("relative_importance")
    axes[0].barh(g.feature_group, g.relative_importance * 100, color="#176B87")
    axes[0].set_xlabel("Aggregated relative importance (%)")
    axes[0].set_title("Feature groups")
    top = feature.head(10).sort_values("relative_importance")
    axes[1].barh(top.feature, top.relative_importance * 100, color="#D17A22")
    axes[1].set_xlabel("Aggregated relative importance (%)")
    axes[1].set_title("Ten highest-ranked features")
    fig.suptitle("Final-model inspection (descriptive, not causal)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "14_model_feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    frame = pd.read_csv(INPUT).reset_index(drop=True)
    frame["task_family"] = frame.label_type.map(task_family)
    frame["target_log_ratio"] = np.log(frame.experimental_sigma_A2 / frame.beb_sigma_A2)
    x, feature_names = final194.feature_matrix(frame)
    summary = json.loads((MODEL_OUT / "metrics.json").read_text(encoding="utf-8"))
    selected = selected_outer_models(summary)
    original = pd.read_csv(MODEL_OUT / "expanded_194_oof_predictions.csv")
    # Validity arrays are positional: reject stale or differently ordered OOF files.
    for column in ["original_csv_line", "original_row_sha256", "identity_key", "label_type"]:
        if not frame[column].astype(str).equals(original[column].astype(str)):
            raise RuntimeError(f"Input and OOF rows are not aligned: {column}")
    for column in ["experimental_sigma_A2", "beb_sigma_A2", "energy_eV"]:
        if not np.allclose(frame[column].to_numpy(float), original[column].to_numpy(float), equal_nan=True):
            raise RuntimeError(f"Input and OOF numeric values differ: {column}")

    names = np.asarray(feature_names, object)
    shape_names = {
        "beb_shape_peak", "beb_shape_peak_energy", "beb_normalized",
        "energy_over_beb_peak", "beb_log_slope", "beb_log_curvature",
        "beb_has_multi_point_shape", "log_energy_from_peak", "shape_at_missing_energy",
    }
    composition_tokens = {"total_atoms", "heavy_atoms", "total_electrons",
                          "halogen_atoms", "heavy_atom_fraction"}
    composition_mask = np.array([
        not (name in composition_tokens or name.startswith("n_") or
             any(token in name for token in ["total_atoms", "heavy_atoms", "total_electrons"]))
        for name in names
    ])
    no_shape_mask = np.array([name not in shape_names for name in names])
    energy_beb_only = np.array([
        name in {"log_beb", "log_energy", "inv_sqrt_energy", "energy_missing",
                 "log_beb*log_energy", "log_beb^2", "log_energy^2"}
        for name in names
    ])

    predictions = {
        "raw_beb": original.prediction_raw_beb_A2.to_numpy(float),
        "constant_scale": original.prediction_constant_scale_A2.to_numpy(float),
        "full_routed_residual": original.prediction_expanded_ml_A2.to_numpy(float),
    }
    predictions["direct_log_prediction"] = direct_log_oof(frame, x, selected, 410000)
    predictions["residual_no_molecular_composition"] = residual_ablation_oof(
        frame, x, composition_mask, selected, 420000
    )
    predictions["residual_no_beb_shape"] = residual_ablation_oof(
        frame, x, no_shape_mask, selected, 430000
    )
    predictions["residual_energy_beb_only"] = residual_ablation_oof(
        frame, x, energy_beb_only, selected, 440000
    )
    predictions["pooled_no_task_routing"] = pooled_oof(frame, x, 450000)

    rows = []
    curves = frame.task_family.eq("curve").to_numpy(bool)
    for method, pred in predictions.items():
        all_m = metrics(frame, pred)
        curve_m = metrics(frame.loc[curves].reset_index(drop=True), pred[curves])
        rows.append({
            "method": method,
            "all_evidence_primary_macro_NMAE": all_m["primary_molecule_macro_NMAE"],
            "all_evidence_R2": all_m["R2"],
            "curve_primary_macro_NMAE": curve_m["primary_molecule_macro_NMAE"],
            "curve_R2": curve_m["R2"],
        })
    order = ["raw_beb", "constant_scale", "direct_log_prediction",
             "residual_energy_beb_only", "residual_no_molecular_composition",
             "residual_no_beb_shape", "pooled_no_task_routing", "full_routed_residual"]
    ablation = pd.DataFrame(rows).set_index("method").loc[order].reset_index()
    ablation.to_csv(OUT / "ablation_metrics.csv", index=False, encoding="utf-8-sig")

    ablation_bootstrap = []
    curve_frame = frame.loc[curves].reset_index(drop=True)
    curve_predictions = {name: values[curves] for name, values in predictions.items()}
    ablation_bootstrap.extend([
        paired_method_bootstrap(
            curve_frame, curve_predictions["direct_log_prediction"],
            curve_predictions["full_routed_residual"],
            "routed residual versus direct log prediction", "curve"
        ),
        paired_method_bootstrap(
            curve_frame, curve_predictions["full_routed_residual"],
            curve_predictions["residual_no_beb_shape"],
            "no-shape residual versus full routed residual", "curve"
        ),
        paired_method_bootstrap(
            curve_frame, curve_predictions["full_routed_residual"],
            curve_predictions["pooled_no_task_routing"],
            "pooled residual versus full routed residual", "curve"
        ),
        paired_method_bootstrap(
            frame, predictions["full_routed_residual"],
            predictions["pooled_no_task_routing"],
            "pooled residual versus full routed residual", "all evidence"
        ),
    ])
    ablation_bootstrap_frame = pd.DataFrame(ablation_bootstrap)
    ablation_bootstrap_frame.to_csv(
        OUT / "ablation_paired_bootstrap.csv", index=False, encoding="utf-8-sig"
    )

    curve_x, _ = final194.feature_matrix(curve_frame)
    batch = leave_one_batch_out(curve_frame, curve_x, 500000)
    batch.to_csv(OUT / "leave_one_source_batch_out.csv", index=False, encoding="utf-8-sig")
    membership = source_batch_membership(curve_frame)
    membership.to_csv(OUT / "source_batch_membership.csv", index=False, encoding="utf-8-sig")
    tested = membership[membership.included_in_batch_stress_test]
    excluded = membership[~membership.included_in_batch_stress_test]
    tested_molecules = int(tested.identity_key.nunique())
    weighted_raw = float(np.average(batch.raw_primary_macro_NMAE, weights=batch.molecules))
    weighted_ml = float(np.average(batch.ml_primary_macro_NMAE, weights=batch.molecules))
    source_summary = {
        "curve_molecules_total": int(curve_frame.identity_key.nunique()),
        "curve_molecules_in_stress_test": tested_molecules,
        "curve_molecules_excluded_from_stress_test": int(excluded.identity_key.nunique()),
        "exclusion_rule": "Source-ingestion batches require at least three curve-bearing identities.",
        "excluded_identities": excluded[[
            "identity_key", "molecule_name", "formula", "held_out_batch", "source_locator"
        ]].to_dict(orient="records"),
        "weighted_raw_primary_macro_NMAE": weighted_raw,
        "weighted_ml_primary_macro_NMAE": weighted_ml,
        "weighted_improvement_percentage_points": 100 * (weighted_raw - weighted_ml),
    }

    # Sensitivity analysis changes one numerical convention at a time while
    # retaining the frozen folds, selected configurations and ensemble mixtures.
    main_curve_x = x[curves]
    sensitivity_settings = [
        ("default refit", {}),
        ("clip 0.5-2.0", {"clip_low": 0.5, "clip_high": 2.0}),
        ("equal region weights 1.0/1.0", {"primary_weight": 1.0, "secondary_weight": 1.0}),
        ("moderate region weights 1.5/1.0", {"primary_weight": 1.5, "secondary_weight": 1.0}),
        ("missing uncertainty 8%", {"missing_uncertainty": 0.08}),
        ("missing uncertainty 20%", {"missing_uncertainty": 0.20}),
    ]
    sensitivity_rows = []
    for label, policy in sensitivity_settings:
        pred = sensitivity_oof(curve_frame, main_curve_x, selected, **policy)
        value = metrics(curve_frame, pred)["primary_molecule_macro_NMAE"]
        sensitivity_rows.append({
            "setting": label,
            "curve_primary_macro_NMAE": value,
            "change_vs_frozen_headline_percentage_points": 100 * (
                value - boot["ml_primary_molecule_macro_NMAE"]
            ) if "boot" in locals() else np.nan,
            "policy": json.dumps(policy, sort_keys=True),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    curve_old = predictions["raw_beb"][curves]
    curve_new = predictions["full_routed_residual"][curves]
    boot = curve_bootstrap(curve_frame, curve_old, curve_new, repeats=20000)
    sensitivity["change_vs_frozen_headline_percentage_points"] = 100 * (
        sensitivity.curve_primary_macro_NMAE - boot["ml_primary_molecule_macro_NMAE"]
    )
    sensitivity.to_csv(OUT / "numerical_sensitivity.csv", index=False, encoding="utf-8-sig")
    feature, grouped = importance_tables(feature_names)
    feature.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(OUT / "feature_group_importance.csv", index=False, encoding="utf-8-sig")

    fold = (frame.groupby(["cv_fold", "task_family"])
            .agg(rows=("identity_key", "size"), molecules=("identity_key", "nunique"))
            .reset_index())
    fold.to_csv(OUT / "fold_composition.csv", index=False, encoding="utf-8-sig")
    configurations = []
    for task, item in summary["final_selection"].items():
        configurations.append({
            "task": task, "molecules": item["molecules"], "rows": item["rows"],
            "config": json.dumps(item["config"], sort_keys=True),
            "weights": json.dumps(item["weights"], sort_keys=True),
            "inner_primary_macro_NMAE": item["inner_primary_macro_NMAE"],
        })
    pd.DataFrame(configurations).to_csv(OUT / "final_model_configuration.csv",
                                        index=False, encoding="utf-8-sig")

    plot_ablation(ablation, int(curve_frame.identity_key.nunique()), int(frame.identity_key.nunique()))
    plot_batches(batch)
    plot_importance(feature, grouped)
    elapsed = time.perf_counter() - started
    result = {
        "curve_primary_result": boot,
        "ablation": ablation.to_dict(orient="records"),
        "ablation_paired_bootstrap": ablation_bootstrap_frame.to_dict(orient="records"),
        "leave_one_source_batch_out": batch.to_dict(orient="records"),
        "source_batch_summary": source_summary,
        "numerical_sensitivity": sensitivity.to_dict(orient="records"),
        "feature_group_importance": grouped.to_dict(orient="records"),
        "runtime": {
            "elapsed_seconds": elapsed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "random_seed_ranges": "410000-500000; bootstrap seed 20260802",
        },
        "interpretation_note": (
            "The three evidence heads are trained independently. Fixed/sparse and peak identities "
            f"do not add training rows to the {curve_frame.identity_key.nunique()}-molecule curve head. Leave-one-source-batch-out "
            f"tests use curve data only and cover {tested_molecules} of {curve_frame.identity_key.nunique()} curve-bearing identities because the "
            "standalone 2-butanol source contains fewer than three identities. The seven groups are "
            "mutually exclusive source-ingestion batches rather than seven independent papers. The "
            "Hudson ester holdout jointly changes source and family, so those two effects cannot be separated."
        ),
    }
    (OUT / "validity_ablation_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
