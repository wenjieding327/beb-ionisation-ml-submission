from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("BEB_WORKSPACE_ROOT", Path(__file__).absolute().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

sys.path.insert(0, str(ROOT / "tools"))
from train_final_beb_residual_model import (
    balanced_folds,
    expanded_features,
    fit_ridge,
    macro_nmae,
    ridge_predict,
    weighted_mean,
)


INPUT = ROOT / "outputs" / "final_ml_training_2026_08_02" / "training_points_and_oof_predictions.csv"
OUT = ROOT / "outputs" / "improved_multitask_ml_2026_08_02"
OUT.mkdir(parents=True, exist_ok=True)

TREE_CONFIGS = [
    {"kind": "extra", "min_samples_leaf": 2, "max_features": 0.65, "max_depth": None},
    {"kind": "extra", "min_samples_leaf": 5, "max_features": 1.0, "max_depth": 16},
    {"kind": "hist", "min_samples_leaf": 20, "max_leaf_nodes": 15, "l2_regularization": 4.0},
]


def family(label):
    return "peak_amplitude" if "peak" in str(label).lower() else "pointwise_curve"


def family_weights(frame):
    counts = frame.groupby("identity_key").size()
    base = frame.identity_key.map(lambda x: 1.0 / counts[x]).to_numpy(float)
    exp = frame.experimental_sigma_A2.to_numpy(float)
    unc = pd.to_numeric(frame.experimental_uncertainty_A2, errors="coerce").to_numpy(float)
    relative = np.divide(unc, exp, out=np.full_like(exp, 0.12), where=np.isfinite(unc))
    quality = 1.0 / (1.0 + (relative / 0.12) ** 2)
    priority = np.where(frame.primary_low_energy.to_numpy(bool), 2.0, 0.75)
    return base * quality * priority


def make_tree(config, seed, n_estimators=160):
    if config["kind"] == "extra":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            max_depth=config["max_depth"],
            n_jobs=-1,
            random_state=seed,
        )
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=140,
        min_samples_leaf=config["min_samples_leaf"],
        max_leaf_nodes=config["max_leaf_nodes"],
        l2_regularization=config["l2_regularization"],
        early_stopping=False,
        random_state=seed,
    )


def score_predictions(frame, pred):
    exp = frame.experimental_sigma_A2.to_numpy(float)
    groups = frame.identity_key.to_numpy(str)
    primary = frame.primary_low_energy.to_numpy(bool)
    return macro_nmae(exp, pred, groups, primary)


def inner_oof(frame, x, config, seed):
    groups = frame.identity_key.to_numpy(str)
    y = frame.target_log_ratio.to_numpy(float)
    beb = frame.beb_sigma_A2.to_numpy(float)
    weights = family_weights(frame)
    folds, _ = balanced_folds(groups, min(3, len(set(groups))))
    tree_resid = np.zeros(len(frame))
    ridge_resid = np.zeros(len(frame))
    global_resid = np.zeros(len(frame))
    for fold in sorted(set(folds)):
        tr, va = folds != fold, folds == fold
        tree = make_tree(config, seed + int(fold), n_estimators=70)
        tree.fit(x[tr], y[tr], sample_weight=weights[tr])
        tree_resid[va] = tree.predict(x[va])
        ridge = fit_ridge(x[tr], y[tr], weights[tr], alpha=1.0)
        ridge_resid[va] = ridge_predict(ridge, x[va])
        global_resid[va] = weighted_mean(y[tr], weights[tr])
    components = [tree_resid, ridge_resid, global_resid]
    best = None
    grid = np.arange(0, 1.001, 0.25)
    for wt in grid:
        for wr in grid:
            wg = 1.0 - wt - wr
            if wg < -1e-9:
                continue
            residual = wt * components[0] + wr * components[1] + wg * components[2]
            residual = np.clip(residual, math.log(0.25), math.log(4.0))
            pred = beb * np.exp(residual)
            score = score_predictions(frame, pred)
            item = (score, float(wt), float(wr), float(max(0, wg)))
            if best is None or item < best:
                best = item
    return best, tree_resid


def select_config(frame, x, seed):
    results = []
    for idx, config in enumerate(TREE_CONFIGS):
        best, _ = inner_oof(frame, x, config, seed + idx * 100)
        results.append((best[0], config, best[1:]))
    return min(results, key=lambda item: item[0]), results


def fit_predict(frame_train, x_train, x_test, config, weights_tuple, seed, final=False):
    y = frame_train.target_log_ratio.to_numpy(float)
    weights = family_weights(frame_train)
    tree = make_tree(config, seed, n_estimators=400 if final else 180)
    tree.fit(x_train, y, sample_weight=weights)
    tree_resid = tree.predict(x_test)
    ridge = fit_ridge(x_train, y, weights, alpha=1.0)
    ridge_resid = ridge_predict(ridge, x_test)
    global_resid = weighted_mean(y, weights)
    wt, wr, wg = weights_tuple
    residual = wt * tree_resid + wr * ridge_resid + wg * global_resid
    residual = np.clip(residual, math.log(0.25), math.log(4.0))
    return residual, {"tree": tree, "ridge": ridge, "global_residual": global_resid,
                      "ensemble_weights": {"tree": wt, "ridge": wr, "global": wg}, "config": config}


def metrics(frame, prediction):
    exp = frame.experimental_sigma_A2.to_numpy(float)
    groups = frame.identity_key.to_numpy(str)
    primary = frame.primary_low_energy.to_numpy(bool)
    ae = np.abs(prediction - exp)
    return {
        "primary_molecule_macro_NMAE": macro_nmae(exp, prediction, groups, primary),
        "all_molecule_macro_NMAE": macro_nmae(exp, prediction, groups),
        "primary_point_weighted_NMAE": float(ae[primary].sum() / exp[primary].sum()),
        "all_point_weighted_NMAE": float(ae.sum() / exp.sum()),
        "median_absolute_percentage_error": float(np.median(ae / exp)),
        "RMSE_A2": float(np.sqrt(np.mean((prediction-exp)**2))),
        "R2": float(1 - np.sum((prediction-exp)**2) / np.sum((exp-exp.mean())**2)),
    }


def bootstrap(frame, old_pred, new_pred, repeats=10000):
    identities = sorted(frame.identity_key.unique())
    paired = []
    for identity in identities:
        g = frame[frame.identity_key == identity]
        idx = g.index.to_numpy(int)
        primary = g.primary_low_energy.to_numpy(bool)
        if not primary.any():
            primary[:] = True
        idx = idx[primary]
        exp = frame.loc[idx, "experimental_sigma_A2"].to_numpy(float)
        old = old_pred[idx]
        new = new_pred[idx]
        paired.append((np.abs(old-exp).sum()/exp.sum(), np.abs(new-exp).sum()/exp.sum()))
    paired = np.asarray(paired)
    rng = np.random.default_rng(20260802)
    draw = rng.integers(0, len(paired), (repeats, len(paired)))
    diff = (paired[:, 0]-paired[:, 1])[draw].mean(axis=1) * 100
    return {"improvement_percentage_points_95_CI": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))],
            "probability_improved": float(np.mean(diff > 0))}


def save_plots(frame, old_metrics, new_metrics):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Raw BEB", "Previous ridge", "Improved multitask"]
    values = [old_metrics["raw"]["primary_molecule_macro_NMAE"]*100,
              old_metrics["ridge"]["primary_molecule_macro_NMAE"]*100,
              new_metrics["primary_molecule_macro_NMAE"]*100]
    bars = ax.bar(labels, values, color=["#2a788e", "#7e57c2", "#e76f51"], width=.62)
    ax.bar_label(bars, fmt="%.2f%%", padding=5, fontsize=12, fontweight="bold")
    ax.set_ylabel("Primary molecule-macro NMAE (%)")
    ax.set_title("Strict molecule-level cross-validation", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(values)*1.25)
    fig.tight_layout(); fig.savefig(OUT / "01_improved_model_comparison.png", dpi=180); plt.close(fig)

    exp = frame.experimental_sigma_A2.to_numpy(float)
    old = frame.prediction_ml_A2.to_numpy(float)
    new = frame.prediction_improved_multitask_A2.to_numpy(float)
    lim = np.quantile(exp, .995)*1.08
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, pred, title, color in [(axes[0], old, "Previous ridge residual", "#7e57c2"),
                                   (axes[1], new, "Improved nonlinear multitask", "#e76f51")]:
        ax.scatter(exp, pred, s=9, alpha=.62, color=color, edgecolor="none")
        ax.plot([0, lim], [0, lim], "k--", lw=1.5)
        ax.set(xlim=(0, lim), ylim=(0, lim), title=title, xlabel="Experiment (A²)")
    axes[0].set_ylabel("Held-out prediction (A²)")
    fig.suptitle("Out-of-fold predictions on unseen molecules", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "02_improved_out_of_fold_parity.png", dpi=180); plt.close(fig)

    rows = []
    for identity, g in frame.groupby("identity_key"):
        idx = g.index.to_numpy(int)
        m = g.primary_low_energy.to_numpy(bool)
        if not m.any(): m[:] = True
        idx = idx[m]
        expg = frame.loc[idx, "experimental_sigma_A2"].to_numpy(float)
        prev = np.abs(frame.loc[idx, "prediction_ml_A2"].to_numpy(float)-expg).sum()/expg.sum()
        newg = np.abs(frame.loc[idx, "prediction_improved_multitask_A2"].to_numpy(float)-expg).sum()/expg.sum()
        rows.append((identity, g.molecule_name.iloc[0], g.formula.iloc[0], (prev-newg)*100))
    rank = pd.DataFrame(rows, columns=["identity_key", "molecule_name", "formula", "improvement_pp"]).sort_values("improvement_pp")
    show = pd.concat([rank.head(12), rank.tail(12)])
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = np.where(show.improvement_pp >= 0, "#2a9d8f", "#d1495b")
    labels = [f"{n} ({f})" for n, f in zip(show.molecule_name, show.formula)]
    ax.barh(labels, show.improvement_pp, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Improvement vs previous ridge (percentage points)")
    ax.set_title("Which molecules changed most", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "03_change_vs_previous_model.png", dpi=180); plt.close(fig)
    return rank


def main():
    frame = pd.read_csv(INPUT)
    frame["task_family"] = frame.label_type.map(family)
    x, feature_names = expanded_features(frame)
    groups = frame.identity_key.to_numpy(str)
    outer_folds = frame.cv_fold.to_numpy(int)
    oof_residual = np.zeros(len(frame))
    audit = []

    for fold in sorted(set(outer_folds)):
        for task in ["pointwise_curve", "peak_amplitude"]:
            tr = (outer_folds != fold) & (frame.task_family.to_numpy(str) == task)
            va = (outer_folds == fold) & (frame.task_family.to_numpy(str) == task)
            if not va.any():
                continue
            selected, candidates = select_config(frame.loc[tr].reset_index(drop=True), x[tr], 10000+fold*1000)
            inner_score, config, ensemble_weights = selected
            residual, _ = fit_predict(frame.loc[tr].reset_index(drop=True), x[tr], x[va], config, ensemble_weights,
                                      seed=20000+fold*100+(0 if task == "pointwise_curve" else 1))
            oof_residual[va] = residual
            audit.append({"fold": int(fold), "task_family": task, "train_molecules": int(len(set(groups[tr]))),
                          "test_molecules": int(len(set(groups[va]))), "selected_tree": config,
                          "ensemble_weights": {"tree": ensemble_weights[0], "ridge": ensemble_weights[1], "global": ensemble_weights[2]},
                          "inner_primary_macro_NMAE": inner_score,
                          "candidate_tree_scores": [{"config": c, "best_ensemble_score": s} for s, c, _ in candidates]})
            print(f"fold={fold} task={task} inner={inner_score:.4f} config={config} weights={ensemble_weights}", flush=True)

    frame["prediction_improved_multitask_A2"] = frame.beb_sigma_A2.to_numpy(float) * np.exp(oof_residual)
    old_metrics = {"raw": metrics(frame, frame.beb_sigma_A2.to_numpy(float)),
                   "ridge": metrics(frame, frame.prediction_ml_A2.to_numpy(float))}
    new_metrics = metrics(frame, frame.prediction_improved_multitask_A2.to_numpy(float))
    family_metrics = {}
    for task, g in frame.groupby("task_family"):
        family_metrics[task] = {
            "molecules": int(g.identity_key.nunique()), "points": int(len(g)),
            "raw": metrics(g, g.beb_sigma_A2.to_numpy(float)),
            "previous_ridge": metrics(g, g.prediction_ml_A2.to_numpy(float)),
            "improved": metrics(g, g.prediction_improved_multitask_A2.to_numpy(float)),
        }
    boot = bootstrap(frame, frame.prediction_ml_A2.to_numpy(float), frame.prediction_improved_multitask_A2.to_numpy(float))
    boot_raw = bootstrap(frame, frame.beb_sigma_A2.to_numpy(float), frame.prediction_improved_multitask_A2.to_numpy(float))

    final_models = {}
    final_selection = {}
    for task in ["pointwise_curve", "peak_amplitude"]:
        m = frame.task_family.to_numpy(str) == task
        selected, candidates = select_config(frame.loc[m].reset_index(drop=True), x[m], 50000)
        inner_score, config, ensemble_weights = selected
        _, fitted = fit_predict(frame.loc[m].reset_index(drop=True), x[m], x[m], config, ensemble_weights, 60000, final=True)
        final_models[task] = fitted
        final_selection[task] = {"inner_score": inner_score, "config": config,
                                 "ensemble_weights": {"tree": ensemble_weights[0], "ridge": ensemble_weights[1], "global": ensemble_weights[2]},
                                 "molecules": int(frame.loc[m, "identity_key"].nunique()), "points": int(m.sum())}

    rank = save_plots(frame, old_metrics, new_metrics)
    frame.to_csv(OUT / "improved_oof_predictions.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT / "per_molecule_change_vs_previous.csv", index=False, encoding="utf-8-sig")
    package = {"models": final_models, "feature_names": feature_names,
               "task_rule": "label_type containing 'peak' -> peak_amplitude; otherwise pointwise_curve",
               "prediction": "sigma_corrected = sigma_BEB * exp(clipped ensemble residual)"}
    joblib.dump(package, OUT / "improved_multitask_model.joblib", compress=3)

    improvement = 100*(old_metrics["ridge"]["primary_molecule_macro_NMAE"]-new_metrics["primary_molecule_macro_NMAE"])
    summary = {
        "unique_molecules": int(frame.identity_key.nunique()), "points": int(len(frame)),
        "validation": "Nested selection inside five identity-disjoint outer folds",
        "old_metrics": old_metrics, "improved_metrics": new_metrics, "family_metrics": family_metrics,
        "absolute_primary_improvement_vs_previous_percentage_points": improvement,
        "relative_primary_improvement_vs_previous_percent": 100*(1-new_metrics["primary_molecule_macro_NMAE"]/old_metrics["ridge"]["primary_molecule_macro_NMAE"]),
        "bootstrap_vs_previous": boot, "bootstrap_vs_raw_beb": boot_raw,
        "outer_fold_audit": audit, "final_selection": final_selection,
        "scientific_change": "Separate pointwise-curve and peak-amplitude heads; nonlinear tree/ridge/global ensembles selected without using outer-fold molecules.",
    }
    (OUT / "improved_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Improved multitask BEB correction\n\n"
        f"- Molecules: **{summary['unique_molecules']}**\n"
        f"- Numeric points: **{summary['points']}**\n"
        "- Validation: nested model selection with five molecule-disjoint outer folds\n"
        f"- Previous primary macro-NMAE: **{old_metrics['ridge']['primary_molecule_macro_NMAE']*100:.2f}%**\n"
        f"- Improved primary macro-NMAE: **{new_metrics['primary_molecule_macro_NMAE']*100:.2f}%**\n"
        f"- Additional relative reduction: **{summary['relative_primary_improvement_vs_previous_percent']:.2f}%**\n"
        f"- Bootstrap improvement 95% CI: **{boot['improvement_percentage_points_95_CI'][0]:.2f} to {boot['improvement_percentage_points_95_CI'][1]:.2f} percentage points**\n\n"
        f"Against raw BEB, the total relative primary error reduction is **{100*(1-new_metrics['primary_molecule_macro_NMAE']/old_metrics['raw']['primary_molecule_macro_NMAE']):.2f}%**, with a paired bootstrap 95% interval of **{boot_raw['improvement_percentage_points_95_CI'][0]:.2f} to {boot_raw['improvement_percentage_points_95_CI'][1]:.2f} percentage points**.\n\n"
        "The improvement separates peak-amplitude supervision from pointwise curve supervision and uses a nonlinear ExtraTrees/gradient-boosting candidate pool blended with ridge/global residuals. The final choice is based only on inner folds; reported scores come from untouched outer molecules.\n",
        encoding="utf-8")
    print(json.dumps({"previous": old_metrics["ridge"]["primary_molecule_macro_NMAE"],
                      "improved": new_metrics["primary_molecule_macro_NMAE"],
                      "relative_change_percent": summary["relative_primary_improvement_vs_previous_percent"],
                      "bootstrap_vs_previous": boot, "bootstrap_vs_raw": boot_raw}, indent=2))


if __name__ == "__main__":
    main()
