from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("BEB_WORKSPACE_ROOT", Path(__file__).absolute().parents[1]))
OUT = ROOT / "outputs" / "final_ml_training_2026_08_02"
OUT.mkdir(parents=True, exist_ok=True)

ELEMENT_Z = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Mg": 12,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ge": 32, "Br": 35,
    "I": 53, "W": 74,
}
CORE_ELEMENTS = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Ge", "Br", "I", "W"]


def fnum(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def slug(value: str) -> str:
    value = value.strip().lower().replace("n-", "n-")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "unknown"


def add(rows, *, identity, name, formula, energy, exp, beb, uncertainty=None,
        label_type, source, primary=True, evidence="numeric_ready"):
    energy, exp, beb = fnum(energy), fnum(exp), fnum(beb)
    uncertainty = fnum(uncertainty)
    if energy is None or exp is None or beb is None:
        return
    if energy <= 0 or exp <= 0.02 or beb <= 0.02:
        return
    ratio = exp / beb
    if ratio < 0.04 or ratio > 25:
        return
    rows.append({
        "identity_key": slug(identity),
        "molecule_name": name.strip(),
        "formula": formula.strip(),
        "energy_eV": energy,
        "experimental_sigma_A2": exp,
        "beb_sigma_A2": beb,
        "experimental_uncertainty_A2": uncertainty,
        "label_type": label_type,
        "primary_low_energy": bool(primary),
        "evidence": evidence,
        "source": source,
    })


def read_curve(path: Path):
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            names = [str(c).lower() for c in df.columns]
            eidx = next((i for i, c in enumerate(names) if "energy" in c), 0)
            sidx = next((i for i, c in enumerate(names) if any(k in c for k in ["cross", "sigma", "tics", "beb"])), 1)
            e = pd.to_numeric(df.iloc[:, eidx], errors="coerce").to_numpy(float)
            s = pd.to_numeric(df.iloc[:, sidx], errors="coerce").to_numpy(float)
        else:
            raise ValueError
    except Exception:
        raw = np.genfromtxt(path, delimiter=",")
        if raw.ndim != 2 or raw.shape[1] < 2:
            raw = np.genfromtxt(path)
        e, s = raw[:, 0], raw[:, 1]
    mask = np.isfinite(e) & np.isfinite(s)
    e, s = e[mask], s[mask]
    order = np.argsort(e)
    return e[order], s[order]


def interp_curve(path_value, energy):
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        e, s = read_curve(path)
        if len(e) < 2 or energy < e.min() or energy > e.max():
            return None
        return float(np.interp(energy, e, s))
    except Exception:
        return None


def curve_peak(path_value):
    if not path_value:
        return None, None
    path = Path(path_value)
    if not path.exists():
        return None, None
    try:
        e, s = read_curve(path)
        if len(e) < 2:
            return None, None
        idx = int(np.nanargmax(s))
        return float(e[idx]), float(s[idx])
    except Exception:
        return None, None


def load_nist_strict(rows):
    base = ROOT / "nist_strict_usable_numeric_pairs_2026_07_30"
    audit = pd.read_csv(base / "strict_usable_pair_audit.csv")
    for _, a in audit.iterrows():
        path = Path(str(a["paired_curve_file"]))
        if not path.is_absolute():
            path = base / path
        if not path.exists():
            continue
        peak_e = fnum(a.get("experimental_peak_energy_eV"))
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            unc_vals = [fnum(r.get("experimental_error_minus_A2")), fnum(r.get("experimental_error_plus_A2"))]
            unc = max([x for x in unc_vals if x is not None], default=None)
            e = fnum(r.get("energy_eV"))
            add(rows, identity=a["molecule_name"], name=a["molecule_name"], formula=a["formula"],
                energy=e, exp=r.get("experimental_total_cross_section_A2"), beb=r.get("beb_interpolated_A2"),
                uncertainty=unc, label_type="curve_point", source=str(path),
                primary=(peak_e is None or (e is not None and e <= peak_e)), evidence="NIST_strict_numeric")


def load_esters(rows):
    base = ROOT / "outputs" / "no_qec_priority_2026_07_31" / "ready15_esters_experiment_beb"
    manifest = pd.read_csv(base / "pair_manifest.csv")
    for _, a in manifest.iterrows():
        path = base / str(a["paired_curve_file"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            add(rows, identity=a["identity_key"], name=a["molecule_name"], formula=a["formula"],
                energy=r.get("energy_eV"), exp=r.get("experimental_total_cross_section_A2"),
                beb=r.get("beb_cross_section_A2"), uncertainty=r.get("experimental_uncertainty_A2"),
                label_type="curve_point", source=str(path),
                primary=str(r.get("primary_onset_to_peak_mask", "true")).lower() == "true",
                evidence=str(a.get("evidence_tier", "digitized_curve")))


def load_alcohol_curves(rows):
    base = ROOT / "outputs" / "no_qec_priority_2026_07_31"
    for path in sorted(base.glob("agent_*_strict_numeric_pairs.csv")):
        df = pd.read_csv(path)
        needed = {"energy_eV", "experiment_tics_1e16_cm2", "published_beb_tics_1e16_cm2"}
        if not needed.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            pct = fnum(r.get("experiment_uncertainty_percent"))
            exp = fnum(r.get("experiment_tics_1e16_cm2"))
            unc = exp * pct / 100 if exp is not None and pct is not None else None
            add(rows, identity=r["identity_key"], name=r["molecule_name"], formula=r["formula"],
                energy=r["energy_eV"], exp=exp, beb=r["published_beb_tics_1e16_cm2"],
                uncertainty=unc, label_type="curve_point", source=str(path),
                primary=str(r.get("eligible_primary_low_energy_loss", "true")).lower() == "true",
                evidence="published_numeric_curve")


def load_current_digitized(rows):
    base = ROOT / "outputs" / "current_experiment_beb_master_2026_08_01"
    specs = [
        ("aip_fig9_12_13_experiment_ecp_beb_READY_TIER_B.csv", "energy_eV_digitized", "experiment_sigma_A2_digitized", "beb_sigma_A2_digitized"),
        ("siclx_experiment_raw_beb_READY_TIER_B.csv", "energy_eV_digitized", "experiment_sigma_A2_digitized", "beb_sigma_A2_digitized"),
        ("sf3_sf5_experiment_beb_READY_TIER_B.csv", "energy_eV_digitized", "experiment_sigma_A2_digitized", "beb_sigma_A2_digitized"),
        ("cf2_cf3_nf2_nf3_experiment_beb_READY_TIER_B.csv", "energy_eV_digitized", "experiment_sigma_A2_digitized", "beb_sigma_A2_digitized"),
    ]
    for filename, ecol, xcol, bcol in specs:
        path = base / filename
        df = pd.read_csv(path)
        peaks = df.groupby("identity_key")[xcol].idxmax()
        peak_energy = {df.loc[i, "identity_key"]: fnum(df.loc[i, ecol]) for i in peaks}
        for _, r in df.iterrows():
            unc = max([x for x in [fnum(r.get("experiment_uncertainty_minus_A2_digitized")), fnum(r.get("experiment_uncertainty_plus_A2_digitized"))] if x is not None], default=None)
            e = fnum(r.get(ecol))
            add(rows, identity=r["identity_key"], name=r["molecule_name"], formula=r["formula"],
                energy=e, exp=r[xcol], beb=r[bcol], uncertainty=unc,
                label_type="curve_point" if len(df[df.identity_key == r["identity_key"]]) > 2 else "sparse_point",
                source=str(path), primary=e <= peak_energy.get(r["identity_key"], e), evidence="digitized_Tier_B")
    path = base / "geh4_single_point_READY_TIER_B.csv"
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        add(rows, identity=r["identity_key"], name=r["molecule_name"], formula=r["formula"],
            energy=r["energy_eV"], exp=r["experiment_sigma_A2"], beb=r["beb_sigma_A2"],
            uncertainty=r.get("experiment_uncertainty_A2"), label_type="sparse_point",
            source=str(path), primary=True, evidence="digitized_Tier_B")


def load_fixed_energy(rows):
    base = ROOT / "outputs" / "no_qec_priority_2026_07_31"
    ready = pd.read_csv(base / "ready100_fixed_energy_with_existing_beb_no_qec.csv")
    for _, r in ready.iterrows():
        p = r.get("local_beb_source_path", "")
        vals = [interp_curve(p, 70.0), interp_curve(p, 75.0)]
        vals = [x for x in vals if x is not None]
        if not vals:
            continue
        add(rows, identity=r["local_beb_identity_key"], name=r["molecule_name"], formula=r["formula"],
            energy=72.5, exp=r["mean_cross_section_A2"], beb=float(np.mean(vals)),
            uncertainty=r.get("confidence_95_A2"), label_type="fixed_70_75eV",
            source=str(base / "ready100_fixed_energy_with_existing_beb_no_qec.csv"),
            primary=True, evidence="published_fixed_energy")

    for rel in [
        "outputs/alecs_bose396_match_2026_08_01/approved_same_identity_11/approved_11_manifest.csv",
        "outputs/alecs_bose396_match_2026_08_01/approved_ambiguous_identity_16/approved_16_manifest.csv",
    ]:
        path = ROOT / rel
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            add(rows, identity=r["molecule_name"], name=r["molecule_name"], formula=r["formula"],
                energy=72.5, exp=r["experimental_cross_section_A2"], beb=r["beb_interval_endpoint_mean_A2"],
                uncertainty=r.get("experimental_uncertainty_95_A2"), label_type="fixed_70_75eV",
                source=str(path), primary=True, evidence="published_fixed_energy_ALeCS")

    ledger = base / "new25_incremental_identity_ledger_2026_08_01.csv"
    df = pd.read_csv(ledger)
    for _, r in df.iterrows():
        energy = fnum(r.get("experimental_energy_eV")) or fnum(r.get("beb_energy_eV")) or 72.5
        add(rows, identity=r["identity_key"], name=r["molecule_name"], formula=r["formula"],
            energy=energy, exp=r["experimental_value_A2"], beb=r["beb_value_A2"],
            uncertainty=r.get("experimental_uncertainty_A2"),
            label_type=str(r.get("experimental_label_type", "sparse_point")), source=str(ledger),
            primary=True, evidence="strict_incremental_numeric")


def load_peak_labels(rows):
    path = ROOT / "outputs" / "no_qec_priority_2026_07_31" / "agent_no_qec_immediate_numeric_peak_labels_50.csv"
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        exp = fnum(r.get("experimental_cross_section_A2"))
        e = fnum(r.get("experimental_energy_eV"))
        peak_e, peak_b = curve_peak(r.get("accepted_BEB_path", ""))
        if e is None:
            e = peak_e
        beb = interp_curve(r.get("accepted_BEB_path", ""), e) if e is not None else None
        if beb is None:
            beb = peak_b
        add(rows, identity=r["identity_key"], name=r["molecule_name"], formula=r["formula"],
            energy=e, exp=exp, beb=beb, uncertainty=r.get("experimental_uncertainty_A2"),
            label_type="peak_height", source=str(path), primary=True, evidence="published_numeric_peak")


def build_dataset():
    rows = []
    load_nist_strict(rows)
    load_esters(rows)
    load_alcohol_curves(rows)
    load_current_digitized(rows)
    load_fixed_energy(rows)
    load_peak_labels(rows)
    df = pd.DataFrame(rows)
    # Exact duplicate numeric observations from cumulative ledgers/manifests are collapsed.
    df["dedup_key"] = (
        df["identity_key"] + "|" + df["label_type"] + "|" +
        df["energy_eV"].round(3).astype(str) + "|" + df["experimental_sigma_A2"].round(4).astype(str)
    )
    df = df.drop_duplicates("dedup_key", keep="first").drop(columns="dedup_key")
    # Keep the scientifically relevant range; tails above 300 eV are secondary and sparse.
    df = df[(df.energy_eV <= 300) & (df.experimental_sigma_A2 > 0.02) & (df.beb_sigma_A2 > 0.02)].copy()
    counts = df.groupby("identity_key").size()
    df["molecule_weight"] = df.identity_key.map(lambda x: 1.0 / counts[x])
    unc_rel = df.experimental_uncertainty_A2 / df.experimental_sigma_A2
    quality = 1.0 / (1.0 + (unc_rel.fillna(0.12) / 0.12) ** 2)
    df["sample_weight"] = df.molecule_weight * quality * np.where(df.primary_low_energy, 2.0, 0.75)
    df["target_log_ratio"] = np.log(df.experimental_sigma_A2 / df.beb_sigma_A2)
    df = df.sort_values(["identity_key", "energy_eV", "source"]).reset_index(drop=True)
    return df


def formula_counts(formula):
    counts = Counter()
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", str(formula)):
        counts[element] += float(number) if number else 1.0
    return counts


def base_features(df):
    matrix = []
    names = ["log_beb", "log_energy", "inv_sqrt_energy", "total_atoms", "heavy_atoms", "total_electrons"]
    names += [f"n_{e}" for e in CORE_ELEMENTS]
    for _, r in df.iterrows():
        c = formula_counts(r.formula)
        total = sum(c.values()) or 1.0
        heavy = total - c.get("H", 0.0)
        electrons = sum(ELEMENT_Z.get(k, 0) * v for k, v in c.items())
        row = [math.log(r.beb_sigma_A2), math.log(r.energy_eV), 1 / math.sqrt(r.energy_eV), total, heavy, electrons]
        row += [c.get(e, 0.0) for e in CORE_ELEMENTS]
        matrix.append(row)
    return np.asarray(matrix, float), names


def expanded_features(df):
    x, names = base_features(df)
    # Add a compact set of physically interpretable interactions.
    extra, extra_names = [], []
    pairs = [(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4), (4, 5)]
    for i, j in pairs:
        extra.append(x[:, i] * x[:, j])
        extra_names.append(f"{names[i]}*{names[j]}")
    for i in [0, 1, 3, 4, 5]:
        extra.append(x[:, i] ** 2)
        extra_names.append(f"{names[i]}^2")
    return np.column_stack([x] + extra), names + extra_names


def weighted_mean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def fit_ridge(x, y, w, alpha):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-10] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    root_w = np.sqrt(w)[:, None]
    a = design * root_w
    b = y * root_w[:, 0]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ b)
    return {"mean": mean, "scale": scale, "coef": coef, "alpha": alpha}


def ridge_predict(model, x):
    z = (x - model["mean"]) / model["scale"]
    return np.column_stack([np.ones(len(z)), z]) @ model["coef"]


def balanced_folds(groups, n_splits=5):
    sizes = Counter(groups)
    ordered = sorted(sizes, key=lambda g: (-sizes[g], hashlib.sha1(g.encode()).hexdigest()))
    loads = [0] * n_splits
    assignment = {}
    for group in ordered:
        fold = min(range(n_splits), key=lambda i: loads[i])
        assignment[group] = fold
        loads[fold] += sizes[group]
    return np.array([assignment[g] for g in groups]), assignment


def macro_nmae(exp, pred, groups, mask=None):
    if mask is None:
        mask = np.ones(len(exp), dtype=bool)
    values = []
    for g in sorted(set(groups[mask])):
        m = mask & (groups == g)
        denom = np.sum(exp[m])
        if denom > 0:
            values.append(np.sum(np.abs(pred[m] - exp[m])) / denom)
    return float(np.mean(values)) if values else float("nan")


def choose_hyperparameters(x, y, exp, beb, groups, weights):
    folds, _ = balanced_folds(groups, min(4, len(set(groups))))
    candidates = [(a, b) for a in [0.1, 1.0, 10.0, 100.0, 1000.0] for b in [0.25, 0.5, 0.75, 1.0]]
    scores = []
    for alpha, blend in candidates:
        pred = np.zeros(len(y))
        for fold in sorted(set(folds)):
            tr, va = folds != fold, folds == fold
            model = fit_ridge(x[tr], y[tr], weights[tr], alpha)
            global_resid = weighted_mean(y[tr], weights[tr])
            residual = blend * ridge_predict(model, x[va]) + (1 - blend) * global_resid
            residual = np.clip(residual, math.log(0.2), math.log(5.0))
            pred[va] = beb[va] * np.exp(residual)
        scores.append((macro_nmae(exp, pred, groups), alpha, blend))
    return min(scores)


def cross_validate(df, x):
    y = df.target_log_ratio.to_numpy(float)
    exp = df.experimental_sigma_A2.to_numpy(float)
    beb = df.beb_sigma_A2.to_numpy(float)
    groups = df.identity_key.to_numpy(str)
    weights = df.sample_weight.to_numpy(float)
    folds, assignment = balanced_folds(groups, 5)
    pred_ml = np.zeros(len(df))
    pred_const = np.zeros(len(df))
    choices = []
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        score, alpha, blend = choose_hyperparameters(x[tr], y[tr], exp[tr], beb[tr], groups[tr], weights[tr])
        model = fit_ridge(x[tr], y[tr], weights[tr], alpha)
        global_resid = weighted_mean(y[tr], weights[tr])
        residual = blend * ridge_predict(model, x[va]) + (1 - blend) * global_resid
        residual = np.clip(residual, math.log(0.2), math.log(5.0))
        pred_ml[va] = beb[va] * np.exp(residual)
        pred_const[va] = beb[va] * math.exp(global_resid)
        choices.append({"fold": fold, "alpha": alpha, "blend": blend, "inner_macro_nmae": score,
                        "train_molecules": int(len(set(groups[tr]))), "test_molecules": int(len(set(groups[va])))})
    return folds, pred_const, pred_ml, choices, assignment


def metrics_bundle(df, pred):
    exp = df.experimental_sigma_A2.to_numpy(float)
    groups = df.identity_key.to_numpy(str)
    primary = df.primary_low_energy.to_numpy(bool)
    ae = np.abs(pred - exp)
    ape = ae / exp
    return {
        "primary_macro_NMAE": macro_nmae(exp, pred, groups, primary),
        "all_macro_NMAE": macro_nmae(exp, pred, groups),
        "primary_point_weighted_NMAE": float(np.sum(ae[primary]) / np.sum(exp[primary])),
        "all_point_weighted_NMAE": float(np.sum(ae) / np.sum(exp)),
        "median_absolute_percentage_error": float(np.median(ape)),
        "RMSE_A2": float(np.sqrt(np.mean((pred - exp) ** 2))),
        "R2_pointwise": float(1 - np.sum((pred - exp) ** 2) / np.sum((exp - np.mean(exp)) ** 2)),
    }


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def canvas(title, subtitle="", size=(1400, 900)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    d.text((60, 35), title, fill="#14213d", font=font(34, True))
    if subtitle:
        d.text((60, 82), subtitle, fill="#4b5563", font=font(20))
    return im, d


def save_metric_chart(metrics, intervals):
    im, d = canvas("Molecule-level cross-validation: primary low-energy error",
                   "Lower is better; test molecules are never present in training folds")
    labels = ["Raw BEB", "Constant scale", "BEB + ML residual"]
    vals = [metrics[k]["primary_macro_NMAE"] * 100 for k in ["raw_beb", "constant_scale", "ml_residual"]]
    colors = ["#2a788e", "#7e57c2", "#e76f51"]
    x0, y0, w, h = 170, 160, 1050, 600
    d.line((x0, y0, x0, y0+h), fill="#111827", width=3)
    d.line((x0, y0+h, x0+w, y0+h), fill="#111827", width=3)
    vmax = max(vals) * 1.35
    for i, (lab, val, col) in enumerate(zip(labels, vals, colors)):
        bx = x0 + 120 + i * 320
        bh = h * val / vmax
        d.rounded_rectangle((bx, y0+h-bh, bx+180, y0+h), radius=12, fill=col)
        d.text((bx+15, y0+h-bh-45), f"{val:.1f}%", fill="#111827", font=font(28, True))
        key = ["raw_beb", "constant_scale", "ml_residual"][i]
        lo, hi = [100*x for x in intervals[key]]
        cx = bx + 90
        ylo, yhi = y0+h-h*lo/vmax, y0+h-h*hi/vmax
        d.line((cx, ylo, cx, yhi), fill="#111827", width=3)
        d.line((cx-15, ylo, cx+15, ylo), fill="#111827", width=3)
        d.line((cx-15, yhi, cx+15, yhi), fill="#111827", width=3)
        d.text((bx-5, y0+h+25), lab, fill="#111827", font=font(22))
    im.save(OUT / "01_primary_metric_comparison.png")


def save_parity(df):
    im, d = canvas("Out-of-fold parity: experiment vs prediction",
                   "Each point is predicted by a model that did not see that molecule", (1600, 800))
    exp = df.experimental_sigma_A2.to_numpy(float)
    vmax = float(np.quantile(exp, 0.995) * 1.1)
    for panel, (col, title, color) in enumerate([
        ("prediction_raw_beb_A2", "Raw BEB", "#2a788e"),
        ("prediction_ml_A2", "BEB + ML residual", "#e76f51")]):
        ox = 90 + panel * 780
        oy, w, h = 150, 650, 540
        d.rectangle((ox, oy, ox+w, oy+h), outline="#d1d5db", width=2)
        d.line((ox, oy+h, ox+w, oy), fill="#111827", width=3)
        pred = df[col].to_numpy(float)
        for a, b in zip(exp, pred):
            if a <= vmax and b <= vmax:
                px = ox + w * a / vmax
                py = oy + h - h * b / vmax
                d.ellipse((px-3, py-3, px+3, py+3), fill=color)
        d.text((ox+240, oy-45), title, fill="#111827", font=font(25, True))
        d.text((ox+240, oy+h+45), "Experiment (A²)", fill="#111827", font=font(20))
        d.text((ox+8, oy+8), f"0–{vmax:.1f} A²", fill="#6b7280", font=font(16))
    im.save(OUT / "02_out_of_fold_parity.png")


def molecule_errors(df):
    out = []
    for identity, g in df.groupby("identity_key"):
        exp = g.experimental_sigma_A2.to_numpy(float)
        raw = g.prediction_raw_beb_A2.to_numpy(float)
        ml = g.prediction_ml_A2.to_numpy(float)
        primary = g.primary_low_energy.to_numpy(bool)
        if not primary.any():
            primary[:] = True
        raw_n = np.sum(np.abs(raw[primary] - exp[primary])) / np.sum(exp[primary])
        ml_n = np.sum(np.abs(ml[primary] - exp[primary])) / np.sum(exp[primary])
        out.append({"identity_key": identity, "molecule_name": g.molecule_name.iloc[0], "formula": g.formula.iloc[0],
                    "points": len(g), "raw_primary_NMAE": raw_n, "ml_primary_NMAE": ml_n,
                    "improvement_percentage_points": 100 * (raw_n - ml_n)})
    return pd.DataFrame(out).sort_values("improvement_percentage_points", ascending=False)


def save_rank_chart(mol):
    ranked = pd.concat([mol.head(12), mol.tail(12)]).drop_duplicates("identity_key")
    im, d = canvas("Where ML helps—and where it does not",
                   "Change in primary low-energy NMAE; positive means improvement", (1500, 1100))
    xmid, y0, rowh = 850, 150, 36
    scale = 7.0
    d.line((xmid, y0-15, xmid, y0+rowh*len(ranked)), fill="#111827", width=2)
    for i, (_, r) in enumerate(ranked.iterrows()):
        y = y0 + i*rowh
        val = float(r.improvement_percentage_points)
        x1 = xmid + val*scale
        color = "#2a9d8f" if val >= 0 else "#d1495b"
        d.rectangle((min(xmid, x1), y, max(xmid, x1), y+22), fill=color)
        d.text((20, y-2), f"{r.molecule_name} ({r.formula})", fill="#111827", font=font(17))
        d.text((x1+8 if val >= 0 else x1-72, y-2), f"{val:+.1f}", fill=color, font=font(16, True))
    im.save(OUT / "03_per_molecule_improvement_extremes.png")


def save_error_hist(df):
    im, d = canvas("Absolute percentage error distribution",
                   "Out-of-fold predictions; values above 150% are shown in the last bin")
    bins = np.linspace(0, 150, 16)
    exp = df.experimental_sigma_A2.to_numpy(float)
    series = [("Raw BEB", df.prediction_raw_beb_A2.to_numpy(float), "#2a788e", 0),
              ("BEB + ML", df.prediction_ml_A2.to_numpy(float), "#e76f51", 1)]
    x0, y0, w, h = 120, 180, 1160, 560
    bw = w / len(bins)
    for label, pred, color, offset in series:
        err = np.minimum(np.abs(pred-exp)/exp*100, 149.9)
        hist, _ = np.histogram(err, bins=bins)
        hist = hist / hist.sum()
        for i, val in enumerate(hist):
            bh = val / max(0.01, max(hist.max(), 0.01)) * (h*0.85)
            left = x0 + i*bw + offset*bw*0.42
            d.rectangle((left, y0+h-bh, left+bw*0.4, y0+h), fill=color)
    d.line((x0, y0+h, x0+w, y0+h), fill="#111827", width=3)
    d.text((x0, y0+h+30), "0%", fill="#111827", font=font(18))
    d.text((x0+w-80, y0+h+30), "150%+", fill="#111827", font=font(18))
    d.rectangle((980, 105, 1010, 130), fill="#2a788e"); d.text((1020, 103), "Raw BEB", fill="#111827", font=font(18))
    d.rectangle((1160, 105, 1190, 130), fill="#e76f51"); d.text((1200, 103), "BEB + ML", fill="#111827", font=font(18))
    im.save(OUT / "04_error_distribution.png")


def save_coverage(df):
    im, d = canvas("Training supervision coverage",
                   "Unique molecule identities and numeric observations by label type")
    stats = []
    for label, g in df.groupby("label_type"):
        stats.append((label, g.identity_key.nunique(), len(g)))
    stats.sort(key=lambda x: -x[1])
    x0, y0, maxw = 470, 180, 720
    maxmol = max(x[1] for x in stats)
    for i, (label, mols, points) in enumerate(stats):
        y = y0 + i*85
        d.text((70, y+8), label, fill="#111827", font=font(21))
        bw = maxw*mols/maxmol
        d.rounded_rectangle((x0, y, x0+bw, y+45), radius=8, fill="#457b9d")
        d.text((x0+bw+15, y+8), f"{mols} molecules / {points} points", fill="#111827", font=font(20, True))
    im.save(OUT / "05_training_coverage.png")


def save_curve_gallery(df, mol):
    gallery = OUT / "representative_curves"
    gallery.mkdir(exist_ok=True)
    eligible = mol[mol.points >= 8].copy()
    if len(eligible) < 6:
        return []
    picks = list(eligible.head(2).identity_key)
    middle = eligible.iloc[(np.linspace(0.4, 0.6, 2) * (len(eligible)-1)).astype(int)]
    picks += list(middle.identity_key)
    picks += list(eligible.tail(2).identity_key)
    saved = []
    for idx, identity in enumerate(dict.fromkeys(picks), 1):
        g = df[df.identity_key == identity].sort_values("energy_eV")
        im, d = canvas(f"{g.molecule_name.iloc[0]} ({g.formula.iloc[0]})",
                       "Experiment vs raw BEB vs out-of-fold ML correction", (1200, 760))
        x0, y0, w, h = 110, 150, 980, 500
        xmin, xmax = g.energy_eV.min(), g.energy_eV.max()
        ymax = max(g.experimental_sigma_A2.max(), g.prediction_raw_beb_A2.max(), g.prediction_ml_A2.max()) * 1.12
        d.rectangle((x0, y0, x0+w, y0+h), outline="#d1d5db", width=2)
        def xy(e, s):
            return (x0+w*(e-xmin)/(xmax-xmin or 1), y0+h-h*s/(ymax or 1))
        for col, color, width in [("prediction_raw_beb_A2", "#2a788e", 4), ("prediction_ml_A2", "#e76f51", 4)]:
            pts = [xy(e, s) for e, s in zip(g.energy_eV, g[col])]
            if len(pts) > 1: d.line(pts, fill=color, width=width)
        for e, s in zip(g.energy_eV, g.experimental_sigma_A2):
            px, py = xy(e, s); d.ellipse((px-5, py-5, px+5, py+5), fill="#111827")
        d.text((x0, y0+h+24), f"Energy: {xmin:.1f}–{xmax:.1f} eV", fill="#111827", font=font(18))
        d.text((760, 105), "● experiment", fill="#111827", font=font(18))
        d.text((900, 105), "— raw BEB", fill="#2a788e", font=font(18))
        d.text((1030, 105), "— ML", fill="#e76f51", font=font(18))
        path = gallery / f"{idx:02d}_{identity}.png"
        im.save(path); saved.append(str(path))
    return saved


def bootstrap_intervals(df, repeats=10000):
    rng = np.random.default_rng(20260802)
    identities = sorted(df.identity_key.unique())
    per_model = {k: [] for k in ["raw_beb", "constant_scale", "ml_residual"]}
    diffs = []
    columns = {"raw_beb": "prediction_raw_beb_A2", "constant_scale": "prediction_constant_scale_A2", "ml_residual": "prediction_ml_A2"}
    for identity in identities:
        g = df[df.identity_key == identity]
        m = g.primary_low_energy.to_numpy(bool)
        if not m.any():
            m[:] = True
        exp = g.experimental_sigma_A2.to_numpy(float)[m]
        for key, col in columns.items():
            pred = g[col].to_numpy(float)[m]
            per_model[key].append(float(np.sum(np.abs(pred-exp))/np.sum(exp)))
    arrays = {k: np.asarray(v) for k, v in per_model.items()}
    samples = rng.integers(0, len(identities), size=(repeats, len(identities)))
    result = {}
    for key, arr in arrays.items():
        means = arr[samples].mean(axis=1)
        result[key] = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    difference = (arrays["raw_beb"] - arrays["ml_residual"])[samples].mean(axis=1)
    result["paired_raw_minus_ml_percentage_points"] = [float(np.quantile(difference, 0.025)*100), float(np.quantile(difference, 0.975)*100)]
    result["probability_ml_improves_macro_nmae"] = float(np.mean(difference > 0))
    return result


def main():
    df = build_dataset()
    x, feature_names = expanded_features(df)
    folds, pred_const, pred_ml, choices, assignment = cross_validate(df, x)
    df["cv_fold"] = folds
    df["prediction_raw_beb_A2"] = df.beb_sigma_A2
    df["prediction_constant_scale_A2"] = pred_const
    df["prediction_ml_A2"] = pred_ml
    metrics = {
        "raw_beb": metrics_bundle(df, df.prediction_raw_beb_A2.to_numpy(float)),
        "constant_scale": metrics_bundle(df, pred_const),
        "ml_residual": metrics_bundle(df, pred_ml),
    }
    bootstrap = bootstrap_intervals(df)
    score, alpha, blend = choose_hyperparameters(
        x, df.target_log_ratio.to_numpy(float), df.experimental_sigma_A2.to_numpy(float),
        df.beb_sigma_A2.to_numpy(float), df.identity_key.to_numpy(str), df.sample_weight.to_numpy(float))
    final = fit_ridge(x, df.target_log_ratio.to_numpy(float), df.sample_weight.to_numpy(float), alpha)
    global_residual = weighted_mean(df.target_log_ratio.to_numpy(float), df.sample_weight.to_numpy(float))

    df.to_csv(OUT / "training_points_and_oof_predictions.csv", index=False, encoding="utf-8-sig")
    mol = molecule_errors(df)
    mol.to_csv(OUT / "per_molecule_oof_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez(OUT / "final_model_parameters.npz", feature_mean=final["mean"], feature_scale=final["scale"],
             coefficients=final["coef"], alpha=np.array([alpha]), blend=np.array([blend]),
             global_log_residual=np.array([global_residual]))

    summary = {
        "training_scope": "BEB residual correction using every locally available, identity-matched numeric experimental/BEB observation",
        "unique_molecules": int(df.identity_key.nunique()),
        "numeric_points": int(len(df)),
        "primary_low_energy_points": int(df.primary_low_energy.sum()),
        "label_type_molecules": {k: int(v) for k, v in df.groupby("label_type").identity_key.nunique().items()},
        "label_type_points": {k: int(v) for k, v in df.label_type.value_counts().items()},
        "metrics": metrics,
        "molecule_bootstrap_95_CI": bootstrap,
        "relative_primary_NMAE_change_vs_raw_percent": 100 * (metrics["ml_residual"]["primary_macro_NMAE"] / metrics["raw_beb"]["primary_macro_NMAE"] - 1),
        "outer_fold_hyperparameters": choices,
        "final_model": {"alpha": alpha, "blend_with_ridge": blend, "constant_component": 1-blend,
                        "inner_cv_macro_nmae": score, "feature_names": feature_names,
                        "target": "log(experimental cross section / raw BEB cross section)"},
        "validation_policy": "Five-fold molecule-level cross-validation. No identity appears in both train and test within a fold.",
        "limitations": [
            "The 194-entry ready registry is an identity/evidence registry; only identities with explicit locally recoverable numeric experiment-BEB pairs enter this fitted model.",
            "Full curves, sparse points, peak heights, and 70-75 eV interval means are heterogeneous labels and are retained as such.",
            "Peak-only and fixed-energy labels constrain amplitude but do not validate an entire curve or peak position.",
            "Tier-B digitised points retain source/digitisation uncertainty and are not represented as Tier-A tables.",
        ],
    }
    (OUT / "metrics_and_training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "model_card.md").write_text(
        "# Final BEB residual-correction model\n\n"
        f"- Training identities: **{summary['unique_molecules']}**\n"
        f"- Numeric experiment/BEB observations: **{summary['numeric_points']}**\n"
        f"- Primary low-energy observations: **{summary['primary_low_energy_points']}**\n"
        "- Target: `log(sigma_experiment / sigma_BEB)`\n"
        "- Validation: five-fold molecule-level CV (identity-disjoint).\n"
        "- Weighting: each molecule has equal total base weight; uncertainty and low-energy priority modify within-molecule weights.\n"
        f"- Raw BEB primary macro-NMAE: **{metrics['raw_beb']['primary_macro_NMAE']*100:.2f}%**\n"
        f"- ML primary macro-NMAE: **{metrics['ml_residual']['primary_macro_NMAE']*100:.2f}%**\n"
        f"- Change: **{summary['relative_primary_NMAE_change_vs_raw_percent']:+.2f}% relative**.\n\n"
        "The model is saved as NumPy parameters with its feature order in the JSON summary. It is intended as a reproducible residual correction to raw BEB, not a replacement for experimental validation. Peak-only and fixed-energy labels do not become invented full curves.\n",
        encoding="utf-8")

    save_metric_chart(metrics, bootstrap)
    save_parity(df)
    save_rank_chart(mol)
    save_error_hist(df)
    save_coverage(df)
    gallery = save_curve_gallery(df, mol)
    summary["representative_curve_images"] = gallery
    (OUT / "metrics_and_training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    relative_reduction = -summary["relative_primary_NMAE_change_vs_raw_percent"]
    (OUT / "README.md").write_text(
        "# Final ML training package — 2026-08-02\n\n"
        "## What was trained\n\n"
        "The fitted model predicts `log(experimental / BEB)` from raw BEB, collision energy, and formula-derived descriptors.\n\n"
        f"- **{summary['unique_molecules']}** unique molecular identities\n"
        f"- **{summary['numeric_points']}** numeric experiment–BEB observations\n"
        f"- **{summary['primary_low_energy_points']}** primary low-energy observations\n"
        "- five-fold molecule-level cross-validation; no molecule is shared between train and test within a fold\n"
        "- equal base weight per molecule, with uncertainty and low-energy priority weighting\n\n"
        "The 194-entry ready registry is an evidence registry. Model fitting additionally requires an explicit numeric experimental value and an explicit matched-energy BEB value; missing points are never invented.\n\n"
        "## Held-out result\n\n"
        f"- Raw BEB primary molecule-macro NMAE: **{metrics['raw_beb']['primary_macro_NMAE']*100:.2f}%**\n"
        f"- ML-corrected primary molecule-macro NMAE: **{metrics['ml_residual']['primary_macro_NMAE']*100:.2f}%**\n"
        f"- Relative primary error reduction: **{relative_reduction:.2f}%**\n"
        f"- Paired molecule-bootstrap improvement (95% CI): **{bootstrap['paired_raw_minus_ml_percentage_points'][0]:.2f} to {bootstrap['paired_raw_minus_ml_percentage_points'][1]:.2f} percentage points**\n"
        f"- Raw / ML pointwise R²: **{metrics['raw_beb']['R2_pointwise']:.3f} / {metrics['ml_residual']['R2_pointwise']:.3f}**\n\n"
        "The constant-scale control is also reported and performs worse than raw BEB, so the ML gain is not a single global multiplier.\n\n"
        "## Main files\n\n"
        "- `training_points_and_oof_predictions.csv`: full numeric table and held-out predictions\n"
        "- `per_molecule_oof_metrics.csv`: one-row-per-molecule comparison\n"
        "- `metrics_and_training_summary.json`: metrics, folds, feature order, confidence intervals, limitations\n"
        "- `final_model_parameters.npz`: final model fitted on every accepted training identity\n"
        "- `01`–`05` PNG: supervisor-ready summary figures\n"
        "- `representative_curves/`: good, typical, and difficult examples\n\n"
        "## Reproduce / apply\n\n"
        "Run `tools/train_final_beb_residual_model.py` to reproduce training. Apply `tools/apply_final_beb_residual_model.py` to a CSV with `formula`, `energy_eV`, and `beb_sigma_A2`.\n\n"
        "## Scientific boundary\n\n"
        "Peak-only and 70–75 eV labels constrain amplitude but are not full experimental curves. Tier-B digitised points remain Tier B. This is a validated residual correction within the represented chemistry, not a universal-accuracy claim.\n",
        encoding="utf-8")
    print(json.dumps({"out": str(OUT), "molecules": summary["unique_molecules"], "points": summary["numeric_points"],
                      "raw_primary_nmae": metrics["raw_beb"]["primary_macro_NMAE"],
                      "ml_primary_nmae": metrics["ml_residual"]["primary_macro_NMAE"],
                      "alpha": alpha, "blend": blend, "gallery": len(gallery)}, indent=2))


if __name__ == "__main__":
    main()
