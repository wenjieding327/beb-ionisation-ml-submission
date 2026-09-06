from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "expanded_194_training_2026_08_02"
OUT.mkdir(parents=True, exist_ok=True)
NOQEC = ROOT / "outputs" / "no_qec_priority_2026_07_31"
MASTER_DIR = ROOT / "outputs" / "current_experiment_beb_master_2026_08_01"
PROJECT_ROOT = Path(os.environ.get("BEB_PROJECT_ROOT", ROOT)).expanduser()
LEGACY_ROOTS = [
    Path(value).expanduser()
    for value in os.environ.get("BEB_LEGACY_ROOTS", "").split(os.pathsep)
    if value
]

sys.path.insert(0, str(ROOT / "tools"))
import reconstruct_scott_irikura_ecp_beb as reconstruct  # noqa: E402


ALIAS_TO_MASTER = {
    "methylcyanide": "acetonitrile",
    "c2h6": "ethane",
    "dimethylether": "dimethyl-ether",
    "1-2-ethanediol-or-ethylene-glycol": "ethylene-glycol",
    "propan-1-yne": "propyne",
    "prop-2-enal": "acrolein",
    "ethylcyanide": "propionitrile",
    "methyl-oxirane": "propylene-oxide",
    "propanaldehyde": "propionaldehyde",
    "butan-1-3-diene": "1-3-butadiene",
    "isopropylcyanide": "isobutyronitrile",
    "propylcyanide": "butyronitrile",
    "hexan-1-ene": "1-hexene",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def bool_value(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        frame = pd.read_csv(path, comment="#")
        if frame.shape[1] >= 2:
            names = [str(c).strip().lower() for c in frame.columns]
            e_idx = next(
                (i for i, c in enumerate(names)
                 if "energy" in c or c in {"e", "t", "e_beb(ev)", "e_beb"} or c.startswith("e_")),
                0,
            )
            # Do not mistake an ``E_BEB`` energy column for a BEB cross section.
            # Prefer the explicit Q_BEB/combined/cross-section columns used by the
            # accepted numeric sources.
            s_idx = next(
                (i for i, c in enumerate(names)
                 if any(k in c for k in ["q_beb", "cross", "sigma", "tics", "combined"]) and i != e_idx),
                1,
            )
            e = pd.to_numeric(frame.iloc[:, e_idx], errors="coerce").to_numpy(float)
            s = pd.to_numeric(frame.iloc[:, s_idx], errors="coerce").to_numpy(float)
        else:
            raise ValueError
    except Exception:
        raw = np.genfromtxt(path, delimiter=",", comments="#")
        if raw.ndim != 2 or raw.shape[1] < 2:
            raw = np.genfromtxt(path, comments="#")
        e, s = raw[:, 0], raw[:, 1]
    mask = np.isfinite(e) & np.isfinite(s)
    e, s = e[mask], s[mask]
    order = np.argsort(e)
    return e[order], s[order]


def interpolate(path: Path, energy: float):
    try:
        e, s = read_curve(path)
        if len(e) < 2 or energy < e.min() or energy > e.max():
            return None
        return float(np.interp(energy, e, s))
    except Exception:
        return None


def curve_peak(path: Path):
    try:
        e, s = read_curve(path)
        if len(e) < 2:
            return None, None
        idx = int(np.nanargmax(s))
        return float(e[idx]), float(s[idx])
    except Exception:
        return None, None


def canonical(identity: str) -> str:
    identity = slug(identity)
    return ALIAS_TO_MASTER.get(identity, identity)


def add(rows: list[dict], *, identity: str, name: str, formula: str, energy, exp, beb,
        uncertainty=None, label_type: str, source: str, primary=True, evidence: str,
        energy_is_reported=True) -> None:
    exp = finite(exp)
    beb = finite(beb)
    energy = finite(energy)
    uncertainty = finite(uncertainty)
    if exp is None or beb is None or exp <= 0.02 or beb <= 0.02:
        return
    if exp / beb < 0.04 or exp / beb > 25:
        return
    rows.append({
        "identity_key": canonical(identity),
        "molecule_name": str(name).strip(),
        "formula": str(formula).strip(),
        "energy_eV": energy,
        "experimental_sigma_A2": exp,
        "beb_sigma_A2": beb,
        "experimental_uncertainty_A2": uncertainty,
        "label_type": label_type,
        "primary_low_energy": bool(primary),
        "evidence": evidence,
        "source": source,
        "energy_is_reported": bool(energy_is_reported and energy is not None),
    })


def corrected_local_path(value: str) -> Path | None:
    if not value or str(value).lower() == "nan":
        return None
    raw = str(value).strip()
    candidates = [Path(raw)]
    for legacy_root in LEGACY_ROOTS:
        token = str(legacy_root)
        if raw.startswith(token):
            candidates.append(ROOT / raw[len(token):].lstrip("\\/"))
    if not Path(raw).is_absolute():
        candidates.append(ROOT / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    master = pd.read_csv(MASTER_DIR / "ready_numeric_now.csv")
    master["identity_key"] = master.identity_key.map(canonical)
    master = master.drop_duplicates("identity_key", keep="first")
    master_ids = set(master.identity_key)
    master_info = master.set_index("identity_key")

    old = pd.read_csv(ROOT / "outputs" / "final_nested_blend_v5_ml_2026_08_02" / "nested_blend_v5_oof_predictions.csv")
    base_cols = [
        "identity_key", "molecule_name", "formula", "energy_eV", "experimental_sigma_A2",
        "beb_sigma_A2", "experimental_uncertainty_A2", "label_type", "primary_low_energy",
        "evidence", "source",
    ]
    old = old[base_cols].copy()
    old["identity_key_original"] = old.identity_key
    old["identity_key"] = old.identity_key.map(canonical)
    old["energy_is_reported"] = True
    for identity, group in old.groupby("identity_key"):
        if identity in master_info.index:
            old.loc[group.index, "molecule_name"] = master_info.loc[identity, "molecule_name"]
            old.loc[group.index, "formula"] = master_info.loc[identity, "formula"]

    base_ids = set(old.identity_key)
    target_ids = master_ids - base_ids
    rows: list[dict] = []
    source_counts: dict[str, int] = {}

    audit_path = PROJECT_ROOT / "outputs" / "final_beb_ml_corrected_2026_07_30" / "paired_molecule_audit.csv"
    paired = pd.read_csv(audit_path)
    paired = paired[paired.pair_eligible_for_ml.astype(bool)].copy()
    paired["identity_key"] = paired.identity_key.map(canonical)
    paired_lookup = {}
    for r in paired.itertuples():
        paired_lookup[canonical(r.identity_key)] = r
        paired_lookup[slug(r.molecule_name)] = r

    qec_points_path = PROJECT_ROOT / "data" / "processed" / "qec_low_energy_priority_curve_points_2026_07_26.csv"
    qec_points = pd.read_csv(qec_points_path)
    qec_identity_paths: dict[str, Path] = {}
    qec_common_to_identity: dict[str, str] = {}
    for common in qec_points.common_name_key.dropna().unique():
        rec = paired_lookup.get(slug(common))
        if rec is not None:
            ident = canonical(rec.identity_key)
            qec_common_to_identity[str(common)] = ident
            path = corrected_local_path(qec_points.loc[qec_points.common_name_key == common, "qec_result_file"].iloc[0])
            if path:
                qec_identity_paths[ident] = path

    def resolve_beb(identity: str, locator: str | None) -> Path | None:
        ident = canonical(identity)
        direct = corrected_local_path(locator or "")
        if direct:
            return direct
        raw = str(locator or "")
        if "new32_qec_beb_successful30_2026_07_30.tar.gz::" in raw:
            rel = raw.split("::", 1)[1].replace("/", "\\")
            candidate = ROOT / "_qec30_audit" / rel
            if candidate.is_file():
                return candidate
        if ident in qec_identity_paths:
            return qec_identity_paths[ident]
        if ident in paired.set_index("identity_key").index:
            loc = paired.set_index("identity_key").loc[ident, "beb_source_path"]
            if isinstance(loc, pd.Series):
                loc = loc.iloc[0]
            direct = corrected_local_path(loc)
            if direct:
                return direct
            if "new32_qec_beb_successful30_2026_07_30.tar.gz::" in str(loc):
                rel = str(loc).split("::", 1)[1].replace("/", "\\")
                candidate = ROOT / "_qec30_audit" / rel
                if candidate.is_file():
                    return candidate
        return None

    # 1) High-quality QEC/Bull-Harland curve table with already interpolated BEB values.
    before = len(rows)
    for r in qec_points.itertuples():
        identity = qec_common_to_identity.get(str(r.common_name_key))
        if identity not in target_ids:
            continue
        unc = finite(r.uncertainty_percent)
        exp = finite(getattr(r, "_22", None))  # pandas renames cross_section_1e-16_cm2
        if exp is None:
            exp = finite(qec_points.loc[r.Index, "cross_section_1e-16_cm2"])
        add(rows, identity=identity, name=paired_lookup[identity].molecule_name,
            formula=paired_lookup[identity].formula, energy=r.energy_eV, exp=exp, beb=r.qec_beb_A2,
            uncertainty=(exp * unc / 100 if exp is not None and unc is not None else None),
            label_type="curve_point", source=str(qec_points_path), primary=bool_value(r.is_prepeak_region),
            evidence=str(r.quality_flag))
    source_counts["qec_priority_curves"] = len(rows) - before

    # 2) Other audited measured curves, paired with the accepted BEB curve at the same energy.
    expanded_path = PROJECT_ROOT / "data" / "processed" / "expanded_measured_total_master_2026_07_26.csv"
    expanded = pd.read_csv(expanded_path)
    expanded["molecule_slug"] = expanded.molecule_name.map(slug)
    before = len(rows)
    for rec in paired.itertuples():
        identity = canonical(rec.identity_key)
        if identity not in target_ids or identity in qec_identity_paths:
            continue
        names = {slug(rec.identity_key), slug(rec.molecule_name)}
        evidence = expanded[expanded.molecule_slug.isin(names) & expanded.use_for_extended_ml_analysis.astype(bool)]
        path = resolve_beb(identity, rec.beb_source_path)
        if evidence.empty or not path:
            continue
        peak_e = finite(rec.experimental_peak_energy_eV)
        for r in evidence.itertuples():
            energy = finite(r.energy_eV)
            if energy is None or energy > 300:
                continue
            beb = interpolate(path, energy)
            exp = finite(r.cross_section_1e_minus_16_cm2)
            unc = finite(r.uncertainty_1e_minus_16_cm2)
            add(rows, identity=identity, name=rec.molecule_name, formula=rec.formula,
                energy=energy, exp=exp, beb=beb, uncertainty=unc, label_type="curve_point",
                source=str(expanded_path), primary=(peak_e is None or energy <= peak_e),
                evidence=str(r.evidence_tier))
    source_counts["expanded_audited_curves"] = len(rows) - before

    # 3) All fixed-energy Bose labels whose BEB locators can be resolved.
    fixed_path = NOQEC / "ready100_fixed_energy_with_existing_beb_no_qec.csv"
    fixed = pd.read_csv(fixed_path)
    before = len(rows)
    for r in fixed.itertuples():
        identity = canonical(r.local_beb_identity_key)
        if identity not in target_ids:
            continue
        path = resolve_beb(identity, r.local_beb_source_path)
        if not path:
            continue
        values = [interpolate(path, 70.0), interpolate(path, 75.0)]
        values = [x for x in values if x is not None]
        if not values:
            continue
        add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
            energy=72.5, exp=r.mean_cross_section_A2, beb=float(np.mean(values)),
            uncertainty=r.confidence_95_A2, label_type="fixed_70_75eV", source=str(fixed_path),
            primary=True, evidence="published_fixed_energy")
    source_counts["fixed_70_75"] = len(rows) - before

    # 4) Bull 2012 numeric peak labels. Peak energy is taken from the accepted BEB curve, not invented.
    bull_path = NOQEC / "agent_bull2012_peak32.csv"
    bull = pd.read_csv(bull_path)
    before = len(rows)
    for r in bull.itertuples():
        identity = canonical(r.local_beb_identity_key)
        if identity not in target_ids:
            continue
        path = resolve_beb(identity, "")
        if not path:
            continue
        energy, beb = curve_peak(path)
        add(rows, identity=identity, name=r.name, formula=r.formula, energy=energy,
            exp=r.peak_value, beb=beb, uncertainty=r.uncertainty, label_type="peak_height",
            source=str(bull_path), primary=True, evidence="published_numeric_peak",
            energy_is_reported=False)
    source_counts["bull2012_peaks"] = len(rows) - before

    # 5) NIST digitised curves that passed visual QA; restrict to missing identities.
    nist_path = MASTER_DIR / "nist_fast12_experiment_beb_digitized_READY_TIER_B.csv"
    nist = pd.read_csv(nist_path)
    peak_energy = nist.groupby("identity_key").apply(
        lambda g: float(g.loc[g.experiment_sigma_A2_digitized.idxmax(), "energy_eV_digitized"]),
        include_groups=False,
    ).to_dict()
    before = len(rows)
    for r in nist.itertuples():
        identity = canonical(r.identity_key)
        if identity not in target_ids:
            continue
        uncertainties = [finite(r.experiment_uncertainty_minus_A2_digitized), finite(r.experiment_uncertainty_plus_A2_digitized)]
        uncertainties = [x for x in uncertainties if x is not None]
        add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
            energy=r.energy_eV_digitized, exp=r.experiment_sigma_A2_digitized,
            beb=r.beb_sigma_A2_interpolated, uncertainty=max(uncertainties) if uncertainties else None,
            label_type="curve_point", source=str(nist_path),
            primary=float(r.energy_eV_digitized) <= peak_energy[r.identity_key], evidence="digitized_Tier_B")
    source_counts["nist_fast12"] = len(rows) - before

    # 6) Bart/Hudson independent experimental peaks versus raw, uncorrected BEB peaks.
    before = len(rows)
    for filename in [
        "bart_hudson2001_new6_strict_peak_pairs_2026_08_02.csv",
        "bart_hudson2001_dichloromethane_strict_peak_pair_2026_08_02.csv",
    ]:
        path = NOQEC / filename
        frame = pd.read_csv(path)
        for r in frame.itertuples():
            identity = canonical(r.identity_key)
            if identity not in target_ids:
                continue
            unc = float(r.experimental_peak_A2) * float(r.experimental_uncertainty_percent) / 100
            add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
                energy=r.raw_BEB_peak_energy_eV, exp=r.experimental_peak_A2, beb=r.raw_BEB_peak_A2,
                uncertainty=unc, label_type="peak_height_numeric", source=str(path), primary=True,
                evidence="published_raw_BEB_peak_pair", energy_is_reported=False)
    source_counts["bart_hudson_peaks"] = len(rows) - before

    # 7) Heavy-element ECP-BEB peaks. Reconstruct peak energy from SI when possible.
    heavy_peak_path = NOQEC / "scott_irikura2005_new5_strict_peak_pairs_2026_08_02.csv"
    heavy_peak = pd.read_csv(heavy_peak_path)
    before = len(rows)
    for r in heavy_peak.itertuples():
        identity = canonical(r.identity_key)
        if identity not in target_ids:
            continue
        energy = None
        curve_file = corrected_local_path(r.beb_curve_file)
        if curve_file and curve_file.suffix.lower() == ".bun":
            orbitals = reconstruct.parse_bun(curve_file)
            threshold = min(o.binding_ev for o in orbitals)
            energies = np.geomspace(threshold * 1.001, 300.0, 1200)
            values = np.asarray([reconstruct.total_beb_a2(float(e), orbitals) for e in energies])
            energy = float(energies[int(np.argmax(values))])
        add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
            energy=energy, exp=r.experimental_peak_A2, beb=r.raw_BEB_peak_A2,
            uncertainty=r.experimental_uncertainty_A2, label_type="peak_height_numeric",
            source=str(heavy_peak_path), primary=True, evidence="published_ECP_BEB_peak_pair",
            energy_is_reported=False)
    source_counts["heavy_element_peaks"] = len(rows) - before

    # 8) Zhong/Bose fixed-energy pairs.
    zhong_path = NOQEC / "zhong_bose8_strict_fixed_energy_pairs_2026_08_01.csv"
    zhong = pd.read_csv(zhong_path)
    before = len(rows)
    for r in zhong.itertuples():
        identity = canonical(r.identity_key)
        if identity not in target_ids:
            continue
        path = corrected_local_path(r.beb_curve_path)
        if not path:
            continue
        values = [interpolate(path, 70.0), interpolate(path, 75.0)]
        values = [x for x in values if x is not None]
        if not values:
            continue
        add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
            energy=72.5, exp=r.experimental_cross_section_A2, beb=float(np.mean(values)),
            uncertainty=r.experimental_uncertainty_95_A2, label_type="fixed_70_75eV",
            source=str(zhong_path), primary=True, evidence="published_fixed_energy_Zhong")
    source_counts["zhong_fixed"] = len(rows) - before

    # 9) Formula-reconstructed heavy-element fixed points.
    heavy_fixed_path = NOQEC / "scott_irikura_heavy_ready_vs_pending_2026_08_01.csv"
    heavy_fixed = pd.read_csv(heavy_fixed_path)
    before = len(rows)
    for r in heavy_fixed.itertuples():
        if r.status != "ready_formula_reconstructed":
            continue
        identity = canonical(r.molecule_name)
        if identity not in target_ids:
            continue
        add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
            energy=r.experimental_energy_eV, exp=r.experimental_value_A2,
            beb=r.beb_at_experimental_energy_A2, uncertainty=r.experimental_uncertainty_A2,
            label_type="fixed_energy_numeric", source=str(heavy_fixed_path), primary=True,
            evidence="published_SI_formula_reconstructed_BEB")
    source_counts["heavy_fixed"] = len(rows) - before

    # 10) Nitromethane same-energy/peak pair.
    nitro_path = NOQEC / "nitromethane_jiao_luthra_strict_peak_pair_2026_08_01.csv"
    nitro = pd.read_csv(nitro_path)
    before = len(rows)
    for r in nitro.itertuples():
        identity = canonical(r.identity_key)
        if identity not in target_ids:
            continue
        add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
            energy=r.experimental_peak_energy_eV, exp=r.experimental_peak_A2,
            beb=r.published_beb_at_experimental_peak_A2, uncertainty=r.experimental_uncertainty_A2,
            label_type="peak_height_numeric", source=str(nitro_path), primary=True,
            evidence="published_same_energy_peak_pair")
    source_counts["nitromethane"] = len(rows) - before

    # 11) New Bull 2014 identity with a numeric peak and an accepted QEC curve.
    bull14_path = NOQEC / "bull2014_peak15_strict_pairs_2026_07_31.csv"
    bull14 = pd.read_csv(bull14_path)
    before = len(rows)
    for r in bull14.itertuples():
        identity = canonical(r.molecule_name)
        if identity not in target_ids or r.current_registry_status != "new_identity_peak_ready":
            continue
        path = resolve_beb(identity, r.beb_source_path)
        energy, beb = curve_peak(path) if path else (None, None)
        add(rows, identity=identity, name=r.molecule_name, formula=r.formula,
            energy=energy, exp=r.experimental_peak_A2, beb=beb,
            uncertainty=r.experimental_uncertainty_A2, label_type="peak_height_numeric",
            source=str(bull14_path), primary=True, evidence="published_numeric_peak",
            energy_is_reported=False)
    source_counts["bull2014_new_identity"] = len(rows) - before

    extra = pd.DataFrame(rows)
    combined = pd.concat([old.drop(columns=["identity_key_original"]), extra], ignore_index=True, sort=False)
    combined["primary_low_energy"] = combined.primary_low_energy.map(bool_value)
    combined["energy_is_reported"] = combined.energy_is_reported.map(bool_value)
    combined["dedup_key"] = (
        combined.identity_key.astype(str) + "|" + combined.label_type.astype(str) + "|" +
        combined.energy_eV.fillna(-1).round(3).astype(str) + "|" +
        combined.experimental_sigma_A2.round(4).astype(str) + "|" +
        combined.beb_sigma_A2.round(4).astype(str)
    )
    combined = combined.drop_duplicates("dedup_key", keep="first").drop(columns="dedup_key")
    combined = combined[(combined.energy_eV.isna() | (combined.energy_eV <= 300)) &
                        (combined.experimental_sigma_A2 > 0.02) & (combined.beb_sigma_A2 > 0.02)].copy()

    counts = combined.groupby("identity_key").size()
    combined["molecule_weight"] = combined.identity_key.map(lambda x: 1.0 / counts[x])
    unc_rel = combined.experimental_uncertainty_A2 / combined.experimental_sigma_A2
    quality = 1.0 / (1.0 + (unc_rel.fillna(0.12) / 0.12) ** 2)
    combined["sample_weight"] = combined.molecule_weight * quality * np.where(combined.primary_low_energy, 2.0, 0.75)
    combined["target_log_ratio"] = np.log(combined.experimental_sigma_A2 / combined.beb_sigma_A2)
    combined = combined.sort_values(["identity_key", "energy_eV", "label_type", "source"], na_position="last").reset_index(drop=True)

    final_ids = set(combined.identity_key)
    missing = sorted(master_ids - final_ids)
    unexpected = sorted(final_ids - master_ids)
    if missing or unexpected:
        audit_missing = master[master.identity_key.isin(missing)]
        audit_missing.to_csv(OUT / "unresolved_master_identities.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(f"Expanded dataset identity mismatch: missing={missing}; unexpected={unexpected}")

    # Deterministic balanced folds by physical identity.
    sizes = combined.groupby("identity_key").size().to_dict()
    ordered = sorted(sizes, key=lambda k: (-sizes[k], k))
    loads = [0] * 5
    assignment = {}
    for identity in ordered:
        fold = min(range(5), key=lambda i: loads[i])
        assignment[identity] = fold
        loads[fold] += sizes[identity]
    combined["cv_fold"] = combined.identity_key.map(assignment).astype(int)

    output_path = OUT / "expanded_194_training_points.csv"
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"identity_key": x, "rows": int(counts[x]), "cv_fold": assignment[x],
         "source_group": master_info.loc[x, "source_group"]}
        for x in sorted(master_ids)
    ]).to_csv(OUT / "expanded_194_identity_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "master_ready_identities": len(master_ids),
        "old_identity_keys": int(old.identity_key.nunique()),
        "old_physical_identities_after_alias_merge": len(base_ids),
        "new_physical_identities_added": len(target_ids),
        "final_physical_identities": int(combined.identity_key.nunique()),
        "final_observations": int(len(combined)),
        "primary_observations": int(combined.primary_low_energy.sum()),
        "curve_identities": int(combined.loc[combined.label_type == "curve_point", "identity_key"].nunique()),
        "fold_row_loads": loads,
        "source_rows_added": source_counts,
        "unresolved_identities": missing,
        "unexpected_identities": unexpected,
        "scientific_note": "All 194 ready master identities are represented exactly once at the physical-identity grouping level; historical aliases are merged before fold assignment.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
