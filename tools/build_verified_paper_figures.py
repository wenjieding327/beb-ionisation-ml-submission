"""Rebuild report Figures 1--14 using only the corrected release outputs.

Run from any working directory: python tools/build_verified_paper_figures.py
The release root is resolved from this script, never from historical project data.
Figures summarise fixed out-of-fold predictions; case selection is descriptive,
not an additional validation or model-selection step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

EXP = "experimental_sigma_A2"
RAW = "prediction_raw_beb_A2"
CONST = "prediction_constant_scale_A2"
ML = "prediction_expanded_ml_A2"
INK, RAW_C, CONST_C, ML_C = "#202020", "#2F6F89", "#B8A27A", "#B54A2A"
GOOD, BAD, SHADE = "#3D7C68", "#A33C3C", "#EEF3F1"
FIGURES = [
    "01_dataset_composition", "02_analysis_workflow", "03_primary_curve_nmae",
    "04_oof_prediction_parity", "05_point_level_percentage_error",
    "06_scope_and_task_performance", "07_controlled_ablations",
    "08_feature_importance", "09_leave_one_source_group_out",
    "10_molecular_heterogeneity", "11_classic_small_molecule_benchmarks",
    "12_strong_improvements", "13_data_quality_warnings", "14_severe_failure",
]
# Exact row-to-series provenance checked against the archived NIST O3 graph HTML.
# These are original source-table line numbers, retained in the corrected OOF file.
OZONE_SERIES = {
    "Newson et al. (1995)": {2843,2846,2848,2851,2853,2856,2858,2867,2869,2871},
    "Siegel (1982)": {2841,2842,2845,2850,2855,2859,2862,2864,2866},
}


def initialise_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.titlesize": 12, "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 11,
        "legend.fontsize": 10.5, "figure.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "grid.color": "#E2E4E5", "grid.linewidth": .6,
        "axes.axisbelow": True, "savefig.facecolor": "white",
        "svg.fonttype": "none", "axes.unicode_minus": True,
    })


def nmae(f, col):
    return float(np.abs(f[col] - f[EXP]).sum() / np.abs(f[EXP]).sum())


def primary(f):
    p = f.loc[f.primary_low_energy]
    return p if len(p) else f


def molecule_errors(f):
    rows = []
    for identity, g in f.groupby("identity_key", sort=True):
        p = primary(g)
        row = {
            "identity_key": identity, "molecule_name": str(g.molecule_name.iloc[0]),
            "formula": str(g.formula.iloc[0]), "rows": int(len(g)),
            "primary_rows": int(len(p)), "sources": sorted(g.source.astype(str).unique()),
            "energy_min_eV": float(g.energy_eV.min()),
            "energy_max_eV": float(g.energy_eV.max()),
            "primary_energy_max_eV": float(p.energy_eV.max()),
            "cv_fold": int(g.cv_fold.iloc[0]),
        }
        for short, col in [("raw", RAW), ("constant", CONST), ("ml", ML)]:
            row[short + "_primary_nmae_percent"] = 100 * nmae(p, col)
            row[short + "_whole_nmae_percent"] = 100 * nmae(g, col)
        row["improvement_pp"] = row["raw_primary_nmae_percent"] - row["ml_primary_nmae_percent"]
        rows.append(row)
    return pd.DataFrame(rows)


def save(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=240, bbox_inches="tight", pad_inches=.10)
    fig.savefig(out / (name + ".svg"), bbox_inches="tight", pad_inches=.10)
    plt.close(fig)


def wrap(value, width=24):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def label_bars(ax, bars, fmt="{:.2f}", pad=3):
    ax.bar_label(bars, fmt=fmt, padding=pad, fontsize=10.5)


def dataset_figure(f, out):
    order = ["curve", "fixed_sparse", "peak"]
    names = ["Energy-dependent\ncurve points", "Fixed-energy /\nsparse points", "Peak-height\nobservations"]
    rowcounts = [int(f.task_family.eq(k).sum()) for k in order]
    identities = [int(f.loc[f.task_family.eq(k), "identity_key"].nunique()) for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(3)
    for ax, vals, title in zip(axes, [rowcounts, identities], ["Observations", "Molecules"]):
        bars = ax.barh(y, vals, color=[RAW_C, CONST_C, ML_C], height=.57)
        ax.set_yticks(y, names if ax is axes[0] else [""] * 3)
        ax.invert_yaxis()
        ax.set_title(title, pad=10)
        ax.set_xlim(0, max(vals) * 1.23)
        ax.grid(axis="x")
        label_bars(ax, bars, "{:,.0f}")
        ax.set_xlabel("Count")
    fig.suptitle(f"{f.identity_key.nunique()} identities; {len(f):,} retained observations", y=1.01)
    fig.text(.5, -.03, "Molecule groups overlap: one identity can provide more than one evidence type.",
             ha="center", fontsize=10.5)
    fig.tight_layout(w_pad=1.7)
    save(fig, out, FIGURES[0])


def workflow_figure(f, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    texts = [
        "1  Traceable inputs\nPublished experiments\n+ published / QEC BEB",
        "2  Match and audit\nIdentity, observable, units\nUnresolved rows quarantined",
        "3  Construct features\n49 BEB, energy and\ncomposition descriptors",
        "6  Report evidence\nOOF errors and cases\nBatch-transfer stress test",
        "5  Predict by task\nIndependent curve, fixed-\npoint and peak heads",
        "4  Nested validation\nFive molecular folds\nInner training-only selection",
    ]
    locations = [(0, 1), (1, 1), (2, 1), (0, 0), (1, 0), (2, 0)]
    for text, (x, y) in zip(texts, locations):
        ax.text(x+.5, y+.52, text, ha="center", va="center", fontsize=10.5,
                linespacing=1.45, bbox=dict(boxstyle="round,pad=.55", fc=SHADE, ec="#91A4AA"))
    for start, end in [((.93,1.52),(1.07,1.52)),((1.93,1.52),(2.07,1.52)),
                       ((2.5,1.17),(2.5,.86)),((2.07,.52),(1.93,.52)),((1.07,.52),(.93,.52))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color=RAW_C))
    fig.text(.5, .005, "BEB shape summaries use the retained matched energy grid, not experimental cross sections.",
             fontsize=10.1, ha="center")
    fig.tight_layout(rect=[0,.035,1,1])
    save(fig, out, FIGURES[1])


def primary_figure(m, v, out):
    d = m["per_task"]["curve"]
    vals = [100*d[k]["primary_molecule_macro_NMAE"] for k in ["raw", "constant", "ml"]]
    b = v["curve_primary_result"]
    ci = b["improvement_percentage_points_95_CI"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    bars = ax.bar(["Raw BEB", "Constant scale", "Residual ML"], vals,
                  color=[RAW_C, CONST_C, ML_C], width=.59)
    label_bars(ax, bars)
    ax.set_ylim(0, max(vals)*1.29)
    ax.set_ylabel("Primary molecule-macro NMAE (%)")
    ax.set_title(f"Energy-dependent curve evaluation: {d['molecules']} molecules")
    ax.grid(axis="y")
    ax.text(.5,.94, f"BEB − ML: {b['absolute_improvement_percentage_points']:+.2f} pp"
            f"  |  conditional 95% CI [{ci[0]:.2f}, {ci[1]:.2f}] pp",
            ha="center", va="top", transform=ax.transAxes, fontsize=10.5)
    fig.text(.5,.012,"Fixed-OOF molecular bootstrap; uncertainty excludes retraining and source clustering.",
             ha="center", fontsize=10.2)
    fig.tight_layout(rect=[0,.045,1,1])
    save(fig,out,FIGURES[2])


def parity_figure(f, m, out):
    fig, axes = plt.subplots(1,2,figsize=(7.2,3.6),sharex=True,sharey=True)
    lo = min(float(f[c].min()) for c in [EXP,RAW,ML])*.75
    hi = max(float(f[c].max()) for c in [EXP,RAW,ML])*1.25
    for ax,col,title,color,key in zip(axes,[RAW,ML],["Raw BEB","Residual ML"],
                                     [RAW_C,ML_C],["raw_metrics","expanded_ml_metrics"]):
        ax.scatter(f[EXP],f[col],s=7,alpha=.24,color=color,edgecolors="none",rasterized=True)
        ax.plot([lo,hi],[lo,hi],"--",color="#555555",lw=1)
        ax.set(xscale="log",yscale="log",xlim=(lo,hi),ylim=(lo,hi))
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nAll-point $R^2$ = {m[key]['R2']:.3f}")
        ax.set_xlabel("Experiment (Å²)")
        ax.grid(alpha=.6)
    axes[0].set_ylabel("Prediction (Å²)")
    fig.text(.5,-.015,f"All {len(f):,} observations shown on logarithmic axes; points within a molecule are correlated.",
             ha="center",fontsize=10.3)
    fig.tight_layout(w_pad=1.7)
    save(fig,out,FIGURES[3])


def point_error_figure(f, out):
    fig,ax = plt.subplots(figsize=(7,4.0))
    for col,label,color in [(RAW,"Raw BEB",RAW_C),(CONST,"Constant scale",CONST_C),(ML,"Residual ML",ML_C)]:
        e = np.sort(100*np.abs(f[col]-f[EXP])/f[EXP])
        ax.plot(e, np.arange(1,len(e)+1)/len(e)*100, lw=1.9,color=color,
                label=f"{label}: median {np.median(e):.2f}%")
    ax.set_xscale("symlog",linthresh=1)
    ax.set(xlabel="Absolute percentage error (%) — linear below 1%, logarithmic above",
           ylabel="Cumulative observations (%)",ylim=(0,101))
    ax.set_title("Point-level error distribution (all evidence tasks)")
    ax.grid(); ax.legend(loc="lower right",frameon=False)
    fig.tight_layout()
    save(fig,out,FIGURES[4])


def scope_figure(m,out):
    fig,axes=plt.subplots(2,1,figsize=(7.1,6.0),gridspec_kw={"height_ratios":[1,1.1]})
    scopes=[("Curve: primary","curve","primary_molecule_macro_NMAE"),
            ("Curve: all energies","curve","all_molecule_macro_NMAE"),
            ("All evidence: primary","all","primary_molecule_macro_NMAE")]
    y=np.arange(3); h=.22
    for offset,k,label,c in [(-h,"raw","Raw BEB",RAW_C),(0,"constant","Constant scale",CONST_C),(h,"ml","Residual ML",ML_C)]:
        vals=[100*(m["per_task"][scope][k] if scope!="all" else m[{"raw":"raw_metrics","constant":"constant_scale_metrics","ml":"expanded_ml_metrics"}[k]])[metric]
              for _,scope,metric in scopes]
        bars=axes[0].barh(y+offset,vals,h,color=c,label=label);label_bars(axes[0],bars)
    axes[0].set_yticks(y,[s[0] for s in scopes]);axes[0].invert_yaxis()
    axes[0].set_title("Evaluation scope");axes[0].grid(axis="x")
    axes[0].legend(loc="upper center",bbox_to_anchor=(.4,1.28),ncol=3,frameon=False)
    axes[0].set_xlim(0,26)
    tasks=["curve","fixed_sparse","peak"]
    names=[f"Curve (n={m['per_task']['curve']['molecules']})",f"Fixed / sparse (n={m['per_task']['fixed_sparse']['molecules']})",f"Peak (n={m['per_task']['peak']['molecules']})"]
    for offset,k,c in [(-.18,"raw",RAW_C),(.18,"ml",ML_C)]:
        vals=[100*m["per_task"][task][k]["primary_molecule_macro_NMAE"] for task in tasks]
        bars=axes[1].barh(y+offset,vals,.33,color=c);label_bars(axes[1],bars)
    axes[1].set_yticks(y,names);axes[1].invert_yaxis();axes[1].grid(axis="x");axes[1].set_xlim(0,27)
    axes[1].set_title("Separate task heads; identities overlap across tasks")
    axes[1].set_xlabel("Molecule-macro NMAE (%) — lower is better")
    fig.tight_layout(h_pad=1.5)
    save(fig,out,FIGURES[5])


def ablation_figure(a, out):
    labels={"raw_beb":"Raw BEB","constant_scale":"Constant scale","direct_log_prediction":"Direct log prediction",
            "residual_energy_beb_only":"Energy + BEB only","residual_no_molecular_composition":"No composition features",
            "residual_no_beb_shape":"No BEB shape features","pooled_no_task_routing":"Pooled, no task routing",
            "full_routed_residual":"Full routed residual"}
    fig,ax=plt.subplots(figsize=(7.2,5.0));y=np.arange(len(a))
    for offset,col,label,c in [(-.18,"curve_primary_macro_NMAE","Curve head",RAW_C),(.18,"all_evidence_primary_macro_NMAE","All evidence",ML_C)]:
        bars=ax.barh(y+offset,100*a[col],.33,color=c,label=label);label_bars(ax,bars)
    ax.set_yticks(y,[labels[x] for x in a.method]);ax.invert_yaxis();ax.grid(axis="x")
    ax.set_xlabel("Primary molecule-macro NMAE (%)")
    ax.set_xlim(0,max(a.curve_primary_macro_NMAE.max(),a.all_evidence_primary_macro_NMAE.max())*123)
    ax.legend(loc="lower right",frameon=False);ax.set_title("Controlled alternatives on fixed molecular folds")
    fig.tight_layout();save(fig,out,FIGURES[6])


def importance_figure(feature,grouped,out):
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.8),gridspec_kw={"width_ratios":[.95,1.05]})
    for ax,data,names in [(axes[0],grouped,"feature_group"),(axes[1],feature.head(9),"feature")]:
        labels=[str(s).replace("BEB curve-shape descriptors","BEB shape").replace("Molecular composition","Composition").replace("Interaction terms","Interactions").replace("Collision energy","Energy").replace("_"," ") for s in data[names]]
        bars=ax.barh(np.arange(len(data)),100*data.relative_importance,color=RAW_C if ax is axes[0] else ML_C)
        ax.set_yticks(np.arange(len(data)),[wrap(s,18) for s in labels]);ax.invert_yaxis();ax.grid(axis="x")
        ax.set_xlabel("Relative weight (%)");ax.set_xlim(0,float(data.relative_importance.max())*125)
        label_bars(ax,bars,"{:.1f}")
    axes[0].set_title("Feature groups");axes[1].set_title("Nine highest weights")
    fig.text(.5,-.006,"Descriptive aggregate of tree importance and |ridge coefficients|; not causal evidence or SHAP.",
             ha="center",fontsize=10.2)
    fig.tight_layout(w_pad=1.4);save(fig,out,FIGURES[7])


def source_figure(batch,v,out):
    fig,ax=plt.subplots(figsize=(7.2,5.3));y=np.arange(len(batch))
    short={"NIST curated curve batch":"NIST curated","QEC priority batch":"QEC priority","Hudson ester family":"Hudson esters",
           "NIST graph-digitised batch":"NIST graph-digitised","Expanded measured batch":"Expanded measured",
           "Heavy-ECP figure batch":"Heavy ECP","SiClx figure batch":"SiClx"}
    for offset,col,label,c in [(-.18,"raw_primary_macro_NMAE","Raw BEB",RAW_C),(.18,"ml_primary_macro_NMAE","Residual ML",ML_C)]:
        bars=ax.barh(y+offset,100*batch[col],.33,color=c,label=label);label_bars(ax,bars)
    ax.set_yticks(y,[f"{short.get(row.held_out_batch,row.held_out_batch)} (n={row.molecules})" for row in batch.itertuples()])
    ax.invert_yaxis();ax.set_xlim(0,max(batch.raw_primary_macro_NMAE.max(),batch.ml_primary_macro_NMAE.max())*127)
    ax.grid(axis="x");ax.set_xlabel("Primary molecule-macro NMAE (%)")
    n=int(batch.molecules.sum()); raw=float(np.average(batch.raw_primary_macro_NMAE,weights=batch.molecules)*100)
    ml=float(np.average(batch.ml_primary_macro_NMAE,weights=batch.molecules)*100)
    ax.set_title(f"Source-ingestion batch holdout ({n} curve molecules)\nWeighted aggregate: {raw:.2f}% → {ml:.2f}%",pad=14)
    ax.legend(loc="upper right",frameon=False)
    fig.text(.5,.005,"Batches are not independent papers; source and chemical-family effects can be confounded.",ha="center",fontsize=10.2)
    fig.tight_layout(rect=[0,.035,1,1]);save(fig,out,FIGURES[8])


def heterogeneity_figure(pm,out):
    gains=pm.improvement_pp
    fig,axes=plt.subplots(2,1,figsize=(7.1,6.8),gridspec_kw={"height_ratios":[.65,1.35]})
    axes[0].hist(gains,bins=17,color=RAW_C,edgecolor="white")
    axes[0].axvline(0,color=BAD,lw=1.2,ls="--");axes[0].grid(axis="y")
    axes[0].set(xlabel="BEB NMAE − ML NMAE (percentage points)",ylabel="Molecules")
    axes[0].set_title(f"{int((gains>0).sum())} improve, {int((gains<0).sum())} worsen; {len(pm)} curve molecules")
    ranked=pm.sort_values("improvement_pp")
    show=pd.concat([ranked.head(5),ranked.tail(5)]).sort_values("improvement_pp")
    bars=axes[1].barh(np.arange(len(show)),show.improvement_pp,color=[GOOD if x>0 else BAD for x in show.improvement_pp])
    axes[1].set_yticks(np.arange(len(show)),[wrap(x,26) for x in show.molecule_name]);axes[1].axvline(0,color="#888888",lw=.8)
    label_bars(axes[1],bars,"{:+.2f}");axes[1].set_xlim(float(gains.min())*1.23,float(gains.max())*1.30)
    axes[1].grid(axis="x");axes[1].set_xlabel("Change in primary NMAE (pp); positive = improvement")
    axes[1].set_title("Five largest improvements and five largest deteriorations")
    fig.tight_layout(h_pad=1.5);save(fig,out,FIGURES[9])


def draw_case(ax,f,identity,pm):
    g=f.loc[(f.identity_key==identity)&f.task_family.eq("curve")].copy().sort_values("energy_eV")
    if g.empty:
        raise ValueError(f"Case {identity} has no retained curve evidence")
    p=primary(g);rec=pm.set_index("identity_key").loc[identity]
    ax.axvspan(float(g.energy_eV.min()),float(p.energy_eV.max()),color=SHADE,zorder=0)
    # Same-energy theoretical predictions are averaged only for display; metrics use every row.
    line=g.groupby("energy_eV",sort=True)[[RAW,ML]].median()
    ax.plot(line.index,line[RAW],color=RAW_C,lw=1.8,label="Raw BEB")
    ax.plot(line.index,line[ML],color=ML_C,lw=1.8,label="Residual ML")
    if identity == "ozone":
        covered=set()
        for (label,lines),marker,color in zip(OZONE_SERIES.items(),["o","^"],[INK,"#6B5A8D"]):
            selected=g.original_csv_line.isin(lines)
            covered.update(g.loc[selected,"original_csv_line"].astype(int))
            ax.scatter(g.loc[selected,"energy_eV"],g.loc[selected,EXP],color=color,s=19,
                       marker=marker,zorder=4,label=label)
        if covered != set(g.original_csv_line.astype(int)):
            raise ValueError("Ozone contains a series outside the two audited original experiments")
        ax.legend(loc="lower right",fontsize=8.8,frameon=False,
                  handles=ax.get_legend_handles_labels()[0][2:])
    else:
        ax.scatter(g.energy_eV,g[EXP],color=INK,s=14,zorder=4,label="Experiment")
    u=pd.to_numeric(g.experimental_uncertainty_A2,errors="coerce")
    good=u.notna()&(u>0)
    if good.any():
        ax.errorbar(g.loc[good,"energy_eV"],g.loc[good,EXP],yerr=u[good],fmt="none",
                    ecolor="#879097",elinewidth=.65,alpha=.75,zorder=3)
    ax.set_xscale("log")
    lo,hi=float(g.energy_eV.min()),float(g.energy_eV.max())
    candidates=np.array([1,2,5,10,20,50,100,200,500,1000,2000,5000,10000],float)
    ticks=candidates[(candidates>=lo*.98)&(candidates<=hi*1.02)]
    if len(ticks)>5:
        ticks=ticks[::int(np.ceil(len(ticks)/5))]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f"{value:g}"));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Electron energy (eV, log scale)")
    ax.set_ylabel("Ionisation cross section (Å²)")
    ax.set_ylim(bottom=0);ax.grid()
    ax.set_title(f"{wrap(rec.molecule_name,30)} ({rec.formula})\nNMAE: {rec.raw_primary_nmae_percent:.2f}% → {rec.ml_primary_nmae_percent:.2f}%",fontsize=11.5)
    return rec.to_dict()


def cases_figure(f,pm,identities,name,out,headline):
    fig,axes=plt.subplots(1,len(identities),figsize=(7.2,3.8 if len(identities)>1 else 4.4),squeeze=False)
    records=[]
    for ax,ident in zip(axes[0],identities):
        records.append(draw_case(ax,f,ident,pm))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.015),ncol=3,frameon=False)
    if len(identities)>1:
        axes[0,1].set_ylabel("")
    fig.suptitle(headline,y=1.015,fontsize=12.5)
    fig.tight_layout(rect=[0,.085,1,.985],w_pad=1.3)
    save(fig,out,name)
    return records


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve()
    model=root/"outputs/final_expanded_194_ml_2026_08_02"
    validity=root/"outputs/final_validity_ablation_corrected_2026_08_05"
    required={"oof":model/"expanded_194_oof_predictions.csv","metrics":model/"metrics.json",
              "validity":validity/"validity_ablation_results.json","ablations":validity/"ablation_metrics.csv",
              "ablation_bootstrap":validity/"ablation_paired_bootstrap.csv",
              "batch":validity/"leave_one_source_batch_out.csv","feature":validity/"feature_importance.csv",
              "feature_group":validity/"feature_group_importance.csv"}
    for path in required.values():
        if not path.is_file(): raise FileNotFoundError(path)
    f=pd.read_csv(required["oof"])
    f["primary_low_energy"]=f.primary_low_energy.astype(str).str.lower().isin(["true","1","yes"])
    m=json.loads(required["metrics"].read_text(encoding="utf-8"))
    v=json.loads(required["validity"].read_text(encoding="utf-8"))
    curves=f.loc[f.task_family.eq("curve")].copy()
    assert len(f)==m["observations"],"OOF and metrics observation counts disagree"
    assert f.identity_key.nunique()==m["physical_molecules"]
    assert curves.identity_key.nunique()==m["curve_molecules"]
    for task in ["curve","fixed_sparse","peak"]:
        assert len(f.loc[f.task_family.eq(task)])==m["per_task"][task]["rows"]
    assert "original_row_sha256" in f and "correction_note" in f,"Expected provenance-preserving corrected release"
    assert (f[[EXP,RAW,CONST,ML]]>0).all().all(),"Plots expect retained positive data"
    pm=molecule_errors(curves)
    np.testing.assert_allclose(pm.raw_primary_nmae_percent.mean(),100*m["per_task"]["curve"]["raw"]["primary_molecule_macro_NMAE"],rtol=1e-10)
    np.testing.assert_allclose(pm.ml_primary_nmae_percent.mean(),100*m["per_task"]["curve"]["ml"]["primary_molecule_macro_NMAE"],rtol=1e-10)
    out=root/"figures";out.mkdir(parents=True,exist_ok=True)
    initialise_style()
    dataset_figure(f,out);workflow_figure(f,out);primary_figure(m,v,out);parity_figure(f,m,out)
    point_error_figure(f,out);scope_figure(m,out)
    a=pd.read_csv(required["ablations"]);b=pd.read_csv(required["batch"])
    fe=pd.read_csv(required["feature"]);fg=pd.read_csv(required["feature_group"])
    ablation_figure(a,out);importance_figure(fe,fg,out);source_figure(b,v,out);heterogeneity_figure(pm,out)
    classic=cases_figure(f,pm,["methane","ethylene"],FIGURES[10],out,"Classic small-molecule benchmarks")
    # Descriptive extreme-case selection occurs after evaluation and is explicitly labelled.
    best=pm.sort_values(["improvement_pp","identity_key"],ascending=[False,True]).head(2).identity_key.tolist()
    good=cases_figure(f,pm,best,FIGURES[11],out,"Two largest observed improvements (retained curve set)")
    warnings=cases_figure(f,pm,["carbon-monosulfide","ozone"],FIGURES[12],out,"Source-sensitive and conflicting-evidence examples")
    worst=pm.sort_values(["improvement_pp","identity_key"]).iloc[0].identity_key
    bad=cases_figure(f,pm,[worst],FIGURES[13],out,"Largest observed deterioration (retained curve set)")
    summary={
        "plotting_environment":{"python":platform.python_version(),"matplotlib":matplotlib.__version__,
                                "numpy":np.__version__,"pandas":pd.__version__},
        "scope":{"identities":int(f.identity_key.nunique()),"observations":len(f),
                 "curve_molecules":int(curves.identity_key.nunique()),"curve_rows":len(curves),
                 "fixed_sparse_molecules":int(m["per_task"]["fixed_sparse"]["molecules"]),
                 "fixed_sparse_rows":int(m["per_task"]["fixed_sparse"]["rows"]),
                 "peak_molecules":int(m["per_task"]["peak"]["molecules"]),"peak_rows":int(m["per_task"]["peak"]["rows"])},
        "inputs":{k:{"path":str(p.relative_to(root)).replace("\\","/"),"sha256":sha256(p)} for k,p in required.items()},
        "primary_curve_result":v["curve_primary_result"],
        "all_evidence_result":{"raw":m["raw_metrics"],"constant":m["constant_scale_metrics"],"ml":m["expanded_ml_metrics"]},
        "per_task":m["per_task"],"source_batch_summary":v["source_batch_summary"],
        "source_batch_results":b.to_dict("records"),"ablation_results":a.to_dict("records"),
        "ablation_paired_bootstrap":pd.read_csv(required["ablation_bootstrap"]).to_dict("records"),
        "feature_group_importance":fg.to_dict("records"),"top_features":fe.head(9).to_dict("records"),
        "molecular_heterogeneity":{"improved":int((pm.improvement_pp>0).sum()),"worsened":int((pm.improvement_pp<0).sum()),
                                    "unchanged":int((pm.improvement_pp==0).sum())},
        "case_selection":{"classic":{"reason":"Previously reported common benchmarks, retained regardless of outcome","cases":classic},
                          "strong":{"reason":f"Two largest positive primary-NMAE differences among {len(pm)} retained curve identities; both same ester family, not proof of new-family generalisation","cases":good},
                          "warnings":{"reason":"Retained source-sensitive and conflicting-branch diagnostic examples; not selected for improvement","cases":warnings},
                          "failure":{"reason":f"Largest negative primary-NMAE difference among {len(pm)} retained curve identities","cases":bad}},
        "plot_notes":["Only corrected-release OOF and validity outputs are read.",
                      "NMAE values are recomputed per identity and checked against metrics.json.",
                      "No new fits, interpolation or predictions are used for displayed curves; display joins median predictions at matching energies.",
                      "Case-panel NMAE refers to the primary region. Shading denotes its retained energy range; error bars are available reported uncertainties, not confidence bands.",
                      "All points are displayed in parity and CDF plots; no outlier clipping.",
                      "Feature weights are descriptive model internals, not SHAP or causal importance.",
                      "PDF/DOCX insertion should preserve aspect ratio and use about 6.5--6.7 inch width."],
        "figures":[str((out/(name+".png")).relative_to(root)).replace("\\","/") for name in FIGURES],
        "caption_templates":{
            FIGURES[0]:"Composition of retained paired evidence. Molecule counts overlap across curve, fixed/sparse and peak tasks; they must not be added to obtain the identity total.",
            FIGURES[1]:"Data audit, feature construction and molecularly separated nested validation workflow. The three task heads are fitted independently. Shape descriptors use theoretical cross sections on the retained matched sampling grid.",
            FIGURES[2]:"Primary-region molecule-macro NMAE for energy-dependent curve evidence. The reported interval is a conditional paired molecular bootstrap of fixed OOF predictions, not a confidence interval accounting for refitting or source clustering.",
            FIGURES[3]:"Predicted versus measured cross sections across all retained evidence tasks on logarithmic axes. The dashed line denotes exact agreement. All-point R-squared is descriptive and does not turn correlated energy points into independent samples.",
            FIGURES[4]:"Empirical cumulative distributions of pointwise absolute percentage error across all retained observations. The horizontal scale is linear below 1% and logarithmic above 1%; no outliers are clipped. A leftward curve indicates lower errors.",
            FIGURES[5]:"Performance under different aggregation scopes and for the independent evidence-task heads. The all-evidence result mixes task types and is not an estimate of complete-curve prediction for every identity.",
            FIGURES[6]:"Controlled alternatives on the fixed molecular folds. Bars show observed primary-region molecule-macro NMAE. Small differences must be interpreted together with the paired bootstrap intervals and adaptive model-development limitations.",
            FIGURES[7]:"Descriptive relative feature weights aggregated from tree importance and absolute ridge coefficients. These are model-internal summaries, not SHAP, causal effects or proof that one feature group is necessary.",
            FIGURES[8]:"Leave-one-source-ingestion-batch-out stress test. The title gives the molecule-count-weighted aggregate. These groups mix source and ingestion history; the ester batch additionally changes chemical family. The standalone one-molecule batch is excluded from this stress test.",
            FIGURES[9]:"Distribution of per-molecule primary-NMAE improvement, with the five largest positive and negative differences shown. Positive values indicate improvement over raw BEB. Extreme cases are descriptive examples selected after evaluation.",
            FIGURES[10]:"Methane and ethene benchmark curves, retained regardless of the outcome. Panel NMAE refers to the shaded primary region. Lines join retained OOF theoretical predictions; symbols and available error bars represent experimental observations.",
            FIGURES[11]:"The two largest observed primary-NMAE improvements in the retained curve set. Both belong to the ester family, so these cases illustrate within-domain gains rather than unseen-family transfer. Shading and line conventions are as in the benchmark figure.",
            FIGURES[12]:"Source-sensitive comparisons for carbon monosulfide and ozone. The ozone symbols distinguish the two original measurements, Newson et al. (1995) and Siegel (1982); the duplicated renormalised variant is excluded. Both retained ozone series remain substantially below the predictions.",
            FIGURES[13]:"The largest observed deterioration in the retained curve set. The shaded region defines the primary NMAE. The ML correction increases the discrepancy, demonstrating that correction is not uniformly beneficial.",
        },
    }
    (out/"figure_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    pm.assign(sources=pm.sources.map(json.dumps)).to_csv(out/"per_molecule_curve_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(classic+good+warnings+bad).assign(sources=lambda d:d.sources.map(json.dumps)).to_csv(out/"representative_case_metrics.csv",index=False,encoding="utf-8-sig")
    print(json.dumps({"figures":len(FIGURES),"directory":str(out),"summary":str(out/"figure_summary.json"),
                      "scope":summary["scope"],"curve_result":summary["primary_curve_result"]},indent=2))


if __name__=="__main__":
    main()
