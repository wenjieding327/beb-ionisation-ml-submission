# Evidence-corrected BEB residual learning release

This release supersedes `submission-2026-09-06`; use the corrected `submission-2026-09-06-r2` tag for submission. The earlier tag is retained as history, not as the final result.

This is the 6 September 2026 corrected numerical release for the MSc project on electron-impact ionisation cross sections. BEB means **Binary-Encounter-Bethe**. The aim is to test whether a bounded machine-learning correction improves agreement between an existing BEB baseline and published experimental observations, with particular attention to the observed onset-to-peak region.

## Scope and result

There are **194 distinct molecular identities and 3,533 retained observations**: 3,316 energy-dependent curve observations from 74 identities, 150 fixed/sparse-energy observations from 144 identities, and 67 peak-height observations from 67 identities. Evidence classes overlap in identity. The three model heads are trained separately; 194 must not be described as 194 complete experimental curves.

The original five outer molecular folds, 49 input features, candidate models, random-seed protocol and model-selection rules were retained. The corrected data were not selected to improve model performance.

| Evaluation | Raw BEB NMAE | Routed residual ML NMAE |
|---|---:|---:|
| Curve task, 74 held-out identities | 15.2764% | 14.0230% |
| Source-ingestion-batch holdout, 73 curve identities | 15.3781% | 17.3815% |
| All evidence types, 194 identities | 19.7445% | 12.6591% |

The primary curve improvement is **1.2535 percentage points (8.2052% relative)**. Its conditional molecule-bootstrap 95% interval is **-0.4187 to +2.8156 percentage points**; it crosses zero. Therefore this release does **not** establish a statistically robust curve improvement. Whole-source-batch prediction worsens overall by 2.0034 percentage points. The mixed-evidence result is supplementary, not a complete-curve result. Bootstrap resampling uses fixed OOF predictions and does not capture refitting, fold-selection or source-cluster uncertainty.

## Installation

The numerical run was verified with Python 3.12.14 on Windows 11, using the exact versions in `requirements.txt`. Create a separate environment; no system installation is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python tools/verify_release.py
```

On Linux/macOS use `.venv/bin/python` instead of `.\.venv\Scripts\python`. The verifier and independent metric audit use only the Python standard library; full training requires the numerical dependencies.

## Reproduce the numerical results

Run the following from this release root. The evidence builder, audit and verifier resolve their root from their own script location. The training and validity scripts also accept `BEB_WORKSPACE_ROOT` for an explicit alternative output workspace. For a full isolated replay, copy the release to another directory and run that copy's scripts.

```powershell
.\.venv\Scripts\python tools/build_corrected_inputs.py --check-only
.\.venv\Scripts\python tools/build_corrected_inputs.py
.\.venv\Scripts\python tools/train_expanded_194_nested_model.py
.\.venv\Scripts\python tools/run_validity_ablation_studies_194.py
.\.venv\Scripts\python tools/audit_corrected_results.py
.\.venv\Scripts\python tools/build_verified_paper_figures.py --root .
.\.venv\Scripts\python tools/verify_release.py --skip-checksums
```

The frozen CSV already contains the required paired sample inputs. `build_corrected_inputs.py` also rebuilds it from the packaged original 3,883 rows and explicit source-correction manifests, with a byte-identical `--check-only` mode. A QEC or Molpro installation is not needed to reproduce this ML stage. QEC (Quantemol Electron Collisions) was used upstream for part of the BEB baseline. The complete source ledger records experimental and BEB provenance. Full raw-PDF-to-numerical extraction is not claimed to be reproducible from this package alone. `${ARCHIVE_ROOT}` is a historical source locator, not a runtime dependency for the commands above.

On the audited 24-logical-CPU workstation, nested training took about 69 seconds and the validity/ablation suite about 132 seconds, excluding environment installation and figures. Runtime varies by machine. A fresh run rewrites outputs and includes a new elapsed runtime, so byte-level checksums will change even when numerical results agree; use `--skip-checksums` after rerunning. The default verifier checks the untouched frozen deliverable.

A separate-directory local replay rebuilt the input from the packaged original rows and reran nested training plus the full validity/ablation suite. All 974 non-runtime JSON values and 81,259 OOF cells agreed within a 1e-12 tolerance; the maximum numerical cell difference was 7.11e-15. The input was byte-identical. Details are in `audit/clean_path_replay.json`. This is a same-environment local reproducibility check, not replication by an independent laboratory. To compare another full replay, run `python tools/compare_clean_replay.py <other-release-directory>`.

## Files

- `outputs/expanded_194_training_2026_08_02/expanded_194_training_points.csv`: corrected input, with original row locators and original folds. Historical directory dates are retained for compatibility; the input is the September corrected dataset.
- `outputs/final_expanded_194_ml_2026_08_02/`: corrected OOF predictions, nested-fold configurations and scores, final fitted model, and `metrics.json`.
- `outputs/final_validity_ablation_corrected_2026_08_05/`: curve bootstrap, ablations, source-batch holdout, sensitivities, feature summaries and fold compositions.
- `figures/`: regenerated report figures. No legacy figure is evidence for corrected numerical results.
- `data/source_evidence_ledger_194.csv`: experimental source, BEB source, page/figure/table locator, evidence status and limitations for each identity.
- `data/final_194_molecule_provenance.csv`: concise identity/provenance table.
- `audit/correction_scope.json`: final corrections, counts and boundary policy.
- `audit/quarantined_rows.csv`: 350 excluded original rows with explicit reasons.
- `audit/peak_metadata_corrections.csv`: 15 corrected peak-energy/uncertainty records.
- `audit/late_source_corrections.json`: 52 row-specific source patches, including 51 common-energy removals, the nitromethane task correction and six original-table amplitude replacements.
- `audit/experimental_source_replacements.json`: automatically derived audit of the six experimental amplitude and uncertainty replacements; all BEB amplitudes remain unchanged.
- `audit/reported_peak_energy_metadata.csv`: 27 source-reported experimental peak energies, retained as provenance only and not read by the feature matrix.
- `audit/bose125_numeric_carrier_column_audit.json`: row-wise comparison of 125 retained observations with the experimental mean column of the numerical compilation.
- `audit/primary_source_spotcheck_final_20260906.txt`: the scope and limits of the final primary-source and numerical-carrier checks; this is not a claim that every experiment was independently reconstructed.
- `audit/legacy_r2/` and `audit/legacy_v2/`: earlier inputs and result snapshots for audit only; not current training data or current performance claims. A historical machine locator in one legacy JSON is normalised, with its original hash retained in a separate audit record; historical metrics are unchanged.
- `audit/independent_metric_recalculation.json`: independent standard-library recomputation from OOF CSV.
- `audit/clean_path_replay.json`: numerical agreement of the full separate-directory replay, with runtime excluded from comparison.
- `audit/per_molecule_corrected_curve_metrics.csv`: all 74 curve-case results, including failures.
- `checksums_sha256.csv`: frozen file inventory with sizes and SHA-256 checksums.

## Evidence correction

From the earlier 3,883-row input, the audit withdrew seven water values that were theoretical rather than experimental, 330 curve rows from six ambiguous ester isomer traces, and 13 renormalised ozone values that duplicated the original Newson series. The six esters retain their separately supported fixed-energy and published table peak-height constraints. Ozone retains 10 original Newson and nine Siegel observations. Its evaluation peak boundary was recomputed from those retained observations.

All 15 Hudson ester peak heights were checked against Table 1 (journal p. 45, PDF p. 4). The table provides no peak energy, so those energies are blank with an explicit missingness flag; uncertainty uses the authors' reported 5% absolute accuracy. Nine retained ester curves are **peak-anchored figure-trace extractions**, not independent author-supplied numerical tables. Their dependence on the table peak and extraction procedure is retained as a limitation.

The final energy-field audit clears another 51 finite peak-record coordinates. Some were backfilled theoretical peak energies; others were genuine experimental peak energies paired with a BEB maximum at a different energy. Thus **all 67 peak-to-peak amplitude pairs have no shared model-energy coordinate**. The `energy_is_reported` field denotes whether a common energy is assigned to that pair; it does not deny that a publication may report a separate experimental peak energy. Twenty-seven such values are preserved outside the feature input. Missing model energy follows the original 72.5 eV numerical imputation plus missingness flag. Nitromethane is instead routed as a genuine fixed-energy pair at 60 eV (experiment 6.8 A², BEB 5.95 A²).

A uniform comparison of 15 related Bart peak records found six legacy experimental amplitudes that did not match the directly readable original table. Those six are explicitly replaced by Bart's Table 3.4 values (PDF p. 62, printed p. 43); their uncertainty fields use the documented 3.9% instrumental bound from Table 3.2 (PDF p. 56, printed p. 37), not a standard deviation. Nine values are unchanged. Before/after amplitudes and source locators are preserved in the audit manifests; this is a disclosed source replacement, not a claim that all original values were already correct.

For 125 fixed-energy observations, the compilation's actual Table A.7 experimental mean column was replayed and matched all 125 retained values, with zero discrepancies. These are not the BEB or ML prediction columns. This is numerical-carrier transcription verification, not an independent reconstruction of every original laboratory's standardisation or uncertainty analysis.

Every retained identity keeps its original outer fold. Sample weights are recomputed for the corrected task data. There is no post-audit search for a more favourable headline model. The observed pooled ablation is reported as an ablation, not silently promoted to replace the frozen routed model.

## Method and code structure

`train_expanded_194_nested_model.py` implements task routing, deterministic three-fold inner selection, five molecular outer folds, bounded log-residual correction and final fitted heads. It imports shared feature and metric functions from `train_final_beb_residual_model.py` and `improve_final_beb_model.py`. `run_validity_ablation_studies_194.py` evaluates the unchanged alternatives and stress tests. Source/chemistry-correlated samples may remain within ordinary molecular folds; nested molecular CV does not prove new-source or new-family generalisation.

`audit/legacy_r2/build_expanded_194_training_dataset.py` is a historical assembly snapshot. It requires the upstream collection archive and additional reconstruction modules, is not called by the corrected training pipeline, and must not be used to overwrite this corrected input. `synchronize_corrected_metadata.py` is an audit-construction helper, not a routine training step. The authoritative reproducible starting point is the included corrected input CSV.

Never load a `.joblib` file from an untrusted source: it can execute Python code. To avoid using the supplied model binary, run training to generate it locally. The supplied numerical outputs can be inspected without loading the binary.
