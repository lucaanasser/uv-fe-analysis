## UV & Fe Microbial Resistance Analysis

Created by Luca Marinho Nasser (Molecular Sciences, University of São Paulo – USP) during a one-year undergraduate research project at the Astrobiology Laboratory (IQ-USP).
Although the project included both benchwork (microbial culturing, Fe³⁺ stress assays, UVC irradiation, plating) and computational analysis, this repository focuses on the analysis code, ensuring full reproducibility of results.
It also hosts the final report and laboratory notebook, documenting the broader investigation into microbial resilience under oxidative and photonic stress.

Contact: lucaanasser@gmail.com  
License: MIT (see `LICENSE`).

### 1. Purpose
This repository provides a reproducible analysis pipeline for two microbiological resistance experiments:

1. **Fe³⁺ (ferric ion) challenge** – compares a single treatment against a control, quantifying viability, log₁₀ reduction, and detection limits.
2. **UVC irradiation inactivation** – builds survival curves across multiple doses, estimating kinetic parameters (log‑linear slope, D10, LD50), Weibull model shape/scale, and uncertainty via bootstrap.

The code standardizes calculations (CFU/mL, viability, survival fraction), handles censored zero‑colony plates (rule of three), and exports clean summary + supplemental tables and publication‑ready figures.

### 2. Main Features
* Automatic CFU/mL computation from colony counts, dilution (log10) and plated volume (µL).
* Viability per replicate and bootstrap 95% CI (Fe³⁺ experiment).
* Log₁₀ reduction with detection‑limit based lower bounds when treatment plates are all zero.
* Survival fraction & log₁₀ survival relative to dose 0 (UVC experiment).
* Log‑linear (through origin) model: slope, SE, adjusted R², AIC, D10, LD50.
* Weighted least squares variant using approximate Poisson variance.
* Weibull model linearization (δ, p parameters).
* Mixed‑effects (random intercept) slope (if ≥2 experiments present).
* Bootstrap confidence intervals (slopes, D10, LD50, dose‑wise means, viability, survival).
* Rule of three upper bounds when all plates at a condition are zero.
* Publication figures: Fe³⁺ bar log plot; UVC survival curve with band; CFU vs dose.
* JSON + CSV exports (main & supplemental) for downstream reporting.

### 3. Data Input Format
Provide CSV files with the columns below (lowercase headers). Additional columns are ignored unless used (e.g., `experiment`).

| Fe³⁺ Analysis (single treatment) | UVC Inactivation |
| -------------------------------- | ---------------- |
| experiment                       | experiment       |
| organism                         | organism         |
| replicate                        | replicate        |
| treatment (e.g. control / Fe3+)  | dose_J_m2        |
| colonies                         | colonies         |
| dilution_log (log10)             | dilution_log     |
| plated_uL                        | plated_uL        |

Notes:
* `colonies` = counted colonies on the plate (integer ≥0).
* `dilution_log` = log10 of the dilution factor (e.g. 3 for 10⁻³ plated aliquot).
* `plated_uL` = plated volume in microlitres.
* For Fe³⁺, exactly one treatment different from the control is assumed per organism.
* Dose 0 rows are required for UVC to define baseline survival.

### 4. Installation
Clone the repository and install dependencies (Python ≥3.10 recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Running the Analyses
Fe³⁺ resistance (outputs to `results_fe/` by default):
```bash
python fe_analysis.py --csv data/fe_resistance_data.csv --out results_fe
```

UVC survival (outputs to `results_uv/` by default):
```bash
python uvc_analysis.py --csv data/uvc_resistance_data.csv --out results_uv
```

Both commands generate JSON + CSV summaries and figures (PNG) per organism.

### 6. Output Overview
Fe (`results_fe/`):
* `fe_results_main.(csv|json)` – condensed per organism (log₁₀ reduction, means, viability CI or upper bound if all zero).
* `fe_results_supplemental.(csv|json)` – full dataclass fields.
* `<organism>_fe_resistance.png` – log bar plot (hatch + label when no colonies).

UVC (`results_uv/`):
* `uvc_results_main.(csv|json)` – organism-level kinetic & model parameters (slopes, D10, LD50, Weibull, mixed model, detection limit flags).
* `uvc_results_supplemental.(csv|json)` – dose-wise survival stats & log reductions.
* `<organism>_uvc_survival.png` – log₁₀ survival vs dose with log-linear fit + 95% bootstrap band.
* `<organism>_uvc_cfu.png` – CFU/mL (log scale) vs dose.
* `<organism>_weibull_params.json` – δ and p estimates.

### 7. Statistical Notes
* Bootstrap: percentile CI (2.5%, 97.5%) with fixed RNG seed for reproducibility.
* Detection limit: computed assuming 1 colony at the lowest plated dilution/volume combination present for the organism/condition.
* Rule of three: upper 95% ≈ 3/n when all plates are zero (used for viability or survival upper bounds).
* Log‑linear model forced through origin (log₁₀ survival at dose 0 = 0).
* D10 = dose achieving 1 log reduction (−1/slope). LD50 uses log10(0.5)/slope if slope < 0.

### 8. Reproducibility
Random seeds are fixed (NumPy `default_rng(42)`) across bootstraps. Changing seeds requires editing the code (CLI parameterization can be added if needed).

### 9. Extending
Ideas:
* Add CLI options for number of bootstrap replicates.
* Parallelize heavy bootstraps (multiprocessing / joblib) for larger datasets.
* Add model comparison (AIC between Weibull vs log-linear with full likelihood).
* Integrate Cook’s distance diagnostics export.

### 10. Citation / Acknowledgment
If you use this code in academic work, please cite:  
“Luca Marinho Nasser, Astrobiology Laboratory, IQ USP – UV & Fe microbial resistance analysis toolkit (2025).”

### 11. License
MIT – free to use, modify, and distribute with attribution (see `LICENSE`).

---
For questions or collaboration: lucaanasser@gmail.com

