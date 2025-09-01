# UV-Fe Analysis

This repository contains tools to analyze **microbial survival under UVC irradiation in the presence of iron (Fe³⁺)**.  
The project is an extension of experiments comparing microorganisms exposed to UVC with iron in **solution** and on **desiccated (solid) iron layers**.

The analysis reproduces the statistical rigor of the original study:
- Dose–response fitting with the **Brain-Cousens hormesis model**.
- Estimation of **LD₁₀, LD₅₀, LD₉₀** with **bootstrap confidence intervals**.
- Model comparison (hormetic vs. non-hormetic).
- Growth curve fitting (logistic/Baranyi approximation).
- Handling of **presence/absence data** (Fisher exact test, Clopper–Pearson intervals).
- Publication-ready plots.

---

## 🚀 How to use

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/uv-fe-analysis.git
cd uv-fe-analysis
````

### 2. Create and activate a virtual environment

Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare your input data

A CSV file with at least these columns:

```
sample_id, organism, treatment, fluence_Jm2, replicate, CFU
```

* `fluence_Jm2`: UVC dose
* `CFU`: colony forming units (0 if no growth)
* `treatment`: e.g. control, 5mM Fe³⁺, Fe-solid

Place your file under the `data/` folder.

### 5. Run the analysis

```bash
python uv_fe_analysis.py --csv data/my_data.csv --out results
```

---

## 📂 Repository structure

```
uv-fe-analysis/
│
├── uv_fe_analysis.py     # Main analysis script
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── data/                 # Input CSV files
└── results/              # Output (plots, tables, summaries)
```

---

## 📊 Outputs

The script creates the following in the `results/` folder:

* **Dose–response plots**: `viab_<organism>_<treatment>.png`
* **Forest plot of LD₅₀**: `forest_ld50.png`
* **Summary table**: `ld50_summary.json` with LD₅₀ values and CI95%

---

## ✨ Notes

* Results help evaluate whether iron (in solution or desiccated form) protects or kills cells under UVC.
* Negative results (no growth detected) are statistically handled using **confidence intervals for proportions**.
* The code is modular and can be extended with additional models or statistical tests.

---

## 📜 License

MIT – feel free to use and adapt.

```