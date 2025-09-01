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
### Example CSV

```csv
sample_id,organism,treatment,fluence_Jm2,replicate,dilution_log,colonies,plated_uL
1,E.coli,control,0,1,-5,29,10
2,E.coli,control,0,1,-5,29,10
3,E.coli,control,0,1,-5,21,10
4,E.coli,Fe10mM,0,1,-5,0,10
```


A CSV file with **these required columns**:

```
sample_id,organism,treatment,fluence_Jm2,replicate,dilution_log,colonies,plated_uL
```

* `colonies`: number of colonies counted (0 if no growth)
* `dilution_log`: log of dilution (e.g. -3 for 10⁻³, -5 for 10⁻⁵)
* `plated_uL`: plated volume in µL (typically 10 µL per drop)
* `fluence_Jm2`: UVC dose
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


* The raw data are entered as colony counts (`colonies`) obtained from drop-plating of serial dilutions (`dilution_log`). The script automatically computes **CFU/mL** according to the formula:

	$$
	CFU/mL = \frac{\text{colonies}}{\text{plated volume in mL}} \times 10^{|\text{dilution_log}|}
	$$

	Example: 30 colonies at dilution 10⁻⁵ with 10 µL plated = 3.0 × 10⁸ CFU/mL.

* All analyses, plots and statistics use `CFU_per_mL` (not raw colony counts).
* If `colonies=0`, the value of `CFU_per_mL` will be zero and handled correctly in all calculations.
* Results help evaluate whether iron (in solution or desiccated form) protects or kills cells under UVC.
* Negative results (no growth detected) are statistically handled using **confidence intervals for proportions**.
* The code is modular and can be extended with additional models or statistical tests.

---

## 📜 License

MIT – feel free to use and adapt.

```