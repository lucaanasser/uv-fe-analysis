import os
import json
import argparse
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.io_utils import load_fe_data, compute_cfu
from analysis.stat_utils import bootstrap as generic_bootstrap, rule_of_three_zero_events
from analysis.results import OrganismResult
from analysis.plots_fe import plot_cfu_comprehensive
from analysis.export_utils import (
	ensure_dir,
	write_json,
	write_csv,
)


# =============================================================
# Core utility functions
# =============================================================

"""Fe³⁺ analysis routine using shared utilities.

Provides:
1. Viability calculation and per-organism statistics.
2. Bootstrap of treatment/control ratios.
3. Generation of log10 CFU plots.
4. Export of detailed and summary results.
"""

# Backwards compatible loader alias
def load_data(csv_path: str) -> pd.DataFrame:  # noqa: D401
	"""Load Fe data from CSV and compute CFU_per_mL."""
	return load_fe_data(csv_path)


def summarize_controls(df: pd.DataFrame, control_label: str = "control") -> pd.DataFrame:
	"""Return mean control CFU per organism and replicate."""
	controls = df[df["treatment"] == control_label]
	n0 = (
		controls.groupby(["organism", "replicate"]).agg(N0_CFU=("CFU_per_mL", "mean")).reset_index()
	)
	return n0


def compute_viability(df: pd.DataFrame, control_label: str = "control") -> pd.DataFrame:
	"""Compute viability (treatment CFU / control CFU) per row."""
	n0 = summarize_controls(df, control_label)
	out = df.merge(n0, on=["organism", "replicate"], how="left")
	out["viability"] = out["CFU_per_mL"] / out["N0_CFU"]
	out.loc[out["N0_CFU"] <= 0, "viability"] = np.nan
	return out

# =============================================================
# Bootstrap / confidence intervals
# =============================================================

def bootstrap_mean_ratio(values, n_boot: int = 5000, ci: float = 95, random_state: int = 42):
	"""Bootstrap the mean of ratio values returning mean and CI."""
	res = generic_bootstrap(values, func=np.mean, n_boot=n_boot, ci=ci, random_state=random_state)
	return {"mean": res["point"], "ci_low": res["ci_low"], "ci_high": res["ci_high"]}


## rule_of_three_zero_events importado diretamente de stat_utils

# =============================================================
# Plotting
# =============================================================

## plot_cfu_comprehensive agora importado de analysis_common.plots_fe


    
    
# =============================================================
# Result data structure
# =============================================================

## OrganismResult agora centralizado em analysis_common.results

# =============================================================
# Main analysis routine
# =============================================================

def analyze_fe_experiment(df: pd.DataFrame, outdir: str = "results_fe", control_label: str = "control"):
	"""Run full Fe analysis pipeline and write tables + plots."""
	ensure_dir(outdir)
	df = compute_viability(df, control_label=control_label)
	organisms = df["organism"].unique()
	results: list[OrganismResult] = []

	for org in organisms:
		org_df = df[df["organism"] == org].copy()
		treatments = [t for t in org_df["treatment"].unique() if t != control_label]
		if not treatments:
			continue
		treatment_label = treatments[0]  # single treatment assumed (Fe3+)
		ctrl = org_df[org_df["treatment"] == control_label]
		trt = org_df[org_df["treatment"] == treatment_label]

		rep_ctrl = ctrl.groupby("replicate")["CFU_per_mL"].mean().rename("CFU_ctrl")
		rep_trt = trt.groupby("replicate")["CFU_per_mL"].mean().rename("CFU_trt")
		rep_df = pd.concat([rep_ctrl, rep_trt], axis=1)
		rep_df["viab_rep"] = rep_df["CFU_trt"] / rep_df["CFU_ctrl"]

		mean_CFU_control = rep_ctrl.mean()
		mean_CFU_treatment = rep_trt.mean()
		std_CFU_control = rep_ctrl.std(ddof=1) if len(rep_ctrl) > 1 else 0.0
		std_CFU_treatment = rep_trt.std(ddof=1) if len(rep_trt) > 1 else 0.0
		log10_reduction = (
			np.log10(mean_CFU_control / mean_CFU_treatment) if mean_CFU_treatment > 0 else np.inf
		)

		boot_stats = bootstrap_mean_ratio(rep_df["viab_rep"].values)
		viability_mean = boot_stats["mean"]
		viability_ci_low = boot_stats["ci_low"]
		viability_ci_high = boot_stats["ci_high"]
		n_reps = rep_df.shape[0]

		if (trt["colonies"] == 0).all():
			n_plates = trt.shape[0]
			upper95 = rule_of_three_zero_events(n_plates)
		else:
			upper95 = None

		detection_limits = compute_cfu(1, org_df["dilution_log"], org_df["plated_uL"])
		detection_limit = float(detection_limits.min())

		plot_cfu_comprehensive(org_df, org, outdir)

		if (trt["colonies"] == 0).all():
			viab_mean_out = viab_low_out = viab_high_out = None
		else:
			viab_mean_out = float(viability_mean)
			viab_low_out = float(viability_ci_low)
			viab_high_out = float(viability_ci_high)

		ctrl_log10 = float(np.log10(mean_CFU_control)) if mean_CFU_control > 0 else None
		trt_log10 = float(np.log10(mean_CFU_treatment)) if mean_CFU_treatment > 0 else None
		log10_det = float(np.log10(detection_limit)) if detection_limit > 0 else float("nan")

		results.append(
			OrganismResult(
				organism=org,
				mean_CFU_control=float(mean_CFU_control),
				std_CFU_control=float(std_CFU_control),
				mean_CFU_treatment=float(mean_CFU_treatment),
				std_CFU_treatment=float(std_CFU_treatment),
				log10_reduction=float(log10_reduction),
				viability_mean=viab_mean_out,
				viability_ci_low=viab_low_out,
				viability_ci_high=viab_high_out,
				upper95_viability_if_all_zero=None if upper95 is None else float(upper95),
				detection_limit_CFU_per_mL=detection_limit,
				log10_detection_limit=log10_det,
				mean_CFU_control_log10=ctrl_log10,
				mean_CFU_treatment_log10=trt_log10,
				n_replicates=int(n_reps),
			)
		)

	# Supplemental full table (CSV + JSON)
	res_records = [asdict(r) for r in results]
	supp_df = pd.DataFrame(res_records)
	write_csv(os.path.join(outdir, "fe_results_supplemental.csv"), supp_df)
	write_json(os.path.join(outdir, "fe_results_supplemental.json"), res_records)

	# Main simplified JSON/CSV
	main_list = []
	for r in res_records:	
		if np.isinf(r["log10_reduction"]):
			if r["mean_CFU_treatment"] == 0 and r["detection_limit_CFU_per_mL"] > 0:
				lower_bound = np.log10(r["mean_CFU_control"] / r["detection_limit_CFU_per_mL"])
				log10_red_repr = f"> {lower_bound:.2f}"
			else:
				log10_red_repr = "> value"
		else:
			log10_red_repr = f"{r['log10_reduction']:.2f}"
		entry = {
			"organism": r["organism"],
			"n_replicates": r["n_replicates"],
			"control_mean_log10_CFU_per_mL": r.get("mean_CFU_control_log10"),
			"treatment_mean_log10_CFU_per_mL": r.get("mean_CFU_treatment_log10"),
			"log10_reduction": log10_red_repr,
			"log10_detection_limit": r.get("log10_detection_limit"),
		}
		if r["viability_mean"] is not None:
			entry["viability_mean"] = r["viability_mean"]
			entry["viability_CI95"] = [r["viability_ci_low"], r["viability_ci_high"]]
		elif r["upper95_viability_if_all_zero"] is not None:
			entry["upper95_viability_if_all_zero"] = r["upper95_viability_if_all_zero"]
		main_list.append(entry)

	main_csv_rows = []
	for m in main_list:
		row = {
			"organism": m["organism"],
			"n_replicates": m["n_replicates"],
			"control_mean_log10_CFU_per_mL": m["control_mean_log10_CFU_per_mL"],
			"treatment_mean_log10_CFU_per_mL": m["treatment_mean_log10_CFU_per_mL"],
			"log10_reduction": m["log10_reduction"],
			"log10_detection_limit": m["log10_detection_limit"],
		}
		if "viability_mean" in m:
			row["viability_mean"] = m["viability_mean"]
			row["viability_CI95_low"] = m["viability_CI95"][0]
			row["viability_CI95_high"] = m["viability_CI95"][1]
		if "upper95_viability_if_all_zero" in m:
			row["upper95_viability_if_all_zero"] = m["upper95_viability_if_all_zero"]
		main_csv_rows.append(row)

	# Export main results
	main_df = pd.DataFrame(main_csv_rows)
	write_csv(os.path.join(outdir, "fe_results_main.csv"), main_df)
	write_json(os.path.join(outdir, "fe_results_main.json"), main_list)

	print(
		"Analysis complete. Generated files: fe_results_main.(csv|json), fe_results_supplemental.(csv|json) in",
		outdir,
	)

# CLI entry point

def main():
	"""Command-line entry point. Parses arguments and runs analysis."""
	parser = argparse.ArgumentParser(
		description="Fe³⁺ experiment analysis (control vs single treatment)."
	)
	parser.add_argument("--csv", required=True, help="Input CSV file.")
	parser.add_argument("--out", default="results_fe", help="Output directory.")
	args = parser.parse_args()
	df = load_data(args.csv)
	if "experiment" in df.columns:
		df = df[df["experiment"].str.lower() == "fe"]
	analyze_fe_experiment(df, outdir=args.out)


if __name__ == "__main__":
	main()

