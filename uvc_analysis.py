import os
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import asdict
from analysis.io_utils import load_uvc_data, compute_cfu
from analysis.stat_utils import (
    bootstrap,
    rule_of_three_zero_events,
    fit_log_linear,
    fit_weibull,
    approximate_log10_surv_variance,
    fit_log_linear_wls,
    compute_cooks_distance_through_origin,
    fit_mixed_model,
)
from analysis.plot_utils import (
    plot_survival_curve,
    plot_cfu_vs_dose,
    plot_residuals,
    plot_qq,
)
from analysis.results import UVCResultMain, UVCResultSupplement
from analysis.export_utils import ensure_dir, write_json, write_csv

# =============================================================
# Data classes
# =============================================================

## Dataclasses agora importados de analysis_common.results

# =============================================================
# Core computations
# =============================================================

def compute_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula fração e log10 sobrevivência usando dose 0 como referência.

    Args:
        df: DataFrame com CFU_per_mL, dose_J_m2, experiment, replicate.

    Returns:
        DataFrame com survival_fraction e log10_survival.
    """
    ref = df[df["dose_J_m2"] == 0].groupby(["experiment", "organism", "replicate"]).agg(CFU0=("CFU_per_mL", "mean")).reset_index()
    out = df.merge(ref, on=["experiment", "organism", "replicate"], how="left")
    out["survival_fraction"] = out["CFU_per_mL"] / out["CFU0"]
    out.loc[out["CFU0"] <= 0, "survival_fraction"] = np.nan
    out["log10_survival"] = np.log10(out["survival_fraction"])  # can be -inf if zero
    return out


def censored_adjustments(df: pd.DataFrame, detection_floor: float):
    """Aplica ajuste censurado substituindo zeros por limite de detecção.

    Args:
        df: DataFrame pós compute_survival.
        detection_floor: Menor CFU/mL detectável.

    Returns:
        DataFrame ajustado.
    """
    zero_mask = df["CFU_per_mL"] <= 0
    if zero_mask.any():
        df.loc[zero_mask, "survival_fraction"] = detection_floor / df.loc[zero_mask, "CFU0"]
        df.loc[zero_mask, "log10_survival"] = np.log10(df.loc[zero_mask, "survival_fraction"])
    return df


def bootstrap_survival_band(df: pd.DataFrame, organism: str, B=2000):
    """Constrói banda bootstrap média de log10 sobrevivência por dose.

    Args:
        df: DataFrame completo de observações.
        organism: Nome do organismo (não usado diretamente, apenas interface consistente).
        B: Número de réplicas bootstrap.

    Returns:
        Dict com dose, low, high.
    """
    doses = sorted(df["dose_J_m2"].unique())
    dose_arr = np.array(doses, dtype=float)
    rng = np.random.default_rng(42)
    units = df.groupby(["experiment", "replicate", "dose_J_m2"]).agg(log10_survival=("log10_survival", "mean")).reset_index()
    exp_reps = units[["experiment", "replicate"]].drop_duplicates().values.tolist()
    for b in range(B):
        sample_units = rng.choice(len(exp_reps), size=len(exp_reps), replace=True)
        sampled_rows = []
        for idx in sample_units:
            er = exp_reps[idx]
            sub = units[(units["experiment"] == er[0]) & (units["replicate"] == er[1])]
            sampled_rows.append(sub)
        sample_df = pd.concat(sampled_rows, ignore_index=True)
        mean_by_dose = sample_df.groupby("dose_J_m2")["log10_survival"].mean()
        row = [mean_by_dose.get(d, np.nan) for d in doses]
        if b == 0:
            acc = np.array(row)[None, :]
        else:
            acc = np.vstack([acc, row])
    low = np.nanpercentile(acc, 2.5, axis=0)
    high = np.nanpercentile(acc, 97.5, axis=0)
    return {"dose": dose_arr, "low": low, "high": high}


def compute_LD50(slope):
    """Calcula LD50 dado slope log10 (sobrevivência = 10^{slope*dose}).

    Args:
        slope: Inclinação log10 (negativa em mortalidade).

    Returns:
        Dose para 50% de sobrevivência ou NaN.
    """
    if slope >= 0:
        return np.nan
    return np.log10(0.5) / slope

# =============================================================
# Main analysis
# =============================================================

def analyze_uvc(df: pd.DataFrame, outdir: str = "results_uv"):
    """Pipeline principal de análise UVC.

    Args:
        df: DataFrame experimental UVC com CFU_per_mL.
        outdir: Diretório de saída.

    Returns:
        None. Gera arquivos de resultados e gráficos.
    """
    ensure_dir(outdir)
    organisms = df["organism"].unique()
    main_results: list[UVCResultMain] = []
    supplement_rows: list[UVCResultSupplement] = []

    for org in organisms:
        org_df = df[df["organism"] == org].copy()
        org_df = compute_survival(org_df)
        detection_limits = compute_cfu(1, org_df["dilution_log"], org_df["plated_uL"])
        detection_floor = float(np.nanmin(detection_limits)) if detection_limits.size else 1.0
        org_df = censored_adjustments(org_df, detection_floor)

        grp = org_df.groupby(["experiment", "replicate", "dose_J_m2"]).agg(
            log10_surv=("log10_survival", "mean"),
            surv_frac=("survival_fraction", "mean"),
            CFU_mean=("CFU_per_mL", "mean"),
        ).reset_index()
        slope_fit = fit_log_linear(grp["dose_J_m2"], grp["log10_surv"])
        baseline = grp[grp["dose_J_m2"] == 0][["replicate", "CFU_mean"]].rename(columns={"CFU_mean": "CFU0"})
        grp_w = grp.merge(baseline, on="replicate", how="left")
        var_log10 = approximate_log10_surv_variance(grp_w["CFU_mean"], grp_w["CFU0"]) if len(grp_w) else []
        wls_fit = fit_log_linear_wls(grp_w["dose_J_m2"], grp_w["log10_surv"], var_log10) if len(grp_w) else {"slope": np.nan}

        # Bootstrap slope/D10/LD50
        rng = np.random.default_rng(42)
        exp_reps = grp[["experiment", "replicate"]].drop_duplicates().values.tolist()
        slopes: list[float] = []
        D10s: list[float] = []
        LD50s: list[float] = []
        for _ in range(2000):
            sample_units = rng.choice(len(exp_reps), size=len(exp_reps), replace=True)
            sampled = []
            for idx in sample_units:
                er = exp_reps[idx]
                sampled.append(grp[(grp["experiment"] == er[0]) & (grp["replicate"] == er[1])])
            boot_df = pd.concat(sampled, ignore_index=True)
            fit_b = fit_log_linear(boot_df["dose_J_m2"], boot_df["log10_surv"])  # slope only
            slopes.append(fit_b["slope"])
            D10s.append(fit_b["D10"])
            LD50s.append(compute_LD50(fit_b["slope"]))
        slope_ci = (np.nanpercentile(slopes, 2.5), np.nanpercentile(slopes, 97.5))
        D10_ci = (np.nanpercentile(D10s, 2.5), np.nanpercentile(D10s, 97.5))
        LD50_ci = (np.nanpercentile(LD50s, 2.5), np.nanpercentile(LD50s, 97.5))
        LD50_point = compute_LD50(slope_fit["slope"]) if np.isfinite(slope_fit["slope"]) else np.nan

        band = bootstrap_survival_band(org_df, org, B=500)
        plot_survival_curve(org_df, org, outdir, slope_fit=slope_fit, boot_bands=band, title_suffix="UVC survival")
        plot_cfu_vs_dose(org_df, org, outdir)

        weibull_par = fit_weibull(grp["dose_J_m2"], grp["surv_frac"]) if grp["surv_frac"].between(0,1).any() else {"delta": np.nan, "p": np.nan}
        mixed_fit = fit_mixed_model(org_df)

        main_results.append(UVCResultMain(
            organism=org,
            slope_log10=slope_fit.get("slope"),
            slope_log10_se=slope_fit.get("slope_se"),
            slope_log10_CI95=[slope_ci[0], slope_ci[1]],
            slope_log10_wls=wls_fit.get("slope"),
            slope_log10_wls_se=wls_fit.get("slope_se"),
            slope_log10_wls_R2_adj=wls_fit.get("R2_adj"),
            D10=slope_fit.get("D10"),
            D10_CI95=[D10_ci[0], D10_ci[1]],
            D10_wls=-1.0 / wls_fit["slope"] if wls_fit.get("slope") and wls_fit.get("slope") < 0 else None,
            LD50_dose=float(LD50_point),
            LD50_CI95=[LD50_ci[0], LD50_ci[1]],
            LD50_wls=np.log10(0.5) / wls_fit["slope"] if wls_fit.get("slope") and wls_fit.get("slope") < 0 else None,
            R2_adj=slope_fit.get("R2_adj"),
            AIC_log_linear=slope_fit.get("AIC"),
            AIC_weibull=None,  # could compute with full residuals method
            weibull_delta=weibull_par.get("delta"),
            weibull_p=weibull_par.get("p"),
            model_preferred="log-linear",
            mixed_slope=mixed_fit.get("mixed_slope"),
            mixed_slope_se=mixed_fit.get("mixed_slope_se"),
            mixed_var_intercept=mixed_fit.get("var_intercept"),
            detection_limit_CFU_per_mL=detection_floor,
            any_full_zero_dose=bool((org_df.groupby("dose_J_m2")["colonies"].sum() == 0).any()),
            n_doses=int(org_df["dose_J_m2"].nunique()),
        ))

        dose_grp = org_df.groupby("dose_J_m2").agg(mean_CFU=("CFU_per_mL", "mean")).reset_index()
        for _, row in dose_grp.iterrows():
            dose = row["dose_J_m2"]
            subset = org_df[org_df["dose_J_m2"] == dose]
            surv_boot = bootstrap(subset["survival_fraction"], func=np.mean, n_boot=2000)
            log_boot = bootstrap(subset["log10_survival"], func=np.mean, n_boot=2000)
            log_red_point = -log_boot["point"]
            log_red_ci = [-log_boot["ci_high"], -log_boot["ci_low"]]
            all_zero = (subset["colonies"] == 0).all()
            upper95 = rule_of_three_zero_events(subset.shape[0]) if all_zero else None
            supplement_rows.append(UVCResultSupplement(
                organism=org,
                dose_J_m2=float(dose),
                mean_CFU_per_mL=float(row["mean_CFU"]),
                mean_survival_fraction=surv_boot["point"],
                survival_CI95=[surv_boot["ci_low"], surv_boot["ci_high"]],
                log10_survival_mean=log_boot["point"],
                log10_survival_CI95=[log_boot["ci_low"], log_boot["ci_high"]],
                log10_reduction_mean=log_red_point,
                log10_reduction_CI95=log_red_ci,
                upper95_survival_if_all_zero=upper95,
            ))

        with open(os.path.join(outdir, f"{org.replace(' ', '_')}_weibull_params.json"), "w") as f:
            json.dump({"organism": org, **weibull_par}, f, indent=2)

    # Write main results
    main_records = [asdict(r) for r in main_results]
    main_df = pd.DataFrame(main_records)
    write_csv(os.path.join(outdir, "uvc_results_main.csv"), main_df)
    write_json(os.path.join(outdir, "uvc_results_main.json"), main_records)

    supp_records = [asdict(r) for r in supplement_rows]
    supp_df = pd.DataFrame(supp_records)
    write_csv(os.path.join(outdir, "uvc_results_supplemental.csv"), supp_df)
    write_json(os.path.join(outdir, "uvc_results_supplemental.json"), supp_records)

    print("UVC analysis complete. Files: uvc_results_main.(csv|json), uvc_results_supplemental.(csv|json)")

# =============================================================
# CLI
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="UVC irradiation survival analysis.")
    parser.add_argument("--csv", required=True, help="Input CSV file with UVC data.")
    parser.add_argument("--out", default="results_uv", help="Output directory.")
    args = parser.parse_args()
    df = load_uvc_data(args.csv)
    analyze_uvc(df, outdir=args.out)

if __name__ == "__main__":
    main()
