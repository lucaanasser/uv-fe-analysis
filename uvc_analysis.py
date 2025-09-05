"""UVC survival analysis pipeline (Weibull removed)."""

from __future__ import annotations

import os
import argparse
from dataclasses import asdict
import numpy as np
import pandas as pd

from analysis.io_utils import load_uvc_data, compute_cfu
from analysis.stat_utils import (
    bootstrap,
    rule_of_three_zero_events,
    fit_log_linear,
    approximate_log10_surv_variance,
    fit_log_linear_wls,
    fit_mixed_model,
    fit_power_loglog,
    fit_biphasic,
    fit_shoulder,
    fit_glm_counts,
    one_way_anova,
    lack_of_fit_test_through_origin,
)
from analysis.plot_utils import (
    plot_survival_curve,
    plot_cfu_vs_dose,
    plot_survival_with_models,
    plot_model_comparison,
)
from analysis.results import UVCResultMain, UVCResultSupplement
from analysis.export_utils import ensure_dir, write_json, write_csv


def compute_survival(df: pd.DataFrame) -> pd.DataFrame:
    ref = (
        df[df["dose_J_m2"] == 0]
        .groupby(["experiment", "organism", "replicate"]).agg(CFU0=("CFU_per_mL", "mean"))
        .reset_index()
    )
    out = df.merge(ref, on=["experiment", "organism", "replicate"], how="left")
    out["survival_fraction"] = out["CFU_per_mL"] / out["CFU0"]
    out.loc[out["CFU0"] <= 0, "survival_fraction"] = np.nan
    out["log10_survival"] = np.log10(out["survival_fraction"])
    return out


def censored_adjustments(df: pd.DataFrame, detection_floor: float) -> pd.DataFrame:
    mask = df["CFU_per_mL"] <= 0
    if mask.any():
        df.loc[mask, "survival_fraction"] = detection_floor / df.loc[mask, "CFU0"]
        df.loc[mask, "log10_survival"] = np.log10(df.loc[mask, "survival_fraction"])
    return df


def bootstrap_survival_band(df: pd.DataFrame, organism: str, B: int = 300):  # organism kept for interface clarity
    doses = sorted(df["dose_J_m2"].unique())
    rng = np.random.default_rng(42)
    units = df.groupby(["experiment", "replicate", "dose_J_m2"]).agg(log10_survival=("log10_survival", "mean")).reset_index()
    exp_reps = units[["experiment", "replicate"]].drop_duplicates().values.tolist()
    acc: list[list[float]] = []
    for _ in range(B):
        idx = rng.choice(len(exp_reps), size=len(exp_reps), replace=True)
        sampled = [exp_reps[i] for i in idx]
        parts = [units[(units["experiment"] == er[0]) & (units["replicate"] == er[1])] for er in sampled]
        boot_df = pd.concat(parts, ignore_index=True)
        mean_by_dose = boot_df.groupby("dose_J_m2")["log10_survival"].mean()
        acc.append([mean_by_dose.get(d, np.nan) for d in doses])
    arr = np.array(acc)
    return {"dose": np.array(doses), "low": np.nanpercentile(arr, 2.5, axis=0), "high": np.nanpercentile(arr, 97.5, axis=0)}


def compute_LD50(slope: float) -> float:
    if slope is None or not np.isfinite(slope) or slope >= 0:
        return np.nan
    return np.log10(0.5) / slope


def analyze_uvc(df: pd.DataFrame, outdir: str = "results_uv") -> None:
    ensure_dir(outdir)
    main_results: list[UVCResultMain] = []
    supplement_rows: list[UVCResultSupplement] = []
    for org in df["organism"].unique():
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

        rng = np.random.default_rng(42)
        exp_reps = grp[["experiment", "replicate"]].drop_duplicates().values.tolist()
        slopes: list[float] = []
        D10s: list[float] = []
        LD50s: list[float] = []
        for _ in range(800):
            idxs = rng.choice(len(exp_reps), size=len(exp_reps), replace=True)
            sampled = [exp_reps[i] for i in idxs]
            parts = [grp[(grp["experiment"] == er[0]) & (grp["replicate"] == er[1])] for er in sampled]
            boot_df = pd.concat(parts, ignore_index=True)
            fit_b = fit_log_linear(boot_df["dose_J_m2"], boot_df["log10_surv"])
            slopes.append(fit_b["slope"])
            D10s.append(fit_b["D10"])
            LD50s.append(compute_LD50(fit_b["slope"]))
        slope_ci = (np.nanpercentile(slopes, 2.5), np.nanpercentile(slopes, 97.5))
        D10_ci = (np.nanpercentile(D10s, 2.5), np.nanpercentile(D10s, 97.5))
        LD50_ci = (np.nanpercentile(LD50s, 2.5), np.nanpercentile(LD50s, 97.5))
        LD50_point = compute_LD50(slope_fit["slope"]) if np.isfinite(slope_fit["slope"]) else np.nan

        band = bootstrap_survival_band(org_df, org, B=300)
        plot_survival_curve(org_df, org, outdir, slope_fit=slope_fit, boot_bands=band, title_suffix="UVC survival")
        plot_cfu_vs_dose(org_df, org, outdir)

        mixed_fit = fit_mixed_model(org_df)
        if grp["surv_frac"].between(0, 1).any():
            power_fit = fit_power_loglog(grp["dose_J_m2"], grp["surv_frac"])
            biphasic_fit = fit_biphasic(grp["dose_J_m2"], grp["surv_frac"])
            shoulder_fit = fit_shoulder(grp["dose_J_m2"], grp["surv_frac"])
        else:
            power_fit = {"a": np.nan, "b": np.nan, "AIC": np.nan, "R2": np.nan}
            biphasic_fit = {"f": np.nan, "k1": np.nan, "k2": np.nan, "AIC": np.nan}
            shoulder_fit = {"D0": np.nan, "k": np.nan, "AIC": np.nan}
        glm_fit = fit_glm_counts(org_df)
        anova_res = one_way_anova(grp["dose_J_m2"], grp["log10_surv"]) if len(grp) else {"anova_F": np.nan, "anova_p": np.nan, "df_between": 0, "df_within": 0}
        lof_res = (
            lack_of_fit_test_through_origin(grp["dose_J_m2"], grp["log10_surv"], slope_fit.get("slope", np.nan))
            if np.isfinite(slope_fit.get("slope", np.nan)) else {"lof_F": np.nan, "lof_p": np.nan, "df_lof": 0, "df_pe": 0}
        )

        primary_slope = slope_fit.get("slope")
        D90 = -np.log10(0.1) / primary_slope if primary_slope and primary_slope < 0 else np.nan
        D99 = -np.log10(0.01) / primary_slope if primary_slope and primary_slope < 0 else np.nan
        D999 = -np.log10(0.001) / primary_slope if primary_slope and primary_slope < 0 else np.nan

        candidate_aics = {"log-linear": slope_fit.get("AIC"), "power": power_fit.get("AIC"), "biphasic": biphasic_fit.get("AIC"), "shoulder": shoulder_fit.get("AIC")}
        finite = {k: v for k, v in candidate_aics.items() if v is not None and np.isfinite(v)}
        best_model = min(finite, key=finite.get) if finite else "log-linear"
        model_params = {
            "log-linear": {"slope": slope_fit.get("slope")},
            "power": {"a": power_fit.get("a"), "b": power_fit.get("b")},
            "biphasic": {"f": biphasic_fit.get("f"), "k1": biphasic_fit.get("k1"), "k2": biphasic_fit.get("k2")},
            "shoulder": {"D0": shoulder_fit.get("D0"), "k": shoulder_fit.get("k")},
        }
        overlays: dict[str, dict] = {}
        if best_model != "log-linear":
            overlays["log-linear"] = model_params["log-linear"]
        overlays[best_model] = model_params.get(best_model, {})
        plot_survival_with_models(org_df, org, outdir, mean_points=True, boot_bands=band, models=overlays)
        plot_model_comparison(org_df, org, outdir, models=model_params, aic=candidate_aics)

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
            D90=float(D90),
            D99=float(D99),
            D999=float(D999),
            LD50_dose=float(LD50_point),
            LD50_CI95=[LD50_ci[0], LD50_ci[1]],
            LD50_wls=np.log10(0.5) / wls_fit["slope"] if wls_fit.get("slope") and wls_fit.get("slope") < 0 else None,
            R2_adj=slope_fit.get("R2_adj"),
            AIC_log_linear=slope_fit.get("AIC"),
            AIC_power=power_fit.get("AIC"),
            AIC_biphasic=biphasic_fit.get("AIC"),
            AIC_shoulder=shoulder_fit.get("AIC"),
            power_a=power_fit.get("a"),
            power_b=power_fit.get("b"),
            biphasic_f=biphasic_fit.get("f"),
            biphasic_k1=biphasic_fit.get("k1"),
            biphasic_k2=biphasic_fit.get("k2"),
            shoulder_D0=shoulder_fit.get("D0"),
            shoulder_k=shoulder_fit.get("k"),
            model_preferred=best_model,
            mixed_slope=mixed_fit.get("mixed_slope"),
            mixed_slope_se=mixed_fit.get("mixed_slope_se"),
            mixed_var_intercept=mixed_fit.get("var_intercept"),
            poisson_slope_log10=glm_fit.get("poisson_slope_log10"),
            poisson_slope_log10_se=glm_fit.get("poisson_slope_log10_se"),
            nb_slope_log10=glm_fit.get("nb_slope_log10"),
            nb_slope_log10_se=glm_fit.get("nb_slope_log10_se"),
            AIC_poisson=glm_fit.get("AIC_poisson"),
            AIC_nb=glm_fit.get("AIC_nb"),
            dispersion_poisson=glm_fit.get("dispersion_poisson"),
            model_preferred_glm=glm_fit.get("model_preferred_glm"),
            detection_limit_CFU_per_mL=detection_floor,
            any_full_zero_dose=bool((org_df.groupby("dose_J_m2")["colonies"].sum() == 0).any()),
            n_doses=int(org_df["dose_J_m2"].nunique()),
            anova_F=anova_res.get("anova_F"),
            anova_p=anova_res.get("anova_p"),
            df_between=anova_res.get("df_between"),
            df_within=anova_res.get("df_within"),
            lof_F=lof_res.get("lof_F"),
            lof_p=lof_res.get("lof_p"),
            df_lof=lof_res.get("df_lof"),
            df_pe=lof_res.get("df_pe"),
        ))

        dose_grp = org_df.groupby("dose_J_m2").agg(mean_CFU=("CFU_per_mL", "mean")).reset_index()
        for _, row in dose_grp.iterrows():
            dose = row["dose_J_m2"]
            subset = org_df[org_df["dose_J_m2"] == dose]
            surv_boot = bootstrap(subset["survival_fraction"], func=np.mean, n_boot=600)
            log_boot = bootstrap(subset["log10_survival"], func=np.mean, n_boot=600)
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

    main_records = [asdict(r) for r in main_results]
    write_csv(os.path.join(outdir, "uvc_results_main.csv"), pd.DataFrame(main_records))
    write_json(os.path.join(outdir, "uvc_results_main.json"), main_records)
    supp_records = [asdict(r) for r in supplement_rows]
    write_csv(os.path.join(outdir, "uvc_results_supplemental.csv"), pd.DataFrame(supp_records))
    write_json(os.path.join(outdir, "uvc_results_supplemental.json"), supp_records)
    print("UVC analysis complete. Files: uvc_results_main.(csv|json), uvc_results_supplemental.(csv|json)")


def main():
    parser = argparse.ArgumentParser(description="UVC irradiation survival analysis")
    parser.add_argument("--csv", required=True, help="Input CSV with UVC data")
    parser.add_argument("--out", default="results_uv", help="Output directory")
    args = parser.parse_args()
    df = load_uvc_data(args.csv)
    analyze_uvc(df, outdir=args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
