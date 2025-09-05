import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Dict, Any

PLOT_STYLE = {
    "figsize": (8,6),
    "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#4E937A"],
}

def style_axes(ax):
    """Apply consistent styling to axes.

    Args:
        ax: Matplotlib axes object.

    Returns:
        None. Modifies axes in-place.
    """
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def save_fig(fig, outdir, filename):
    """Save figure to directory, creating it if necessary.

    Args:
        fig: Matplotlib figure.
        outdir: Output directory.
        filename: File name (png).

    Returns:
        None.
    """
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_survival_curve(df, organism, outdir, slope_fit=None, boot_bands=None, title_suffix="UVC survival"):
    """Plot mean log10 survival curve vs dose.

    Args:
        df: DataFrame containing survival_fraction and log10_survival.
        organism: Organism name.
        outdir: Output directory for the PNG.
        slope_fit: Optional dict with 'slope' from log-linear fit.
        boot_bands: Dict with arrays dose, low, high for uncertainty band.
        title_suffix: Title suffix.

    Returns:
        None. Saves figure.
    """
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    agg = df.groupby("dose_J_m2").agg(mean_surv=("survival_fraction", "mean"), log10_survival=("log10_survival", "mean")).reset_index()
    doses = agg["dose_J_m2"].values
    log10_mean = agg["log10_survival"].values
    ax.scatter(doses, log10_mean, color=PLOT_STYLE["colors"][0], s=70, edgecolor="black", zorder=3)
    if slope_fit and np.isfinite(slope_fit.get("slope", np.nan)):
        x_line = np.linspace(0, doses.max()*1.05, 200)
        y_line = slope_fit["slope"] * x_line
        ax.plot(x_line, y_line, color="black", linewidth=2, label=f"Log-linear slope={slope_fit['slope']:.3f}")
    if boot_bands:
        ax.fill_between(boot_bands["dose"], boot_bands["low"], boot_bands["high"], color="#2E86AB", alpha=0.15, label="95% band")
    ax.set_xlabel("Dose (J/m²)", fontsize=12, weight="bold")
    ax.set_ylabel("log10 Survival", fontsize=12, weight="bold")
    ax.set_title(f"{organism} — {title_suffix}", fontsize=14, weight="bold", pad=20)
    style_axes(ax)
    ax.legend(frameon=False)
    save_fig(fig, outdir, f"{organism.replace(' ', '_')}_uvc_survival.png")


def plot_survival_with_models(df, organism, outdir, mean_points=True, boot_bands=None, models: Dict[str, Dict[str, Any]]|None=None, title_suffix="UVC survival (model overlay)"):
    """Plot empirical mean log10 survival with optional bootstrap band and selected model curves.

    Args:
        df: Raw replicated dataframe (needs columns dose_J_m2, log10_survival, survival_fraction).
        organism: Organism name.
        outdir: Output directory.
        mean_points: If True scatter mean per dose; else all points.
        boot_bands: Optional dict dose/low/high for band.
        models: Dict mapping model label -> dict of parameters (expects keys used below).
        title_suffix: Title text.
    """
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    if mean_points:
        agg = df.groupby("dose_J_m2").agg(log10_survival=("log10_survival","mean")).reset_index()
        ax.scatter(agg["dose_J_m2"], agg["log10_survival"], s=70, color=PLOT_STYLE["colors"][0], edgecolor="black", label="Média")
    else:
        ax.scatter(df["dose_J_m2"], df["log10_survival"], s=40, alpha=0.6, color=PLOT_STYLE["colors"][0], label="Replicados")
    if boot_bands:
        ax.fill_between(boot_bands["dose"], boot_bands["low"], boot_bands["high"], color=PLOT_STYLE["colors"][0], alpha=0.15, label="95% bootstrap")
    if models:
        doses = df["dose_J_m2"].unique()
        d_min_pos = np.min(doses[doses>0]) if np.any(doses>0) else 1.0
        x_line = np.linspace(0, doses.max()*1.05, 400)
        x_line_pos = np.linspace(d_min_pos, doses.max()*1.05, 400)
        color_cycle = plt.cm.tab10(np.linspace(0,1,len(models)))
        for (i,(label, pars)) in enumerate(models.items()):
            y_pred = None
            if label == "log-linear" and np.isfinite(pars.get("slope", np.nan)):
                y_pred = pars["slope"] * x_line
                ax.plot(x_line, y_pred, color=color_cycle[i], lw=2, label=f"{label}")
                continue
            if label == "power" and all(np.isfinite([pars.get("a"), pars.get("b")])):
                # log10(S)=a + b*log10(D)
                xv = x_line_pos
                y_pred = pars["a"] + pars["b"]*np.log10(xv)
                ax.plot(xv, y_pred, color=color_cycle[i], lw=2, label=f"{label}")
                continue
            if label == "biphasic" and all(np.isfinite([pars.get("f"), pars.get("k1"), pars.get("k2")])):
                xv = x_line
                S = pars["f"]*np.exp(-pars["k1"]*xv) + (1-pars["f"])*np.exp(-pars["k2"]*xv)
                y_pred = np.log10(np.clip(S,1e-12,1))
                ax.plot(xv, y_pred, color=color_cycle[i], lw=2, label=f"{label}")
                continue
            if label == "shoulder" and all(np.isfinite([pars.get("D0"), pars.get("k")])):
                xv = x_line
                S = np.where(xv <= pars["D0"], 1.0, np.exp(-pars["k"]*(xv-pars["D0"])) )
                y_pred = np.log10(np.clip(S,1e-12,1))
                ax.plot(xv, y_pred, color=color_cycle[i], lw=2, label=f"{label}")
                continue
            # Weibull model removed
    ax.set_xlabel("Dose (J/m²)", fontsize=12, weight="bold")
    ax.set_ylabel("log10 Sobrevivência", fontsize=12, weight="bold")
    ax.set_title(f"{organism} — {title_suffix}", fontsize=14, weight="bold", pad=20)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, outdir, f"{organism.replace(' ', '_')}_uvc_survival_best.png")


def plot_model_comparison(df, organism, outdir, models: Dict[str, Dict[str, Any]], aic: Dict[str, float]|None=None):
    """Create a dedicated model comparison figure with all model curves.

    Args:
        df: DataFrame with log10_survival.
        organism: Name.
        outdir: Output directory.
        models: Dict of model parameters (same labels as overlay).
        aic: Optional dict label->AIC to annotate.
    """
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    agg = df.groupby("dose_J_m2").agg(log10_survival=("log10_survival","mean")).reset_index()
    ax.scatter(agg["dose_J_m2"], agg["log10_survival"], color="black", s=65, zorder=4, label="Média")
    doses = df["dose_J_m2"].unique()
    d_min_pos = np.min(doses[doses>0]) if np.any(doses>0) else 1.0
    x_line = np.linspace(0, doses.max()*1.05, 500)
    x_line_pos = np.linspace(d_min_pos, doses.max()*1.05, 500)
    color_cycle = plt.cm.Dark2(np.linspace(0,1,len(models)))
    legend_labels = []
    for (i,(label, pars)) in enumerate(models.items()):
        y_pred=None
        if label == "log-linear" and np.isfinite(pars.get("slope", np.nan)):
            y_pred = pars["slope"]*x_line
            ax.plot(x_line, y_pred, color=color_cycle[i], lw=2)
        elif label == "power" and all(np.isfinite([pars.get("a"), pars.get("b")])):
            xv = x_line_pos
            y_pred = pars["a"] + pars["b"]*np.log10(xv)
            ax.plot(xv, y_pred, color=color_cycle[i], lw=2)
        elif label == "biphasic" and all(np.isfinite([pars.get("f"), pars.get("k1"), pars.get("k2")])):
            xv = x_line
            S = pars["f"]*np.exp(-pars["k1"]*xv) + (1-pars["f"])*np.exp(-pars["k2"]*xv)
            y_pred = np.log10(np.clip(S,1e-12,1))
            ax.plot(xv, y_pred, color=color_cycle[i], lw=2)
        elif label == "shoulder" and all(np.isfinite([pars.get("D0"), pars.get("k")])):
            xv = x_line
            S = np.where(xv <= pars["D0"], 1.0, np.exp(-pars["k"]*(xv-pars["D0"])) )
            y_pred = np.log10(np.clip(S,1e-12,1))
            ax.plot(xv, y_pred, color=color_cycle[i], lw=2)
    # Weibull model removed
        if y_pred is not None:
            aic_txt = ''
            if aic and label in aic and np.isfinite(aic[label]):
                aic_txt = f" (AIC {aic[label]:.1f})"
            legend_labels.append(f"{label}{aic_txt}")
    ax.set_xlabel("Dose (J/m²)")
    ax.set_ylabel("log10 Sobrevivência")
    ax.set_title(f"{organism} — Comparação de Modelos")
    style_axes(ax)
    ax.legend(legend_labels, frameon=False, fontsize=9)
    # Optional small AIC ranking textbox
    if aic:
        finite = {k:v for k,v in aic.items() if v is not None and np.isfinite(v)}
        if finite:
            rank = sorted(finite.items(), key=lambda kv: kv[1])
            txt = "\n".join([f"{i+1}. {k}: {v:.1f}" for i,(k,v) in enumerate(rank)])
            ax.text(0.98,0.02, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, edgecolor='none'))
    save_fig(fig, outdir, f"{organism.replace(' ', '_')}_uvc_model_comparison.png")


def plot_cfu_vs_dose(df, organism, outdir):
    """Plot CFU/mL (log scale) vs dose.

    Args:
        df: DataFrame with CFU_per_mL and dose_J_m2.
        organism: Organism name.
        outdir: Output directory.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    agg = df.groupby("dose_J_m2").agg(mean_CFU=("CFU_per_mL", "mean")).reset_index()
    ax.plot(agg["dose_J_m2"], agg["mean_CFU"], marker="o", color=PLOT_STYLE["colors"][1])
    ax.set_yscale("log")
    ax.set_xlabel("Dose (J/m²)")
    ax.set_ylabel("CFU/mL (log)")
    ax.set_title(f"{organism} — Raw CFU vs Dose")
    style_axes(ax)
    save_fig(fig, outdir, f"{organism.replace(' ', '_')}_uvc_cfu.png")


def plot_residuals(doses, residuals, organism, outdir, filename_suffix="residuals"):
    """Plot residuals vs dose.

    Args:
        doses: Sequence of doses.
        residuals: Corresponding residuals (log10).
        organism: Organism name.
        outdir: Output directory.
        filename_suffix: File name suffix.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axhline(0, color="black", lw=1)
    ax.scatter(doses, residuals, color=PLOT_STYLE["colors"][2])
    ax.set_xlabel("Dose (J/m²)")
    ax.set_ylabel("Residual (log10)")
    ax.set_title(f"{organism} residuals")
    style_axes(ax)
    save_fig(fig, outdir, f"{organism.replace(' ', '_')}_uvc_{filename_suffix}.png")


def plot_qq(residuals, organism, outdir):
    """Plot QQ-plot of residuals.

    Args:
        residuals: Array of residuals.
        organism: Organism name.
        outdir: Output directory.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(5,5))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{organism} residuals QQ-plot")
    save_fig(fig, outdir, f"{organism.replace(' ', '_')}_uvc_residuals_qq.png")
