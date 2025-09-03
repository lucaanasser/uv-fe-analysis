import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
