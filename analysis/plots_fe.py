"""Plotting functions specific to Fe analysis."""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cfu_comprehensive(df_org, organism: str, outdir: str, control_label: str = "control"):
    """Plot log CFU/mL bars comparing treatment vs control.

    Args:
        df_org: DataFrame for a single organism containing treatment, replicate, CFU_per_mL, colonies.
        organism: Organism name.
        outdir: Output directory.
        control_label: Control treatment label.

    Returns:
        None. Saves figure.
    """
    agg = (
        df_org.groupby(["treatment", "replicate"])["CFU_per_mL"].mean().reset_index()
    )
    order = [control_label] + [t for t in agg["treatment"].unique() if t != control_label]
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    means = []
    errors = []
    x_positions = []
    bar_positions = []
    bar_means = []
    bar_colors = []
    detection_floor = 0.1
    pseudo_zero_value = detection_floor * 1.1
    zero_bar_indices: list[int] = []
    real_max = 0
    for i, treatment in enumerate(order):
        data = agg[agg["treatment"] == treatment]["CFU_per_mL"].values
        non_zero_data = data[data > 0]
        if len(non_zero_data) > 0:
            mean_val = float(np.mean(non_zero_data))
            real_max = max(real_max, mean_val)
            means.append(mean_val)
            bar_positions.append(i)
            bar_means.append(mean_val)
            bar_colors.append(colors[i % len(colors)])
            if len(non_zero_data) > 1:
                sem = np.std(non_zero_data, ddof=1) / np.sqrt(len(non_zero_data))
                error_val = 1.96 * sem
                errors.append([error_val, error_val])
            else:
                errors.append([0, 0])
        else:
            means.append(None)
            bar_positions.append(i)
            bar_means.append(pseudo_zero_value)
            bar_colors.append("#999999")
            zero_bar_indices.append(len(bar_positions) - 1)
            errors.append([0, 0])
            ax.text(
                i,
                pseudo_zero_value * 1.25,
                "no colonies",
                ha="center",
                va="bottom",
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.75),
            )
        x_positions.append(i)
    if bar_positions:
        bars = ax.bar(
            bar_positions,
            bar_means,
            color=bar_colors,
            width=0.55,
            edgecolor="black",
            linewidth=1,
        )
        for idx in zero_bar_indices:
            bars[idx].set_hatch("////")
    error_idx = 0
    for i, treatment in enumerate(order):
        data = agg[agg["treatment"] == treatment]["CFU_per_mL"].values
        non_zero_data = data[data > 0]
        if len(non_zero_data) > 0 and error_idx < len(errors):
            error = errors[error_idx]
            mean = bar_means[error_idx]
            if error[1] > 0:
                ax.errorbar(
                    i,
                    mean,
                    yerr=error[1],
                    color="black",
                    capsize=12,
                    capthick=1.1,
                    linewidth=1.1,
                    alpha=0.95,
                )
            error_idx += 1
    ax.set_yscale("log")
    max_with_error = 0
    for idx, mean_val in enumerate(bar_means):
        if idx in zero_bar_indices:
            continue
        error_up = 0
        if idx < len(errors):
            error_up = errors[idx][1]
        max_with_error = max(max_with_error, mean_val + error_up)
    if max_with_error > 0:
        upper = max_with_error * 1.6
        log_decades = np.log10(upper) - np.log10(detection_floor)
        if log_decades < 2:
            upper = 10 ** (np.log10(max_with_error) + 0.3)
        ax.set_ylim(detection_floor, upper)
    else:
        ax.set_ylim(detection_floor, detection_floor * 10)
    ax.set_xlabel("Treatment", fontsize=12, weight="bold")
    ax.set_ylabel("CFU/mL (log scale)", fontsize=12, weight="bold")
    ax.set_title(f"{organism} — Fe³⁺ resistance", fontsize=14, weight="bold", pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([t.replace("Fe3+", "Fe³⁺") for t in order])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    if len(bar_means) > 0 and len(order) > 0:
        control_data = agg[agg["treatment"] == order[0]]["CFU_per_mL"].values
        control_non_zero = control_data[control_data > 0]
        if len(control_non_zero) > 0:
            control_mean = np.mean(control_non_zero)
            ax.axhline(y=control_mean, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(
        os.path.join(outdir, f"{organism.replace(' ', '_').replace('.', '')}_fe_resistance.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
