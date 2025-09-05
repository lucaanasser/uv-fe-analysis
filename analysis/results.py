"""Result dataclasses for Fe and UVC analyses.

Centralizes output structures for reuse and serialization.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Any, Dict


@dataclass
class OrganismResult:
    """Per-organism statistical summary (Fe experiment).

    Attributes:
        organism: Organism identifier.
        mean_CFU_control: Mean control CFU.
        std_CFU_control: Std dev control CFU.
        mean_CFU_treatment: Mean treatment CFU.
        std_CFU_treatment: Std dev treatment CFU.
        log10_reduction: log10(mean control / mean treatment) or inf.
        viability_mean: Bootstrap mean viability.
        viability_ci_low: Lower CI bound.
        viability_ci_high: Upper CI bound.
        upper95_viability_if_all_zero: Rule-of-three upper bound if all treatment plates zero.
        detection_limit_CFU_per_mL: Detection limit.
        log10_detection_limit: Log10 detection limit.
        mean_CFU_control_log10: Log10 mean control CFU.
        mean_CFU_treatment_log10: Log10 mean treatment CFU.
        n_replicates: Number of replicates.
    """
    organism: str
    mean_CFU_control: float
    std_CFU_control: float
    mean_CFU_treatment: float
    std_CFU_treatment: float
    log10_reduction: float
    viability_mean: float | None
    viability_ci_low: float | None
    viability_ci_high: float | None
    upper95_viability_if_all_zero: float | None
    detection_limit_CFU_per_mL: float
    log10_detection_limit: float
    mean_CFU_control_log10: float | None
    mean_CFU_treatment_log10: float | None
    n_replicates: int

    def to_dict(self) -> Dict[str, Any]:  # conveniência
        return asdict(self)


@dataclass
class UVCResultMain:
    organism: str
    slope_log10: float
    slope_log10_se: float
    slope_log10_CI95: list
    slope_log10_wls: float | None
    slope_log10_wls_se: float | None
    slope_log10_wls_R2_adj: float | None
    D10: float
    D10_CI95: list
    D10_wls: float | None
    D90: float | None
    D99: float | None
    D999: float | None
    LD50_dose: float | None
    LD50_CI95: list | None
    LD50_wls: float | None
    R2_adj: float | None
    AIC_log_linear: float | None
    AIC_power: float | None
    AIC_biphasic: float | None
    AIC_shoulder: float | None
    power_a: float | None
    power_b: float | None
    biphasic_f: float | None
    biphasic_k1: float | None
    biphasic_k2: float | None
    shoulder_D0: float | None
    shoulder_k: float | None
    model_preferred: str | None
    mixed_slope: float | None
    mixed_slope_se: float | None
    mixed_var_intercept: float | None
    poisson_slope_log10: float | None
    poisson_slope_log10_se: float | None
    nb_slope_log10: float | None
    nb_slope_log10_se: float | None
    AIC_poisson: float | None
    AIC_nb: float | None
    dispersion_poisson: float | None
    model_preferred_glm: str | None
    detection_limit_CFU_per_mL: float
    any_full_zero_dose: bool
    n_doses: int
    anova_F: float | None = None
    anova_p: float | None = None
    df_between: int | None = None
    df_within: int | None = None
    lof_F: float | None = None
    lof_p: float | None = None
    df_lof: int | None = None
    df_pe: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UVCResultSupplement:
    organism: str
    dose_J_m2: float
    mean_CFU_per_mL: float
    mean_survival_fraction: float
    survival_CI95: list
    log10_survival_mean: float
    log10_survival_CI95: list
    log10_reduction_mean: float
    log10_reduction_CI95: list
    upper95_survival_if_all_zero: float | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
