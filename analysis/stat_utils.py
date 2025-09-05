import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from numpy.typing import ArrayLike
from dataclasses import dataclass

from scipy import optimize  
from scipy import stats  

try:
    import statsmodels.api as sm 
    from statsmodels.regression.mixed_linear_model import MixedLM  # type: ignore
except Exception: 
    sm = None
    MixedLM = None


def bootstrap(values: ArrayLike, func=np.mean, n_boot: int = 5000, ci: float = 95, random_state: int = 42):
    """Simple bootstrap to estimate a statistic and its confidence interval.

    Args:
        values: Samples (array-like).
        func: Statistic function applied to each bootstrap sample (default mean).
        n_boot: Number of bootstrap replicates.
        ci: Two-sided confidence interval (%).
        random_state: Random generator seed.

    Returns:
        Dict with keys point, ci_low, ci_high.
    """
    rng = np.random.default_rng(random_state)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    stat = func(boots, axis=1)
    low, high = np.percentile(stat, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"point": float(func(arr)), "ci_low": float(low), "ci_high": float(high)}


def rule_of_three_zero_events(n: int):
    """Apply the rule of three for zero events (approx 95% upper bound).

    Args:
        n: Number of independent trials.

    Returns:
        Approximate upper bound of true proportion.
    """
    if n <= 0:
        return np.nan
    return 3 / n


def fit_log_linear(doses, log10_surv):
    """Unweighted log-linear fit (line through origin).

    Args:
        doses: Doses (array-like).
        log10_surv: log10 of survival fraction.

    Returns:
        Dict with slope, slope_se, D10, D10_se, R2_adj, AIC and support vectors.
    """
    x = np.asarray(doses, dtype=float)
    y = np.asarray(log10_surv, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"slope": np.nan, "slope_se": np.nan, "D10": np.nan, "D10_se": np.nan, "R2_adj": np.nan, "AIC": np.nan}
    sx2 = np.sum(x * x)
    slope = np.sum(x * y) / sx2
    y_hat = slope * x
    resid = y - y_hat
    df = x.size - 1
    SSE = np.sum(resid ** 2)
    SST = np.sum((y - np.mean(y)) ** 2)
    mse = SSE / df if df > 0 else np.nan
    slope_se = np.sqrt(mse / sx2) if np.isfinite(mse) else np.nan
    R2 = 1 - SSE / SST if SST > 0 else np.nan
    R2_adj = 1 - (1 - R2) * (x.size - 1) / (x.size - 1 - 1) if x.size > 2 and np.isfinite(R2) else R2
    AIC = x.size * np.log(SSE / x.size) + 2 * 1 if SSE > 0 else np.nan
    if slope >= 0:
        D10 = np.nan
        D10_se = np.nan
    else:
        D10 = -1.0 / slope
        D10_se = slope_se / (slope ** 2) if np.isfinite(slope_se) else np.nan
    return {
        "slope": float(slope),
        "slope_se": float(slope_se),
        "D10": float(D10) if D10 is not None else np.nan,
        "D10_se": float(D10_se) if D10_se is not None else np.nan,
        "R2_adj": float(R2_adj) if R2_adj is not None else np.nan,
        "AIC": float(AIC) if AIC is not None else np.nan,
        "residuals": resid,
        "x": x,
        "y": y,
        "y_hat": y_hat,
    }


def approximate_log10_surv_variance(CFU_dose, CFU0):
    """Approximate variance of log10(survival) assuming Poisson counts.

    Var(log10 S) ≈ (1/ln 10)^2 (1/CFU_dose + 1/CFU0).

    Args:
        CFU_dose: CFU at each dose.
        CFU0: Reference CFU (dose 0) for each observation.

    Returns:
        Array of approximate variances.
    """
    CFU_dose = np.clip(np.asarray(CFU_dose, dtype=float), 1e-12, None)
    CFU0 = np.clip(np.asarray(CFU0, dtype=float), 1e-12, None)
    return (1 / (np.log(10) ** 2)) * (1 / CFU_dose + 1 / CFU0)


def fit_log_linear_wls(doses, log10_surv, var_log10):
    """Weighted least squares fit (through origin) with known variances.

    Args:
        doses: Doses.
        log10_surv: log10 survival.
        var_log10: Variances of log10_surv.

    Returns:
        Dict with slope, slope_se, R2_adj, AIC, residuals and weights.
    """
    x = np.asarray(doses, dtype=float)
    y = np.asarray(log10_surv, dtype=float)
    v = np.asarray(var_log10, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(v) & (v > 0)
    x = x[mask]
    y = y[mask]
    v = v[mask]
    if x.size < 2:
        return {"slope": np.nan, "slope_se": np.nan, "R2_adj": np.nan, "AIC": np.nan, "weights": np.array([]), "residuals": np.array([]), "x": x, "y": y, "y_hat": np.array([])}
    w = 1 / v
    sx2w = np.sum(w * x * x)
    slope = np.sum(w * x * y) / sx2w
    y_hat = slope * x
    resid = y - y_hat
    # Weighted SSE & variance of slope
    SSEw = np.sum(w * resid ** 2)
    dof = x.size - 1
    mse_w = SSEw / dof if dof > 0 else np.nan
    slope_se = np.sqrt(1 / sx2w * mse_w) if np.isfinite(mse_w) else np.nan
    # Weighted R2 (pseudo)
    y_bar_w = np.sum(w * y) / np.sum(w)
    SSTw = np.sum(w * (y - y_bar_w) ** 2)
    R2w = 1 - SSEw / SSTw if SSTw > 0 else np.nan
    R2w_adj = 1 - (1 - R2w) * (x.size - 1) / (x.size - 1 - 1) if x.size > 2 and np.isfinite(R2w) else R2w
    AICw = x.size * np.log(SSEw / x.size) + 2 * 1 if SSEw > 0 else np.nan
    return {
        "slope": float(slope),
        "slope_se": float(slope_se),
        "R2_adj": float(R2w_adj) if R2w_adj is not None else np.nan,
        "AIC": float(AICw) if AICw is not None else np.nan,
        "residuals": resid,
        "weights": w,
        "x": x,
        "y": y,
        "y_hat": y_hat,
    }


def fit_weibull(doses, surv_frac, max_iter=2000):
    """Estimate Weibull parameters (delta, p) via linearization.

    Args:
        doses: Doses > 0.
        surv_frac: Survival fractions (0-1).
        max_iter: (Not used, reserved for future use).

    Returns:
        Dict with delta and p.
    """
    x = np.asarray(doses, dtype=float)
    s = np.asarray(surv_frac, dtype=float)
    mask = (x > 0) & (s > 0) & (s < 1)
    x = x[mask]
    s = s[mask]
    if x.size < 3:
        return {"delta": np.nan, "p": np.nan}
    Y = np.log(-np.log(s))
    X = np.log(x)
    A = np.vstack([X, np.ones_like(X)]).T
    p, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    # From Y = p*log(x) + b ; b = -p*log(delta) => delta = exp(-b/p)
    delta = np.exp(-b / p) if p != 0 else np.nan
    return {"delta": float(delta), "p": float(p)}


# =============================================================
# Additional inactivation model fits (for peer-review robustness)
# =============================================================

def _safe_aic(n_obs: int, sse: float, k_params: int) -> float:
    if not np.isfinite(sse) or sse <= 0 or n_obs <= 0:
        return np.nan
    return n_obs * np.log(sse / n_obs) + 2 * k_params


def fit_power_loglog(doses, surv_frac):
    """Fit power-law (1/x-type) model in log-log space excluding dose 0.

    Model: log10(S) = a + b * log10(dose)
    If b == -1 e^{a} equiv a curva ~ 1/dose.

    Returns dict with a, b, slope_log10_equivalent (per J/m2 at mean dose), AIC, R2.
    """
    x = np.asarray(doses, dtype=float)
    s = np.asarray(surv_frac, dtype=float)
    mask = (x > 0) & (s > 0) & (s <= 1)
    x = x[mask]
    s = s[mask]
    if x.size < 3:
        return {"a": np.nan, "b": np.nan, "AIC": np.nan, "R2": np.nan}
    X = np.log10(x)
    Y = np.log10(s)
    A = np.vstack([np.ones_like(X), X]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    a, b = coeffs
    y_hat = a + b * X
    resid = Y - y_hat
    SSE = np.sum(resid**2)
    SST = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - SSE / SST if SST > 0 else np.nan
    AIC = _safe_aic(x.size, SSE, 2)
    return {"a": float(a), "b": float(b), "AIC": float(AIC), "R2": float(R2)}


def fit_biphasic(doses, surv_frac):
    """Fit biphasic model: S = f*exp(-k1*D) + (1-f)*exp(-k2*D).

    Constraints: 0<f<1; k1>0, k2>0.
    Returns dict with f,k1,k2,AIC (log10 residual SSE).
    """
    x = np.asarray(doses, dtype=float)
    s = np.asarray(surv_frac, dtype=float)
    mask = (x >= 0) & (s > 0) & (s <= 1)
    x = x[mask]
    s = s[mask]
    if x.size < 5:  
        return {"f": np.nan, "k1": np.nan, "k2": np.nan, "AIC": np.nan}

    def model(D, f, k1, k2):
        return f * np.exp(-k1 * D) + (1 - f) * np.exp(-k2 * D)

    guess = [0.7, 1e-3, 1e-4]

    bounds = ([1e-3, 1e-8, 1e-8], [0.999, 10, 10])
    try:
        popt, _ = optimize.curve_fit(model, x, s, p0=guess, bounds=bounds, maxfev=20000)
        f, k1, k2 = popt
        pred = model(x, f, k1, k2)
        Y = np.log10(s)
        Yhat = np.log10(np.clip(pred, 1e-12, 1))
        resid = Y - Yhat
        SSE = np.sum(resid**2)
        AIC = _safe_aic(x.size, SSE, 3)
        return {"f": float(f), "k1": float(k1), "k2": float(k2), "AIC": float(AIC)}
    except Exception:
        return {"f": np.nan, "k1": np.nan, "k2": np.nan, "AIC": np.nan}


def fit_shoulder(doses, surv_frac):
    """Fit simple shoulder model (Geeraerd-type simplified):

    S = 1                           if D <= D0
        exp(-k*(D-D0))              if D >  D0

    Fit via nonlinear least squares on survival fraction.
    Returns dict with D0, k, AIC (log10 residual SSE).
    """
    x = np.asarray(doses, dtype=float)
    s = np.asarray(surv_frac, dtype=float)
    mask = (x >= 0) & (s > 0) & (s <= 1)
    x = x[mask]
    s = s[mask]
    if x.size < 5:
        return {"D0": np.nan, "k": np.nan, "AIC": np.nan}

    def model(D, D0, k):
        D = np.asarray(D)
        out = np.where(D <= D0, 1.0, np.exp(-k * (D - D0)))
        return out

    guess = [x.min() * 0.2, 1e-3]
    bounds = ([0, 1e-8], [x.max() * 0.8, 1])
    try:
        popt, _ = optimize.curve_fit(model, x, s, p0=guess, bounds=bounds, maxfev=20000)
        D0, k = popt
        pred = model(x, D0, k)
        Y = np.log10(s)
        Yhat = np.log10(np.clip(pred, 1e-12, 1))
        resid = Y - Yhat
        SSE = np.sum(resid**2)
        AIC = _safe_aic(x.size, SSE, 2)
        return {"D0": float(D0), "k": float(k), "AIC": float(AIC)}
    except Exception:
        return {"D0": np.nan, "k": np.nan, "AIC": np.nan}


def fit_glm_counts(df: pd.DataFrame):
    """Fit Poisson and Negative Binomial GLMs on raw colony counts.

    Expects columns: colonies, dose_J_m2, dilution_log, plated_uL.
    Uses offset = log( dilution_factor * plated_mL ).
    Returns dict with poisson_slope_log10, nb_slope_log10, their SEs, AICs, dispersion.
    """
    if sm is None:
        return {k: np.nan for k in [
            'poisson_slope_log10','poisson_slope_log10_se','AIC_poisson','dispersion_poisson',
            'nb_slope_log10','nb_slope_log10_se','AIC_nb','model_preferred_glm'
        ]}
    sub = df.dropna(subset=["colonies", "dose_J_m2", "dilution_log", "plated_uL"]).copy()
    if sub.empty or sub["dose_J_m2"].nunique() < 2:
        return {k: np.nan for k in [
            'poisson_slope_log10','poisson_slope_log10_se','AIC_poisson','dispersion_poisson',
            'nb_slope_log10','nb_slope_log10_se','AIC_nb','model_preferred_glm'
        ]}
    sub["dilution_factor"] = 10 ** np.abs(sub["dilution_log"])
    sub["plated_mL"] = sub["plated_uL"] / 1000.0
    sub["offset"] = np.log(sub["dilution_factor"] * sub["plated_mL"])
    try:
        X = sm.add_constant(sub["dose_J_m2"])  # intercept
        poisson_model = sm.GLM(sub["colonies"], X, family=sm.families.Poisson(), offset=sub["offset"]).fit()
        beta1 = poisson_model.params.get("dose_J_m2", np.nan)
        beta1_se = poisson_model.bse.get("dose_J_m2", np.nan)
        mu_hat = poisson_model.fittedvalues
        pearson = np.sum((sub["colonies"] - mu_hat) ** 2 / mu_hat)
        dispersion = pearson / (sub.shape[0] - 2) if sub.shape[0] > 2 else np.nan
        try:
            nb_model = sm.GLM(sub["colonies"], X, family=sm.families.NegativeBinomial(), offset=sub["offset"]).fit()
            nb_beta1 = nb_model.params.get("dose_J_m2", np.nan)
            nb_beta1_se = nb_model.bse.get("dose_J_m2", np.nan)
            AIC_nb = nb_model.aic
        except Exception:
            nb_beta1 = np.nan
            nb_beta1_se = np.nan
            AIC_nb = np.nan
        slope_log10 = beta1 / np.log(10) if np.isfinite(beta1) else np.nan
        slope_log10_se = beta1_se / np.log(10) if np.isfinite(beta1_se) else np.nan
        nb_slope_log10 = nb_beta1 / np.log(10) if np.isfinite(nb_beta1) else np.nan
        nb_slope_log10_se = nb_beta1_se / np.log(10) if np.isfinite(nb_beta1_se) else np.nan
        model_pref = None
        if np.isfinite(AIC_nb) and poisson_model.aic:
            model_pref = 'NB' if AIC_nb + 2 < poisson_model.aic else 'Poisson'
        return {
            'poisson_slope_log10': float(slope_log10),
            'poisson_slope_log10_se': float(slope_log10_se),
            'AIC_poisson': float(poisson_model.aic),
            'dispersion_poisson': float(dispersion),
            'nb_slope_log10': float(nb_slope_log10),
            'nb_slope_log10_se': float(nb_slope_log10_se),
            'AIC_nb': float(AIC_nb) if np.isfinite(AIC_nb) else np.nan,
            'model_preferred_glm': model_pref
        }
    except Exception:
        return {k: np.nan for k in [
            'poisson_slope_log10','poisson_slope_log10_se','AIC_poisson','dispersion_poisson',
            'nb_slope_log10','nb_slope_log10_se','AIC_nb','model_preferred_glm'
        ]}


# =============================================================
# ANOVA & lack-of-fit tests
# =============================================================

def one_way_anova(doses, responses):
    """One-way ANOVA of responses grouped by exact dose values.

    Args:
        doses: array-like dose values (categorical levels for ANOVA).
        responses: array-like response (e.g., log10_surv per replicate).

    Returns dict with F, p, df_between, df_within.
    """
    x = np.asarray(doses)
    y = np.asarray(responses, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2:
        return {"anova_F": np.nan, "anova_p": np.nan, "df_between": 0, "df_within": 0}
    levels = np.unique(x)
    k = levels.size
    groups = [y[x == lv] for lv in levels]
    n_total = y.size
    overall_mean = np.mean(y)
    ss_between = sum(g.size * (np.mean(g) - overall_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0 or ss_within <= 0:
        return {"anova_F": np.nan, "anova_p": np.nan, "df_between": df_between, "df_within": df_within}
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    F = ms_between / ms_within if ms_within > 0 else np.nan
    p = 1 - stats.f.cdf(F, df_between, df_within) if np.isfinite(F) else np.nan
    return {"anova_F": float(F), "anova_p": float(p), "df_between": int(df_between), "df_within": int(df_within)}


def lack_of_fit_test_through_origin(doses, responses, slope):
    """Lack-of-fit F-test for model y = slope * dose (no intercept) using replicate pure error.

    Args:
        doses: dose values (can repeat).
        responses: observed log10_surv values (per replicate).
        slope: fitted slope from all data.

    Returns dict with F_lof, p_lof, df_lof, df_pe.
    """
    x = np.asarray(doses, dtype=float)
    y = np.asarray(responses, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    unique_doses = np.unique(x)
    k = unique_doses.size
    if k < 2:
        return {"lof_F": np.nan, "lof_p": np.nan, "df_lof": 0, "df_pe": 0}
    y_hat = slope * x
    resid = y - y_hat
    sse_model = np.sum(resid ** 2)
    sse_pe = 0.0
    df_pe = 0
    for d in unique_doses:
        yd = y[x == d]
        if yd.size > 1:
            sse_pe += np.sum((yd - np.mean(yd)) ** 2)
            df_pe += yd.size - 1
    df_lof = k - 1  
    ss_lof = sse_model - sse_pe
    if df_pe <= 0 or ss_lof < 0:
        return {"lof_F": np.nan, "lof_p": np.nan, "df_lof": int(df_lof), "df_pe": int(df_pe)}
    ms_lof = ss_lof / df_lof if df_lof > 0 else np.nan
    ms_pe = sse_pe / df_pe if df_pe > 0 else np.nan
    F = ms_lof / ms_pe if (ms_pe and ms_pe > 0) else np.nan
    p = 1 - stats.f.cdf(F, df_lof, df_pe) if np.isfinite(F) else np.nan
    return {"lof_F": float(F), "lof_p": float(p), "df_lof": int(df_lof), "df_pe": int(df_pe)}
    sub = df.dropna(subset=["colonies", "dose_J_m2", "dilution_log", "plated_uL"]).copy()
    if sub.empty or sub["dose_J_m2"].nunique() < 2:
        return {k: np.nan for k in [
            'poisson_slope_log10','poisson_slope_log10_se','AIC_poisson','dispersion_poisson',
            'nb_slope_log10','nb_slope_log10_se','AIC_nb','model_preferred_glm'
        ]}
    # Offset construction
    sub["dilution_factor"] = 10 ** np.abs(sub["dilution_log"])
    sub["plated_mL"] = sub["plated_uL"] / 1000.0
    sub["offset"] = np.log(sub["dilution_factor"] * sub["plated_mL"])  # log exposure volume
    try:
        # Poisson GLM
        X = sm.add_constant(sub["dose_J_m2"])  # intercept + slope
        poisson_model = sm.GLM(sub["colonies"], X, family=sm.families.Poisson(), offset=sub["offset"]).fit()
        beta1 = poisson_model.params.get("dose_J_m2", np.nan)
        beta1_se = poisson_model.bse.get("dose_J_m2", np.nan)
        # Dispersion (Pearson)
        mu_hat = poisson_model.fittedvalues
        pearson = np.sum(((sub["colonies"] - mu_hat) ** 2 - mu_hat) / mu_hat)
        dispersion = pearson / (sub.shape[0] - 2) if sub.shape[0] > 2 else np.nan
        # NB GLM
        try:
            nb_model = sm.GLM(sub["colonies"], X, family=sm.families.NegativeBinomial(), offset=sub["offset"]).fit()
            nb_beta1 = nb_model.params.get("dose_J_m2", np.nan)
            nb_beta1_se = nb_model.bse.get("dose_J_m2", np.nan)
            AIC_nb = nb_model.aic
        except Exception:
            nb_beta1 = np.nan
            nb_beta1_se = np.nan
            AIC_nb = np.nan
        # Convert slopes to log10 survival slopes: log(mu_d / mu_0) = beta1 * dose; log10(S) = beta1/ln10 * dose
        slope_log10 = beta1 / np.log(10) if np.isfinite(beta1) else np.nan
        slope_log10_se = beta1_se / np.log(10) if np.isfinite(beta1_se) else np.nan
        nb_slope_log10 = nb_beta1 / np.log(10) if np.isfinite(nb_beta1) else np.nan
        nb_slope_log10_se = nb_beta1_se / np.log(10) if np.isfinite(nb_beta1_se) else np.nan
        model_pref = None
        if np.isfinite(AIC_nb) and poisson_model.aic:
            model_pref = 'NB' if AIC_nb + 2 < poisson_model.aic else 'Poisson'
        return {
            'poisson_slope_log10': float(slope_log10),
            'poisson_slope_log10_se': float(slope_log10_se),
            'AIC_poisson': float(poisson_model.aic),
            'dispersion_poisson': float(dispersion),
            'nb_slope_log10': float(nb_slope_log10),
            'nb_slope_log10_se': float(nb_slope_log10_se),
            'AIC_nb': float(AIC_nb) if np.isfinite(AIC_nb) else np.nan,
            'model_preferred_glm': model_pref
        }
    except Exception:
        return {k: np.nan for k in [
            'poisson_slope_log10','poisson_slope_log10_se','AIC_poisson','dispersion_poisson',
            'nb_slope_log10','nb_slope_log10_se','AIC_nb','model_preferred_glm'
        ]}


def compute_cooks_distance_through_origin(x, y, resid):
    """Cook's distance for model y = b x (no intercept).

    Args:
        x: Predictor values.
        y: Observed values (not used directly, kept for interface compatibility).
        resid: Residuals from the fit.

    Returns:
        Array with Cook's distance per observation.
    """
    x = np.asarray(x)
    resid = np.asarray(resid)
    sx2 = np.sum(x * x)
    h = (x * x) / sx2
    p = 1
    n = x.size
    MSE = np.sum(resid ** 2) / (n - p) if n - p > 0 else np.nan
    if not np.isfinite(MSE) or MSE <= 0:
        return np.full_like(x, np.nan)
    D = (resid ** 2 / (p * MSE)) * (h / (1 - h) ** 2)
    return D


def fit_mixed_model(df: pd.DataFrame):
    """Fit mixed model with random intercept.

    Args:
        df: DataFrame containing log10_survival, dose_J_m2, experiment.

    Returns:
        Dict with mixed_slope, mixed_slope_se, var_intercept, var_slope, AIC (NaN if fails/insufficient data).
    """
    if sm is None or MixedLM is None:
        return {"mixed_slope": np.nan, "mixed_slope_se": np.nan, "var_intercept": np.nan, "var_slope": np.nan, "AIC": np.nan}
    sub = df.dropna(subset=["log10_survival", "dose_J_m2", "experiment"]).copy()
    if sub["experiment"].nunique() < 2:
        return {"mixed_slope": np.nan, "mixed_slope_se": np.nan, "var_intercept": np.nan, "var_slope": np.nan, "AIC": np.nan}
    try:
        model = MixedLM(sub["log10_survival"], sm.add_constant(sub["dose_J_m2"]), groups=sub["experiment"])
        res = model.fit()
        slope = res.params.get("dose_J_m2", np.nan)
        slope_se = res.bse.get("dose_J_m2", np.nan)
        var_intercept = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] > 0 else np.nan
        return {"mixed_slope": float(slope), "mixed_slope_se": float(slope_se), "var_intercept": var_intercept, "var_slope": np.nan, "AIC": float(res.aic)}
    except Exception:
        return {"mixed_slope": np.nan, "mixed_slope_se": np.nan, "var_intercept": np.nan, "var_slope": np.nan, "AIC": np.nan}
