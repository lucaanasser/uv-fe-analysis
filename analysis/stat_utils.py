import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from numpy.typing import ArrayLike

try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.regression.mixed_linear_model import MixedLM  # type: ignore
except Exception:  # pragma: no cover
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
