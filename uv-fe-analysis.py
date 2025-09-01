

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Model, Parameters
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import NegativeBinomial, Binomial
import os
import json




def compute_cfu(colonies, dilution_log, plated_uL):
    """Calculate CFU/mL from colony count, dilution log, and plated volume (µL)."""
    dilution_factor = 10 ** abs(dilution_log)
    return colonies / (plated_uL / 1000.0) * dilution_factor


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['fluence_Jm2'] = df['fluence_Jm2'].astype(float)
    if {'colonies', 'dilution_log', 'plated_uL'}.issubset(df.columns):
        df['CFU_per_mL'] = compute_cfu(df['colonies'], df['dilution_log'], df['plated_uL'])
    else:
        raise ValueError("CSV must contain columns: colonies, dilution_log, plated_uL")
    return df

# ---------------------------------------------
# Função: calcular viabilidade (N/N0) por condição
# ---------------------------------------------


def compute_viability(df, control_label='control'):
    """Calculate relative viability (N/N0) using CFU_per_mL."""
    df = df.copy()
    controls = df[(df['fluence_Jm2']==0) & (df['treatment']==control_label)]
    n0 = controls.groupby(['organism','replicate'])['CFU_per_mL'].mean().reset_index().rename(columns={'CFU_per_mL':'N0'})
    df = df.merge(n0, on=['organism','replicate'], how='left')
    df['viability'] = df['CFU_per_mL'] / df['N0']
    df.loc[df['N0']<=0,'viability'] = np.nan
    return df

"""
Brain-Cousens (hormesis) model:
Flexible function for viability curves, with LD50, slope, hormesis amplitude, and upper asymptote.
Uses: viability = d * 1/(1 + (x/LD50)**b) + f * exp(-g*x)
"""
def brain_cousens_func(x, ld50, b, d, f, g):
    # ld50 >0, b>0, d ~1 (control)
    return d / (1.0 + (x / (ld50 + 1e-12))**b) + f * np.exp(-g * x)

def fit_brain_cousens(x, y, p0=None):
    model = Model(brain_cousens_func)
    params = Parameters()
    params.add('ld50', value=p0.get('ld50', 100.0), min=1e-6)
    params.add('b', value=p0.get('b', 1.0), min=0.01)
    params.add('d', value=p0.get('d', 1.0), min=0.0)
    params.add('f', value=p0.get('f', 0.0))
    params.add('g', value=p0.get('g', 0.001), min=0.0)
    out = model.fit(y, params, x=x)
    return out

"""
Bootstrap LD50 and confidence intervals
"""
def bootstrap_ld50(x, y, fit_func, n_boot=2000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    ld50s = []
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            out = fit_func(x[idx], y[idx])
            ld50s.append(out.params['ld50'].value)
        except Exception:
            continue
    arr = np.array(ld50s)
    return {'ld50_median': np.median(arr), 'ld50_mean': np.mean(arr),
            'ld50_ci95': np.percentile(arr, [2.5,97.5]),
            'raw': arr}

"""
Compare models: hormetic vs non-hormetic (set f=0)
Uses AIC from lmfit
"""
def compare_models(result_hormetic, result_nohorm):
    aic_h = result_hormetic.aic
    aic_n = result_nohorm.aic
    delta_aic = aic_n - aic_h
    return {'aic_hormetic': aic_h, 'aic_nohormetic': aic_n, 'delta_aic': delta_aic}

"""
Baranyi growth model fitting (simple parametric)
Uses logistic with lag approximation
"""
def logistic_with_lag(t, N0, ymax, mu, lag):
    t_eff = np.maximum(t - lag, 0)
    K = ymax
    r = mu
    Nt = N0 * np.exp(r * t_eff) / (1 + (N0 / K) * (np.exp(r * t_eff) - 1))
    return Nt

def fit_growth(t, N, p0=None):
    model = Model(logistic_with_lag)
    params = Parameters()
    params.add('N0', value=p0.get('N0', N.min()), min=1e-6)
    params.add('ymax', value=p0.get('ymax', N.max()*1.2), min=1e-6)
    params.add('mu', value=p0.get('mu', 0.5), min=1e-6)
    params.add('lag', value=p0.get('lag', 1.0), min=0.0)
    out = model.fit(N, params, t=t)
    return out

"""
Presence/absence stats: Fisher exact, Clopper-Pearson
"""
def fisher_test_from_table(a, b, c, d):
    oddsratio, pvalue = stats.fisher_exact([[a,b],[c,d]])
    return oddsratio, pvalue

def clopper_pearson(k, n, alpha=0.05):
    lower = 0.0 if k==0 else stats.beta.ppf(alpha/2, k, n-k+1)
    upper = 1.0 if k==n else stats.beta.ppf(1-alpha/2, k+1, n-k)
    return lower, upper

"""
Plotting utilities
"""
def plot_viability_with_fit(df, organism, treatment, fit_result=None, savepath=None):
    sub = df[(df.organism==organism) & (df.treatment==treatment)]
    x = sub['fluence_Jm2'].values
    y = sub['viability'].values
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, label='data')
    xs = np.linspace(0, max(x)*1.05, 200)
    if fit_result is not None:
        ys = brain_cousens_func(xs,
                                fit_result.params['ld50'].value,
                                fit_result.params['b'].value,
                                fit_result.params['d'].value,
                                fit_result.params['f'].value,
                                fit_result.params['g'].value)
        plt.plot(xs, ys, label='fit')
    plt.xscale('log')
    plt.xlabel('Fluence (J/m²) [log scale]')
    plt.ylabel('Viability (N/N0)')
    plt.title(f'{organism} — {treatment}')
    plt.legend()
    # Always save in results folder
    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    fname = f'viab_{organism}_{treatment}.png'
    savepath = os.path.join(outdir, fname)
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

def forest_ld50(ld_results, savepath=None):
    # ld_results: dict of label -> (ld50, ci_low, ci_high)
    labels = list(ld_results.keys())
    ld = np.array([ld_results[k][0] for k in labels])
    cil = np.array([ld_results[k][1] for k in labels])
    cih = np.array([ld_results[k][2] for k in labels])
    fig, ax = plt.subplots(figsize=(6, max(3, len(labels)*0.5)))
    y = np.arange(len(labels))
    ax.errorbar(ld, y, xerr=[ld-cil, cih-ld], fmt='o')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xscale('log')
    ax.set_xlabel('LD50 (J/m²) [log scale]')
    # Always save in results folder
    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    fname = 'forest_ld50.png'
    savepath = os.path.join(outdir, fname)
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------
# Example pipeline
# ------------------------------------------------
def run_pipeline(csv_path, outdir='results'):
    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    df = load_data(csv_path)
    df = compute_viability(df)

    organisms = df['organism'].unique()
    summary_ld = {}
    for org in organisms:
        for treat in df[df.organism==org]['treatment'].unique():
            sub = df[(df.organism==org)&(df.treatment==treat)]
            sub = sub.dropna(subset=['viability','fluence_Jm2'])
            if len(sub) < 6:
                print(f'Few points for {org} {treat} — skipping fit.')
                continue
            x = sub['fluence_Jm2'].values
            y = sub['viability'].values
            p0 = {'ld50': np.median(x[x>0]) if np.any(x>0) else 100.0, 'b':1.0, 'd':1.0, 'f':0.0, 'g':0.001}
            try:
                res_h = fit_brain_cousens(x,y,p0=p0)
            except Exception as e:
                print('Hormetic fit error:', e)
                continue
            p0_no = p0.copy(); p0_no['f'] = 0.0
            try:
                res_noh = fit_brain_cousens(x,y,p0=p0_no)
            except:
                res_noh = None
            boot = bootstrap_ld50(x,y, lambda xx,yy: fit_brain_cousens(xx,yy,p0=p0), n_boot=800)
            ld50 = res_h.params['ld50'].value
            ci_low, ci_high = boot['ld50_ci95']
            summary_ld[f'{org}_{treat}'] = (ld50, ci_low, ci_high)
            plot_viability_with_fit(sub, org, treat, fit_result=res_h)
    forest_ld50(summary_ld)
    with open(os.path.join(outdir,'ld50_summary.json'),'w') as f:
        json.dump(summary_ld, f, indent=2)
    print('Pipeline finished — results in', outdir)

"""
Main entry point for CLI usage
"""
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV com dados conforme descrito')
    parser.add_argument('--out', default='results', help='pasta de saída')
    args = parser.parse_args()
    run_pipeline(args.csv, outdir=args.out)
