"""
Model Consistency and Bias in Extreme Tails

Auto-converted from Jupyter notebook. Paths are configured via environment
variables / a local ./data directory so this runs outside the original
institutional file system. Place the required input CSVs (see project
README) under DATA_DIR before running.
"""
import os

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "figures"))


# -*- coding: utf-8 -*-
"""
Model consistency and bias in extreme tails (robust)
- Fig4_percentiles.png (always produced from available data)
- Fig5_taylor.png (produced only if finite RL50 exist; otherwise a placeholder panel)

Author: you
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# --------------------------
# Paths / files
# --------------------------
CLUSTER_FOLDER = DATA_DIR
CLUSTER_FILES = {
    "ERA":         "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP":  "ECH_full_filled_with_clusters_k9_named.csv",
}

# Optional declustered events (if present). Adjust patterns if your names differ.
EVENT_FOLDER = CLUSTER_FOLDER
EVENT_PATTERNS = {
    "ERA":         ["*ERA*events*.parquet", "*ERA*events*.csv"],
    "ScenarioMIP": ["*ECN*events*.parquet", "*ScenarioMIP*events*.parquet",
                    "*ECN*events*.csv",     "*ScenarioMIP*events*.csv"],
    "HighResMIP":  ["*ECH*events*.parquet", "*HighResMIP*events*.parquet",
                    "*ECH*events*.csv",     "*HighResMIP*events*.csv"],
}

OUT = CLUSTER_FOLDER
os.makedirs(OUT, exist_ok=True)

# --------------------------
# Config
# --------------------------
REGIME_COL_CAND = ["cluster_abbrev", "cluster", "cluster_name"]
COL_HS, COL_U10 = "Hs", "U10"
PERIOD_YEARS = 36
RL_T_YEARS = 50
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]

# --------------------------
# Helpers
# --------------------------
def detect_reg_col(df):
    for c in REGIME_COL_CAND:
        if c in df.columns:
            return c
    raise KeyError(f"No regime column in {list(df.columns)}")

def load_cluster_csv(ds):
    path = os.path.join(CLUSTER_FOLDER, CLUSTER_FILES[ds])
    df = pd.read_csv(path)
    reg = detect_reg_col(df)
    if "mean_hs" in df.columns:
        df = df.rename(columns={"mean_hs":"Hs_proxy"})
    if "annual_mean_wind_speed" in df.columns:
        df = df.rename(columns={"annual_mean_wind_speed":"U10_proxy"})
    return df, reg

def find_event_file(ds):
    for pat in EVENT_PATTERNS.get(ds, []):
        fs = glob.glob(os.path.join(EVENT_FOLDER, pat))
        if fs:
            return fs[0]
    return None

def load_events(ds, reg_hint=None):
    f = find_event_file(ds)
    if not f:
        return None, None
    if f.lower().endswith(".parquet"):
        df = pd.read_parquet(f)
    else:
        df = pd.read_csv(f)
    reg = reg_hint if (reg_hint and reg_hint in df.columns) else detect_reg_col(df)
    if (COL_HS not in df.columns) or (COL_U10 not in df.columns):
        warnings.warn(f"{os.path.basename(f)} missing '{COL_HS}'/'{COL_U10}'.")
        return None, None
    return df, reg

def p99_by_regime(df, reg, var):
    # observed=False silences the future warning and preserves current behavior
    return df.groupby(reg, observed=False)[var].quantile(0.99).rename("p99").reset_index()

def gpd_rl_by_regime(df, reg, var, T=RL_T_YEARS, period=PERIOD_YEARS):
    out = []
    for rg, sub in df.groupby(reg, observed=False):
        x = sub[var].dropna().values
        if len(x) < 50:
            out.append((rg, np.nan)); continue
        u = np.quantile(x, 0.95)
        y = x[x > u] - u
        if len(y) < 20:
            out.append((rg, np.nan)); continue
        xi, loc, beta = genpareto.fit(y, floc=0)
        lam = len(y)/period
        if (lam <= 0) or (beta <= 0):
            rl = np.nan
        else:
            a = T*lam
            rl = u + (beta*np.log(a) if abs(xi) < 1e-6 else (beta/xi)*(a**xi - 1.0))
        out.append((rg, rl))
    return pd.DataFrame(out, columns=[reg, "RL50"])

def order_regimes(df, reg):
    cats = [c for c in CLUSTER_ORDER if c in set(df[reg])]
    if not cats:
        cats = sorted(df[reg].unique())
    df[reg] = pd.Categorical(df[reg], categories=cats, ordered=True)
    return df.sort_values(reg)

# --------------------------
# Load base CSVs and events
# --------------------------
cluster_dfs, regcol = {}, {}
for ds in ["ERA","ScenarioMIP","HighResMIP"]:
    cluster_dfs[ds], regcol[ds] = load_cluster_csv(ds)

event_dfs = {}
for ds in ["ERA","ScenarioMIP","HighResMIP"]:
    ev, re = load_events(ds, regcol[ds])
    if ev is not None:
        event_dfs[ds] = (ev, re)

# --------------------------
# Build p99 tables (event-level if available; else proxy)
# --------------------------
p99_tabs = []
for ds in ["ERA","ScenarioMIP","HighResMIP"]:
    if ds in event_dfs:
        ev, rc = event_dfs[ds]
        t_hs  = p99_by_regime(ev, rc, COL_HS).rename(columns={"p99":"p99_Hs"})
        t_u10 = p99_by_regime(ev, rc, COL_U10).rename(columns={"p99":"p99_U10"})
        tab = t_hs.merge(t_u10, on=rc, how="outer"); tab["dataset"]=ds
        tab = order_regimes(tab, rc)
        tab = tab.rename(columns={rc:"regime"})
        p99_tabs.append(tab)
    else:
        warnings.warn(f"Using proxy p99 from clustered CSV for {ds} (no event-level file found).")
        base, rc = cluster_dfs[ds], regcol[ds]
        for needed in ("Hs_proxy","U10_proxy"):
            if needed not in base.columns:
                raise KeyError(f"{needed} not found in {ds} CSV.")
        t_hs  = p99_by_regime(base, rc, "Hs_proxy").rename(columns={"p99":"p99_Hs"})
        t_u10 = p99_by_regime(base, rc, "U10_proxy").rename(columns={"p99":"p99_U10"})
        tab = t_hs.merge(t_u10, on=rc, how="outer"); tab["dataset"]=ds
        tab = order_regimes(tab, rc)
        tab = tab.rename(columns={rc:"regime"})
        p99_tabs.append(tab)

p99_all = pd.concat(p99_tabs, ignore_index=True)

# --------------------------
# Build RL50 tables (only if events exist)
# --------------------------
rl_tabs = []
for ds in ["ERA","ScenarioMIP","HighResMIP"]:
    if ds in event_dfs:
        ev, rc = event_dfs[ds]
        r_hs = gpd_rl_by_regime(ev, rc, COL_HS).rename(columns={"RL50":"RL50_Hs"})
        r_u  = gpd_rl_by_regime(ev, rc, COL_U10).rename(columns={"RL50":"RL50_U10"})
        t = r_hs.merge(r_u, on=rc, how="outer"); t["dataset"]=ds
        t = order_regimes(t, rc).rename(columns={rc:"regime"})
        rl_tabs.append(t)
    else:
        warnings.warn(f"RL50 set to NaN for {ds} (no event-level file found). Provide declustered events for GPD fits).")
        base, rc = cluster_dfs[ds], regcol[ds]
        regs = order_regimes(base[[rc]].drop_duplicates(), rc).rename(columns={rc:"regime"})
        regs["dataset"]=ds; regs["RL50_Hs"]=np.nan; regs["RL50_U10"]=np.nan
        rl_tabs.append(regs)

rl_all = pd.concat(rl_tabs, ignore_index=True)

# --------------------------
# FIGURE 4: p99 bars
# --------------------------
def bar_group(df, var, outpng, ylabel, title):
    # pivot with observed=False to keep current pandas behaviour
    dp = df.pivot_table(index="regime", columns="dataset", values=var, aggfunc="first", observed=False)
    cats = [c for c in CLUSTER_ORDER if c in dp.index]
    if not cats: cats = sorted(dp.index)
    dp = dp.reindex(cats)

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    datasets = ["ERA","ScenarioMIP","HighResMIP"]
    x = np.arange(len(dp.index)); width = 0.75/len(datasets)
    colors = {"ERA":"#595959","ScenarioMIP":"#1f77b4","HighResMIP":"#d62728"}
    for i, ds in enumerate(datasets):
        y = dp[ds].values if ds in dp.columns else np.zeros(len(dp.index))
        ax.bar(x + i*width, y, width=width, color=colors.get(ds), edgecolor="black", linewidth=0.6, label=ds)
    ax.set_xticks(x + (len(datasets)-1)*width/2)
    ax.set_xticklabels(dp.index)
    ax.set_xlabel("Regime"); ax.set_ylabel(ylabel); ax.set_title(title, pad=8)
    ax.grid(axis="y", alpha=0.3); ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.savefig(outpng, dpi=300); plt.close(fig)

bar_group(
    p99_all, "p99_Hs",
    os.path.join(OUT, "Fig4_percentiles_Hs.png"),
    "$H_s$ 99th percentile (m)",
    "Regime-wise 99th-percentile of significant wave height ($H_s$)"
)
bar_group(
    p99_all, "p99_U10",
    os.path.join(OUT, "Fig4_percentiles_U10.png"),
    "$U_{10}$ 99th percentile (m s$^{-1}$)",
    "Regime-wise 99th-percentile of 10-m wind speed ($U_{10}$)"
)

# Combine into a single Fig4_percentiles.png (stacked)
def combine_p99(df, outpng):
    d1 = df[["regime","dataset","p99_Hs"]].dropna()
    d2 = df[["regime","dataset","p99_U10"]].dropna()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    for ax, dsub, var, yl, ttl in [
        (axes[0], d1, "p99_Hs", "$H_s$ 99th percentile (m)", "Regime-wise 99th-percentile of significant wave height ($H_s$)"),
        (axes[1], d2, "p99_U10", "$U_{10}$ 99th percentile (m s$^{-1}$)", "Regime-wise 99th-percentile of 10-m wind speed ($U_{10}$)")
    ]:
        dp = dsub.pivot_table(index="regime", columns="dataset", values=var, aggfunc="first", observed=False)
        cats = [c for c in CLUSTER_ORDER if c in dp.index];  cats = cats or sorted(dp.index)
        dp = dp.reindex(cats)
        datasets = ["ERA","ScenarioMIP","HighResMIP"]; x = np.arange(len(dp.index)); w = 0.75/len(datasets)
        colors = {"ERA":"#595959","ScenarioMIP":"#1f77b4","HighResMIP":"#d62728"}
        for i, ds in enumerate(datasets):
            y = dp[ds].values if ds in dp.columns else np.zeros(len(dp.index))
            ax.bar(x + i*w, y, width=w, color=colors.get(ds), edgecolor="black", linewidth=0.6, label=ds if ax is axes[0] else None)
        ax.set_ylabel(yl); ax.set_title(ttl, pad=8); ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(x + (len(datasets)-1)*w/2); ax.set_xticklabels(dp.index)
    axes[0].legend(frameon=False, ncol=3, loc="upper right"); axes[-1].set_xlabel("Regime")
    fig.savefig(outpng, dpi=300); plt.close(fig)

combine_p99(p99_all, os.path.join(OUT, "Fig4_percentiles.png"))

# --------------------------
# FIGURE 5: RL50 1:1 scatter (robust)
# --------------------------
def rl50_scatter(rl_df, var, outpng, title, ylabel):
    # Pivot to wide for ERA vs models
    era = rl_df[rl_df["dataset"]=="ERA"][["regime", var]].rename(columns={var:"ERA"})
    ecn = rl_df[rl_df["dataset"]=="ScenarioMIP"][["regime", var]].rename(columns={var:"ScenarioMIP"})
    ech = rl_df[rl_df["dataset"]=="HighResMIP"][["regime", var]].rename(columns={var:"HighResMIP"})
    m = era.merge(ecn, on="regime", how="left").merge(ech, on="regime", how="left")

    # Any finite values?
    arr = m[["ERA","ScenarioMIP","HighResMIP"]].to_numpy().astype(float)
    if not np.isfinite(arr).any():
        # Save placeholder figure to keep manuscript workflow consistent
        fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.55, "RL$_{50}$ plot unavailable",
                ha="center", va="center", fontsize=12)
        ax.text(0.5, 0.45, "No declustered event-level data found for GPD fitting.\n"
                           "Provide events (Hs, U10) to compute return levels.",
                ha="center", va="center", fontsize=10)
        fig.suptitle(title, y=0.98)
        fig.savefig(outpng, dpi=300); plt.close(fig)
        print(f"⚠️  Saved placeholder (no RL50 data): {outpng}")
        return

    # Compute axis limits from finite data only
    finite = arr[np.isfinite(arr)]
    lo = max(0, float(np.min(finite)*0.95))
    hi = float(np.max(finite)*1.05)

    # Regime order for annotation
    cats = [c for c in CLUSTER_ORDER if c in m["regime"].unique()]
    if not cats: cats = sorted(m["regime"].unique())
    m["regime"] = pd.Categorical(m["regime"], categories=cats, ordered=True)
    m = m.sort_values("regime")

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="k", alpha=0.7, label="1:1")
    ax.scatter(m["ERA"], m["ScenarioMIP"], s=55, facecolor="#1f77b4", edgecolor="black", linewidth=0.6, label="ScenarioMIP")
    ax.scatter(m["ERA"], m["HighResMIP"], s=55, facecolor="#d62728", edgecolor="black", linewidth=0.6, label="HighResMIP")
    for _, r in m.iterrows():
        if np.isfinite(r["ScenarioMIP"]):
            ax.annotate(str(r["regime"]), (r["ERA"], r["ScenarioMIP"]), xytext=(3,3), textcoords="offset points", fontsize=8, alpha=0.8)
        if np.isfinite(r["HighResMIP"]):
            ax.annotate(str(r["regime"]), (r["ERA"], r["HighResMIP"]), xytext=(3,-10), textcoords="offset points", fontsize=8, alpha=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"ERA5 {ylabel}"); ax.set_ylabel(f"Model {ylabel}")
    ax.set_title(title, pad=8); ax.grid(alpha=0.3); ax.legend(frameon=False, loc="upper left")
    fig.savefig(outpng, dpi=300); plt.close(fig)

# Save RL50 panels (placeholder if needed)
rl50_scatter(rl_all, "RL50_Hs",
             os.path.join(OUT, "Fig5_taylor_Hs.png"),
             "GPD 50-year return levels of $H_s$ (models vs ERA5)",
             "RL$_{50}$($H_s$) (m)")
rl50_scatter(rl_all, "RL50_U10",
             os.path.join(OUT, "Fig5_taylor_U10.png"),
             "GPD 50-year return levels of $U_{10}$ (models vs ERA5)",
             "RL$_{50}$($U_{10}$) (m s$^{-1}$)")

# Optional combine
def combine_two(img1, img2, outpng):
    try:
        from PIL import Image
        if (not os.path.exists(img1)) or (not os.path.exists(img2)):
            return
        a, b = Image.open(img1), Image.open(img2)
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width+b.width, h), (255,255,255))
        canvas.paste(a, (0, 0)); canvas.paste(b, (a.width, 0))
        canvas.save(outpng)
        print(f"✅ Saved {outpng}")
    except Exception as e:
        print(f"Skipping combine: {e}")

combine_two(os.path.join(OUT, "Fig5_taylor_Hs.png"),
            os.path.join(OUT, "Fig5_taylor_U10.png"),
            os.path.join(OUT, "Fig5_taylor.png"))

print("✅ Done. Outputs:")
print(" -", os.path.join(OUT, "Fig4_percentiles.png"))
print(" -", os.path.join(OUT, "Fig5_taylor.png"), "(placeholder if events unavailable)")

# ---- next cell ----

# -*- coding: utf-8 -*-
"""
End-to-end: Model Consistency & Bias in Extreme Tails (PROXY, from k9 CSVs only)

Outputs (in OUT folder):
 - Fig4_percentiles_proxy.png
 - Fig5_taylor_proxy_Hs.png
 - Fig5_taylor_proxy_U10.png
 - Fig5_taylor_proxy.png     (stitched Hs|U10 panel, if PIL available)
 - proxy_extremes_by_regime.csv

CAVEAT (state in Methods & captions):
These are proxy extremes derived from distributional assumptions using regime-level
mean & std (no event-level time-series). Hs ~ Lognormal; U10 ~ Weibull; n=365 samples/yr.
Not a substitute for POT/GEV/GPD with declustered events.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths & dataset filenames
# =========================
FOLDER = DATA_DIR
FILES = {
    "ERA":         "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP":  "ECH_full_filled_with_clusters_k9_named.csv",
}
OUT = FOLDER  # write outputs next to inputs
os.makedirs(OUT, exist_ok=True)

# =========================
# Configuration
# =========================
REGIME_COL_CAND = ["cluster_abbrev", "cluster", "cluster_name"]
CLUSTER_ORDER = ["NWAC", "NEC", "WENS", "WMIC", "EAC", "SMC", "EMC", "SCES", "BSC"]

# columns available in your k9 CSVs (as you showed)
COL_MEAN_HS   = "mean_hs"                  # m (annual)
COL_STD_HS    = "std_hs"                   # m (annual)
COL_MEAN_U10  = "annual_mean_wind_speed"   # m s^-1 (annual)
COL_STD_U10   = "annual_std_wind_speed"    # m s^-1 (annual)

# Seasonal columns (OPTIONAL: for seasonal mixture proxy)
SEAS_MEAN = {
    "DJF": ("mean_hs_DJF",  "mean_wind_speed_DJF"),
    "MAM": ("mean_hs_MAM",  "mean_wind_speed_MAM"),
    "JJA": ("mean_hs_JJA",  "mean_wind_speed_JJA"),
    "SON": ("mean_hs_SON",  "mean_wind_speed_SON"),
}
SEAS_STD = {
    "DJF": ("std_hs_DJF",  "std_wind_speed_DJF"),
    "MAM": ("std_hs_MAM",  "std_wind_speed_MAM"),
    "JJA": ("std_hs_JJA",  "std_wind_speed_JJA"),
    "SON": ("std_hs_SON",  "std_wind_speed_SON"),
}

# “samples per year” to link quantiles to return levels (proxy)
# for daily maxima, n ≈ 365; adjust in sensitivity
N_PER_YEAR = 365
T_YEARS = 50

# Toggle: also compute a seasonal-mixture proxy RL_T (weighted equally across seasons)
DO_SEASONAL_MIX = True   # set False to skip


# =========================
# Helper functions
# =========================
def detect_regime_col(df):
    for c in REGIME_COL_CAND:
        if c in df.columns:
            return c
    raise KeyError("No regime/cluster column found among: " + ", ".join(REGIME_COL_CAND))

def spatial_p99(series):
    """Spatial 99th percentile across grid cells (proxy for 'largest typical' regions)."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, 0.99))

def order_regimes_index(index_vals):
    cats = [c for c in CLUSTER_ORDER if c in set(index_vals)]
    return cats if cats else sorted(index_vals)

# Weibull (U10) parameterization from mean & std (common in wind analysis)
def weibull_k_from_cv(cv):
    # Approximation widely used for 0.1 < CV < ~1
    if cv <= 0 or not np.isfinite(cv):
        return np.nan
    return cv ** (-1.086)

def weibull_scale_from_mean(mu, k):
    from scipy.special import gamma
    if mu <= 0 or k <= 0 or not np.isfinite(mu*k):
        return np.nan
    return mu / gamma(1.0 + 1.0/k)

def weibull_quantile(p, k, c):
    if not (0 < p < 1) or not np.isfinite(k*c) or k <= 0 or c <= 0:
        return np.nan
    return c * (-np.log(1.0 - p)) ** (1.0 / k)

# Lognormal (Hs) parameterization from mean & std
def lognormal_params_from_mean_std(mu, sigma):
    if mu <= 0 or sigma <= 0 or not np.isfinite(mu*sigma):
        return np.nan, np.nan
    s2 = np.log(1.0 + (sigma**2) / (mu**2))
    s = np.sqrt(s2)
    m = np.log(mu) - 0.5*s2
    return m, s

def lognormal_quantile(p, m, s):
    from scipy.stats import norm
    if not (0 < p < 1) or not np.isfinite(m*s) or s <= 0:
        return np.nan
    return np.exp(m + s * norm.ppf(p))

def p_eff_for_return_level(T, n_per_year):
    # effective non-exceedance probability for proxy T-year quantile
    return 1.0 - (1.0 - 1.0 / float(T)) ** (1.0 / float(n_per_year))


# =========================
# Load all three CSVs
# =========================
print(">> Loading clustered CSVs …")
dfs = {}
regcols = {}
for ds, fn in FILES.items():
    path = os.path.join(FOLDER, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{ds}: {path} not found.")
    df = pd.read_csv(path)
    reg = detect_regime_col(df)
    dfs[ds] = df
    regcols[ds] = reg
    print(f"   {ds:<12} {fn}  (rows={len(df):,})  regime_col={reg}")

# =========================
# 1) Spatial p99 (proxy) by regime
# =========================
print("\n>> Computing spatial 99th percentiles (proxy) by regime …")
rows = []
for ds, df in dfs.items():
    reg = regcols[ds]
    g = df.groupby(reg, observed=False)
    p99_hs  = g[COL_MEAN_HS].apply(spatial_p99)
    p99_u10 = g[COL_MEAN_U10].apply(spatial_p99)
    tmp = pd.DataFrame({
        "regime": p99_hs.index,
        "dataset": ds,
        "p99_Hs_spatial": p99_hs.values,
        "p99_U10_spatial": p99_u10.values
    })
    rows.append(tmp)

p99_spatial_all = pd.concat(rows, ignore_index=True)

# =========================
# 2) Proxy RL_T (annual fit) from mean/std per regime
# =========================
print(">> Computing proxy RL_T (annual) per regime (Hs~Lognormal, U10~Weibull) …")
p_eff = p_eff_for_return_level(T_YEARS, N_PER_YEAR)

rows_rl_annual = []
for ds, df in dfs.items():
    reg = regcols[ds]
    grp = df.groupby(reg, observed=False)

    # regime-level summaries (averaging across grid cells)
    mu_hs  = grp[COL_MEAN_HS].mean()
    sd_hs  = grp[COL_STD_HS].mean()
    mu_u10 = grp[COL_MEAN_U10].mean()
    sd_u10 = grp[COL_STD_U10].mean()

    regimes = mu_hs.index.tolist()
    rl_hs = []
    rl_u10 = []
    for r in regimes:
        # Hs ~ Lognormal
        m_ln, s_ln = lognormal_params_from_mean_std(mu_hs.loc[r], sd_hs.loc[r])
        q_hs = lognormal_quantile(p_eff, m_ln, s_ln)

        # U10 ~ Weibull
        mu = mu_u10.loc[r]; sd = sd_u10.loc[r]
        if mu > 0 and sd > 0:
            cv = sd / mu
            k  = weibull_k_from_cv(cv)
            c  = weibull_scale_from_mean(mu, k)
            q_u = weibull_quantile(p_eff, k, c)
        else:
            q_u = np.nan

        rl_hs.append(q_hs)
        rl_u10.append(q_u)

    t = pd.DataFrame({
        "regime": regimes,
        "dataset": ds,
        "RL50_Hs_proxy_annual": rl_hs,
        "RL50_U10_proxy_annual": rl_u10
    })
    rows_rl_annual.append(t)

rl50_proxy_annual = pd.concat(rows_rl_annual, ignore_index=True)

# =========================
# 3) (Optional) Seasonal-mixture proxy RL_T
# =========================
if DO_SEASONAL_MIX:
    print(">> Computing seasonal-mixture proxy RL_T (equal-weight DJF/MAM/JJA/SON) …")
    rows_seas = []
    seas = ["DJF","MAM","JJA","SON"]
    n_season = 365/4.0  # ~90 samples per season
    p_eff_seas = p_eff_for_return_level(T_YEARS, n_season)

    for ds, df in dfs.items():
        reg = regcols[ds]
        grp = df.groupby(reg, observed=False)

        # Prepare regime lists
        regimes = sorted(grp.size().index.tolist())

        # For each season, compute RL_T proxy, then average across seasons (equal weight)
        rl_hs_mix = []
        rl_u10_mix = []
        for r in regimes:
            q_hs_list = []
            q_u_list  = []
            for sea in seas:
                mhs_col, mu10_col = SEAS_MEAN[sea]
                shs_col, su10_col = SEAS_STD[sea]

                # regime-level mean of means and stds (across grid cells)
                mu_hs  = grp[mhs_col].mean().get(r, np.nan)
                sd_hs  = grp[shs_col].mean().get(r, np.nan)
                mu_u10 = grp[mu10_col].mean().get(r, np.nan)
                sd_u10 = grp[su10_col].mean().get(r, np.nan)

                # Hs seasonal quantile
                m_ln, s_ln = lognormal_params_from_mean_std(mu_hs, sd_hs)
                q_hs = lognormal_quantile(p_eff_seas, m_ln, s_ln)

                # U10 seasonal quantile
                if mu_u10 > 0 and sd_u10 > 0 and np.isfinite(mu_u10*sd_u10):
                    cv = sd_u10 / mu_u10
                    k  = weibull_k_from_cv(cv)
                    c  = weibull_scale_from_mean(mu_u10, k)
                    q_u = weibull_quantile(p_eff_seas, k, c)
                else:
                    q_u = np.nan

                q_hs_list.append(q_hs)
                q_u_list.append(q_u)

            # Equal-weight mixture
            rl_hs_mix.append(np.nanmean(q_hs_list))
            rl_u10_mix.append(np.nanmean(q_u_list))

        rows_seas.append(pd.DataFrame({
            "regime": regimes,
            "dataset": ds,
            "RL50_Hs_proxy_seasMix": rl_hs_mix,
            "RL50_U10_proxy_seasMix": rl_u10_mix
        }))

    rl50_proxy_seasMix = pd.concat(rows_seas, ignore_index=True)
else:
    rl50_proxy_seasMix = pd.DataFrame(columns=["regime","dataset","RL50_Hs_proxy_seasMix","RL50_U10_proxy_seasMix"])

# =========================
# 4) Merge all numbers & export CSV
# =========================
proxy_table = (
    p99_spatial_all.merge(rl50_proxy_annual, on=["regime","dataset"], how="outer")
)
if len(rl50_proxy_seasMix):
    proxy_table = proxy_table.merge(rl50_proxy_seasMix, on=["regime","dataset"], how="outer")

out_csv = os.path.join(OUT, "proxy_extremes_by_regime.csv")
proxy_table.to_csv(out_csv, index=False)
print(f">> Wrote table: {out_csv}")

# =========================
# 5) Figures
# =========================

def grouped_bar(ax, df_sub, value_col, ylabel, title):
    dp = df_sub.pivot_table(index="regime", columns="dataset", values=value_col,
                            aggfunc="first", observed=False)
    cats = order_regimes_index(dp.index)
    dp = dp.reindex(cats)

    x = np.arange(len(dp.index))
    datasets = ["ERA","ScenarioMIP","HighResMIP"]
    width = 0.75 / len(datasets)

    # (No custom colors to keep style portable)
    for i, ds in enumerate(datasets):
        y = dp[ds].values if ds in dp.columns else np.zeros(len(dp.index))
        ax.bar(x + i*width, y, width=width, edgecolor="black", linewidth=0.6, label=ds)
    ax.set_xticks(x + (len(datasets)-1)*width/2)
    ax.set_xticklabels(dp.index)
    ax.set_xlabel("Regime")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="upper right")

# Fig 4 (proxy): regime-wise spatial p99 bars for Hs & U10
print(">> Rendering Fig4_percentiles_proxy.png …")
fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
grouped_bar(
    axes[0],
    p99_spatial_all[["regime","dataset","p99_Hs_spatial"]],
    "p99_Hs_spatial",
    "$H_s$ 99th percentile (spatial proxy, m)",
    "Regime-wise spatial 99th percentile of mean significant wave height ($H_s$)"
)
grouped_bar(
    axes[1],
    p99_spatial_all[["regime","dataset","p99_U10_spatial"]],
    "p99_U10_spatial",
    "$U_{10}$ 99th percentile (spatial proxy, m s$^{-1}$)",
    "Regime-wise spatial 99th percentile of mean 10-m wind speed ($U_{10}$)"
)
axes[-1].set_xlabel("Regime")
fig.savefig(os.path.join(OUT, "Fig4_percentiles_proxy.png"), dpi=300)
plt.close(fig)
print("   saved -> Fig4_percentiles_proxy.png")

# Fig 5 (proxy): 1:1 scatter of proxy RL50 (models vs ERA), for Hs and U10
def rl50_proxy_scatter(df_proxy, var, outpng, title, ylabel):
    era = df_proxy[df_proxy["dataset"]=="ERA"][["regime", var]].rename(columns={var:"ERA"})
    ecn = df_proxy[df_proxy["dataset"]=="ScenarioMIP"][["regime", var]].rename(columns={var:"ScenarioMIP"})
    ech = df_proxy[df_proxy["dataset"]=="HighResMIP"][["regime", var]].rename(columns={var:"HighResMIP"})
    m = era.merge(ecn, on="regime", how="left").merge(ech, on="regime", how="left")

    arr = m[["ERA","ScenarioMIP","HighResMIP"]].to_numpy().astype(float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.55, "No finite proxy RL values.", ha="center", va="center", fontsize=12)
        fig.suptitle(title, y=0.98)
        fig.savefig(outpng, dpi=300); plt.close(fig)
        print(f"   saved placeholder -> {os.path.basename(outpng)}")
        return

    lo = max(0, float(np.min(finite)*0.95))
    hi = float(np.max(finite)*1.05)

    cats = order_regimes_index(m["regime"].unique())
    m["regime"] = pd.Categorical(m["regime"], categories=cats, ordered=True)
    m = m.sort_values("regime")

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="k", alpha=0.7, label="1:1")

    # scenario vs ERA
    ax.scatter(m["ERA"], m["ScenarioMIP"], s=55, edgecolor="black", linewidth=0.6, label="ScenarioMIP")
    ax.scatter(m["ERA"], m["HighResMIP"], s=55, edgecolor="black", linewidth=0.6, label="HighResMIP")

    # label points with regime abbreviations
    for _, r in m.iterrows():
        if np.isfinite(r["ScenarioMIP"]):
            ax.annotate(str(r["regime"]), (r["ERA"], r["ScenarioMIP"]),
                        xytext=(3,3), textcoords="offset points", fontsize=8, alpha=0.85)
        if np.isfinite(r["HighResMIP"]):
            ax.annotate(str(r["regime"]), (r["ERA"], r["HighResMIP"]),
                        xytext=(3,-10), textcoords="offset points", fontsize=8, alpha=0.85)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"ERA5 {ylabel}")
    ax.set_ylabel(f"Model {ylabel}")
    ax.set_title(title, pad=8)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(outpng, dpi=300)
    plt.close(fig)
    print(f"   saved -> {os.path.basename(outpng)}")

# Build the proxy RL frame to use (use seasonal mixture if computed, else annual)
if DO_SEASONAL_MIX and len(rl50_proxy_seasMix):
    rl_plot = proxy_table.rename(columns={
        "RL50_Hs_proxy_seasMix": "RL50_Hs_proxy",
        "RL50_U10_proxy_seasMix": "RL50_U10_proxy"
    })
else:
    rl_plot = proxy_table.rename(columns={
        "RL50_Hs_proxy_annual": "RL50_Hs_proxy",
        "RL50_U10_proxy_annual": "RL50_U10_proxy"
    })

print(">> Rendering Fig5 (proxy) …")
rl50_proxy_scatter(
    rl_plot, "RL50_Hs_proxy",
    os.path.join(OUT, "Fig5_taylor_proxy_Hs.png"),
    "Proxy 50-year return levels of $H_s$ (models vs ERA5)",
    "proxy RL$_{50}$($H_s$) (m)"
)
rl50_proxy_scatter(
    rl_plot, "RL50_U10_proxy",
    os.path.join(OUT, "Fig5_taylor_proxy_U10.png"),
    "Proxy 50-year return levels of $U_{10}$ (models vs ERA5)",
    "proxy RL$_{50}$($U_{10}$) (m s$^{-1}$)"
)

# optional: stitch the two panels horizontally into Fig5_taylor_proxy.png
try:
    from PIL import Image
    a = Image.open(os.path.join(OUT, "Fig5_taylor_proxy_Hs.png"))
    b = Image.open(os.path.join(OUT, "Fig5_taylor_proxy_U10.png"))
    H = max(a.height, b.height)
    canvas = Image.new("RGB", (a.width + b.width, H), (255,255,255))
    canvas.paste(a, (0,0)); canvas.paste(b, (a.width,0))
    canvas.save(os.path.join(OUT, "Fig5_taylor_proxy.png"))
    print("   stitched -> Fig5_taylor_proxy.png")
except Exception as e:
    warnings.warn(f"Panel stitch skipped: {e}")

print("\n✅ Done.\n"
      " - Fig4_percentiles_proxy.png\n"
      " - Fig5_taylor_proxy_Hs.png\n"
      " - Fig5_taylor_proxy_U10.png\n"
      " - Fig5_taylor_proxy.png (stitched if available)\n"
      " - proxy_extremes_by_regime.csv\n"
      "\nNOTE: These are PROXY extremes (distributional assumptions from mean/std).\n"
      "For true GPD/GEV return levels, supply declustered event-level time series.")
