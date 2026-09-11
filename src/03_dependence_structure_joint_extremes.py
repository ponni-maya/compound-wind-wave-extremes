"""
Dependence Structure of Joint Extremes

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
Dependence Structure of Joint Extremes (Proxy, Regime-wise)
- Builds proxy joint samples from seasonal/annual (Hs, U10) pairs.
- Computes Kendall's tau and empirical upper-tail dependence proxy.
- Fits Gaussian copula via normal-scores correlation for contours.
- Produces Fig6_tau_map.png and Fig7_copula_contours.png.
- Exports dependence_metrics.csv (per regime & dataset).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from scipy.stats import kendalltau, norm

# ----------------------- USER INPUTS -----------------------
FOLDER = DATA_DIR
FILES = {
    "ERA5":        "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP":  "ECH_full_filled_with_clusters_k9_named.csv",
}
REGIME_COL = "cluster_abbrev"  # your regime label
# seasonal/annual variable names in your headers:
HS_SEAS = ["mean_hs_DJF", "mean_hs_MAM", "mean_hs_JJA", "mean_hs_SON"]
U10_SEAS = ["mean_wind_speed_DJF", "mean_wind_speed_MAM", "mean_wind_speed_JJA", "mean_wind_speed_SON"]
HS_ANN, U10_ANN = "mean_hs", "annual_mean_wind_speed"

# Representative regimes for copula contours (open-ocean & semi-enclosed)
REP_REGIMES = ["EAC", "EMC"]  # adjust if needed (e.g., ["EAC","BSC"])

OUT = FOLDER  # save alongside data
FIG6 = os.path.join(OUT, "Fig6_tau_map.png")
FIG7 = os.path.join(OUT, "Fig7_copula_contours.png")
CSV_OUT = os.path.join(OUT, "dependence_metrics.csv")

# Optional fixed order for plotting
REGIME_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]

# --------------------- HELPER FUNCTIONS --------------------
def load_all(files_dict):
    dfs = []
    for ds, fname in files_dict.items():
        path = os.path.join(FOLDER, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        df["dataset"] = ds
        # Basic sanity
        needed = [REGIME_COL, "longitude", "latitude", HS_ANN, U10_ANN] + HS_SEAS + U10_SEAS
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"{ds}: required columns missing: {missing}")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full

def pseudo_obs(x):
    """Rank-based pseudo-observations U = rank(x)/(n+1)."""
    x = np.asarray(x)
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return ranks / (len(x) + 1.0)

def tail_dependence_empirical(U, V, q=0.95):
    """Empirical upper-tail dependence proxy: λ_U(q) = P(U>q, V>q)/(1-q)."""
    U = np.asarray(U); V = np.asarray(V)
    mask = np.isfinite(U) & np.isfinite(V)
    U = U[mask]; V = V[mask]
    if U.size < 50:
        return np.nan
    joint = np.mean((U > q) & (V > q))
    denom = (1.0 - q)
    return np.nan if denom <= 0 else joint / denom

def gaussian_copula_rho(U, V):
    """Estimate Gaussian copula correlation via normal score transform."""
    eps = 1e-10
    U = np.clip(U, eps, 1 - eps)
    V = np.clip(V, eps, 1 - eps)
    Z1 = norm.ppf(U)
    Z2 = norm.ppf(V)
    return np.corrcoef(Z1, Z2)[0, 1]

def collect_regime_pairs(df_reg):
    """
    Build proxy joint sample for one regime: concatenate seasonal (Hs, U10) pairs
    across all rows/years/grids, plus annual (Hs, U10).
    Returns arrays Hs_all, U10_all (1D).
    """
    parts_hs, parts_u10 = [], []
    # seasonal
    for hs_col, u_col in zip(HS_SEAS, U10_SEAS):
        hs = df_reg[hs_col].to_numpy()
        u = df_reg[u_col].to_numpy()
        m = np.isfinite(hs) & np.isfinite(u)
        parts_hs.append(hs[m]); parts_u10.append(u[m])
    # annual
    hs = df_reg[HS_ANN].to_numpy()
    u = df_reg[U10_ANN].to_numpy()
    m = np.isfinite(hs) & np.isfinite(u)
    parts_hs.append(hs[m]); parts_u10.append(u[m])

    Hs_all = np.concatenate(parts_hs) if parts_hs else np.array([])
    U_all  = np.concatenate(parts_u10) if parts_u10 else np.array([])
    return Hs_all, U_all

def compute_dependence_metrics(df):
    """
    For each dataset & regime:
      - Kendall's tau between Hs and U10 (proxy joint sample)
      - Empirical tail dependence at q=0.95 from pseudo observations
      - Gaussian copula rho (normal-scores correlation)
      - n (sample size used)
    """
    out = []
    for ds, dsd in df.groupby("dataset"):
        for reg, dreg in dsd.groupby(REGIME_COL):
            Hs, U = collect_regime_pairs(dreg)
            Hs = Hs.astype(float); U = U.astype(float)
            m = np.isfinite(Hs) & np.isfinite(U)
            Hs, U = Hs[m], U[m]
            if Hs.size < 50:
                out.append(dict(dataset=ds, regime=reg, n=Hs.size,
                                tau=np.nan, lambdaU=np.nan, rho=np.nan))
                continue
            # Kendall tau
            tau, _ = kendalltau(Hs, U, nan_policy="omit")
            # Pseudo-obs and tail dependence
            Uh = pseudo_obs(Hs)
            Uu = pseudo_obs(U)
            lamU = tail_dependence_empirical(Uh, Uu, q=0.95)
            # Gaussian copula rho
            rho = gaussian_copula_rho(Uh, Uu)
            out.append(dict(dataset=ds, regime=reg, n=Hs.size,
                            tau=tau, lambdaU=lamU, rho=rho))
    dep = pd.DataFrame(out)
    if REGIME_ORDER:
        dep["regime"] = pd.Categorical(dep["regime"], categories=REGIME_ORDER, ordered=True)
        dep = dep.sort_values(["dataset","regime"])
    return dep

# --------------------- PLOTTING FUNCTIONS -------------------
COLORS = {
    "ERA5": "#1f77b4",        # blue
    "ScenarioMIP": "#ff7f0e", # orange
    "HighResMIP": "#2ca02c",  # green
}

def make_tau_map(df, dep, outpath):
    """
    Map-like scatter: each grid cell colored by regime's tau (per dataset).
    One row per dataset. Uses regime-level tau from dep table.
    """
    # Build (dataset, regime) -> tau lookup
    tauLUT = dep.set_index(["dataset","regime"])["tau"].to_dict()

    # prepare figure
    ds_list = list(FILES.keys())  # ["ERA5", "ScenarioMIP", "HighResMIP"]
    fig, axes = plt.subplots(nrows=1, ncols=len(ds_list), figsize=(14, 4.4), constrained_layout=True)
    if len(ds_list) == 1:
        axes = [axes]

    # Fix domain
    xmin = df["longitude"].min(); xmax = df["longitude"].max()
    ymin = df["latitude"].min();  ymax = df["latitude"].max()

    # Common normalization for tau color
    # tau theoretically in [-1, 1]; here clip to [-0.2, 1.0] for color scale
    norm = Normalize(vmin=-0.2, vmax=1.0)
    cmap = plt.cm.viridis

    for ax, ds in zip(axes, ds_list):
        sub = df[df["dataset"] == ds].copy()
        # map each row to its regime tau
        key = list(zip([ds]*len(sub), sub[REGIME_COL].astype(str)))
        tvals = np.array([tauLUT.get(k, np.nan) for k in key])
        m = np.isfinite(tvals)
        ax.scatter(sub.loc[m, "longitude"], sub.loc[m, "latitude"],
                   c=tvals[m], s=6, cmap=cmap, norm=norm, edgecolor="none")
        ax.set_title(ds, fontsize=11)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Longitude"); 
        if ax is axes[0]:
            ax.set_ylabel("Latitude")
        ax.grid(True, lw=0.2, alpha=0.4)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("Kendall's $\\tau$ (proxy)")

    fig.suptitle("Kendall's $\\tau$ for proxy joint $(H_s, U_{10})$ within fixed regimes", y=1.02, fontsize=12)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def gaussian_copula_pdf(u, v, rho):
    """Gaussian copula density c(u,v;rho) on (0,1)^2."""
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    x = norm.ppf(u); y = norm.ppf(v)
    denom = np.sqrt(1 - rho**2)
    exponent = - (x**2 - 2*rho*x*y + y**2) / (2*(1 - rho**2)) + (x**2 + y**2)/2.0
    # Copula density: c(u,v) = phi_2(x,y;rho) / [phi(x)*phi(y)]
    # Simplifies to exp(exponent) / denom
    return np.exp(exponent) / denom

def plot_copula_contours(df, dep, regimes, outpath):
    """
    For a small set of regimes (e.g., ["EAC","EMC"]), create a 2x3 panel:
       rows = regimes, cols = datasets (ERA5, ScenarioMIP, HighResMIP)
    For each axis: scatter (U,V) pseudo-obs + Gaussian copula contour with fitted rho.
    """
    ds_list = list(FILES.keys())  # ["ERA5","ScenarioMIP","HighResMIP"]
    R = len(regimes); C = len(ds_list)
    fig, axes = plt.subplots(R, C, figsize=(4.5*C, 3.8*R), constrained_layout=True)
    if R == 1: axes = np.array([axes])
    if C == 1: axes = axes.reshape(R, 1)

    levels = np.linspace(1.0, 6.0, 10)  # arbitrary density levels

    for i, reg in enumerate(regimes):
        for j, ds in enumerate(ds_list):
            ax = axes[i, j]
            dsub = df[(df["dataset"] == ds) & (df[REGIME_COL] == reg)]

            # build proxy joint pairs and pseudo-obs
            Hs, U = collect_regime_pairs(dsub)
            m = np.isfinite(Hs) & np.isfinite(U)
            Hs, U = Hs[m], U[m]
            if Hs.size < 50:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_title(f"{ds} — {reg}")
                continue

            Uh = pseudo_obs(Hs); Uu = pseudo_obs(U)
            rho = gaussian_copula_rho(Uh, Uu)

            # scatter of pseudo-obs
            ax.scatter(Uu, Uh, s=6, alpha=0.35, edgecolor="none", color=COLORS.get(ds, "gray"))

            # copula density contours
            grid = np.linspace(0.01, 0.99, 120)
            U_grid, V_grid = np.meshgrid(grid, grid)
            Z = gaussian_copula_pdf(U_grid, V_grid, rho=rho)
            cs = ax.contour(U_grid, V_grid, Z, levels=levels, linewidths=0.7, colors="k")
            ax.clabel(cs, inline=True, fmt="%.1f", fontsize=7)

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == R-1: ax.set_xlabel("$F_{U_{10}}$")
            if j == 0:   ax.set_ylabel("$F_{H_s}$")
            ax.set_title(f"{ds} — {reg}  (ρ={rho:.2f})", fontsize=10)
            ax.grid(True, lw=0.2, alpha=0.4)

    fig.suptitle("Gaussian-copula contours over pseudo-observations by regime (proxy dependence)", y=1.02, fontsize=12)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# ------------------------- RUN ------------------------------
if __name__ == "__main__":
    print(">> Loading clustered CSVs …")
    df = load_all(FILES)
    print(f"   Loaded rows: {len(df):,}")

    # Compute dependence metrics
    print(">> Computing regime-wise dependence metrics (tau, lambda_U, rho) …")
    dep = compute_dependence_metrics(df)
    dep.to_csv(CSV_OUT, index=False)
    print(f">> Wrote metrics: {CSV_OUT}")

    # Fig 6: tau "map" panels (one per dataset)
    print(">> Rendering Fig6_tau_map.png …")
    make_tau_map(df, dep, FIG6)
    print(f"   saved -> {FIG6}")

    # Fig 7: copula contours for representative regimes
    # If a requested regime not present (e.g., BSC missing in a dataset), it will just show “Insufficient data”.
    print(">> Rendering Fig7_copula_contours.png …")
    plot_copula_contours(df, dep, REP_REGIMES, FIG7)
    print(f"   saved -> {FIG7}")

    # Console summary (optional): show tau by regime/dataset
    print("\nKendall's tau (proxy) by regime:")
    print(dep.pivot_table(index="regime", columns="dataset", values="tau", aggfunc="first").round(2))
    print("\nUpper-tail proxy λ_U (q=0.95):")
    print(dep.pivot_table(index="regime", columns="dataset", values="lambdaU", aggfunc="first").round(2))

