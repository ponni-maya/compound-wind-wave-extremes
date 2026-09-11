"""
Joint Design Return Levels and Model Implications

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
Proxy joint design contours & model–reference bias (from clustered CSVs)

Outputs:
  - Fig12_design_contours.png  (copula iso-probability contours, selected regimes)
  - Fig13_design_bias.png      (ΔHs_T, ΔU10_T bars by regime for T=10/25/50)
  - design_table_proxy.csv     (diagonal design points per regime/dataset)

Assumptions (proxy):
  Hs ~ Lognormal(mean, std),  U10 ~ Weibull(k, λ) via method-of-moments.
  Dependence via Gaussian copula with rho estimated from Kendall's tau on
  spatial samples within each regime. Return contours solve 1 - C(u, v; rho) = 1/T.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, lognorm, weibull_min, norm, multivariate_normal
from scipy.optimize import bisect

# -----------------------
# Paths & input files
# -----------------------
BASE = DATA_DIR
FILES = {
    "ERA":         os.path.join(BASE, "ERA_full_filled_with_clusters_k9_named.csv"),
    "ScenarioMIP": os.path.join(BASE, "ECN_full_filled_with_clusters_k9_named.csv"),
    "HighResMIP":  os.path.join(BASE, "ECH_full_filled_with_clusters_k9_named.csv"),
}
OUT = OUT_DIR

# -----------------------
# Columns & regimes
# -----------------------
HS_COL   = "mean_hs"
U10_COL  = "annual_mean_wind_speed"
CLUS_COL = "cluster_abbrev"

# Order for plotting
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]

# Regimes to show in contour figure (edit as you like)
REGIMES_FOR_CONTOURS = ["EAC", "EMC"]  # e.g., open-ocean & semi-enclosed

# Return periods (years)
RETURNS = [10, 25, 50]

# -----------------------
# Helper functions
# -----------------------
def fit_lognormal_from_mean_std(m, s):
    """Return scipy lognorm parameters (shape=sigma, loc=0, scale=exp(mu)) from mean & std."""
    if m <= 0 or s <= 0:
        # fallback tiny variance
        sigma = 0.10
        mu = np.log(max(m, 1e-6)) - 0.5*sigma**2
        return lognorm(sigma, scale=np.exp(mu))
    # lognormal equations
    sigma2 = np.log(1.0 + (s**2)/(m**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(m) - 0.5*sigma2
    return lognorm(sigma, scale=np.exp(mu))

def fit_weibull_moments(mean, std):
    """Fit Weibull(k, λ) from mean & std via simple search on k; return a frozen scipy distribution."""
    if mean <= 0 or std <= 0:
        # fallback near-deterministic
        k = 5.0
        lam = max(mean, 1e-6)
        return weibull_min(c=k, scale=lam)
    # method-of-moments: solve for k numerically using coefficients of variation
    from scipy.special import gamma
    cv = std / mean
    # search k in a reasonable range
    ks = np.linspace(0.5, 10.0, 300)
    cvs = []
    for k in ks:
        m1 = gamma(1 + 1.0/k)
        m2 = gamma(1 + 2.0/k)
        cv_k = np.sqrt(m2 - m1**2) / m1
        cvs.append(cv_k)
    cvs = np.array(cvs)
    i = np.argmin(np.abs(cvs - cv))
    k = float(ks[i])
    m1 = gamma(1 + 1.0/k)
    lam = mean / m1
    return weibull_min(c=k, scale=lam)

def gaussian_copula_cdf(u, v, rho):
    """C(u,v; rho) using Gaussian copula with correlation rho."""
    x = norm.ppf(np.clip(u, 1e-12, 1-1e-12))
    y = norm.ppf(np.clip(v, 1e-12, 1-1e-12))
    cov = [[1.0, rho],[rho, 1.0]]
    return multivariate_normal(mean=[0,0], cov=cov).cdf([x, y])

def tau_to_rho(tau):
    """Map Kendall's tau to Gaussian-copula rho."""
    return np.sin(0.5*np.pi*np.clip(tau, -0.999, 0.999))

def diag_u_for_return_T(rho, T):
    """Solve for u on the diagonal (u=v) such that 1 - C(u,u; rho) = 1/T."""
    target = 1.0 - 1.0/T
    f = lambda u: gaussian_copula_cdf(u, u, rho) - target
    # search in [0.5..0.999999]
    return bisect(f, 0.50, 0.999999, maxiter=200, xtol=1e-9)

def contour_uv_for_T(rho, T, n=140, u_min=0.70):
    """Generate (u,v) contour points for 1 - C(u,v) = 1/T by sweeping u."""
    target = 1.0 - 1.0/T
    us = np.linspace(u_min, 0.9999, n)
    vs = np.empty_like(us)
    for i, u in enumerate(us):
        # solve in v for C(u,v) = target
        g = lambda v: gaussian_copula_cdf(u, v, rho) - target
        try:
            v = bisect(g, 1e-6, 0.999999, maxiter=200, xtol=1e-9)
        except ValueError:
            v = np.nan
        vs[i] = v
    return us, vs

def safe_kendall_tau(x, y):
    """Kendall tau with guard for constant arrays / small samples."""
    if len(x) < 10 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return 0.3  # mild positive proxy
    t, _ = kendalltau(x, y, nan_policy="omit")
    if not np.isfinite(t):
        t = 0.3
    return float(t)

# -----------------------
# Load data
# -----------------------
print(">> Loading clustered CSVs …")
data = {}
for ds, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # minimal clean
    df = df.dropna(subset=[HS_COL, U10_COL, CLUS_COL])
    # keep in domain
    data[ds] = df
    print(f"   {ds:<11} {os.path.basename(path)}  (rows={len(df):,})  regimes={df[CLUS_COL].nunique()}")

# -----------------------
# Fit proxy marginals & dependence per regime
# -----------------------
records = []  # to hold diagonal design points & parameters

for ds, df in data.items():
    for reg in CLUSTER_ORDER:
        sub = df[df[CLUS_COL] == reg]
        if sub.empty:
            continue

        # Means & stds across gridpoints (spatial proxy)
        m_hs, s_hs = float(sub[HS_COL].mean()), float(sub[HS_COL].std(ddof=0))
        m_u,  s_u  = float(sub[U10_COL].mean()), float(sub[U10_COL].std(ddof=0))

        # Fit marginals
        dist_hs = fit_lognormal_from_mean_std(m_hs, s_hs)
        dist_u  = fit_weibull_moments(m_u, s_u)

        # Proxy Kendall's tau across spatial samples, then Gaussian rho
        tau = safe_kendall_tau(sub[U10_COL].values, sub[HS_COL].values)
        rho = tau_to_rho(tau)

        # Diagonal design points for each return period
        for T in RETURNS:
            u_diag = diag_u_for_return_T(rho, T)
            Hs_T   = dist_hs.ppf(u_diag)
            U_T    = dist_u.ppf(u_diag)
            records.append({
                "dataset": ds, "regime": reg, "T": T,
                "tau_proxy": tau, "rho_proxy": rho,
                "u_diag": u_diag, "Hs_T": Hs_T, "U10_T": U_T,
                "Hs_mean": m_hs, "U10_mean": m_u
            })

design_df = pd.DataFrame.from_records(records)
design_csv = os.path.join(OUT, "design_table_proxy.csv")
design_df.to_csv(design_csv, index=False)
print(f">> Wrote table: {design_csv}")

# -----------------------
# Fig. 12 — Copula design contours (selected regimes)
# -----------------------
print(">> Rendering Fig12_design_contours.png …")
nreg = len(REGIMES_FOR_CONTOURS)
fig, axes = plt.subplots(nreg, 3, figsize=(14, 4.8*nreg), sharex=False, sharey=False, constrained_layout=True)
if nreg == 1:
    axes = np.array([axes])  # ensure 2D

colors = {10: "#2c7fb8", 25: "#fdae61", 50: "#d7191c"}  # default colors by T
for r_idx, reg in enumerate(REGIMES_FOR_CONTOURS):
    for c_idx, ds in enumerate(["ERA", "ScenarioMIP", "HighResMIP"]):
        ax = axes[r_idx, c_idx]
        base = data[ds]
        sub  = base[base[CLUS_COL] == reg]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Refit for this dataset/regime
        m_hs, s_hs = float(sub[HS_COL].mean()), float(sub[HS_COL].std(ddof=0))
        m_u,  s_u  = float(sub[U10_COL].mean()), float(sub[U10_COL].std(ddof=0))
        dist_hs = fit_lognormal_from_mean_std(m_hs, s_hs)
        dist_u  = fit_weibull_moments(m_u, s_u)
        tau = safe_kendall_tau(sub[U10_COL].values, sub[HS_COL].values)
        rho = tau_to_rho(tau)

        # Contours
        for T in RETURNS:
            us, vs = contour_uv_for_T(rho, T, n=160, u_min=0.70)
            # map to physical space
            hs = dist_hs.ppf(us)
            u10 = dist_u.ppf(vs)
            ax.plot(hs, u10, label=f"T={T} yr", lw=2.0, color=colors[T])

        ax.set_title(f"{ds} — {reg}  (τ≈{tau:.2f})", fontsize=12)
        ax.set_xlabel("$H_s$ (m)")
        ax.set_ylabel("$U_{10}$ (m s$^{-1}$)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=10, loc="lower right")

fig.suptitle("Proxy joint design contours (Gaussian copula; Hs~Lognormal, U10~Weibull)", fontsize=14)
f12 = os.path.join(OUT, "Fig12_design_contours.png")
fig.savefig(f12, dpi=300)
plt.close(fig)
print(f"   saved -> {f12}")

# -----------------------
# Fig. 13 — Model–reference bias bars
# -----------------------
print(">> Rendering Fig13_design_bias.png …")
# Build Δ relative to ERA by regime and T
era = design_df[design_df["dataset"] == "ERA"]
rows = []
for ds in ["ScenarioMIP", "HighResMIP"]:
    sub = design_df[design_df["dataset"] == ds]
    merged = pd.merge(
        sub, era,
        on=["regime", "T"],
        suffixes=(f"_{ds}", "_ERA")
    )
    for _, r in merged.iterrows():
        rows.append({
            "dataset": ds,
            "regime": r["regime"],
            "T": int(r["T"]),
            "dHs_T": r["Hs_T_"+ds]   - r["Hs_T_ERA"],
            "dU10_T": r["U10_T_"+ds] - r["U10_T_ERA"]
        })

bias = pd.DataFrame(rows)

# Plot 2 panels: ΔHs_T and ΔU10_T (stack regimes on x; colors by T)
fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)

for i, var in enumerate(["dHs_T", "dU10_T"]):
    ax = axs[i]
    x = np.arange(len(CLUSTER_ORDER))
    width = 0.28
    # plot ScenarioMIP and HighResMIP side-by-side per T
    offsets = {"ScenarioMIP": -width/2, "HighResMIP": width/2}
    for ds in ["ScenarioMIP", "HighResMIP"]:
        for j, T in enumerate(RETURNS):
            bsub = bias[(bias["dataset"]==ds) & (bias["T"]==T)].set_index("regime").reindex(CLUSTER_ORDER)
            vals = bsub[var].values
            ax.bar(x + offsets[ds] + (j-1)*width/3,
                   vals, width=width/3,
                   label=f"{ds} T={T} yr" if i==0 else None,
                   edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"$\Delta H_{s,T}$ (m)" if var=="dHs_T" else r"$\Delta U_{10,T}$ (m s$^{-1}$)")
    ax.grid(axis="y", alpha=0.25)

axs[-1].set_xticks(np.arange(len(CLUSTER_ORDER)))
axs[-1].set_xticklabels(CLUSTER_ORDER)
axs[0].set_title("Model–reference deviation in diagonal design points (proxy)")

# One legend at top
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0,1.12))

f13 = os.path.join(OUT, "Fig13_design_bias.png")
fig.savefig(f13, dpi=300)
plt.close(fig)
print(f"   saved -> {f13}")

print("✅ Done.\n - Fig12_design_contours.png\n - Fig13_design_bias.png\n - design_table_proxy.csv\n\nNOTE: Proxy results for comparative use only (see header).")
