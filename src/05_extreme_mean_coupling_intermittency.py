"""
Extreme–Mean Coupling and Energy Intermittency

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
Extreme–Mean Coupling and Intermittency (EMCR) from clustered CSVs (proxy)
Outputs:
  - emcr_by_regime.csv
  - Fig10_emcr_boxplot.png
  - Fig11_emcr_scatter.png

EMCR definition (proxy):
  EMCR = p95( X ) / mean( X )  computed spatially within each regime
where X is one of:
  - U10  -> annual_mean_wind_speed
  - Hs   -> mean_hs
  - Pw   -> const * (mean_hs**2) * mean_t02   (deep-water proxy using T02≈Te)

Author: you
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# -------------------
# PATHS
# -------------------
BASE = DATA_DIR
FILES = {
    "ERA":         os.path.join(BASE, "ERA_full_filled_with_clusters_k9_named.csv"),
    "ScenarioMIP": os.path.join(BASE, "ECN_full_filled_with_clusters_k9_named.csv"),
    "HighResMIP":  os.path.join(BASE, "ECH_full_filled_with_clusters_k9_named.csv"),
}
OUT = OUT_DIR

# -------------------
# SETTINGS
# -------------------
CLUS_COL = "cluster_abbrev"
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]

# required columns in the clustered CSVs
COL_U10 = "annual_mean_wind_speed"  # m s^-1 (climatological mean per grid cell)
COL_HS  = "mean_hs"                 # m (climatological mean per grid cell)
COL_T02 = "mean_t02"                # s (proxy for Te)

# constants for wave power (unit scale not important for EMCR ratio)
RHO_W = 1025.0
G = 9.81
PW_CONST = (RHO_W * G**2) / (64.0 * np.pi)

# plotting styles
COLORS = {"ERA":"#444444", "ScenarioMIP":"#1f77b4", "HighResMIP":"#d62728"}

# -------------------
# helper
# -------------------
def pct95_over_mean(arr):
    """EMCR = p95/mean on finite values"""
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0 or np.nanmean(a) == 0:
        return np.nan
    return np.nanpercentile(a, 95) / np.nanmean(a)

def safe_mean(arr):
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    return np.nan if a.size == 0 else float(np.nanmean(a))

# -------------------
# compute EMCR per regime
# -------------------
records = []

print(">> Computing EMCR (spatial proxy) by regime …")
for ds, path in FILES.items():
    if not os.path.exists(path):
        print(f"!! Missing file for {ds}: {path}")
        continue

    df = pd.read_csv(path)
    # basic clean
    need = [CLUS_COL, COL_U10, COL_HS, COL_T02, "latitude", "longitude"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{ds} missing required columns: {miss}")

    # proxy wave power at each grid cell (units arbitrary for EMCR since ratio)
    df["Pw_proxy"] = PW_CONST * (df[COL_HS]**2) * df[COL_T02]

    # compute regimewise EMCR and means
    for reg, sub in df.groupby(CLUS_COL):
        emcr_u10 = pct95_over_mean(sub[COL_U10])
        emcr_hs  = pct95_over_mean(sub[COL_HS])
        emcr_pw  = pct95_over_mean(sub["Pw_proxy"])

        mean_u10 = safe_mean(sub[COL_U10])
        mean_hs  = safe_mean(sub[COL_HS])
        mean_pw  = safe_mean(sub["Pw_proxy"])

        records.append({
            "dataset": ds,
            "regime": reg,
            "EMCR_U10": emcr_u10,
            "EMCR_Hs":  emcr_hs,
            "EMCR_Pw":  emcr_pw,
            "Mean_U10": mean_u10,
            "Mean_Hs":  mean_hs,
            "Mean_Pw":  mean_pw,
            "n_cells":  len(sub)
        })

emcr = pd.DataFrame.from_records(records)
# enforce regime order for plotting consistency
emcr["regime"] = pd.Categorical(emcr["regime"], categories=CLUSTER_ORDER, ordered=True)
emcr = emcr.sort_values(["regime","dataset"]).reset_index(drop=True)

# write table
csv_path = os.path.join(OUT, "emcr_by_regime.csv")
emcr.to_csv(csv_path, index=False)
print(f">> Wrote table: {csv_path}")

# -------------------
# FIG 10: Boxplots of EMCR per dataset (3 panels: Hs, U10, Pw)
# -------------------
print(">> Rendering Fig10_emcr_boxplot.png …")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True, sharey=False)

panels = [
    ("EMCR_Hs",  r"$H_s$"),
    ("EMCR_U10", r"$U_{10}$"),
    ("EMCR_Pw",  r"$P_w$ (proxy)"),
]

for ax, (col, label) in zip(axes, panels):
    box_data = [emcr.loc[emcr["dataset"]==ds, col].dropna().values for ds in FILES.keys()]
    bp = ax.boxplot(box_data, patch_artist=True, labels=list(FILES.keys()))
    for patch, ds in zip(bp["boxes"], FILES.keys()):
        patch.set_facecolor(COLORS[ds])
        patch.set_alpha(0.7)
    ax.set_title(f"EMCR for {label}")
    ax.set_ylabel("EMCR = p95 / mean")
    ax.grid(True, alpha=0.25)

fig.suptitle("Regime-wise EMCR distributions across models (spatial proxy)", fontsize=13)
fig.savefig(os.path.join(OUT, "Fig10_emcr_boxplot.png"), dpi=300)
plt.close(fig)
print("   saved -> Fig10_emcr_boxplot.png")

# -------------------
# FIG 11: Mean vs EMCR scatter (points = regimes), colored by dataset
# -------------------
print(">> Rendering Fig11_emcr_scatter.png …")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

scat_specs = [
    ("Mean_Hs",  "EMCR_Hs",  r"Mean $H_s$ (m)",         r"EMCR($H_s$)"),
    ("Mean_U10", "EMCR_U10", r"Mean $U_{10}$ (m s$^{-1}$)", r"EMCR($U_{10}$)"),
    ("Mean_Pw",  "EMCR_Pw",  r"Mean $P_w$ (proxy)",     r"EMCR($P_w$)"),
]

for ax, (xcol, ycol, xlab, ylab) in zip(axes, scat_specs):
    for ds in FILES.keys():
        sub = emcr[emcr["dataset"]==ds]
        ax.scatter(sub[xcol], sub[ycol], s=60, color=COLORS[ds], label=ds, alpha=0.8, edgecolor="white", linewidth=0.6)
        # add regime labels
        for _, r in sub.iterrows():
            ax.text(r[xcol], r[ycol], str(r["regime"]), fontsize=8, ha="center", va="bottom", color="black", alpha=0.9)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.25)

# one legend for all
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[:3], labels[:3], ncol=3, loc="upper center", frameon=False)
fig.suptitle("Mean vs. EMCR (spatial proxy) by regime and dataset", fontsize=13, y=1.05)
fig.savefig(os.path.join(OUT, "Fig11_emcr_scatter.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("   saved -> Fig11_emcr_scatter.png")

print("✅ Done.")
print(" - emcr_by_regime.csv")
print(" - Fig10_emcr_boxplot.png")
print(" - Fig11_emcr_scatter.png")
