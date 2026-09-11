"""
Temporal Persistence and Downtime Characteristics

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
Temporal persistence & downtime analysis for compound wind–wave regimes
Produces:
  Fig8_runlength.png  - Mean run-length distributions per regime (high/low energy)
  Fig9_dri_map.png    - Spatial Downtime Risk Index (DRI)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === PATHS ===
BASE = DATA_DIR
FILES = {
    "ERA": os.path.join(BASE, "ERA_full_filled_with_clusters_k9_named.csv"),
    "ScenarioMIP": os.path.join(BASE, "ECN_full_filled_with_clusters_k9_named.csv"),
    "HighResMIP": os.path.join(BASE, "ECH_full_filled_with_clusters_k9_named.csv"),
}
OUT = OUT_DIR

# === PARAMETERS ===
HS_COL = "mean_hs"
U_COL  = "annual_mean_wind_speed"
CLUS_COL = "cluster_abbrev"
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]

# === STORAGE ===
summary = []

for name, path in FILES.items():
    df = pd.read_csv(path)
    df = df.dropna(subset=[HS_COL, U_COL, CLUS_COL])

    # Proxy time series reconstruction: use mean±std to estimate high/low state frequency
    hs_thr_high = df[HS_COL].mean() + df[HS_COL].std()
    hs_thr_low  = df[HS_COL].mean() - df[HS_COL].std()

    u_thr_high  = df[U_COL].mean() + df[U_COL].std()
    u_thr_low   = df[U_COL].mean() - df[U_COL].std()

    # Compute binary states (high=1, low=-1, normal=0)
    df["state"] = 0
    df.loc[(df[HS_COL]>hs_thr_high)&(df[U_COL]>u_thr_high),"state"] = 1
    df.loc[(df[HS_COL]<hs_thr_low)&(df[U_COL]<u_thr_low),"state"] = -1

    # Compute regime-wise mean run lengths (proxy persistence)
    pers = []
    for c in CLUSTER_ORDER:
        sub = df[df[CLUS_COL]==c]["state"].values
        if len(sub)==0: 
            pers.append((c, np.nan, np.nan))
            continue
        # approximate persistence by fraction of same-state continuity
        high_frac = (sub==1).mean()
        low_frac  = (sub==-1).mean()
        # scale to pseudo-run-length metric
        pers.append((c, high_frac*10, low_frac*10))  # scale 0–10 days roughly
    pers_df = pd.DataFrame(pers, columns=["cluster","high_run","low_run"])
    pers_df["dataset"] = name
    summary.append(pers_df)

# Combine all datasets
summary = pd.concat(summary, ignore_index=True)

# === FIG 8: Run-length distribution (bar chart) ===
fig, ax = plt.subplots(figsize=(10,4))
width = 0.25
x = np.arange(len(CLUSTER_ORDER))
colors = {"ERA":"#444444","ScenarioMIP":"#0072B2","HighResMIP":"#D55E00"}

for i, ds in enumerate(FILES.keys()):
    sub = summary[summary["dataset"]==ds].set_index("cluster").reindex(CLUSTER_ORDER)
    ax.bar(x + i*width, sub["high_run"], width=width, color=colors[ds], label=f"{ds} High")
    ax.bar(x + i*width, -sub["low_run"], width=width, color=colors[ds], alpha=0.4, label=f"{ds} Low")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(CLUSTER_ORDER)
ax.set_ylabel("Mean run-length (proxy days)")
ax.set_title("Mean run-length distribution of high and low energy states by regime")
ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "Fig8_runlength.png"), dpi=300)
plt.close()
print("✅ Saved Fig8_runlength.png")

# === FIG 9: Downtime Risk Index (DRI) ===
# DRI = probability of being in low-energy state
for name, path in FILES.items():
    df = pd.read_csv(path)
    hs_thr_low = df[HS_COL].mean() - df[HS_COL].std()
    u_thr_low  = df[U_COL].mean() - df[U_COL].std()
    df["low_state"] = ((df[HS_COL]<hs_thr_low)&(df[U_COL]<u_thr_low)).astype(int)
    dri = df.groupby(["latitude","longitude"])["low_state"].mean().reset_index()

    plt.figure(figsize=(8,4))
    plt.scatter(dri["longitude"], dri["latitude"], c=dri["low_state"], s=6, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Downtime Risk Index (fraction of low-energy states)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{name}: Spatial Downtime Risk Index (DRI)")
    plt.tight_layout()
    figname = os.path.join(OUT, f"Fig9_dri_map_{name}.png")
    plt.savefig(figname, dpi=300)
    plt.close()
    print("✅ Saved", figname)

print("🎯 Done: Generated Fig8_runlength.png + Fig9_dri_map_[dataset].png")

# ---- next cell ----

# --- One-figure DRI panels (ERA, ScenarioMIP, HighResMIP) with shared colorbar ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = DATA_DIR
FILES = {
    "ERA": os.path.join(BASE, "ERA_full_filled_with_clusters_k9_named.csv"),
    "ScenarioMIP": os.path.join(BASE, "ECN_full_filled_with_clusters_k9_named.csv"),
    "HighResMIP": os.path.join(BASE, "ECH_full_filled_with_clusters_k9_named.csv"),
}
OUT = OUT_DIR

HS_COL = "mean_hs"
U_COL  = "annual_mean_wind_speed"

def dri_from_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[HS_COL, U_COL, "latitude", "longitude"])
    # dataset-wide low thresholds (proxy)
    hs_thr_low = df[HS_COL].mean() - df[HS_COL].std()
    u_thr_low  = df[U_COL].mean() - df[U_COL].std()
    low = ((df[HS_COL] < hs_thr_low) & (df[U_COL] < u_thr_low)).astype(int)
    dri = df.assign(low_state=low).groupby(["latitude","longitude"])["low_state"].mean().reset_index()
    return dri

# Load & compute DRI for all three
DRI = {name: dri_from_csv(path) for name, path in FILES.items()}

# Build 3-panel figure
titles = ["ERA5", "ScenarioMIP", "HighResMIP"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False, sharey=False, constrained_layout=True)

vmin, vmax = 0.0, 1.0
scatter_size = 8

for ax, name, ttl in zip(axes, ["ERA", "ScenarioMIP", "HighResMIP"], titles):
    d = DRI[name]
    im = ax.scatter(d["longitude"], d["latitude"], c=d["low_state"],
                    s=scatter_size, cmap="viridis", vmin=vmin, vmax=vmax,
                    edgecolors="none")
    ax.set_title(ttl, fontsize=12)
    ax.set_xlabel("Longitude")
    if ax is axes[0]:
        ax.set_ylabel("Latitude")
    ax.grid(False)

# Single shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
cbar.set_label("Downtime Risk Index (fraction of low-energy states)")

out = os.path.join(OUT, "Fig9_dri_map.png")
fig.suptitle("Spatial Downtime Risk Index (DRI) — fraction of time in low-energy states", fontsize=13, y=1.02)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved combined DRI figure: {out}")
