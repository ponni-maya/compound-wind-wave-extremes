"""
Frequency and Intensity of Compound Extremes

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
Quick inspection of ERA, ScenarioMIP, and HighResMIP clustered CSVs.
Checks size, columns, NaNs, cluster distribution, and summary stats.
"""

import os
import pandas as pd

# === File paths ===
folder = DATA_DIR
files = {
    "ERA":         "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP":  "ECH_full_filled_with_clusters_k9_named.csv",
}

# === Function to explore each dataset ===
def inspect_csv(name, path):
    print("="*80)
    print(f"{name} → {os.path.basename(path)}")
    print("="*80)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading file: {e}\n")
        return

    # --- Basic info ---
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print("Columns:", list(df.columns[:10]), "...\n")

    # --- Missing values ---
    na_counts = df.isna().sum()
    na_total = na_counts.sum()
    print(f"Missing values: {na_total} ({na_total/df.size*100:.2f}% of total cells)\n")
    if na_total > 0:
        print(na_counts[na_counts > 0].sort_values(ascending=False).head())

    # --- Cluster info ---
    cluster_cols = [c for c in df.columns if "cluster" in c.lower() or "regime" in c.lower()]
    if cluster_cols:
        col = cluster_cols[0]
        print(f"\nCluster column detected: '{col}'")
        print(df[col].value_counts())
    else:
        print("\n⚠️ No cluster column found.")

    # --- Summary of key numerical variables ---
    key_vars = [v for v in ["Hs", "U10", "Tp", "T02", "Pw"] if v in df.columns]
    if key_vars:
        print("\nSummary of key variables:")
        print(df[key_vars].describe().T[["mean", "std", "min", "max"]])
    else:
        print("\nNo standard physical variables (Hs, U10, etc.) found.")

    print("\n")

# === Run inspection for all ===
for name, fname in files.items():
    path = os.path.join(folder, fname)
    inspect_csv(name, path)

# ---- next cell ----

# -*- coding: utf-8 -*-
"""
Display column headers (names) from ERA, ScenarioMIP, and HighResMIP CSVs.
"""

import os
import pandas as pd

# === File paths ===
folder = DATA_DIR
files = {
    "ERA":         "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP":  "ECH_full_filled_with_clusters_k9_named.csv",
}

# === Function to show headers ===
def show_headers(name, path):
    print("="*80)
    print(f"{name} → {os.path.basename(path)}")
    print("="*80)

    try:
        # Only read the first row for speed
        df = pd.read_csv(path, nrows=0)
        headers = list(df.columns)
        for i, col in enumerate(headers, 1):
            print(f"{i:02d}. {col}")
        print(f"\nTotal columns: {len(headers)}\n")

    except Exception as e:
        print(f"❌ Error readin file: {e}\n")

# === Loop through all datasets ===
for name, fname in files.items():
    path = os.path.join(folder, fname)
    show_headers(name, path)

# ---- next cell ----

import pandas as pd, os

folder = DATA_DIR
files = {
    "ERA": "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP": "ECH_full_filled_with_clusters_k9_named.csv",
}

for name, fname in files.items():
    df = pd.read_csv(os.path.join(folder, fname), usecols=[
        "mean_hs", "std_hs", "mean_t02", "mean_wind_speed_DJF",
        "annual_mean_wind_speed"
    ])
    print(f"\n{name} — summary statistics:")
    print(df.describe(percentiles=[0.01, 0.5, 0.99]).T)

# ---- next cell ----

# -*- coding: utf-8 -*-
"""
Compound wind–wave extremes (Section 3.2): frequency and intensity
-------------------------------------------------------------------
This script:
  • Loads ERA, ScenarioMIP, HighResMIP CSVs with regime labels
  • Computes joint exceedance probability P_joint per regime:
        P_joint = P(Hs > p95  AND  U10 > p95)
  • Supports two threshold modes:
        - 'intrinsic': p95 computed per dataset (default)
        - 'era5_anchored': ERA5 p95 applied to all datasets
  • Exports a CSV summary and produces two publication-ready figures:
        - Fig2_joint_prob.png (P_joint by regime, grouped bars)
        - Fig3_coextreme_counts.png (counts by regime, grouped bars)

Author: (you)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent

# ----------------------------
# User configuration
# ----------------------------
FOLDER = DATA_DIR
FILES = {
    "ERA":         "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP":  "ECH_full_filled_with_clusters_k9_named.csv",
}

# threshold_mode: 'intrinsic' (per dataset p95) or 'era5_anchored' (use ERA5 p95 for all)
THRESHOLD_MODE = "intrinsic"   # change to "era5_anchored" for common benchmark

# Output directory (created if missing)
OUTDIR = os.path.join(FOLDER, "compound_extremes_outputs")
os.makedirs(OUTDIR, exist_ok=True)

# Column names to use
COL_HS   = "mean_hs"
COL_U10  = "annual_mean_wind_speed"
COL_CLUS = "cluster_abbrev"

# Nice cluster display order (edit to your preferred regime order)
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]


# ----------------------------
# Helpers
# ----------------------------
def load_data(folder, files):
    dfs = {}
    for name, fname in files.items():
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        for col in [COL_HS, COL_U10, COL_CLUS]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {name} ({fname})")
        dfs[name] = df
    return dfs

def compute_thresholds(dfs, mode="intrinsic"):
    """
    Returns dict of thresholds per dataset:
        thresholds[name] = { 'hs': value, 'u10': value }
    """
    thresholds = {}
    if mode not in ("intrinsic", "era5_anchored"):
        raise ValueError("mode must be 'intrinsic' or 'era5_anchored'")

    if mode == "era5_anchored":
        era = dfs["ERA"]
        hs95 = era[COL_HS].quantile(0.95)
        u95  = era[COL_U10].quantile(0.95)
        for name in dfs:
            thresholds[name] = {"hs": hs95, "u10": u95}
    else:
        for name, df in dfs.items():
            hs95 = df[COL_HS].quantile(0.95)
            u95  = df[COL_U10].quantile(0.95)
            thresholds[name] = {"hs": hs95, "u10": u95}

    return thresholds

def joint_exceedance_prob(g, hs_th, u_th):
    mask = (g[COL_HS] > hs_th) & (g[COL_U10] > u_th)
    return mask.mean(), mask.sum(), len(g)

def summarize_by_regime(dfs, thresholds, cluster_order=None):
    """
    Returns a tidy DataFrame with P_joint and counts per dataset × cluster.
    """
    records = []
    for name, df in dfs.items():
        hs_th = thresholds[name]["hs"]
        u_th  = thresholds[name]["u10"]
        # group by cluster
        for clus, g in df.groupby(COL_CLUS):
            p, n, N = joint_exceedance_prob(g, hs_th, u_th)
            records.append({
                "dataset": name,
                "cluster": clus,
                "P_joint": p,
                "coextreme_count": int(n),
                "total_count": int(N),
                "hs_p95": hs_th,
                "u10_p95": u_th,
            })
        # overall (all clusters)
        p_all, n_all, N_all = joint_exceedance_prob(df, hs_th, u_th)
        records.append({
            "dataset": name,
            "cluster": "ALL",
            "P_joint": p_all,
            "coextreme_count": int(n_all),
            "total_count": int(N_all),
            "hs_p95": hs_th,
            "u10_p95": u_th,
        })

    out = pd.DataFrame.from_records(records)
    if cluster_order:
        # include ALL at the end
        all_order = [c for c in cluster_order if c in out["cluster"].unique()] + ["ALL"]
        out["cluster"] = pd.Categorical(out["cluster"], categories=all_order, ordered=True)
        out = out.sort_values(["cluster","dataset"])
    return out

def _bar_grouped(ax, df_plot, value_col, ylabel, title, colors=None):
    """
    Grouped bars by cluster (x) and dataset (hue).
    """
    clusters = df_plot["cluster"].unique().tolist()
    datasets = df_plot["dataset"].unique().tolist()
    x = np.arange(len(clusters))
    width = 0.75 / max(len(datasets), 3)  # leave margins

    if colors is None:
        colors = {"ERA": "#595959", "ScenarioMIP": "#1f77b4", "HighResMIP": "#d62728"}

    for i, ds in enumerate(datasets):
        vals = df_plot[df_plot["dataset"] == ds][value_col].values
        ax.bar(x + i*width, vals, width=width, label=ds, color=colors.get(ds, None), edgecolor="black", linewidth=0.6)

    ax.set_xticks(x + (len(datasets)-1)*width/2)
    ax.set_xticklabels(clusters, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="upper right")

def plot_fig2_pjoint(summary, outpath, threshold_mode):
    df_plot = summary[summary["cluster"]!="ALL"].copy()
    # keep dataset order ERA, ScenarioMIP, HighResMIP
    df_plot["dataset"] = pd.Categorical(df_plot["dataset"], categories=["ERA","ScenarioMIP","HighResMIP"], ordered=True)
    df_plot = df_plot.sort_values(["cluster","dataset"])

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    _bar_grouped(
        ax,
        df_plot,
        value_col="P_joint",
        ylabel="Joint exceedance probability",
        title=f"Frequency of Compound Wind–Wave Extremes by Regime (mode: {threshold_mode})"
    )
    # y-format as percent
    ax.set_ylim(0, max(0.15, df_plot["P_joint"].max()*1.15))
    ax.set_yticks(np.linspace(0, ax.get_ylim()[1], 6))
    ax.set_yticklabels([f"{100*t:.0f}%" for t in ax.get_yticks()])
    ax.set_xlabel("Regime")
    fig.suptitle("Fig. 2. Probability of joint exceedance events ($P_{\\text{joint}}$) across regimes", y=1.04, fontsize=11)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_fig3_counts(summary, outpath, threshold_mode):
    df_plot = summary[summary["cluster"]!="ALL"].copy()
    df_plot["dataset"] = pd.Categorical(df_plot["dataset"], categories=["ERA","ScenarioMIP","HighResMIP"], ordered=True)
    df_plot = df_plot.sort_values(["cluster","dataset"])

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    _bar_grouped(
        ax,
        df_plot,
        value_col="coextreme_count",
        ylabel="Co-extreme count (n)",
        title=f"Counts of Compound Wind–Wave Extremes by Regime (mode: {threshold_mode})"
    )
    ax.set_xlabel("Regime")
    fig.suptitle("Fig. 3. Cluster-wise frequency (counts) of joint exceedances (ERA5 vs. models)", y=1.04, fontsize=11)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(">> Loading datasets …")
    dfs = load_data(FOLDER, FILES)

    print(f">> Computing thresholds (mode = {THRESHOLD_MODE}) …")
    thresholds = compute_thresholds(dfs, mode=THRESHOLD_MODE)

    # Print thresholds
    print("\n95th-percentile thresholds used:")
    for name in ["ERA","ScenarioMIP","HighResMIP"]:
        th = thresholds[name]
        print(f"  {name}:  Hs_p95 = {th['hs']:.3f} m,  U10_p95 = {th['u10']:.3f} m/s")

    print("\n>> Summarizing by regime …")
    summary = summarize_by_regime(dfs, thresholds, cluster_order=CLUSTER_ORDER)

    # Save tidy summary CSV
    csv_out = os.path.join(OUTDIR, f"compound_extremes_summary_{THRESHOLD_MODE}.csv")
    summary.sort_values(["cluster","dataset"]).to_csv(csv_out, index=False)
    print(f">> Wrote summary: {csv_out}")

    # Print top regimes by P_joint (ERA5 perspective)
    top_era = summary[(summary["dataset"]=="ERA") & (summary["cluster"]!="ALL")].sort_values("P_joint", ascending=False)
    print("\nTop ERA5 regimes by P_joint:")
    print(top_era[["cluster","P_joint","coextreme_count","total_count"]].head(5).to_string(index=False, formatters={"P_joint":lambda x: f"{100*x:.1f}%"}))

    # Figures
    fig2_path = os.path.join(OUTDIR, f"Fig2_joint_prob_{THRESHOLD_MODE}.png")
    fig3_path = os.path.join(OUTDIR, f"Fig3_coextreme_counts_{THRESHOLD_MODE}.png")
    print("\n>> Rendering figures …")
    plot_fig2_pjoint(summary, fig2_path, THRESHOLD_MODE)
    plot_fig3_counts(summary, fig3_path, THRESHOLD_MODE)
    print(f">> Saved: {fig2_path}")
    print(f">> Saved: {fig3_path}")

    # Console note for manuscript phrasing
    note = dedent(f"""
    Notes for manuscript (Section 3.2):
      • Threshold mode = {THRESHOLD_MODE}.
      • Fig. 2 shows regime-wise joint exceedance probability (P_joint).
      • Fig. 3 shows regime-wise counts of co-extreme events.
      • Consider reporting both 'intrinsic' and 'era5_anchored' in main vs. supplement.
    """).strip()
    print("\n" + note)

# ---- next cell ----

# -*- coding: utf-8 -*-
"""
Generate Fig2_joint_prob.png and Fig3_histogram.png
for 'Frequency and Intensity of Compound Extremes'.

- Loads ERA, ScenarioMIP, HighResMIP CSVs with regime labels
- Computes joint exceedance probability P_joint per regime:
      P_joint = P(Hs > p95 AND U10 > p95)
- Saves two figures with EXACT requested filenames:
      Fig2_joint_prob.png
      Fig3_histogram.png

Change THRESHOLD_MODE to "era5_anchored" if you want ERA5 p95 applied to all.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Paths and files (your setup)
# ----------------------------
FOLDER = DATA_DIR
FILES = {
    "ERA":         "ERA_full_filled_with_clusters_k9_named.csv",
    "ScenarioMIP": "ECN_full_filled_with_clusters_k9_named.csv",
    "HighResMIP":  "ECH_full_filled_with_clusters_k9_named.csv",
}

# thresholding: 'intrinsic' (per dataset p95) or 'era5_anchored' (ERA5 p95 for all)
THRESHOLD_MODE = "intrinsic"   # change to "era5_anchored" if needed

# Column names
COL_HS   = "mean_hs"
COL_U10  = "annual_mean_wind_speed"
COL_CLUS = "cluster_abbrev"

# Regime display order (adjust if you prefer)
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]

# Figure output directory (use the same folder for simplicity)
OUTDIR = FOLDER
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def load_df(folder, fname):
    path = os.path.join(folder, fname)
    df = pd.read_csv(path)
    for c in [COL_HS, COL_U10, COL_CLUS]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {path}")
    return df

def compute_thresholds(dfs, mode):
    if mode not in ("intrinsic", "era5_anchored"):
        raise ValueError("THRESHOLD_MODE must be 'intrinsic' or 'era5_anchored'")
    thr = {}
    if mode == "era5_anchored":
        era = dfs["ERA"]
        hs95 = era[COL_HS].quantile(0.95)
        u95  = era[COL_U10].quantile(0.95)
        for k in dfs:
            thr[k] = {"hs": hs95, "u10": u95}
    else:
        for k, df in dfs.items():
            thr[k] = {
                "hs": df[COL_HS].quantile(0.95),
                "u10": df[COL_U10].quantile(0.95),
            }
    return thr

def joint_stats_by_regime(df, hs_th, u_th):
    def _calc(g):
        mask = (g[COL_HS] > hs_th) & (g[COL_U10] > u_th)
        return pd.Series({
            "P_joint": mask.mean(),
            "coextreme_count": int(mask.sum()),
            "total_count": int(len(g))
        })
    out = df.groupby(COL_CLUS, observed=True).apply(_calc, include_groups=False).reset_index()
    return out

def tidy_summary(dfs, thresholds, cluster_order):
    rows = []
    for name, df in dfs.items():
        hs_th, u_th = thresholds[name]["hs"], thresholds[name]["u10"]
        tmp = joint_stats_by_regime(df, hs_th, u_th)
        tmp["dataset"] = name
        rows.append(tmp)
    res = pd.concat(rows, ignore_index=True)
    # order clusters + datasets
    present = [c for c in cluster_order if c in set(res[COL_CLUS])]
    res[COL_CLUS] = pd.Categorical(res[COL_CLUS], categories=present, ordered=True)
    res["dataset"] = pd.Categorical(res["dataset"], categories=["ERA","ScenarioMIP","HighResMIP"], ordered=True)
    res = res.sort_values([COL_CLUS, "dataset"])
    return res

def grouped_bar(ax, df, value_col, ylabel, title, percent=False):
    clusters = df[COL_CLUS].cat.categories.tolist()
    datasets = df["dataset"].cat.categories.tolist()
    x = np.arange(len(clusters))
    width = 0.75 / len(datasets)

    colors = {"ERA": "#595959", "ScenarioMIP": "#1f77b4", "HighResMIP": "#d62728"}

    for i, ds in enumerate(datasets):
        vals = df[df["dataset"] == ds][value_col].values
        ax.bar(x + i*width, vals, width=width, label=ds, color=colors.get(ds), edgecolor="black", linewidth=0.6)

    ax.set_xticks(x + (len(datasets)-1)*width/2)
    ax.set_xticklabels(clusters, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    if percent:
        # format as percentage on y-axis
        ymin, ymax = 0, max(0.15, df[value_col].max()*1.15)
        ax.set_ylim(ymin, ymax)
        yticks = np.linspace(ymin, ymax, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{100*t:.0f}%" for t in yticks])

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(">> Loading data")
    dfs = {k: load_df(FOLDER, v) for k, v in FILES.items()}

    print(f">> Threshold mode = {THRESHOLD_MODE}")
    thr = compute_thresholds(dfs, THRESHOLD_MODE)
    for k in ["ERA","ScenarioMIP","HighResMIP"]:
        print(f"  {k}: Hs_p95={thr[k]['hs']:.3f} m, U10_p95={thr[k]['u10']:.3f} m/s")

    print(">> Computing summary by regime")
    summary = tidy_summary(dfs, thr, CLUSTER_ORDER)

    # ---------------- Figure 2: Probability (P_joint) ----------------
    fig2_path = os.path.join(OUTDIR, "Fig2_joint_prob.png")
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    grouped_bar(
        ax,
        summary[[COL_CLUS,"dataset","P_joint"]],
        value_col="P_joint",
        ylabel="Joint exceedance probability",
        title=f"Frequency of Compound Wind–Wave Extremes by Regime (mode: {THRESHOLD_MODE})",
        percent=True
    )
    fig.suptitle("Fig. 2. Probability of joint exceedance events ($P_{\\text{joint}}$) across all regimes", y=1.04, fontsize=11)
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f">> Saved: {fig2_path}")

    # ---------------- Figure 3: Counts (histogram-style bars) ----------------
    fig3_path = os.path.join(OUTDIR, "Fig3_histogram.png")
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    grouped_bar(
        ax,
        summary[[COL_CLUS,"dataset","coextreme_count"]],
        value_col="coextreme_count",
        ylabel="Co-extreme count (n)",
        title=f"Counts of Compound Wind–Wave Extremes by Regime (mode: {THRESHOLD_MODE})",
        percent=False
    )
    fig.suptitle("Fig. 3. Cluster-wise frequency of co-extreme counts (ERA5 vs. EC-Earth3 models)", y=1.04, fontsize=11)
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f">> Saved: {fig3_path}")

    print("\nAll done. Drop these into LaTeX with:")
    print(r"\includegraphics[width=0.9\textwidth]{Fig2_joint_prob.png}")
    print(r"\includegraphics[width=0.9\textwidth]{Fig3_histogram.png}")

# ---- next cell ----

# -*- coding: utf-8 -*-
"""
Plot Fig2_joint_prob.png and Fig3_histogram.png from the saved summary CSV.
Robust to 'cluster' vs 'cluster_abbrev', drops 'ALL', and aligns bars by cluster.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
BASE = os.path.join(DATA_DIR, "compound_extremes_outputs")
CSV  = os.path.join(BASE, "compound_extremes_summary_intrinsic.csv")  # or ..._era5_anchored.csv

FIG2 = os.path.join(BASE, "Fig2_joint_prob.png")
FIG3 = os.path.join(BASE, "Fig3_histogram.png")

# === Load summary ===
df = pd.read_csv(CSV)

# Detect cluster column
if "cluster" in df.columns:
    CLUS_COL = "cluster"
elif "cluster_abbrev" in df.columns:
    CLUS_COL = "cluster_abbrev"
else:
    raise KeyError(f"No cluster column found. Columns: {list(df.columns)}")

# Drop the pseudo-cluster 'ALL' if present
df = df[df[CLUS_COL] != "ALL"].copy()

# Canonical order (use only those present)
PREFERRED_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]
present = [c for c in PREFERRED_ORDER if c in df[CLUS_COL].unique()]
if not present:
    present = sorted(df[CLUS_COL].unique())

# Order datasets
df["dataset"] = pd.Categorical(df["dataset"], categories=["ERA","ScenarioMIP","HighResMIP"], ordered=True)

# Pivot helper: align values by cluster for each dataset
def make_wide(value_col: str) -> pd.DataFrame:
    wide = df.pivot_table(
        index=CLUS_COL, columns="dataset", values=value_col, aggfunc="first"
    ).reindex(index=present)
    # If any cluster is missing for a dataset, fill with 0 so bars align
    return wide.fillna(0.0)

# Shared grouped-bar plotter (aligned by cluster)
def plot_grouped_bars(wide: pd.DataFrame, ylabel: str, title: str, percent=False, outpath:str=""):
    clusters = wide.index.tolist()
    datasets = ["ERA","ScenarioMIP","HighResMIP"]
    x = np.arange(len(clusters))
    width = 0.75 / len(datasets)

    colors = {"ERA":"#595959","ScenarioMIP":"#1f77b4","HighResMIP":"#d62728"}

    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    for i, ds in enumerate(datasets):
        heights = wide[ds].values if ds in wide.columns else np.zeros(len(clusters))
        ax.bar(
            x + i*width, heights, width=width,
            color=colors.get(ds, None), edgecolor="black", linewidth=0.6, label=ds
        )

    ax.set_xticks(x + (len(datasets)-1)*width/2)
    ax.set_xticklabels(clusters)
    ax.set_xlabel("Regime")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    if percent:
        ymin, ymax = 0, max(0.15, float(wide[datasets].to_numpy().max())*1.15)
        ax.set_ylim(ymin, ymax)
        yticks = np.linspace(ymin, ymax, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{100*t:.0f}%" for t in yticks])

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# === Fig 2: Probability (P_joint) ===
wide_p = make_wide("P_joint")
plot_grouped_bars(
    wide_p,
    ylabel="Joint exceedance probability",
    title="Frequency of compound wind–wave extremes by regime",
    percent=True,
    outpath=FIG2
)

# === Fig 3: Counts (coextreme_count) ===
wide_c = make_wide("coextreme_count")
plot_grouped_bars(
    wide_c,
    ylabel="Co-extreme count (n)",
    title="Counts of compound wind–wave extremes by regime",
    percent=False,
    outpath=FIG3
)

print("✅ Figures saved:")
print(FIG2)
print(FIG3)
