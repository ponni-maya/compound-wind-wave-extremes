"""
Synthesis of Regime-Wise Patterns

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
Synthesis of regime-wise patterns across metrics:
 - ΔP_joint (ScenarioMIP|HighResMIP - ERA)
 - Δtau, ΔlambdaU
 - Δ run-length (high/low)
 - ΔEMCR(Hs), ΔEMCR(U10)
 - ΔDRI (low-energy fraction)

Outputs:
  - Fig14_summary_matrix.png  (two heatmaps: ScenarioMIP vs ERA, HighResMIP vs ERA)
  - synthesis_matrix.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- PATHS -------------------------
BASE = DATA_DIR
FILES = {
    "ERA":         os.path.join(BASE, "ERA_full_filled_with_clusters_k9_named.csv"),
    "ScenarioMIP": os.path.join(BASE, "ECN_full_filled_with_clusters_k9_named.csv"),
    "HighResMIP":  os.path.join(BASE, "ECH_full_filled_with_clusters_k9_named.csv"),
}
# optional prior outputs (if present they’ll be used)
CE_SUMMARY  = os.path.join(BASE, "compound_extremes_outputs", "compound_extremes_summary_intrinsic.csv")
DEP_METRICS = os.path.join(BASE, "dependence_metrics.csv")

OUT_FIG = os.path.join(BASE, "Fig14_summary_matrix.png")
OUT_CSV = os.path.join(BASE, "synthesis_matrix.csv")

# ------------------------- SETTINGS -------------------------
CLUS_COL = "cluster_abbrev"
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]

HS_COL = "mean_hs"                    # spatial mean of Hs per grid point (proxy)
U_COL  = "annual_mean_wind_speed"     # spatial mean of U10 per grid point (proxy)

def _reindex_regime(df, col="regime"):
    """Ensure regime ordering even if some regimes are missing."""
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=CLUSTER_ORDER, ordered=True)
        df = df.sort_values(col)
    return df

# ------------------------- STEP 1: LOAD CLUSTERED CSVs -------------------------
print(">> Loading clustered CSVs …")
pool = []
for ds, path in FILES.items():
    d = pd.read_csv(path)
    d = d.dropna(subset=[HS_COL, U_COL, CLUS_COL])
    d["dataset"] = ds
    d["regime"] = d[CLUS_COL].astype(str)
    pool.append(d[["dataset","regime","longitude","latitude",HS_COL,U_COL]])
allpts = pd.concat(pool, ignore_index=True)

# ------------------------- STEP 2: EMCR (p95/mean) per regime -------------------------
def emcr_by_regime(df, value_col, rename_as):
    out = (
        df.groupby(["dataset","regime"])[value_col]
          .agg(mu="mean", p95=lambda x: np.nanpercentile(x, 95))
          .reset_index()
    )
    out[rename_as] = out["p95"] / out["mu"]
    return out[["dataset","regime", rename_as]]

emcr_hs  = emcr_by_regime(allpts, HS_COL, "EMCR_Hs")
emcr_u10 = emcr_by_regime(allpts, U_COL,  "EMCR_U10")

# ------------------------- STEP 3: DRI (low-energy fraction) per regime -------------------------
def dri_by_point(df):
    hs_thr_low = df[HS_COL].mean() - df[HS_COL].std()
    u_thr_low  = df[U_COL].mean() - df[U_COL].std()
    return ((df[HS_COL] < hs_thr_low) & (df[U_COL] < u_thr_low)).astype(int)

dri = []
for ds, path in FILES.items():
    d = pd.read_csv(path).dropna(subset=[HS_COL, U_COL, CLUS_COL])
    d["regime"] = d[CLUS_COL].astype(str)
    d["low"] = dri_by_point(d)
    g = d.groupby("regime")["low"].mean().rename("DRI").reset_index()
    g["dataset"] = ds
    dri.append(g[["dataset","regime","DRI"]])
dri = pd.concat(dri, ignore_index=True)

# ------------------------- STEP 4: Persistence proxy (mean run-length) -------------------------
def runlength_by_regime(df):
    hs_thr_h = df[HS_COL].mean() + df[HS_COL].std()
    u_thr_h  = df[U_COL].mean() + df[U_COL].std()
    hs_thr_l = df[HS_COL].mean() - df[HS_COL].std()
    u_thr_l  = df[U_COL].mean() - df[U_COL].std()

    d = df.copy()
    d["state"] = 0
    d.loc[(d[HS_COL]>hs_thr_h) & (d[U_COL]>u_thr_h), "state"] = 1
    d.loc[(d[HS_COL]<hs_thr_l) & (d[U_COL]<u_thr_l), "state"] = -1

    rows = []
    for reg, sub in d.groupby("regime"):
        s = sub["state"].to_numpy()
        rows.append((reg, (s==1).mean()*10.0, (s==-1).mean()*10.0))  # pseudo-days 0–10
    return pd.DataFrame(rows, columns=["regime","RL_high","RL_low"])

rl_list = []
for ds in FILES.keys():
    df_ds = allpts[allpts["dataset"]==ds]
    rldf = runlength_by_regime(df_ds)
    rldf["dataset"] = ds
    rl_list.append(rldf[["dataset","regime","RL_high","RL_low"]])
rl = pd.concat(rl_list, ignore_index=True)

# ------------------------- STEP 5: P_joint (robust loader) -------------------------
pj = None
if os.path.exists(CE_SUMMARY):
    pj_raw = pd.read_csv(CE_SUMMARY)

    # find regime column
    regcol = next((c for c in ["regime","cluster","cluster_abbrev"] if c in pj_raw.columns), None)
    # find dataset column
    dscol  = next((c for c in ["dataset","name","model","source"] if c in pj_raw.columns), None)
    # find or build P_joint
    pcol   = next((c for c in ["P_joint","P_joint_frac","P_joint_percent"] if c in pj_raw.columns), None)
    if pcol is None and {"coextreme_count","total_count"}.issubset(set(pj_raw.columns)):
        pj_raw["P_joint"] = pj_raw["coextreme_count"] / pj_raw["total_count"]
        pcol = "P_joint"

    if regcol and dscol and pcol:
        pj = pj_raw.rename(columns={regcol:"regime", dscol:"dataset", pcol:"P_joint"})[["dataset","regime","P_joint"]]
        if "percent" in pcol.lower() or pj["P_joint"].max() > 1.01:
            pj["P_joint"] = pj["P_joint"] / 100.0  # normalize if percent
        print(">> Found compound_extremes_summary_intrinsic.csv and will include ΔP_joint.")
    else:
        print("!! ΔP_joint skipped (required columns not found in CE summary).")
else:
    print("!! ΔP_joint skipped (compound_extremes_summary_intrinsic.csv not found)")

# ------------------------- STEP 6: Dependence metrics (robust loader) -------------------------
dep = None
if os.path.exists(DEP_METRICS):
    dep_raw = pd.read_csv(DEP_METRICS)
    regcol = next((c for c in ["regime","cluster","cluster_abbrev"] if c in dep_raw.columns), None)
    dscol  = next((c for c in ["dataset","name","model","source"] if c in dep_raw.columns), None)
    taucol = next((c for c in ["tau","kendall_tau","kendall_tau_proxy"] if c in dep_raw.columns), None)
    lambcol= next((c for c in ["lambdaU","lambda_U","upper_tail","lambda_u"] if c in dep_raw.columns), None)
    if regcol and dscol and taucol and lambcol:
        dep = dep_raw.rename(columns={regcol:"regime", dscol:"dataset", taucol:"tau", lambcol:"lambdaU"})
        dep = dep[["dataset","regime","tau","lambdaU"]]
        print(">> Found dependence_metrics.csv and will include Δtau and ΔlambdaU.")
    else:
        print("!! Δτ/Δλ_U skipped (required columns not found in dependence metrics).")
else:
    print("!! Δτ/Δλ_U skipped (dependence_metrics.csv not found)")

# ------------------------- STEP 7: Merge all metrics -------------------------
parts = [emcr_hs, emcr_u10, rl, dri]
if pj is not None:  parts.append(pj)
if dep is not None: parts.append(dep)

metrics = parts[0]
for p in parts[1:]:
    metrics = metrics.merge(p, on=["dataset","regime"], how="outer")

metrics = _reindex_regime(metrics, "regime")

# ------------------------- STEP 8: Compute deltas vs ERA per regime -------------------------
def deltas_vs_era(df, varname):
    piv = df.pivot_table(index="regime", columns="dataset", values=varname, aggfunc="first")
    for col in ["ERA","ScenarioMIP","HighResMIP"]:
        if col not in piv.columns:
            piv[col] = np.nan
    out = pd.DataFrame({
        f"Δ{varname}_ScenarioMIP": piv["ScenarioMIP"] - piv["ERA"],
        f"Δ{varname}_HighResMIP":  piv["HighResMIP"]  - piv["ERA"],
    })
    out.index.name = "regime"
    return out.reset_index()

delta_vars = []
for v in ["P_joint","tau","lambdaU","EMCR_Hs","EMCR_U10","RL_high","RL_low","DRI"]:
    if v in metrics.columns:
        delta_vars.append(deltas_vs_era(metrics[["dataset","regime",v]].dropna(subset=[v]), v))

deltas = delta_vars[0] if delta_vars else pd.DataFrame({"regime": CLUSTER_ORDER})
for dv in delta_vars[1:]:
    deltas = deltas.merge(dv, on="regime", how="outer")

full = metrics.merge(deltas, on="regime", how="left")
full = _reindex_regime(full, "regime")
full.to_csv(OUT_CSV, index=False)
print(f">> Wrote synthesis table: {OUT_CSV}")

# ------------------------- STEP 9: Heatmaps (ScenarioMIP−ERA, HighResMIP−ERA) -------------------------
present = []
for base in ["P_joint","tau","lambdaU","EMCR_Hs","EMCR_U10","RL_high","RL_low","DRI"]:
    if f"Δ{base}_ScenarioMIP" in deltas.columns and f"Δ{base}_HighResMIP" in deltas.columns:
        present.append(base)

mat_scn = deltas[["regime"] + [f"Δ{b}_ScenarioMIP" for b in present]].copy()
mat_hr  = deltas[["regime"] + [f"Δ{b}_HighResMIP"  for b in present]].copy()

label_map = {
    "P_joint":   "ΔP_joint",
    "tau":       "Δτ",
    "lambdaU":   "Δλ_U",
    "EMCR_Hs":   "ΔEMCR(Hs)",
    "EMCR_U10":  "ΔEMCR(U10)",
    "RL_high":   "ΔRL_high",
    "RL_low":    "ΔRL_low",
    "DRI":       "ΔDRI",
}
mat_scn.columns = ["regime"] + [label_map[b] for b in present]
mat_hr.columns  = ["regime"] + [label_map[b] for b in present]

def _to_matrix(df):
    df = _reindex_regime(df, "regime")
    return df["regime"].tolist(), df.drop(columns=["regime"]).to_numpy(dtype=float), df.columns[1:].tolist()

regs, A_scn, cols = _to_matrix(mat_scn)
_,    A_hr,  _    = _to_matrix(mat_hr)

vabs = np.nanmax(np.abs(np.concatenate([A_scn.flatten(), A_hr.flatten()])))
vabs = 1.0 if not np.isfinite(vabs) else float(vabs)
vmin, vmax = -vabs, vabs

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), constrained_layout=True)
cm = "coolwarm"

def heat(ax, A, title):
    im = ax.imshow(A, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(regs)))
    ax.set_yticklabels(regs)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_title(title, pad=10, fontsize=13)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(regs), 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.15)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            val = A[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    return im

im0 = heat(axes[0], A_scn, "ScenarioMIP − ERA (by regime and metric)")
heat(axes[1], A_hr,  "HighResMIP − ERA (by regime and metric)")

cbar = fig.colorbar(im0, ax=axes, orientation="vertical", fraction=0.035, pad=0.02)
cbar.set_label("Model − ERA (native units)", rotation=90)

fig.suptitle("Synthesis across metrics: frequency, dependence, persistence, intermittency, downtime",
             fontsize=14, y=1.03)
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)
print(f">> Saved: {OUT_FIG}")

# ------------------------- STEP 10: Quick textual synthesis -------------------------
absmean_scn = np.nanmean(np.abs(A_scn), axis=1)
absmean_hr  = np.nanmean(np.abs(A_hr),  axis=1)
rank_scn = sorted(zip(regs, absmean_scn), key=lambda x: x[1], reverse=True)
rank_hr  = sorted(zip(regs, absmean_hr),  key=lambda x: x[1], reverse=True)

print("\n=== Regimes most sensitive (average |Δ| across metrics) ===")
print("ScenarioMIP:", ", ".join([f"{r}({v:.2f})" for r,v in rank_scn[:4]]))
print("HighResMIP :", ", ".join([f"{r}({v:.2f})" for r,v in rank_hr[:4]]))
print("\nNote: Values reflect the magnitude of deviations from ERA across all available metrics.\n")

# ---- next cell ----

# -*- coding: utf-8 -*-
"""
Synthesis of regime-wise patterns across metrics:
 - ΔP_joint (ScenarioMIP|HighResMIP - ERA)
 - Δtau, ΔlambdaU
 - ΔRL_high, ΔRL_low
 - ΔEMCR(Hs), ΔEMCR(U10)
 - ΔDRI
Outputs:
  - Fig14_summary_matrix.png  (two heatmaps: ScenarioMIP vs ERA, HighResMIP vs ERA)
  - synthesis_matrix.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- PATHS -------------------------
BASE = DATA_DIR
FILES = {
    "ERA":         os.path.join(BASE, "ERA_full_filled_with_clusters_k9_named.csv"),
    "ScenarioMIP": os.path.join(BASE, "ECN_full_filled_with_clusters_k9_named.csv"),
    "HighResMIP":  os.path.join(BASE, "ECH_full_filled_with_clusters_k9_named.csv"),
}
CE_SUMMARY  = os.path.join(BASE, "compound_extremes_outputs", "compound_extremes_summary_intrinsic.csv")
DEP_METRICS = os.path.join(BASE, "dependence_metrics.csv")

OUT_FIG = os.path.join(BASE, "Fig14_summary_matrix.png")
OUT_CSV = os.path.join(BASE, "synthesis_matrix.csv")

# ------------------------- SETTINGS -------------------------
CLUS_COL = "cluster_abbrev"
CLUSTER_ORDER = ["NWAC","NEC","WENS","WMIC","EAC","SMC","EMC","SCES","BSC"]
HS_COL = "mean_hs"
U_COL  = "annual_mean_wind_speed"

NAME_MAP = {  # normalize dataset names everywhere
    "ERA5": "ERA", "ERA-5": "ERA", "ERA_5": "ERA", "ERA_5.0": "ERA",
    "ECN": "ScenarioMIP", "ECH": "HighResMIP"
}

def normalize_dataset_names(df):
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip().replace(NAME_MAP)
    return df

def normalize_regime_col(df):
    # allow 'regime' or 'cluster' or CLUS_COL
    if "regime" in df.columns:
        col = "regime"
    elif "cluster" in df.columns:
        df = df.rename(columns={"cluster": "regime"})
        col = "regime"
    elif CLUS_COL in df.columns:
        df = df.rename(columns={CLUS_COL: "regime"})
        col = "regime"
    else:
        raise KeyError("No regime/cluster column found.")
    df[col] = df[col].astype(str)
    # enforce order
    df[col] = pd.Categorical(df[col], categories=CLUSTER_ORDER, ordered=True)
    return df

def _reindex_regime(df, col="regime"):
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=CLUSTER_ORDER, ordered=True)
        df = df.sort_values(col)
    return df

print(">> Loading clustered CSVs …")
pool = []
for ds, path in FILES.items():
    d = pd.read_csv(path)
    d = d.dropna(subset=[HS_COL, U_COL, CLUS_COL])
    d["dataset"] = ds
    d = d.rename(columns={CLUS_COL: "regime"})
    d["regime"] = d["regime"].astype(str)
    pool.append(d[["dataset","regime","longitude","latitude",HS_COL,U_COL]])
allpts = pd.concat(pool, ignore_index=True)

# ------------------------- EMCR -------------------------
def emcr_by_regime(df, value_col, rename_as):
    out = (
        df.groupby(["dataset","regime"])[value_col]
          .agg(mu="mean", p95=lambda x: np.nanpercentile(x, 95))
          .reset_index()
    )
    out[rename_as] = out["p95"] / out["mu"]
    return out[["dataset","regime", rename_as]]

emcr_hs  = emcr_by_regime(allpts, HS_COL, "EMCR_Hs")
emcr_u10 = emcr_by_regime(allpts, U_COL,  "EMCR_U10")

# ------------------------- DRI -------------------------
def dri_by_point(df):
    hs_thr_low = df[HS_COL].mean() - df[HS_COL].std()
    u_thr_low  = df[U_COL].mean() - df[U_COL].std()
    return ((df[HS_COL] < hs_thr_low) & (df[U_COL] < u_thr_low)).astype(int)

dri_list = []
for ds, path in FILES.items():
    d = pd.read_csv(path).dropna(subset=[HS_COL, U_COL, CLUS_COL])
    d = d.rename(columns={CLUS_COL: "regime"})
    d["regime"] = d["regime"].astype(str)
    d["low"]    = dri_by_point(d)
    g = d.groupby("regime")["low"].mean().rename("DRI").reset_index()
    g["dataset"] = ds
    dri_list.append(g[["dataset","regime","DRI"]])
dri = pd.concat(dri_list, ignore_index=True)

# ------------------------- Persistence (proxy RL) -------------------------
def runlength_by_regime(df):
    hs_thr_h = df[HS_COL].mean() + df[HS_COL].std()
    u_thr_h  = df[U_COL].mean() + df[U_COL].std()
    hs_thr_l = df[HS_COL].mean() - df[HS_COL].std()
    u_thr_l  = df[U_COL].mean() - df[U_COL].std()
    x = df.copy()
    x["state"] = 0
    x.loc[(x[HS_COL]>hs_thr_h) & (x[U_COL]>u_thr_h), "state"] = 1
    x.loc[(x[HS_COL]<hs_thr_l) & (x[U_COL]<u_thr_l), "state"] = -1
    out = []
    for reg, sub in x.groupby("regime"):
        s = sub["state"].to_numpy()
        out.append((reg, (s==1).mean()*10.0, (s==-1).mean()*10.0))
    return pd.DataFrame(out, columns=["regime","RL_high","RL_low"])

rl_list = []
for ds in FILES.keys():
    rldf = runlength_by_regime(allpts[allpts["dataset"]==ds])
    rldf["dataset"] = ds
    rl_list.append(rldf[["dataset","regime","RL_high","RL_low"]])
rl = pd.concat(rl_list, ignore_index=True)

# ------------------------- Compound frequency (optional) -------------------------
pj = None
if os.path.exists(CE_SUMMARY):
    pj = pd.read_csv(CE_SUMMARY)
    pj = normalize_dataset_names(pj)
    pj = normalize_regime_col(pj)
    if "P_joint" not in pj.columns:
        # tolerate alternative spellings
        for alt in ["Pjoint","p_joint","P_joint_intrinsic"]:
            if alt in pj.columns:
                pj = pj.rename(columns={alt:"P_joint"})
                break
    if pj["P_joint"].max() > 1.01:
        pj["P_joint"] = pj["P_joint"] / 100.0
    pj = pj[["dataset","regime","P_joint"]]
    print(">> Using ΔP_joint (from compound_extremes_summary_intrinsic.csv)")
else:
    print("!! ΔP_joint skipped (file not found)")

# ------------------------- Dependence (τ, λ_U) -------------------------
dep = None
if os.path.exists(DEP_METRICS):
    dep = pd.read_csv(DEP_METRICS)
    dep = normalize_dataset_names(dep)
    dep = normalize_regime_col(dep)

    # tolerate alternate column names
    col_map = {}
    if "lambda_U" in dep.columns and "lambdaU" not in dep.columns:
        col_map["lambda_U"] = "lambdaU"
    if "tau_proxy" in dep.columns and "tau" not in dep.columns:
        col_map["tau_proxy"] = "tau"
    if col_map:
        dep = dep.rename(columns=col_map)

    missing = [c for c in ["dataset","regime","tau","lambdaU"] if c not in dep.columns]
    if missing:
        raise KeyError(f"dependence_metrics.csv is missing required columns: {missing}")

    # diagnostics
    print(">> Dependence file datasets:", sorted(dep["dataset"].unique()))
    dep = dep[["dataset","regime","tau","lambdaU"]]
else:
    print("!! Δτ/Δλ_U skipped (dependence_metrics.csv not found)")

# ------------------------- Merge all metrics -------------------------
parts = [emcr_hs, emcr_u10, rl, dri]
if pj is not None:  parts.append(pj)
if dep is not None: parts.append(dep)

metrics = parts[0]
for p in parts[1:]:
    metrics = metrics.merge(p, on=["dataset","regime"], how="outer")

metrics = _reindex_regime(metrics, "regime")

# ------------------------- Deltas vs ERA -------------------------
def deltas_vs_era(df, varname):
    piv = df.pivot_table(index="regime", columns="dataset", values=varname, aggfunc="first")
    # ensure columns exist
    for col in ["ERA","ScenarioMIP","HighResMIP"]:
        if col not in piv.columns: piv[col] = np.nan
    out = pd.DataFrame({
        f"Δ{varname}_ScenarioMIP": piv["ScenarioMIP"] - piv["ERA"],
        f"Δ{varname}_HighResMIP":  piv["HighResMIP"]  - piv["ERA"],
    })
    out.index.name = "regime"
    return out.reset_index()

delta_vars = []
for v in ["P_joint","tau","lambdaU","EMCR_Hs","EMCR_U10","RL_high","RL_low","DRI"]:
    if v in metrics.columns:
        delta_vars.append(deltas_vs_era(metrics[["dataset","regime",v]].dropna(subset=[v]), v))

deltas = delta_vars[0] if delta_vars else pd.DataFrame({"regime": CLUSTER_ORDER})
for dv in delta_vars[1:]:
    deltas = deltas.merge(dv, on="regime", how="outer")

# archive table (raw + deltas)
full = metrics.merge(deltas, on="regime", how="left")
full = _reindex_regime(full, "regime")
full.to_csv(OUT_CSV, index=False)
print(f">> Wrote synthesis table: {OUT_CSV}")

# ------------------------- Heatmaps -------------------------
present = []
for base in ["P_joint","tau","lambdaU","EMCR_Hs","EMCR_U10","RL_high","RL_low","DRI"]:
    if f"Δ{base}_ScenarioMIP" in deltas.columns and f"Δ{base}_HighResMIP" in deltas.columns:
        present.append(base)

mat_scn = deltas[["regime"] + [f"Δ{b}_ScenarioMIP" for b in present]].copy()
mat_hr  = deltas[["regime"] + [f"Δ{b}_HighResMIP"  for b in present]].copy()

label_map = {
    "P_joint":   "ΔP_joint",
    "tau":       "Δτ",
    "lambdaU":   "Δλ_U",
    "EMCR_Hs":   "ΔEMCR(Hs)",
    "EMCR_U10":  "ΔEMCR(U10)",
    "RL_high":   "ΔRL_high",
    "RL_low":    "ΔRL_low",
    "DRI":       "ΔDRI",
}
mat_scn.columns = ["regime"] + [label_map[b] for b in present]
mat_hr.columns  = ["regime"] + [label_map[b] for b in present]

def _to_matrix(df):
    df = _reindex_regime(df, "regime")
    return df["regime"].tolist(), df.drop(columns=["regime"]).to_numpy(dtype=float), df.columns[1:].tolist()

regs_scn, A_scn, cols = _to_matrix(mat_scn)
regs_hr,  A_hr,  _    = _to_matrix(mat_hr)

vabs = np.nanmax(np.abs(np.concatenate([A_scn.flatten(), A_hr.flatten()])))
vabs = 1.0 if not np.isfinite(vabs) else float(vabs)
vmin, vmax = -vabs, vabs

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), constrained_layout=True)
cm = "coolwarm"

def heat(ax, A, title, yticks):
    im = ax.imshow(A, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_title(title, pad=10, fontsize=13)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(yticks), 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.15)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            val = A[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    return im

im0 = heat(axes[0], A_scn, "ScenarioMIP − ERA (by regime and metric)", regs_scn)
im1 = heat(axes[1], A_hr,  "HighResMIP − ERA (by regime and metric)",  regs_hr)

cbar = fig.colorbar(im0, ax=axes, orientation="vertical", fraction=0.035, pad=0.02)
cbar.set_label("Model − ERA (native units)", rotation=90)

fig.suptitle("Synthesis across metrics: frequency, dependence, persistence, intermittency, downtime", fontsize=14, y=1.04)
fig.savefig(OUT_FIG, dpi=300)
plt.close(fig)
print(f">> Saved: {OUT_FIG}")

# quick sensitivity summary
absmean_scn = np.nanmean(np.abs(A_scn), axis=1)
absmean_hr  = np.nanmean(np.abs(A_hr),  axis=1)
rank_scn = sorted(zip(regs_scn, absmean_scn), key=lambda x: x[1], reverse=True)
rank_hr  = sorted(zip(regs_hr,  absmean_hr),  key=lambda x: x[1], reverse=True)
print("\n=== Regimes most sensitive (average |Δ| across metrics) ===")
print("ScenarioMIP:", ", ".join([f"{r}({v:.2f})" for r,v in rank_scn[:4]]))
print("HighResMIP :", ", ".join([f"{r}({v:.2f})" for r,v in rank_hr[:4]]))
print("\nNote: Values reflect the magnitude of deviations from ERA across all available metrics.\n")
