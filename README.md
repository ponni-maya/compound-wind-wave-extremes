# Compound Wind–Wave Extremes: Regime-Wise Diagnostics Across Reanalysis and Model Ensembles

A reproducible analysis pipeline for characterizing **compound wind–wave extremes** — their frequency, dependence structure, persistence, and design return levels — across ERA5 reanalysis and CMIP-class model output (ScenarioMIP, HighResMIP), grouped into nine spatial wind–wave regimes.

This work sits at the intersection of **extreme value theory, compound/multivariate risk, and climate model evaluation**: rather than only checking whether models reproduce mean wind–wave climate, it asks whether they reproduce the *statistics of the extremes* — joint exceedance, tail dependence, persistence, and intermittency — which is what actually matters for downtime, design loads, and compound risk in marine and coastal applications.

## Analysis stages

| Script | Question | Methods |
|---|---|---|
| `01_frequency_intensity_compound_extremes.py` | How often do wind and wave extremes co-occur, per regime? | Joint exceedance probability P(Hs>p95 ∧ U10>p95), intrinsic vs. ERA5-anchored thresholds |
| `02_model_consistency_bias_extreme_tails.py` | Do models reproduce the extreme tail, not just the mean? | GPD fitting, percentile comparison, Taylor diagrams |
| `03_dependence_structure_joint_extremes.py` | How strongly are wind and wave extremes dependent, and does that dependence hold in the tail? | Kendall's τ, empirical upper-tail dependence, Gaussian copula fits |
| `04_temporal_persistence_downtime.py` | How long do high- and low-energy states persist? | Run-length statistics, spatial Downtime Risk Index (DRI) |
| `05_extreme_mean_coupling_intermittency.py` | How "peaky" is each regime relative to its mean state? | Extreme–Mean Coupling Ratio (EMCR = p95/mean) for Hs, U10, wave power |
| `07_joint_design_return_levels.py` | What do compound design conditions (10/25/50-yr) look like, and how biased are models? | Copula-based joint return level contours, method-of-moments marginal fits |
| `08_synthesis_regime_patterns.py` | Pulling it together: where do models diverge from ERA5 across *all* metrics at once? | Cross-metric delta matrix, heatmap synthesis |

Each script writes its outputs (figures + summary tables) to `figures/`.

## Data

Input data are per-grid-cell clustered CSVs (ERA5, ScenarioMIP, HighResMIP) with wind/wave statistics and a 9-regime cluster label (`cluster_abbrev`). These are institutional/project data and are **not included in this repository**. To run the scripts, place the following files under `data/` (or point `DATA_DIR` to their location):

- `ERA_full_filled_with_clusters_k9_named.csv`
- `ECN_full_filled_with_clusters_k9_named.csv` (ScenarioMIP)
- `ECH_full_filled_with_clusters_k9_named.csv` (HighResMIP)

Expected columns include `mean_hs`, `std_hs`, `mean_t02`, `annual_mean_wind_speed`, `mean_wind_speed_DJF`, `cluster_abbrev`, `latitude`, `longitude`.

## Usage

```bash
pip install -r requirements.txt

# point at your data directory (defaults to ./data)
export DATA_DIR=/path/to/clustered_csvs
export OUT_DIR=/path/to/figures

python src/01_frequency_intensity_compound_extremes.py
python src/02_model_consistency_bias_extreme_tails.py
# ...etc, in order
```

## Notes on method choices

Several diagnostics (persistence, EMCR, design return levels) are computed as **spatial proxies** from regime-level summary statistics rather than from declustered event time series, since the underlying pipeline works from pre-aggregated per-cell CSVs rather than raw time series. This is stated explicitly in each script's docstring and is intended to be transparent about what is a proxy vs. a direct estimate — an important distinction for compound-extremes work, where naive proxies can understate genuine dependence.

## Repository structure

```
compound-wind-wave-extremes/
├── README.md
├── requirements.txt
├── data/            # not included — see Data section
├── figures/         # generated outputs
└── src/
    ├── 01_frequency_intensity_compound_extremes.py
    ├── 02_model_consistency_bias_extreme_tails.py
    ├── 03_dependence_structure_joint_extremes.py
    ├── 04_temporal_persistence_downtime.py
    ├── 05_extreme_mean_coupling_intermittency.py
    ├── 07_joint_design_return_levels.py
    └── 08_synthesis_regime_patterns.py
```

## Author

Ponni Maya — ponnithampy2016@gmail.com
