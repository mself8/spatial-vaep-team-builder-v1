# Spatial VAEP Team Builder (v1)

Implementation of the paper *"Soccer Lineup Optimization based on Spatial Action Values"* (KCC 2026) — a **spatial VAEP → HAN graph neural network → genetic algorithm** pipeline for optimizing soccer starting lineups.

> **Versioning** — this repository is the KCC 2026 snapshot (**v1**). Active development continues in [spatial-vaep-team-builder-v2](https://github.com/mself8/spatial-vaep-team-builder-v2).

The code is organized into four runnable pillars:

- `common_pipeline/`: SPADL conversion, VAEP training, tactic detection, and synergy computation
- `baseline_teambuilder/`: scalar ILP Team-Builder baseline + the paper Table 2 "Ours (Without GNN)" ablation (Team-Builder objective + GA)
- `proposed_spatial_gnn_ga/`: 12-zone spatial GNN dataset builder, HAN training, and GA lineup optimization
- `evaluation_comparison/`: batch experiment runner + hit-rate / Jaccard / win-rate(hit≥9) metrics

`main_pipeline.ipynb` runs the entire paper pipeline end-to-end and reproduces Table 1, Table 2, and Figure 3.

## Repository Layout

```text
spatial-vaep-team-builder-v1/
├── main_pipeline.ipynb           # End-to-end notebook (paper reproduction)
├── project_paths.py              # PROJECT_ROOT / DATA_DIR resolver
├── utils.py                      # Shared parsing helpers
├── requirements.txt
├── common_pipeline/
├── baseline_teambuilder/
├── proposed_spatial_gnn_ga/
├── evaluation_comparison/
└── data/                         # Local artifacts (not committed)
    └── archive/                  # Place raw Wyscout CSVs here
```

## 💾 Data Preparation

The pipeline requires the original Wyscout 2017/18 dataset. Due to its size (~3 GB), the data is **not** included in this repository.

1. Download the [Wyscout Soccer Match Event Dataset](https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset) from Kaggle.
2. Unzip it and place all CSV files under `data/archive/`.

Required input files (all inside `data/archive/`):

```text
data/archive/
├── matches_England.csv
├── matches_France.csv
├── matches_Germany.csv
├── matches_Italy.csv
├── matches_Spain.csv
├── matches_World_Cup.csv
├── matches_European_Championship.csv
├── events_England.csv  (and events_*.csv for each league)
├── players.csv
├── teams.csv
├── coaches.csv
├── player_games.csv
└── ...
```

If the data is missing, the notebook fails from Cell 2 (which builds `matches_non_england.csv`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On GPU machines, note that the `torch==1.13.1` pinned in `requirements.txt` installs as a CPU wheel. For fast GNN training on a GPU, install a CUDA wheel from the [official PyTorch index](https://pytorch.org/get-started/previous-versions/) first, then a compatible `torch-geometric` wheel:

```bash
# example (CUDA 11.7)
pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric==2.3.1
```

## Quick Start (run everything from the notebook)

```bash
jupyter lab main_pipeline.ipynb
# run the cells top to bottom
```

Running the cells in order reproduces:

- **Table 1**: VAEP/GNN predictive performance (AUC, ECE, Log Loss, Brier)
- **Table 2**: lineup Hit-Rate, Jaccard, Win-Rate(hit≥9)
- **Figure 3**: Manchester United case study (vs Arsenal, vs West Bromwich Albion)

### Model checkpoints

Trained GNN checkpoints (`*.pt`) are not committed (see `.gitignore`). Notebook Cell 16 (GNN training) creates `data/phase_5_lineup/data/gnn_phase5/hetero_edge_gat_win_epoch10_l2.pt` from scratch.

The pretrained model used for the paper results, `hetero_edge_gat_win_ood_final.pt`, must be provided separately. Without it, point the `--model-ckpt` argument in Cell 20 (the EPL-380 evaluation cell) at your own trained checkpoint instead.

## Pipeline (running scripts individually)

### 1. Common pipeline

```bash
python common_pipeline/convert_wyscout_to_spadl.py
python common_pipeline/train_vaep_xgboost.py
python common_pipeline/detect_tactics_phase3.py
python common_pipeline/compute_synergy_phase4.py
```

### 2. Baseline Team-Builder (scalar ILP)

```bash
python baseline_teambuilder/optimize_lineup_phase5.py
```

### 3. Proposed spatial GNN + GA

```bash
python proposed_spatial_gnn_ga/build_gnn_dataset_phase4_5.py
python proposed_spatial_gnn_ga/train_gnn_phase5.py
python proposed_spatial_gnn_ga/optimize_lineup_ga_phase6.py
```

### 4. Evaluation

```bash
# batch evaluation over 380 matches (reproduces Table 2)
python evaluation_comparison/run_experiment_batch.py

# compute Hit-Rate / Jaccard from two prediction CSVs
python evaluation_comparison/evaluate_hitrate_gnn_vs_teambuilder.py \
  --gnn-pred-csv <path-to-gnn-preds> \
  --teambuilder-pred-csv <path-to-teambuilder-preds>
```

## Notes

- `io_event_surfaces_base.parquet` and `id_event_surfaces_base.parquet` are raw event logs, not spatial tensors.
- The 12-zone tensors are built inside `proposed_spatial_gnn_ga/build_gnn_dataset_phase4_5.py`.
- The baseline Team-Builder is a scalar ILP model, not the spatial GNN branch.
- `baseline_teambuilder/optimize_lineup_phase5_ga.py` implements the paper Table 2 ablation "Ours (Without GNN)" — the Team-Builder objective optimized via GA.
- Shared parsing helpers (`_safe_literal`, `_to_int`) live in `utils.py` and are imported by the individual scripts.
- Paths are resolved via `project_paths.PROJECT_ROOT` (looks for a directory named `team-builder` among `__file__.parents`, falling back to the file's parent directory). The repo can therefore be cloned under any directory name — including this repository's renamed default — as long as the script files keep their relative layout.
