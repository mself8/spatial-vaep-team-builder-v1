# Team Builder

This repository contains the experimental lineup optimization pipeline described in the paper.

The code is organized into four runnable pillars:

- `common_pipeline/`: SPADL conversion, VAEP training, tactic detection, and synergy computation
- `baseline_teambuilder/`: scalar ILP Team-Builder baseline
- `proposed_spatial_gnn_ga/`: 12-zone spatial GNN dataset builder, GNN training, and GA lineup optimization
- `evaluation_comparison/`: final hit-rate comparison between GNN and Team-Builder

## 💾 Data Preparation (데이터 준비)

이 코드를 실행하려면 원본 데이터셋이 필요합니다. 
용량 문제로 데이터는 깃허브에 포함되어 있지 않습니다.

1. [Kaggle 데이터셋 링크(여기에 링크 삽입)]에서 데이터를 다운로드합니다.
2. 다운로드한 압축 파일을 풀고, 아래 구조와 같이 `data/archive/` 폴더 안에 넣어주세요.

```text
프로젝트_폴더/
├── data/
│   └── archive/
│       ├── matches.csv      <-- 여기에 파일 위치!
│       ├── players.csv      <-- 여기에 파일 위치!
│       └── ...
├── optimize_lineup_ga_phase6.py
└── README.md

## Repository Layout

```text
team-builder/
├── common_pipeline/
├── baseline_teambuilder/
├── proposed_spatial_gnn_ga/
├── evaluation_comparison/
├── data/
├── project_paths.py
├── requirements.txt
└── README.md
```

## Data Placement

The scripts now resolve paths relative to the repository root and expect local data under `team-builder/data/`.

Typical inputs include:

- `data/archive/`
- `data/spadl/`
- `data/vaep/`
- `data/tactics/`
- `data/synergy/`
- `data/phase_4_synergy/`
- `data/phase_5_lineup/`
- `data/phase_6_validation/`
- `data/vaep_phase2_no_ltr_preproc_all/`
- `data/synergy_ilp_unified_non_england/`
- `data/synergy_ioid_england_eval_preproc_all/`

If your files are stored elsewhere, pass `--data-root` or the specific input/output paths exposed by each script.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch-geometric` needs platform-specific wheels for your environment, install the matching PyTorch and PyG wheels first, then rerun `pip install -r requirements.txt`.

## Pipeline

### 1. Common pipeline

```bash
python common_pipeline/convert_wyscout_to_spadl.py
python common_pipeline/train_vaep_xgboost.py
python common_pipeline/detect_tactics_phase3.py
python common_pipeline/compute_synergy_phase4.py
```

These scripts produce SPADL, VAEP, tactic, and synergy artifacts under `data/`.

### 2. Baseline Team-Builder

```bash
python baseline_teambuilder/optimize_lineup_phase5.py
```

This is the scalar ILP baseline. It uses Phase 4 synergy outputs when available and does not require spatial graph inputs.

### 3. Proposed spatial GNN + GA

Build the graph dataset first:

```bash
python proposed_spatial_gnn_ga/build_gnn_dataset_phase4_5.py
```

Train the model:

```bash
python proposed_spatial_gnn_ga/train_gnn_phase5.py
```

Run GA-based lineup optimization:

```bash
python proposed_spatial_gnn_ga/optimize_lineup_ga_phase6.py
```

### 4. Evaluation

Batch comparison:

```bash
python evaluation_comparison/run_experiment_batch.py
```

Standalone hit-rate evaluation:

```bash
python evaluation_comparison/evaluate_hitrate_gnn_vs_teambuilder.py \
  --gnn-pred-csv <path-to-gnn-preds> \
  --teambuilder-pred-csv <path-to-teambuilder-preds>
```

## Notes

- `io_event_surfaces_base.parquet` and `id_event_surfaces_base.parquet` are raw event logs, not spatial tensors.
- The 12-zone tensors are built inside `proposed_spatial_gnn_ga/build_gnn_dataset_phase4_5.py`.
- The baseline Team-Builder is a scalar ILP model, not the spatial GNN branch.
- `baseline_teambuilder/optimize_lineup_phase5_ga.py` implements the paper Table 2 ablation "Ours (Without GNN)" — Team-Builder objective optimized via GA.
- Shared parsing helpers (`_safe_literal`, `_to_int`) live in `utils.py` and are imported by individual scripts.
