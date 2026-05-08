# Team Builder

Implementation of the paper *"Soccer Lineup Optimization based on Spatial Action Values"* — a spatial VAEP → HAN graph neural network → genetic algorithm pipeline for optimizing soccer starting lineups.

The code is organized into four runnable pillars:

- `common_pipeline/`: SPADL conversion, VAEP training, tactic detection, and synergy computation
- `baseline_teambuilder/`: scalar ILP Team-Builder baseline + the paper Table 2 "Ours (Without GNN)" ablation (Team-Builder objective + GA)
- `proposed_spatial_gnn_ga/`: 12-zone spatial GNN dataset builder, HAN training, and GA lineup optimization
- `evaluation_comparison/`: batch experiment runner + hit-rate / Jaccard / win-rate(hit≥9) metrics

`main_pipeline.ipynb` runs the entire paper pipeline end-to-end and reproduces Table 1, Table 2, and Figure 3.

## Repository Layout

```text
team-builder/
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

이 코드를 실행하려면 원본 Wyscout 17/18 데이터셋이 필요합니다. 용량(~3GB) 문제로 데이터는 깃허브에 포함되어 있지 않습니다.

1. Kaggle에서 [Wyscout Soccer Match Event Dataset](https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset)을 다운로드합니다.
2. 압축을 풀고 모든 CSV 파일을 `data/archive/` 아래에 배치합니다.

필수 입력 파일 목록 (모두 `data/archive/` 안에):

```text
data/archive/
├── matches_England.csv
├── matches_France.csv
├── matches_Germany.csv
├── matches_Italy.csv
├── matches_Spain.csv
├── matches_World_Cup.csv
├── matches_European_Championship.csv
├── events_England.csv  (각 리그별 events_*.csv)
├── players.csv
├── teams.csv
├── coaches.csv
├── player_games.csv
└── ...
```

데이터 누락 시 노트북 Cell 2 (matches_non_england.csv 빌드)부터 실패합니다.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU 환경의 경우 `requirements.txt`에 명시된 `torch==1.13.1`은 CPU 휠로 설치됩니다. GPU에서 GNN 학습을 빠르게 하려면 [PyTorch 공식 인덱스](https://pytorch.org/get-started/previous-versions/)에서 CUDA 휠을 선택해 별도 설치 후, 호환되는 `torch-geometric` 휠을 설치하세요.

```bash
# 예시 (CUDA 11.7)
pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric==2.3.1
```

## Quick Start (노트북으로 한 번에 실행)

```bash
jupyter lab main_pipeline.ipynb
# 위에서부터 셀을 순차 실행
```

순서대로 실행하면 다음을 재현합니다:

- **Table 1**: VAEP/GNN 예측 성능 (AUC, ECE, Log Loss, Brier)
- **Table 2**: 라인업 Hit-Rate, Jaccard, Win-Rate(hit≥9)
- **Figure 3**: Manchester United 응용 사례 (vs Arsenal, vs West Bromwich Albion)

### 모델 체크포인트 안내

GNN 학습된 체크포인트(`*.pt`)는 git에 포함되어 있지 않습니다 (.gitignore). 노트북 Cell 16(GNN 학습)이 `data/phase_5_lineup/data/gnn_phase5/hetero_edge_gat_win_epoch10_l2.pt`를 새로 생성합니다.

논문 결과 재현용 사전학습 모델 `hetero_edge_gat_win_ood_final.pt`는 별도로 제공되어야 합니다 (없으면 Cell 20의 EPL 380 평가 셀이 `--model-ckpt` 경로를 학습된 체크포인트로 변경 필요).

## Pipeline (스크립트 단독 실행)

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
# 380경기 일괄 평가 (Table 2 재현)
python evaluation_comparison/run_experiment_batch.py

# 두 모델 예측 CSV에서 Hit-Rate / Jaccard만 계산
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
- Paths are resolved via `project_paths.PROJECT_ROOT` (looks up the directory named `team-builder` in `__file__.parents`, falling back to the file's parent directory). Repo can be cloned under any directory name as long as the script files keep their relative layout.
