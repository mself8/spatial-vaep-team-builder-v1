# Phase 파일 매핑

## Phase 0 (환경)
- code: `phase_0_setup/code/requirements_vaep.txt`

## Phase 1 (SPADL 전처리)
- code: `phase_1_spadl/code/convert_wyscout_to_spadl.py`
- data: `phase_1_spadl/data/spadl/`

## Phase 2 (VAEP 학습/점수)
- code: `phase_2_vaep/code/train_vaep_xgboost.py`
- code: `phase_2_vaep/code/README_VAEP.md`
- data: `phase_2_vaep/data/vaep/`

## Phase 3 (전술 감지)
- code: `phase_3_tactics/code/detect_tactics_phase3.py`
- data: `phase_3_tactics/data/tactics/`

## Phase 4 (시너지 지표)
- code: `phase_4_synergy/code/compute_synergy_phase4.py`
- data: `phase_4_synergy/data/synergy/`

## Phase 5 (라인업 최적화)
- code: `phase_5_lineup/code/optimize_lineup_phase5.py`
- code: `phase_5_lineup/code/README_PHASE5.md`
- data: `phase_5_lineup/data/lineup/`

## Phase 6 (승률 예측 교차검증)
- code: `phase_6_validation/code/validate_lineup_winrate_phase6.py`
- code: `phase_6_validation/code/README_PHASE6.md`
- data: `phase_6_validation/data/`

## 호환성
기존 데이터 경로(`데이터/spadl|vaep|tactics|synergy|lineup`)는 심볼릭 링크로 유지됩니다.
`전처리/` 폴더는 중복 제거를 위해 비워둔 상태입니다.
