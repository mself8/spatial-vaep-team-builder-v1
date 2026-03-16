#!/usr/bin/env python3
"""Phase 6: Match outcome cross-validation with lineup features.

- Train independent Win/Draw/Loss model (RandomForest) from historical lineups.
- Compare predicted win rate of AI optimized lineup vs actual coach lineup.
- Supports opponent-specific scenario via latest head-to-head match.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold


METRIC_COLS = ["vi", "v_total", "io", "id", "vaep_mean", "n_actions"]


# 기능: 문자열 표현 리스트/딕셔너리를 안전 파싱한다.
# 동작/맥락: matches 라인업 JSON 유사 문자열을 해석하는 _parse_lineup_players의 보조 유틸로 사용된다.
def _safe_literal(value):
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


# 기능: 경기 라인업 필드에서 선수 ID 11명 리스트를 추출한다.
# 동작/맥락: dict/list 혼합 포맷을 정규화하고 중복 제거·순서 보존해 학습/평가 피처 입력으로 사용한다.
def _parse_lineup_players(value) -> list[int]:
    parsed = _safe_literal(value)
    out: list[int] = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                pid = item.get("playerId") or item.get("player_id") or item.get("wyId")
            else:
                pid = item
            try:
                if pid is not None:
                    out.append(int(pid))
            except Exception:
                continue

    # enforce deterministic unique order
    seen = set()
    uniq = []
    for pid in out:
        if pid not in seen:
            uniq.append(pid)
            seen.add(pid)
    return uniq[:11]


# 기능: 선수 점수 테이블을 player_id 인덱스 맵과 전역 기본값으로 변환한다.
# 동작/맥락: 라인업 내 미등록 선수 발생 시 global_default로 대체해 피처 결측을 방지한다.
def _build_player_metric_map(player_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cols = ["player_id"] + [c for c in METRIC_COLS if c in player_scores.columns]
    p = player_scores[cols].copy()
    p["player_id"] = pd.to_numeric(p["player_id"], errors="coerce")
    p = p.dropna(subset=["player_id"]).copy()
    p["player_id"] = p["player_id"].astype(int)

    global_default = {}
    for col in METRIC_COLS:
        if col in p.columns:
            val = pd.to_numeric(p[col], errors="coerce").fillna(0.0)
            global_default[col] = float(val.mean())
            p[col] = val
        else:
            global_default[col] = 0.0
            p[col] = 0.0

    p = p.set_index("player_id")
    return p, global_default


# 기능: 단일 라인업을 합계/평균/최댓값/최솟값 기반 수치 피처로 집계한다.
# 동작/맥락: home/away prefix를 받아 METRIC_COLS별 통계와 known_ratio를 생성한다.
def _aggregate_lineup_features(
    lineup: list[int],
    player_metric_map: pd.DataFrame,
    global_default: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = {m: [] for m in METRIC_COLS}
    known = 0

    for pid in lineup[:11]:
        if pid in player_metric_map.index:
            known += 1
            row = player_metric_map.loc[pid]
            for m in METRIC_COLS:
                values[m].append(float(row[m]))
        else:
            for m in METRIC_COLS:
                values[m].append(float(global_default[m]))

    # pad to 11 for malformed lineups
    while len(values[METRIC_COLS[0]]) < 11:
        for m in METRIC_COLS:
            values[m].append(float(global_default[m]))

    feats: dict[str, float] = {
        f"{prefix}_known_players": float(known),
        f"{prefix}_known_ratio": float(known / 11.0),
    }

    for m in METRIC_COLS:
        arr = np.array(values[m], dtype=float)
        feats[f"{prefix}_{m}_sum"] = float(arr.sum())
        feats[f"{prefix}_{m}_mean"] = float(arr.mean())
        feats[f"{prefix}_{m}_max"] = float(arr.max())
        feats[f"{prefix}_{m}_min"] = float(arr.min())

    return feats


# 기능: 홈·원정 라인업 피처를 결합하고 차이(diff) 피처를 생성한다.
# 동작/맥락: 모델 입력 한 행을 구성하며 승무패 분류기가 직접 사용하는 X 스키마를 만든다.
def _build_match_feature_row(
    home_lineup: list[int],
    away_lineup: list[int],
    player_metric_map: pd.DataFrame,
    global_default: dict[str, float],
) -> dict[str, float]:
    fh = _aggregate_lineup_features(home_lineup, player_metric_map, global_default, prefix="home")
    fa = _aggregate_lineup_features(away_lineup, player_metric_map, global_default, prefix="away")

    out = {**fh, **fa}
    for m in METRIC_COLS:
        out[f"diff_{m}_sum"] = out[f"home_{m}_sum"] - out[f"away_{m}_sum"]
        out[f"diff_{m}_mean"] = out[f"home_{m}_mean"] - out[f"away_{m}_mean"]
        out[f"diff_{m}_max"] = out[f"home_{m}_max"] - out[f"away_{m}_max"]
        out[f"diff_{m}_min"] = out[f"home_{m}_min"] - out[f"away_{m}_min"]

    out["diff_known_ratio"] = out["home_known_ratio"] - out["away_known_ratio"]
    return out


# 기능: 홈팀 기준 스코어를 승/무/패 라벨로 변환한다.
# 동작/맥락: 학습 타깃 y 생성 시 W/D/L 다중분류 클래스를 정의한다.
def _outcome_label(team1_score: float, team2_score: float) -> str:
    if team1_score > team2_score:
        return "W"  # home win
    if team1_score < team2_score:
        return "L"  # home loss
    return "D"


# 기능: 과거 matches로 학습용 테이블(피처+라벨)을 구축한다.
# 동작/맥락: 양 팀 라인업/스코어가 유효한 경기만 사용해 RF 학습 데이터 품질을 보장한다.
def _build_training_table(matches: pd.DataFrame, player_metric_map: pd.DataFrame, global_default: dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, row in matches.iterrows():
        home_lineup = _parse_lineup_players(row.get("team1.formation.lineup"))
        away_lineup = _parse_lineup_players(row.get("team2.formation.lineup"))
        if len(home_lineup) == 0 or len(away_lineup) == 0:
            continue

        team1_score = pd.to_numeric(row.get("team1.score"), errors="coerce")
        team2_score = pd.to_numeric(row.get("team2.score"), errors="coerce")
        if pd.isna(team1_score) or pd.isna(team2_score):
            continue

        feat = _build_match_feature_row(home_lineup, away_lineup, player_metric_map, global_default)
        feat["match_id"] = int(row["wyId"]) if not pd.isna(row.get("wyId")) else None
        feat["home_team_id"] = int(row["team1.teamId"]) if not pd.isna(row.get("team1.teamId")) else None
        feat["away_team_id"] = int(row["team2.teamId"]) if not pd.isna(row.get("team2.teamId")) else None
        feat["label"] = _outcome_label(float(team1_score), float(team2_score))
        feat["dateutc"] = row.get("dateutc")
        rows.append(feat)

    if not rows:
        raise RuntimeError("No valid training rows built from matches data.")

    return pd.DataFrame(rows)


# 기능: RandomForest 승무패 모델의 교차검증 성능을 계산한다.
# 동작/맥락: fold별 accuracy/logloss를 반환해 최종 비교 결과의 신뢰도를 점검한다.
def _cross_validate_rf(X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=random_state + fold,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        )
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_va)
        pred = clf.predict(X_va)

        labels = list(clf.classes_)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(X_tr),
                "n_valid": len(X_va),
                "accuracy": float(accuracy_score(y_va, pred)),
                "logloss": float(log_loss(y_va, proba, labels=labels)),
            }
        )

    return pd.DataFrame(fold_rows)


# 기능: 전체 학습 데이터로 최종 RandomForest 모델을 학습한다.
# 동작/맥락: 실제/AI 라인업 확률 비교에 사용할 단일 배포 모델을 생성한다.
def _train_final_rf(X: pd.DataFrame, y: pd.Series, random_state: int) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=700,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
        min_samples_leaf=2,
    )
    clf.fit(X, y)
    return clf


# 기능: 모델 확률을 타깃 팀 관점(win/draw/loss)으로 변환한다.
# 동작/맥락: 모델이 홈팀 관점 W/D/L을 반환하므로 target_is_home 여부에 따라 승패 확률을 재매핑한다.
def _predict_target_probs(
    clf: RandomForestClassifier,
    feat_row: dict[str, float],
    target_is_home: bool,
) -> dict[str, float]:
    X_one = pd.DataFrame([feat_row])
    proba = clf.predict_proba(X_one)[0]
    c2p = {str(c): float(p) for c, p in zip(clf.classes_, proba)}

    # model classes are from home perspective: W(home win), D, L(home loss)
    if target_is_home:
        p_win = c2p.get("W", 0.0)
        p_draw = c2p.get("D", 0.0)
        p_loss = c2p.get("L", 0.0)
    else:
        p_win = c2p.get("L", 0.0)
        p_draw = c2p.get("D", 0.0)
        p_loss = c2p.get("W", 0.0)

    return {"win": p_win, "draw": p_draw, "loss": p_loss}


# 기능: 팀-상대 조합의 최신 맞대결 경기 1건을 찾는다.
# 동작/맥락: dateutc/wyId 기준 최신 정렬로 Phase6 시나리오의 기준(match context)을 고정한다.
def _latest_head_to_head(matches: pd.DataFrame, team_id: int, opponent_id: int) -> pd.Series:
    mask = (
        ((matches["team1.teamId"] == team_id) & (matches["team2.teamId"] == opponent_id))
        | ((matches["team1.teamId"] == opponent_id) & (matches["team2.teamId"] == team_id))
    )
    subset = matches[mask].copy()
    if subset.empty:
        raise RuntimeError(f"No head-to-head match found for team_id={team_id} vs opponent_id={opponent_id}")

    subset["dateutc_parsed"] = pd.to_datetime(subset.get("dateutc"), errors="coerce")
    subset = subset.sort_values(["dateutc_parsed", "wyId"], ascending=[False, False])
    return subset.iloc[0]


# 기능: Phase6 단일 상대 검증(학습→예측→실제vsAI 비교)을 수행한다.
# 동작/맥락: reference H2H 기준으로 동일 상대/동일 조건에서 AI 라인업의 승률 개선량을 산출한다.
def run_phase6(
    train_matches_path: Path,
    test_matches_path: Path,
    player_scores_path: Path,
    ai_lineup_path: Path,
    teams_path: Path,
    output_dir: Path,
    team_id: int,
    opponent_id: int,
    n_splits: int,
    random_state: int,
) -> None:
    train_matches = pd.read_csv(train_matches_path)
    test_matches = pd.read_csv(test_matches_path)
    teams = pd.read_csv(teams_path)
    player_scores = pd.read_parquet(player_scores_path)
    ai_lineup = pd.read_parquet(ai_lineup_path)

    player_metric_map, global_default = _build_player_metric_map(player_scores)

    train_df = _build_training_table(train_matches, player_metric_map, global_default)
    feature_cols = [c for c in train_df.columns if c not in {"label", "match_id", "home_team_id", "away_team_id", "dateutc"}]

    X = train_df[feature_cols].astype(float)
    y = train_df["label"].astype(str)

    cv_df = _cross_validate_rf(X, y, n_splits=n_splits, random_state=random_state)
    clf = _train_final_rf(X, y, random_state=random_state)

    h2h = _latest_head_to_head(test_matches, team_id=team_id, opponent_id=opponent_id)

    home_team = int(h2h["team1.teamId"])
    away_team = int(h2h["team2.teamId"])

    actual_home = _parse_lineup_players(h2h.get("team1.formation.lineup"))
    actual_away = _parse_lineup_players(h2h.get("team2.formation.lineup"))

    ai_ids = [int(x) for x in ai_lineup["player_id"].dropna().astype(int).tolist()][:11]
    if len(ai_ids) < 11:
        raise RuntimeError("AI lineup file must contain 11 selected player_ids.")

    if team_id == home_team:
        target_is_home = True
        actual_target = actual_home
        opp_lineup = actual_away
        ai_target = ai_ids

        feat_actual = _build_match_feature_row(actual_target, opp_lineup, player_metric_map, global_default)
        feat_ai = _build_match_feature_row(ai_target, opp_lineup, player_metric_map, global_default)
    elif team_id == away_team:
        target_is_home = False
        actual_target = actual_away
        opp_lineup = actual_home
        ai_target = ai_ids

        # model is home perspective, so keep home=opponent, away=target
        feat_actual = _build_match_feature_row(opp_lineup, actual_target, player_metric_map, global_default)
        feat_ai = _build_match_feature_row(opp_lineup, ai_target, player_metric_map, global_default)
    else:
        raise RuntimeError("Target team is not in selected head-to-head match.")

    probs_actual = _predict_target_probs(clf, feat_actual, target_is_home=target_is_home)
    probs_ai = _predict_target_probs(clf, feat_ai, target_is_home=target_is_home)

    team_name_map = teams.set_index("wyId")["name"].to_dict() if "wyId" in teams.columns and "name" in teams.columns else {}

    compare = pd.DataFrame(
        [
            {
                "scenario": "actual_lineup",
                "team_id": team_id,
                "team_name": team_name_map.get(team_id, str(team_id)),
                "opponent_id": opponent_id,
                "opponent_name": team_name_map.get(opponent_id, str(opponent_id)),
                "reference_match_id": int(h2h["wyId"]),
                "reference_match_dateutc": h2h.get("dateutc"),
                "target_is_home": bool(target_is_home),
                "pred_win": probs_actual["win"],
                "pred_draw": probs_actual["draw"],
                "pred_loss": probs_actual["loss"],
            },
            {
                "scenario": "ai_optimized_lineup",
                "team_id": team_id,
                "team_name": team_name_map.get(team_id, str(team_id)),
                "opponent_id": opponent_id,
                "opponent_name": team_name_map.get(opponent_id, str(opponent_id)),
                "reference_match_id": int(h2h["wyId"]),
                "reference_match_dateutc": h2h.get("dateutc"),
                "target_is_home": bool(target_is_home),
                "pred_win": probs_ai["win"],
                "pred_draw": probs_ai["draw"],
                "pred_loss": probs_ai["loss"],
            },
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "n_matches_train": int(len(train_df)),
                "cv_accuracy_mean": float(cv_df["accuracy"].mean()),
                "cv_logloss_mean": float(cv_df["logloss"].mean()),
                "win_prob_gain_ai_minus_actual": float(probs_ai["win"] - probs_actual["win"]),
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "rf_training_table.parquet", index=False)
    cv_df.to_csv(output_dir / "rf_cv_metrics.csv", index=False)
    compare.to_csv(output_dir / "lineup_winrate_comparison.csv", index=False)
    summary.to_csv(output_dir / "lineup_winrate_summary.csv", index=False)

    report = {
        "cv": cv_df.to_dict(orient="records"),
        "comparison": compare.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "lineup_winrate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Train matches file: {train_matches_path}")
    print(f"[OK] Test matches file: {test_matches_path}")
    print(f"[OK] Train matches: {len(train_df):,}")
    print(f"[OK] CV accuracy mean={cv_df['accuracy'].mean():.4f}, logloss mean={cv_df['logloss'].mean():.4f}")
    print(f"[OK] Actual lineup win prob={probs_actual['win']:.4f}")
    print(f"[OK] AI lineup win prob={probs_ai['win']:.4f}")
    print(f"[OK] Delta (AI-Actual)={probs_ai['win'] - probs_actual['win']:+.4f}")
    print(f"[OK] saved: {output_dir / 'lineup_winrate_comparison.csv'}")


# 기능: CLI 인자를 파싱하고 run_phase6를 실행하는 진입점이다.
# 동작/맥락: 팀/상대/경로/CV 파라미터를 외부에서 주입해 재현 가능한 비교 리포트를 생성한다.
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: RandomForest match outcome validation")
    parser.add_argument(
        "--matches-path",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/archive/matches_England.csv"),
        help="Historical matches CSV with lineup and score columns (legacy: used for both train/test if split args omitted)",
    )
    parser.add_argument(
        "--train-matches-path",
        type=Path,
        default=None,
        help="Optional train matches CSV path",
    )
    parser.add_argument(
        "--test-matches-path",
        type=Path,
        default=None,
        help="Optional test/reference matches CSV path",
    )
    parser.add_argument(
        "--player-scores-path",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/synergy/player_synergy_scores.parquet"),
        help="Phase 4 player score table",
    )
    parser.add_argument(
        "--ai-lineup-path",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/lineup/lineup_selected.parquet"),
        help="Phase 5 selected AI lineup parquet",
    )
    parser.add_argument(
        "--teams-path",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/archive/teams.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/phase_6_validation/data"),
    )

    parser.add_argument("--team-id", type=int, required=True, help="Target team id")
    parser.add_argument("--opponent-id", type=int, required=True, help="Opponent team id")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    train_matches_path = args.train_matches_path if args.train_matches_path is not None else args.matches_path
    test_matches_path = args.test_matches_path if args.test_matches_path is not None else train_matches_path

    run_phase6(
        train_matches_path=train_matches_path,
        test_matches_path=test_matches_path,
        player_scores_path=args.player_scores_path,
        ai_lineup_path=args.ai_lineup_path,
        teams_path=args.teams_path,
        output_dir=args.output_dir,
        team_id=args.team_id,
        opponent_id=args.opponent_id,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
