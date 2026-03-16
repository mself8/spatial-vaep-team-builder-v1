#!/usr/bin/env python3
"""Train VAEP (scores/concedes) XGBoost models and assign VAEP values to actions."""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import socceraction.spadl as spadl
from socceraction.vaep import features, formula, labels
from socceraction.vaep.base import xfns_default


# 기능: 문자열 기반 리스트/딕셔너리를 안전하게 파싱해 후속 컬럼 추출 오류를 방지한다.
# 동작/맥락: matches의 teamsData 등 JSON 유사 텍스트를 ast로 해석하며 실패 시 None 반환.
def _safe_literal(value):
    if isinstance(value, (dict, list)):
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
    except (ValueError, SyntaxError):
        return None


# 기능: csv/json/jsonl 입력을 통합 로더로 DataFrame화한다.
# 동작/맥락: 파일 확장자 분기 파싱 후 Phase2 전처리 함수들이 공통으로 사용한다.
def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return pd.DataFrame()
        if suffix in {".jsonl", ".ndjson"}:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            return pd.DataFrame(rows)
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return pd.DataFrame(loaded)
        if isinstance(loaded, dict):
            for key in ("data", "matches", "events", "players"):
                if isinstance(loaded.get(key), list):
                    return pd.DataFrame(loaded[key])
            return pd.json_normalize(loaded)
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported file extension: {path.suffix}")


# 기능: stem(events_England 등)에 해당하는 실제 입력 파일 경로를 찾는다.
# 동작/맥락: 지원 확장자를 순회해 첫 존재 파일을 반환하며 없으면 예외를 발생시킨다.
def _find_input_file(data_dir: Path, stem: str) -> Path:
    for ext in (".csv", ".json", ".jsonl", ".ndjson"):
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No input file found for {stem}")


# 기능: matches 데이터에서 경기별 홈팀 ID 시리즈를 추출한다.
# 동작/맥락: team1.teamId/home_team_id/teamsData 우선순위로 해석해 game_id→home_team_id 매핑을 만든다.
def _extract_home_team_series(matches: pd.DataFrame) -> pd.Series:
    match_id_col = "wyId" if "wyId" in matches.columns else "match_id"
    if match_id_col not in matches.columns:
        raise ValueError("matches file must contain 'wyId' or 'match_id'")

    if "team1.teamId" in matches.columns:
        home = pd.to_numeric(matches["team1.teamId"], errors="coerce")
        return pd.Series(home.values, index=matches[match_id_col]).dropna().astype(int)

    if "home_team_id" in matches.columns:
        home = pd.to_numeric(matches["home_team_id"], errors="coerce")
        return pd.Series(home.values, index=matches[match_id_col]).dropna().astype(int)

    if "teamsData" in matches.columns:
        def _extract(value):
            parsed = _safe_literal(value)
            if not isinstance(parsed, dict):
                return None
            for team_key, team_data in parsed.items():
                if isinstance(team_data, dict) and str(team_data.get("side", "")).lower() == "home":
                    return team_data.get("teamId", team_key)
            return None

        home = pd.to_numeric(matches["teamsData"].map(_extract), errors="coerce")
        return pd.Series(home.values, index=matches[match_id_col]).dropna().astype(int)

    raise ValueError("matches file must contain one of: team1.teamId, home_team_id, teamsData")


# 기능: SPADL 디렉터리의 처리 대상 리그 목록을 만든다.
# 동작/맥락: spadl_*.parquet를 스캔하고 include 인자가 있으면 필터링한다.
def _iter_spadl_leagues(spadl_dir: Path, include: list[str] | None) -> list[str]:
    leagues = sorted(
        p.stem.replace("spadl_", "")
        for p in spadl_dir.glob("spadl_*.parquet")
        if p.name != "spadl_all.parquet"
    )
    if include:
        include_set = set(include)
        return [league for league in leagues if league in include_set]
    return leagues


# 기능: 리그별 match 메타(game_id, home_team_id)를 구축한다.
# 동작/맥락: convert 단계와 동일한 매칭 기준으로 features.play_left_to_right 입력용 홈팀 정보를 제공한다.
def _build_games_meta(data_dir: Path, league: str) -> pd.DataFrame:
    matches_path = _find_input_file(data_dir, f"matches_{league}")
    matches = _load_table(matches_path)
    home_teams = _extract_home_team_series(matches)
    games = pd.DataFrame({"game_id": home_teams.index.astype(int), "home_team_id": home_teams.values.astype(int)})
    games = games.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
    return games


# 기능: 단일 경기 액션에서 VAEP 학습용 X/y를 생성한다.
# 동작/맥락: gamestates→left_to_right→xfns_default 특성, labels.scores/concedes 라벨을 만든다.
def _compute_features_and_labels_for_game(
    actions: pd.DataFrame,
    home_team_id: int,
    nb_prev_actions: int,
    nr_actions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_actions = actions.sort_values(["action_id", "time_seconds"]).reset_index(drop=True)
    named_actions = spadl.add_names(game_actions)

    gs = features.gamestates(named_actions, nb_prev_actions=nb_prev_actions)
    gs_ltr = features.play_left_to_right(gs, home_team_id=home_team_id)
    X_game = pd.concat([fn(gs_ltr) for fn in xfns_default], axis=1)

    y_scores = labels.scores(named_actions, nr_actions=nr_actions)
    y_concedes = labels.concedes(named_actions, nr_actions=nr_actions)
    y_game = pd.concat([y_scores, y_concedes], axis=1)

    key_cols = game_actions[["game_id", "action_id"]].reset_index(drop=True)
    X_game = pd.concat([key_cols, X_game.reset_index(drop=True)], axis=1)
    y_game = pd.concat([key_cols, y_game.reset_index(drop=True)], axis=1)
    return X_game, y_game


# 기능: 이진 XGBoost 분류기(득점/실점)를 학습한다.
# 동작/맥락: run_phase2에서 train/valid 분할 데이터를 받아 p_scores/p_concedes 추정 모델로 사용된다.
def _fit_binary_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


# 기능: Phase2 전체 파이프라인(특성 생성→모델 학습→VAEP 산출→저장)을 수행한다.
# 동작/맥락: 리그/경기 루프를 돌며 OOF가 아닌 전체 예측 기반 p_scores/p_concedes와 vaep_actions를 생성한다.
def run_phase2(
    data_dir: Path,
    spadl_dir: Path,
    output_dir: Path,
    leagues: list[str] | None,
    nb_prev_actions: int,
    nr_actions: int,
    val_size: float,
    random_state: int,
    max_games: int | None,
) -> None:
    selected_leagues = _iter_spadl_leagues(spadl_dir, leagues)
    if not selected_leagues:
        raise RuntimeError("No SPADL league parquet files found to train VAEP.")

    feature_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []

    for league in selected_leagues:
        spadl_path = spadl_dir / f"spadl_{league}.parquet"
        actions = pd.read_parquet(spadl_path)
        games_meta = _build_games_meta(data_dir, league)
        home_team_by_game = games_meta.set_index("game_id")["home_team_id"].to_dict()

        game_ids = [gid for gid in actions["game_id"].drop_duplicates().tolist() if gid in home_team_by_game]
        if max_games is not None:
            game_ids = game_ids[:max_games]

        print(f"[INFO] League {league}: games={len(game_ids):,}, actions={len(actions):,}")

        for game_id in game_ids:
            game_actions = actions[actions["game_id"] == game_id].copy()
            if game_actions.empty:
                continue
            X_game, y_game = _compute_features_and_labels_for_game(
                actions=game_actions,
                home_team_id=int(home_team_by_game[game_id]),
                nb_prev_actions=nb_prev_actions,
                nr_actions=nr_actions,
            )
            feature_frames.append(X_game)
            label_frames.append(y_game)
            action_frames.append(game_actions[["game_id", "action_id"]].copy())

    if not feature_frames:
        raise RuntimeError("No game features were generated. Check matches/home_team mapping.")

    X_all = pd.concat(feature_frames, ignore_index=True)
    y_all = pd.concat(label_frames, ignore_index=True)

    key_cols = ["game_id", "action_id"]
    feature_cols = [c for c in X_all.columns if c not in key_cols]

    X_matrix = X_all[feature_cols].copy()
    X_matrix = X_matrix.astype(float)

    y_scores = y_all["scores"].astype(int)
    y_concedes = y_all["concedes"].astype(int)

    idx = np.arange(len(X_matrix))
    train_idx, valid_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    X_train = X_matrix.iloc[train_idx]
    X_valid = X_matrix.iloc[valid_idx]

    print("[INFO] Training XGBoost(score)...")
    model_scores = _fit_binary_xgb(
        X_train,
        y_scores.iloc[train_idx],
        X_valid,
        y_scores.iloc[valid_idx],
        random_state=random_state,
    )

    print("[INFO] Training XGBoost(concede)...")
    model_concedes = _fit_binary_xgb(
        X_train,
        y_concedes.iloc[train_idx],
        X_valid,
        y_concedes.iloc[valid_idx],
        random_state=random_state + 7,
    )

    p_scores = model_scores.predict_proba(X_matrix)[:, 1]
    p_concedes = model_concedes.predict_proba(X_matrix)[:, 1]

    pred_df = X_all[key_cols].copy()
    pred_df["p_scores"] = p_scores
    pred_df["p_concedes"] = p_concedes

    vaep_parts: list[pd.DataFrame] = []
    for league in selected_leagues:
        spadl_path = spadl_dir / f"spadl_{league}.parquet"
        actions = pd.read_parquet(spadl_path)
        if max_games is not None:
            keep_games = pred_df["game_id"].drop_duplicates().tolist()
            actions = actions[actions["game_id"].isin(keep_games)]

        for game_id, game_actions in actions.groupby("game_id", sort=False):
            game_actions = game_actions.sort_values(["action_id", "time_seconds"]).reset_index(drop=True)
            game_preds = pred_df[pred_df["game_id"] == game_id].sort_values("action_id").reset_index(drop=True)
            if len(game_actions) != len(game_preds):
                continue

            named_actions = spadl.add_names(game_actions)
            values = formula.value(
                named_actions,
                game_preds["p_scores"],
                game_preds["p_concedes"],
            ).reset_index(drop=True)

            out = pd.concat(
                [
                    game_actions.reset_index(drop=True),
                    game_preds[["p_scores", "p_concedes"]],
                    values,
                ],
                axis=1,
            )
            vaep_parts.append(out)

    if not vaep_parts:
        raise RuntimeError("Failed to compute VAEP values. No per-game outputs were produced.")

    vaep_actions = pd.concat(vaep_parts, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    vaep_path = output_dir / "vaep_actions.parquet"
    vaep_actions.to_parquet(vaep_path, index=False)

    models_path = output_dir / "vaep_xgb_models.pkl"
    with models_path.open("wb") as f:
        pickle.dump(
            {
                "model_scores": model_scores,
                "model_concedes": model_concedes,
                "feature_cols": feature_cols,
                "nb_prev_actions": nb_prev_actions,
                "nr_actions": nr_actions,
            },
            f,
        )

    print(f"[OK] Saved VAEP actions: {vaep_path}")
    print(f"[OK] Saved models: {models_path}")
    print(f"[OK] Rows: {len(vaep_actions):,}")


# 기능: CLI 인자를 파싱하고 run_phase2를 실행하는 진입점이다.
# 동작/맥락: 실험 파라미터(nb_prev_actions, nr_actions, max_games 등)를 받아 재현 가능한 실행을 보장한다.
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Train VAEP XGBoost and assign VAEP values")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/archive"),
        help="Directory containing matches_* and players/events source files",
    )
    parser.add_argument(
        "--spadl-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/spadl"),
        help="Directory containing spadl_*.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/vaep"),
        help="Directory to write VAEP outputs",
    )
    parser.add_argument("--leagues", nargs="*", default=None, help="Optional league names")
    parser.add_argument("--nb-prev-actions", type=int, default=3)
    parser.add_argument("--nr-actions", type=int, default=10, help="Label horizon for scores/concedes")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-games", type=int, default=None, help="Optional quick-run cap for games")
    args = parser.parse_args()

    run_phase2(
        data_dir=args.data_dir,
        spadl_dir=args.spadl_dir,
        output_dir=args.output_dir,
        leagues=args.leagues,
        nb_prev_actions=args.nb_prev_actions,
        nr_actions=args.nr_actions,
        val_size=args.val_size,
        random_state=args.random_state,
        max_games=args.max_games,
    )


if __name__ == "__main__":
    main()
