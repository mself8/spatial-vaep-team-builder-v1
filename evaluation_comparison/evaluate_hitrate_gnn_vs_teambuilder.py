#!/usr/bin/env python3
"""Evaluate actual-lineup hit rate: GNN vs Team-Builder.

Input expectation:
- Actual lineup source: Wyscout matches csv (team1/team2 lineup fields)
- Prediction source for each model (CSV), one of:
  1) Wide format: columns include game/team and lineup list column (json/list/pipe/comma string)
  2) Long format: columns include game/team/player_id (optional selected flag)

Outputs:
- hitrate_detail.csv: per match-team hit counts/rates
- hitrate_summary.csv: model-level mean hit rate and Jaccard similarity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils import _safe_literal, _to_int  # noqa: E402
# 기능: _dedupe_keep_order는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다.
# 데이터 입출력:
#   - Input: values: list[int], max_len: int
#   - Output: list[int]
def _dedupe_keep_order(values: list[int], max_len: int = 11) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(int(v))
        if len(out) >= max_len:
            break
    return out
# 기능: _parse_player_id_list는 컬럼 'playerId', 'player_id'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: value: object, max_len: int
#   - Output: list[int]
def _parse_player_id_list(value: object, max_len: int = 11) -> list[int]:
    if isinstance(value, list):
        raw = value
    else:
        parsed = _safe_literal(value)
        if isinstance(parsed, list):
            raw = parsed
        elif isinstance(value, str):
            txt = value.strip()
            if "|" in txt:
                raw = [tok.strip() for tok in txt.split("|") if tok.strip()]
            elif "," in txt:
                raw = [tok.strip() for tok in txt.split(",") if tok.strip()]
            else:
                raw = [txt] if txt else []
        else:
            raw = []

    ids: list[int] = []
    for row in raw:
        if isinstance(row, dict):
            pid = _to_int(row.get("playerId", row.get("player_id")))
        else:
            pid = _to_int(row)
        if pid is None:
            continue
        ids.append(pid)
    return _dedupe_keep_order(ids, max_len=max_len)
# 기능: _pick_column는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다.
# 데이터 입출력:
#   - Input: df: pd.DataFrame, explicit: str | None, candidates: list[str], label: str
#   - Output: str
def _pick_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{label} column not found: {explicit}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"failed to infer {label} column. candidates={candidates}")
# 기능: _parse_role_code는 컬럼 'code2', 'name'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다.
# 데이터 입출력:
#   - Input: role_value: object
#   - Output: str | None
def _parse_role_code(role_value: object) -> str | None:
    parsed = _safe_literal(role_value)
    if isinstance(parsed, dict):
        code2 = parsed.get("code2")
        if isinstance(code2, str):
            code2 = code2.upper().strip()
            if code2 in {"GK", "DF", "MD", "MF", "FW"}:
                return "MF" if code2 == "MD" else code2
        name = str(parsed.get("name", "")).lower()
        if "goal" in name:
            return "GK"
        if "def" in name:
            return "DF"
        if "mid" in name:
            return "MF"
        if "for" in name or "att" in name:
            return "FW"
    if isinstance(role_value, str):
        txt = role_value.upper()
        if "GK" in txt:
            return "GK"
        if "DF" in txt:
            return "DF"
        if "MD" in txt or "MF" in txt:
            return "MF"
        if "FW" in txt:
            return "FW"
    return None
# 기능: _load_goalkeepers는 컬럼 'player_id', 'wyId', 'position', 'role', 연산 pd.read_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성; 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: players_csv: Path
#   - Output: set[int]
def _load_goalkeepers(players_csv: Path) -> set[int]:
    players = pd.read_csv(players_csv)
    req = {"wyId", "role"}
    miss = sorted(req - set(players.columns))
    if miss:
        raise ValueError(f"players csv missing columns: {miss}")

    players = players[["wyId", "role"]].copy()
    players["player_id"] = pd.to_numeric(players["wyId"], errors="coerce")
    players = players.dropna(subset=["player_id"]).copy()
    players["player_id"] = players["player_id"].astype(int)
    players["position"] = players["role"].map(_parse_role_code)
    return set(players.loc[players["position"] == "GK", "player_id"].astype(int).tolist())
# 기능: _load_actual_lineups는 컬럼 'wyId', 'team1.teamId', 'team2.teamId', 'team1.formation.lineup', 'team2.formation.lineup', 연산 pd.read_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_csv: Path
#   - Output: pd.DataFrame
def _load_actual_lineups(matches_csv: Path) -> pd.DataFrame:
    matches = pd.read_csv(matches_csv)
    required = {"wyId", "team1.teamId", "team2.teamId", "team1.formation.lineup", "team2.formation.lineup"}
    miss = sorted(required - set(matches.columns))
    if miss:
        raise ValueError(f"matches csv missing columns: {miss}")

    rows: list[dict] = []
    for row in matches.to_dict(orient="records"):
        game_id = _to_int(row.get("wyId"))
        t1 = _to_int(row.get("team1.teamId"))
        t2 = _to_int(row.get("team2.teamId"))
        if game_id is None or t1 is None or t2 is None:
            continue

        l1 = _parse_player_id_list(row.get("team1.formation.lineup"), max_len=11)
        l2 = _parse_player_id_list(row.get("team2.formation.lineup"), max_len=11)
        if len(l1) == 11:
            rows.append({"game_id": game_id, "team_id": t1, "actual_ids": l1})
        if len(l2) == 11:
            rows.append({"game_id": game_id, "team_id": t2, "actual_ids": l2})

    if not rows:
        raise RuntimeError("no valid actual lineups parsed from matches csv")

    return pd.DataFrame(rows).drop_duplicates(["game_id", "team_id"]).reset_index(drop=True)
# 기능: _load_prediction_table는 컬럼 'team_id', 'game_id', 'pred_ids', 'player_id', 연산 pd.read_csv/groupby/agg을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: pred_csv: Path, game_col: str | None, team_col: str | None, lineup_col: str | None
#   - Output: pd.DataFrame
def _load_prediction_table(
    pred_csv: Path,
    game_col: str | None,
    team_col: str | None,
    lineup_col: str | None,
) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    game_id_col = _pick_column(pred, game_col, ["game_id", "match_id", "wyId"], "game_id")
    team_id_col = _pick_column(pred, team_col, ["team_id"], "team_id")

    # Mode A: explicit lineup list per row.
    inferred_lineup_col = lineup_col
    if inferred_lineup_col is None:
        for cand in ["lineup_ids", "player_ids", "pred_ids", "recommended_ids", "lineup"]:
            if cand in pred.columns:
                inferred_lineup_col = cand
                break

    if inferred_lineup_col is not None and inferred_lineup_col in pred.columns:
        use = pred[[game_id_col, team_id_col, inferred_lineup_col]].copy()
        use["game_id"] = pd.to_numeric(use[game_id_col], errors="coerce")
        use["team_id"] = pd.to_numeric(use[team_id_col], errors="coerce")
        use = use.dropna(subset=["game_id", "team_id"]).copy()
        use[["game_id", "team_id"]] = use[["game_id", "team_id"]].astype(int)
        use["pred_ids"] = use[inferred_lineup_col].map(lambda v: _parse_player_id_list(v, max_len=11))
        use = use[use["pred_ids"].map(len) > 0].copy()
        return use[["game_id", "team_id", "pred_ids"]].drop_duplicates(["game_id", "team_id"])

    # Mode B: long format (game_id, team_id, player_id, [selected]).
    if "player_id" not in pred.columns:
        raise ValueError(
            "prediction csv format not supported: need lineup column or player_id long format"
        )

    use = pred.copy()
    use["game_id"] = pd.to_numeric(use[game_id_col], errors="coerce")
    use["team_id"] = pd.to_numeric(use[team_id_col], errors="coerce")
    use["player_id"] = pd.to_numeric(use["player_id"], errors="coerce")
    use = use.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    use[["game_id", "team_id", "player_id"]] = use[["game_id", "team_id", "player_id"]].astype(int)

    selected_col = None
    for cand in ["selected", "is_selected", "pick", "chosen"]:
        if cand in use.columns:
            selected_col = cand
            break
    if selected_col is not None:
        use[selected_col] = pd.to_numeric(use[selected_col], errors="coerce").fillna(0)
        use = use[use[selected_col] > 0].copy()

    grouped = (
        use.groupby(["game_id", "team_id"], as_index=False)
        .agg(pred_ids=("player_id", lambda s: _dedupe_keep_order([int(x) for x in s.tolist()], max_len=11)))
    )
    grouped = grouped[grouped["pred_ids"].map(len) > 0].copy()
    return grouped
# 기능: _evaluate_one_model는 연산 merge을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다.
# 데이터 입출력:
#   - Input: model_name: str, pred_df: pd.DataFrame, actual_df: pd.DataFrame, goalkeeper_ids: set[int]
#   - Output: pd.DataFrame
def _evaluate_one_model(
    model_name: str,
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    goalkeeper_ids: set[int],
) -> pd.DataFrame:
    merged = actual_df.merge(pred_df, on=["game_id", "team_id"], how="inner")
    rows: list[dict] = []

    for row in merged.itertuples(index=False):
        actual_ids = _dedupe_keep_order([int(x) for x in row.actual_ids], max_len=11)
        pred_ids = _dedupe_keep_order([int(x) for x in row.pred_ids], max_len=11)

        actual_set = set(actual_ids)
        pred_set = set(pred_ids)

        hit_11 = len(actual_set & pred_set)
        hr_11 = hit_11 / 11.0
        union_11 = len(actual_set | pred_set)
        jaccard_11 = (hit_11 / float(union_11)) if union_11 > 0 else 0.0

        actual_out = [pid for pid in actual_ids if pid not in goalkeeper_ids]
        pred_out = [pid for pid in pred_ids if pid not in goalkeeper_ids]
        denom_out = max(len(actual_out), 1)
        actual_out_set = set(actual_out)
        pred_out_set = set(pred_out)
        hit_10 = len(actual_out_set & pred_out_set)
        hr_10 = hit_10 / float(denom_out)
        union_10 = len(actual_out_set | pred_out_set)
        jaccard_10 = (hit_10 / float(union_10)) if union_10 > 0 else 0.0

        rows.append(
            {
                "model": model_name,
                "game_id": int(row.game_id),
                "team_id": int(row.team_id),
                "pred_n": int(len(pred_ids)),
                "actual_n": int(len(actual_ids)),
                "hit_11": int(hit_11),
                "hit_rate_11": float(hr_11),
                "jaccard_11": float(jaccard_11),
                "hit_10": int(hit_10),
                "hit_rate_10": float(hr_10),
                "jaccard_10": float(jaccard_10),
                "pred_ids": "|".join(map(str, pred_ids)),
                "actual_ids": "|".join(map(str, actual_ids)),
            }
        )

    return pd.DataFrame(rows)
# 기능: 배치 입력 경기들을 순회하며 GNN/Team-Builder 양쪽 예측 CSV 및 오류 CSV를 저장하고 필요시 외부 evaluator를 서브프로세스로 호출한다.
# 동작/맥락: 실제 선발(ground truth)과 예측 선발을 같은 키(game_id, team_id)로 맞춘 뒤 Hit-Rate/Jaccard를 논문 표 형태로 산출하기 위해 필요하다.
# 데이터 입출력:
#   - Input: args: argparse.Namespace
#   - Output: None
def run(args: argparse.Namespace) -> None:
    actual_df = _load_actual_lineups(args.matches_csv)
    goalkeeper_ids = _load_goalkeepers(args.players_csv)

    gnn_pred = _load_prediction_table(
        pred_csv=args.gnn_pred_csv,
        game_col=args.pred_game_id_col,
        team_col=args.pred_team_id_col,
        lineup_col=args.gnn_lineup_col,
    )
    tb_pred = _load_prediction_table(
        pred_csv=args.teambuilder_pred_csv,
        game_col=args.pred_game_id_col,
        team_col=args.pred_team_id_col,
        lineup_col=args.teambuilder_lineup_col,
    )

    gnn_detail = _evaluate_one_model("GNN", gnn_pred, actual_df, goalkeeper_ids)
    tb_detail = _evaluate_one_model("Team-Builder", tb_pred, actual_df, goalkeeper_ids)
    detail = pd.concat([gnn_detail, tb_detail], ignore_index=True)

    if detail.empty:
        raise RuntimeError("no overlap between predictions and actual match-team keys")

    summary = (
        detail.groupby("model", as_index=False)
        .agg(
            n_rows=("game_id", "count"),
            hit_rate_11_mean=("hit_rate_11", "mean"),
            jaccard_11_mean=("jaccard_11", "mean"),
            hit_rate_10_mean=("hit_rate_10", "mean"),
            jaccard_10_mean=("jaccard_10", "mean"),
            hit_11_mean=("hit_11", "mean"),
            hit_10_mean=("hit_10", "mean"),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "hitrate_detail.csv"
    summary_path = output_dir / "hitrate_summary.csv"
    meta_path = output_dir / "run_metadata.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    meta = {
        "matches_csv": str(args.matches_csv),
        "players_csv": str(args.players_csv),
        "gnn_pred_csv": str(args.gnn_pred_csv),
        "teambuilder_pred_csv": str(args.teambuilder_pred_csv),
        "n_actual_rows": int(len(actual_df)),
        "n_gnn_pred_rows": int(len(gnn_pred)),
        "n_teambuilder_pred_rows": int(len(tb_pred)),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved: {detail_path}")
    print(f"[OK] saved: {summary_path}")
    print(f"[OK] saved: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare actual lineup hit rate: GNN vs Team-Builder")
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=DATA_DIR / "archive/matches_England.csv",
        help="Wyscout matches csv with team1/team2 lineup fields",
    )
    parser.add_argument(
        "--players-csv",
        type=Path,
        default=DATA_DIR / "archive/players.csv",
        help="players metadata csv (for GK exclusion in outfield hit rate)",
    )
    parser.add_argument("--gnn-pred-csv", type=Path, required=True, help="GNN lineup prediction csv")
    parser.add_argument("--teambuilder-pred-csv", type=Path, required=True, help="Team-Builder lineup prediction csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "phase_6_validation/data/hitrate_gnn_vs_teambuilder",
    )

    parser.add_argument("--pred-game-id-col", type=str, default=None, help="Optional override for prediction game id column")
    parser.add_argument("--pred-team-id-col", type=str, default=None, help="Optional override for prediction team id column")
    parser.add_argument("--gnn-lineup-col", type=str, default=None, help="Optional override for GNN lineup list column")
    parser.add_argument(
        "--teambuilder-lineup-col",
        type=str,
        default=None,
        help="Optional override for Team-Builder lineup list column",
    )

    run(parser.parse_args())
