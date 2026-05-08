#!/usr/bin/env python3
"""Phase 5: Constraint-based lineup optimization (ILP with PuLP).

Objective (paper-style):
    V = lambda1 * V_I + lambda2 * I_O + lambda3 * I_D

- V_I: per-player individual value (from Phase 4 `player_vi`)
- I_O: offensive interaction value between selected teammate pairs
- I_D: defensive interaction value attributed to defenders at possession switches

Tactic weighting:
- If an interaction belongs to targeted attack/defense tactics (cluster or zone keyword),
  multiply by `w_k > 1` before optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils import _safe_literal  # noqa: E402
# 기능: _resolve_synergy_file는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: synergy_dir: Path, filename: str
#   - Output: Path
def _resolve_synergy_file(synergy_dir: Path, filename: str) -> Path:
    """Resolve synergy files from project-root anchored absolute paths first."""
    candidates = [
        (Path(synergy_dir) / filename).resolve(),
        (PROJECT_ROOT / "data" / "synergy" / filename).resolve(),
        (DATA_DIR / "synergy" / filename).resolve(),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"synergy file not found: {filename}. checked={[str(c) for c in candidates]}"
    )
# 기능: _load_player_synergy_scores는 컬럼 'team_id', 연산 pd.read_csv/pd.read_parquet을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: synergy_dir: Path
#   - Output: pd.DataFrame
def _load_player_synergy_scores(synergy_dir: Path) -> pd.DataFrame:
    """Load player synergy scores with robust parquet/csv fallback."""
    parquet_path = _resolve_synergy_file(synergy_dir, "player_synergy_scores.parquet")
    csv_path = parquet_path.with_suffix(".csv")

    try:
        df = pd.read_parquet(parquet_path)
        src = parquet_path
    except Exception as exc:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            src = csv_path
            print(f"[WARN] failed to read parquet ({parquet_path}): {exc}. fallback to csv={csv_path}")
        else:
            raise RuntimeError(f"failed to load player synergy scores from {parquet_path}: {exc}")

    if df.empty:
        raise RuntimeError(f"player synergy table is empty: {src}")

    if "team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")

    print(f"[INFO] loaded player synergy scores: {src} rows={len(df)}")
    return df
# 기능: _parse_role_code는 컬럼 'code2', 'name'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: role_value: object
#   - Output: str | None
def _parse_role_code(role_value: object) -> str | None:
    parsed = _safe_literal(role_value)
    if isinstance(parsed, dict):
        code2 = parsed.get("code2")
        if isinstance(code2, str):
            code2 = code2.upper().strip()
            if code2 in {"GK", "DF", "MF", "FW"}:
                return code2
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
        for code in ("GK", "DF", "MF", "FW"):
            if code in txt:
                return code
    return None
# 기능: _parse_formation는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: formation: str
#   - Output: tuple[int, int, int]
def _parse_formation(formation: str) -> tuple[int, int, int]:
    parts = formation.split("-")
    if len(parts) != 3:
        raise ValueError("formation must be in format DF-MF-FW (e.g. 4-3-3)")
    df_n, mf_n, fw_n = [int(x) for x in parts]
    if df_n + mf_n + fw_n != 10:
        raise ValueError("formation must sum to 10 outfield players (GK is fixed to 1)")
    return df_n, mf_n, fw_n
# 기능: _build_event_vaep_map는 컬럼 'original_event_id', 연산 groupby/agg을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: vaep_actions: pd.DataFrame
#   - Output: pd.DataFrame
def _build_event_vaep_map(vaep_actions: pd.DataFrame) -> pd.DataFrame:
    return (
        vaep_actions.dropna(subset=["original_event_id"])
        .groupby(["game_id", "original_event_id"], as_index=False)
        .agg(vaep_value=("vaep_value", "mean"))
    )
# 기능: _contains_any_keyword는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: text: str | None, keywords: list[str]
#   - Output: bool
def _contains_any_keyword(text: str | None, keywords: list[str]) -> bool:
    if not keywords:
        return False
    text_norm = str(text or "").lower()
    return any(k.lower() in text_norm for k in keywords)
# 기능: _load_opponent_lineup_ids는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: opponent_lineup_ids: list[int]
#   - Output: list[int]
def _load_opponent_lineup_ids(opponent_lineup_ids: list[int]) -> list[int]:
    cleaned: list[int] = []
    seen: set[int] = set()
    for pid in opponent_lineup_ids:
        p = int(pid)
        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)
    if len(cleaned) != 11:
        raise ValueError(
            "opponent lineup must contain exactly 11 unique player ids for Equation 5 I_D filtering"
        )
    return cleaned
# 기능: _load_available_player_ids는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: available_player_ids: list[int]
#   - Output: list[int]
def _load_available_player_ids(available_player_ids: list[int]) -> list[int]:
    cleaned: list[int] = []
    seen: set[int] = set()
    for pid in available_player_ids:
        p = int(pid)
        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)
    if len(cleaned) < 11:
        raise ValueError("available_player_ids must contain at least 11 unique player ids")
    return cleaned
# 기능: _load_phase4_id_pairs는 컬럼 'defending_team_id', 'id_sum', 'defender_player_id', 'opponent_team_id', 'opponent_player_id', 연산 pd.read_parquet/groupby/agg을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: synergy_dir: Path, target_team_id: int
#   - Output: pd.DataFrame
def _load_phase4_id_pairs(
    synergy_dir: Path,
    target_team_id: int,
) -> pd.DataFrame:
    id_pair = pd.read_parquet(synergy_dir / "defense_interaction_id.parquet")

    required = {"defending_team_id", "defender_player_id", "opponent_team_id", "opponent_player_id"}
    miss = sorted(required - set(id_pair.columns))
    if miss:
        raise RuntimeError(f"defense_interaction_id.parquet missing required columns: {miss}")

    id_pair["defending_team_id"] = pd.to_numeric(id_pair["defending_team_id"], errors="coerce")
    id_pair = id_pair[id_pair["defending_team_id"] == int(target_team_id)].copy()

    value_col = None
    for cand in ["id_sum", "id", "id_90_raw", "id_weighted_sum"]:
        if cand in id_pair.columns:
            value_col = cand
            break
    if value_col is None:
        raise RuntimeError(
            "defense_interaction_id.parquet must include one of: id_sum, id, id_90_raw, id_weighted_sum"
        )

    id_pair["id_sum"] = pd.to_numeric(id_pair[value_col], errors="coerce").fillna(0.0)
    id_pair["defender_player_id"] = pd.to_numeric(id_pair["defender_player_id"], errors="coerce")
    id_pair["opponent_team_id"] = pd.to_numeric(id_pair["opponent_team_id"], errors="coerce")
    id_pair["opponent_player_id"] = pd.to_numeric(id_pair["opponent_player_id"], errors="coerce")

    id_pair = id_pair.dropna(subset=["defender_player_id", "opponent_team_id", "opponent_player_id"]).copy()
    id_pair[["defender_player_id", "opponent_team_id", "opponent_player_id"]] = id_pair[
        ["defender_player_id", "opponent_team_id", "opponent_player_id"]
    ].astype(int)

    return (
        id_pair.groupby(
            ["defender_player_id", "opponent_team_id", "opponent_player_id"],
            as_index=False,
        )
        .agg(id_sum=("id_sum", "sum"), id_count=("id_sum", "count"))
        .assign(id_weighted_share=1.0)
    )
# 기능: _load_lambda_scaler_stats는 컬럼 'scaler_type', 'vi_mean', 'vi_scale', 'vi_median', 'vi_q25', 연산 pd.read_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: path: Path | None
#   - Output: dict | None
def _load_lambda_scaler_stats(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"lambda scaler stats not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"lambda scaler stats csv is empty: {path}")
    row = df.iloc[-1]
    scaler_type = str(row.get("scaler_type", "")).strip().lower()
    if scaler_type not in {"standard", "minmax", "robust"}:
        raise ValueError("lambda scaler stats must include scaler_type in {standard,minmax,robust}")
    return {
        "scaler_type": scaler_type,
        "feature_order": ["vi_game", "io_game", "id_game"],
        "feature_stats": {
            "vi_game": {
                "mean": float(row.get("vi_mean", np.nan)),
                "scale": float(row.get("vi_scale", np.nan)),
                "median": float(row.get("vi_median", np.nan)),
                "q25": float(row.get("vi_q25", np.nan)),
                "q75": float(row.get("vi_q75", np.nan)),
                "min": float(row.get("vi_min", np.nan)),
                "max": float(row.get("vi_max", np.nan)),
            },
            "io_game": {
                "mean": float(row.get("io_mean", np.nan)),
                "scale": float(row.get("io_scale", np.nan)),
                "median": float(row.get("io_median", np.nan)),
                "q25": float(row.get("io_q25", np.nan)),
                "q75": float(row.get("io_q75", np.nan)),
                "min": float(row.get("io_min", np.nan)),
                "max": float(row.get("io_max", np.nan)),
            },
            "id_game": {
                "mean": float(row.get("id_mean", np.nan)),
                "scale": float(row.get("id_scale", np.nan)),
                "median": float(row.get("id_median", np.nan)),
                "q25": float(row.get("id_q25", np.nan)),
                "q75": float(row.get("id_q75", np.nan)),
                "min": float(row.get("id_min", np.nan)),
                "max": float(row.get("id_max", np.nan)),
            },
        },
    }
# 기능: _transform_value는 컬럼 'feature_stats', 'scaler_type', 'mean', 'scale', 'min'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: value: float, feature: str, scaler_stats: dict | None
#   - Output: float
def _transform_value(value: float, feature: str, scaler_stats: dict | None) -> float:
    if scaler_stats is None:
        return float(value)
    fs = scaler_stats.get("feature_stats", {}).get(feature, {})
    scaler_type = str(scaler_stats.get("scaler_type", "")).lower()
    x = float(value)
    if scaler_type == "standard":
        mean = float(fs.get("mean", 0.0))
        scale = float(fs.get("scale", 1.0))
        if not np.isfinite(scale) or abs(scale) <= 1e-12:
            scale = 1.0
        return float((x - mean) / scale)
    if scaler_type == "minmax":
        xmin = float(fs.get("min", 0.0))
        xmax = float(fs.get("max", 1.0))
        denom = xmax - xmin
        if not np.isfinite(denom) or abs(denom) <= 1e-12:
            return 0.0
        return float((x - xmin) / denom)
    return float(value)
# 기능: _linear_scale_factor는 컬럼 'feature_stats', 'scaler_type', 'scale', 'min', 'max'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: feature: str, scaler_stats: dict | None
#   - Output: float
def _linear_scale_factor(feature: str, scaler_stats: dict | None) -> float:
    """Return multiplicative factor to align raw objective terms with scaled-lambda space.

    Lambda estimation in Phase4 uses game-level scaled features. In Phase5 objective,
    we optimize sums of raw player/pair terms. To keep linearity and avoid per-term
    sign distortion, apply only multiplicative scaling (1/scale or 1/range).
    Additive offsets from centering are constants for fixed-size lineups and do not
    affect argmax.
    """
    if scaler_stats is None:
        return 1.0
    fs = scaler_stats.get("feature_stats", {}).get(feature, {})
    scaler_type = str(scaler_stats.get("scaler_type", "")).lower()
    if scaler_type == "standard":
        scale = float(fs.get("scale", 1.0))
        if not np.isfinite(scale) or abs(scale) <= 1e-12:
            return 1.0
        return float(1.0 / scale)
    if scaler_type == "minmax":
        xmin = float(fs.get("min", 0.0))
        xmax = float(fs.get("max", 1.0))
        denom = xmax - xmin
        if not np.isfinite(denom) or abs(denom) <= 1e-12:
            return 1.0
        return float(1.0 / denom)
    if scaler_type == "robust":
        scale = float(fs.get("scale", fs.get("iqr", 1.0)))
        if not np.isfinite(scale) or abs(scale) <= 1e-12:
            return 1.0
        return float(1.0 / scale)
    return 1.0
# 기능: _compute_weighted_io는 컬럼 'vaep_value', 'next_player_id', 'player_id', 'next_team_id', 'team_id', 연산 groupby/agg/merge/sort_values을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: atomic_with_phase: pd.DataFrame, phase_summary: pd.DataFrame, event_vaep: pd.DataFrame, target_team_id: int, attack_weight: float, attack_clusters: set[int], ...
#   - Output: pd.DataFrame
def _compute_weighted_io(
    atomic_with_phase: pd.DataFrame,
    phase_summary: pd.DataFrame,
    event_vaep: pd.DataFrame,
    target_team_id: int,
    attack_weight: float,
    attack_clusters: set[int],
    attack_keywords: list[str],
) -> pd.DataFrame:
    actions = atomic_with_phase.merge(event_vaep, on=["game_id", "original_event_id"], how="left")
    if "vaep_value" not in actions.columns:
        vx = pd.to_numeric(actions.get("vaep_value_x"), errors="coerce") if "vaep_value_x" in actions.columns else pd.Series(index=actions.index, dtype=float)
        vy = pd.to_numeric(actions.get("vaep_value_y"), errors="coerce") if "vaep_value_y" in actions.columns else pd.Series(index=actions.index, dtype=float)
        actions["vaep_value"] = vx.fillna(vy).fillna(0.0)
    else:
        actions["vaep_value"] = pd.to_numeric(actions["vaep_value"], errors="coerce").fillna(0.0)

    phase_meta = phase_summary[["game_id", "phase_id", "attack_cluster", "attack_zone_sequence"]].copy()
    actions = actions.merge(phase_meta, on=["game_id", "phase_id"], how="left")

    actions = actions.sort_values(["game_id", "phase_id", "time_seconds", "action_id"]).reset_index(drop=True)
    grp = ["game_id", "phase_id"]
    actions["next_player_id"] = actions.groupby(grp)["player_id"].shift(-1)
    actions["next_team_id"] = actions.groupby(grp)["team_id"].shift(-1)
    actions["next_vaep"] = actions.groupby(grp)["vaep_value"].shift(-1)

    pairs = actions[actions["next_player_id"].notna()].copy()
    pairs = pairs[pairs["team_id"] == pairs["next_team_id"]]
    pairs = pairs[pairs["player_id"] != pairs["next_player_id"]]
    pairs = pairs[pairs["team_id"] == target_team_id]

    pairs["io_raw"] = pairs["vaep_value"] + pairs["next_vaep"]

    attack_cluster_col = "attack_cluster"
    if attack_cluster_col not in pairs.columns:
        if "attack_cluster_x" in pairs.columns:
            attack_cluster_col = "attack_cluster_x"
        elif "attack_cluster_y" in pairs.columns:
            attack_cluster_col = "attack_cluster_y"

    attack_zone_col = "attack_zone_sequence"
    if attack_zone_col not in pairs.columns:
        if "attack_zone_sequence_x" in pairs.columns:
            attack_zone_col = "attack_zone_sequence_x"
        elif "attack_zone_sequence_y" in pairs.columns:
            attack_zone_col = "attack_zone_sequence_y"

    use_weight = (
        pd.to_numeric(pairs.get(attack_cluster_col), errors="coerce").fillna(-999).astype(int).isin(attack_clusters)
        | pairs.get(attack_zone_col, pd.Series(index=pairs.index, dtype=object)).map(lambda s: _contains_any_keyword(s, attack_keywords))
    )
    pairs["tactic_weight"] = 1.0
    pairs.loc[use_weight, "tactic_weight"] = attack_weight
    pairs["io_weighted"] = pairs["io_raw"] * pairs["tactic_weight"]

    io_pair = (
        pairs.groupby(["player_id", "next_player_id"], as_index=False)
        .agg(
            io_sum=("io_weighted", "sum"),
            io_count=("io_weighted", "count"),
            io_weighted_share=("tactic_weight", "mean"),
        )
        .rename(columns={"player_id": "player_a_id", "next_player_id": "player_b_id"})
    )
    io_pair["player_a_id"] = io_pair["player_a_id"].astype(int)
    io_pair["player_b_id"] = io_pair["player_b_id"].astype(int)
    return io_pair
# 기능: _compute_weighted_id는 컬럼 'vaep_value', 'defense_cluster', 'defense_zone_sequence', 'next_phase_id', 'phase_id', 연산 groupby/agg/merge/sort_values을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: atomic_with_phase: pd.DataFrame, phase_summary: pd.DataFrame, event_vaep: pd.DataFrame, target_team_id: int, defense_weight: float, defense_clusters: set[int], ...
#   - Output: pd.DataFrame
def _compute_weighted_id(
    atomic_with_phase: pd.DataFrame,
    phase_summary: pd.DataFrame,
    event_vaep: pd.DataFrame,
    target_team_id: int,
    defense_weight: float,
    defense_clusters: set[int],
    defense_keywords: list[str],
) -> pd.DataFrame:
    actions = atomic_with_phase.merge(event_vaep, on=["game_id", "original_event_id"], how="left")
    if "vaep_value" not in actions.columns:
        vx = pd.to_numeric(actions.get("vaep_value_x"), errors="coerce") if "vaep_value_x" in actions.columns else pd.Series(index=actions.index, dtype=float)
        vy = pd.to_numeric(actions.get("vaep_value_y"), errors="coerce") if "vaep_value_y" in actions.columns else pd.Series(index=actions.index, dtype=float)
        actions["vaep_value"] = vx.fillna(vy).fillna(0.0)
    else:
        actions["vaep_value"] = pd.to_numeric(actions["vaep_value"], errors="coerce").fillna(0.0)

    ordered = actions.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)

    phase_first = (
        ordered.groupby(["game_id", "phase_id"], as_index=False)
        .agg(
            period_id=("period_id", "first"),
            first_team_id=("team_id", "first"),
            first_player_id=("player_id", "first"),
            first_vaep=("vaep_value", "first"),
            start_time=("time_seconds", "first"),
        )
    )
    phase_last = (
        ordered.groupby(["game_id", "phase_id"], as_index=False)
        .agg(
            last_team_id=("team_id", "last"),
            last_player_id=("player_id", "last"),
            last_vaep=("vaep_value", "last"),
        )
    )

    phase_table = phase_first.merge(phase_last, on=["game_id", "phase_id"], how="inner")
    phase_table = phase_table.merge(
        phase_summary[["game_id", "phase_id", "defense_cluster", "defense_zone_sequence"]],
        on=["game_id", "phase_id"],
        how="left",
    )

    if "defense_cluster" not in phase_table.columns:
        dcx = pd.to_numeric(phase_table.get("defense_cluster_x"), errors="coerce") if "defense_cluster_x" in phase_table.columns else pd.Series(index=phase_table.index, dtype=float)
        dcy = pd.to_numeric(phase_table.get("defense_cluster_y"), errors="coerce") if "defense_cluster_y" in phase_table.columns else pd.Series(index=phase_table.index, dtype=float)
        phase_table["defense_cluster"] = dcx.fillna(dcy)

    if "defense_zone_sequence" not in phase_table.columns:
        dzx = phase_table.get("defense_zone_sequence_x") if "defense_zone_sequence_x" in phase_table.columns else pd.Series(index=phase_table.index, dtype=object)
        dzy = phase_table.get("defense_zone_sequence_y") if "defense_zone_sequence_y" in phase_table.columns else pd.Series(index=phase_table.index, dtype=object)
        phase_table["defense_zone_sequence"] = dzx.fillna(dzy)

    phase_table = phase_table.sort_values(["game_id", "period_id", "start_time", "phase_id"]).reset_index(drop=True)

    phase_table["next_phase_id"] = phase_table.groupby("game_id")["phase_id"].shift(-1)
    phase_table["next_period_id"] = phase_table.groupby("game_id")["period_id"].shift(-1)
    phase_table["next_first_team_id"] = phase_table.groupby("game_id")["first_team_id"].shift(-1)
    phase_table["next_first_player_id"] = phase_table.groupby("game_id")["first_player_id"].shift(-1)
    phase_table["next_first_vaep"] = phase_table.groupby("game_id")["first_vaep"].shift(-1)
    phase_table["next_defense_cluster"] = phase_table.groupby("game_id")["defense_cluster"].shift(-1)
    phase_table["next_defense_zone_sequence"] = phase_table.groupby("game_id")["defense_zone_sequence"].shift(-1)

    trans = phase_table[phase_table["next_phase_id"].notna()].copy()
    trans = trans[trans["period_id"] == trans["next_period_id"]]
    trans = trans[trans["last_team_id"] != trans["next_first_team_id"]]
    trans = trans[trans["next_first_team_id"] == target_team_id]

    trans["next_first_player_id"] = pd.to_numeric(trans["next_first_player_id"], errors="coerce")
    trans["last_player_id"] = pd.to_numeric(trans["last_player_id"], errors="coerce")
    trans["last_team_id"] = pd.to_numeric(trans["last_team_id"], errors="coerce")
    trans = trans.dropna(subset=["next_first_player_id", "last_player_id", "last_team_id"]).copy()
    if trans.empty:
        return pd.DataFrame(
            columns=[
                "defender_player_id",
                "opponent_team_id",
                "opponent_player_id",
                "id_sum",
                "id_count",
                "id_weighted_share",
            ]
        )

    trans[["next_first_player_id", "last_player_id", "last_team_id"]] = trans[
        ["next_first_player_id", "last_player_id", "last_team_id"]
    ].astype(int)

    trans["id_raw"] = trans["last_vaep"] + trans["next_first_vaep"]

    use_weight = (
        trans["next_defense_cluster"].fillna(-999).astype(int).isin(defense_clusters)
        | trans["next_defense_zone_sequence"].map(lambda s: _contains_any_keyword(s, defense_keywords))
    )
    trans["tactic_weight"] = 1.0
    trans.loc[use_weight, "tactic_weight"] = defense_weight
    trans["id_weighted"] = trans["id_raw"] * trans["tactic_weight"]

    id_pair = (
        trans.groupby(["next_first_player_id", "last_team_id", "last_player_id"], as_index=False)
        .agg(
            id_sum=("id_weighted", "sum"),
            id_count=("id_weighted", "count"),
            id_weighted_share=("tactic_weight", "mean"),
        )
        .rename(
            columns={
                "next_first_player_id": "defender_player_id",
                "last_team_id": "opponent_team_id",
                "last_player_id": "opponent_player_id",
            }
        )
    )
    id_pair[["defender_player_id", "opponent_team_id", "opponent_player_id"]] = id_pair[
        ["defender_player_id", "opponent_team_id", "opponent_player_id"]
    ].astype(int)
    return id_pair
# 기능: team_id·formation 제약(GK/DF/MF/FW 인원수) 아래 PuLP 이진변수(x,y)로 VI/IO/ID 목적함수를 최적화해 Team-Builder 라인업을 산출한다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성; lambda_vi/lambda_io/lambda_id 선형 결합 목적함수를 사용한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: synergy_dir: Path, tactics_dir: Path, archive_dir: Path, output_dir: Path, team_id: int | None, formation: str, ...
#   - Output: None
def run_phase5(
    synergy_dir: Path,
    tactics_dir: Path,
    archive_dir: Path,
    output_dir: Path,
    team_id: int | None,
    formation: str,
    lambda_vi: float,
    lambda_io: float,
    lambda_id: float,
    attack_weight: float,
    defense_weight: float,
    attack_clusters: list[int],
    defense_clusters: list[int],
    attack_keywords: list[str],
    defense_keywords: list[str],
    solver_time_limit: int,
    io_source: str,
    id_source: str,
    lambda_csv: Path | None,
    lambda_scaler_stats: Path | None,
    opponent_lineup_ids: list[int] | None = None,
    opponent_team_id: int | None = None,
    available_player_ids: list[int] | None = None,
) -> None:
    player_scores = _load_player_synergy_scores(synergy_dir)
    players = pd.read_csv(archive_dir / "players.csv")

    if lambda_csv is not None:
        if not lambda_csv.exists():
            raise FileNotFoundError(f"lambda_csv not found: {lambda_csv}")
        ldf = pd.read_csv(lambda_csv)
        if ldf.empty:
            raise RuntimeError(f"lambda_csv is empty: {lambda_csv}")
        row = ldf.iloc[-1]
        if {"lambda_vi", "lambda_io", "lambda_id"}.issubset(ldf.columns):
            lambda_vi = float(row["lambda_vi"])
            lambda_io = float(row["lambda_io"])
            lambda_id = float(row["lambda_id"])
        else:
            raise ValueError("lambda_csv must include columns: lambda_vi, lambda_io, lambda_id")

    scaler_stats = _load_lambda_scaler_stats(lambda_scaler_stats)

    use_equation5_id = opponent_lineup_ids is not None and len(opponent_lineup_ids) > 0
    if use_equation5_id:
        opponent_lineup_ids = _load_opponent_lineup_ids(opponent_lineup_ids)
        opponent_lineup_set = set(opponent_lineup_ids)
    else:
        opponent_lineup_ids = []
        opponent_lineup_set: set[int] = set()
        print("[WARN] opponent lineup not provided. Falling back to legacy ID aggregation (not Equation 5 strict mode).")

    use_available_filter = available_player_ids is not None and len(available_player_ids) > 0
    if use_available_filter:
        available_player_ids = _load_available_player_ids(available_player_ids)
        available_player_set = set(available_player_ids)
    else:
        available_player_ids = []
        available_player_set: set[int] = set()

    if team_id is None:
        team_id = int(player_scores["team_id"].value_counts().index[0])

    team_players = player_scores[player_scores["team_id"] == team_id].copy()
    if team_players.empty:
        known = sorted(
            pd.to_numeric(player_scores.get("team_id", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"No player scores found for team_id={team_id}. "
            f"loaded_rows={len(player_scores)} known_team_ids_sample={known[:20]}"
        )

    players_meta = players[["wyId", "shortName", "role"]].copy()
    players_meta["position"] = players_meta["role"].map(_parse_role_code)
    players_meta = players_meta.rename(columns={"wyId": "player_id", "shortName": "player_name"})

    team_players = team_players.merge(players_meta[["player_id", "player_name", "position"]], on="player_id", how="left")
    team_players = team_players[team_players["position"].isin(["GK", "DF", "MF", "FW"])].copy()

    if use_available_filter:
        team_players = team_players[team_players["player_id"].isin(available_player_set)].copy()
        if team_players.empty:
            raise RuntimeError(
                "No valid players remain after available_player_ids filter; "
                "check matchday squad extraction and player id schema"
            )

    if team_players.empty:
        raise RuntimeError("No players with valid positions (GK/DF/MF/FW) after merge with players metadata.")

    if io_source == "phase4":
        io_pair = pd.read_parquet(synergy_dir / "attack_interaction_io.parquet")
        io_pair = io_pair[pd.to_numeric(io_pair["team_id"], errors="coerce") == int(team_id)].copy()
        keep_cols = [c for c in ["player_a_id", "player_b_id", "io"] if c in io_pair.columns]
        if len(keep_cols) < 3:
            raise RuntimeError("attack_interaction_io.parquet must include player_a_id, player_b_id, io")
        io_pair = io_pair[keep_cols].copy()
        io_pair["io_sum"] = pd.to_numeric(io_pair["io"], errors="coerce").fillna(0.0)
    else:
        vaep_actions = pd.read_parquet(DATA_DIR / "vaep/vaep_actions.parquet")
        atomic_with_phase = pd.read_parquet(tactics_dir / "atomic_actions_with_phase.parquet")
        phase_summary = pd.read_parquet(tactics_dir / "phase_summary.parquet")
        event_vaep = _build_event_vaep_map(vaep_actions)
        io_pair = _compute_weighted_io(
            atomic_with_phase=atomic_with_phase,
            phase_summary=phase_summary,
            event_vaep=event_vaep,
            target_team_id=team_id,
            attack_weight=attack_weight,
            attack_clusters=set(attack_clusters),
            attack_keywords=attack_keywords,
        )

    if use_equation5_id:
        if id_source == "phase4":
            id_pair = _load_phase4_id_pairs(synergy_dir=synergy_dir, target_team_id=int(team_id))
        else:
            vaep_actions = pd.read_parquet(DATA_DIR / "vaep/vaep_actions.parquet")
            atomic_with_phase = pd.read_parquet(tactics_dir / "atomic_actions_with_phase.parquet")
            phase_summary = pd.read_parquet(tactics_dir / "phase_summary.parquet")
            event_vaep = _build_event_vaep_map(vaep_actions)
            id_pair = _compute_weighted_id(
                atomic_with_phase=atomic_with_phase,
                phase_summary=phase_summary,
                event_vaep=event_vaep,
                target_team_id=team_id,
                defense_weight=defense_weight,
                defense_clusters=set(defense_clusters),
                defense_keywords=defense_keywords,
            )

        id_pair_filtered = id_pair[id_pair["opponent_player_id"].isin(opponent_lineup_set)].copy()
        if opponent_team_id is not None:
            id_pair_filtered = id_pair_filtered[
                pd.to_numeric(id_pair_filtered["opponent_team_id"], errors="coerce") == int(opponent_team_id)
            ].copy()

        id_pair_filtered["id_sum"] = pd.to_numeric(id_pair_filtered.get("id_sum", 0.0), errors="coerce").fillna(0.0)
        if id_pair_filtered.empty:
            print("[WARN] No defender-opponent ID pairs after Equation 5 filtering; ID term will be zero.")

        id_player = (
            id_pair_filtered.groupby("defender_player_id", as_index=False)
            .agg(
                id_sum=("id_sum", "sum"),
                id_count=("id_count", "sum"),
                id_weighted_share=("id_weighted_share", "mean"),
            )
            .rename(columns={"defender_player_id": "player_id"})
        )
        team_players = team_players.merge(id_player, on="player_id", how="left")
        team_players[["id_sum", "id_count", "id_weighted_share"]] = team_players[
            ["id_sum", "id_count", "id_weighted_share"]
        ].fillna(0.0)
    else:
        if id_source == "phase4":
            team_players["id_sum"] = pd.to_numeric(team_players.get("id", 0.0), errors="coerce").fillna(0.0)
            team_players["id_count"] = pd.to_numeric(team_players.get("id_events", 0.0), errors="coerce").fillna(0.0)
            team_players["id_weighted_share"] = 1.0
        else:
            vaep_actions = pd.read_parquet(DATA_DIR / "vaep/vaep_actions.parquet")
            atomic_with_phase = pd.read_parquet(tactics_dir / "atomic_actions_with_phase.parquet")
            phase_summary = pd.read_parquet(tactics_dir / "phase_summary.parquet")
            event_vaep = _build_event_vaep_map(vaep_actions)
            id_pair_all = _compute_weighted_id(
                atomic_with_phase=atomic_with_phase,
                phase_summary=phase_summary,
                event_vaep=event_vaep,
                target_team_id=team_id,
                defense_weight=defense_weight,
                defense_clusters=set(defense_clusters),
                defense_keywords=defense_keywords,
            )
            id_pair_all["id_sum"] = pd.to_numeric(id_pair_all.get("id_sum", 0.0), errors="coerce").fillna(0.0)
            id_player = (
                id_pair_all.groupby("defender_player_id", as_index=False)
                .agg(
                    id_sum=("id_sum", "sum"),
                    id_count=("id_count", "sum"),
                    id_weighted_share=("id_weighted_share", "mean"),
                )
                .rename(columns={"defender_player_id": "player_id"})
            )
            team_players = team_players.merge(id_player, on="player_id", how="left")
            team_players[["id_sum", "id_count", "id_weighted_share"]] = team_players[
                ["id_sum", "id_count", "id_weighted_share"]
            ].fillna(0.0)

        id_pair_filtered = pd.DataFrame(
            columns=[
                "defender_player_id",
                "opponent_team_id",
                "opponent_player_id",
                "id_sum",
                "id_count",
                "id_weighted_share",
            ]
        )

    df_n, mf_n, fw_n = _parse_formation(formation)

    pos_need = {"GK": 1, "DF": df_n, "MF": mf_n, "FW": fw_n}
    for pos, need in pos_need.items():
        have = int((team_players["position"] == pos).sum())
        if have < need:
            raise RuntimeError(f"Not enough {pos} players for formation {formation}: need {need}, have {have}")

    player_ids = team_players["player_id"].astype(int).tolist()
    player_set = set(player_ids)

    io_pair = io_pair[io_pair["player_a_id"].isin(player_set) & io_pair["player_b_id"].isin(player_set)].copy()

    vi_map_raw = team_players.set_index("player_id")["vi"].to_dict()
    id_map_raw = team_players.set_index("player_id")["id_sum"].to_dict()
    pos_map = team_players.set_index("player_id")["position"].to_dict()

    vi_scale = _linear_scale_factor("vi_game", scaler_stats)
    io_scale = _linear_scale_factor("io_game", scaler_stats)
    id_scale = _linear_scale_factor("id_game", scaler_stats)

    vi_map = {int(pid): float(val) * vi_scale for pid, val in vi_map_raw.items()}
    id_map = {int(pid): float(val) * id_scale for pid, val in id_map_raw.items()}

    io_map_raw = {(int(r.player_a_id), int(r.player_b_id)): float(r.io_sum) for r in io_pair.itertuples(index=False)}
    io_map = {k: float(v) * io_scale for k, v in io_map_raw.items()}

    prob = pulp.LpProblem("LineupOptimization", pulp.LpMaximize)

    x = {pid: pulp.LpVariable(f"x_{pid}", lowBound=0, upBound=1, cat="Binary") for pid in player_ids}

    y = {}
    for a, b in io_map:
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in y:
            continue
        y[key] = pulp.LpVariable(f"y_{key[0]}_{key[1]}", lowBound=0, upBound=1, cat="Binary")

    prob += pulp.lpSum([x[pid] for pid in player_ids]) == 11

    for pos, need in pos_need.items():
        prob += pulp.lpSum([x[pid] for pid in player_ids if pos_map[pid] == pos]) == need

    for (a, b), y_var in y.items():
        prob += y_var <= x[a]
        prob += y_var <= x[b]
        prob += y_var >= x[a] + x[b] - 1

    obj_vi_id = pulp.lpSum([
        x[pid] * (lambda_vi * float(vi_map.get(pid, 0.0)) + lambda_id * float(id_map.get(pid, 0.0)))
        for pid in player_ids
    ])

    pair_coef = {}
    for (a, b), _ in y.items():
        coef = float(io_map.get((a, b), 0.0) + io_map.get((b, a), 0.0))
        pair_coef[(a, b)] = coef

    obj_io = pulp.lpSum([lambda_io * pair_coef[k] * y[k] for k in y.keys()])

    prob += obj_vi_id + obj_io

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    status = prob.solve(solver)
    status_name = pulp.LpStatus[status]
    
    # Accept both optimal and near-optimal (suboptimal) solutions from CBC solver
    # CBC returns "Optimal" for proven optimal, or "Not Solved" if timeout/interrupted with best feasible found
    if status_name not in {"Optimal", "Not Solved"}:
        # Only reject if problem is fundamentally infeasible or unbounded
        if status_name in {"Infeasible", "Unbounded"}:
            raise RuntimeError(f"ILP problem is {status_name}. Check formation constraints or synergy data.")
        else:
            raise RuntimeError(f"Unexpected solver status: {status_name}")
    
    if status_name == "Not Solved":
        print(f"[WARN] ILP solver hit time limit ({solver_time_limit}s). Using best feasible solution found.")

    selected = [pid for pid in player_ids if pulp.value(x[pid]) > 0.5]
    if len(selected) != 11:
        raise RuntimeError(
            f"No feasible XI extracted from solver output. status={status_name} selected={len(selected)}"
        )

    lineup = team_players[team_players["player_id"].isin(selected)].copy()
    lineup["selected"] = 1
    lineup["vi_component"] = lambda_vi * lineup["vi"]
    lineup["id_component"] = lambda_id * lineup["id_sum"]

    selected_pairs = []
    for (a, b), y_var in y.items():
        if pulp.value(y_var) > 0.5:
            selected_pairs.append({"player_a_id": a, "player_b_id": b, "io_pair_value": pair_coef[(a, b)]})

    id_pair_selected_df = id_pair_filtered[id_pair_filtered["defender_player_id"].isin(selected)].copy()

    lineup_name = lineup[["player_id", "player_name"]].drop_duplicates()
    selected_pairs_df = pd.DataFrame(selected_pairs)
    if not selected_pairs_df.empty:
        selected_pairs_df = selected_pairs_df.merge(
            lineup_name.rename(columns={"player_id": "player_a_id", "player_name": "player_a_name"}),
            on="player_a_id",
            how="left",
        )
        selected_pairs_df = selected_pairs_df.merge(
            lineup_name.rename(columns={"player_id": "player_b_id", "player_name": "player_b_name"}),
            on="player_b_id",
            how="left",
        )

    if not id_pair_selected_df.empty:
        id_pair_selected_df = id_pair_selected_df.merge(
            lineup_name.rename(columns={"player_id": "defender_player_id", "player_name": "defender_player_name"}),
            on="defender_player_id",
            how="left",
        )
        id_pair_selected_df = id_pair_selected_df.merge(
            players_meta[["player_id", "player_name"]].rename(
                columns={"player_id": "opponent_player_id", "player_name": "opponent_player_name"}
            ),
            on="opponent_player_id",
            how="left",
        )

    realized_vi = float((lambda_vi * lineup["vi"]).sum())
    realized_id = float((lambda_id * lineup["id_sum"]).sum())
    realized_io = float((lambda_io * selected_pairs_df.get("io_pair_value", pd.Series(dtype=float))).sum()) if not selected_pairs_df.empty else 0.0
    realized_total = realized_vi + realized_io + realized_id

    lineup = lineup.sort_values(["position", "v_total"], ascending=[True, False]).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "team_id": team_id,
                "formation": formation,
                "lambda_vi": lambda_vi,
                "lambda_io": lambda_io,
                "lambda_id": lambda_id,
                "attack_weight": attack_weight,
                "defense_weight": defense_weight,
                "attack_clusters": ",".join(map(str, attack_clusters)) if attack_clusters else "",
                "defense_clusters": ",".join(map(str, defense_clusters)) if defense_clusters else "",
                "attack_keywords": ",".join(attack_keywords) if attack_keywords else "",
                "defense_keywords": ",".join(defense_keywords) if defense_keywords else "",
                "io_source": io_source,
                "id_source": id_source,
                "id_mode": "equation5_vs_target_opponent_starting11" if use_equation5_id else "legacy_player_aggregated",
                "opponent_team_id": opponent_team_id if opponent_team_id is not None else "",
                "opponent_lineup_size": len(opponent_lineup_ids),
                "opponent_lineup_ids": "|".join(map(str, opponent_lineup_ids)),
                "available_player_count": len(available_player_ids),
                "available_player_ids": "|".join(map(str, available_player_ids)),
                "id_pairs_used": int(len(id_pair_filtered)),
                "lambda_csv": str(lambda_csv) if lambda_csv is not None else "",
                "lambda_scaler_stats": str(lambda_scaler_stats) if lambda_scaler_stats is not None else "",
                "objective_uses_scaled_features": int(scaler_stats is not None),
                "objective_scaling_mode": "linear_factor_only" if scaler_stats is not None else "none",
                "objective_vi_scale_factor": vi_scale,
                "objective_io_scale_factor": io_scale,
                "objective_id_scale_factor": id_scale,
                "objective_vi": realized_vi,
                "objective_io": realized_io,
                "objective_id": realized_id,
                "objective_total": realized_total,
                "solver_status": status_name,
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    lineup.to_csv(output_dir / "lineup_selected.csv", index=False)
    lineup.to_parquet(output_dir / "lineup_selected.parquet", index=False)
    named_cols = [c for c in ["player_id", "player_name", "position", "selected", "vi", "io", "id_sum", "vi_component", "id_component"] if c in lineup.columns]
    lineup[named_cols].to_csv(output_dir / "lineup_selected_named.csv", index=False)
    selected_pairs_df.to_csv(output_dir / "lineup_selected_io_pairs.csv", index=False)
    selected_pairs_df.to_parquet(output_dir / "lineup_selected_io_pairs.parquet", index=False)
    id_pair_selected_df.to_csv(output_dir / "lineup_selected_id_pairs_vs_opponent.csv", index=False)
    id_pair_selected_df.to_parquet(output_dir / "lineup_selected_id_pairs_vs_opponent.parquet", index=False)
    summary.to_csv(output_dir / "lineup_optimization_summary.csv", index=False)
    summary.to_parquet(output_dir / "lineup_optimization_summary.parquet", index=False)

    print(f"[OK] team_id={team_id}, formation={formation}, status={status_name}")
    print(f"[OK] objective_total={realized_total:.6f} (V_I={realized_vi:.6f}, I_O={realized_io:.6f}, I_D={realized_id:.6f})")
    print(f"[OK] saved: {output_dir / 'lineup_selected.parquet'}")
# 기능: main는 lambda_vi/lambda_io/lambda_id 선형 결합 목적함수를 사용한다을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase5 Team-Builder ILP 단계에서 목적함수 V=lambda1*V_I + lambda2*I_O + lambda3*I_D를 구성/최적화하기 위해 필요하다. 특히 lambda_vi/lambda_io/lambda_id 선형 결합 목적함수를 사용한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: None
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: ILP lineup optimization with tactic-weighted synergy")
    parser.add_argument(
        "--synergy-dir",
        type=Path,
        default=DATA_DIR / "synergy",
        help="Directory containing Phase 4 outputs",
    )
    parser.add_argument(
        "--tactics-dir",
        type=Path,
        default=DATA_DIR / "tactics",
        help="Directory containing Phase 3 outputs",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DATA_DIR / "archive",
        help="Directory containing players.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "lineup",
        help="Output directory for lineup optimization results",
    )
    parser.add_argument("--team-id", type=int, default=None, help="Target team id. If omitted, uses top team in synergy file")
    parser.add_argument("--formation", type=str, default="4-3-3", help="DF-MF-FW, GK is fixed to 1")

    parser.add_argument("--lambda-vi", type=float, default=1.0)
    parser.add_argument("--lambda-io", type=float, default=1.0)
    parser.add_argument("--lambda-id", type=float, default=1.0)

    parser.add_argument("--attack-weight", type=float, default=1.0, help="w_k multiplier for targeted attack tactics")
    parser.add_argument("--defense-weight", type=float, default=1.0, help="w_k multiplier for targeted defense tactics")
    parser.add_argument("--attack-clusters", nargs="*", type=int, default=[])
    parser.add_argument("--defense-clusters", nargs="*", type=int, default=[])
    parser.add_argument("--attack-keywords", nargs="*", default=[], help="substring match on attack zone sequence")
    parser.add_argument("--defense-keywords", nargs="*", default=[], help="substring match on defense zone sequence")

    parser.add_argument("--solver-time-limit", type=int, default=120)
    parser.add_argument(
        "--io-source",
        type=str,
        choices=["phase4", "recompute"],
        default="phase4",
        help="Use IO from Phase4 outputs (recommended) or recompute in Phase5",
    )
    parser.add_argument(
        "--id-source",
        type=str,
        choices=["phase4", "recompute"],
        default="phase4",
        help="Use ID from Phase4 outputs (recommended for consistency) or recompute in Phase5",
    )
    parser.add_argument(
        "--lambda-csv",
        type=Path,
        default=None,
        help="Optional CSV containing lambda_vi, lambda_io, lambda_id (e.g. phase6 reestimated lambda file)",
    )
    parser.add_argument(
        "--lambda-scaler-stats",
        type=Path,
        default=None,
        help="Optional scaler stats (.json/.csv) from Phase4 lambda estimation; if provided, vi/io/id are transformed before objective",
    )
    parser.add_argument(
        "--opponent-lineup-ids",
        nargs=11,
        type=int,
        required=False,
        default=None,
        help="Optional target match opponent starting 11 player IDs for Equation 5 I_D filtering",
    )
    parser.add_argument(
        "--opponent-team-id",
        type=int,
        default=None,
        help="Optional opponent team id for stricter Equation 5 I_D filtering",
    )
    parser.add_argument(
        "--available-player-ids",
        nargs="*",
        type=int,
        required=False,
        default=None,
        help="Optional available player ids (e.g., matchday squad). If provided, ILP can select only from this set.",
    )

    args = parser.parse_args()

    run_phase5(
        synergy_dir=args.synergy_dir,
        tactics_dir=args.tactics_dir,
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        team_id=args.team_id,
        formation=args.formation,
        lambda_vi=args.lambda_vi,
        lambda_io=args.lambda_io,
        lambda_id=args.lambda_id,
        attack_weight=args.attack_weight,
        defense_weight=args.defense_weight,
        attack_clusters=args.attack_clusters,
        defense_clusters=args.defense_clusters,
        attack_keywords=args.attack_keywords,
        defense_keywords=args.defense_keywords,
        solver_time_limit=args.solver_time_limit,
        io_source=args.io_source,
        id_source=args.id_source,
        lambda_csv=args.lambda_csv,
        lambda_scaler_stats=args.lambda_scaler_stats,
        opponent_lineup_ids=args.opponent_lineup_ids,
        opponent_team_id=args.opponent_team_id,
        available_player_ids=args.available_player_ids,
    )


if __name__ == "__main__":
    main()
