#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 4.5: Build time-aware GNN training dataset with rolling-window leakage control.

This script creates one HeteroData graph per target match.
Core ideas:
1) For match N, only events from matches strictly BEFORE match N timestamp are used.
2) Spatial signals are compressed to asymmetric 12 zones (not uniform grids).
3) Node features: Off + Def 12D vectors per player (24D concat).
4) Edge features:
   - IO (same-team directed): 12D vectors
   - ID (cross-team directed): 12D vectors

Output:
- torch serialized list[HeteroData] at --output-pt
- optional metadata CSV at --output-meta-csv
Note:
- No feature scaling is applied here. Train-fold-only scaler fitting is done in Phase 5 training.
"""

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import HeteroData
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required. Install with: pip install torch-geometric"
    ) from exc

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"


# ----------------------------
# Zone mapping (asymmetric 12)
# ----------------------------
# 기능: 피치 좌표 x(0~105), y(0~68)를 임계값 x=[26.25, 52.5, 78.75], y 분할=[3,1,3,5] 규칙으로 0~11 전술 존 인덱스로 변환한다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 x 경계값 26.25/52.5/78.75와 y 구간 분할식을 사용한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: x: float, y: float
#   - Output: int
def map_to_12_zones(x: float, y: float) -> int:
    """Map pitch coordinate (x,y) to asymmetric 12 tactical zones.

    Pitch definition:
    - x: 0..105 (length)
    - y: 0..68  (width)

    X segments:
    1) [0, 26.25): split Y into 3 zones -> 0,1,2
    2) [26.25, 52.5): full-width single zone -> 3
    3) [52.5, 78.75): split Y into 3 zones -> 4,5,6
    4) [78.75, 105]: split Y into 5 zones -> 7,8,9,10,11
    """
    xx = float(np.clip(x, 0.0, 105.0))
    yy = float(np.clip(y, 0.0, 68.0))

    if xx < 26.25:
        b = 68.0 / 3.0
        return 0 if yy < b else (1 if yy < 2.0 * b else 2)

    if xx < 52.5:
        return 3

    if xx < 78.75:
        b = 68.0 / 3.0
        return 4 if yy < b else (5 if yy < 2.0 * b else 6)

    b = 68.0 / 5.0
    if yy < 1.0 * b:
        return 7
    if yy < 2.0 * b:
        return 8
    if yy < 3.0 * b:
        return 9
    if yy < 4.0 * b:
        return 10
    return 11
# 기능: _accumulate_zone_vector는 컬럼 'x', 'y'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: events: pd.DataFrame, value_col: str
#   - Output: np.ndarray
def _accumulate_zone_vector(events: pd.DataFrame, value_col: str = "value") -> np.ndarray:
    """Accumulate event values into a 12D zone vector."""
    vec = np.zeros(12, dtype=np.float32)
    if events.empty:
        return vec

    x = pd.to_numeric(events["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(events["y"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(events[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x = x[valid]
    y = y[valid]
    v = v[valid]

    for xx, yy, vv in zip(x, y, v):
        z = map_to_12_zones(float(xx), float(yy))
        vec[z] += float(vv)
    return vec
# 기능: _safe_density_divide는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: vec: np.ndarray, exposure_90: float
#   - Output: np.ndarray
def _safe_density_divide(vec: np.ndarray, exposure_90: float) -> np.ndarray:
    """Convert summed vector into density vector using per-90 exposure."""
    denom = float(max(exposure_90, 1e-6))
    return (vec / denom).astype(np.float32)
# 기능: OFF/DEF 이벤트의 ['player_id','game_id'] 고유 경기 수를 이용해 분모(exposure_90)를 계산하고 밀도 정규화용 맵을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: off_events: pd.DataFrame, def_events: pd.DataFrame
#   - Output: Dict[int, float]
def _build_player_exposure90(off_events: pd.DataFrame, def_events: pd.DataFrame) -> Dict[int, float]:
    """Build exposure map per player.

    Preferred denominator is cumulative minutes / 90.
    If minutes are unavailable, fallback to cumulative distinct matches played.
    """
    frames = []
    if not off_events.empty:
        frames.append(off_events[["player_id", "game_id"]].copy())
    if not def_events.empty:
        frames.append(def_events[["player_id", "game_id"]].copy())

    if not frames:
        return {}

    base = pd.concat(frames, axis=0, ignore_index=True).dropna(subset=["player_id", "game_id"]).copy()
    if base.empty:
        return {}

    base["player_id"] = pd.to_numeric(base["player_id"], errors="coerce")
    base["game_id"] = pd.to_numeric(base["game_id"], errors="coerce")
    base = base.dropna(subset=["player_id", "game_id"]).copy()
    if base.empty:
        return {}

    # Fallback exposure: number of distinct historical matches (equivalent to minutes/90 when minutes are missing).
    games = base.groupby("player_id")["game_id"].nunique().astype(float)
    return {int(pid): float(max(n_games, 1.0)) for pid, n_games in games.items()}


# ----------------------------
# Match / lineup utilities
# ----------------------------
# 기능: _to_int는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: v: object
#   - Output: int | None
def _to_int(v: object) -> int | None:
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except Exception:
        return None
# 기능: _safe_eval_list는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: v: object
#   - Output: list
def _safe_eval_list(v: object) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return []
        try:
            obj = ast.literal_eval(t)
        except Exception:
            return []
        return obj if isinstance(obj, list) else []
    return []
# 기능: _parse_lineup_ids는 컬럼 'playerId'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: lineup_raw: object, max_players: int
#   - Output: List[int]
def _parse_lineup_ids(lineup_raw: object, max_players: int = 11) -> List[int]:
    lineup = _safe_eval_list(lineup_raw)
    ids: List[int] = []
    for row in lineup:
        if isinstance(row, dict):
            pid = _to_int(row.get("playerId"))
        else:
            pid = _to_int(row)
        if pid is None:
            continue
        if pid not in ids:
            ids.append(pid)
        if len(ids) >= max_players:
            break
    return ids
# 기능: _require_match_row는 컬럼 'wyId'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_df: pd.DataFrame, match_id: int
#   - Output: pd.Series
def _require_match_row(matches_df: pd.DataFrame, match_id: int) -> pd.Series:
    m = matches_df[pd.to_numeric(matches_df["wyId"], errors="coerce") == int(match_id)]
    if m.empty:
        raise ValueError(f"match_id not found in matches csv: {match_id}")
    return m.iloc[0]


# ----------------------------
# Standardize event tables
# ----------------------------

@dataclass
class EventTables:
    off: pd.DataFrame
    deff: pd.DataFrame
    io: pd.DataFrame
    idd: pd.DataFrame
# 기능: _pick_best_parquet_by_overlap는 연산 pd.read_parquet을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: candidates: List[Path], match_ids: Set[int], key_col: str
#   - Output: Path
def _pick_best_parquet_by_overlap(candidates: List[Path], match_ids: Set[int], key_col: str = "game_id") -> Path:
    """Pick parquet file with maximum game_id overlap against target match ids."""
    best_path: Path | None = None
    best_score = -1

    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=[key_col])
            gids = set(pd.to_numeric(df[key_col], errors="coerce").dropna().astype(int).tolist())
            score = len(gids & match_ids)
            if score > best_score:
                best_score = score
                best_path = p
        except Exception:
            continue

    if best_path is None:
        raise FileNotFoundError(f"No readable parquet among candidates: {candidates}")
    return best_path
# 기능: _pick_first_existing_path는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: candidates: List[Path]
#   - Output: Path
def _pick_first_existing_path(candidates: List[Path]) -> Path:
    """Return the first existing path from a candidate list."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No existing file among candidates: {candidates}")
# 기능: VAEP/IO/ID 후보 경로(예: vaep_actions_england_eval.parquet, synergy_england)를 dataset_hint와 match_ids 기준으로 선택해 OFF/DEF/IO/ID 테이블을 표준 스키마로 맞춘다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: data_root: Path, match_ids: Set[int], dataset_hint: str
#   - Output: EventTables
def _build_event_tables(data_root: Path, match_ids: Set[int], dataset_hint: str = "") -> EventTables:
    """Load and standardize OFF/DEF/IO/ID event tables.

    OFF/DEF source:
    - data/vaep/vaep_actions.parquet for non-England
    - data/vaep/vaep_actions_england_eval.parquet for England eval

    IO source:
    - data/synergy/io_event_surfaces_base.parquet for non-England
    - data/synergy_england/io_event_surfaces_base.parquet for England eval

    ID source:
    - data/synergy/id_event_surfaces_base.parquet for non-England
    - data/synergy_england/id_event_surfaces_base.parquet for England eval
    """
    hint = dataset_hint.lower()
    non_eng_vaep_path = data_root / "vaep/vaep_actions.parquet"
    eng_eval_vaep_path = data_root / "vaep/vaep_actions_england_eval.parquet"
    non_eng_io_candidates = [
        data_root / "synergy/io_event_surfaces_base.parquet",
        data_root / "synergy_ilp_unified_non_england/io_event_surfaces_base.parquet",
    ]
    non_eng_id_candidates = [
        data_root / "synergy/id_event_surfaces_base.parquet",
        data_root / "synergy_ilp_unified_non_england/id_event_surfaces_base.parquet",
    ]
    eng_io_candidates = [
        data_root / "synergy_england/io_event_surfaces_base.parquet",
        data_root / "synergy_ioid_england_eval_preproc_all/io_event_surfaces_base.parquet",
    ]
    eng_id_candidates = [
        data_root / "synergy_england/id_event_surfaces_base.parquet",
        data_root / "synergy_ioid_england_eval_preproc_all/id_event_surfaces_base.parquet",
    ]
    if "non_england" in hint:
        vaep_path = non_eng_vaep_path
        io_path = _pick_first_existing_path(non_eng_io_candidates)
        id_path = _pick_first_existing_path(non_eng_id_candidates)
    elif "england_eval" in hint:
        vaep_path = _pick_first_existing_path([eng_eval_vaep_path, non_eng_vaep_path])
        io_path = _pick_first_existing_path(eng_io_candidates)
        id_path = _pick_first_existing_path(eng_id_candidates)
    else:
        vaep_path = _pick_best_parquet_by_overlap(
            [
                non_eng_vaep_path,
                eng_eval_vaep_path,
            ],
            match_ids=match_ids,
            key_col="game_id",
        )
        io_path = _pick_best_parquet_by_overlap(
            non_eng_io_candidates + eng_io_candidates,
            match_ids=match_ids,
            key_col="game_id",
        )
        id_path = _pick_best_parquet_by_overlap(
            non_eng_id_candidates + eng_id_candidates,
            match_ids=match_ids,
            key_col="game_id",
        )

    print(f"[INFO] selected VAEP: {vaep_path}")
    print(f"[INFO] selected IO:   {io_path}")
    print(f"[INFO] selected ID:   {id_path}")

    vaep = pd.read_parquet(
        vaep_path,
        columns=["game_id", "team_id", "player_id", "start_x", "start_y", "offensive_value", "defensive_value"],
    ).rename(columns={"start_x": "x", "start_y": "y"})

    off = vaep[["game_id", "team_id", "player_id", "x", "y", "offensive_value"]].rename(
        columns={"offensive_value": "value"}
    )
    deff = vaep[["game_id", "team_id", "player_id", "x", "y", "defensive_value"]].rename(
        columns={"defensive_value": "value"}
    )

    io_raw = pd.read_parquet(io_path)
    needed_io = {
        "game_id",
        "team_id",
        "actor_player_id",
        "receiver_player_id",
        "contribution_source",
        "x",
        "y",
        "io_event_weighted",
    }
    if not needed_io.issubset(io_raw.columns):
        raise ValueError(f"io_event_surfaces_base missing columns: {sorted(needed_io - set(io_raw.columns))}")

    io = io_raw[list(needed_io)].copy()
    io["contribution_source"] = io["contribution_source"].astype(str).str.lower()

    # Directed pair mapping rule:
    # - first_action  -> src=actor, dst=receiver
    # - second_action -> src=receiver, dst=actor
    is_second = io["contribution_source"].eq("second_action")
    io["src_player_id"] = np.where(is_second, io["receiver_player_id"], io["actor_player_id"])
    io["dst_player_id"] = np.where(is_second, io["actor_player_id"], io["receiver_player_id"])
    io = io.rename(columns={"io_event_weighted": "value"})
    io = io[["game_id", "team_id", "src_player_id", "dst_player_id", "x", "y", "value"]]

    id_raw = pd.read_parquet(id_path)
    needed_id = {
        "game_id",
        "defending_team_id",
        "opponent_team_id",
        "defender_player_id",
        "opponent_player_id",
        "x",
        "y",
        "id_event_weighted",
    }
    if not needed_id.issubset(id_raw.columns):
        raise ValueError(f"id_event_surfaces_base missing columns: {sorted(needed_id - set(id_raw.columns))}")

    idd = id_raw[list(needed_id)].rename(columns={"id_event_weighted": "value"}).copy()

    # Numeric sanitize
    for col in ["game_id", "team_id", "player_id", "x", "y", "value"]:
        if col in off.columns:
            off[col] = pd.to_numeric(off[col], errors="coerce")
    for col in ["game_id", "team_id", "player_id", "x", "y", "value"]:
        if col in deff.columns:
            deff[col] = pd.to_numeric(deff[col], errors="coerce")

    for col in ["game_id", "team_id", "src_player_id", "dst_player_id", "x", "y", "value"]:
        io[col] = pd.to_numeric(io[col], errors="coerce")

    for col in [
        "game_id",
        "defending_team_id",
        "opponent_team_id",
        "defender_player_id",
        "opponent_player_id",
        "x",
        "y",
        "value",
    ]:
        idd[col] = pd.to_numeric(idd[col], errors="coerce")

    off = off.dropna(subset=["game_id", "team_id", "player_id", "x", "y"]).copy()
    deff = deff.dropna(subset=["game_id", "team_id", "player_id", "x", "y"]).copy()
    io = io.dropna(subset=["game_id", "team_id", "src_player_id", "dst_player_id", "x", "y"]).copy()
    idd = idd.dropna(
        subset=["game_id", "defending_team_id", "opponent_team_id", "defender_player_id", "opponent_player_id", "x", "y"]
    ).copy()

    for col in ["game_id", "team_id", "player_id"]:
        if col in off.columns:
            off[col] = off[col].astype(int)
        if col in deff.columns:
            deff[col] = deff[col].astype(int)
    for col in ["game_id", "team_id", "src_player_id", "dst_player_id"]:
        io[col] = io[col].astype(int)
    for col in ["game_id", "defending_team_id", "opponent_team_id", "defender_player_id", "opponent_player_id"]:
        idd[col] = idd[col].astype(int)

    return EventTables(off=off, deff=deff, io=io, idd=idd)


# ----------------------------
# HeteroData builder
# ----------------------------
# 기능: _build_node_feature_matrix는 컬럼 'player_id'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: player_ids: List[int], off_events: pd.DataFrame, def_events: pd.DataFrame, exposure90_map: Dict[int, float]
#   - Output: np.ndarray
def _build_node_feature_matrix(
    player_ids: List[int],
    off_events: pd.DataFrame,
    def_events: pd.DataFrame,
    exposure90_map: Dict[int, float],
) -> np.ndarray:
    """Return node feature matrix [n_players, 24] = concat(off_12, def_12)."""
    feats: List[np.ndarray] = []
    for pid in player_ids:
        off_vec = _accumulate_zone_vector(off_events[off_events["player_id"] == int(pid)], value_col="value")
        def_vec = _accumulate_zone_vector(def_events[def_events["player_id"] == int(pid)], value_col="value")
        exposure90 = float(exposure90_map.get(int(pid), 1.0))
        node_vec = np.concatenate([off_vec, def_vec], axis=0)
        feats.append(_safe_density_divide(node_vec, exposure90))
    if not feats:
        return np.zeros((0, 24), dtype=np.float32)
    return np.stack(feats, axis=0).astype(np.float32)
# 기능: _build_same_team_io_edges는 컬럼 'team_id', 'src_player_id', 'dst_player_id'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: team_id: int, player_ids: List[int], io_events: pd.DataFrame, exposure90_map: Dict[int, float]
#   - Output: Tuple[np.ndarray, np.ndarray]
def _build_same_team_io_edges(
    team_id: int,
    player_ids: List[int],
    io_events: pd.DataFrame,
    exposure90_map: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build directed same-team IO edges and 12D edge_attr.

    Even if sparse, we keep full directed complete graph without self-loops
    so GAT can attend over a consistent structure.
    """
    pid_to_idx = {int(pid): i for i, pid in enumerate(player_ids)}
    edges: List[Tuple[int, int]] = []
    attrs: List[np.ndarray] = []

    team_io = io_events[pd.to_numeric(io_events["team_id"], errors="coerce") == int(team_id)].copy()

    for src_pid in player_ids:
        for dst_pid in player_ids:
            if int(src_pid) == int(dst_pid):
                continue
            e = team_io[
                (team_io["src_player_id"] == int(src_pid))
                & (team_io["dst_player_id"] == int(dst_pid))
            ]
            vec = _accumulate_zone_vector(e, value_col="value")
            edge_exposure90 = 0.5 * (
                float(exposure90_map.get(int(src_pid), 1.0))
                + float(exposure90_map.get(int(dst_pid), 1.0))
            )
            vec = _safe_density_divide(vec, edge_exposure90)
            edges.append((pid_to_idx[int(src_pid)], pid_to_idx[int(dst_pid)]))
            attrs.append(vec)

    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(attrs, axis=0).astype(np.float32) if attrs else np.zeros((0, 12), dtype=np.float32)
    return edge_index, edge_attr
# 기능: _build_cross_id_edges는 컬럼 'defending_team_id', 'opponent_team_id', 'defender_player_id', 'opponent_player_id'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다.
# 데이터 입출력:
#   - Input: def_team_id: int, def_player_ids: List[int], off_team_id: int, off_player_ids: List[int], id_events: pd.DataFrame, def_exposure90_map: Dict[int, float], ...
#   - Output: Tuple[np.ndarray, np.ndarray]
def _build_cross_id_edges(
    def_team_id: int,
    def_player_ids: List[int],
    off_team_id: int,
    off_player_ids: List[int],
    id_events: pd.DataFrame,
    def_exposure90_map: Dict[int, float],
    off_exposure90_map: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build directed cross-team ID edges and 12D edge_attr.

    Requirement: keep structural edges even when no historical matchup exists.
    We therefore create full bipartite edges with zero vectors for missing pairs.
    """
    def_idx = {int(pid): i for i, pid in enumerate(def_player_ids)}
    off_idx = {int(pid): i for i, pid in enumerate(off_player_ids)}

    edges: List[Tuple[int, int]] = []
    attrs: List[np.ndarray] = []

    use = id_events[
        (id_events["defending_team_id"] == int(def_team_id))
        & (id_events["opponent_team_id"] == int(off_team_id))
    ]

    for dpid in def_player_ids:
        for opid in off_player_ids:
            e = use[
                (use["defender_player_id"] == int(dpid))
                & (use["opponent_player_id"] == int(opid))
            ]
            vec = _accumulate_zone_vector(e, value_col="value")
            edge_exposure90 = 0.5 * (
                float(def_exposure90_map.get(int(dpid), 1.0))
                + float(off_exposure90_map.get(int(opid), 1.0))
            )
            vec = _safe_density_divide(vec, edge_exposure90)
            edges.append((def_idx[int(dpid)], off_idx[int(opid)]))
            attrs.append(vec)

    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(attrs, axis=0).astype(np.float32) if attrs else np.zeros((0, 12), dtype=np.float32)
    return edge_index, edge_attr
# 기능: home/away 11인 기준 node(24D=off_12+def_12)와 edge(IO/ID 12D) 텐서를 구성하고 match_y/global_features를 포함한 HeteroData를 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: match_id: int, past_events_df: Dict[str, pd.DataFrame], matches_df: pd.DataFrame, players_per_team: int
#   - Output: HeteroData
def create_match_heterodata(
    match_id: int,
    past_events_df: Dict[str, pd.DataFrame],
    matches_df: pd.DataFrame,
    players_per_team: int = 11,
) -> HeteroData:
    """Create one HeteroData graph for a target match.

    Parameters
    - match_id: target match id
    - past_events_df: dict with keys ['off','def','io','id']
      containing ONLY historical rows before target match timestamp
    - matches_df: match table that contains team ids and lineups

    Node types
    - home_team: 11 player nodes
    - away_team: 11 player nodes

    Node features
    - 24D per node = [off_12, def_12]

    Edge types
    - (home_team, passes_to, home_team): directed IO edges, edge_attr 12D
    - (away_team, passes_to, away_team): directed IO edges, edge_attr 12D
    - (home_team, defends_against, away_team): directed ID edges, edge_attr 12D
    - (away_team, defends_against, home_team): directed ID edges, edge_attr 12D
    """
    row = _require_match_row(matches_df, int(match_id))

    home_team_id = _to_int(row.get("team1.teamId"))
    away_team_id = _to_int(row.get("team2.teamId"))
    if home_team_id is None or away_team_id is None:
        raise ValueError(f"Invalid team ids in match row: {match_id}")

    home_lineup = _parse_lineup_ids(row.get("team1.formation.lineup"), max_players=players_per_team)
    away_lineup = _parse_lineup_ids(row.get("team2.formation.lineup"), max_players=players_per_team)
    if len(home_lineup) < players_per_team or len(away_lineup) < players_per_team:
        raise ValueError(f"Lineup parsing failed for match {match_id}: home={len(home_lineup)}, away={len(away_lineup)}")

    off_hist = past_events_df["off"]
    def_hist = past_events_df["def"]
    io_hist = past_events_df["io"]
    id_hist = past_events_df["id"]

    # For node OFF/DEF signals, aggregate by player_id from all historical events.
    # Team-id strict filtering can zero out nodes when team id systems differ across tables.
    home_off = off_hist[off_hist["player_id"].isin(home_lineup)]
    home_def = def_hist[def_hist["player_id"].isin(home_lineup)]
    away_off = off_hist[off_hist["player_id"].isin(away_lineup)]
    away_def = def_hist[def_hist["player_id"].isin(away_lineup)]

    home_exposure90 = _build_player_exposure90(home_off, home_def)
    away_exposure90 = _build_player_exposure90(away_off, away_def)

    x_home = _build_node_feature_matrix(home_lineup, home_off, home_def, home_exposure90)
    x_away = _build_node_feature_matrix(away_lineup, away_off, away_def, away_exposure90)

    hh_edge_index, hh_edge_attr = _build_same_team_io_edges(
        int(home_team_id),
        home_lineup,
        io_hist,
        home_exposure90,
    )
    aa_edge_index, aa_edge_attr = _build_same_team_io_edges(
        int(away_team_id),
        away_lineup,
        io_hist,
        away_exposure90,
    )

    ha_edge_index, ha_edge_attr = _build_cross_id_edges(
        def_team_id=int(home_team_id),
        def_player_ids=home_lineup,
        off_team_id=int(away_team_id),
        off_player_ids=away_lineup,
        id_events=id_hist,
        def_exposure90_map=home_exposure90,
        off_exposure90_map=away_exposure90,
    )
    ah_edge_index, ah_edge_attr = _build_cross_id_edges(
        def_team_id=int(away_team_id),
        def_player_ids=away_lineup,
        off_team_id=int(home_team_id),
        off_player_ids=home_lineup,
        id_events=id_hist,
        def_exposure90_map=away_exposure90,
        off_exposure90_map=home_exposure90,
    )

    data = HeteroData()
    data["home_team"].x = torch.tensor(x_home, dtype=torch.float32)
    data["away_team"].x = torch.tensor(x_away, dtype=torch.float32)

    data[("home_team", "passes_to", "home_team")].edge_index = torch.tensor(hh_edge_index, dtype=torch.long)
    data[("home_team", "passes_to", "home_team")].edge_attr = torch.tensor(hh_edge_attr, dtype=torch.float32)

    data[("away_team", "passes_to", "away_team")].edge_index = torch.tensor(aa_edge_index, dtype=torch.long)
    data[("away_team", "passes_to", "away_team")].edge_attr = torch.tensor(aa_edge_attr, dtype=torch.float32)

    data[("home_team", "defends_against", "away_team")].edge_index = torch.tensor(ha_edge_index, dtype=torch.long)
    data[("home_team", "defends_against", "away_team")].edge_attr = torch.tensor(ha_edge_attr, dtype=torch.float32)

    data[("away_team", "defends_against", "home_team")].edge_index = torch.tensor(ah_edge_index, dtype=torch.long)
    data[("away_team", "defends_against", "home_team")].edge_attr = torch.tensor(ah_edge_attr, dtype=torch.float32)

    # Match-level labels (for training target)
    home_score = pd.to_numeric(row.get("team1.score"), errors="coerce")
    away_score = pd.to_numeric(row.get("team2.score"), errors="coerce")
    if np.isfinite(home_score) and np.isfinite(away_score):
        if home_score > away_score:
            y = 2  # Win
        elif home_score == away_score:
            y = 1  # Draw
        else:
            y = 0  # Loss
        data["match_y"] = torch.tensor([y], dtype=torch.long)

    data["match_id"] = torch.tensor([int(match_id)], dtype=torch.long)
    data["home_team_id"] = torch.tensor([int(home_team_id)], dtype=torch.long)
    data["away_team_id"] = torch.tensor([int(away_team_id)], dtype=torch.long)
    data["home_player_ids"] = torch.tensor(home_lineup, dtype=torch.long)
    data["away_player_ids"] = torch.tensor(away_lineup, dtype=torch.long)

    # Graph-level context features injected for downstream model (global_features).
    # Keep this leakage-safe: no post-match outcomes are used.
    match_time = row.get("match_time", pd.NaT)
    if pd.notna(match_time):
        ts = pd.Timestamp(match_time)
        month = float(ts.month)
        dow = float(ts.dayofweek)
        month_rad = 2.0 * np.pi * (month - 1.0) / 12.0
        is_weekend = 1.0 if int(dow) >= 5 else 0.0
        global_features = np.array([1.0, is_weekend, np.sin(month_rad), np.cos(month_rad)], dtype=np.float32)
    else:
        global_features = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    data["global_features"] = torch.tensor(global_features, dtype=torch.float32)

    return data


# ----------------------------
# Rolling window dataset build
# ----------------------------
# 기능: matches_df의 'dateutc'/'date'를 datetime으로 파싱하고 'wyId'를 정수화한 뒤 'match_time_unix'·'seq_index'를 붙여 시간순 정렬한다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_df: pd.DataFrame
#   - Output: pd.DataFrame
def _parse_match_time(matches_df: pd.DataFrame) -> pd.DataFrame:
    out = matches_df.copy()
    if "dateutc" in out.columns:
        out["match_time"] = pd.to_datetime(out["dateutc"], errors="coerce", utc=True)
    elif "date" in out.columns:
        out["match_time"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    else:
        raise ValueError("matches csv must include dateutc (or date) for rolling-window filtering")

    out["wyId"] = pd.to_numeric(out["wyId"], errors="coerce")
    out = out.dropna(subset=["wyId", "match_time"]).copy()
    out["wyId"] = out["wyId"].astype(int)
    out = out.sort_values(["match_time", "wyId"]).reset_index(drop=True)
    out["match_time_unix"] = (out["match_time"].astype("int64") // 1_000_000_000).astype(np.int64)
    out["seq_index"] = np.arange(len(out), dtype=np.int64)
    return out
# 기능: 각 타깃 경기마다 과거 경기 집합(past_id_set)을 만들고 OFF/DEF/IO/ID를 game_id 기준으로 잘라 rolling-window HeteroData를 생성한다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성; 엔티티 키(game_id/team_id/player_id) 일관성; 현재 경기보다 과거(match_time_unix < cur_unix) 조건만 남긴다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_df: pd.DataFrame, event_tables: EventTables, players_per_team: int
#   - Output: Tuple[List[HeteroData], pd.DataFrame]
def build_rolling_gnn_dataset(
    matches_df: pd.DataFrame,
    event_tables: EventTables,
    players_per_team: int,
) -> Tuple[List[HeteroData], pd.DataFrame]:
    """Build list[HeteroData] with strict time-aware rolling window.

    For each target match m at time t_m:
    - collect historical game_ids where match_time < t_m
    - filter OFF/DEF/IO/ID events by those historical game_ids only
    - build HeteroData for m from historical-only signals
    """
    graphs: List[HeteroData] = []
    meta_rows: List[dict] = []

    match_times = matches_df[["wyId", "match_time"]].copy()
    game_to_time = dict(zip(match_times["wyId"], match_times["match_time"]))

    for row in matches_df.itertuples(index=False):
        match_id = int(row.wyId)
        t_cur = game_to_time[match_id]
        cur_unix = int(row.match_time_unix)

        # Strict time leakage control:
        # use ONLY matches with strictly earlier unix timestamp.
        past_mask = matches_df["match_time_unix"] < cur_unix
        past_id_set = set(matches_df.loc[past_mask, "wyId"].astype(int).tolist())

        past = {
            "off": event_tables.off[event_tables.off["game_id"].isin(past_id_set)].copy(),
            "def": event_tables.deff[event_tables.deff["game_id"].isin(past_id_set)].copy(),
            "io": event_tables.io[event_tables.io["game_id"].isin(past_id_set)].copy(),
            "id": event_tables.idd[event_tables.idd["game_id"].isin(past_id_set)].copy(),
        }

        # Earliest matches may have too little history; still create graph with sparse/zero vectors.
        try:
            data = create_match_heterodata(
                match_id=int(match_id),
                past_events_df=past,
                matches_df=matches_df,
                players_per_team=int(players_per_team),
            )
            graphs.append(data)
            meta_rows.append(
                {
                    "match_id": int(match_id),
                    "match_time": str(t_cur),
                    "n_past_matches": int(len(past_id_set)),
                    "n_off_events": int(len(past["off"])),
                    "n_def_events": int(len(past["def"])),
                    "n_io_events": int(len(past["io"])),
                    "n_id_events": int(len(past["id"])),
                }
            )
        except Exception as exc:
            meta_rows.append(
                {
                    "match_id": int(match_id),
                    "match_time": str(t_cur),
                    "n_past_matches": int(len(past_id_set)),
                    "error": str(exc),
                }
            )

    return graphs, pd.DataFrame(meta_rows)
# 기능: main는 컬럼 'wyId', 연산 pd.read_csv/torch.save/to_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase4.5 그래프 생성에서 경기 시점 이전(match_time_unix < cur_unix) 데이터만 사용해 시계열 누수를 차단하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: None
def main() -> None:
    parser = argparse.ArgumentParser(description="Build rolling-window GNN dataset (Phase 4.5)")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=DATA_DIR / "archive/matches_non_england.csv",
        help="Training split matches csv (recommended non-England to avoid target leakage)",
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DATA_DIR / "phase_4_synergy/data/gnn_phase4_5/hetero_graphs_non_england.pt",
    )
    parser.add_argument(
        "--output-meta-csv",
        type=Path,
        default=DATA_DIR / "phase_4_synergy/data/gnn_phase4_5/hetero_graphs_non_england_meta.csv",
    )
    parser.add_argument(
        "--output-scaler-pt",
        type=Path,
        default=None,
        help="Deprecated. Scaler is now fitted on train fold in Phase 5 and stored with model checkpoint.",
    )
    parser.add_argument("--players-per-team", type=int, default=11)
    parser.add_argument("--max-matches", type=int, default=0, help="Use only first N matches after time sort (0=all)")
    args = parser.parse_args()

    matches = pd.read_csv(args.matches_csv)
    matches = _parse_match_time(matches)
    if int(args.max_matches) > 0:
        matches = matches.head(int(args.max_matches)).copy()

    match_ids = set(matches["wyId"].astype(int).tolist())
    event_tables = _build_event_tables(args.data_root, match_ids=match_ids, dataset_hint=str(args.matches_csv))

    graphs, meta = build_rolling_gnn_dataset(
        matches_df=matches,
        event_tables=event_tables,
        players_per_team=int(args.players_per_team),
    )

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta_csv.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, args.output_pt)
    meta.to_csv(args.output_meta_csv, index=False)

    print(f"[OK] saved graphs: {args.output_pt}")
    print(f"[OK] saved meta:   {args.output_meta_csv}")
    if args.output_scaler_pt is not None:
        print(
            "[WARN] --output-scaler-pt is deprecated and ignored. "
            "Scaler fitting now happens in Phase 5 train-fold preprocessing."
        )
    print(f"[OK] n_graphs:     {len(graphs)}")
    if not meta.empty:
        print(meta.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
