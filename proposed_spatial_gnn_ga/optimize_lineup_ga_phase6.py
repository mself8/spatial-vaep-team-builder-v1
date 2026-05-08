#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 6: GA lineup optimization with cached hetero features.

Goal:
- Maximize expected points E[Pts] vs a fixed away lineup using trained GNN.
- Avoid rebuilding full historical tables inside GA loop by caching tensors.

Pipeline:
1) Load trained model checkpoint.
2) Build squad pools for home/away (15~20 players by recent usage).
3) Precompute and cache OFF/DEF/IO/ID 12D tensors in memory.
4) Run GA for home starting XI (11 unique players).
5) Print best XI with P(win/draw/loss) and max expected points.
"""

import argparse
import ast
import importlib.util
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import global_mean_pool

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils import _safe_literal, _to_int  # noqa: E402

EPS = 1e-6


# ----------------------------
# Spatial mapping
# ----------------------------
# 기능: 피치 좌표 x(0~105), y(0~68)를 임계값 x=[26.25, 52.5, 78.75], y 분할=[3,1,3,5] 규칙으로 0~11 전술 존 인덱스로 변환한다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 x 경계값 26.25/52.5/78.75와 y 구간 분할식을 사용한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: x: float, y: float
#   - Output: int
def map_to_12_zones(x: float, y: float) -> int:
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
    if yy < b:
        return 7
    if yy < 2.0 * b:
        return 8
    if yy < 3.0 * b:
        return 9
    if yy < 4.0 * b:
        return 10
    return 11
# 기능: _accumulate_zone_vector는 컬럼 'x', 'y'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: df: pd.DataFrame, value_col: str
#   - Output: np.ndarray
def _accumulate_zone_vector(df: pd.DataFrame, value_col: str = "value") -> np.ndarray:
    vec = np.zeros(12, dtype=np.float32)
    if df.empty:
        return vec
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    for xx, yy, vv in zip(x[valid], y[valid], v[valid]):
        vec[map_to_12_zones(float(xx), float(yy))] += float(vv)
    return vec
# 기능: 누적 12D 벡터를 per-90 밀도 벡터로 변환한다 (분모 0 방지 포함).
# 데이터 입출력:
#   - Input: vec [12], exposure90: float — 경기 수 분모
#   - Output: np.ndarray [12] float32
def _safe_density_divide(vec: np.ndarray, exposure90: float) -> np.ndarray:
    return (vec / float(max(exposure90, 1e-6))).astype(np.float32)


# 기능: 문자열 또는 리스트 값을 Python 리스트로 안전하게 변환한다. 파싱 실패 시 [] 반환.
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
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    return []
# 기능: _extract_player_ids는 컬럼 'playerId'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: raw: object, max_take: int | None
#   - Output: List[int]
def _extract_player_ids(raw: object, max_take: int | None = None) -> List[int]:
    arr = _safe_eval_list(raw)
    out: List[int] = []
    for r in arr:
        pid = _to_int(r.get("playerId") if isinstance(r, dict) else r)
        if pid is None:
            continue
        if pid not in out:
            out.append(pid)
        if max_take is not None and len(out) >= max_take:
            break
    return out


# ----------------------------
# Data sources
# ----------------------------

@dataclass
class EventTables:
    off: pd.DataFrame
    deff: pd.DataFrame
    io: pd.DataFrame
    idd: pd.DataFrame
# 기능: league_mode(england/non_england)에 따라 VAEP 소스를 고르고 IO/ID parquet를 읽어 src/dst 또는 defender/opponent 스키마로 정규화한다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: data_root: Path, league_mode: str
#   - Output: EventTables
def _load_event_tables(data_root: Path, league_mode: str) -> EventTables:
    base_vaep_path = data_root / "vaep/vaep_actions.parquet"
    eng_vaep_path = data_root / "vaep/vaep_actions_england_eval.parquet"
    io_candidates = [
        data_root / "synergy_england/io_event_surfaces_base.parquet",
        data_root / "synergy_ilp_unified_non_england/io_event_surfaces_base.parquet",
        data_root / "synergy_ioid_england_eval_preproc_all/io_event_surfaces_base.parquet",
    ]
    id_candidates = [
        data_root / "synergy_england/id_event_surfaces_base.parquet",
        data_root / "synergy_ilp_unified_non_england/id_event_surfaces_base.parquet",
        data_root / "synergy_ioid_england_eval_preproc_all/id_event_surfaces_base.parquet",
    ]
    # 기능: 후보 경로 중 첫 번째로 존재하는 경로를 반환한다.
    def _pick_first_existing_path(candidates: list[Path]) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No existing file among candidates: {candidates}")

    if league_mode == "england":
        vaep_path = _pick_first_existing_path([eng_vaep_path, base_vaep_path])
    else:
        vaep_path = _pick_first_existing_path([base_vaep_path, eng_vaep_path])

    io_path = _pick_first_existing_path(io_candidates)
    id_path = _pick_first_existing_path(id_candidates)

    if not vaep_path.exists():
        vaep_path = base_vaep_path

    vaep = pd.read_parquet(
        vaep_path,
        columns=["game_id", "team_id", "player_id", "start_x", "start_y", "offensive_value", "defensive_value"],
    ).rename(columns={"start_x": "x", "start_y": "y"})

    off = vaep[["game_id", "team_id", "player_id", "x", "y", "offensive_value"]].rename(columns={"offensive_value": "value"})
    deff = vaep[["game_id", "team_id", "player_id", "x", "y", "defensive_value"]].rename(columns={"defensive_value": "value"})

    io_raw = pd.read_parquet(io_path)
    io = io_raw[
        ["game_id", "team_id", "actor_player_id", "receiver_player_id", "contribution_source", "x", "y", "io_event_weighted"]
    ].copy()
    io["contribution_source"] = io["contribution_source"].astype(str).str.lower()
    is_second = io["contribution_source"].eq("second_action")
    io["src_player_id"] = np.where(is_second, io["receiver_player_id"], io["actor_player_id"])
    io["dst_player_id"] = np.where(is_second, io["actor_player_id"], io["receiver_player_id"])
    io = io.rename(columns={"io_event_weighted": "value"})
    io = io[["game_id", "team_id", "src_player_id", "dst_player_id", "x", "y", "value"]]

    id_raw = pd.read_parquet(id_path)
    idd = id_raw[
        ["game_id", "defending_team_id", "opponent_team_id", "defender_player_id", "opponent_player_id", "x", "y", "id_event_weighted"]
    ].rename(columns={"id_event_weighted": "value"}).copy()

    for df in (off, deff):
        for c in ["game_id", "team_id", "player_id", "x", "y", "value"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["game_id", "team_id", "player_id", "x", "y"], inplace=True)
        df[["game_id", "team_id", "player_id"]] = df[["game_id", "team_id", "player_id"]].astype(int)

    for c in ["game_id", "team_id", "src_player_id", "dst_player_id", "x", "y", "value"]:
        io[c] = pd.to_numeric(io[c], errors="coerce")
    io = io.dropna(subset=["game_id", "team_id", "src_player_id", "dst_player_id", "x", "y"])
    io[["game_id", "team_id", "src_player_id", "dst_player_id"]] = io[["game_id", "team_id", "src_player_id", "dst_player_id"]].astype(int)

    for c in ["game_id", "defending_team_id", "opponent_team_id", "defender_player_id", "opponent_player_id", "x", "y", "value"]:
        idd[c] = pd.to_numeric(idd[c], errors="coerce")
    idd = idd.dropna(subset=["game_id", "defending_team_id", "opponent_team_id", "defender_player_id", "opponent_player_id", "x", "y"])
    idd[["game_id", "defending_team_id", "opponent_team_id", "defender_player_id", "opponent_player_id"]] = idd[["game_id", "defending_team_id", "opponent_team_id", "defender_player_id", "opponent_player_id"]].astype(int)

    return EventTables(off=off, deff=deff, io=io, idd=idd)


# ----------------------------
# Squad and team helpers
# ----------------------------
# 기능: _resolve_team_id는 컬럼 'name_l', 'name', 'wyId'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: teams_df: pd.DataFrame, name_query: str
#   - Output: int
def _resolve_team_id(teams_df: pd.DataFrame, name_query: str) -> int:
    q = name_query.strip().lower()
    candidates = teams_df.copy()
    candidates["name_l"] = candidates["name"].astype(str).str.lower()
    hit = candidates[candidates["name_l"].str.contains(q, regex=False)]
    if hit.empty:
        raise ValueError(f"team not found for query: {name_query}")
    return int(hit.iloc[0]["wyId"])
# 기능: 매치 CSV에서 'wyId','team1.teamId','team2.teamId','dateutc/date'를 정규화해 시간순 배치 실행이 가능한 테이블로 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_csv: Path
#   - Output: pd.DataFrame
def _prepare_matches(matches_csv: Path) -> pd.DataFrame:
    m = pd.read_csv(matches_csv)
    if "dateutc" in m.columns:
        m["match_time"] = pd.to_datetime(m["dateutc"], errors="coerce", utc=True)
    else:
        m["match_time"] = pd.to_datetime(m["date"], errors="coerce", utc=True)
    m["wyId"] = pd.to_numeric(m["wyId"], errors="coerce")
    m = m.dropna(subset=["wyId", "match_time"]).copy()
    m["wyId"] = m["wyId"].astype(int)
    return m.sort_values(["match_time", "wyId"]).reset_index(drop=True)
# 기능: _extract_registered_squad_from_match는 컬럼 'wyId', 'team1.teamId', 'team2.teamId', 'lineup', 'bench'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_df: pd.DataFrame, match_id: int, team_id: int
#   - Output: List[int]
def _extract_registered_squad_from_match(matches_df: pd.DataFrame, match_id: int, team_id: int) -> List[int]:
    hit = matches_df[matches_df["wyId"] == int(match_id)]
    if hit.empty:
        return []

    row = hit.iloc[0]
    side = None
    if _to_int(row.get("team1.teamId")) == int(team_id):
        side = "team1"
    elif _to_int(row.get("team2.teamId")) == int(team_id):
        side = "team2"
    if side is None:
        return []

    lineup = _extract_player_ids(row.get(f"{side}.formation.lineup"), max_take=None)
    bench = _extract_player_ids(row.get(f"{side}.formation.bench"), max_take=None)

    if (not lineup or not bench) and f"{side}.formation" in row.index:
        parsed = _safe_literal(row.get(f"{side}.formation"))
        if isinstance(parsed, dict):
            if not lineup:
                lineup = _extract_player_ids(parsed.get("lineup"), max_take=None)
            if not bench:
                bench = _extract_player_ids(parsed.get("bench"), max_take=None)

    squad: List[int] = []
    for pid in lineup + bench:
        if pid not in squad:
            squad.append(pid)
    return squad
# 기능: 입력 player_id 시퀀스에서 None 제거·중복 제거·정수 변환을 수행하여 정제된 리스트를 반환한다.
# 데이터 입출력:
#   - Input: available_player_ids: Sequence[int] | None
#   - Output: List[int] — 중복 없는 정수 player_id 리스트
def _sanitize_available_player_ids(available_player_ids: Sequence[int] | None) -> List[int]:
    if available_player_ids is None:
        return []
    out: List[int] = []
    seen = set()
    for pid in available_player_ids:
        p = _to_int(pid)
        if p is None:
            continue
        if p in seen:
            continue
        seen.add(int(p))
        out.append(int(p))
    return out
# 기능: _build_squad_pool_by_minutes는 컬럼 'team_id', 'game_id', 'player_id', 'minutes_played', 연산 groupby/agg/sort_values을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: player_games_df: pd.DataFrame, team_id: int, valid_game_ids: set[int], squad_size: int
#   - Output: List[int]
def _build_squad_pool_by_minutes(
    player_games_df: pd.DataFrame,
    team_id: int,
    valid_game_ids: set[int],
    squad_size: int,
) -> List[int]:
    pg = player_games_df.copy()
    pg["team_id"] = pd.to_numeric(pg["team_id"], errors="coerce")
    pg["game_id"] = pd.to_numeric(pg["game_id"], errors="coerce")
    pg["player_id"] = pd.to_numeric(pg["player_id"], errors="coerce")
    pg["minutes_played"] = pd.to_numeric(pg.get("minutes_played"), errors="coerce").fillna(0.0)

    pg = pg.dropna(subset=["team_id", "game_id", "player_id"]).copy()
    pg = pg[(pg["team_id"].astype(int) == int(team_id)) & (pg["game_id"].astype(int).isin(valid_game_ids))].copy()
    if pg.empty:
        raise ValueError(f"No player_games rows for team_id={team_id} in historical matches")

    ranked = (
        pg.groupby("player_id", as_index=False)["minutes_played"]
        .sum()
        .sort_values(["minutes_played", "player_id"], ascending=[False, True])
    )
    out = ranked["player_id"].astype(int).tolist()[: int(squad_size)]
    if len(out) < 11:
        raise ValueError(f"Insufficient squad pool by minutes for team_id={team_id}: got {len(out)}")
    return out


# ----------------------------
# Feature cache
# ----------------------------

@dataclass
class TeamCache:
    team_id: int
    squad_player_ids: List[int]
    node_feat_24: torch.Tensor            # [N,24]
    exposure90: Dict[int, float]


@dataclass
class MatchupCache:
    home: TeamCache
    away: TeamCache
    home_io: torch.Tensor                 # [Nh,Nh,12]
    away_io: torch.Tensor                 # [Na,Na,12]
    home_to_away_id: torch.Tensor         # [Nh,Na,12]
    away_to_home_id: torch.Tensor         # [Na,Nh,12]


@dataclass
class OutcomePrediction:
    win_prob: float
    draw_prob: float
    loss_prob: float
    expected_points: float
# 기능: scaler payload 딕셔너리에서 mean/std를 가진 블록을 우선순위 키 목록으로 검색한다.
# 데이터 입출력:
#   - Input: payload: dict, key_candidates: List[str] — 검색할 키 우선순위 목록
#   - Output: dict({'mean':..., 'std':...}) | None
def _resolve_scaler_block(payload: dict, key_candidates: List[str]) -> dict | None:
    for k in key_candidates:
        if k in payload and isinstance(payload[k], dict):
            blk = payload[k]
            if "mean" in blk and "std" in blk:
                return blk
    return None
# 기능: _parse_feature_scaler_payload는 컬럼 'node_mean', 'node_std', 'passes_mean', 'passes_std', 'defends_mean'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: payload: object
#   - Output: dict
def _parse_feature_scaler_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("scaler payload must be a dict")

    node_blk = _resolve_scaler_block(payload, ["node", "nodes"])
    pass_blk = _resolve_scaler_block(payload, ["passes_to", "io", "edge_io"])
    def_blk = _resolve_scaler_block(payload, ["defends_against", "id", "edge_id"])

    # Backward-compat: flat keys style.
    if node_blk is None and ("node_mean" in payload and "node_std" in payload):
        node_blk = {"mean": payload["node_mean"], "std": payload["node_std"]}
    if pass_blk is None and ("passes_mean" in payload and "passes_std" in payload):
        pass_blk = {"mean": payload["passes_mean"], "std": payload["passes_std"]}
    if def_blk is None and ("defends_mean" in payload and "defends_std" in payload):
        def_blk = {"mean": payload["defends_mean"], "std": payload["defends_std"]}

    if node_blk is None or pass_blk is None or def_blk is None:
        raise ValueError("scaler payload missing required blocks: node, passes_to, defends_against")
    # 기능: scaler 값을 1D float32 텐서로 변환한다. 빈 텐서이면 ValueError를 발생시킨다.
    def _to_1d(name: str, x: object) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32).view(-1)
        if t.numel() == 0:
            raise ValueError(f"scaler tensor is empty: {name}")
        return t

    scaler = {
        "node_mean": _to_1d("node_mean", node_blk["mean"]),
        "node_std": _to_1d("node_std", node_blk["std"]),
        "passes_mean": _to_1d("passes_mean", pass_blk["mean"]),
        "passes_std": _to_1d("passes_std", pass_blk["std"]),
        "defends_mean": _to_1d("defends_mean", def_blk["mean"]),
        "defends_std": _to_1d("defends_std", def_blk["std"]),
    }
    for k in ["node_std", "passes_std", "defends_std"]:
        s = scaler[k]
        scaler[k] = torch.where(s.abs() > EPS, s, torch.ones_like(s))
    return scaler
# 기능: 별도 scaler .pt 파일이 있으면 로드하여 파싱된 scaler 딕셔너리를 반환한다. None이면 None 반환.
# 데이터 입출력:
#   - Input: scaler_pt: Path | None
#   - Output: dict | None (scaler_pt=None이거나 파일 없으면 None)
def _load_feature_scaler(scaler_pt: Path | None) -> dict | None:
    if scaler_pt is None:
        return None
    if not scaler_pt.exists():
        raise FileNotFoundError(f"scaler pt not found: {scaler_pt}")
    payload = torch.load(scaler_pt, map_location="cpu", weights_only=False)
    return _parse_feature_scaler_payload(payload)
# 기능: 텐서에 (x-mean)/std z-score 변환을 적용한다. 브로드캐스팅으로 임의 차원 텐서에 동작한다.
# 데이터 입출력:
#   - Input: t: Tensor[..., D], mean: Tensor[D], std: Tensor[D]
#   - Output: Tensor[..., D]
def _zscore_tensor(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t.to(torch.float32) - mean.view(*([1] * (t.ndim - 1)), -1)) / std.view(*([1] * (t.ndim - 1)), -1)
# 기능: _apply_scaler_to_cache_inplace는 컬럼 'node_mean', 'node_std', 'passes_mean', 'passes_std', 'defends_mean'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: cache: MatchupCache, scaler: dict | None
#   - Output: None
def _apply_scaler_to_cache_inplace(cache: MatchupCache, scaler: dict | None) -> None:
    if scaler is None:
        return
    cache.home.node_feat_24 = _zscore_tensor(cache.home.node_feat_24, scaler["node_mean"], scaler["node_std"])
    cache.away.node_feat_24 = _zscore_tensor(cache.away.node_feat_24, scaler["node_mean"], scaler["node_std"])

    cache.home_io = _zscore_tensor(cache.home_io, scaler["passes_mean"], scaler["passes_std"])
    cache.away_io = _zscore_tensor(cache.away_io, scaler["passes_mean"], scaler["passes_std"])

    cache.home_to_away_id = _zscore_tensor(cache.home_to_away_id, scaler["defends_mean"], scaler["defends_std"])
    cache.away_to_home_id = _zscore_tensor(cache.away_to_home_id, scaler["defends_mean"], scaler["defends_std"])
# 기능: _apply_scaler_to_graph_inplace는 컬럼 'home_team', 'node_mean', 'node_std', 'away_team', 'passes_mean'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: data: HeteroData, scaler: dict | None
#   - Output: None
def _apply_scaler_to_graph_inplace(data: HeteroData, scaler: dict | None) -> None:
    if scaler is None:
        return

    home_x = _zscore_tensor(data["home_team"].x, scaler["node_mean"], scaler["node_std"])
    away_x = _zscore_tensor(data["away_team"].x, scaler["node_mean"], scaler["node_std"])
    data["home_team"].x = home_x
    data["away_team"].x = away_x
    data["home_team"].x_off = home_x[:, :12]
    data["home_team"].x_def = home_x[:, 12:24]
    data["away_team"].x_off = away_x[:, :12]
    data["away_team"].x_def = away_x[:, 12:24]

    rel = ("home_team", "passes_to", "home_team")
    attr = _zscore_tensor(data[rel].edge_attr, scaler["passes_mean"], scaler["passes_std"])
    data[rel].edge_attr = attr
    data[rel].io_attr = attr

    rel = ("away_team", "passes_to", "away_team")
    attr = _zscore_tensor(data[rel].edge_attr, scaler["passes_mean"], scaler["passes_std"])
    data[rel].edge_attr = attr
    data[rel].io_attr = attr

    rel = ("home_team", "defends_against", "away_team")
    attr = _zscore_tensor(data[rel].edge_attr, scaler["defends_mean"], scaler["defends_std"])
    data[rel].edge_attr = attr
    data[rel].id_attr = attr

    rel = ("away_team", "defends_against", "home_team")
    attr = _zscore_tensor(data[rel].edge_attr, scaler["defends_mean"], scaler["defends_std"])
    data[rel].edge_attr = attr
    data[rel].id_attr = attr
# 기능: _build_exposure90_map는 컬럼 'player_id', 'game_id', 연산 groupby/agg을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: off_df: pd.DataFrame, def_df: pd.DataFrame
#   - Output: Dict[int, float]
def _build_exposure90_map(off_df: pd.DataFrame, def_df: pd.DataFrame) -> Dict[int, float]:
    if off_df.empty and def_df.empty:
        return {}
    base = pd.concat([
        off_df[["player_id", "game_id"]],
        def_df[["player_id", "game_id"]],
    ], ignore_index=True)
    base = base.dropna(subset=["player_id", "game_id"]).copy()
    base["player_id"] = pd.to_numeric(base["player_id"], errors="coerce")
    base["game_id"] = pd.to_numeric(base["game_id"], errors="coerce")
    base = base.dropna(subset=["player_id", "game_id"]).copy()
    if base.empty:
        return {}
    g = base.groupby("player_id")["game_id"].nunique().astype(float)
    return {int(pid): float(max(n, 1.0)) for pid, n in g.items()}
# 기능: _build_team_node_cache는 컬럼 'game_id', 'player_id'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: team_id: int, squad_ids: List[int], events: EventTables, past_ids: set[int]
#   - Output: TeamCache
def _build_team_node_cache(team_id: int, squad_ids: List[int], events: EventTables, past_ids: set[int]) -> TeamCache:
    off_hist = events.off[events.off["game_id"].isin(past_ids)]
    def_hist = events.deff[events.deff["game_id"].isin(past_ids)]

    off_team = off_hist[off_hist["player_id"].isin(squad_ids)]
    def_team = def_hist[def_hist["player_id"].isin(squad_ids)]

    exposure90 = _build_exposure90_map(off_team, def_team)

    feats = []
    for pid in squad_ids:
        off_vec = _accumulate_zone_vector(off_team[off_team["player_id"] == int(pid)], value_col="value")
        def_vec = _accumulate_zone_vector(def_team[def_team["player_id"] == int(pid)], value_col="value")
        vec = np.concatenate([off_vec, def_vec], axis=0)
        vec = _safe_density_divide(vec, float(exposure90.get(int(pid), 1.0)))
        feats.append(vec)

    node_feat = torch.tensor(np.stack(feats, axis=0).astype(np.float32), dtype=torch.float32)
    return TeamCache(team_id=team_id, squad_player_ids=squad_ids, node_feat_24=node_feat, exposure90=exposure90)
# 기능: _build_homeaway_edge_caches는 컬럼 'game_id', 'team_id', 'src_player_id', 'dst_player_id', 'defending_team_id'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: home: TeamCache, away: TeamCache, events: EventTables, past_ids: set[int]
#   - Output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
def _build_homeaway_edge_caches(home: TeamCache, away: TeamCache, events: EventTables, past_ids: set[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    io_hist = events.io[events.io["game_id"].isin(past_ids)]
    id_hist = events.idd[events.idd["game_id"].isin(past_ids)]

    # IO caches
    # 기능: build_io_for_team는 컬럼 'team_id', 'src_player_id', 'dst_player_id'을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
    # 데이터 입출력:
    #   - Input: team: TeamCache
    #   - Output: torch.Tensor
    def build_io_for_team(team: TeamCache) -> torch.Tensor:
        n = len(team.squad_player_ids)
        cache = np.zeros((n, n, 12), dtype=np.float32)
        tdf = io_hist[io_hist["team_id"] == int(team.team_id)]
        for i, src in enumerate(team.squad_player_ids):
            for j, dst in enumerate(team.squad_player_ids):
                if i == j:
                    continue
                e = tdf[(tdf["src_player_id"] == int(src)) & (tdf["dst_player_id"] == int(dst))]
                vec = _accumulate_zone_vector(e, value_col="value")
                ex = 0.5 * (float(team.exposure90.get(int(src), 1.0)) + float(team.exposure90.get(int(dst), 1.0)))
                cache[i, j, :] = _safe_density_divide(vec, ex)
        return torch.tensor(cache, dtype=torch.float32)

    home_io = build_io_for_team(home)
    away_io = build_io_for_team(away)

    # ID caches
    home_to_away = np.zeros((len(home.squad_player_ids), len(away.squad_player_ids), 12), dtype=np.float32)
    away_to_home = np.zeros((len(away.squad_player_ids), len(home.squad_player_ids), 12), dtype=np.float32)

    h2a_df = id_hist[(id_hist["defending_team_id"] == int(home.team_id)) & (id_hist["opponent_team_id"] == int(away.team_id))]
    a2h_df = id_hist[(id_hist["defending_team_id"] == int(away.team_id)) & (id_hist["opponent_team_id"] == int(home.team_id))]

    for i, dpid in enumerate(home.squad_player_ids):
        for j, opid in enumerate(away.squad_player_ids):
            e = h2a_df[(h2a_df["defender_player_id"] == int(dpid)) & (h2a_df["opponent_player_id"] == int(opid))]
            vec = _accumulate_zone_vector(e, value_col="value")
            ex = 0.5 * (float(home.exposure90.get(int(dpid), 1.0)) + float(away.exposure90.get(int(opid), 1.0)))
            home_to_away[i, j, :] = _safe_density_divide(vec, ex)

    for i, dpid in enumerate(away.squad_player_ids):
        for j, opid in enumerate(home.squad_player_ids):
            e = a2h_df[(a2h_df["defender_player_id"] == int(dpid)) & (a2h_df["opponent_player_id"] == int(opid))]
            vec = _accumulate_zone_vector(e, value_col="value")
            ex = 0.5 * (float(away.exposure90.get(int(dpid), 1.0)) + float(home.exposure90.get(int(opid), 1.0)))
            away_to_home[i, j, :] = _safe_density_divide(vec, ex)

    return (
        home_io,
        away_io,
        torch.tensor(home_to_away, dtype=torch.float32),
        torch.tensor(away_to_home, dtype=torch.float32),
    )
# 기능: asof_time 이전 경기 기록으로 home/away 스쿼드와 IO/ID 3D 캐시 텐서를 사전 계산해 GA 루프에서 재사용 가능한 MatchupCache를 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성; 엔티티 키(game_id/team_id/player_id) 일관성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_df: pd.DataFrame, player_games_df: pd.DataFrame, events: EventTables, home_team_id: int, away_team_id: int, match_id: int, ...
#   - Output: MatchupCache
def build_matchup_cache(
    matches_df: pd.DataFrame,
    player_games_df: pd.DataFrame,
    events: EventTables,
    home_team_id: int,
    away_team_id: int,
    match_id: int,
    asof_time: pd.Timestamp,
    squad_size: int,
    available_home_player_ids: Sequence[int] | None = None,
    available_away_player_ids: Sequence[int] | None = None,
) -> MatchupCache:
    past_ids = set(matches_df.loc[matches_df["match_time"] < asof_time, "wyId"].astype(int).tolist())
    if not past_ids:
        raise ValueError("No historical matches before asof_time")

    # Event tables may use a different game_id namespace than matches.wyId.
    # If overlap is zero, using wyId-filtered history would collapse all features to zeros.
    event_game_ids = set(pd.to_numeric(events.off["game_id"], errors="coerce").dropna().astype(int).tolist())
    event_past_ids = past_ids & event_game_ids
    if not event_past_ids:
        event_past_ids = event_game_ids
        print(
            "[WARN] No overlap between matches.wyId and event.game_id before asof_time; "
            "falling back to full event history for feature cache assembly."
        )

    # Priority 0: explicit available-player constraints (e.g., matchday lineup + bench).
    home_squad = _sanitize_available_player_ids(available_home_player_ids)
    away_squad = _sanitize_available_player_ids(available_away_player_ids)

    # Priority 1: actual registered matchday squad from target match (lineup + bench).
    if not home_squad:
        home_squad = _extract_registered_squad_from_match(matches_df, match_id=match_id, team_id=home_team_id)
    if not away_squad:
        away_squad = _extract_registered_squad_from_match(matches_df, match_id=match_id, team_id=away_team_id)

    # Fallback: cumulative minutes ranking from historical matches.
    if not home_squad:
        home_squad = _build_squad_pool_by_minutes(
            player_games_df=player_games_df,
            team_id=home_team_id,
            valid_game_ids=past_ids,
            squad_size=squad_size,
        )
    else:
        home_squad = home_squad[: int(squad_size)]

    if not away_squad:
        away_squad = _build_squad_pool_by_minutes(
            player_games_df=player_games_df,
            team_id=away_team_id,
            valid_game_ids=past_ids,
            squad_size=squad_size,
        )
    else:
        away_squad = away_squad[: int(squad_size)]

    if len(home_squad) < 11 or len(away_squad) < 11:
        raise ValueError(
            f"Squad pool too small: home={len(home_squad)} away={len(away_squad)} (need >=11 each)"
        )

    home_cache = _build_team_node_cache(home_team_id, home_squad, events, event_past_ids)
    away_cache = _build_team_node_cache(away_team_id, away_squad, events, event_past_ids)

    home_io, away_io, h2a, a2h = _build_homeaway_edge_caches(home_cache, away_cache, events, event_past_ids)
    return MatchupCache(
        home=home_cache,
        away=away_cache,
        home_io=home_io,
        away_io=away_io,
        home_to_away_id=h2a,
        away_to_home_id=a2h,
    )


# ----------------------------
# Fast graph assembly
# ----------------------------
# 기능: N개 노드에 대한 완전 방향 그래프(자기 자신 제외)의 edge_index를 생성한다.
# 동작/맥락: IO 관계(passes_to)의 같은 팀 엣지 구조에 사용된다.
#            N=11이면 11×10=110개 방향 엣지 → edge_index shape [2, 110]
# 데이터 입출력:
#   - Input: n: int — 노드 수 (선택된 11명)
#   - Output: torch.Tensor [2, n*(n-1)] — [src_indices, dst_indices]
def _complete_directed_no_self_edges(n: int) -> torch.Tensor:
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rows.append(i)
            cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)
# 기능: n_left × n_right 완전 이분 그래프의 edge_index를 생성한다.
# 동작/맥락: ID 관계(defends_against)의 교차 팀 엣지 구조에 사용된다.
#            11×11=121개 엣지 → edge_index shape [2, 121]
# 데이터 입출력:
#   - Input: n_left: int — 출발 노드 수 (수비팀 11명)
#            n_right: int — 도착 노드 수 (공격팀 11명)
#   - Output: torch.Tensor [2, n_left*n_right]
def _full_bipartite_edges(n_left: int, n_right: int) -> torch.Tensor:
    rows, cols = [], []
    for i in range(n_left):
        for j in range(n_right):
            rows.append(i)
            cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)
# 기능: 3D 캐시 텐서에서 선택된 선수 인덱스 쌍의 엣지 특징을 추출하여 2D 행렬로 반환한다.
# 동작/맥락: cache_3d[i, j, :] 는 선수 i → 선수 j의 12D 공간 시너지 벡터를 사전 계산한 값이다.
#            GA 루프 내 매 genome 평가마다 직접 VAEP를 재계산하는 대신 이 캐시를 조회하여 O(N²) 재계산을 방지한다.
#            directed_no_self=True: IO 관계에서 자기 자신 제외 (passes_to 엣지에 self-loop 없음)
# 데이터 입출력:
#   - Input: cache_3d: torch.Tensor [N_squad, N_squad, 12] — 스쿼드 전체 IO 캐시
#            selected: Sequence[int] — 선택된 11명의 스쿼드 내 인덱스
#            directed_no_self: bool — True이면 i==j 조합 제외
#   - Output: torch.Tensor [E, 12] — 선택된 쌍들의 엣지 특징 (E=110 for IO, E=121 for ID)
def _gather_pair_attrs(cache_3d: torch.Tensor, selected: Sequence[int], directed_no_self: bool = True) -> torch.Tensor:
    attrs = []
    for i_idx in selected:
        for j_idx in selected:
            if directed_no_self and i_idx == j_idx:
                continue
            attrs.append(cache_3d[int(i_idx), int(j_idx), :])
    if not attrs:
        return torch.zeros((0, 12), dtype=torch.float32)
    return torch.stack(attrs, dim=0)
# 기능: 3D 캐시 텐서에서 교차 팀(이분 그래프) 엣지 특징을 추출한다.
# 동작/맥락: ID 관계의 home×away 엣지 특징 조회에 사용한다.
# 데이터 입출력:
#   - Input: cache_3d [N_left, N_right, 12], left_sel [11], right_sel [11]
#   - Output: Tensor [121, 12]
def _gather_cross_attrs(cache_3d: torch.Tensor, left_sel: Sequence[int], right_sel: Sequence[int]) -> torch.Tensor:
    attrs = []
    for i_idx in left_sel:
        for j_idx in right_sel:
            attrs.append(cache_3d[int(i_idx), int(j_idx), :])
    if not attrs:
        return torch.zeros((0, 12), dtype=torch.float32)
    return torch.stack(attrs, dim=0)
# 기능: GA 루프 내에서 genome(선수 인덱스 11개) 하나에 대한 HeteroData를 캐시 조회로 즉시 조립한다.
# 동작/맥락: build_rolling_gnn_dataset과 동일한 그래프 구조를 생성하되, 이벤트 DataFrame 재처리 없이
#            사전 계산된 MatchupCache의 3D 텐서에서 선택된 선수 부분만 슬라이싱하여 조립한다.
#   조립 결과:
#     - home_team.x: cache.home.node_feat_24[home_sel] → [11, 24]
#     - away_team.x: cache.away.node_feat_24[away_sel] → [11, 24]
#     - (home_team, passes_to, home_team).edge_attr: cache.home_io[home_sel][:, home_sel] → [110, 12]
#     - (away_team, passes_to, away_team).edge_attr: cache.away_io[away_sel][:, away_sel] → [110, 12]
#     - (home_team, defends_against, away_team).edge_attr: cache.home_to_away_id[home_sel][:, away_sel] → [121, 12]
#     - (away_team, defends_against, home_team).edge_attr: cache.away_to_home_id[away_sel][:, home_sel] → [121, 12]
#   GA genome 하나 평가: 이 함수 호출(캐시 조회) + GNN forward pass 만으로 끝난다 → 매우 빠름
# 데이터 입출력:
#   - Input: cache: MatchupCache — 사전 계산된 스쿼드 전체의 3D 특징 캐시
#            home_sel: Sequence[int] — 홈팀에서 선택된 11명의 스쿼드 내 인덱스
#            away_sel: Sequence[int] — 어웨이팀에서 선택된 11명의 스쿼드 내 인덱스
#   - Output: HeteroData — GNN forward에 바로 넣을 수 있는 완성된 그래프
def build_fast_heterodata(
    cache: MatchupCache,
    home_sel: Sequence[int],
    away_sel: Sequence[int],
) -> HeteroData:
    n_home = len(home_sel)
    n_away = len(away_sel)

    data = HeteroData()
    home_x = cache.home.node_feat_24[torch.tensor(home_sel, dtype=torch.long)]
    away_x = cache.away.node_feat_24[torch.tensor(away_sel, dtype=torch.long)]

    data["home_team"].x = home_x
    data["home_team"].x_off = home_x[:, :12]
    data["home_team"].x_def = home_x[:, 12:24]
    data["away_team"].x = away_x
    data["away_team"].x_off = away_x[:, :12]
    data["away_team"].x_def = away_x[:, 12:24]

    hh_idx = _complete_directed_no_self_edges(n_home)
    aa_idx = _complete_directed_no_self_edges(n_away)
    ha_idx = _full_bipartite_edges(n_home, n_away)
    ah_idx = _full_bipartite_edges(n_away, n_home)

    data[("home_team", "passes_to", "home_team")].edge_index = hh_idx
    home_io_attr = _gather_pair_attrs(cache.home_io, home_sel, directed_no_self=True)
    data[("home_team", "passes_to", "home_team")].edge_attr = home_io_attr
    data[("home_team", "passes_to", "home_team")].io_attr = home_io_attr

    data[("away_team", "passes_to", "away_team")].edge_index = aa_idx
    away_io_attr = _gather_pair_attrs(cache.away_io, away_sel, directed_no_self=True)
    data[("away_team", "passes_to", "away_team")].edge_attr = away_io_attr
    data[("away_team", "passes_to", "away_team")].io_attr = away_io_attr

    data[("home_team", "defends_against", "away_team")].edge_index = ha_idx
    home_id_attr = _gather_cross_attrs(cache.home_to_away_id, home_sel, away_sel)
    data[("home_team", "defends_against", "away_team")].edge_attr = home_id_attr
    data[("home_team", "defends_against", "away_team")].id_attr = home_id_attr

    data[("away_team", "defends_against", "home_team")].edge_index = ah_idx
    away_id_attr = _gather_cross_attrs(cache.away_to_home_id, away_sel, home_sel)
    data[("away_team", "defends_against", "home_team")].edge_attr = away_id_attr
    data[("away_team", "defends_against", "home_team")].id_attr = away_id_attr

    return data
# 기능: _validate_subgraph_slice_shapes는 컬럼 'home_team', 'away_team'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: data: HeteroData, expected_home_n: int, expected_away_n: int
#   - Output: None
def _validate_subgraph_slice_shapes(data: HeteroData, expected_home_n: int, expected_away_n: int) -> None:
    rel_home_io = ("home_team", "passes_to", "home_team")
    rel_away_io = ("away_team", "passes_to", "away_team")
    rel_home_id = ("home_team", "defends_against", "away_team")
    rel_away_id = ("away_team", "defends_against", "home_team")

    actual_home_n = int(data["home_team"].x.size(0))
    actual_away_n = int(data["away_team"].x.size(0))
    if actual_home_n != int(expected_home_n) or actual_away_n != int(expected_away_n):
        raise RuntimeError(
            f"Subgraph node slicing mismatch: home={actual_home_n}/{expected_home_n}, away={actual_away_n}/{expected_away_n}"
        )

    expected_home_io = expected_home_n * (expected_home_n - 1)
    expected_away_io = expected_away_n * (expected_away_n - 1)
    expected_cross = expected_home_n * expected_away_n

    checks = [
        (int(data[rel_home_io].edge_index.size(1)), expected_home_io, "home_io_edge_index"),
        (int(data[rel_home_io].edge_attr.size(0)), expected_home_io, "home_io_edge_attr"),
        (int(data[rel_away_io].edge_index.size(1)), expected_away_io, "away_io_edge_index"),
        (int(data[rel_away_io].edge_attr.size(0)), expected_away_io, "away_io_edge_attr"),
        (int(data[rel_home_id].edge_index.size(1)), expected_cross, "home_id_edge_index"),
        (int(data[rel_home_id].edge_attr.size(0)), expected_cross, "home_id_edge_attr"),
        (int(data[rel_away_id].edge_index.size(1)), expected_cross, "away_id_edge_index"),
        (int(data[rel_away_id].edge_attr.size(0)), expected_cross, "away_id_edge_attr"),
    ]
    for actual, expected, name in checks:
        if actual != int(expected):
            raise RuntimeError(f"Subgraph edge slicing mismatch: {name}={actual}, expected={expected}")
# 기능: _lineup_signature는 컬럼 'home_team'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: data: HeteroData
#   - Output: tuple[float, float]
def _lineup_signature(data: HeteroData) -> tuple[float, float]:
    rel_home_io = ("home_team", "passes_to", "home_team")
    node_sum = float(data["home_team"].x.sum().item()) if data["home_team"].x.numel() > 0 else 0.0
    edge_sum = float(data[rel_home_io].edge_attr.sum().item()) if data[rel_home_io].edge_attr.numel() > 0 else 0.0
    return (round(node_sum, 6), round(edge_sum, 6))


# ----------------------------
# Model loading and GA
# ----------------------------
# 기능: train_gnn_phase5.py를 동적으로 import하여 HeteroEdgeGATWinPredictor 클래스에 접근한다.
# 동작/맥락: optimize_lineup_ga_phase6.py가 train_gnn_phase5.py와 다른 디렉토리에 있어도 동작하도록 경로 기반 import를 사용한다.
# 데이터 입출력:
#   - Input: model_def_path: Path — train_gnn_phase5.py 절대 경로
#   - Output: 모듈 객체 (mod.HeteroEdgeGATWinPredictor 접근 가능)
def _import_model_module(model_def_path: Path):
    spec = importlib.util.spec_from_file_location("train_gnn_phase5", model_def_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _LegacyHeteroEdgeGATWinPredictor(nn.Module):
    """Backward-compatible model for older checkpoints saved with HeteroConv keys."""
    # 기능: __init__는 연산 GATConv을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: hidden_channels: int, num_layers: int, heads: int, dropout: float, num_classes: int
    #   - Output: None
    def __init__(
        self,
        hidden_channels: int = 96,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.15,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)

        rel_home_io = ("home_team", "passes_to", "home_team")
        rel_away_io = ("away_team", "passes_to", "away_team")
        rel_home_id = ("home_team", "defends_against", "away_team")
        rel_away_id = ("away_team", "defends_against", "home_team")

        self.convs = nn.ModuleList()
        for _ in range(int(num_layers)):
            conv = HeteroConv(
                {
                    rel_home_io: GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                    rel_away_io: GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                    rel_home_id: GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                    rel_away_id: GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.head = nn.Sequential(
            nn.LazyLinear(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, self.num_classes),
        )
    # 기능: 단일 그래프 추론 시 batch 속성 없어도 안전하게 0-벡터 batch를 반환한다.
    @staticmethod
    def _node_batch(node_store) -> torch.Tensor:
        if hasattr(node_store, "batch") and node_store.batch is not None:
            return node_store.batch
        return torch.zeros(node_store.x.size(0), dtype=torch.long, device=node_store.x.device)
    # 기능: forward는 컬럼 'home_team', 'away_team'을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: data
    #   - Output: 코드 내부 return 표현식
    def forward(self, data):
        rel_home_io = ("home_team", "passes_to", "home_team")
        rel_away_io = ("away_team", "passes_to", "away_team")
        rel_home_id = ("home_team", "defends_against", "away_team")
        rel_away_id = ("away_team", "defends_against", "home_team")

        x_dict = {
            "home_team": data["home_team"].x,
            "away_team": data["away_team"].x,
        }
        edge_index_dict = {
            rel_home_io: data[rel_home_io].edge_index,
            rel_away_io: data[rel_away_io].edge_index,
            rel_home_id: data[rel_home_id].edge_index,
            rel_away_id: data[rel_away_id].edge_index,
        }
        edge_attr_dict = {
            rel_home_io: data[rel_home_io].edge_attr,
            rel_away_io: data[rel_away_io].edge_attr,
            rel_home_id: data[rel_home_id].edge_attr,
            rel_away_id: data[rel_away_id].edge_attr,
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}

        home_batch = self._node_batch(data["home_team"])
        away_batch = self._node_batch(data["away_team"])
        home_pool = global_mean_pool(x_dict["home_team"], home_batch)
        away_pool = global_mean_pool(x_dict["away_team"], away_batch)
        match_repr = torch.cat([home_pool, away_pool], dim=-1)
        return self.head(match_repr)
# 기능: 학습된 모델 체크포인트를 로드하고 추론 준비가 완료된 모델과 scaler payload를 반환한다.
# 동작/맥락: 구버전/신버전 체크포인트를 자동 감지하여 적합한 모델 클래스로 복원한다:
#   - state_dict 키가 "convs."로 시작 → _LegacyHeteroEdgeGATWinPredictor (HeteroConv 기반 구버전)
#   - 그 외 → HeteroEdgeGATWinPredictor (4-인코더 + SemanticAttn 현재 버전)
#   - LazyLinear 초기화: load_state_dict 전에 dummy forward 1회 수행 (실제 입력 차원으로 lazy param 초기화)
#   - strict=False: 체크포인트와 모델 구조가 완전히 일치하지 않아도 로드 허용 (하위 호환성 유지)
# 데이터 입출력:
#   - Input: model_def_path: Path — train_gnn_phase5.py 경로
#            ckpt_path: Path      — 학습 완료 .pt 체크포인트 경로
#            device: torch.device — 추론 디바이스 (cpu/cuda)
#   - Output: Tuple[model (eval mode), ckpt_scaler_payload (dict or None)]
def load_trained_model(model_def_path: Path, ckpt_path: Path, device: torch.device):
    mod = _import_model_module(model_def_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    state = ckpt.get("state_dict", ckpt)
    legacy_state = any(str(k).startswith("convs.") for k in state.keys())

    if legacy_state:
        head_w = state.get("head.3.weight")
        legacy_num_classes = int(head_w.shape[0]) if hasattr(head_w, "shape") else 1
        model = _LegacyHeteroEdgeGATWinPredictor(
            hidden_channels=int(cfg.get("hidden_channels", 96)),
            num_layers=int(cfg.get("num_layers", 3)),
            heads=int(cfg.get("heads", 4)),
            dropout=float(cfg.get("dropout", 0.15)),
            num_classes=legacy_num_classes,
        )
    else:
        model = mod.HeteroEdgeGATWinPredictor(
            hidden_channels=int(cfg.get("hidden_channels", 96)),
            num_layers=int(cfg.get("num_layers", 3)),
            heads=int(cfg.get("heads", 4)),
            dropout=float(cfg.get("dropout", 0.15)),
            num_classes=int(cfg.get("num_classes", 3)),
        )

    # LazyLinear safety: run one dummy forward once to initialize lazy params.
    dummy = HeteroData()
    dummy["home_team"].x = torch.zeros((11, 24), dtype=torch.float32)
    dummy["away_team"].x = torch.zeros((11, 24), dtype=torch.float32)
    dummy[("home_team", "passes_to", "home_team")].edge_index = _complete_directed_no_self_edges(11)
    dummy[("away_team", "passes_to", "away_team")].edge_index = _complete_directed_no_self_edges(11)
    dummy[("home_team", "defends_against", "away_team")].edge_index = _full_bipartite_edges(11, 11)
    dummy[("away_team", "defends_against", "home_team")].edge_index = _full_bipartite_edges(11, 11)
    dummy[("home_team", "passes_to", "home_team")].edge_attr = torch.zeros((110, 12), dtype=torch.float32)
    dummy[("away_team", "passes_to", "away_team")].edge_attr = torch.zeros((110, 12), dtype=torch.float32)
    dummy[("home_team", "defends_against", "away_team")].edge_attr = torch.zeros((121, 12), dtype=torch.float32)
    dummy[("away_team", "defends_against", "home_team")].edge_attr = torch.zeros((121, 12), dtype=torch.float32)
    with torch.no_grad():
        _ = model(dummy)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    ckpt_scaler_payload = ckpt.get("feature_scaler") if isinstance(ckpt, dict) else None
    return model, ckpt_scaler_payload
# 기능: 어웨이팀 스쿼드에서 exposure90(출전 경기 수) 상위 11명을 초기 선발 XI로 결정한다.
# 동작/맥락: GA 최적화를 수행하지 않는 경우나 초기 어웨이 XI가 필요할 때 결정론적 기준선으로 사용한다.
# 데이터 입출력:
#   - Input: away_cache: TeamCache
#   - Output: List[int] — 스쿼드 풀 내 인덱스 11개
def _default_away_starting11(away_cache: TeamCache) -> List[int]:
    # choose by exposure desc as a deterministic baseline away XI
    scored = [(i, away_cache.exposure90.get(pid, 0.0)) for i, pid in enumerate(away_cache.squad_player_ids)]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [idx for idx, _ in scored[:11]]
# 기능: 홈팀 관점의 결과 확률을 어웨이팀 관점으로 변환한다.
# 동작/맥락: 모델은 항상 홈팀 관점(P_win=홈 승)으로 예측한다. 어웨이팀 GA 최적화 시에는
#            P_win↔P_loss를 뒤집어 어웨이팀 시각에서의 expected_points = 3·P_away_win + P_draw를 계산해야 한다.
# 데이터 입출력:
#   - Input: home_pred: OutcomePrediction — 홈팀 관점 (win_prob=홈 승, loss_prob=홈 패=어웨이 승)
#   - Output: OutcomePrediction — 어웨이팀 관점 (win_prob=어웨이 승=홈 패, loss_prob=어웨이 패=홈 승)
def _to_away_outcome_prediction(home_pred: OutcomePrediction) -> OutcomePrediction:
    """Convert home-perspective probabilities into away-perspective probabilities."""
    away_win = float(home_pred.loss_prob)
    away_draw = float(home_pred.draw_prob)
    away_loss = float(home_pred.win_prob)
    away_expected_points = (3.0 * away_win) + away_draw
    return OutcomePrediction(
        win_prob=away_win,
        draw_prob=away_draw,
        loss_prob=away_loss,
        expected_points=float(away_expected_points),
    )
# 기능: GNN 모델로 홈팀 경기 결과 확률(P_loss, P_draw, P_win)과 기대 승점을 계산한다.
# 동작/맥락: temperature scaling을 적용하여 모델의 과신(overconfidence)을 보정한다.
#   수식:
#     probs = softmax(logits / T)   T=2.0 (기본값, 소프트맥스 분포를 평탄하게)
#     P_loss = probs[0], P_draw = probs[1], P_win = probs[2]
#     expected_points = 3·P_win + P_draw  (축구 승점 체계: 승=3점, 무=1점, 패=0점)
#   GA 적합도 함수(fitness)로 이 expected_points를 사용한다.
#   레거시 체크포인트(logits 1개): sigmoid로 P_win만 계산, P_draw=0으로 처리 (하위 호환성)
# 데이터 입출력:
#   - Input: model — eval 상태의 GNN 모델
#            data: HeteroData — build_fast_heterodata()로 조립된 단일 그래프
#            device: torch.device
#            temperature: float — 소프트맥스 온도 (기본값 2.0, 높을수록 균등 분포에 가까워짐)
#   - Output: OutcomePrediction — {win_prob, draw_prob, loss_prob, expected_points}
def _predict_home_outcome_probs(
    model,
    data: HeteroData,
    device: torch.device,
    temperature: float = 2.0,
) -> OutcomePrediction:
    t = max(float(temperature), EPS)
    with torch.no_grad():
        logits = model(data.to(device))

    if logits.dim() == 1:
        logits = logits.view(1, -1)
    if logits.size(-1) == 1:
        # Legacy binary checkpoint: interpret single logit as P(home win).
        p_win = float(torch.sigmoid(logits[0, 0]).item())
        p_draw = 0.0
        p_loss = float(1.0 - p_win)
    elif logits.size(-1) == 3:
        probs = torch.softmax(logits / t, dim=-1)[0]
        p_loss = float(probs[0].item())
        p_draw = float(probs[1].item())
        p_win = float(probs[2].item())
    else:
        raise ValueError(f"Expected 1 or 3 logits for prediction, got shape={tuple(logits.shape)}")

    expected_points = (3.0 * p_win) + p_draw

    return OutcomePrediction(
        win_prob=p_win,
        draw_prob=p_draw,
        loss_prob=p_loss,
        expected_points=float(expected_points),
    )
# 기능: 홈 승리 확률만 반환하는 하위 호환성 래퍼 (T=1.0 고정, expected_points 불필요한 경우 사용).
# 데이터 입출력:
#   - Input: model, data: HeteroData, device: torch.device
#   - Output: float — P(홈 승)
def _predict_home_win_prob(model, data: HeteroData, device: torch.device) -> float:
    """Backward-compatible helper returning only home win probability."""
    pred = _predict_home_outcome_probs(model, data, device=device, temperature=1.0)
    return float(pred.win_prob)
# 기능: n_pool 크기 풀에서 k개의 인덱스를 무작위 비복원추출하여 초기 genome을 생성한다.
# 데이터 입출력:
#   - Input: n_pool: int — 스쿼드 크기, k: int — 선발 수 (11)
#   - Output: List[int] — 중복 없는 k개 인덱스
def _random_genome(n_pool: int, k: int) -> List[int]:
    return random.sample(range(n_pool), k)
# 기능: fixed_idx 없을 때의 genome 복구 래퍼.
def _repair_unique(genome: List[int], n_pool: int, k: int) -> List[int]:
    return _repair_unique_with_lock(genome=genome, n_pool=n_pool, k=k, fixed_idx=None)
# 기능: genome에서 중복 인덱스를 제거하고 k개가 될 때까지 무작위 보충하여 유효한 genome을 복구한다.
# 동작/맥락: 교차/변이 후 발생하는 중복·부족 문제를 수정하는 GA 복구 연산이다.
#            fixed_idx가 지정되면 그 인덱스를 항상 genome의 첫 번째 자리에 보장한다.
# 데이터 입출력:
#   - Input: genome: List[int], n_pool, k, fixed_idx: int|None
#   - Output: List[int] — 중복 없는 k개 인덱스, fixed_idx 포함 보장
def _repair_unique_with_lock(genome: List[int], n_pool: int, k: int, fixed_idx: int | None) -> List[int]:
    out, used = [], set()

    if fixed_idx is not None:
        fix = int(fixed_idx)
        if not (0 <= fix < int(n_pool)):
            raise ValueError(f"fixed_idx out of range: {fix} not in [0, {n_pool})")
        out.append(fix)
        used.add(fix)

    for g in genome:
        if 0 <= int(g) < n_pool and int(g) not in used:
            out.append(int(g))
            used.add(int(g))
        if len(out) == k:
            break
    while len(out) < k:
        c = random.randrange(n_pool)
        if c not in used:
            out.append(c)
            used.add(c)
    return out
# 기능: fixed_idx 없을 때의 단순 교차 (wrapper).
def _crossover(p1: List[int], p2: List[int], n_pool: int, k: int) -> List[int]:
    return _crossover_with_lock(p1=p1, p2=p2, n_pool=n_pool, k=k, fixed_idx=None)
# 기능: 두 부모 genome을 단일 분기점(cut)에서 교차하여 자식 genome을 생성한다.
# 동작/맥락: single-point crossover: child = p1[:cut] + p2[cut:]
#            교차 후 중복 선수 인덱스가 생길 수 있으므로 _repair_unique_with_lock으로 복구한다.
#            fixed_idx(GK): 교차 후 복구 과정에서도 GK 자리는 항상 fixed_idx로 고정된다.
# 데이터 입출력:
#   - Input: p1, p2: List[int] — 부모 genome (스쿼드 내 선수 인덱스 11개)
#            n_pool: int — 스쿼드 크기
#            k: int — 선발 선수 수 (11)
#            fixed_idx: int | None — 항상 포함할 선수 인덱스 (GK 고정용)
#   - Output: List[int] — 자식 genome (중복 없는 11개 인덱스, fixed_idx 포함 보장)
def _crossover_with_lock(p1: List[int], p2: List[int], n_pool: int, k: int, fixed_idx: int | None) -> List[int]:
    cut = random.randint(1, k - 1)
    child = p1[:cut] + p2[cut:]
    return _repair_unique_with_lock(child, n_pool=n_pool, k=k, fixed_idx=fixed_idx)
# 기능: fixed_idx 없을 때의 단순 변이 (wrapper).
def _mutate(g: List[int], n_pool: int, p_mut: float) -> List[int]:
    return _mutate_with_lock(g=g, n_pool=n_pool, p_mut=p_mut, fixed_idx=None)
# 기능: 확률 p_mut으로 genome의 무작위 한 자리를 다른 선수 인덱스로 교체하는 단일 유전자 변이를 수행한다.
# 동작/맥락: 돌연변이 적용 방식:
#   - random.random() < p_mut이면 변이 발생, 아니면 그대로
#   - 변이 위치: fixed_idx 제외한 자리 중 무작위 선택
#   - 새 선수 인덱스: 기존 선발 목록에 없는 인덱스 중 무작위 선택 (최대 100회 시도)
#   - 변이 후 _repair_unique_with_lock으로 중복 제거 및 fixed_idx 포함 보장
# 데이터 입출력:
#   - Input: g: List[int] — 현재 genome
#            n_pool: int — 스쿼드 크기
#            p_mut: float — 변이 발생 확률
#            fixed_idx: int | None — 변이 불가 자리 (GK 인덱스)
#   - Output: List[int] — 변이된 genome
def _mutate_with_lock(g: List[int], n_pool: int, p_mut: float, fixed_idx: int | None) -> List[int]:
    out = g[:]
    if random.random() < p_mut:
        mutable_positions = [i for i, idx in enumerate(out) if fixed_idx is None or int(idx) != int(fixed_idx)]
        if not mutable_positions:
            return _repair_unique_with_lock(out, n_pool=n_pool, k=len(g), fixed_idx=fixed_idx)
        pos = random.choice(mutable_positions)
        used = set(out)
        cand = random.randrange(n_pool)
        tries = 0
        while cand in used and tries < 100:
            cand = random.randrange(n_pool)
            tries += 1
        out[pos] = cand
    return _repair_unique_with_lock(out, n_pool=n_pool, k=len(g), fixed_idx=fixed_idx)
# 기능: _build_player_position_map는 컬럼 'code2', 연산 pd.read_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: players_csv: Path
#   - Output: Dict[int, str]
def _build_player_position_map(players_csv: Path) -> Dict[int, str]:
    players = pd.read_csv(players_csv)
    if not {"wyId", "role"}.issubset(players.columns):
        return {}

    pos_map: Dict[int, str] = {}
    for r in players[["wyId", "role"]].itertuples(index=False):
        pid = _to_int(r.wyId)
        if pid is None:
            continue
        role = _safe_literal(r.role)
        code = None
        if isinstance(role, dict):
            code2 = role.get("code2")
            if isinstance(code2, str):
                code2 = code2.upper().strip()
                if code2 in {"GK", "DF", "MF", "FW"}:
                    code = code2
        if code is None and isinstance(r.role, str):
            txt = r.role.upper()
            for c in ("GK", "DF", "MF", "FW"):
                if c in txt:
                    code = c
                    break
        if code in {"GK", "DF", "MF", "FW"}:
            pos_map[int(pid)] = str(code)
    return pos_map
# 기능: 스쿼드 풀에서 GK 포지션 선수의 인덱스를 결정하고 GA 전 과정에서 고정한다.
# 동작/맥락: GA는 11명 조합을 자유 탐색하지만, GK는 도메인 제약으로 항상 1명이 필요하다.
#            우선순위: ①preferred_player_ids에 포함된 GK → ②스쿼드 풀에 있는 GK → ③없으면 ValueError
#            GK 후보 중 exposure90(출전 경기 수)가 가장 높은 선수를 선택한다.
# 데이터 입출력:
#   - Input: team_cache: TeamCache — 스쿼드 풀 및 선수 특징 캐시
#            preferred_player_ids: 실제 경기 명단에 있는 선수 ID (우선 GK 검색 범위)
#            player_position_map: Dict[int, str] — {player_id: 포지션 코드 'GK'/'DF'/'MF'/'FW'}
#   - Output: int — 스쿼드 풀 내 GK 선수의 인덱스 (genome에서 항상 포함할 fixed_idx)
def resolve_fixed_gk_index(
    team_cache: TeamCache,
    preferred_player_ids: Sequence[int],
    player_position_map: Dict[int, str],
) -> int:
    preferred = set(int(pid) for pid in preferred_player_ids)
    candidates: List[int] = []

    for idx, pid in enumerate(team_cache.squad_player_ids):
        if player_position_map.get(int(pid)) != "GK":
            continue
        if preferred and int(pid) not in preferred:
            continue
        candidates.append(int(idx))

    if not candidates:
        for idx, pid in enumerate(team_cache.squad_player_ids):
            if player_position_map.get(int(pid)) == "GK":
                candidates.append(int(idx))

    if not candidates:
        raise ValueError(f"No GK found in squad pool for team_id={team_cache.team_id}")
    # 기능: GK 후보 정렬 키. exposure90 내림차순, 인덱스 오름차순으로 결정론적으로 정렬한다.
    def _score(i: int) -> tuple[float, int]:
        pid = int(team_cache.squad_player_ids[int(i)])
        return (float(team_cache.exposure90.get(pid, 0.0)), -int(i))

    candidates.sort(key=_score, reverse=True)
    return int(candidates[0])
# 기능: 유전 알고리즘(GA)으로 홈팀 기대 승점(expected_points)을 최대화하는 11인 조합을 탐색한다.
# 동작/맥락: GA 알고리즘 흐름:
#   ① 초기 집단: pop_size개 genome 무작위 생성 (fixed_player_idx=GK 항상 포함)
#   ② 세대 반복 (generations회):
#      For each genome g in pop:
#        data = build_fast_heterodata(cache, home_sel=g, away_sel)  # 캐시 조회
#        _apply_scaler_to_graph_inplace(data, graph_scaler)          # z-score 변환
#        pred = _predict_home_outcome_probs(model, data, T)          # GNN 추론
#        fitness = pred.expected_points = 3·P_win + P_draw           # 적합도
#      scored.sort(by fitness, descending)
#      elites = scored[:elite_size]           # 엘리트 보존 (truncation selection)
#      next_pop = elites + crossover+mutate   # 다음 세대 생성
#   ③ best_g(최고 genome)과 best_pred(예측 결과) 반환
#   특징: 캐시된 텐서 조회(O(1))로 per-genome 평가가 매우 빠름 → 대규모 집단/세대 탐색 가능
# 데이터 입출력:
#   - Input: model — eval 상태 GNN 모델
#            cache: MatchupCache — 사전 계산된 홈/어웨이 특징 캐시
#            away_sel: List[int] — 고정된 어웨이팀 11인 인덱스
#            pop_size: int — GA 집단 크기 (기본값: 30~50)
#            generations: int — 세대 수 (기본값: 80~100)
#            elite_size: int — 엘리트 보존 수 (기본값: 3~5)
#            mutation_p: float — 변이 확률 (기본값: 0.2~0.3)
#            temperature: float — 소프트맥스 온도 (기본값: 2.0)
#            fixed_player_idx: int|None — 항상 포함할 선수 인덱스 (GK)
#            graph_scaler: dict|None — z-score 스케일러 (None이면 변환 없음)
#   - Output: Tuple[best_genome: List[int], best_pred: OutcomePrediction]
def run_ga_optimize_home(
    model,
    cache: MatchupCache,
    away_sel: List[int],
    device: torch.device,
    pop_size: int,
    generations: int,
    elite_size: int,
    mutation_p: float,
    temperature: float = 2.0,
    fixed_player_idx: int | None = None,
    graph_scaler: dict | None = None,
    verify_slicing: bool = True,
) -> Tuple[List[int], OutcomePrediction]:
    n_pool = len(cache.home.squad_player_ids)
    k = 11

    pop = [_repair_unique_with_lock([], n_pool=n_pool, k=k, fixed_idx=fixed_player_idx) for _ in range(pop_size)]
    best_g = pop[0]
    best_f = -1.0
    best_pred = OutcomePrediction(win_prob=0.0, draw_prob=0.0, loss_prob=1.0, expected_points=0.0)

    slice_shapes_verified = False
    lineup_signature_verified = False
    previous_signature: tuple[float, float] | None = None

    for gen in range(1, generations + 1):
        scored = []
        for g in pop:
            data = build_fast_heterodata(cache, home_sel=g, away_sel=away_sel)
            _apply_scaler_to_graph_inplace(data, graph_scaler)

            if verify_slicing and not slice_shapes_verified:
                _validate_subgraph_slice_shapes(data, expected_home_n=k, expected_away_n=len(away_sel))
                print(
                    "[CHECK] subgraph slicing ok "
                    f"home_x={tuple(data['home_team'].x.shape)} "
                    f"away_x={tuple(data['away_team'].x.shape)} "
                    f"home_io_edges={int(data[('home_team', 'passes_to', 'home_team')].edge_index.size(1))}"
                )
                slice_shapes_verified = True

            if verify_slicing and not lineup_signature_verified:
                sig = _lineup_signature(data)
                if previous_signature is None:
                    previous_signature = sig
                elif sig != previous_signature:
                    print(f"[CHECK] lineup-dependent tensors change detected: prev={previous_signature} curr={sig}")
                    lineup_signature_verified = True

            pred = _predict_home_outcome_probs(model, data, device=device, temperature=temperature)
            f = float(pred.expected_points)
            scored.append((f, g, pred))
            if f > best_f:
                best_f, best_g, best_pred = f, g[:], pred

        if verify_slicing and gen == 1 and not lineup_signature_verified:
            print("[WARN] lineup signature did not differ inside first generation; verify squad diversity and cached features")
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [g[:] for _, g, _ in scored[:elite_size]]

        if gen % 10 == 0 or gen == 1 or gen == generations:
            top_pred = scored[0][2]
            print(
                f"[GA] gen={gen:03d} best_epts={scored[0][0]:.6f} "
                f"mean_epts={float(np.mean([s for s, _, _ in scored])):.6f} "
                f"Pwin={top_pred.win_prob:.4f} Pdraw={top_pred.draw_prob:.4f} Ploss={top_pred.loss_prob:.4f}"
            )

        next_pop = elites[:]
        while len(next_pop) < pop_size:
            p1 = random.choice(scored[: max(3, pop_size // 3)])[1]
            p2 = random.choice(scored[: max(3, pop_size // 3)])[1]
            c = _crossover_with_lock(p1, p2, n_pool=n_pool, k=k, fixed_idx=fixed_player_idx)
            c = _mutate_with_lock(c, n_pool=n_pool, p_mut=mutation_p, fixed_idx=fixed_player_idx)
            next_pop.append(c)
        pop = next_pop

    return best_g, best_pred


# ----------------------------
# Main
# 기능: _player_name_map는 연산 pd.read_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다.
# 데이터 입출력:
#   - Input: players_csv: Path
#   - Output: Dict[int, str]
def _player_name_map(players_csv: Path) -> Dict[int, str]:
    p = pd.read_csv(players_csv)
    out = {}
    for r in p.itertuples(index=False):
        rid = _to_int(getattr(r, "wyId", None))
        if rid is None:
            continue
        fn = str(getattr(r, "firstName", "")).strip()
        ln = str(getattr(r, "lastName", "")).strip()
        short = str(getattr(r, "shortName", "")).strip()
        name = short if short else (fn + " " + ln).strip()
        out[int(rid)] = name if name else str(rid)
    return out
# 기능: main는 연산 pd.read_csv/pd.to_datetime/softmax/GA 세대 반복, temperature가 반영된 outcome 확률로 expected_points를 계산한다을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: Phase6에서 캐시된 OFF/DEF/IO/ID 텐서를 재사용하며 GA 탐색으로 expected_points를 최대화하기 위해 필요하다. 특히 temperature가 반영된 outcome 확률로 expected_points를 계산한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: None
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 GA lineup optimization using trained GNN")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--league-mode", type=str, default="england", choices=["england", "non_england"])
    parser.add_argument("--matches-csv", type=Path, default=DATA_DIR / "archive/matches_England.csv")
    parser.add_argument("--teams-csv", type=Path, default=DATA_DIR / "archive/teams.csv")
    parser.add_argument("--players-csv", type=Path, default=DATA_DIR / "archive/players.csv")
    parser.add_argument("--player-games-csv", type=Path, default=DATA_DIR / "archive/player_games.csv")

    parser.add_argument("--home-team", type=str, default="Manchester United")
    parser.add_argument("--away-team", type=str, default="Chelsea")
    parser.add_argument("--hypothetical-match-id", type=int, default=999999)
    parser.add_argument("--match-date", type=str, default="2026-04-18 15:00:00+00:00")

    parser.add_argument("--squad-size", type=int, default=18)
    parser.add_argument("--pop-size", type=int, default=96)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--elite-size", type=int, default=16)
    parser.add_argument("--mutation-p", "--mutation-rate", dest="mutation_p", type=float, default=0.18)
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for logits scaling before softmax")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model-def",
        type=Path,
        default=PROJECT_ROOT / "proposed_spatial_gnn_ga/train_gnn_phase5.py",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=DATA_DIR / "phase_5_lineup/data/gnn_phase5/hetero_edge_gat_win_epoch10_l2.pt",
    )
    parser.add_argument(
        "--scaler-pt",
        type=Path,
        default=None,
        help="Optional feature_scaler.pt override. If omitted, uses feature_scaler embedded in model checkpoint when available.",
    )
    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teams = pd.read_csv(args.teams_csv)
    home_team_id = _resolve_team_id(teams, args.home_team)
    away_team_id = _resolve_team_id(teams, args.away_team)

    asof_time = pd.to_datetime(args.match_date, utc=True)
    matches = _prepare_matches(args.matches_csv)
    player_games = pd.read_csv(args.player_games_csv)
    events = _load_event_tables(args.data_root, league_mode=args.league_mode)

    print(f"[INFO] match_id={args.hypothetical_match_id} home={args.home_team}({home_team_id}) away={args.away_team}({away_team_id})")
    print("[INFO] building caches...")
    cache = build_matchup_cache(
        matches_df=matches,
        player_games_df=player_games,
        events=events,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        match_id=int(args.hypothetical_match_id),
        asof_time=asof_time,
        squad_size=int(args.squad_size),
    )
    pos_map = _build_player_position_map(args.players_csv)
    fixed_home_gk_idx = resolve_fixed_gk_index(
        team_cache=cache.home,
        preferred_player_ids=cache.home.squad_player_ids,
        player_position_map=pos_map,
    )
    fixed_home_gk_player_id = int(cache.home.squad_player_ids[fixed_home_gk_idx])
    print(
        f"[INFO] cache ready: home_squad={len(cache.home.squad_player_ids)} away_squad={len(cache.away.squad_player_ids)} "
        f"home_io={tuple(cache.home_io.shape)} away_io={tuple(cache.away_io.shape)}"
    )
    print(f"[INFO] fixed home GK index={fixed_home_gk_idx} player_id={fixed_home_gk_player_id}")

    scaler = _load_feature_scaler(args.scaler_pt)
    model, ckpt_scaler_payload = load_trained_model(args.model_def, args.model_ckpt, device=device)

    if scaler is None and ckpt_scaler_payload is not None:
        scaler = _parse_feature_scaler_payload(ckpt_scaler_payload)
        print("[INFO] z-score scaler loaded from model checkpoint payload")

    if scaler is not None:
        if args.scaler_pt is not None:
            print(f"[INFO] z-score scaler loaded for per-graph application: {args.scaler_pt}")
        else:
            print("[INFO] z-score scaler loaded from checkpoint payload for per-graph application")

    away_sel = _default_away_starting11(cache.away)
    best_home_sel, best_pred = run_ga_optimize_home(
        model=model,
        cache=cache,
        away_sel=away_sel,
        device=device,
        pop_size=int(args.pop_size),
        generations=int(args.generations),
        elite_size=int(args.elite_size),
        mutation_p=float(args.mutation_p),
        temperature=float(args.temperature),
        fixed_player_idx=int(fixed_home_gk_idx),
        graph_scaler=scaler,
    )

    name_map = _player_name_map(args.players_csv)
    best_home_ids = [cache.home.squad_player_ids[i] for i in best_home_sel]
    away_ids = [cache.away.squad_player_ids[i] for i in away_sel]

    print("\n===== Phase 6 Result =====")
    print(f"Hypothetical Match ID: {args.hypothetical_match_id}")
    print(f"Home Team: {args.home_team} ({home_team_id})")
    print(f"Away Team: {args.away_team} ({away_team_id})")
    print(f"Temperature: {float(args.temperature):.3f}")
    print(f"Max Home Expected Points: {best_pred.expected_points:.6f}")
    print(
        "Predicted Outcome Probabilities "
        f"P(win)={best_pred.win_prob:.6f} "
        f"P(draw)={best_pred.draw_prob:.6f} "
        f"P(loss)={best_pred.loss_prob:.6f}"
    )

    print("\n[Best Home XI]")
    for i, pid in enumerate(best_home_ids, start=1):
        print(f"{i:02d}. {name_map.get(pid, str(pid))} ({pid})")

    print("\n[Fixed Away XI]")
    for i, pid in enumerate(away_ids, start=1):
        print(f"{i:02d}. {name_map.get(pid, str(pid))} ({pid})")


if __name__ == "__main__":
    main()
