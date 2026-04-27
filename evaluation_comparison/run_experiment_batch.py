#!/usr/bin/env python3
from __future__ import annotations

"""Run full EPL 17/18 lineup hit-rate experiment: GNN vs Team-Builder.

What this script does:
1) Iterates all matches in matches_England.csv (default: 380 matches).
2) Auto-extracts target-match opponent starting XI.
3) Runs Team-Builder Phase 5 with Equation 5 opponent-XI conditioning.
4) Runs GNN Phase 6 GA lineup optimization for both sides of each match.
5) Saves prediction tables and executes evaluate_hitrate_gnn_vs_teambuilder.py.
"""

import argparse
import ast
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"
PROPOSED_DIR = PROJECT_ROOT / "proposed_spatial_gnn_ga"
if str(PROPOSED_DIR) not in sys.path:
    sys.path.insert(0, str(PROPOSED_DIR))

import optimize_lineup_ga_phase6 as p6
# 기능: _to_int는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
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
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: v: object
#   - Output: list
def _safe_eval_list(v: object) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    return []
# 기능: _extract_player_ids는 컬럼 'playerId'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: raw: object, max_take: int
#   - Output: List[int]
def _extract_player_ids(raw: object, max_take: int = 11) -> List[int]:
    arr = _safe_eval_list(raw)
    out: List[int] = []
    for r in arr:
        pid = _to_int(r.get("playerId") if isinstance(r, dict) else r)
        if pid is None:
            continue
        if pid not in out:
            out.append(pid)
        if max_take is not None and len(out) >= int(max_take):
            break
    return out
# 기능: _extract_substitution_player_ids는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: raw: object
#   - Output: List[int]
def _extract_substitution_player_ids(raw: object) -> List[int]:
    arr = _safe_eval_list(raw)
    out: List[int] = []
    for r in arr:
        if isinstance(r, dict):
            for key in ("playerIn", "playerId"):
                pid = _to_int(r.get(key))
                if pid is None:
                    continue
                if pid not in out:
                    out.append(pid)
        else:
            pid = _to_int(r)
            if pid is None:
                continue
            if pid not in out:
                out.append(pid)
    return out
# 기능: _extract_players_from_formation_blob는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: raw_formation: object, key: str
#   - Output: List[int]
def _extract_players_from_formation_blob(raw_formation: object, key: str) -> List[int]:
    parsed = _safe_literal(raw_formation)
    if not isinstance(parsed, dict):
        return []
    return _extract_player_ids(parsed.get(key), max_take=None)
# 기능: _extract_matchday_squad는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: row: dict, side_prefix: str
#   - Output: List[int]
def _extract_matchday_squad(row: dict, side_prefix: str) -> List[int]:
    # Prefer explicit columns, then fallback to nested formation blob.
    lineup = _extract_player_ids(row.get(f"{side_prefix}.formation.lineup"), max_take=11)
    if not lineup:
        lineup = _extract_players_from_formation_blob(row.get(f"{side_prefix}.formation"), "lineup")[:11]

    bench = _extract_player_ids(row.get(f"{side_prefix}.formation.bench"), max_take=None)
    if not bench:
        bench = _extract_players_from_formation_blob(row.get(f"{side_prefix}.formation"), "bench")

    # Last-resort fallback when bench is unavailable in source data.
    if not bench:
        bench = _extract_substitution_player_ids(row.get(f"{side_prefix}.formation.substitutions"))

    squad: List[int] = []
    for pid in lineup + bench:
        if pid not in squad:
            squad.append(pid)
    return squad
# 기능: _parse_role_code는 컬럼 'code2', 'name'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
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
# 기능: _safe_literal는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: value
#   - Output: 코드 내부 return 표현식
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
# 기능: _build_player_position_map는 연산 pd.read_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: players_csv: Path
#   - Output: Dict[int, str]
def _build_player_position_map(players_csv: Path) -> Dict[int, str]:
    players = pd.read_csv(players_csv)
    required = {"wyId", "role"}
    if not required.issubset(players.columns):
        raise RuntimeError(f"players csv missing columns for formation inference: {sorted(required - set(players.columns))}")

    pos_map: Dict[int, str] = {}
    for r in players[["wyId", "role"]].itertuples(index=False):
        pid = _to_int(r.wyId)
        if pid is None:
            continue
        code = _parse_role_code(r.role)
        if code in {"GK", "DF", "MF", "FW"}:
            pos_map[int(pid)] = str(code)
    return pos_map
# 기능: _infer_formation_from_lineup는 컬럼 'GK', 'DF', 'MF', 'FW'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: lineup_ids: Sequence[int], player_position_map: Dict[int, str], fallback: str
#   - Output: str
def _infer_formation_from_lineup(
    lineup_ids: Sequence[int],
    player_position_map: Dict[int, str],
    fallback: str,
) -> str:
    if len(lineup_ids) != 11:
        raise ValueError(f"lineup size must be 11 for formation inference, got {len(lineup_ids)}")

    counts = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
    unknown = 0
    for pid in lineup_ids:
        pos = player_position_map.get(int(pid))
        if pos in counts:
            counts[pos] += 1
        else:
            unknown += 1

    if counts["GK"] != 1:
        if fallback:
            return fallback
        raise ValueError(f"invalid GK count from lineup: {counts['GK']}")

    outfield = counts["DF"] + counts["MF"] + counts["FW"]
    missing = 10 - outfield
    if missing > 0 and unknown >= missing:
        counts["MF"] += missing
        unknown -= missing

    if counts["DF"] + counts["MF"] + counts["FW"] != 10:
        if fallback:
            return fallback
        raise ValueError(
            "failed to infer outfield formation from lineup "
            f"(DF={counts['DF']}, MF={counts['MF']}, FW={counts['FW']}, unknown={unknown})"
        )

    return f"{counts['DF']}-{counts['MF']}-{counts['FW']}"
# 기능: 매치 CSV에서 'wyId','team1.teamId','team2.teamId','dateutc/date'를 정규화해 시간순 배치 실행이 가능한 테이블로 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: matches_csv: Path
#   - Output: pd.DataFrame
def _prepare_matches(matches_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(matches_csv)
    required = {
        "wyId",
        "team1.teamId",
        "team2.teamId",
        "team1.formation.lineup",
        "team2.formation.lineup",
    }
    miss = sorted(required - set(df.columns))
    if miss:
        raise ValueError(f"matches csv missing required columns: {miss}")

    has_bench_cols = {"team1.formation.bench", "team2.formation.bench"}.issubset(df.columns)
    has_formation_blob = {"team1.formation", "team2.formation"}.issubset(df.columns)
    if not (has_bench_cols or has_formation_blob):
        raise ValueError(
            "matches csv must include bench information via team*.formation.bench or team*.formation"
        )

    if "dateutc" in df.columns:
        df["match_time"] = pd.to_datetime(df["dateutc"], errors="coerce", utc=True)
    elif "date" in df.columns:
        df["match_time"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        raise ValueError("matches csv requires dateutc or date column")

    for c in ["wyId", "team1.teamId", "team2.teamId"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["wyId", "team1.teamId", "team2.teamId", "match_time"]).copy()
    df["wyId"] = df["wyId"].astype(int)
    df["team1.teamId"] = df["team1.teamId"].astype(int)
    df["team2.teamId"] = df["team2.teamId"].astype(int)

    return df.sort_values(["match_time", "wyId"]).reset_index(drop=True)
# 기능: _import_module_from_path는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: module_path: Path, module_name: str
#   - Output: 코드 내부 return 표현식
def _import_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
# 기능: _ids_to_indices는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: ids: Sequence[int], pool: Sequence[int], fallback: Sequence[int]
#   - Output: List[int]
def _ids_to_indices(ids: Sequence[int], pool: Sequence[int], fallback: Sequence[int]) -> List[int]:
    pos = {int(pid): i for i, pid in enumerate(pool)}
    out: List[int] = []
    used = set()

    for pid in ids:
        idx = pos.get(int(pid))
        if idx is None or idx in used:
            continue
        out.append(int(idx))
        used.add(int(idx))
        if len(out) >= 11:
            break

    for idx in fallback:
        i = int(idx)
        if i < 0 or i >= len(pool) or i in used:
            continue
        out.append(i)
        used.add(i)
        if len(out) >= 11:
            break

    if len(out) < 11:
        for i in range(len(pool)):
            if i not in used:
                out.append(i)
                used.add(i)
                if len(out) >= 11:
                    break

    return out[:11]
# 기능: _repair_unique는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: genome: List[int], n_pool: int, k: int, fixed_idx: int | None
#   - Output: List[int]
def _repair_unique(genome: List[int], n_pool: int, k: int, fixed_idx: int | None = None) -> List[int]:
    out: List[int] = []
    used = set()

    if fixed_idx is not None:
        fix = int(fixed_idx)
        if not (0 <= fix < n_pool):
            raise ValueError(f"fixed_idx out of range: {fix} not in [0, {n_pool})")
        out.append(fix)
        used.add(fix)

    for g in genome:
        gi = int(g)
        if 0 <= gi < n_pool and gi not in used:
            out.append(gi)
            used.add(gi)
        if len(out) >= k:
            break
    while len(out) < k:
        c = random.randrange(n_pool)
        if c not in used:
            out.append(c)
            used.add(c)
    return out
# 기능: _crossover는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: p1: List[int], p2: List[int], n_pool: int, k: int, fixed_idx: int | None
#   - Output: List[int]
def _crossover(p1: List[int], p2: List[int], n_pool: int, k: int, fixed_idx: int | None = None) -> List[int]:
    cut = random.randint(1, k - 1)
    child = p1[:cut] + p2[cut:]
    return _repair_unique(child, n_pool=n_pool, k=k, fixed_idx=fixed_idx)
# 기능: _mutate는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: g: List[int], n_pool: int, p_mut: float, fixed_idx: int | None
#   - Output: List[int]
def _mutate(g: List[int], n_pool: int, p_mut: float, fixed_idx: int | None = None) -> List[int]:
    out = g[:]
    if random.random() < float(p_mut):
        mutable_positions = [i for i, idx in enumerate(out) if fixed_idx is None or int(idx) != int(fixed_idx)]
        if not mutable_positions:
            return _repair_unique(out, n_pool=n_pool, k=len(g), fixed_idx=fixed_idx)
        pos = random.choice(mutable_positions)
        used = set(out)
        cand = random.randrange(n_pool)
        tries = 0
        while cand in used and tries < 100:
            cand = random.randrange(n_pool)
            tries += 1
        out[pos] = cand
    return _repair_unique(out, n_pool=n_pool, k=len(g), fixed_idx=fixed_idx)
# 기능: _run_ga_optimize_home_quiet는 연산 GA 세대 반복, temperature가 반영된 outcome 확률로 expected_points를 계산한다을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다. 특히 temperature가 반영된 outcome 확률로 expected_points를 계산한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: model, cache: p6.MatchupCache, away_sel: List[int], global_features: torch.Tensor, device: torch.device, pop_size: int, ...
#   - Output: Tuple[List[int], p6.OutcomePrediction]
def _run_ga_optimize_home_quiet(
    model,
    cache: p6.MatchupCache,
    away_sel: List[int],
    global_features: torch.Tensor,
    device: torch.device,
    pop_size: int,
    generations: int,
    elite_size: int,
    mutation_p: float,
    temperature: float,
    early_stop_patience: int,
    fixed_home_idx: int | None,
) -> Tuple[List[int], p6.OutcomePrediction]:
    n_pool = len(cache.home.squad_player_ids)
    k = 11

    pop = [_repair_unique([], n_pool=n_pool, k=k, fixed_idx=fixed_home_idx) for _ in range(int(pop_size))]
    best_g = pop[0]
    best_pred = p6.OutcomePrediction(win_prob=0.0, draw_prob=0.0, loss_prob=1.0, expected_points=0.0)
    best_fit = -1.0
    stale = 0

    for _ in range(int(generations)):
        improved = False
        scored = []
        for g in pop:
            data = p6.build_fast_heterodata(cache, home_sel=g, away_sel=away_sel, global_features=global_features)
            pred = p6._predict_home_outcome_probs(model, data, device=device, temperature=float(temperature))
            fit = float(pred.expected_points)
            scored.append((fit, g, pred))
            if fit > best_fit + 1e-12:
                best_fit = fit
                best_g = g[:]
                best_pred = pred
                improved = True

        if improved:
            stale = 0
        else:
            stale += 1

        if stale >= int(max(1, early_stop_patience)):
            break

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [g[:] for _, g, _ in scored[: int(max(1, elite_size))]]

        next_pop = elites[:]
        while len(next_pop) < int(pop_size):
            p1 = random.choice(scored[: max(3, int(pop_size) // 3)])[1]
            p2 = random.choice(scored[: max(3, int(pop_size) // 3)])[1]
            c = _crossover(p1, p2, n_pool=n_pool, k=k, fixed_idx=fixed_home_idx)
            c = _mutate(c, n_pool=n_pool, p_mut=float(mutation_p), fixed_idx=fixed_home_idx)
            next_pop.append(c)
        pop = next_pop

    return best_g, best_pred
# 기능: _run_ga_optimize_away_quiet는 연산 GA 세대 반복, temperature가 반영된 outcome 확률로 expected_points를 계산한다을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다. 특히 temperature가 반영된 outcome 확률로 expected_points를 계산한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: model, cache: p6.MatchupCache, home_sel: List[int], global_features: torch.Tensor, device: torch.device, pop_size: int, ...
#   - Output: Tuple[List[int], p6.OutcomePrediction]
def _run_ga_optimize_away_quiet(
    model,
    cache: p6.MatchupCache,
    home_sel: List[int],
    global_features: torch.Tensor,
    device: torch.device,
    pop_size: int,
    generations: int,
    elite_size: int,
    mutation_p: float,
    temperature: float,
    early_stop_patience: int,
    fixed_away_idx: int | None,
) -> Tuple[List[int], p6.OutcomePrediction]:
    n_pool = len(cache.away.squad_player_ids)
    k = 11

    pop = [_repair_unique([], n_pool=n_pool, k=k, fixed_idx=fixed_away_idx) for _ in range(int(pop_size))]
    best_g = pop[0]
    best_pred = p6.OutcomePrediction(win_prob=0.0, draw_prob=0.0, loss_prob=1.0, expected_points=0.0)
    best_fit = -1.0
    stale = 0

    for _ in range(int(generations)):
        improved = False
        scored = []
        for g in pop:
            data = p6.build_fast_heterodata(cache, home_sel=home_sel, away_sel=g, global_features=global_features)
            home_pred = p6._predict_home_outcome_probs(model, data, device=device, temperature=float(temperature))
            away_pred = p6._to_away_outcome_prediction(home_pred)
            fit = float(away_pred.expected_points)
            scored.append((fit, g, away_pred))
            if fit > best_fit + 1e-12:
                best_fit = fit
                best_g = g[:]
                best_pred = away_pred
                improved = True

        if improved:
            stale = 0
        else:
            stale += 1

        if stale >= int(max(1, early_stop_patience)):
            break

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [g[:] for _, g, _ in scored[: int(max(1, elite_size))]]

        next_pop = elites[:]
        while len(next_pop) < int(pop_size):
            p1 = random.choice(scored[: max(3, int(pop_size) // 3)])[1]
            p2 = random.choice(scored[: max(3, int(pop_size) // 3)])[1]
            c = _crossover(p1, p2, n_pool=n_pool, k=k, fixed_idx=fixed_away_idx)
            c = _mutate(c, n_pool=n_pool, p_mut=float(mutation_p), fixed_idx=fixed_away_idx)
            next_pop.append(c)
        pop = next_pop

    return best_g, best_pred
# 기능: _clear_teambuilder_outputs는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: run_dir: Path
#   - Output: None
def _clear_teambuilder_outputs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "lineup_selected.csv",
        "lineup_selected.parquet",
        "lineup_selected_named.csv",
        "lineup_selected_io_pairs.csv",
        "lineup_selected_io_pairs.parquet",
        "lineup_selected_id_pairs_vs_opponent.csv",
        "lineup_selected_id_pairs_vs_opponent.parquet",
        "lineup_optimization_summary.csv",
        "lineup_optimization_summary.parquet",
    ]:
        p = run_dir / name
        if p.exists():
            p.unlink()
# 기능: _run_teambuilder_once는 컬럼 'player_id', 연산 pd.read_csv, lambda_vi/lambda_io/lambda_id 선형 결합 목적함수를 사용한다을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다. 특히 엔티티 키(game_id/team_id/player_id) 일관성; lambda_vi/lambda_io/lambda_id 선형 결합 목적함수를 사용한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: tb_mod, run_dir: Path, synergy_dir: Path, tactics_dir: Path, archive_dir: Path, team_id: int, ...
#   - Output: List[int]
def _run_teambuilder_once(
    tb_mod,
    run_dir: Path,
    synergy_dir: Path,
    tactics_dir: Path,
    archive_dir: Path,
    team_id: int,
    formation: str,
    lambda_vi: float,
    lambda_io: float,
    lambda_id: float,
    solver_time_limit: int,
    io_source: str,
    id_source: str,
    lambda_csv: Path | None,
    lambda_scaler_stats: Path | None,
    opponent_lineup_ids: List[int],
    opponent_team_id: int,
    available_player_ids: List[int],
) -> List[int]:
    _clear_teambuilder_outputs(run_dir)

    tb_mod.run_phase5(
        synergy_dir=synergy_dir,
        tactics_dir=tactics_dir,
        archive_dir=archive_dir,
        output_dir=run_dir,
        team_id=int(team_id),
        formation=formation,
        lambda_vi=float(lambda_vi),
        lambda_io=float(lambda_io),
        lambda_id=float(lambda_id),
        attack_weight=1.0,
        defense_weight=1.0,
        attack_clusters=[],
        defense_clusters=[],
        attack_keywords=[],
        defense_keywords=[],
        solver_time_limit=int(solver_time_limit),
        io_source=io_source,
        id_source=id_source,
        lambda_csv=lambda_csv,
        lambda_scaler_stats=lambda_scaler_stats,
        opponent_lineup_ids=opponent_lineup_ids,
        opponent_team_id=int(opponent_team_id),
        available_player_ids=available_player_ids,
    )

    out_csv = run_dir / "lineup_selected.csv"
    if not out_csv.exists():
        raise RuntimeError(f"Team-Builder output missing: {out_csv}")

    pred = pd.read_csv(out_csv)
    if "player_id" not in pred.columns:
        raise RuntimeError("lineup_selected.csv missing player_id column")

    ids = pd.to_numeric(pred["player_id"], errors="coerce").dropna().astype(int).tolist()
    uniq: List[int] = []
    for pid in ids:
        if pid not in uniq:
            uniq.append(pid)
        if len(uniq) >= 11:
            break

    if len(uniq) != 11:
        raise RuntimeError(f"Team-Builder predicted lineup size is {len(uniq)} (expected 11)")
    return uniq
# 기능: _team_name_map는 연산 pd.read_csv을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: teams_csv: Path
#   - Output: Dict[int, str]
def _team_name_map(teams_csv: Path) -> Dict[int, str]:
    df = pd.read_csv(teams_csv)
    if not {"wyId", "name"}.issubset(df.columns):
        return {}
    out: Dict[int, str] = {}
    for r in df[["wyId", "name"]].itertuples(index=False):
        tid = _to_int(r.wyId)
        if tid is None:
            continue
        out[int(tid)] = str(r.name)
    return out
# 기능: 배치 입력 경기들을 순회하며 GNN/Team-Builder 양쪽 예측 CSV 및 오류 CSV를 저장하고 필요시 외부 evaluator를 서브프로세스로 호출한다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다. 특히 경기 키('wyId')와 시점 컬럼('dateutc'/'match_time') 정합성; lambda_vi/lambda_io/lambda_id 선형 결합 목적함수를 사용한다를 고정 규칙으로 유지한다.
# 데이터 입출력:
#   - Input: args: argparse.Namespace
#   - Output: None
def run(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matches = _prepare_matches(args.matches_csv)
    history_matches = _prepare_matches(args.history_matches_csv) if args.history_matches_csv is not None else matches
    
    # Optional: filter by home/away team names if specified
    filter_home_team = getattr(args, 'filter_home_team', None)
    filter_away_team = getattr(args, 'filter_away_team', None)
    if filter_home_team or filter_away_team:
        teams_df = pd.read_csv(args.teams_csv)
        team_name_to_id = dict(zip(teams_df["name"], teams_df["wyId"]))

        if filter_home_team:
            home_team_id = team_name_to_id.get(filter_home_team)
            if home_team_id is None:
                print(f"[WARN] home team not found: '{filter_home_team}'")
            else:
                matches = matches[matches["team1.teamId"] == int(home_team_id)].copy()
                print(f"[INFO] filtered to home team '{filter_home_team}' ({home_team_id}): {len(matches)} matches")

        if filter_away_team:
            away_team_id = team_name_to_id.get(filter_away_team)
            if away_team_id is None:
                print(f"[WARN] away team not found: '{filter_away_team}'")
            else:
                matches = matches[matches["team2.teamId"] == int(away_team_id)].copy()
                print(f"[INFO] filtered to away team '{filter_away_team}' ({away_team_id}): {len(matches)} matches")
    
    if int(args.max_matches) > 0:
        matches = matches.head(int(args.max_matches)).copy()

    team_name_map = _team_name_map(args.teams_csv)
    player_position_map = _build_player_position_map(args.players_csv)
    player_games = pd.read_csv(args.player_games_csv)
    events = p6._load_event_tables(args.data_root, league_mode=args.league_mode)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] loading GNN model on device={device}")
    model, ckpt_scaler_payload = p6.load_trained_model(args.model_def, args.model_ckpt, device=device)
    scaler = p6._load_feature_scaler(args.scaler_pt)
    if scaler is None and ckpt_scaler_payload is not None:
        scaler = p6._parse_feature_scaler_payload(ckpt_scaler_payload)
        print("[INFO] using feature scaler from model checkpoint")

    tb_mod = None
    tb_run_home = output_dir / "_tmp_teambuilder_home"
    tb_run_away = output_dir / "_tmp_teambuilder_away"
    if not args.skip_teambuilder:
        tb_mod = _import_module_from_path(args.tb_script, module_name="optimize_lineup_phase5_mod")

    gnn_rows: List[dict] = []
    tb_rows: List[dict] = []
    err_rows: List[dict] = []

    total_matches = len(matches)
    print(f"[INFO] start batch: matches={total_matches} (pred rows target={total_matches * 2})")

    for i, row in enumerate(matches.to_dict(orient="records"), start=1):
        game_id = int(row["wyId"])
        home_id = int(row["team1.teamId"])
        away_id = int(row["team2.teamId"])
        match_time = pd.to_datetime(row["match_time"], utc=True)

        home_actual = _extract_player_ids(row.get("team1.formation.lineup"), max_take=11)
        away_actual = _extract_player_ids(row.get("team2.formation.lineup"), max_take=11)
        home_matchday = _extract_matchday_squad(row=row, side_prefix="team1")
        away_matchday = _extract_matchday_squad(row=row, side_prefix="team2")
        if len(home_actual) != 11 or len(away_actual) != 11:
            err_rows.append(
                {
                    "game_id": game_id,
                    "team_id": "",
                    "model": "both",
                    "stage": "lineup_parse",
                    "error": f"invalid lineup size: home={len(home_actual)} away={len(away_actual)}",
                }
            )
            continue

        if len(home_matchday) < 11 or len(away_matchday) < 11:
            err_rows.append(
                {
                    "game_id": game_id,
                    "team_id": "",
                    "model": "both",
                    "stage": "matchday_squad_parse",
                    "error": f"invalid matchday squad size: home={len(home_matchday)} away={len(away_matchday)}",
                }
            )
            continue

        home_tb_formation = _infer_formation_from_lineup(
            lineup_ids=home_actual,
            player_position_map=player_position_map,
            fallback=str(args.tb_formation),
        )
        away_tb_formation = _infer_formation_from_lineup(
            lineup_ids=away_actual,
            player_position_map=player_position_map,
            fallback=str(args.tb_formation),
        )

        # ---------- GNN ----------
        if not args.skip_gnn:
            try:
                cache = p6.build_matchup_cache(
                    matches_df=history_matches,
                    player_games_df=player_games,
                    events=events,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    match_id=game_id,
                    asof_time=match_time,
                    squad_size=int(args.gnn_squad_size),
                    available_home_player_ids=home_matchday,
                    available_away_player_ids=away_matchday,
                )
                p6._apply_scaler_to_cache_inplace(cache, scaler)
                gf = p6._build_global_features(match_time)

                fixed_home_gk_idx = p6.resolve_fixed_gk_index(
                    team_cache=cache.home,
                    preferred_player_ids=home_matchday,
                    player_position_map=player_position_map,
                )
                fixed_away_gk_idx = p6.resolve_fixed_gk_index(
                    team_cache=cache.away,
                    preferred_player_ids=away_matchday,
                    player_position_map=player_position_map,
                )

                away_fallback = p6._default_away_starting11(cache.away)
                fixed_away_sel = _ids_to_indices(away_actual, cache.away.squad_player_ids, away_fallback)
                best_home_sel, home_pred = _run_ga_optimize_home_quiet(
                    model=model,
                    cache=cache,
                    away_sel=fixed_away_sel,
                    global_features=gf,
                    device=device,
                    pop_size=int(args.gnn_pop_size),
                    generations=int(args.gnn_generations),
                    elite_size=int(args.gnn_elite_size),
                    mutation_p=float(args.gnn_mutation_p),
                    temperature=float(args.gnn_temperature),
                    early_stop_patience=int(args.gnn_early_stop_patience),
                    fixed_home_idx=int(fixed_home_gk_idx),
                )
                home_pred_ids = [int(cache.home.squad_player_ids[idx]) for idx in best_home_sel]

                home_fallback = p6._default_away_starting11(cache.home)
                fixed_home_sel = _ids_to_indices(home_actual, cache.home.squad_player_ids, home_fallback)
                best_away_sel, away_pred = _run_ga_optimize_away_quiet(
                    model=model,
                    cache=cache,
                    home_sel=fixed_home_sel,
                    global_features=gf,
                    device=device,
                    pop_size=int(args.gnn_pop_size),
                    generations=int(args.gnn_generations),
                    elite_size=int(args.gnn_elite_size),
                    mutation_p=float(args.gnn_mutation_p),
                    temperature=float(args.gnn_temperature),
                    early_stop_patience=int(args.gnn_early_stop_patience),
                    fixed_away_idx=int(fixed_away_gk_idx),
                )
                away_pred_ids = [int(cache.away.squad_player_ids[idx]) for idx in best_away_sel]

                gnn_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": home_id,
                        "lineup_ids": "|".join(map(str, home_pred_ids)),
                        "match_time": str(match_time),
                        "opponent_team_id": away_id,
                        "opponent_team_name": team_name_map.get(away_id, str(away_id)),
                        "side": "home",
                        "win_prob": float(home_pred.win_prob),
                        "draw_prob": float(home_pred.draw_prob),
                        "loss_prob": float(home_pred.loss_prob),
                        "expected_points": float(home_pred.expected_points),
                    }
                )
                gnn_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": away_id,
                        "lineup_ids": "|".join(map(str, away_pred_ids)),
                        "match_time": str(match_time),
                        "opponent_team_id": home_id,
                        "opponent_team_name": team_name_map.get(home_id, str(home_id)),
                        "side": "away",
                        "win_prob": float(away_pred.win_prob),
                        "draw_prob": float(away_pred.draw_prob),
                        "loss_prob": float(away_pred.loss_prob),
                        "expected_points": float(away_pred.expected_points),
                    }
                )
            except Exception as e:
                err_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": "",
                        "model": "GNN",
                        "stage": "optimize",
                        "error": str(e),
                    }
                )

        # ---------- Team-Builder ----------
        if tb_mod is not None:
            try:
                pred_home = _run_teambuilder_once(
                    tb_mod=tb_mod,
                    run_dir=tb_run_home,
                    synergy_dir=args.tb_synergy_dir,
                    tactics_dir=args.tb_tactics_dir,
                    archive_dir=args.tb_archive_dir,
                    team_id=home_id,
                    formation=home_tb_formation,
                    lambda_vi=float(args.tb_lambda_vi),
                    lambda_io=float(args.tb_lambda_io),
                    lambda_id=float(args.tb_lambda_id),
                    solver_time_limit=int(args.tb_solver_time_limit),
                    io_source=args.tb_io_source,
                    id_source=args.tb_id_source,
                    lambda_csv=args.tb_lambda_csv,
                    lambda_scaler_stats=args.tb_lambda_scaler_stats,
                    opponent_lineup_ids=away_actual,
                    opponent_team_id=away_id,
                    available_player_ids=home_matchday,
                )
                tb_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": home_id,
                        "lineup_ids": "|".join(map(str, pred_home)),
                        "match_time": str(match_time),
                        "opponent_team_id": away_id,
                        "opponent_team_name": team_name_map.get(away_id, str(away_id)),
                        "side": "home",
                    }
                )
            except Exception as e:
                err_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": home_id,
                        "model": "Team-Builder",
                        "stage": "optimize_home",
                        "error": str(e),
                    }
                )

            try:
                pred_away = _run_teambuilder_once(
                    tb_mod=tb_mod,
                    run_dir=tb_run_away,
                    synergy_dir=args.tb_synergy_dir,
                    tactics_dir=args.tb_tactics_dir,
                    archive_dir=args.tb_archive_dir,
                    team_id=away_id,
                    formation=away_tb_formation,
                    lambda_vi=float(args.tb_lambda_vi),
                    lambda_io=float(args.tb_lambda_io),
                    lambda_id=float(args.tb_lambda_id),
                    solver_time_limit=int(args.tb_solver_time_limit),
                    io_source=args.tb_io_source,
                    id_source=args.tb_id_source,
                    lambda_csv=args.tb_lambda_csv,
                    lambda_scaler_stats=args.tb_lambda_scaler_stats,
                    opponent_lineup_ids=home_actual,
                    opponent_team_id=home_id,
                    available_player_ids=away_matchday,
                )
                tb_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": away_id,
                        "lineup_ids": "|".join(map(str, pred_away)),
                        "match_time": str(match_time),
                        "opponent_team_id": home_id,
                        "opponent_team_name": team_name_map.get(home_id, str(home_id)),
                        "side": "away",
                    }
                )
            except Exception as e:
                err_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": away_id,
                        "model": "Team-Builder",
                        "stage": "optimize_away",
                        "error": str(e),
                    }
                )

        if i % int(max(1, args.log_every)) == 0 or i == total_matches:
            print(
                f"[PROGRESS] {i}/{total_matches} matches processed | "
                f"gnn_rows={len(gnn_rows)} tb_rows={len(tb_rows)} errors={len(err_rows)}"
            )

    gnn_df = pd.DataFrame(gnn_rows)
    tb_df = pd.DataFrame(tb_rows)
    err_df = pd.DataFrame(err_rows)

    gnn_csv = output_dir / "gnn_predictions.csv"
    tb_csv = output_dir / "teambuilder_predictions.csv"
    err_csv = output_dir / "batch_errors.csv"

    gnn_df.to_csv(gnn_csv, index=False)
    tb_df.to_csv(tb_csv, index=False)
    err_df.to_csv(err_csv, index=False)

    print(f"[OK] saved: {gnn_csv}")
    print(f"[OK] saved: {tb_csv}")
    print(f"[OK] saved: {err_csv}")

    eval_summary_path = output_dir / "hitrate_eval" / "hitrate_summary.csv"
    if (not args.skip_evaluate) and (not gnn_df.empty) and (not tb_df.empty):
        eval_out_dir = output_dir / "hitrate_eval"
        eval_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(args.eval_script),
            "--matches-csv",
            str(args.matches_csv),
            "--players-csv",
            str(args.players_csv),
            "--gnn-pred-csv",
            str(gnn_csv),
            "--teambuilder-pred-csv",
            str(tb_csv),
            "--output-dir",
            str(eval_out_dir),
            "--gnn-lineup-col",
            "lineup_ids",
            "--teambuilder-lineup-col",
            "lineup_ids",
        ]
        print("[INFO] running hit-rate evaluator...")
        subprocess.run(cmd, check=True)

        if eval_summary_path.exists():
            summary = pd.read_csv(eval_summary_path)
            print("\n===== Final Hit Rate Summary =====")
            print(summary.to_string(index=False))
        else:
            print(f"[WARN] evaluator finished but summary file missing: {eval_summary_path}")
    else:
        print("[WARN] evaluator skipped (skip flag set or missing prediction rows)")

    metadata = {
        "matches_csv": str(args.matches_csv),
        "total_matches_processed": int(total_matches),
        "gnn_rows": int(len(gnn_df)),
        "teambuilder_rows": int(len(tb_df)),
        "error_rows": int(len(err_df)),
        "gnn_model_ckpt": str(args.model_ckpt),
        "teambuilder_script": str(args.tb_script),
        "evaluator_script": str(args.eval_script),
        "evaluation_summary_csv": str(eval_summary_path),
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {meta_path}")
# 기능: build_argparser는 연산 GA 세대 반복을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: argparse.ArgumentParser
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch experiment runner: EPL 17/18 GNN vs Team-Builder hit-rate comparison"
    )

    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--league-mode", type=str, default="england", choices=["england", "non_england"])
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=DATA_DIR / "archive/matches_England.csv",
    )
    parser.add_argument(
        "--teams-csv",
        type=Path,
        default=DATA_DIR / "archive/teams.csv",
    )
    parser.add_argument(
        "--players-csv",
        type=Path,
        default=DATA_DIR / "archive/players.csv",
    )
    parser.add_argument(
        "--player-games-csv",
        type=Path,
        default=DATA_DIR / "archive/player_games.csv",
    )

    parser.add_argument("--max-matches", type=int, default=0, help="0 means full season (380 matches)")
    parser.add_argument("--filter-home-team", type=str, default=None, help="Optional: filter by home team name (e.g., 'Manchester United')")
    parser.add_argument("--filter-away-team", type=str, default=None, help="Optional: filter by away team name (e.g., 'Manchester City')")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # GNN settings
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gnn-squad-size", type=int, default=18)
    parser.add_argument("--gnn-pop-size", type=int, default=48)
    parser.add_argument("--gnn-generations", type=int, default=80)
    parser.add_argument("--gnn-elite-size", type=int, default=8)
    parser.add_argument("--gnn-mutation-p", type=float, default=0.18)
    parser.add_argument("--gnn-temperature", type=float, default=2.0)
    parser.add_argument("--gnn-early-stop-patience", type=int, default=10)

    parser.add_argument(
        "--model-def",
        type=Path,
        default=PROJECT_ROOT / "proposed_spatial_gnn_ga/train_gnn_phase5.py",
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=DATA_DIR / "phase_5_lineup/data/gnn_phase5/hetero_edge_gat_win_ddp_full.pt",
    )
    parser.add_argument(
        "--scaler-pt",
        type=Path,
        default=None,
        help="Optional scaler override. If omitted, tries checkpoint embedded scaler",
    )

    # Team-Builder settings
    parser.add_argument("--skip-teambuilder", action="store_true")
    parser.add_argument(
        "--tb-script",
        type=Path,
        default=PROJECT_ROOT / "baseline_teambuilder/optimize_lineup_phase5.py",
    )
    parser.add_argument(
        "--tb-synergy-dir",
        type=Path,
        default=DATA_DIR / "synergy",
    )
    parser.add_argument(
        "--tb-tactics-dir",
        type=Path,
        default=DATA_DIR / "tactics",
    )
    parser.add_argument(
        "--tb-archive-dir",
        type=Path,
        default=DATA_DIR / "archive",
    )
    parser.add_argument("--tb-formation", type=str, default="4-3-3")
    parser.add_argument("--tb-lambda-vi", type=float, default=1.0)
    parser.add_argument("--tb-lambda-io", type=float, default=1.0)
    parser.add_argument("--tb-lambda-id", type=float, default=1.0)
    parser.add_argument("--tb-solver-time-limit", type=int, default=45)
    parser.add_argument("--tb-io-source", type=str, default="phase4", choices=["phase4", "recompute"])
    parser.add_argument("--tb-id-source", type=str, default="phase4", choices=["phase4", "recompute"])
    parser.add_argument("--tb-lambda-csv", type=Path, default=None)
    parser.add_argument("--tb-lambda-scaler-stats", type=Path, default=None)

    # Evaluation settings
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=PROJECT_ROOT / "evaluation_comparison/evaluate_hitrate_gnn_vs_teambuilder.py",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "phase_6_validation/data/hitrate_batch_epl17_18",
    )
    parser.add_argument(
        "--history-matches-csv",
        type=Path,
        default=None,
        help="Optional full matches CSV for historical feature construction (defaults to --matches-csv)",
    )

    return parser
# 기능: main는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 380경기 배치 실험에서 GNN/Team-Builder를 동일 조건으로 실행하고 비교 지표 CSV를 재현 가능하게 만들기 위해 필요하다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: None
def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
