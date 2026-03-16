#!/usr/bin/env python3
"""Phase 4: Compute normalized synergy metrics (V_I, I_O, I_D).

핵심 구현:
- 90분 정규화: 선수/선수쌍별 분모 M 사용
- 전술 가중치 w_k: Phase 3의 tactic_id 기준 가중
- 저출전/저공존 예외처리: filter 또는 downweight

수식(구현형):
- V_I(p) = sum(VAEP_a) * 90 / M(p)
- I_O(i,j) = (sum_k w_k * (VAEP_i + VAEP_j)) * 90 / M(i,j)
- I_D(i,q) = (sum_k w_k * (VAEP_def + VAEP_off_last)) * 90 / M(i,q)
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


DEFENSIVE_SUCCESS_TYPES = {"interception", "tackle"}


# 기능: Phase4 입력(vaep_actions, atomic_actions_with_phase) 로드와 기본 스키마 검증을 수행한다.
# 동작/맥락: game_id 교집합 정렬 및 max_games 컷오프로 이후 VI/IO/ID 계산의 일관된 입력을 보장한다.
def _load_phase4_inputs(
    vaep_path: Path,
    atomic_phase_path: Path,
    max_games: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not vaep_path.exists():
        raise FileNotFoundError(f"Missing VAEP file: {vaep_path}")
    if not atomic_phase_path.exists():
        raise FileNotFoundError(f"Missing phase file: {atomic_phase_path}")

    vaep = pd.read_parquet(vaep_path)
    atomic = pd.read_parquet(atomic_phase_path)

    required_vaep = {"game_id", "player_id", "team_id", "action_id", "original_event_id", "vaep_value"}
    required_atomic = {
        "game_id",
        "phase_id",
        "action_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "original_event_id",
    }

    miss_vaep = sorted(required_vaep - set(vaep.columns))
    miss_atomic = sorted(required_atomic - set(atomic.columns))
    if miss_vaep:
        raise ValueError(f"vaep_actions missing columns: {miss_vaep}")
    if miss_atomic:
        raise ValueError(f"atomic_actions_with_phase missing columns: {miss_atomic}")

    atomic_game_ids = set(pd.to_numeric(atomic["game_id"], errors="coerce").dropna().astype(int).tolist())
    vaep = vaep[vaep["game_id"].isin(atomic_game_ids)].copy()

    if max_games is not None:
        keep_games = atomic["game_id"].drop_duplicates().head(max_games)
        atomic = atomic[atomic["game_id"].isin(keep_games)].copy()
        vaep = vaep[vaep["game_id"].isin(set(keep_games))].copy()

    return vaep, atomic


# 기능: atomic 액션 테이블에 event 단위 VAEP(및 offensive/defensive 값)를 결합한다.
# 동작/맥락: original_event_id 기준 병합 후 vaep_value 결측을 0으로 보정해 상호작용 계산 입력을 만든다.
def _attach_vaep_to_atomic(vaep: pd.DataFrame, atomic: pd.DataFrame) -> pd.DataFrame:
    agg_map: dict[str, tuple[str, str]] = {"vaep_value": ("vaep_value", "mean")}
    if "offensive_value" in vaep.columns:
        agg_map["offensive_value"] = ("offensive_value", "mean")
    if "defensive_value" in vaep.columns:
        agg_map["defensive_value"] = ("defensive_value", "mean")

    event_vaep = (
        vaep.dropna(subset=["original_event_id"])
        .groupby(["game_id", "original_event_id"], as_index=False)
        .agg(**agg_map)
    )

    atomic_enriched = atomic.merge(event_vaep, on=["game_id", "original_event_id"], how="left")

    if "vaep_value_x" in atomic_enriched.columns and "vaep_value_y" in atomic_enriched.columns:
        atomic_enriched["vaep_value"] = (
            pd.to_numeric(atomic_enriched["vaep_value_y"], errors="coerce")
            .fillna(pd.to_numeric(atomic_enriched["vaep_value_x"], errors="coerce"))
            .fillna(0.0)
        )
        atomic_enriched = atomic_enriched.drop(columns=["vaep_value_x", "vaep_value_y"])
    elif "vaep_value" not in atomic_enriched.columns:
        atomic_enriched["vaep_value"] = 0.0
    else:
        atomic_enriched["vaep_value"] = pd.to_numeric(atomic_enriched["vaep_value"], errors="coerce").fillna(0.0)

    for col in ["offensive_value", "defensive_value"]:
        if col not in atomic_enriched.columns:
            atomic_enriched[col] = 0.0
        atomic_enriched[col] = pd.to_numeric(atomic_enriched[col], errors="coerce").fillna(0.0)

    atomic_enriched["period_id"] = pd.to_numeric(atomic_enriched["period_id"], errors="coerce").fillna(0).astype(int)
    atomic_enriched["time_seconds"] = pd.to_numeric(atomic_enriched["time_seconds"], errors="coerce").fillna(0.0)

    atomic_enriched = atomic_enriched.sort_values(
        ["game_id", "period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)
    return atomic_enriched


# 기능: 전술 가중치 CSV를 (team_id or global, tactic_id)→weight 딕셔너리로 로드한다.
# 동작/맥락: IO/ID 이벤트에 w_k를 곱할 때 팀별 우선-전역 fallback 구조를 사용한다.
def _load_tactic_weights(path: Path | None, id_col: str) -> dict[tuple[int | None, int], float]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Tactic weight file not found: {path}")

    df = pd.read_csv(path)
    if id_col not in df.columns or "weight" not in df.columns:
        raise ValueError(f"{path} must include columns: {id_col}, weight (optional: team_id)")

    out: dict[tuple[int | None, int], float] = {}
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        tactic_id = row_dict.get(id_col)
        if pd.isna(tactic_id):
            continue
        tactic_id_int = int(tactic_id)
        weight = float(row_dict.get("weight", 1.0))
        team_id = row_dict.get("team_id")
        team_key = None if pd.isna(team_id) else int(team_id)
        out[(team_key, tactic_id_int)] = weight
    return out


# 기능: 특정 팀/전술 조합의 가중치를 선택한다.
# 동작/맥락: team-specific 키 우선, 없으면 global 키, 마지막으로 default_weight를 반환한다.
def _pick_weight(
    weight_map: dict[tuple[int | None, int], float],
    team_id: int,
    tactic_id: float | int | None,
    default_weight: float,
) -> float:
    if pd.isna(tactic_id):
        return float(default_weight)
    tid = int(tactic_id)
    if (int(team_id), tid) in weight_map:
        return float(weight_map[(int(team_id), tid)])
    if (None, tid) in weight_map:
        return float(weight_map[(None, tid)])
    return float(default_weight)


# 기능: 선수 출전 분모(minutes) 계산용 presence 테이블을 만든다.
# 동작/맥락: 분 단위 공존/대치 카운트를 통해 player_minutes, same_team_minutes, opp_minutes를 생성한다.
def _compute_presence_tables(
    atomic_enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    presence = atomic_enriched[["game_id", "period_id", "time_seconds", "team_id", "player_id"]].copy()
    presence = presence.dropna(subset=["player_id", "team_id"]).copy()

    presence["player_id"] = presence["player_id"].astype(int)
    presence["team_id"] = presence["team_id"].astype(int)
    presence["minute_bin"] = np.floor(presence["time_seconds"] / 60.0).astype(int)

    presence = presence[["game_id", "period_id", "minute_bin", "team_id", "player_id"]].drop_duplicates()

    player_min_counter: Counter[int] = Counter()
    same_team_counter: Counter[tuple[int, int, int]] = Counter()
    opp_counter: Counter[tuple[int, int, int, int]] = Counter()

    for _, minute_df in presence.groupby(["game_id", "period_id", "minute_bin"], sort=False):
        team_map: dict[int, list[int]] = {}
        for team_id, team_df in minute_df.groupby("team_id"):
            plist = sorted(team_df["player_id"].unique().tolist())
            team_map[int(team_id)] = plist

        for team_id, plist in team_map.items():
            for pid in plist:
                player_min_counter[pid] += 1
            for a, b in combinations(plist, 2):
                same_team_counter[(team_id, int(a), int(b))] += 1

        teams = sorted(team_map.keys())
        for t1, t2 in combinations(teams, 2):
            p1_list = team_map[t1]
            p2_list = team_map[t2]
            for p1 in p1_list:
                for p2 in p2_list:
                    opp_counter[(int(t1), int(p1), int(t2), int(p2))] += 1
                    opp_counter[(int(t2), int(p2), int(t1), int(p1))] += 1

    player_minutes = pd.DataFrame(
        [{"player_id": int(pid), "minutes_played": float(mins)} for pid, mins in player_min_counter.items()]
    )

    same_team_minutes = pd.DataFrame(
        [
            {"team_id": int(team), "player_i": int(a), "player_j": int(b), "minutes_together": float(mins)}
            for (team, a, b), mins in same_team_counter.items()
        ]
    )

    opp_minutes = pd.DataFrame(
        [
            {
                "defending_team_id": int(t_def),
                "defender_player_id": int(p_def),
                "opponent_team_id": int(t_off),
                "opponent_player_id": int(p_off),
                "minutes_opposed": float(mins),
            }
            for (t_def, p_def, t_off, p_off), mins in opp_counter.items()
        ]
    )

    return player_minutes, same_team_minutes, opp_minutes


# 기능: matches의 lineup/substitutions를 사용해 정확한 출전시간/공존시간을 계산한다.
# 동작/맥락: 선수별 on-field 구간을 구성하고 구간 중첩 길이로 player_minutes, same_team_minutes, opp_minutes를 만든다.
def _compute_presence_tables_from_matches(
    atomic_enriched: pd.DataFrame,
    matches_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(matches_dir.glob("matches_*.csv"))
    if not files:
        raise FileNotFoundError(f"No matches_*.csv found in {matches_dir}")

    needed_game_ids = set(pd.to_numeric(atomic_enriched["game_id"], errors="coerce").dropna().astype(int).tolist())
    if not needed_game_ids:
        raise RuntimeError("No game_id found in atomic_enriched")

    required_cols = [
        "wyId",
        "team1.teamId",
        "team2.teamId",
        "team1.formation.lineup",
        "team1.formation.substitutions",
        "team2.formation.lineup",
        "team2.formation.substitutions",
    ]

    match_parts: list[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file, usecols=lambda c: c in required_cols)
        if "wyId" not in df.columns:
            continue
        df["wyId"] = pd.to_numeric(df["wyId"], errors="coerce")
        df = df[df["wyId"].isin(needed_game_ids)].copy()
        if df.empty:
            continue
        match_parts.append(df)

    if not match_parts:
        raise RuntimeError("No matching rows found in matches_*.csv for current game_ids")

    matches = pd.concat(match_parts, ignore_index=True).drop_duplicates(subset=["wyId"]).copy()

    atomic_tmp = atomic_enriched[["game_id", "period_id", "time_seconds"]].copy()
    atomic_tmp["game_id"] = pd.to_numeric(atomic_tmp["game_id"], errors="coerce")
    atomic_tmp["period_id"] = pd.to_numeric(atomic_tmp["period_id"], errors="coerce").fillna(1)
    atomic_tmp["time_seconds"] = pd.to_numeric(atomic_tmp["time_seconds"], errors="coerce").fillna(0.0)
    atomic_tmp = atomic_tmp.dropna(subset=["game_id"])
    atomic_tmp["game_id"] = atomic_tmp["game_id"].astype(int)
    atomic_tmp["absolute_minute"] = (atomic_tmp["period_id"] - 1.0) * 45.0 + (atomic_tmp["time_seconds"] / 60.0)
    game_end_minutes = (
        atomic_tmp.groupby("game_id", as_index=False)["absolute_minute"]
        .max()
        .rename(columns={"absolute_minute": "game_end_minute"})
    )
    game_end_minutes["game_end_minute"] = game_end_minutes["game_end_minute"].clip(lower=90.0)
    game_end_map = dict(zip(game_end_minutes["game_id"], game_end_minutes["game_end_minute"]))

    def _safe_eval_list(val: object) -> list[dict]:
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
        if isinstance(val, str):
            txt = val.strip()
            if not txt:
                return []
            try:
                parsed = ast.literal_eval(txt)
            except Exception:
                return []
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        return []

    def _to_int(v: object) -> int | None:
        try:
            if pd.isna(v):
                return None
            return int(float(v))
        except Exception:
            return None

    player_min_counter: Counter[int] = Counter()
    same_team_counter: Counter[tuple[int, int, int]] = Counter()
    opp_counter: Counter[tuple[int, int, int, int]] = Counter()

    def _build_intervals(lineup_raw: object, subs_raw: object, game_end: float) -> dict[int, tuple[float, float]]:
        lineup = _safe_eval_list(lineup_raw)
        subs = _safe_eval_list(subs_raw)

        intervals: dict[int, list[float]] = {}
        for row in lineup:
            pid = _to_int(row.get("playerId"))
            if pid is None:
                continue
            intervals[pid] = [0.0, float(game_end)]

        subs_sorted = sorted(subs, key=lambda x: _to_int(x.get("minute")) or 0)
        for sub in subs_sorted:
            minute = _to_int(sub.get("minute"))
            player_in = _to_int(sub.get("playerIn"))
            player_out = _to_int(sub.get("playerOut"))
            if minute is None:
                continue
            m = float(np.clip(minute, 0, game_end))

            if player_out is not None and player_out in intervals:
                intervals[player_out][1] = min(intervals[player_out][1], m)

            if player_in is not None:
                if player_in in intervals:
                    intervals[player_in][0] = min(intervals[player_in][0], m)
                    intervals[player_in][1] = max(intervals[player_in][1], game_end)
                else:
                    intervals[player_in] = [m, float(game_end)]

        out: dict[int, tuple[float, float]] = {}
        for pid, (start, end) in intervals.items():
            s = float(np.clip(start, 0.0, game_end))
            e = float(np.clip(end, 0.0, game_end))
            if e > s:
                out[int(pid)] = (s, e)
        return out

    for rowd in matches.to_dict(orient="records"):
        game_id = _to_int(rowd.get("wyId"))
        team1 = _to_int(rowd.get("team1.teamId"))
        team2 = _to_int(rowd.get("team2.teamId"))
        if game_id is None or team1 is None or team2 is None:
            continue
        game_end = float(game_end_map.get(game_id, 90.0))

        team1_intervals = _build_intervals(
            rowd.get("team1.formation.lineup"),
            rowd.get("team1.formation.substitutions"),
            game_end,
        )
        team2_intervals = _build_intervals(
            rowd.get("team2.formation.lineup"),
            rowd.get("team2.formation.substitutions"),
            game_end,
        )

        for pid, (s, e) in team1_intervals.items():
            player_min_counter[pid] += float(e - s)
        for pid, (s, e) in team2_intervals.items():
            player_min_counter[pid] += float(e - s)

        t1_ids = sorted(team1_intervals.keys())
        t2_ids = sorted(team2_intervals.keys())

        for a, b in combinations(t1_ids, 2):
            sa, ea = team1_intervals[a]
            sb, eb = team1_intervals[b]
            overlap = max(0.0, min(ea, eb) - max(sa, sb))
            if overlap > 0:
                same_team_counter[(int(team1), int(a), int(b))] += float(overlap)

        for a, b in combinations(t2_ids, 2):
            sa, ea = team2_intervals[a]
            sb, eb = team2_intervals[b]
            overlap = max(0.0, min(ea, eb) - max(sa, sb))
            if overlap > 0:
                same_team_counter[(int(team2), int(a), int(b))] += float(overlap)

        for p1 in t1_ids:
            s1, e1 = team1_intervals[p1]
            for p2 in t2_ids:
                s2, e2 = team2_intervals[p2]
                overlap = max(0.0, min(e1, e2) - max(s1, s2))
                if overlap > 0:
                    opp_counter[(int(team1), int(p1), int(team2), int(p2))] += float(overlap)
                    opp_counter[(int(team2), int(p2), int(team1), int(p1))] += float(overlap)

    player_minutes = pd.DataFrame(
        [{"player_id": int(pid), "minutes_played": float(mins)} for pid, mins in player_min_counter.items()]
    )

    same_team_minutes = pd.DataFrame(
        [
            {"team_id": int(team), "player_i": int(a), "player_j": int(b), "minutes_together": float(mins)}
            for (team, a, b), mins in same_team_counter.items()
        ]
    )

    opp_minutes = pd.DataFrame(
        [
            {
                "defending_team_id": int(t_def),
                "defender_player_id": int(p_def),
                "opponent_team_id": int(t_off),
                "opponent_player_id": int(p_off),
                "minutes_opposed": float(mins),
            }
            for (t_def, p_def, t_off, p_off), mins in opp_counter.items()
        ]
    )

    if player_minutes.empty:
        raise RuntimeError("No player minutes built from lineup/substitution data")

    return player_minutes, same_team_minutes, opp_minutes


# 기능: 저출전/저공존 샘플에 대한 신뢰도 계수를 계산한다.
# 동작/맥락: policy(filter/downweight)에 따라 None 또는 0~1 배율을 반환해 VI/IO/ID 스케일을 조정한다.
def _reliability_factor(minutes: float, min_minutes: float, policy: str) -> float | None:
    m = float(minutes)
    if m <= 0:
        return None if policy == "filter" else 0.0
    if policy == "filter":
        return 1.0 if m >= min_minutes else None
    if min_minutes <= 0:
        return 1.0
    return float(min(1.0, m / min_minutes))


# 기능: 개인 가치 V_I를 90분 정규화와 신뢰도 보정으로 계산한다.
# 동작/맥락: vaep_sum, minutes_played, vi_90_raw, reliability를 결합해 최종 vi를 산출한다.
def _compute_vi(
    vaep: pd.DataFrame,
    player_minutes: pd.DataFrame,
    min_player_minutes: float,
    low_minutes_policy: str,
) -> pd.DataFrame:
    vi_raw = (
        vaep.groupby("player_id", as_index=False)
        .agg(
            team_id=("team_id", lambda s: int(s.mode().iat[0]) if not s.mode().empty else int(s.iloc[0])),
            vaep_sum=("vaep_value", "sum"),
            n_actions=("action_id", "count"),
            vaep_mean=("vaep_value", "mean"),
        )
    )

    vi = vi_raw.merge(player_minutes, on="player_id", how="left")
    vi["minutes_played"] = pd.to_numeric(vi["minutes_played"], errors="coerce").fillna(0.0)

    vi["vi_90_raw"] = np.where(vi["minutes_played"] > 0, vi["vaep_sum"] * 90.0 / vi["minutes_played"], 0.0)
    vi["reliability"] = vi["minutes_played"].map(
        lambda m: _reliability_factor(m, min_player_minutes, low_minutes_policy)
    )

    if low_minutes_policy == "filter":
        vi = vi[vi["reliability"].notna()].copy()
        vi["reliability"] = 1.0
    else:
        vi["reliability"] = pd.to_numeric(vi["reliability"], errors="coerce").fillna(0.0)

    vi["vi"] = vi["vi_90_raw"]
    vi = vi.sort_values("vi", ascending=False).reset_index(drop=True)
    return vi


# 기능: 같은 팀 연속 액션 쌍으로 공격 상호작용 I_O를 계산한다.
# 동작/맥락: pair_vaep에 공격 전술 가중치와 minutes_together 기반 정규화를 적용해 io/io_player를 만든다.
def _compute_io(
    atomic_enriched: pd.DataFrame,
    same_team_minutes: pd.DataFrame,
    attack_weight_map: dict[tuple[int | None, int], float],
    attack_default_weight: float,
    min_pair_minutes: float,
    low_pair_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = atomic_enriched.copy()

    grp = ["game_id", "phase_id"]
    df["next_player_id"] = df.groupby(grp)["player_id"].shift(-1)
    df["next_team_id"] = df.groupby(grp)["team_id"].shift(-1)
    df["next_vaep_value"] = df.groupby(grp)["vaep_value"].shift(-1)

    io_pairs = df[df["next_player_id"].notna()].copy()
    io_pairs = io_pairs[io_pairs["player_id"] != io_pairs["next_player_id"]]
    io_pairs = io_pairs[io_pairs["team_id"] == io_pairs["next_team_id"]]

    io_pairs["player_id"] = io_pairs["player_id"].astype(int)
    io_pairs["next_player_id"] = io_pairs["next_player_id"].astype(int)
    io_pairs["team_id"] = io_pairs["team_id"].astype(int)

    io_pairs["pair_vaep_raw"] = io_pairs["vaep_value"] + io_pairs["next_vaep_value"]

    io_pairs["attack_weight"] = io_pairs.apply(
        lambda r: _pick_weight(
            attack_weight_map,
            int(r["team_id"]),
            r.get("attack_tactic_id"),
            attack_default_weight,
        ),
        axis=1,
    )
    io_pairs["pair_vaep_weighted"] = io_pairs["pair_vaep_raw"] * io_pairs["attack_weight"]

    io_pairs["player_i"] = io_pairs[["player_id", "next_player_id"]].min(axis=1).astype(int)
    io_pairs["player_j"] = io_pairs[["player_id", "next_player_id"]].max(axis=1).astype(int)

    io_agg = (
        io_pairs.groupby(["team_id", "player_i", "player_j"], as_index=False)
        .agg(
            io_weighted_sum=("pair_vaep_weighted", "sum"),
            io_raw_sum=("pair_vaep_raw", "sum"),
            io_count=("pair_vaep_weighted", "count"),
            io_weight_mean=("attack_weight", "mean"),
        )
    )

    io_agg = io_agg.merge(
        same_team_minutes,
        on=["team_id", "player_i", "player_j"],
        how="left",
    )
    io_agg["minutes_together"] = pd.to_numeric(io_agg["minutes_together"], errors="coerce").fillna(0.0)

    io_agg["io_90_raw"] = np.where(
        io_agg["minutes_together"] > 0,
        io_agg["io_weighted_sum"] * 90.0 / io_agg["minutes_together"],
        0.0,
    )

    io_agg["reliability"] = io_agg["minutes_together"].map(
        lambda m: _reliability_factor(m, min_pair_minutes, low_pair_policy)
    )
    if low_pair_policy == "filter":
        io_agg = io_agg[io_agg["reliability"].notna()].copy()
        io_agg["reliability"] = 1.0
    else:
        io_agg["reliability"] = pd.to_numeric(io_agg["reliability"], errors="coerce").fillna(0.0)

    io_agg["io"] = io_agg["io_90_raw"]

    io_pair_list = io_agg.rename(columns={"player_i": "player_a_id", "player_j": "player_b_id"}).copy()
    io_pair_list["io_sum"] = io_pair_list["io"]
    io_pair_list["io_mean"] = np.where(io_pair_list["io_count"] > 0, io_pair_list["io_weighted_sum"] / io_pair_list["io_count"], 0.0)

    contrib_a = io_pair_list[["player_a_id", "io"]].rename(columns={"player_a_id": "player_id"})
    contrib_b = io_pair_list[["player_b_id", "io"]].rename(columns={"player_b_id": "player_id"})
    io_player = (
        pd.concat([contrib_a, contrib_b], ignore_index=True)
        .assign(io=lambda d: d["io"] * 0.5)
        .groupby("player_id", as_index=False)
        .agg(io=("io", "sum"), io_events=("io", "count"))
    )

    return io_pair_list.sort_values("io", ascending=False).reset_index(drop=True), io_player


# 기능: 소유권 전환 시 수비 성공 이벤트를 기반으로 방어 상호작용 I_D를 계산한다.
# 동작/맥락: defender-opponent 쌍에 defense weight와 minutes_opposed 정규화를 적용해 id/id_player를 만든다.
def _compute_id(
    atomic_enriched: pd.DataFrame,
    opp_minutes: pd.DataFrame,
    defense_weight_map: dict[tuple[int | None, int], float],
    defense_default_weight: float,
    min_pair_minutes: float,
    low_pair_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = atomic_enriched.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True).copy()

    df["prev_game_id"] = df["game_id"].shift(1)
    df["prev_period_id"] = df["period_id"].shift(1)
    df["prev_team_id"] = df["team_id"].shift(1)
    df["prev_player_id"] = df["player_id"].shift(1)
    df["prev_vaep_value"] = df["vaep_value"].shift(1)
    df["prev_action_id"] = df["action_id"].shift(1)

    candidates = df[
        (df["prev_game_id"] == df["game_id"])
        & (df["prev_period_id"] == df["period_id"])
        & (df["prev_team_id"] != df["team_id"])
        & (df["type_name"].astype(str).str.lower().isin(DEFENSIVE_SUCCESS_TYPES))
    ].copy()

    if candidates.empty:
        empty_pairs = pd.DataFrame(
            columns=[
                "defending_team_id",
                "opponent_team_id",
                "defender_player_id",
                "opponent_player_id",
                "id_weighted_sum",
                "id_raw_sum",
                "id_count",
                "id_weight_mean",
                "minutes_opposed",
                "id_90_raw",
                "reliability",
                "id",
                "id_sum",
                "id_mean",
            ]
        )
        empty_player = pd.DataFrame(columns=["player_id", "id", "id_events"])
        return empty_pairs, empty_player

    candidates["defending_team_id"] = candidates["team_id"].astype(int)
    candidates["defender_player_id"] = candidates["player_id"].astype(int)
    candidates["opponent_team_id"] = candidates["prev_team_id"].astype(int)
    candidates["opponent_player_id"] = candidates["prev_player_id"].astype(int)

    candidates["id_raw_event"] = candidates["vaep_value"] + candidates["prev_vaep_value"]

    candidates["defense_weight"] = candidates.apply(
        lambda r: _pick_weight(
            defense_weight_map,
            int(r["defending_team_id"]),
            r.get("defense_tactic_id"),
            defense_default_weight,
        ),
        axis=1,
    )
    candidates["id_weighted_event"] = candidates["id_raw_event"] * candidates["defense_weight"]

    id_pair = (
        candidates.groupby(
            ["defending_team_id", "defender_player_id", "opponent_team_id", "opponent_player_id"],
            as_index=False,
        )
        .agg(
            id_weighted_sum=("id_weighted_event", "sum"),
            id_raw_sum=("id_raw_event", "sum"),
            id_count=("id_weighted_event", "count"),
            id_weight_mean=("defense_weight", "mean"),
        )
    )

    id_pair = id_pair.merge(
        opp_minutes,
        on=["defending_team_id", "defender_player_id", "opponent_team_id", "opponent_player_id"],
        how="left",
    )
    id_pair["minutes_opposed"] = pd.to_numeric(id_pair["minutes_opposed"], errors="coerce").fillna(0.0)

    id_pair["id_90_raw"] = np.where(
        id_pair["minutes_opposed"] > 0,
        id_pair["id_weighted_sum"] * 90.0 / id_pair["minutes_opposed"],
        0.0,
    )

    id_pair["reliability"] = id_pair["minutes_opposed"].map(
        lambda m: _reliability_factor(m, min_pair_minutes, low_pair_policy)
    )
    if low_pair_policy == "filter":
        id_pair = id_pair[id_pair["reliability"].notna()].copy()
        id_pair["reliability"] = 1.0
    else:
        id_pair["reliability"] = pd.to_numeric(id_pair["reliability"], errors="coerce").fillna(0.0)

    id_pair["id"] = id_pair["id_90_raw"]
    id_pair["id_sum"] = id_pair["id"]
    id_pair["id_mean"] = np.where(id_pair["id_count"] > 0, id_pair["id_weighted_sum"] / id_pair["id_count"], 0.0)

    id_player = (
        id_pair.groupby("defender_player_id", as_index=False)
        .agg(id=("id", "sum"), id_events=("id", "count"))
        .rename(columns={"defender_player_id": "player_id"})
    )

    return id_pair.sort_values("id", ascending=False).reset_index(drop=True), id_player


# 기능: IO 쌍 리스트를 선수×선수 매트릭스로 변환한다.
# 동작/맥락: (a,b)와 (b,a)를 확장해 pivot_table로 대칭적 조회용 CSV를 생성한다.
def _build_io_matrix(io_pair: pd.DataFrame) -> pd.DataFrame:
    if io_pair.empty:
        return pd.DataFrame()

    undirected = io_pair[["player_a_id", "player_b_id", "io"]].copy()
    reverse = undirected.rename(columns={"player_a_id": "player_b_id", "player_b_id": "player_a_id"})
    full = pd.concat([undirected, reverse], ignore_index=True)

    matrix = (
        full.pivot_table(
            index="player_a_id",
            columns="player_b_id",
            values="io",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return matrix


# 기능: matches_*.csv에서 팀-경기 포인트/승리 라벨을 로드한다.
# 동작/맥락: lambda 추정 학습용 타깃(points/is_win)을 game_id, team_id 단위로 구성한다.
def _load_match_points(matches_dir: Path, game_ids: set[int] | None = None) -> pd.DataFrame:
    files = sorted(matches_dir.glob("matches_*.csv"))
    if not files:
        raise FileNotFoundError(f"No matches_*.csv found in {matches_dir}")

    parts: list[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file)
        required = {"wyId", "team1.teamId", "team2.teamId", "team1.score", "team2.score"}
        if not required.issubset(set(df.columns)):
            continue

        m = df[["wyId", "team1.teamId", "team2.teamId", "team1.score", "team2.score"]].copy()
        m = m.rename(
            columns={
                "wyId": "game_id",
                "team1.teamId": "home_team_id",
                "team2.teamId": "away_team_id",
                "team1.score": "home_score",
                "team2.score": "away_score",
            }
        )

        for col in ["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]:
            m[col] = pd.to_numeric(m[col], errors="coerce")
        m = m.dropna(subset=["game_id", "home_team_id", "away_team_id", "home_score", "away_score"])

        m["game_id"] = m["game_id"].astype(int)
        m["home_team_id"] = m["home_team_id"].astype(int)
        m["away_team_id"] = m["away_team_id"].astype(int)

        if game_ids is not None:
            m = m[m["game_id"].isin(game_ids)]
        if m.empty:
            continue

        home_points = np.where(m["home_score"] > m["away_score"], 3, np.where(m["home_score"] == m["away_score"], 1, 0))
        away_points = np.where(m["away_score"] > m["home_score"], 3, np.where(m["away_score"] == m["home_score"], 1, 0))

        home = pd.DataFrame(
            {
                "game_id": m["game_id"].astype(int),
                "team_id": m["home_team_id"].astype(int),
                "points": home_points.astype(float),
                "is_win": (home_points == 3).astype(int),
            }
        )
        away = pd.DataFrame(
            {
                "game_id": m["game_id"].astype(int),
                "team_id": m["away_team_id"].astype(int),
                "points": away_points.astype(float),
                "is_win": (away_points == 3).astype(int),
            }
        )
        parts.append(pd.concat([home, away], ignore_index=True))

    if not parts:
        raise RuntimeError("No valid match rows found for lambda estimation.")

    out = pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "team_id"]).reset_index(drop=True)
    return out


# 기능: 경기 단위 VI/IO/ID 합성 피처(team_game)를 구축한다.
# 동작/맥락: lambda 추정에서 설명변수(vi_game, io_game, id_game)로 사용되는 표를 생성한다.
def _build_team_game_components(
    atomic_enriched: pd.DataFrame,
    vaep: pd.DataFrame,
    attack_weight_map: dict[tuple[int | None, int], float],
    defense_weight_map: dict[tuple[int | None, int], float],
    attack_default_weight: float,
    defense_default_weight: float,
) -> pd.DataFrame:
    team_game_base = atomic_enriched[["game_id", "team_id"]].drop_duplicates().copy()
    team_game_base["game_id"] = team_game_base["game_id"].astype(int)
    team_game_base["team_id"] = team_game_base["team_id"].astype(int)

    vi_game = (
        vaep.groupby(["game_id", "team_id"], as_index=False)
        .agg(vi_game=("vaep_value", "sum"))
        .assign(game_id=lambda d: d["game_id"].astype(int), team_id=lambda d: d["team_id"].astype(int))
    )

    io_df = atomic_enriched.copy()
    grp = ["game_id", "phase_id"]
    io_df["next_player_id"] = io_df.groupby(grp)["player_id"].shift(-1)
    io_df["next_team_id"] = io_df.groupby(grp)["team_id"].shift(-1)
    io_df["next_vaep_value"] = io_df.groupby(grp)["vaep_value"].shift(-1)

    io_pairs = io_df[io_df["next_player_id"].notna()].copy()
    io_pairs = io_pairs[io_pairs["team_id"] == io_pairs["next_team_id"]]
    io_pairs = io_pairs[io_pairs["player_id"] != io_pairs["next_player_id"]]
    io_pairs["team_id"] = io_pairs["team_id"].astype(int)
    io_pairs["io_event"] = io_pairs["vaep_value"] + io_pairs["next_vaep_value"]
    io_pairs["attack_weight"] = io_pairs.apply(
        lambda r: _pick_weight(
            attack_weight_map,
            int(r["team_id"]),
            r.get("attack_tactic_id"),
            attack_default_weight,
        ),
        axis=1,
    )
    io_pairs["io_event_weighted"] = io_pairs["io_event"] * io_pairs["attack_weight"]
    io_game = (
        io_pairs.groupby(["game_id", "team_id"], as_index=False)
        .agg(io_game=("io_event_weighted", "sum"))
        .assign(game_id=lambda d: d["game_id"].astype(int), team_id=lambda d: d["team_id"].astype(int))
    )

    id_df = atomic_enriched.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True).copy()
    id_df["prev_game_id"] = id_df["game_id"].shift(1)
    id_df["prev_period_id"] = id_df["period_id"].shift(1)
    id_df["prev_team_id"] = id_df["team_id"].shift(1)
    id_df["prev_player_id"] = id_df["player_id"].shift(1)
    id_df["prev_vaep_value"] = id_df["vaep_value"].shift(1)

    transitions = id_df[
        (id_df["prev_game_id"] == id_df["game_id"])
        & (id_df["prev_period_id"] == id_df["period_id"])
        & (id_df["prev_team_id"] != id_df["team_id"])
        & (id_df["type_name"].astype(str).str.lower().isin(DEFENSIVE_SUCCESS_TYPES))
    ].copy()

    if transitions.empty:
        id_game = pd.DataFrame(columns=["game_id", "team_id", "id_game"])
    else:
        transitions["team_id"] = transitions["team_id"].astype(int)
        transitions["id_event"] = transitions["vaep_value"] + transitions["prev_vaep_value"]
        transitions["defense_weight"] = transitions.apply(
            lambda r: _pick_weight(
                defense_weight_map,
                int(r["team_id"]),
                r.get("defense_tactic_id"),
                defense_default_weight,
            ),
            axis=1,
        )
        transitions["id_event_weighted"] = transitions["id_event"] * transitions["defense_weight"]
        id_game = (
            transitions.groupby(["game_id", "team_id"], as_index=False)
            .agg(id_game=("id_event_weighted", "sum"))
            .assign(game_id=lambda d: d["game_id"].astype(int), team_id=lambda d: d["team_id"].astype(int))
        )

    team_game = team_game_base.merge(vi_game, on=["game_id", "team_id"], how="left")
    team_game = team_game.merge(io_game, on=["game_id", "team_id"], how="left")
    team_game = team_game.merge(id_game, on=["game_id", "team_id"], how="left")
    for col in ["vi_game", "io_game", "id_game"]:
        team_game[col] = pd.to_numeric(team_game[col], errors="coerce").fillna(0.0)
    return team_game


# 기능: 경기 결과(points 또는 is_win)로 lambda_vi/io/id를 데이터 기반 추정한다.
# 동작/맥락: linear/logistic 회귀 계수 절댓값 비율을 정규화해 람다를 만들고 리포트를 반환한다.
def _estimate_lambdas_from_results(
    team_game: pd.DataFrame,
    match_points: pd.DataFrame,
    estimator: str,
    min_games: int,
) -> tuple[dict[str, float] | None, pd.DataFrame]:
    train = team_game.merge(match_points, on=["game_id", "team_id"], how="inner")
    train = train[["game_id", "team_id", "vi_game", "io_game", "id_game", "points", "is_win"]].dropna().copy()

    report_rows: list[dict] = [
        {
            "n_samples": int(len(train)),
            "estimator": estimator,
            "status": "ready" if len(train) >= min_games else "insufficient_samples",
        }
    ]

    if len(train) < min_games:
        return None, pd.DataFrame(report_rows)

    x = train[["vi_game", "io_game", "id_game"]].to_numpy(dtype=float)
    if estimator == "linear":
        y = train["points"].to_numpy(dtype=float)
        if np.std(y) == 0:
            return None, pd.DataFrame(report_rows + [{"status": "no_target_variance"}])
        model = LinearRegression()
        model.fit(x, y)
        coefs = np.asarray(model.coef_, dtype=float)
        target_name = "points"
    else:
        y = train["is_win"].to_numpy(dtype=int)
        if np.unique(y).shape[0] < 2:
            return None, pd.DataFrame(report_rows + [{"status": "no_class_variance"}])
        model = LogisticRegression(max_iter=2000)
        model.fit(x, y)
        coefs = np.asarray(model.coef_[0], dtype=float)
        target_name = "is_win"

    abs_coefs = np.abs(coefs)
    coef_sum = float(abs_coefs.sum())
    if coef_sum <= 1e-12:
        lambdas = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        status = "zero_coefficients_fallback_equal"
    else:
        lambdas = abs_coefs / coef_sum
        status = "ok"

    report_rows.append(
        {
            "status": status,
            "target": target_name,
            "coef_vi": float(coefs[0]),
            "coef_io": float(coefs[1]),
            "coef_id": float(coefs[2]),
            "lambda_vi": float(lambdas[0]),
            "lambda_io": float(lambdas[1]),
            "lambda_id": float(lambdas[2]),
            "lambda_sum": float(lambdas.sum()),
        }
    )

    return (
        {
            "lambda_vi": float(lambdas[0]),
            "lambda_io": float(lambdas[1]),
            "lambda_id": float(lambdas[2]),
        },
        pd.DataFrame(report_rows),
    )


# 기능: Phase4 전체(VI/IO/ID 계산, 람다 추정, 파일 저장)를 실행한다.
# 동작/맥락: 각 하위 계산 함수를 오케스트레이션해 player_synergy_scores와 상호작용 산출물을 생성한다.
def run_phase4(
    vaep_path: Path,
    atomic_phase_path: Path,
    output_dir: Path,
    lambda_vi: float,
    lambda_io: float,
    lambda_id: float,
    min_player_minutes: float,
    min_pair_minutes: float,
    low_minutes_policy: str,
    low_pair_policy: str,
    attack_weights_csv: Path | None,
    defense_weights_csv: Path | None,
    attack_default_weight: float,
    defense_default_weight: float,
    max_games: int | None,
    estimate_lambda: bool,
    lambda_estimator: str,
    matches_dir: Path,
    min_games_for_lambda: int,
) -> None:
    vaep, atomic = _load_phase4_inputs(
        vaep_path=vaep_path,
        atomic_phase_path=atomic_phase_path,
        max_games=max_games,
    )
    atomic_enriched = _attach_vaep_to_atomic(vaep=vaep, atomic=atomic)

    try:
        player_minutes, same_team_minutes, opp_minutes = _compute_presence_tables_from_matches(
            atomic_enriched=atomic_enriched,
            matches_dir=matches_dir,
        )
        print("[OK] Presence built from lineup/substitutions in matches_*.csv")
    except Exception as e:
        print(f"[WARN] Fallback to action-based presence due to: {e}")
        player_minutes, same_team_minutes, opp_minutes = _compute_presence_tables(atomic_enriched)

    attack_weight_map = _load_tactic_weights(attack_weights_csv, id_col="attack_tactic_id")
    defense_weight_map = _load_tactic_weights(defense_weights_csv, id_col="defense_tactic_id")

    vi = _compute_vi(
        vaep=vaep,
        player_minutes=player_minutes,
        min_player_minutes=min_player_minutes,
        low_minutes_policy=low_minutes_policy,
    )
    io_pair_agg, io_player = _compute_io(
        atomic_enriched=atomic_enriched,
        same_team_minutes=same_team_minutes,
        attack_weight_map=attack_weight_map,
        attack_default_weight=attack_default_weight,
        min_pair_minutes=min_pair_minutes,
        low_pair_policy=low_pair_policy,
    )
    id_pair_agg, id_player = _compute_id(
        atomic_enriched=atomic_enriched,
        opp_minutes=opp_minutes,
        defense_weight_map=defense_weight_map,
        defense_default_weight=defense_default_weight,
        min_pair_minutes=min_pair_minutes,
        low_pair_policy=low_pair_policy,
    )

    lambda_report = pd.DataFrame()
    team_game_features = pd.DataFrame()

    if estimate_lambda:
        team_game_features = _build_team_game_components(
            atomic_enriched=atomic_enriched,
            vaep=vaep,
            attack_weight_map=attack_weight_map,
            defense_weight_map=defense_weight_map,
            attack_default_weight=attack_default_weight,
            defense_default_weight=defense_default_weight,
        )
        match_points = _load_match_points(matches_dir=matches_dir, game_ids=set(team_game_features["game_id"].astype(int)))
        estimated, lambda_report = _estimate_lambdas_from_results(
            team_game=team_game_features,
            match_points=match_points,
            estimator=lambda_estimator,
            min_games=min_games_for_lambda,
        )
        if estimated is not None:
            lambda_vi = float(estimated["lambda_vi"])
            lambda_io = float(estimated["lambda_io"])
            lambda_id = float(estimated["lambda_id"])
            print(
                "[OK] Estimated lambdas from match results: "
                f"lambda_vi={lambda_vi:.4f}, lambda_io={lambda_io:.4f}, lambda_id={lambda_id:.4f}"
            )
        else:
            print("[WARN] Lambda estimation skipped/fallback; using provided lambda values.")

    player_scores = vi[["player_id", "team_id", "vi", "vaep_sum", "vi_90_raw", "minutes_played", "reliability", "n_actions", "vaep_mean"]].copy()
    player_scores = player_scores.merge(io_player, on="player_id", how="left")
    player_scores = player_scores.merge(id_player, on="player_id", how="left")

    for col in ["io", "io_events", "id", "id_events"]:
        player_scores[col] = pd.to_numeric(player_scores[col], errors="coerce").fillna(0.0)

    player_scores["lambda_vi"] = float(lambda_vi)
    player_scores["lambda_io"] = float(lambda_io)
    player_scores["lambda_id"] = float(lambda_id)

    player_scores["v_total"] = (
        lambda_vi * player_scores["vi"]
        + lambda_io * player_scores["io"]
        + lambda_id * player_scores["id"]
    )

    player_scores = player_scores.sort_values("v_total", ascending=False).reset_index(drop=True)

    io_matrix = _build_io_matrix(io_pair_agg)

    output_dir.mkdir(parents=True, exist_ok=True)

    vi.to_parquet(output_dir / "player_vi.parquet", index=False)
    vi.to_csv(output_dir / "player_vi.csv", index=False)

    io_pair_agg.to_parquet(output_dir / "attack_interaction_io.parquet", index=False)
    io_pair_agg.to_csv(output_dir / "attack_interaction_io.csv", index=False)

    id_pair_agg.to_parquet(output_dir / "defense_interaction_id.parquet", index=False)
    id_pair_agg.to_csv(output_dir / "defense_interaction_id.csv", index=False)

    if not io_matrix.empty:
        io_matrix.to_csv(output_dir / "attack_interaction_io_matrix.csv")

    player_minutes.to_csv(output_dir / "player_minutes_estimated.csv", index=False)
    same_team_minutes.to_csv(output_dir / "pair_minutes_together_estimated.csv", index=False)
    opp_minutes.to_csv(output_dir / "pair_minutes_opposed_estimated.csv", index=False)

    player_scores.to_parquet(output_dir / "player_synergy_scores.parquet", index=False)
    player_scores.to_csv(output_dir / "player_synergy_scores.csv", index=False)

    if estimate_lambda:
        if not team_game_features.empty:
            team_game_features.to_csv(output_dir / "lambda_team_game_features.csv", index=False)
        if not lambda_report.empty:
            lambda_report.to_csv(output_dir / "lambda_estimation_report.csv", index=False)

    print(f"[OK] Saved V_I: {output_dir / 'player_vi.parquet'}")
    print(f"[OK] Saved I_O: {output_dir / 'attack_interaction_io.parquet'}")
    print(f"[OK] Saved I_D: {output_dir / 'defense_interaction_id.parquet'}")
    print(f"[OK] Saved final V: {output_dir / 'player_synergy_scores.parquet'}")
    print(
        f"[OK] players={len(player_scores):,}, io_pairs={len(io_pair_agg):,}, "
        f"id_pairs={len(id_pair_agg):,}, minutes_rows={len(player_minutes):,}"
    )


# 기능: CLI 인자를 파싱하고 run_phase4를 호출하는 진입점이다.
# 동작/맥락: min_minutes/policy/weights/lambda 옵션을 받아 동일 설정 실험을 재현 가능하게 한다.
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Compute normalized V_I, I_O, I_D and total player score")
    parser.add_argument(
        "--vaep-path",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/vaep/vaep_actions.parquet"),
        help="Path to vaep_actions.parquet",
    )
    parser.add_argument(
        "--atomic-phase-path",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/tactics/atomic_actions_with_phase.parquet"),
        help="Path to atomic_actions_with_phase.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/synergy"),
        help="Directory to save Phase 4 outputs",
    )
    parser.add_argument("--lambda-vi", type=float, default=1.0)
    parser.add_argument("--lambda-io", type=float, default=1.0)
    parser.add_argument("--lambda-id", type=float, default=1.0)

    parser.add_argument("--min-player-minutes", type=float, default=90.0)
    parser.add_argument("--min-pair-minutes", type=float, default=90.0)
    parser.add_argument(
        "--low-minutes-policy",
        choices=["filter", "downweight"],
        default="downweight",
        help="How to handle players with low minutes",
    )
    parser.add_argument(
        "--low-pair-policy",
        choices=["filter", "downweight"],
        default="downweight",
        help="How to handle pairs with low shared/opposed minutes",
    )

    parser.add_argument(
        "--attack-weights-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: attack_tactic_id, weight, (optional team_id)",
    )
    parser.add_argument(
        "--defense-weights-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: defense_tactic_id, weight, (optional team_id)",
    )
    parser.add_argument("--attack-default-weight", type=float, default=1.0)
    parser.add_argument("--defense-default-weight", type=float, default=1.0)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--estimate-lambda", action="store_true", help="Estimate lambda weights from historical match results")
    parser.add_argument(
        "--lambda-estimator",
        choices=["linear", "logistic"],
        default="linear",
        help="Regression model for lambda estimation",
    )
    parser.add_argument(
        "--matches-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/archive"),
        help="Directory containing matches_*.csv",
    )
    parser.add_argument(
        "--min-games-for-lambda",
        type=int,
        default=100,
        help="Minimum team-game samples required for lambda estimation",
    )

    args = parser.parse_args()

    run_phase4(
        vaep_path=args.vaep_path,
        atomic_phase_path=args.atomic_phase_path,
        output_dir=args.output_dir,
        lambda_vi=args.lambda_vi,
        lambda_io=args.lambda_io,
        lambda_id=args.lambda_id,
        min_player_minutes=args.min_player_minutes,
        min_pair_minutes=args.min_pair_minutes,
        low_minutes_policy=args.low_minutes_policy,
        low_pair_policy=args.low_pair_policy,
        attack_weights_csv=args.attack_weights_csv,
        defense_weights_csv=args.defense_weights_csv,
        attack_default_weight=args.attack_default_weight,
        defense_default_weight=args.defense_default_weight,
        max_games=args.max_games,
        estimate_lambda=args.estimate_lambda,
        lambda_estimator=args.lambda_estimator,
        matches_dir=args.matches_dir,
        min_games_for_lambda=args.min_games_for_lambda,
    )


if __name__ == "__main__":
    main()
