#!/usr/bin/env python3
"""Phase 3: Possession-based tactic clustering with 3x3 zones.

Implements requested logic:
1) Split phases mainly by possession change (team_id change)
2) Map every action to 3x3 zone (1..9)
3) Categorize phase by first action type
   - Attack: pass / corner / freekick
   - Defense: tackle / interception / foul
4) Attack tactics: group by trajectory (zone sequence)
5) Defense tactics: group by single zone of first defensive action
6) Compute per-tactic usage and success
   - shot_success_rate: proportion of phases with shot creation
   - vaep_success_mean: mean phase VAEP sum (optional if VAEP file exists)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import socceraction.atomic.spadl as atomic_spadl


SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}

ATTACK_PASS_TYPES = {"pass", "cross"}
ATTACK_CORNER_TYPES = {"corner", "corner_crossed", "corner_short"}
ATTACK_FREEKICK_TYPES = {"freekick", "freekick_crossed", "freekick_short", "shot_freekick"}

DEFENSE_TACKLE_TYPES = {"tackle"}
DEFENSE_INTERCEPT_TYPES = {"interception"}
DEFENSE_FOUL_TYPES = {"foul"}


# 기능: SPADL 파일 기준 리그 목록을 수집/필터링한다.
# 동작/맥락: run_phase3의 리그 반복 대상을 정의해 전체 전술 추출 범위를 결정한다.
def _iter_leagues(spadl_dir: Path, include: list[str] | None) -> list[str]:
    leagues = sorted(
        p.stem.replace("spadl_", "")
        for p in spadl_dir.glob("spadl_*.parquet")
        if p.name != "spadl_all.parquet"
    )
    if include:
        wanted = set(include)
        return [league for league in leagues if league in wanted]
    return leagues


# 기능: 연속 중복 zone을 제거해 궤적을 압축한다.
# 동작/맥락: trajectory_zone_ids 생성 시 동일 구역 반복 노이즈를 줄여 전술 키 안정성을 높인다.
def _collapse_consecutive(values: list[int]) -> list[int]:
    out: list[int] = []
    prev = None
    for value in values:
        if value != prev:
            out.append(value)
        prev = value
    return out


# 기능: 액션 좌표(x,y)를 3x3 zone 라벨/번호(1~9)로 매핑한다.
# 동작/맥락: zone_x/zone_y/zone_3x3/zone_id를 생성해 공격·수비 전술 그룹화의 기본 변수로 사용한다.
def _assign_zone_3x3(df: pd.DataFrame, x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    out = df.copy()

    x = pd.to_numeric(out[x_col], errors="coerce").fillna(0.0).clip(0, 105)
    y = pd.to_numeric(out[y_col], errors="coerce").fillna(0.0).clip(0, 68)

    x_idx = np.minimum((x / (105 / 3)).astype(int), 2)  # 0:def, 1:mid, 2:att
    y_idx = np.minimum((y / (68 / 3)).astype(int), 2)   # 0:left,1:center,2:right

    x_labels = np.array(["defense", "middle", "attack"])
    y_labels = np.array(["left", "center", "right"])

    out["zone_x"] = x_labels[x_idx]
    out["zone_y"] = y_labels[y_idx]
    out["zone_3x3"] = np.char.add(np.char.add(out["zone_x"].to_numpy(dtype=str), "_"), out["zone_y"].to_numpy(dtype=str))

    # numbering 1..9 with requested interpretation:
    # defense_left=1, defense_center=2, defense_right=3,
    # middle_left=4, middle_center=5, middle_right=6,
    # attack_left=7, attack_center=8, attack_right=9
    out["zone_id"] = (x_idx * 3 + y_idx + 1).astype(int)
    return out


# 기능: 소유권(team_id) 전환 중심으로 phase_id를 부여한다.
# 동작/맥락: game/period 변경과 시간 간극(max_gap_seconds)도 새 phase 조건으로 반영한다.
def _build_phase_ids_possession(actions: pd.DataFrame, max_gap_seconds: float) -> pd.DataFrame:
    ordered = actions.sort_values(["league", "game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)

    prev_game = ordered["game_id"].shift(1)
    prev_period = ordered["period_id"].shift(1)
    prev_team = ordered["team_id"].shift(1)
    prev_time = ordered["time_seconds"].shift(1)

    new_phase = (
        (ordered["game_id"] != prev_game)
        | (ordered["period_id"] != prev_period)
        | (ordered["team_id"] != prev_team)  # possession switch core rule
        | ((ordered["time_seconds"] - prev_time) > max_gap_seconds)
    )
    new_phase.iloc[0] = True

    ordered["phase_id"] = new_phase.cumsum().astype(int)
    return ordered


# 기능: phase 첫 액션 type_name을 공격/수비 및 세부 카테고리로 분류한다.
# 동작/맥락: 후속 attack_tactic_id/defense_tactic_id 생성 시 기준 레이블로 사용된다.
def _first_action_category(type_name: str) -> tuple[str | None, str | None]:
    if type_name in ATTACK_PASS_TYPES:
        return "attack", "pass"
    if type_name in ATTACK_CORNER_TYPES:
        return "attack", "corner"
    if type_name in ATTACK_FREEKICK_TYPES:
        return "attack", "freekick"

    if type_name in DEFENSE_TACKLE_TYPES:
        return "defense", "tackle"
    if type_name in DEFENSE_INTERCEPT_TYPES:
        return "defense", "interception"
    if type_name in DEFENSE_FOUL_TYPES:
        return "defense", "foul"

    return None, None


# 기능: atomic 액션에 event 단위 VAEP 값을 결합한다.
# 동작/맥락: original_event_id 기준 merge로 phase_vaep_sum 계산 입력 변수(vaep_value)를 만든다.
def _attach_vaep_to_atomic(atomic_actions: pd.DataFrame, vaep_path: Path | None) -> pd.DataFrame:
    out = atomic_actions.copy()
    out["vaep_value"] = 0.0

    if vaep_path is None or not vaep_path.exists():
        return out

    vaep = pd.read_parquet(vaep_path)
    if not {"game_id", "original_event_id", "vaep_value"}.issubset(set(vaep.columns)):
        return out

    event_vaep = (
        vaep.dropna(subset=["original_event_id"])
        .groupby(["game_id", "original_event_id"], as_index=False)
        .agg(vaep_value=("vaep_value", "mean"))
    )

    out = out.merge(event_vaep, on=["game_id", "original_event_id"], how="left", suffixes=("", "_m"))
    if "vaep_value_m" in out.columns:
        out["vaep_value"] = pd.to_numeric(out["vaep_value_m"], errors="coerce").fillna(0.0)
        out = out.drop(columns=["vaep_value_m"])
    else:
        out["vaep_value"] = pd.to_numeric(out["vaep_value"], errors="coerce").fillna(0.0)

    return out


# 기능: phase 단위 요약 테이블을 생성한다.
# 동작/맥락: trajectory, has_shot, phase_vaep_sum, attack/defense 호환 컬럼을 만들어 Phase4/5 입력으로 넘긴다.
def _build_phase_summary(all_atomic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    grouped = all_atomic.groupby("phase_id", sort=False)
    for phase_id, g in grouped:
        g = g.sort_values(["time_seconds", "action_id"]).reset_index(drop=True)

        first_type = str(g["type_name"].iloc[0])
        mode, first_cat = _first_action_category(first_type)

        zone_ids = g["zone_id"].astype(int).tolist()
        traj = _collapse_consecutive(zone_ids)
        traj_key = ">".join(map(str, traj))

        has_shot = int(g["type_name"].isin(SHOT_TYPES).any())
        phase_vaep_sum = float(pd.to_numeric(g["vaep_value"], errors="coerce").fillna(0.0).sum())

        rec = {
            "phase_id": int(phase_id),
            "league": g["league"].iloc[0],
            "game_id": int(g["game_id"].iloc[0]),
            "period_id": int(g["period_id"].iloc[0]),
            "team_id": int(g["team_id"].iloc[0]),
            "start_time": float(g["time_seconds"].min()),
            "end_time": float(g["time_seconds"].max()),
            "n_actions": int(len(g)),
            "first_action_type": first_type,
            "tactic_mode": mode,
            "first_action_category": first_cat,
            "trajectory_zone_ids": traj_key,
            "first_zone_id": int(traj[0]) if len(traj) > 0 else -1,
            "has_shot": has_shot,
            "phase_vaep_sum": phase_vaep_sum,
        }

        # compatibility columns used by next phases
        rec["attack_zone_sequence"] = traj_key if mode == "attack" else ""
        rec["defense_zone_sequence"] = str(rec["first_zone_id"]) if mode == "defense" else ""

        rows.append(rec)

    summary = pd.DataFrame(rows)
    return summary


# 기능: 공격/수비 phase를 전술 키로 묶어 tactic id를 부여한다.
# 동작/맥락: 팀별 빈도 순으로 attack_tactic_id/defense_tactic_id를 매기고 cluster 호환 컬럼을 유지한다.
def _assign_tactic_ids(phase_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = phase_summary.copy()

    attack = summary[summary["tactic_mode"] == "attack"].copy()
    defense = summary[summary["tactic_mode"] == "defense"].copy()

    if not attack.empty:
        attack_groups = (
            attack.groupby(["team_id", "first_action_category", "trajectory_zone_ids"], as_index=False)
            .size()
            .rename(columns={"size": "n_phases"})
            .sort_values(["team_id", "n_phases"], ascending=[True, False])
            .reset_index(drop=True)
        )
        attack_groups["attack_tactic_id"] = attack_groups.groupby("team_id").cumcount() + 1
        attack = attack.merge(
            attack_groups[["team_id", "first_action_category", "trajectory_zone_ids", "attack_tactic_id"]],
            on=["team_id", "first_action_category", "trajectory_zone_ids"],
            how="left",
        )
        attack["attack_tactic_id"] = attack["attack_tactic_id"].astype(int)
    else:
        attack_groups = pd.DataFrame(columns=["team_id", "first_action_category", "trajectory_zone_ids", "n_phases", "attack_tactic_id"])

    if not defense.empty:
        defense_groups = (
            defense.groupby(["team_id", "first_action_category", "first_zone_id"], as_index=False)
            .size()
            .rename(columns={"size": "n_phases"})
            .sort_values(["team_id", "n_phases"], ascending=[True, False])
            .reset_index(drop=True)
        )
        defense_groups["defense_tactic_id"] = defense_groups.groupby("team_id").cumcount() + 1
        defense = defense.merge(
            defense_groups[["team_id", "first_action_category", "first_zone_id", "defense_tactic_id"]],
            on=["team_id", "first_action_category", "first_zone_id"],
            how="left",
        )
        defense["defense_tactic_id"] = defense["defense_tactic_id"].astype(int)
    else:
        defense_groups = pd.DataFrame(columns=["team_id", "first_action_category", "first_zone_id", "n_phases", "defense_tactic_id"])

    merged = pd.concat([attack, defense, summary[summary["tactic_mode"].isna()]], ignore_index=True, sort=False)

    merged["attack_tactic_id"] = pd.to_numeric(merged.get("attack_tactic_id"), errors="coerce")
    merged["defense_tactic_id"] = pd.to_numeric(merged.get("defense_tactic_id"), errors="coerce")

    # compatibility with previous pipeline names
    merged["attack_cluster"] = merged["attack_tactic_id"]
    merged["defense_cluster"] = merged["defense_tactic_id"]

    return merged, attack_groups, defense_groups


# 기능: 전술별 사용률 및 성공률 통계를 계산한다.
# 동작/맥락: phase_count, shot_success_rate, vaep_success_mean, usage_rate를 attack/defense 각각 산출한다.
def _compute_tactic_stats(phase_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = phase_summary.copy()

    attack = summary[summary["tactic_mode"] == "attack"].copy()
    defense = summary[summary["tactic_mode"] == "defense"].copy()

    if not attack.empty:
        team_attack_total = attack.groupby("team_id").size().rename("team_attack_total")
        atk = (
            attack.groupby(["team_id", "attack_tactic_id", "first_action_category", "trajectory_zone_ids"], as_index=False)
            .agg(
                phase_count=("phase_id", "count"),
                shot_success_rate=("has_shot", "mean"),
                vaep_success_mean=("phase_vaep_sum", "mean"),
            )
        )
        atk = atk.merge(team_attack_total.reset_index(), on="team_id", how="left")
        atk["usage_rate"] = atk["phase_count"] / atk["team_attack_total"]
    else:
        atk = pd.DataFrame(columns=["team_id", "attack_tactic_id", "first_action_category", "trajectory_zone_ids", "phase_count", "shot_success_rate", "vaep_success_mean", "team_attack_total", "usage_rate"])

    if not defense.empty:
        team_def_total = defense.groupby("team_id").size().rename("team_defense_total")
        dfn = (
            defense.groupby(["team_id", "defense_tactic_id", "first_action_category", "first_zone_id"], as_index=False)
            .agg(
                phase_count=("phase_id", "count"),
                shot_success_rate=("has_shot", "mean"),
                vaep_success_mean=("phase_vaep_sum", "mean"),
            )
        )
        dfn = dfn.merge(team_def_total.reset_index(), on="team_id", how="left")
        dfn["usage_rate"] = dfn["phase_count"] / dfn["team_defense_total"]
    else:
        dfn = pd.DataFrame(columns=["team_id", "defense_tactic_id", "first_action_category", "first_zone_id", "phase_count", "shot_success_rate", "vaep_success_mean", "team_defense_total", "usage_rate"])

    return atk, dfn


# 기능: Phase3 전체 전술 추출 파이프라인을 실행한다.
# 동작/맥락: SPADL→atomic→zone/phase/tactic/stats를 순차 생성해 parquet/csv 산출물을 저장한다.
def run_phase3(
    spadl_dir: Path,
    output_dir: Path,
    leagues: list[str] | None,
    max_gap_seconds: float,
    max_games: int | None,
    vaep_path: Path | None,
) -> None:
    selected_leagues = _iter_leagues(spadl_dir, leagues)
    if not selected_leagues:
        raise RuntimeError("No SPADL parquet files found for Phase 3.")

    atomic_parts: list[pd.DataFrame] = []
    for league in selected_leagues:
        spadl_path = spadl_dir / f"spadl_{league}.parquet"
        actions = pd.read_parquet(spadl_path)

        if max_games is not None:
            keep_games = actions["game_id"].drop_duplicates().head(max_games)
            actions = actions[actions["game_id"].isin(keep_games)]

        atomic = atomic_spadl.convert_to_atomic(actions)
        atomic = atomic_spadl.add_names(atomic)
        atomic["league"] = league
        atomic_parts.append(atomic)

        print(f"[INFO] {league}: spadl={len(actions):,}, atomic={len(atomic):,}")

    all_atomic = pd.concat(atomic_parts, ignore_index=True)
    all_atomic = _assign_zone_3x3(all_atomic, x_col="x", y_col="y")
    all_atomic = _attach_vaep_to_atomic(all_atomic, vaep_path=vaep_path)
    all_atomic = _build_phase_ids_possession(all_atomic, max_gap_seconds=max_gap_seconds)

    phase_summary = _build_phase_summary(all_atomic)
    phase_summary, attack_groups, defense_groups = _assign_tactic_ids(phase_summary)

    # push tactic ids back to action-level table
    all_atomic = all_atomic.merge(
        phase_summary[["phase_id", "attack_tactic_id", "defense_tactic_id", "tactic_mode", "first_action_category", "attack_cluster", "defense_cluster"]],
        on="phase_id",
        how="left",
    )

    attack_stats, defense_stats = _compute_tactic_stats(phase_summary)

    output_dir.mkdir(parents=True, exist_ok=True)

    # action-level output (compatibility + richer columns)
    all_atomic.to_parquet(output_dir / "atomic_actions_with_phase.parquet", index=False)

    # phase-level summary (compatibility includes attack_cluster/defense_cluster + sequences)
    phase_summary.to_parquet(output_dir / "phase_summary.parquet", index=False)
    phase_summary.to_csv(output_dir / "phase_summary.csv", index=False)

    # tactic dictionaries and stats
    attack_groups.to_csv(output_dir / "attack_tactic_groups.csv", index=False)
    defense_groups.to_csv(output_dir / "defense_tactic_groups.csv", index=False)
    attack_stats.to_csv(output_dir / "attack_tactic_stats.csv", index=False)
    defense_stats.to_csv(output_dir / "defense_tactic_stats.csv", index=False)

    # keep previous-named artifact for phase4/5 continuity
    defensive_actions = all_atomic[all_atomic["tactic_mode"] == "defense"].copy()
    defensive_actions.to_parquet(output_dir / "defense_actions.parquet", index=False)

    print(f"[OK] Saved atomic actions: {output_dir / 'atomic_actions_with_phase.parquet'}")
    print(f"[OK] Saved phase summary: {output_dir / 'phase_summary.parquet'}")
    print(f"[OK] Saved attack tactic stats: {output_dir / 'attack_tactic_stats.csv'}")
    print(f"[OK] Saved defense tactic stats: {output_dir / 'defense_tactic_stats.csv'}")


# 기능: CLI 인자를 파싱하고 run_phase3를 호출하는 진입점이다.
# 동작/맥락: 리그/간격/VAEP 경로 설정을 받아 동일 로직을 재실행 가능하게 한다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: possession-based tactic clustering with 3x3 zones"
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
        default=Path("/workspace/ai 라인업/데이터/tactics"),
        help="Directory to write Phase 3 outputs",
    )
    parser.add_argument("--leagues", nargs="*", default=None, help="Optional league list")
    parser.add_argument("--max-gap-seconds", type=float, default=20.0)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument(
        "--vaep-path",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/vaep/vaep_actions.parquet"),
        help="Optional VAEP actions parquet for VAEP-based success metric",
    )

    args = parser.parse_args()

    run_phase3(
        spadl_dir=args.spadl_dir,
        output_dir=args.output_dir,
        leagues=args.leagues,
        max_gap_seconds=args.max_gap_seconds,
        max_games=args.max_games,
        vaep_path=args.vaep_path,
    )


if __name__ == "__main__":
    main()
