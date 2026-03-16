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
import ast
from pathlib import Path

import pandas as pd
import pulp


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


def _parse_formation(formation: str) -> tuple[int, int, int]:
    parts = formation.split("-")
    if len(parts) != 3:
        raise ValueError("formation must be in format DF-MF-FW (e.g. 4-3-3)")
    df_n, mf_n, fw_n = [int(x) for x in parts]
    if df_n + mf_n + fw_n != 10:
        raise ValueError("formation must sum to 10 outfield players (GK is fixed to 1)")
    return df_n, mf_n, fw_n


def _build_event_vaep_map(vaep_actions: pd.DataFrame) -> pd.DataFrame:
    return (
        vaep_actions.dropna(subset=["original_event_id"])
        .groupby(["game_id", "original_event_id"], as_index=False)
        .agg(vaep_value=("vaep_value", "mean"))
    )


def _contains_any_keyword(text: str | None, keywords: list[str]) -> bool:
    if not keywords:
        return False
    text_norm = str(text or "").lower()
    return any(k.lower() in text_norm for k in keywords)


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

    use_weight = (
        pairs["attack_cluster"].fillna(-999).astype(int).isin(attack_clusters)
        | pairs["attack_zone_sequence"].map(lambda s: _contains_any_keyword(s, attack_keywords))
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

    trans["id_raw"] = trans["last_vaep"] + trans["next_first_vaep"]

    use_weight = (
        trans["next_defense_cluster"].fillna(-999).astype(int).isin(defense_clusters)
        | trans["next_defense_zone_sequence"].map(lambda s: _contains_any_keyword(s, defense_keywords))
    )
    trans["tactic_weight"] = 1.0
    trans.loc[use_weight, "tactic_weight"] = defense_weight
    trans["id_weighted"] = trans["id_raw"] * trans["tactic_weight"]

    id_player = (
        trans.groupby("next_first_player_id", as_index=False)
        .agg(
            id_sum=("id_weighted", "sum"),
            id_count=("id_weighted", "count"),
            id_weighted_share=("tactic_weight", "mean"),
        )
        .rename(columns={"next_first_player_id": "player_id"})
    )
    id_player["player_id"] = id_player["player_id"].astype(int)
    return id_player


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
) -> None:
    player_scores = pd.read_parquet(synergy_dir / "player_synergy_scores.parquet")
    vaep_actions = pd.read_parquet(Path("/workspace/ai 라인업/데이터/vaep/vaep_actions.parquet"))
    atomic_with_phase = pd.read_parquet(tactics_dir / "atomic_actions_with_phase.parquet")
    phase_summary = pd.read_parquet(tactics_dir / "phase_summary.parquet")
    players = pd.read_csv(archive_dir / "players.csv")

    if team_id is None:
        team_id = int(player_scores["team_id"].value_counts().index[0])

    team_players = player_scores[player_scores["team_id"] == team_id].copy()
    if team_players.empty:
        raise RuntimeError(f"No player scores found for team_id={team_id}")

    players_meta = players[["wyId", "shortName", "role"]].copy()
    players_meta["position"] = players_meta["role"].map(_parse_role_code)
    players_meta = players_meta.rename(columns={"wyId": "player_id", "shortName": "player_name"})

    team_players = team_players.merge(players_meta[["player_id", "player_name", "position"]], on="player_id", how="left")
    team_players = team_players[team_players["position"].isin(["GK", "DF", "MF", "FW"])].copy()

    if team_players.empty:
        raise RuntimeError("No players with valid positions (GK/DF/MF/FW) after merge with players metadata.")

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

    id_player = _compute_weighted_id(
        atomic_with_phase=atomic_with_phase,
        phase_summary=phase_summary,
        event_vaep=event_vaep,
        target_team_id=team_id,
        defense_weight=defense_weight,
        defense_clusters=set(defense_clusters),
        defense_keywords=defense_keywords,
    )

    team_players = team_players.merge(id_player, on="player_id", how="left")
    team_players[["id_sum", "id_count", "id_weighted_share"]] = team_players[
        ["id_sum", "id_count", "id_weighted_share"]
    ].fillna(0.0)

    df_n, mf_n, fw_n = _parse_formation(formation)

    pos_need = {"GK": 1, "DF": df_n, "MF": mf_n, "FW": fw_n}
    for pos, need in pos_need.items():
        have = int((team_players["position"] == pos).sum())
        if have < need:
            raise RuntimeError(f"Not enough {pos} players for formation {formation}: need {need}, have {have}")

    player_ids = team_players["player_id"].astype(int).tolist()
    player_set = set(player_ids)

    io_pair = io_pair[io_pair["player_a_id"].isin(player_set) & io_pair["player_b_id"].isin(player_set)].copy()

    vi_map = team_players.set_index("player_id")["vi"].to_dict()
    id_map = team_players.set_index("player_id")["id_sum"].to_dict()
    pos_map = team_players.set_index("player_id")["position"].to_dict()

    io_map = {(int(r.player_a_id), int(r.player_b_id)): float(r.io_sum) for r in io_pair.itertuples(index=False)}

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
    if status_name not in {"Optimal", "Not Solved", "Undefined", "Infeasible", "Unbounded"}:
        raise RuntimeError(f"Unexpected solver status: {status_name}")
    if status_name != "Optimal":
        raise RuntimeError(f"ILP did not find optimal solution. status={status_name}")

    selected = [pid for pid in player_ids if pulp.value(x[pid]) > 0.5]

    lineup = team_players[team_players["player_id"].isin(selected)].copy()
    lineup["selected"] = 1
    lineup["vi_component"] = lambda_vi * lineup["vi"]
    lineup["id_component"] = lambda_id * lineup["id_sum"]

    selected_pairs = []
    for (a, b), y_var in y.items():
        if pulp.value(y_var) > 0.5:
            selected_pairs.append({"player_a_id": a, "player_b_id": b, "io_pair_value": pair_coef[(a, b)]})

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
    selected_pairs_df.to_csv(output_dir / "lineup_selected_io_pairs.csv", index=False)
    selected_pairs_df.to_parquet(output_dir / "lineup_selected_io_pairs.parquet", index=False)
    summary.to_csv(output_dir / "lineup_optimization_summary.csv", index=False)
    summary.to_parquet(output_dir / "lineup_optimization_summary.parquet", index=False)

    print(f"[OK] team_id={team_id}, formation={formation}, status={status_name}")
    print(f"[OK] objective_total={realized_total:.6f} (V_I={realized_vi:.6f}, I_O={realized_io:.6f}, I_D={realized_id:.6f})")
    print(f"[OK] saved: {output_dir / 'lineup_selected.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: ILP lineup optimization with tactic-weighted synergy")
    parser.add_argument(
        "--synergy-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/synergy"),
        help="Directory containing Phase 4 outputs",
    )
    parser.add_argument(
        "--tactics-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/tactics"),
        help="Directory containing Phase 3 outputs",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/archive"),
        help="Directory containing players.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/ai 라인업/데이터/lineup"),
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
    )


if __name__ == "__main__":
    main()
