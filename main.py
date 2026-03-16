from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="Team Builder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "데이터"
RUNTIME_DIR = BASE_DIR / "team-builder-dashboard" / "runtime"

TEAM_ALIAS = {
    1611: "맨유 (Manchester United)",
    1625: "맨시티 (Manchester City)",
    1644: "리버풀 (Liverpool)",
    1628: "첼시 (Chelsea)",
    1651: "아스널 (Arsenal)",
    1633: "토트넘 (Tottenham)",
}


class MinMax(BaseModel):
    min: int
    max: int


class FormationConstraints(BaseModel):
    forwards: MinMax
    midfielders: MinMax
    defenders: MinMax
    goalkeeper: Optional[MinMax] = MinMax(min=1, max=1)


class Constraints(BaseModel):
    formation: Optional[str] = "4-3-3"
    locked_players: Optional[List[int]] = []
    excluded_players: Optional[List[int]] = []
    formation_constraints: Optional[FormationConstraints] = None


class OptimizeRequest(BaseModel):
    team_id: int
    opponent_id: int
    opponent_team_id: Optional[int] = None
    tactic_weights: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    constraints: Optional[Constraints] = None
    view_mode: Optional[Literal["offensive", "defensive"]] = "offensive"


def _scale(value: float, weight: float) -> float:
    return round(max(0.01, min(0.99, value * (0.8 + 0.2 * weight))), 3)


def _safe_int_list(values: Optional[List[int]]) -> List[int]:
    if not values:
        return []
    out = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _resolve_formation(req: OptimizeRequest) -> tuple[str, int, int, int]:
    fc = req.constraints.formation_constraints if req.constraints and req.constraints.formation_constraints else None
    if fc is not None:
        f_count = max(1, min(5, int(fc.forwards.min)))
        m_count = max(1, min(5, int(fc.midfielders.min)))
        d_count = max(1, min(5, int(fc.defenders.min)))
        while f_count + m_count + d_count > 10:
            if d_count > 1:
                d_count -= 1
            elif m_count > 1:
                m_count -= 1
            else:
                f_count -= 1
        while f_count + m_count + d_count < 10:
            if d_count < 5:
                d_count += 1
            elif m_count < 5:
                m_count += 1
            else:
                f_count += 1
        return f"{d_count}-{m_count}-{f_count}", d_count, m_count, f_count

    formation = (req.constraints.formation if req.constraints and req.constraints.formation else "4-3-3") or "4-3-3"
    try:
        d, m, f = [int(x) for x in formation.split("-")]
        if d + m + f != 10:
            raise ValueError
        return formation, d, m, f
    except Exception:
        return "4-3-3", 4, 3, 3


@lru_cache(maxsize=1)
def _load_phase5_module():
    phase5_path = BASE_DIR / "phase_5_lineup" / "code" / "optimize_lineup_phase5.py"
    spec = importlib.util.spec_from_file_location("phase5_module", phase5_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase5 module from {phase5_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_phase5_realtime(req: OptimizeRequest) -> Optional[pd.DataFrame]:
    try:
        phase5 = _load_phase5_module()
    except Exception:
        return None

    formation, _, _, _ = _resolve_formation(req)
    lock_ids = _safe_int_list(req.constraints.locked_players if req.constraints else [])
    exclude_ids = _safe_int_list(req.constraints.excluded_players if req.constraints else [])

    run_dir = RUNTIME_DIR / "phase5_requests" / f"{req.team_id}_{req.opponent_id}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        phase5.run_phase5(
            synergy_dir=DATA_DIR / "synergy",
            tactics_dir=DATA_DIR / "tactics",
            archive_dir=DATA_DIR / "archive",
            output_dir=run_dir,
            team_id=int(req.team_id),
            opponent_team_id=int(req.opponent_id),
            formation=formation,
            lambda_vi=1.0,
            lambda_io=1.0,
            lambda_id=1.0,
            attack_weight=1.0,
            defense_weight=1.0,
            attack_clusters=[],
            defense_clusters=[],
            attack_keywords=[],
            defense_keywords=[],
            attack_tactic_weights_csv=None,
            defense_tactic_weights_csv=None,
            lock_player_ids=lock_ids,
            exclude_player_ids=exclude_ids,
            use_estimated_lambda=True,
            solver_time_limit=45,
        )
    except Exception:
        return None

    out_csv = run_dir / "lineup_selected.csv"
    if not out_csv.exists():
        return None
    try:
        return pd.read_csv(out_csv)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@lru_cache(maxsize=1)
def _load_players_meta() -> pd.DataFrame:
    players = _load_csv(str(DATA_DIR / "archive/players.csv"))
    if players.empty:
        return pd.DataFrame(columns=["wyId", "shortName", "role"])
    cols = [c for c in ["wyId", "shortName", "role"] if c in players.columns]
    out = players[cols].copy()
    if "shortName" not in out.columns:
        out["shortName"] = out["wyId"].astype(str)
    if "role" not in out.columns:
        out["role"] = ""
    return out


def _role_to_position(role_text: Any) -> str:
    s = str(role_text or "").upper()
    if "'CODE2': 'GK'" in s or '"CODE2": "GK"' in s:
        return "GK"
    if "'CODE2': 'DF'" in s or '"CODE2": "DF"' in s:
        return "DF"
    if "'CODE2': 'FW'" in s or '"CODE2": "FW"' in s:
        return "FW"
    return "MF"


@lru_cache(maxsize=1)
def _player_name_map() -> Dict[int, str]:
    meta = _load_players_meta()
    if meta.empty:
        return {}
    return {int(r["wyId"]): str(r["shortName"]) for _, r in meta.dropna(subset=["wyId"]).iterrows()}


@lru_cache(maxsize=1)
def _player_position_map() -> Dict[int, str]:
    meta = _load_players_meta()
    if meta.empty:
        return {}
    out: Dict[int, str] = {}
    for _, r in meta.dropna(subset=["wyId"]).iterrows():
        out[int(r["wyId"])] = _role_to_position(r.get("role"))
    return out


def _zone_to_xy(zone_id: int) -> List[int]:
    zid = max(1, min(9, int(zone_id)))
    row = (zid - 1) // 3
    col = (zid - 1) % 3
    x = int(round(16.67 + col * 33.33))
    y = int(round(16.67 + row * 33.33))
    return [x, y]


def _parse_trajectory(raw: Any) -> List[int]:
    s = str(raw or "").strip()
    if not s:
        return [2, 5, 8]
    for sep in ["-", ",", " ", ">"]:
        if sep in s:
            out = [int(float(x)) for x in s.split(sep) if str(x).strip().replace(".", "", 1).isdigit()]
            if out:
                return out[:4]
    if s.replace(".", "", 1).isdigit():
        return [int(float(s))]
    return [2, 5, 8]


def _format_action(raw: Any) -> str:
    s = " ".join(str(raw or "pass").split()).strip().lower()
    if s.startswith("free") or s == "freekick":
        return "Free kick"
    if s.startswith("corner"):
        return "Corner kick"
    if s.startswith("inter"):
        return "Intercept"
    if s.startswith("tackle"):
        return "Tackle"
    if s.startswith("foul"):
        return "Foul"
    return "Pass"


def _select_top_tactics(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    use_cols = [id_col, "usage_rate", "shot_success_rate"]
    base = df[use_cols].copy().dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    top_usage = base.sort_values("usage_rate", ascending=False).head(3)
    top_success = base.sort_values("shot_success_rate", ascending=False).head(3)
    chosen_ids = pd.concat([top_usage[id_col], top_success[id_col]], ignore_index=True).drop_duplicates().tolist()
    return df[df[id_col].isin(chosen_ids)].copy().head(6)


def _build_tactics(team_id: int, opponent_id: int, weights: Dict[str, Dict[str, float]]) -> Dict[str, List[Dict[str, Any]]]:
    atk_stats = _load_csv(str(DATA_DIR / "tactics/attack_tactic_stats.csv"))
    dfn_stats = _load_csv(str(DATA_DIR / "tactics/defense_tactic_stats.csv"))

    def build_attack(df: pd.DataFrame, prefix: str, weight_key: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        top = _select_top_tactics(df, "attack_tactic_id")
        for _, row in top.iterrows():
            tid = int(row["attack_tactic_id"])
            item_id = f"{prefix}_{tid}"
            w = float(weights.get(weight_key, {}).get(item_id, 1.0))
            zones = _parse_trajectory(row.get("trajectory_zone_ids"))
            path = [_zone_to_xy(z) for z in zones]
            if len(path) == 1:
                path = [path[0], [min(94, path[0][0] + 18), max(6, path[0][1] - 10)]]
            out.append(
                {
                    "id": item_id,
                    "name": f"Attack Tactic {tid}",
                    "startAction": _format_action(row.get("first_action_category")),
                    "weight": round(w, 2),
                    "frequency": _scale(float(row.get("usage_rate", 0.0)), w),
                    "successRate": _scale(float(row.get("shot_success_rate", 0.0)), w),
                    "path": path,
                }
            )
        return out

    def build_defense(df: pd.DataFrame, prefix: str, weight_key: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        top = _select_top_tactics(df, "defense_tactic_id")
        for _, row in top.iterrows():
            tid = int(row["defense_tactic_id"])
            item_id = f"{prefix}_{tid}"
            w = float(weights.get(weight_key, {}).get(item_id, 1.0))
            start = _zone_to_xy(int(row.get("first_zone_id", 5)))
            path = [start, [50, 50]]
            out.append(
                {
                    "id": item_id,
                    "name": f"Defense Tactic {tid}",
                    "startAction": _format_action(row.get("first_action_category")),
                    "weight": round(w, 2),
                    "frequency": _scale(float(row.get("usage_rate", 0.0)), w),
                    "successRate": _scale(float(row.get("shot_success_rate", 0.0)), w),
                    "path": path,
                }
            )
        return out

    home_atk = atk_stats[atk_stats.get("team_id") == team_id].copy()
    home_dfn = dfn_stats[dfn_stats.get("team_id") == team_id].copy()
    away_atk = atk_stats[atk_stats.get("team_id") == opponent_id].copy()
    away_dfn = dfn_stats[dfn_stats.get("team_id") == opponent_id].copy()

    return {
        "home_attack": build_attack(home_atk, "ha", "homeAttack"),
        "home_defense": build_defense(home_dfn, "hd", "homeDefense"),
        "away_attack": build_attack(away_atk, "aa", "awayAttack"),
        "away_defense": build_defense(away_dfn, "ad", "awayDefense"),
    }

def _layout_players(players: List[Dict[str, Any]], d_count: int, m_count: int, f_count: int) -> List[Dict[str, Any]]:
    d_x = [int(round(v)) for v in (pd.Series(range(d_count)) * (70 / max(1, d_count - 1)) + 15).tolist()] if d_count > 1 else [50]
    m_x = [int(round(v)) for v in (pd.Series(range(m_count)) * (64 / max(1, m_count - 1)) + 18).tolist()] if m_count > 1 else [50]
    f_x = [int(round(v)) for v in (pd.Series(range(f_count)) * (64 / max(1, f_count - 1)) + 18).tolist()] if f_count > 1 else [50]

    out = []
    gk = [p for p in players if p["position"] == "GK"]
    dfs = [p for p in players if p["position"] == "DF"]
    mfs = [p for p in players if p["position"] == "MF"]
    fws = [p for p in players if p["position"] == "FW"]

    if gk:
        p = gk[0].copy()
        p["x"], p["y"] = 50, 92
        out.append(p)
    for idx, p in enumerate(dfs[:d_count]):
        q = p.copy()
        q["x"], q["y"] = d_x[idx], 72
        out.append(q)
    for idx, p in enumerate(mfs[:m_count]):
        q = p.copy()
        q["x"], q["y"] = m_x[idx], 50
        out.append(q)
    for idx, p in enumerate(fws[:f_count]):
        q = p.copy()
        q["x"], q["y"] = f_x[idx], 28
        out.append(q)
    return out[:11]


def _build_lineup(req: OptimizeRequest, realtime_selected: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    _, d_count, m_count, f_count = _resolve_formation(req)

    name_map = _player_name_map()
    pos_map = _player_position_map()

    lineup_df = _load_csv(str(DATA_DIR / "lineup/lineup_selected.csv"))
    synergy_df = _load_csv(str(DATA_DIR / "synergy/player_synergy_scores.csv"))

    selected_src = realtime_selected.copy() if realtime_selected is not None and not realtime_selected.empty else pd.DataFrame()
    if selected_src.empty:
        selected_src = lineup_df[lineup_df.get("team_id") == req.team_id].copy() if not lineup_df.empty else pd.DataFrame()
    if selected_src.empty and not synergy_df.empty:
        selected_src = synergy_df[synergy_df.get("team_id") == req.team_id].sort_values("v_total", ascending=False).head(24).copy()
        selected_src["player_name"] = selected_src["player_id"].map(name_map)
        selected_src["position"] = selected_src["player_id"].map(pos_map)

    all_players_df = synergy_df[synergy_df.get("team_id") == req.team_id].copy() if not synergy_df.empty else pd.DataFrame()
    if all_players_df.empty:
        all_players_df = selected_src.copy()
    all_players_df["player_name"] = all_players_df.get("player_name", pd.Series([""] * len(all_players_df))).fillna(all_players_df.get("player_id").map(name_map)).fillna("Unknown")
    if "position" not in all_players_df.columns:
        all_players_df["position"] = all_players_df.get("player_id").map(pos_map)
    all_players_df["position"] = all_players_df["position"].fillna(all_players_df.get("player_id").map(pos_map)).fillna("MF")

    if "position" not in selected_src.columns:
        selected_src["position"] = selected_src["player_id"].map(pos_map)
    selected_src["position"] = selected_src["position"].fillna(selected_src.get("player_id").map(pos_map)).fillna("MF")
    selected_src["player_name"] = selected_src.get("player_name", pd.Series([""] * len(selected_src))).fillna(selected_src.get("player_id").map(name_map)).fillna("Unknown")
    selected_src = selected_src.sort_values("v_total", ascending=False)

    gk_pool = selected_src[selected_src["position"] == "GK"]
    if gk_pool.empty:
        gk_pool = all_players_df[all_players_df.get("position") == "GK"] if not all_players_df.empty else pd.DataFrame()
    gk_row = gk_pool.sort_values("v_total", ascending=False).head(1)

    df_pool = selected_src[selected_src["position"] == "DF"].sort_values("v_total", ascending=False)
    mf_pool = selected_src[selected_src["position"] == "MF"].sort_values("v_total", ascending=False)
    fw_pool = selected_src[selected_src["position"] == "FW"].sort_values("v_total", ascending=False)

    picked = pd.concat([
        gk_row,
        df_pool.head(d_count),
        mf_pool.head(m_count),
        fw_pool.head(f_count),
    ], ignore_index=True)

    if len(picked) < 11:
        remaining = selected_src[~selected_src["player_id"].isin(picked.get("player_id", pd.Series(dtype=int)))].head(11 - len(picked))
        picked = pd.concat([picked, remaining], ignore_index=True)

    def to_player(row: pd.Series) -> Dict[str, Any]:
        pid = int(row.get("player_id", 0))
        return {
            "id": pid,
            "number": pid,
            "name": str(row.get("player_name") or name_map.get(pid, f"P{pid}")),
            "position": str(row.get("position") or pos_map.get(pid, "MF")),
            "vi": float(row.get("vi", 0.0)),
            "io": float(row.get("io", row.get("io_player_sum", 0.0))),
            "idv": float(row.get("id_sum", row.get("id", 0.0))),
        }

    selected = _layout_players([to_player(r) for _, r in picked.head(11).iterrows()], d_count=d_count, m_count=m_count, f_count=f_count)

    candidate_src = all_players_df.copy() if not all_players_df.empty else selected_src.copy()
    candidate_src["player_name"] = candidate_src.get("player_name", pd.Series([""] * len(candidate_src))).fillna(candidate_src.get("player_id").map(name_map)).fillna("Unknown")
    if "position" not in candidate_src.columns:
        candidate_src["position"] = candidate_src["player_id"].map(pos_map).fillna("MF")
    candidate_src = candidate_src[~candidate_src["player_id"].isin([p["id"] for p in selected])].sort_values("v_total", ascending=False)
    candidate = [to_player(r) for _, r in candidate_src.head(7).iterrows()]

    opp_src = synergy_df[synergy_df.get("team_id") == req.opponent_id].copy() if not synergy_df.empty else pd.DataFrame()
    if not opp_src.empty:
        opp_src = opp_src.sort_values("v_total", ascending=False).head(11)
        opp = []
        for _, r in opp_src.iterrows():
            pid = int(r.get("player_id", 0))
            opp.append({"id": pid, "number": pid, "name": name_map.get(pid, f"Opp {pid}")})
    else:
        opp = [{"id": 200 + idx, "number": 200 + idx, "name": f"Opp {idx + 1}"} for idx in range(11)]

    return {
        "selected_players": selected,
        "candidate_players": candidate,
        "opponent_players": opp,
        "all_players": [*selected, *candidate],
    }


def _build_alternatives(selected: List[Dict[str, Any]], candidate: List[Dict[str, Any]], team_id: int, opponent_id: int) -> List[Dict[str, Any]]:
    sim = _load_csv(str(DATA_DIR / "lineup/lineup_optimization_summary.csv"))
    wr_base = 0.55
    ap_base = 8
    if not sim.empty:
        sub = sim[(sim.get("team_id") == team_id) & (sim.get("opponent_team_id") == opponent_id)]
        use = sub if not sub.empty else sim[sim.get("team_id") == team_id]
        if not use.empty:
            conf = pd.to_numeric(use.get("confidence_score"), errors="coerce").dropna()
            wr_base = float(0.4 + 0.005 * conf.mean()) if len(conf) else 0.55
            ap_base = int(max(3, min(20, len(use) * 4)))

    def make_alt(name: str, wr: float, ap: int, shift: int) -> Dict[str, Any]:
        players = []
        points: List[Tuple[int, int]] = []
        for idx, p in enumerate(selected):
            px = max(8, min(92, int(p["x"] + ((idx % 3) - 1) * shift)))
            py = max(14, min(92, int(p["y"] + ((idx % 2) * shift // 2))))
            players.append({**p, "x": px, "y": py})
            points.append((px, py))
        if candidate:
            swap_idx = min(10, max(1, len(players) - 1))
            c = candidate[(ap + shift) % len(candidate)]
            players[swap_idx] = {**players[swap_idx], "id": c["id"], "number": c["number"], "name": c["name"], "vi": c.get("vi", 0.0), "io": c.get("io", 0.0), "idv": c.get("idv", 0.0)}
        return {
            "name": name,
            "winning_rate": wr,
            "appear_count": ap,
            "expected_stats": {
                "shots": round(11.5 + wr * 4, 1),
                "goals": round(1.2 + wr * 1.3, 2),
                "intercepts": round(8.0 + (1 - wr) * 3, 1),
                "tackles": round(13.0 + (1 - wr) * 4, 1),
            },
            "formation_points": points,
            "players": players,
        }

    return [
        make_alt("대안 1", round(min(0.95, wr_base - 0.03), 3), ap_base + 4, 0),
        make_alt("대안 2", round(min(0.95, wr_base), 3), ap_base, 3),
        make_alt("대안 3", round(min(0.95, wr_base + 0.03), 3), max(3, ap_base - 2), -3),
    ]


def _build_synergy(selected: List[Dict[str, Any]], opponents: List[Dict[str, Any]], team_id: int, opponent_id: int) -> Dict[str, Any]:
    io_df = _load_csv(str(DATA_DIR / "synergy/attack_interaction_io.csv"))
    id_df = _load_csv(str(DATA_DIR / "synergy/defense_interaction_id.csv"))

    io_map: Dict[Tuple[int, int], float] = {}
    if not io_df.empty:
        use = io_df[io_df.get("team_id") == team_id]
        for _, r in use.iterrows():
            a = int(r.get("player_a_id", 0))
            b = int(r.get("player_b_id", 0))
            io_val = float(r.get("io", r.get("io_90_raw", 0.0)))
            io_map[(min(a, b), max(a, b))] = io_val

    id_map: Dict[Tuple[int, int], float] = {}
    if not id_df.empty:
        use = id_df[(id_df.get("defending_team_id") == team_id) & (id_df.get("opponent_team_id") == opponent_id)]
        for _, r in use.iterrows():
            a = int(r.get("defender_player_id", 0))
            b = int(r.get("opponent_player_id", 0))
            id_val = float(r.get("id", r.get("id_90_raw", 0.0)))
            id_map[(a, b)] = id_val

    y_players = [{"number": p["number"], "name": p["name"]} for p in selected]
    fw = [{"number": p["number"], "name": p["name"]} for p in selected if p.get("position") == "FW"]
    mf = [{"number": p["number"], "name": p["name"]} for p in selected if p.get("position") == "MF"]
    df = [{"number": p["number"], "name": p["name"]} for p in selected if p.get("position") == "DF"]
    gk = [{"number": p["number"], "name": p["name"]} for p in selected if p.get("position") == "GK"]
    x_groups = [{"group": "F", "players": fw}, {"group": "M", "players": mf}, {"group": "D", "players": df}, {"group": "G", "players": gk}]

    offensive_matrix: List[List[float]] = []
    for r in selected:
        row: List[float] = []
        for c in selected:
            if int(r["id"]) == int(c["id"]):
                row.append(round(float(r.get("vi", 0.0)), 3))
            else:
                key = (min(int(r["id"]), int(c["id"])), max(int(r["id"]), int(c["id"])))
                row.append(round(float(io_map.get(key, 0.0)), 3))
        offensive_matrix.append(row)

    opp_x = [{"number": p["number"], "name": p["name"]} for p in opponents]
    defensive_matrix: List[List[float]] = []
    for r in selected:
        row: List[float] = []
        for c in opponents:
            row.append(round(float(id_map.get((int(r["id"]), int(c["id"])), 0.0)), 3))
        defensive_matrix.append(row)

    return {
        "offensive": {
            "y_players": y_players,
            "x_groups": x_groups,
            "matrix": offensive_matrix,
        },
        "defensive": {
            "opponent_x_players": opp_x,
            "matrix": defensive_matrix,
        },
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/teams")
def get_teams() -> Dict[str, Any]:
    teams = _load_csv(str(DATA_DIR / "archive/teams.csv"))
    if teams.empty:
        return {"teams": [{"id": k, "name": v} for k, v in TEAM_ALIAS.items()]}

    team_ids = sorted(set([1611, 1625, 1644, 1628, 1651, 1633]))
    rows = teams[teams.get("wyId").isin(team_ids)] if "wyId" in teams.columns else pd.DataFrame()
    out = []
    for tid in team_ids:
        name = TEAM_ALIAS.get(tid)
        if rows.empty:
            out.append({"id": tid, "name": name or f"Team {tid}"})
            continue
        row = rows[rows["wyId"] == tid]
        if row.empty:
            out.append({"id": tid, "name": name or f"Team {tid}"})
            continue
        eng = str(row.iloc[0].get("name", f"Team {tid}"))
        out.append({"id": tid, "name": name or eng})
    return {"teams": out}


@app.post("/api/optimize_lineup")
def optimize_lineup(req: OptimizeRequest) -> Dict[str, Any]:
    tactics = _build_tactics(req.team_id, req.opponent_id, req.tactic_weights)
    realtime_selected = _run_phase5_realtime(req)
    lineup = _build_lineup(req, realtime_selected=realtime_selected)
    alternatives = _build_alternatives(lineup["selected_players"], lineup["candidate_players"], req.team_id, req.opponent_id)
    synergy = _build_synergy(lineup["selected_players"], lineup["opponent_players"], req.team_id, req.opponent_id)

    return {
        "tactics": tactics,
        "lineup": lineup,
        "alternatives": alternatives,
        "synergy": synergy,
    }


# run:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
