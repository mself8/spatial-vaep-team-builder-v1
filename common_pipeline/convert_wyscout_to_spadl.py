#!/usr/bin/env python3
"""Batch-convert Wyscout CSV event data to SPADL using socceraction."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


PERIOD_MAP = {
    "1H": 1,
    "2H": 2,
    "E1": 3,
    "E2": 4,
    "P": 5,
}


# 기능: 문자열로 저장된 리스트/딕셔너리(JSON 유사 텍스트)를 파이썬 객체로 안전 변환한다.
# 동작: NaN/빈문자 처리 후 ast.literal_eval을 시도하고, 실패 시 빈 리스트를 반환한다.
# 입출력/사용: _prepare_events, _load_home_teams 내부에서 positions/tags/teamsData 전처리에 사용된다.
def _safe_literal(value):
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []


# 기능: 입력 파일 확장자(csv/json/jsonl/ndjson)에 맞춰 테이블을 DataFrame으로 로드한다.
# 동작: 경로 존재 확인 후 포맷별 파싱 로직을 분기하고 표준화된 DataFrame을 반환한다.
# 입출력/사용: _load_players, convert_league, _load_home_teams가 원천 데이터 적재에 사용한다.
def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".json", ".jsonl", ".ndjson"}:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return pd.DataFrame()

        if suffix in {".jsonl", ".ndjson"}:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            return pd.DataFrame(records)

        loaded = json.loads(text)
        if isinstance(loaded, dict):
            for key in ("data", "events", "matches", "players"):
                if key in loaded and isinstance(loaded[key], list):
                    return pd.DataFrame(loaded[key])
            return pd.json_normalize(loaded)
        if isinstance(loaded, list):
            return pd.DataFrame(loaded)
        raise ValueError(f"Unsupported JSON structure in {path}")

    raise ValueError(f"Unsupported file extension: {path.suffix}")


# 기능: stem 이름(events_England 등)에 맞는 실제 입력 파일 경로를 탐색한다.
# 동작: 지원 확장자를 순회하며 첫 번째 존재 파일을 반환하고, 없으면 예외를 발생시킨다.
# 입출력/사용: _load_players, convert_league에서 players/events/matches 파일 경로 확보에 사용된다.
def _find_input_file(data_dir: Path, stem: str) -> Path:
    for ext in (".csv", ".json", ".jsonl", ".ndjson"):
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No input file found for {stem} with supported extensions")


# 기능: players 원천 파일을 읽어 선수 메타 데이터를 확보한다.
# 동작: _find_input_file + _load_table를 호출하고 비어있으면 예외를 던진다.
# 입출력/사용: main에서 데이터 무결성 사전 체크(로드 가능 여부) 목적으로 사용한다.
def _load_players(data_dir: Path) -> pd.DataFrame:
    players_path = _find_input_file(data_dir, "players")
    players = _load_table(players_path)
    if players.empty:
        raise RuntimeError(f"Players file is empty: {players_path}")
    return players


def _extract_goalkeeper_ids(players: pd.DataFrame) -> set[int]:
    if players.empty or "wyId" not in players.columns:
        return set()
    role_col = players["role"].astype(str) if "role" in players.columns else pd.Series("", index=players.index)
    gk_mask = role_col.str.contains("'CODE2': 'GK'|\"CODE2\": \"GK\"", regex=True, na=False)
    ids = pd.to_numeric(players.loc[gk_mask, "wyId"], errors="coerce").dropna().astype(int)
    return set(ids.tolist())


# 기능: Wyscout 이벤트 컬럼을 socceraction 변환 입력 스키마로 정규화한다.
# 동작: 컬럼명 매핑, period/milliseconds 파생, 필수 컬럼 검증, 타입 캐스팅/결측 제거를 수행한다.
# 입출력/사용: _prepare_events가 호출하며 convert_league의 wyscout_spadl.convert_to_actions 입력이 된다.
def _normalize_wyscout_columns(events: pd.DataFrame) -> pd.DataFrame:
    renamed = events.rename(
        columns={
            "id": "event_id",
            "eventId": "type_id",
            "subEventId": "subtype_id",
            "playerId": "player_id",
            "teamId": "team_id",
            "matchId": "game_id",
            "eventSec": "seconds",
            "matchPeriod": "period",
        }
    ).copy()

    if "period_id" not in renamed.columns:
        if "period" in renamed.columns:
            renamed["period_id"] = (
                renamed["period"].astype(str).str.strip().map(PERIOD_MAP)
            )
        elif "matchPeriod" in renamed.columns:
            renamed["period_id"] = (
                renamed["matchPeriod"].astype(str).str.strip().map(PERIOD_MAP)
            )

    if "milliseconds" not in renamed.columns:
        if "seconds" in renamed.columns:
            renamed["milliseconds"] = pd.to_numeric(
                renamed["seconds"], errors="coerce"
            ) * 1000.0
        elif "eventSec" in renamed.columns:
            renamed["milliseconds"] = pd.to_numeric(
                renamed["eventSec"], errors="coerce"
            ) * 1000.0

    required_cols = [
        "event_id",
        "type_id",
        "subtype_id",
        "player_id",
        "team_id",
        "game_id",
        "period_id",
        "milliseconds",
    ]
    missing = [col for col in required_cols if col not in renamed.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for Wyscout->SPADL conversion: {missing}"
        )

    renamed["event_id"] = pd.to_numeric(renamed["event_id"], errors="coerce")
    renamed["type_id"] = pd.to_numeric(renamed["type_id"], errors="coerce")
    renamed["subtype_id"] = pd.to_numeric(renamed["subtype_id"], errors="coerce")
    renamed["player_id"] = pd.to_numeric(renamed["player_id"], errors="coerce")
    renamed["team_id"] = pd.to_numeric(renamed["team_id"], errors="coerce")
    renamed["game_id"] = pd.to_numeric(renamed["game_id"], errors="coerce")
    renamed["period_id"] = pd.to_numeric(renamed["period_id"], errors="coerce")
    renamed["milliseconds"] = pd.to_numeric(renamed["milliseconds"], errors="coerce")

    renamed = renamed.dropna(
        subset=[
            "event_id",
            "type_id",
            "subtype_id",
            "player_id",
            "team_id",
            "game_id",
            "period_id",
            "milliseconds",
        ]
    )

    int_cols = ["event_id", "type_id", "subtype_id", "player_id", "team_id", "game_id", "period_id"]
    for col in int_cols:
        renamed[col] = renamed[col].astype(int)

    return renamed


# 기능: 이벤트 DataFrame을 SPADL 변환 직전 형태로 최종 정리한다.
# 동작: _normalize_wyscout_columns 수행 후 positions/tags를 _safe_literal로 객체화한다.
# 입출력/사용: convert_league에서 game 단위 변환 전에 공통 전처리 단계로 사용한다.
def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    events = _normalize_wyscout_columns(events)
    # socceraction expects Python objects in these columns, not stringified JSON.
    if "positions" in events.columns:
        events["positions"] = events["positions"].map(_safe_literal)
    if "tags" in events.columns:
        events["tags"] = events["tags"].map(_safe_literal)
    return events


# 기능: 경기별 홈팀 ID 매핑 Series를 생성한다.
# 동작: matches 파일에서 team1.teamId/home_team_id/teamsData를 우선순위로 읽어 match_id 인덱스로 반환한다.
# 입출력/사용: convert_league가 home_team_id를 받아 convert_to_actions/play_left_to_right에 전달한다.
def _load_home_teams(matches_path: Path) -> pd.Series:
    matches = _load_table(matches_path)
    match_id_col = "wyId" if "wyId" in matches.columns else "match_id"
    if match_id_col not in matches.columns:
        raise ValueError(f"{matches_path} is missing match id column ('wyId' or 'match_id')")

    home_col_candidates = ["team1.teamId", "home_team_id"]
    home_col = next((c for c in home_col_candidates if c in matches.columns), None)
    if home_col is not None:
        series = matches.set_index(match_id_col)[home_col]
        return pd.to_numeric(series, errors="coerce").dropna().astype(int)

    if "teamsData" in matches.columns:
        def _extract_home_team_id(value):
            parsed = _safe_literal(value)
            if not isinstance(parsed, dict):
                return None
            for team_key, team_data in parsed.items():
                if isinstance(team_data, dict) and str(team_data.get("side", "")).lower() == "home":
                    if "teamId" in team_data:
                        return team_data["teamId"]
                    return team_key
            return None

        home = matches["teamsData"].map(_extract_home_team_id)
        series = pd.to_numeric(home, errors="coerce")
        out = pd.Series(series.values, index=matches[match_id_col]).dropna().astype(int)
        if not out.empty:
            return out

    raise ValueError(
        f"{matches_path} is missing home team information. "
        "Expected one of columns ['team1.teamId', 'home_team_id', 'teamsData']."
    )


# 기능: data_dir의 events_* 파일을 기준으로 처리 대상 리그 목록을 만든다.
# 동작: 파일명에서 리그명을 추출하고 include 필터가 있으면 교집합만 반환한다.
# 입출력/사용: main에서 리그 루프를 구성할 때 사용된다.
def _iter_leagues(data_dir: Path, include: Iterable[str] | None) -> list[str]:
    all_leagues = sorted(
        {
            p.stem.replace("events_", "")
            for p in data_dir.iterdir()
            if p.is_file() and p.stem.startswith("events_") and p.suffix.lower() in {".csv", ".json", ".jsonl", ".ndjson"}
        }
    )
    if include:
        include_set = set(include)
        return [league for league in all_leagues if league in include_set]
    return all_leagues


# 기능: 단일 리그(events/matches)를 SPADL 액션으로 변환하여 저장한다.
# 동작: 파일 로드→전처리→game별 convert_to_actions→(옵션)left_to_right 정규화→parquet/csv 저장을 수행한다.
# 입출력/사용: main 루프에서 호출되며 반환 DataFrame은 save_combined 옵션 시 결합에 사용된다.
def convert_league(
    data_dir: Path,
    output_dir: Path,
    league: str,
    save_csv: bool,
    enforce_ltr: bool,
    goalkeeper_ids: set[int] | None = None,
) -> pd.DataFrame:
    events_path = _find_input_file(data_dir, f"events_{league}")
    matches_path = _find_input_file(data_dir, f"matches_{league}")

    home_teams = _load_home_teams(matches_path)
    events = _load_table(events_path)
    events = _prepare_events(events)

    try:
        import socceraction.spadl as spadl
        import socceraction.spadl.wyscout as wyscout_spadl
    except ImportError as exc:
        raise ImportError(
            "Failed to import socceraction dependencies. "
            "Try installing compatible versions, e.g. `pip install socceraction 'multimethod<2'`. "
            f"Original error: {exc}"
        ) from exc

    game_id_col = "game_id"
    if game_id_col not in events.columns:
        raise ValueError(f"{events_path} is missing '{game_id_col}' column")

    spadl_frames = []

    def _align_team_direction_per_game(actions_df: pd.DataFrame) -> pd.DataFrame:
        """Align each team's own-goal side to x=0 using keeper restart/action seeds.

        This fixes occasional Wyscout coordinate inconsistencies where an entire team
        in a game appears mirrored after conversion.
        """
        if actions_df.empty:
            return actions_df

        named = spadl.add_names(actions_df.copy())
        out = named.copy()
        keeper_types = {"keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up"}

        for team_id, idx in out.groupby("team_id").groups.items():
            team_rows = out.loc[idx]

            goalkick_seed = team_rows[team_rows["type_name"] == "goalkick"]
            keeper_seed = team_rows[team_rows["type_name"].isin(list(keeper_types))]

            seed = goalkick_seed if len(goalkick_seed) >= 1 else keeper_seed
            if len(seed) < 1:
                continue

            seed_x = pd.to_numeric(seed["start_x"], errors="coerce")
            if seed_x.isna().all():
                continue

            own_left = float(seed_x.median()) <= (FIELD_LENGTH / 2.0)

            # If defensive-action seed is mostly on the right half, mirror this team.
            if float(seed_x.median()) > (FIELD_LENGTH / 2.0 + 5.0):
                out.loc[idx, "start_x"] = FIELD_LENGTH - pd.to_numeric(out.loc[idx, "start_x"], errors="coerce")
                out.loc[idx, "end_x"] = FIELD_LENGTH - pd.to_numeric(out.loc[idx, "end_x"], errors="coerce")
                out.loc[idx, "start_y"] = FIELD_WIDTH - pd.to_numeric(out.loc[idx, "start_y"], errors="coerce")
                out.loc[idx, "end_y"] = FIELD_WIDTH - pd.to_numeric(out.loc[idx, "end_y"], errors="coerce")
                # Re-evaluate team rows after full-team flip.
                team_rows = out.loc[idx]
                own_left = True

            # Event-level cleanup for GK players: mirror only opposite-half GK actions.
            team_rows = out.loc[idx]
            gk_type_mask = team_rows["type_name"].isin(list(keeper_types))
            gk_player_mask = pd.Series(False, index=team_rows.index)
            if goalkeeper_ids:
                gk_player_mask = team_rows["player_id"].astype("Int64").isin(list(goalkeeper_ids))

            if own_left:
                wrong_side = (gk_type_mask | gk_player_mask) & (pd.to_numeric(team_rows["start_x"], errors="coerce") > (FIELD_LENGTH / 2.0))
            else:
                wrong_side = (gk_type_mask | gk_player_mask) & (pd.to_numeric(team_rows["start_x"], errors="coerce") < (FIELD_LENGTH / 2.0))

            wrong_idx = wrong_side[wrong_side].index
            if len(wrong_idx) > 0:
                out.loc[wrong_idx, "start_x"] = FIELD_LENGTH - pd.to_numeric(out.loc[wrong_idx, "start_x"], errors="coerce")
                out.loc[wrong_idx, "end_x"] = FIELD_LENGTH - pd.to_numeric(out.loc[wrong_idx, "end_x"], errors="coerce")
                out.loc[wrong_idx, "start_y"] = FIELD_WIDTH - pd.to_numeric(out.loc[wrong_idx, "start_y"], errors="coerce")
                out.loc[wrong_idx, "end_y"] = FIELD_WIDTH - pd.to_numeric(out.loc[wrong_idx, "end_y"], errors="coerce")

        return out[actions_df.columns]

    for game_id, game_events in events.groupby(game_id_col, sort=False):
        if game_id not in home_teams.index:
            continue

        home_team_id = int(home_teams.loc[game_id])
        game_actions = wyscout_spadl.convert_to_actions(game_events, home_team_id)

        if enforce_ltr:
            game_actions = spadl.play_left_to_right(game_actions, home_team_id)

        game_actions = _align_team_direction_per_game(game_actions)

        game_actions["game_id"] = game_id
        spadl_frames.append(game_actions)

    if not spadl_frames:
        raise RuntimeError(f"No games converted for {league}")

    league_actions = pd.concat(spadl_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"spadl_{league}.parquet"
    league_actions.to_parquet(parquet_path, index=False)

    if save_csv:
        csv_path = output_dir / f"spadl_{league}.csv"
        league_actions.to_csv(csv_path, index=False)

    return league_actions


# 기능: CLI 인자를 파싱해 Phase1 전체 변환 파이프라인을 오케스트레이션한다.
# 동작: 선수/리그 검증 후 convert_league를 반복 호출하고 필요 시 통합 파일(spadl_all)도 저장한다.
# 입출력/사용: 스크립트 진입점이며 __main__에서 직접 호출된다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Wyscout events files (CSV/JSON/JSONL) to SPADL format in batch"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR / "archive",
        help="Directory with events_* and matches_* files (.csv/.json/.jsonl/.ndjson)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "spadl",
        help="Directory to write SPADL output",
    )
    parser.add_argument(
        "--leagues",
        nargs="*",
        default=None,
        help="Optional list of leagues to convert (e.g. England Spain Italy)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save CSV files (parquet is always saved)",
    )
    parser.add_argument(
        "--no-left-to-right",
        action="store_true",
        help="Disable left-to-right normalization (enabled by default)",
    )
    parser.add_argument(
        "--save-combined",
        action="store_true",
        help="Save one combined SPADL file across converted leagues",
    )

    args = parser.parse_args()

    players = _load_players(args.data_dir)
    goalkeeper_ids = _extract_goalkeeper_ids(players)
    print(f"[INFO] Loaded players: {len(players):,}")

    leagues = _iter_leagues(args.data_dir, args.leagues)
    if not leagues:
        raise RuntimeError("No events_* files found with supported extensions (or no matching --leagues)")

    combined = []
    for league in leagues:
        print(f"[INFO] Converting {league} ...")
        league_df = convert_league(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            league=league,
            save_csv=args.save_csv,
            enforce_ltr=not args.no_left_to_right,
            goalkeeper_ids=goalkeeper_ids,
        )
        combined.append(league_df)
        print(f"[OK] {league}: {len(league_df):,} actions")

    if args.save_combined:
        all_actions = pd.concat(combined, ignore_index=True)
        combined_parquet = args.output_dir / "spadl_all.parquet"
        all_actions.to_parquet(combined_parquet, index=False)
        if args.save_csv:
            all_actions.to_csv(args.output_dir / "spadl_all.csv", index=False)
        print(f"[OK] Combined: {len(all_actions):,} actions")


if __name__ == "__main__":
    main()
