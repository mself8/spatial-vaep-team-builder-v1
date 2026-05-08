#!/usr/bin/env python3
"""Paper Table 2 ablation: "Ours (Without GNN)".

Team-Builder ILP 식 목적함수를 GA로 최적화한 버전. GNN 미사용으로,
GNN 도입의 정량적 정당성(스칼라 합산 vs 공간적 상호작용)을 보여주기 위한
비교 실험 스크립트. main_pipeline.ipynb에서 호출된다.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"

# 어떤 약어가 와도 GK, DF, MF, FW 4가지로 욱여넣는 파서
def _parse_role_code(role_value: object) -> str:
    val = str(role_value).upper()
    if any(k in val for k in ["GK", "GOAL", "KEEP", "PORT"]): return "GK"
    if any(k in val for k in ["DF", "DEF", "BACK", "CB", "LB", "RB", "WB", "DIF"]): return "DF"
    if any(k in val for k in ["FW", "FOR", "ATT", "ST", "WING", "WIN", "LW", "RW"]): return "FW"
    return "MF"  # 애매하면 모두 MF로 처리하여 증발 방지

# =====================================================================
# [GA Core Functions]
# =====================================================================
def init_population(pop_size, pool, needs, fixed_gk_id):
    pop = []
    for _ in range(pop_size):
        genome = []
        if fixed_gk_id: genome.append(fixed_gk_id)
        for pos, count in needs.items():
            if pos == "GK" and fixed_gk_id: continue
            genome.extend(random.sample(pool[pos], count))
        pop.append(genome)
    return pop

def evaluate_fitness(genome, vi_map, io_pair_map, id_map, l_vi, l_io, l_id):
    score = sum(vi_map.get(p, 0.0) * l_vi + id_map.get(p, 0.0) * l_id for p in genome)
    for i in range(len(genome)):
        for j in range(i + 1, len(genome)):
            key = tuple(sorted((genome[i], genome[j])))
            score += io_pair_map.get(key, 0.0) * l_io
    return score

def crossover(p1, p2, pool, needs, pos_map, fixed_gk_id):
    child = [fixed_gk_id] if fixed_gk_id else []
    for pos, count in needs.items():
        if pos == "GK" and fixed_gk_id: continue
        p1_sub = [p for p in p1 if pos_map.get(p) == pos]
        p2_sub = [p for p in p2 if pos_map.get(p) == pos]
        combined = list(set(p1_sub + p2_sub))
        if len(combined) >= count: 
            child.extend(random.sample(combined, count))
        else: 
            child.extend(combined)
            avail = list(set(pool[pos]) - set(combined))
            child.extend(random.sample(avail, count - len(combined)))
    return child

def mutate(genome, pool, needs, pos_map, mut_rate, fixed_gk_id):
    for i in range(len(genome)):
        pid = genome[i]
        if pid == fixed_gk_id: continue
        if random.random() < mut_rate:
            pos = pos_map.get(pid)
            avail = list(set(pool[pos]) - set(genome)) # 중복 선택 방지
            if avail: genome[i] = random.choice(avail)
    return genome

# =====================================================================
# [Main Execution]
# =====================================================================
def run_phase5_ga(synergy_dir, archive_dir, output_dir, team_id, formation, available_player_ids, fixed_gk_id, **kwargs):
    player_scores = pd.read_parquet(synergy_dir / "player_synergy_scores.parquet")
    players_df = pd.read_csv(archive_dir / "players.csv")
    
    players_meta = players_df[["wyId", "shortName", "role"]].rename(columns={"wyId": "player_id", "shortName": "player_name"})
    players_meta["player_id"] = pd.to_numeric(players_meta["player_id"], errors='coerce').fillna(-1).astype(int)
    players_meta["position"] = players_meta["role"].map(_parse_role_code)
    
    if available_player_ids:
        team_players = players_meta[players_meta["player_id"].isin(available_player_ids)].copy()
    else:
        team_players = players_meta.copy()
        
    team_players = team_players[team_players["position"].isin(["GK", "DF", "MF", "FW"])]

    scores = player_scores[pd.to_numeric(player_scores["team_id"], errors='coerce') == team_id].copy()
    scores["player_id"] = pd.to_numeric(scores["player_id"], errors='coerce').fillna(-1).astype(int)
    
    team_players = team_players.merge(scores[["player_id", "vi"]], on="player_id", how="left")
    team_players["vi"] = team_players["vi"].fillna(0.0) # 스탯 없어도 명단 생존
    team_players = team_players.drop_duplicates(subset=["player_id"])

    pos_map = team_players.set_index("player_id")["position"].to_dict()
    vi_map = team_players.set_index("player_id")["vi"].to_dict()
    
    pool_by_pos = {"GK":[], "DF":[], "MF":[], "FW":[]}
    for pid, pos in pos_map.items(): pool_by_pos[pos].append(pid)
    
    # GK 강제 세팅
    if fixed_gk_id and fixed_gk_id not in pool_by_pos["GK"]:
        pool_by_pos["GK"].append(fixed_gk_id)
        pos_map[fixed_gk_id] = "GK"
        vi_map[fixed_gk_id] = 0.0

    f_parts = [int(x) for x in formation.split("-")]
    needs = {"GK": 1, "DF": f_parts[0], "MF": f_parts[1], "FW": f_parts[2]}

    # 🔥 [핵심 수정] 포지션 융통성(Flex) 부여 및 중복(양다리) 완벽 차단
    for pos, count in needs.items():
        if pos == "GK" and fixed_gk_id: continue
        while len(pool_by_pos[pos]) < count:
            spares = []
            # 남는 포지션에서 선수를 수배
            for other_pos in ["DF", "MF", "FW"]:
                if other_pos != pos and len(pool_by_pos[other_pos]) > needs.get(other_pos, 0):
                    spares.extend(pool_by_pos[other_pos])
            
            if not spares:
                spares = list(set(pool_by_pos["DF"] + pool_by_pos["MF"] + pool_by_pos["FW"]) - set(pool_by_pos[pos]))
                if not spares: break
            
            borrowed_id = random.choice(spares)
            
            # 빌려온 선수는 원래 포지션에서 완벽히 삭제 (중복 방지)
            for other_pos in ["DF", "MF", "FW"]:
                if borrowed_id in pool_by_pos[other_pos]:
                    pool_by_pos[other_pos].remove(borrowed_id)
            
            pool_by_pos[pos].append(borrowed_id)
            pos_map[borrowed_id] = pos # 포지션 라벨 강제 변경

    # GA 실행
    pop = init_population(50, pool_by_pos, needs, fixed_gk_id)
    best_genome = None
    best_score = -float('inf')

    for gen in range(1, 101):
        scored = [(evaluate_fitness(g, vi_map, {}, {}, 1.0, 1.0, 1.0), g) for g in pop]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_genome = scored[0][1][:]
            
        elites = [g for _, g in scored[:10]]
        next_pop = elites[:]
        while len(next_pop) < 50:
            child = crossover(random.choice(elites), random.choice(elites), pool_by_pos, needs, pos_map, fixed_gk_id)
            next_pop.append(mutate(child, pool_by_pos, needs, pos_map, 0.2, fixed_gk_id))
        pop = next_pop

    # 🔥 최종 안전장치: 혹시라도 돌연변이 때문에 중복이 생겼을 경우, 강제로 11명을 채움
    final_genome = list(set(best_genome))
    if len(final_genome) < 11:
        all_field_players = pool_by_pos["DF"] + pool_by_pos["MF"] + pool_by_pos["FW"]
        avail_to_fill = list(set(all_field_players) - set(final_genome))
        missing_count = 11 - len(final_genome)
        if len(avail_to_fill) >= missing_count:
            final_genome.extend(random.sample(avail_to_fill, missing_count))

    best_lineup = team_players[team_players["player_id"].isin(final_genome)].copy()
    
    if len(best_lineup) == 11:
        output_dir.mkdir(parents=True, exist_ok=True)
        best_lineup.to_csv(output_dir / "lineup_selected_ga.csv", index=False)
    else:
        print(f"❌ [ERROR] Saved failed. Expected 11, got {len(best_lineup)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-id", type=int)
    parser.add_argument("--available-player-ids", nargs="*", type=int)
    parser.add_argument("--fixed-gk-id", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formation", type=str, default="4-3-3")
    args = parser.parse_args()
    
    run_phase5_ga(**vars(args), synergy_dir=DATA_DIR/"synergy", archive_dir=DATA_DIR/"archive")

if __name__ == "__main__": main()