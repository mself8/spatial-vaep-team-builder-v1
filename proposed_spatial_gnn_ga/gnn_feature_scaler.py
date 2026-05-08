#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch_geometric.data import HeteroData

EPS = 1e-6
NODE_TYPES = ["home_team", "away_team"]
PASS_EDGE_TYPES = [
    ("home_team", "passes_to", "home_team"),
    ("away_team", "passes_to", "away_team"),
]
DEF_EDGE_TYPES = [
    ("home_team", "defends_against", "away_team"),
    ("away_team", "defends_against", "home_team"),
]
# 기능: 텐서의 평균과 표준편차를 계산하되, std가 EPS(1e-6) 미만인 차원은 1로 대체한다.
# 동작/맥락: 특정 공간 존(zone)에 이벤트가 극히 드문 경우 std≈0이 되어 z-score 분모가 0이 될 수 있다.
#            이 경우 (x - mean) / 1 = x - mean 으로 처리해 수치 폭발 없이 안정적으로 정규화한다.
# 데이터 입출력:
#   - Input: x: torch.Tensor [N, D] — 특징 행렬 (N=전체 노드/엣지 수, D=특징 차원)
#            dim: int — 통계를 집계할 축 (dim=0이면 각 특징 차원별 통계)
#   - Output: (mean: Tensor[D], std: Tensor[D]) — std에서 0에 가까운 값은 1로 치환됨
def _safe_mean_std(x: torch.Tensor, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if x.numel() == 0:
        return torch.zeros(0, dtype=torch.float32), torch.ones(0, dtype=torch.float32)
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=False)
    std = torch.where(std > EPS, std, torch.ones_like(std))
    return mean.to(torch.float32), std.to(torch.float32)
# 기능: 전체 그래프 리스트에서 home_team과 away_team의 노드 특징 행렬(24D)을 모두 concat한다.
# 동작/맥락: z-score 통계 계산을 위해 학습 그래프의 모든 노드를 단일 행렬로 수집한다.
#            home/away 구분 없이 동일 24D 피처 공간이므로 합쳐서 통계를 낸다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData]
#   - Output: torch.Tensor [N_total, 24] — 모든 그래프·모든 노드의 특징 행렬
def _collect_node_features(graphs: List[HeteroData]) -> torch.Tensor:
    rows = []
    for g in graphs:
        for ntype in NODE_TYPES:
            if ntype not in g.node_types:
                continue
            store = g[ntype]
            if hasattr(store, "x"):
                x = store.x
                if isinstance(x, torch.Tensor) and x.numel() > 0:
                    rows.append(x.to(torch.float32))
    if not rows:
        return torch.empty((0, 24), dtype=torch.float32)
    return torch.cat(rows, dim=0)
# 기능: 특정 관계 타입(edge_type)의 edge_attr를 모든 그래프에서 수집하여 concat한다.
# 동작/맥락: IO 엣지(12D)와 ID 엣지(12D)는 서로 다른 의미의 공간 시너지 벡터이므로 타입별로 분리해 통계를 낸다.
#            (IO와 ID의 분포가 달라 함께 정규화하면 scale mismatch가 발생)
# 데이터 입출력:
#   - Input: graphs: List[HeteroData]
#            edge_type: tuple[str,str,str] — 예) ("home_team","passes_to","home_team")
#            feat_dim: int — 엣지 특징 차원 (IO=12, ID=12)
#   - Output: torch.Tensor [E_total, feat_dim] — 해당 관계 타입의 모든 엣지 특징
def _collect_edge_features(graphs: List[HeteroData], edge_type: tuple[str, str, str], feat_dim: int) -> torch.Tensor:
    rows = []
    for g in graphs:
        if edge_type not in g.edge_types:
            continue
        store = g[edge_type]
        if hasattr(store, "edge_attr"):
            e = store.edge_attr
            if isinstance(e, torch.Tensor) and e.numel() > 0:
                rows.append(e.to(torch.float32))
    if not rows:
        return torch.empty((0, feat_dim), dtype=torch.float32)
    return torch.cat(rows, dim=0)
# 기능: 그래프 리스트에서 node_x(24D), IO 엣지(12D), ID 엣지(12D) 각각의 전역 mean/std를 계산한다.
# 동작/맥락: 3가지 피처 그룹이 각각 독립적인 z-score 통계를 가진다:
#   - node_mean/std [24D]: off_12(공격 공간 VAEP) + def_12(수비 공간 VAEP) 복합 분포
#   - passes_mean/std [12D]: 같은 팀 내 패스 협력 강도 (IO 시너지 벡터 분포)
#   - defends_mean/std [12D]: 교차 팀 수비 대결 강도 (ID 시너지 벡터 분포)
#   IO/ID 엣지는 home/away 방향 모두 합쳐서 통계를 내어 방향 편향을 줄인다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData]
#   - Output: Dict[str, torch.Tensor] — {'node_mean':24D, 'node_std':24D, 'passes_mean':12D, ...}
def compute_global_zscore_stats(graphs: List[HeteroData]) -> Dict[str, torch.Tensor]:
    """Compute dataset-global per-dimension mean/std for node and edge features.

    - node_x: 24D (off_12 + def_12)
    - passes_to edge_attr: 12D
    - defends_against edge_attr: 12D
    """
    node_x = _collect_node_features(graphs)
    node_mean, node_std = _safe_mean_std(node_x, dim=0)

    pass_edges = _collect_edge_features(graphs, PASS_EDGE_TYPES[0], 12)
    pass_edges_away = _collect_edge_features(graphs, PASS_EDGE_TYPES[1], 12)
    if pass_edges_away.numel() > 0:
        pass_edges = torch.cat([pass_edges, pass_edges_away], dim=0) if pass_edges.numel() > 0 else pass_edges_away
    pass_mean, pass_std = _safe_mean_std(pass_edges, dim=0)

    def_edges = _collect_edge_features(graphs, DEF_EDGE_TYPES[0], 12)
    def_edges_rev = _collect_edge_features(graphs, DEF_EDGE_TYPES[1], 12)
    if def_edges_rev.numel() > 0:
        def_edges = torch.cat([def_edges, def_edges_rev], dim=0) if def_edges.numel() > 0 else def_edges_rev
    def_mean, def_std = _safe_mean_std(def_edges, dim=0)

    stats = {
        "node_mean": node_mean,
        "node_std": node_std,
        "passes_mean": pass_mean,
        "passes_std": pass_std,
        "defends_mean": def_mean,
        "defends_std": def_std,
    }
    return sanitize_feature_scaler_stats(stats)
# 기능: train fold 그래프만으로 z-score 통계를 fit한다. train_gnn_phase5.py에서 호출되는 진입점이다.
# 동작/맥락: 반드시 train fold 그래프만 전달해야 한다. valid/test 그래프를 포함하면 데이터 누수가 발생한다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData] — train fold 그래프만 포함
#   - Output: Dict[str, torch.Tensor] — z-score 통계 딕셔너리 (transform에 재사용)
def fit_feature_scaler(graphs: List[HeteroData]) -> Dict[str, torch.Tensor]:
    """Fit z-score statistics from training graphs only."""
    return compute_global_zscore_stats(graphs)
# 기능: scaler 통계 딕셔너리를 float32/CPU로 표준화하고 std의 0값을 1로 치환하여 안전한 상태로 정규화한다.
# 동작/맥락: 체크포인트 로드 시 dtype이 달라질 수 있고, 저장/복원 과정에서 std가 작아질 수 있으므로 방어적 정제가 필요하다.
# 데이터 입출력:
#   - Input: stats: Dict[str, torch.Tensor] — 6개 키 필수: node_mean, node_std, passes_mean, passes_std, defends_mean, defends_std
#   - Output: Dict[str, torch.Tensor] — 동일 구조, float32/CPU, std≤EPS인 원소는 1로 치환
def sanitize_feature_scaler_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize dtype/device and protect std terms from zero division."""
    out: Dict[str, torch.Tensor] = {}
    for k in ["node_mean", "node_std", "passes_mean", "passes_std", "defends_mean", "defends_std"]:
        if k not in stats:
            raise KeyError(f"Missing scaler key: {k}")
        out[k] = torch.as_tensor(stats[k], dtype=torch.float32).view(-1).cpu()

    for std_k in ["node_std", "passes_std", "defends_std"]:
        s = out[std_k]
        out[std_k] = torch.where(s.abs() > EPS, s, torch.ones_like(s))
    return out
# 기능: 단일 HeteroData 그래프의 node_x와 edge_attr를 train fold 통계 기준으로 z-score 변환한다.
# 동작/맥락: 변환 수식: x_scaled = (x - mean) / std, 텐서를 제자리에서 교체하므로 메모리 효율적이다.
#   - home_team.x, away_team.x: node_mean/std (24D) 적용
#   - (home_team, passes_to, home_team).edge_attr: passes_mean/std (12D) 적용
#   - (away_team, passes_to, away_team).edge_attr: passes_mean/std (12D) 적용
#   - defends_against 두 방향 edge_attr: defends_mean/std (12D) 적용
# 데이터 입출력:
#   - Input: graph: HeteroData, stats: Dict[str, torch.Tensor]
#   - Output: HeteroData (동일 객체, in-place 수정 후 반환)
def apply_global_zscore_inplace(graph: HeteroData, stats: Dict[str, torch.Tensor]) -> HeteroData:
    stats = sanitize_feature_scaler_stats(stats)

    for ntype in NODE_TYPES:
        if ntype in graph.node_types and hasattr(graph[ntype], "x"):
            x = graph[ntype].x
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                graph[ntype].x = (x.to(torch.float32) - stats["node_mean"]) / stats["node_std"]

    for etype in PASS_EDGE_TYPES:
        if etype in graph.edge_types and hasattr(graph[etype], "edge_attr"):
            e = graph[etype].edge_attr
            if isinstance(e, torch.Tensor) and e.numel() > 0:
                graph[etype].edge_attr = (e.to(torch.float32) - stats["passes_mean"]) / stats["passes_std"]

    for etype in DEF_EDGE_TYPES:
        if etype in graph.edge_types and hasattr(graph[etype], "edge_attr"):
            e = graph[etype].edge_attr
            if isinstance(e, torch.Tensor) and e.numel() > 0:
                graph[etype].edge_attr = (e.to(torch.float32) - stats["defends_mean"]) / stats["defends_std"]

    return graph
# 기능: 그래프 컬렉션(train 또는 valid) 전체에 z-score 변환을 일괄 적용한다.
# 동작/맥락: train_gnn_phase5.py의 transform_graphs_with_scaler_inplace()에서 호출되는 진입점이다.
#            통계는 반드시 train fold에서만 fit된 것이어야 한다.
# 데이터 입출력:
#   - Input: graphs: Iterable[HeteroData] — train 또는 valid 그래프 컬렉션
#            stats: Dict[str, torch.Tensor] — fit_feature_scaler()의 결과
#   - Output: None (각 그래프 객체를 in-place 수정)
def apply_global_zscore_to_graphs_inplace(graphs: Iterable[HeteroData], stats: Dict[str, torch.Tensor]) -> None:
    """Transform train/valid/inference graph collections with one fitted scaler."""
    stats = sanitize_feature_scaler_stats(stats)
    for g in graphs:
        apply_global_zscore_inplace(g, stats)
# 기능: 내부 scaler 통계를 중첩 딕셔너리 직렬화 포맷으로 변환한다.
# 동작/맥락: torch.save 대상 모델 체크포인트에 함께 포함될 수 있도록 구조화된 포맷으로 변환한다.
#            포맷: {'node': {'mean':T, 'std':T}, 'passes_to': {...}, 'defends_against': {...}, 'format': 'global_zscore_v1'}
#            optimize_lineup_ga_phase6.py의 _parse_feature_scaler_payload()가 이 포맷을 복원한다.
# 데이터 입출력:
#   - Input: stats: Dict[str, torch.Tensor]
#   - Output: Dict[str, object] — JSON-직렬화 가능한 중첩 딕셔너리
def scaler_stats_to_payload(stats: Dict[str, torch.Tensor]) -> Dict[str, object]:
    stats = sanitize_feature_scaler_stats(stats)
    return {
        "node": {"mean": stats["node_mean"], "std": stats["node_std"]},
        "passes_to": {"mean": stats["passes_mean"], "std": stats["passes_std"]},
        "defends_against": {"mean": stats["defends_mean"], "std": stats["defends_std"]},
        "format": "global_zscore_v1",
    }
# 기능: payload_to_scaler_stats는 컬럼 'mean', 'std'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: payload: Dict[str, object]
#   - Output: Dict[str, torch.Tensor]
def payload_to_scaler_stats(payload: Dict[str, object]) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("scaler payload must be a dict")
    # 기능: _block는 컬럼 'mean', 'std'을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: key: str
    #   - Output: Dict[str, torch.Tensor]
    def _block(key: str) -> Dict[str, torch.Tensor]:
        blk = payload.get(key)
        if not isinstance(blk, dict):
            raise KeyError(f"Missing scaler block: {key}")
        if "mean" not in blk or "std" not in blk:
            raise KeyError(f"Scaler block missing mean/std: {key}")
        return {
            "mean": torch.as_tensor(blk["mean"], dtype=torch.float32),
            "std": torch.as_tensor(blk["std"], dtype=torch.float32),
        }

    node = _block("node")
    passes = _block("passes_to")
    defends = _block("defends_against")
    return sanitize_feature_scaler_stats(
        {
            "node_mean": node["mean"],
            "node_std": node["std"],
            "passes_mean": passes["mean"],
            "passes_std": passes["std"],
            "defends_mean": defends["mean"],
            "defends_std": defends["std"],
        }
    )
# 기능: save_feature_scaler는 연산 torch.save을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: stats: Dict[str, torch.Tensor], output_path: Path
#   - Output: None
def save_feature_scaler(stats: Dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = scaler_stats_to_payload(stats)
    torch.save(payload, output_path)
