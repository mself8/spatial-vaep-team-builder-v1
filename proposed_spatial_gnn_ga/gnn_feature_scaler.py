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
# 기능: 빈 텐서 예외를 포함해 mean/std를 계산하고 std<EPS 구간을 1로 치환해 z-score 분모 0 문제를 방지한다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: x: torch.Tensor, dim: int
#   - Output: tuple[torch.Tensor, torch.Tensor]
def _safe_mean_std(x: torch.Tensor, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if x.numel() == 0:
        return torch.zeros(0, dtype=torch.float32), torch.ones(0, dtype=torch.float32)
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=False)
    std = torch.where(std > EPS, std, torch.ones_like(std))
    return mean.to(torch.float32), std.to(torch.float32)
# 기능: _collect_node_features는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData]
#   - Output: torch.Tensor
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
# 기능: _collect_edge_features는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData], edge_type: tuple[str, str, str], feat_dim: int
#   - Output: torch.Tensor
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
# 기능: 전체 그래프에서 node_x(24D), passes_to(12D), defends_against(12D) 분포를 모아 전역 mean/std 통계를 산출한다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData]
#   - Output: Dict[str, torch.Tensor]
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
# 기능: fit_feature_scaler는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData]
#   - Output: Dict[str, torch.Tensor]
def fit_feature_scaler(graphs: List[HeteroData]) -> Dict[str, torch.Tensor]:
    """Fit z-score statistics from training graphs only."""
    return compute_global_zscore_stats(graphs)
# 기능: sanitize_feature_scaler_stats는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: stats: Dict[str, torch.Tensor]
#   - Output: Dict[str, torch.Tensor]
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
# 기능: 각 노드/엣지 타입별로 (x-mean)/std를 적용해 그래프 객체를 제자리(in-place)로 표준화한다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: graph: HeteroData, stats: Dict[str, torch.Tensor]
#   - Output: HeteroData
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
# 기능: apply_global_zscore_to_graphs_inplace는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: Iterable[HeteroData], stats: Dict[str, torch.Tensor]
#   - Output: None
def apply_global_zscore_to_graphs_inplace(graphs: Iterable[HeteroData], stats: Dict[str, torch.Tensor]) -> None:
    """Transform train/valid/inference graph collections with one fitted scaler."""
    stats = sanitize_feature_scaler_stats(stats)
    for g in graphs:
        apply_global_zscore_inplace(g, stats)
# 기능: scaler_stats_to_payload는 컬럼 'node_mean', 'node_std', 'passes_mean', 'passes_std', 'defends_mean'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: 학습/추론 그래프의 node/edge 분포를 동일 z-score 기준으로 맞춰 fold 간 스케일 드리프트를 줄이기 위해 필요하다.
# 데이터 입출력:
#   - Input: stats: Dict[str, torch.Tensor]
#   - Output: Dict[str, object]
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
