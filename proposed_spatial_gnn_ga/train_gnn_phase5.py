#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 5: Train hetero GNN for 3-class match outcome prediction.

Model requirements implemented:
1) Relation-wise edge-aware GATConv over:
   - (home_team, passes_to, home_team)
   - (away_team, passes_to, away_team)
   - (home_team, defends_against, away_team)
   - (away_team, defends_against, home_team)
2) HAN-style semantic-level attention to fuse relation embeddings per destination node type.
3) Train-fold-only z-score fit/transform (no full-dataset scaler leakage).
"""

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"


REL_HOME_IO = ("home_team", "passes_to", "home_team")
REL_AWAY_IO = ("away_team", "passes_to", "away_team")
REL_HOME_ID = ("home_team", "defends_against", "away_team")
REL_AWAY_ID = ("away_team", "defends_against", "home_team")
RELATIONS = [REL_HOME_IO, REL_AWAY_IO, REL_HOME_ID, REL_AWAY_ID]


# ----------------------------
# Scaler module bridge
# ----------------------------
# 기능: _import_scaler_module는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: 코드 내부 return 표현식
def _import_scaler_module():
    try:
        import gnn_feature_scaler as scaler_mod  # type: ignore

        return scaler_mod
    except Exception:
        scaler_path = Path(__file__).resolve().parent / "gnn_feature_scaler.py"
        spec = importlib.util.spec_from_file_location("gnn_feature_scaler", scaler_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to import scaler module from: {scaler_path}")
        scaler_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scaler_mod)
        return scaler_mod


_SCALER_MOD = _import_scaler_module()
# 기능: fit_train_fold_scaler는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: train_graphs: List
#   - Output: Dict[str, torch.Tensor]
def fit_train_fold_scaler(train_graphs: List) -> Dict[str, torch.Tensor]:
    """Fit scaler stats from train fold graphs only."""
    return _SCALER_MOD.fit_feature_scaler(train_graphs)
# 기능: transform_graphs_with_scaler_inplace는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: List, scaler_stats: Dict[str, torch.Tensor]
#   - Output: None
def transform_graphs_with_scaler_inplace(graphs: List, scaler_stats: Dict[str, torch.Tensor]) -> None:
    """Apply previously fitted scaler to any graph split (train/valid/inference)."""
    _SCALER_MOD.apply_global_zscore_to_graphs_inplace(graphs, scaler_stats)
# 기능: export_scaler_payload는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: scaler_stats: Dict[str, torch.Tensor]
#   - Output: Dict[str, object]
def export_scaler_payload(scaler_stats: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return _SCALER_MOD.scaler_stats_to_payload(scaler_stats)
# 기능: save_scaler_payload는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: scaler_stats: Dict[str, torch.Tensor], output_path: Path
#   - Output: None
def save_scaler_payload(scaler_stats: Dict[str, torch.Tensor], output_path: Path) -> None:
    _SCALER_MOD.save_feature_scaler(scaler_stats, output_path)


# ----------------------------
# Core utilities
# ----------------------------
# 기능: _rel_key는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: rel: Tuple[str, str, str]
#   - Output: str
def _rel_key(rel: Tuple[str, str, str]) -> str:
    return "__".join(rel)
# 기능: set_seed는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: seed: int
#   - Output: None
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SemanticLevelAttention(nn.Module):
    """HAN-style semantic attention over relation-specific node embeddings."""
    # 기능: __init__는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: hidden_channels: int, num_relations: int, dropout: float
    #   - Output: None
    def __init__(self, hidden_channels: int, num_relations: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_relations = int(num_relations)
        self.dropout = float(dropout)

        self.proj = nn.Linear(self.hidden_channels, self.hidden_channels, bias=True)
        self.query = nn.Parameter(torch.empty(self.hidden_channels))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.uniform_(self.query, -0.1, 0.1)
    # 기능: _graph_mean_scalar는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: node_values: torch.Tensor, batch: torch.Tensor, n_graphs: int
    #   - Output: torch.Tensor
    @staticmethod
    def _graph_mean_scalar(node_values: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        summed = node_values.new_zeros((n_graphs,))
        count = node_values.new_zeros((n_graphs,))
        ones = node_values.new_ones((node_values.size(0),))
        summed.index_add_(0, batch, node_values)
        count.index_add_(0, batch, ones)
        return summed / count.clamp_min(1.0)
    # 기능: forward는 연산 softmax을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: relation_embeddings: List[torch.Tensor], batch: torch.Tensor, return_weights: bool
    #   - Output: 코드 내부 return 표현식
    def forward(
        self,
        relation_embeddings: List[torch.Tensor],
        batch: torch.Tensor,
        return_weights: bool = False,
    ):
        if len(relation_embeddings) != self.num_relations:
            raise ValueError(
                f"Expected {self.num_relations} relation embeddings, got {len(relation_embeddings)}"
            )
        if not relation_embeddings:
            raise ValueError("relation_embeddings must not be empty")

        n_nodes = int(relation_embeddings[0].size(0))
        if n_nodes == 0:
            empty = relation_embeddings[0]
            if return_weights:
                return empty, empty.new_zeros((0, self.num_relations))
            return empty

        n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        rel_scores = []
        for h in relation_embeddings:
            if h.size(0) != n_nodes:
                raise ValueError("All relation embeddings must have the same number of nodes")
            node_score = torch.matmul(torch.tanh(self.proj(h)), self.query)
            graph_score = self._graph_mean_scalar(node_score, batch, n_graphs)
            rel_scores.append(graph_score)

        scores = torch.stack(rel_scores, dim=1)  # [B, R]
        beta = torch.softmax(scores, dim=1)

        if self.training and self.dropout > 0.0:
            beta = F.dropout(beta, p=self.dropout, training=True)
            beta = beta / beta.sum(dim=1, keepdim=True).clamp_min(1e-6)

        out = relation_embeddings[0].new_zeros(relation_embeddings[0].shape)
        for r_idx, h in enumerate(relation_embeddings):
            out = out + beta[:, r_idx][batch].unsqueeze(-1) * h

        if return_weights:
            return out, beta
        return out


class HeteroEdgeGATWinPredictor(nn.Module):
    """Edge-aware hetero GAT + semantic-level attention for 3-class outcome logits."""
    # 기능: __init__는 연산 GATConv을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: hidden_channels: int, num_layers: int, heads: int, dropout: float, num_classes: int
    #   - Output: None
    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.15,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)

        self.rel_convs = nn.ModuleList()
        self.home_semantic = nn.ModuleList()
        self.away_semantic = nn.ModuleList()

        for _ in range(int(num_layers)):
            rel_layer = nn.ModuleDict(
                {
                    _rel_key(REL_HOME_IO): GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                    _rel_key(REL_AWAY_IO): GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                    _rel_key(REL_HOME_ID): GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                    _rel_key(REL_AWAY_ID): GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=self.dropout,
                        add_self_loops=False,
                        edge_dim=12,
                    ),
                }
            )
            self.rel_convs.append(rel_layer)
            self.home_semantic.append(SemanticLevelAttention(hidden_channels, num_relations=2, dropout=self.dropout))
            self.away_semantic.append(SemanticLevelAttention(hidden_channels, num_relations=2, dropout=self.dropout))

        self.head = nn.Sequential(
            nn.LazyLinear(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, self.num_classes),
        )
    # 기능: _node_batch는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: node_store
    #   - Output: torch.Tensor
    @staticmethod
    def _node_batch(node_store) -> torch.Tensor:
        if hasattr(node_store, "batch") and node_store.batch is not None:
            return node_store.batch
        return torch.zeros(node_store.x.size(0), dtype=torch.long, device=node_store.x.device)
    # 기능: _extract_global_features는 컬럼 'global_features'을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: data, batch_size: int, device: torch.device
    #   - Output: torch.Tensor
    def _extract_global_features(self, data, batch_size: int, device: torch.device) -> torch.Tensor:
        if "global_features" not in data:
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=device)

        gf = data["global_features"]
        if gf.dim() == 1:
            gf = gf.view(batch_size, -1)
        elif gf.dim() == 2:
            if gf.size(0) != batch_size:
                gf = gf.view(batch_size, -1)
        else:
            gf = gf.view(batch_size, -1)
        return gf.to(device=device, dtype=torch.float32)
    # 기능: forward는 컬럼 'home_team', 'away_team', 'updated_home_x', 'updated_away_x', 'home_pool'을 기준으로 함수 목적에 맞는 산출물을 만든다.
    # 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
    # 데이터 입출력:
    #   - Input: data, return_debug: bool
    #   - Output: 코드 내부 return 표현식
    def forward(self, data, return_debug: bool = False):
        x_home = data["home_team"].x
        x_away = data["away_team"].x

        home_batch = self._node_batch(data["home_team"])
        away_batch = self._node_batch(data["away_team"])

        edge_index_home_io = data[REL_HOME_IO].edge_index
        edge_attr_home_io = data[REL_HOME_IO].edge_attr
        edge_index_away_io = data[REL_AWAY_IO].edge_index
        edge_attr_away_io = data[REL_AWAY_IO].edge_attr
        edge_index_home_id = data[REL_HOME_ID].edge_index
        edge_attr_home_id = data[REL_HOME_ID].edge_attr
        edge_index_away_id = data[REL_AWAY_ID].edge_index
        edge_attr_away_id = data[REL_AWAY_ID].edge_attr

        debug: Dict[str, object] = {
            "input_home_x": tuple(x_home.shape),
            "input_away_x": tuple(x_away.shape),
            "input_home_io_edge_attr": tuple(edge_attr_home_io.shape),
            "input_away_io_edge_attr": tuple(edge_attr_away_io.shape),
            "input_home_id_edge_attr": tuple(edge_attr_home_id.shape),
            "input_away_id_edge_attr": tuple(edge_attr_away_id.shape),
        }

        for li, rel_layer in enumerate(self.rel_convs, start=1):
            home_io_out = rel_layer[_rel_key(REL_HOME_IO)](x_home, edge_index_home_io, edge_attr=edge_attr_home_io)
            away_io_out = rel_layer[_rel_key(REL_AWAY_IO)](x_away, edge_index_away_io, edge_attr=edge_attr_away_io)
            home_id_out = rel_layer[_rel_key(REL_HOME_ID)](
                (x_home, x_away),
                edge_index_home_id,
                edge_attr=edge_attr_home_id,
            )
            away_id_out = rel_layer[_rel_key(REL_AWAY_ID)](
                (x_away, x_home),
                edge_index_away_id,
                edge_attr=edge_attr_away_id,
            )

            x_home, home_beta = self.home_semantic[li - 1]([home_io_out, away_id_out], home_batch, return_weights=True)
            x_away, away_beta = self.away_semantic[li - 1]([away_io_out, home_id_out], away_batch, return_weights=True)

            x_home = F.relu(x_home)
            x_away = F.relu(x_away)
            x_home = F.dropout(x_home, p=self.dropout, training=self.training)
            x_away = F.dropout(x_away, p=self.dropout, training=self.training)

            debug[f"layer{li}_home_semantic_beta_mean"] = home_beta.mean(dim=0).detach().cpu().tolist()
            debug[f"layer{li}_away_semantic_beta_mean"] = away_beta.mean(dim=0).detach().cpu().tolist()

        debug["updated_home_x"] = tuple(x_home.shape)
        debug["updated_away_x"] = tuple(x_away.shape)

        home_pool = global_mean_pool(x_home, home_batch)
        away_pool = global_mean_pool(x_away, away_batch)
        global_features = self._extract_global_features(data, batch_size=home_pool.size(0), device=home_pool.device)

        debug["home_pool"] = tuple(home_pool.shape)
        debug["away_pool"] = tuple(away_pool.shape)
        debug["global_features"] = tuple(global_features.shape)

        match_repr = torch.cat([home_pool, away_pool, global_features], dim=-1)
        debug["concat_match_repr"] = tuple(match_repr.shape)
        logits = self.head(match_repr)
        debug["logits"] = tuple(logits.shape)

        if return_debug:
            return logits, debug
        return logits


# ----------------------------
# Dataset / training utilities
# ----------------------------
# 기능: _prepare_graph_targets는 컬럼 'match_y', 'target_result'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: List
#   - Output: List
def _prepare_graph_targets(graphs: List) -> List:
    """Keep labeled samples and create integer class target (loss=0, draw=1, win=2)."""
    prepared = []
    for g in graphs:
        if "match_y" not in g:
            continue
        y_raw = int(g["match_y"].view(-1)[0].item())
        if y_raw < 0 or y_raw > 2:
            continue
        g["target_result"] = torch.tensor([y_raw], dtype=torch.long)
        prepared.append(g)
    return prepared
# 기능: _split_dataset는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: graphs: List, valid_ratio: float, seed: int
#   - Output: Tuple[List, List]
def _split_dataset(graphs: List, valid_ratio: float, seed: int) -> Tuple[List, List]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(graphs))
    rng.shuffle(idx)

    n_valid = max(1, int(len(graphs) * valid_ratio))
    valid_idx = set(idx[:n_valid].tolist())

    train_graphs, valid_graphs = [], []
    for i, g in enumerate(graphs):
        if i in valid_idx:
            valid_graphs.append(g)
        else:
            train_graphs.append(g)
    return train_graphs, valid_graphs
# 기능: _batch_targets는 컬럼 'target_result'을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: batch
#   - Output: torch.Tensor
def _batch_targets(batch) -> torch.Tensor:
    y = batch["target_result"]
    if y.dim() == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y.long()
# 기능: _macro_f1_from_confusion는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: confusion: torch.Tensor
#   - Output: float
def _macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    f1s = []
    for c in range(confusion.size(0)):
        tp = float(confusion[c, c].item())
        fp = float(confusion[:, c].sum().item() - confusion[c, c].item())
        fn = float(confusion[c, :].sum().item() - confusion[c, c].item())
        denom = (2.0 * tp) + fp + fn
        f1s.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
    return float(np.mean(f1s)) if f1s else float("nan")
# 기능: 검증 로더를 순회하며 loss/accuracy/confusion을 집계하고 macro-F1을 함께 계산한다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: model: nn.Module, loader: DataLoader, device: torch.device
#   - Output: Tuple[float, float, float]
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    confusion = torch.zeros((3, 3), dtype=torch.long)
    criterion_sum = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = _batch_targets(batch).to(device)

        loss = criterion_sum(logits, y)
        total_loss += float(loss.item())
        n += int(y.numel())

        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == y).sum().item())

        yt = y.detach().cpu()
        pt = pred.detach().cpu()
        for i in range(int(yt.numel())):
            confusion[int(yt[i].item()), int(pt[i].item())] += 1

    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return total_loss / n, correct / n, _macro_f1_from_confusion(confusion)
# 기능: graph_pt를 학습/검증으로 분할하고 train_fold scaler를 fit한 뒤 CrossEntropy로 epoch 학습해 best_val_loss 모델 체크포인트를 저장한다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: args: argparse.Namespace
#   - Output: None
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    graphs = torch.load(args.graph_pt, weights_only=False)
    graphs = _prepare_graph_targets(graphs)

    if len(graphs) < 10:
        raise ValueError(f"Too few labeled graphs: {len(graphs)}")

    train_graphs, valid_graphs = _split_dataset(graphs, valid_ratio=args.valid_ratio, seed=args.seed)

    scaler_stats = fit_train_fold_scaler(train_graphs)
    transform_graphs_with_scaler_inplace(train_graphs, scaler_stats)
    transform_graphs_with_scaler_inplace(valid_graphs, scaler_stats)
    print("[INFO] fitted scaler on train fold and transformed train/valid graphs")

    if args.output_scaler is not None:
        save_scaler_payload(scaler_stats, args.output_scaler)
        print(f"[OK] scaler saved: {args.output_scaler}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False)

    model = HeteroEdgeGATWinPredictor(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        num_classes=args.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0

        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            y = _batch_targets(batch).to(device)

            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = int(y.numel())
            epoch_loss += float(loss.item()) * bs
            n += bs

        train_loss = epoch_loss / max(n, 1)
        val_loss, val_acc, val_f1 = evaluate(model, valid_loader, device)

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = float(val_loss)
            
            best_state = {}
            for k, v in model.state_dict().items():
                try:
                    _ = v.shape  # 1. 크기를 확인할 수 있는지 찔러봄
                    best_state[k] = v.detach().cpu() # 2. 정상이면 복사해서 저장
                except (RuntimeError, ValueError):
                    pass  # 3. 에러가 나면(빈 껍데기면) 무시하고 다음 부품으로 넘어감!

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "model_config": {
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "num_classes": args.num_classes,
        },
        "feature_scaler": export_scaler_payload(scaler_stats),
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "valid_ratio": args.valid_ratio,
            "seed": args.seed,
            "scaler_fit": "train_fold_only",
        },
        "n_train": len(train_graphs),
        "n_valid": len(valid_graphs),
        "best_val_loss": best_val,
    }
    torch.save(payload, args.output_model)
    print(f"[OK] model saved: {args.output_model}")
# 기능: build_argparser는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: argparse.ArgumentParser
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train hetero edge-aware GAT with semantic-level attention (Phase 5)")
    parser.add_argument(
        "--graph-pt",
        type=Path,
        default=DATA_DIR / "phase_4_synergy/data/gnn_phase4_5/hetero_graphs_non_england.pt",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DATA_DIR / "phase_5_lineup/data/gnn_phase5/hetero_edge_gat_win.pt",
    )
    parser.add_argument(
        "--output-scaler",
        type=Path,
        default=None,
        help="Optional path to save train-fold fitted scaler payload",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser
# 기능: main는 현재 단계에서 필요한 중간 표현을 기준으로 함수 목적에 맞는 산출물을 만든다.
# 동작/맥락: relation-wise GAT + semantic attention 학습에서 train-fold 전용 스케일링과 검증 분할을 일관되게 적용하기 위해 필요하다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: None
def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
