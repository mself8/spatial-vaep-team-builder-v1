#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 5: 이종 그래프 신경망(Heterogeneous GNN) 학습 — 3-클래스 경기 결과 예측.

=== 전체 모델 파이프라인 ===
입력: HeteroData (home_team 노드 11개 + away_team 노드 11개, 4가지 관계 엣지)
  └─ 노드 특징: 24D = [off_12(공격 공간 VAEP), def_12(수비 공간 VAEP)]  per 90분 밀도 정규화
  └─ 엣지 특징: 12D 공간 시너지 벡터 (IO=패스 협력, ID=수비 대결)

[Layer 1 ~ L]
  1. GATConv 4개 독립 인코더 (EncoderHome_IO / EncoderAway_IO / EncoderHome_ID / EncoderAway_ID)
     어텐션 계수: α_ij = softmax(LeakyReLU( a^T [Wh_i ‖ Wh_j ‖ W^e·e_ij] ))
     - edge_dim=12 → 엣지 특징 e_ij가 어텐션 계산에 직접 참여 (표준 GATv1 확장)
     - concat=False → 멀티헤드 출력을 평균으로 합산
     - IO(passes_to): 같은 팀 내 방향성 패스 협력 관계 (src→dst 공간 협업 강도)
     - ID(defends_against): 홈수비→어웨이공격, 어웨이수비→홈공격 대결 관계

  2. SemanticLevelAttention — HAN(Heterogeneous Attention Network) 방식 의미 수준 어텐션
     각 목적지 노드 타입마다 R=2개 관계 임베딩을 가중합
     수식:
       e_r   = mean_{v ∈ V} [ tanh(W · h_r_v) · q ]   (그래프 레벨 의미 점수)
       β     = softmax([ e_1, e_2 ])                    (관계 중요도 가중치, β_1+β_2=1)
       h_out = β_1·h_r1_v + β_2·h_r2_v                (관계 임베딩 가중합)
     - home: R1=home_io_out(팀 내 패스), R2=away_id_out(어웨이가 홈를 수비한 결과)
     - away: R1=away_io_out(팀 내 패스), R2=home_id_out(홈이 어웨이를 수비한 결과)

  3. Global Mean Pooling → [B, hidden] 그래프 레벨 표현
  4. cat([home_pool, away_pool]) → [B, 2·hidden]
  5. MLP head: Linear→ReLU→Dropout→Linear → 3-class logits [loss=0, draw=1, win=2]
  6. 학습: CrossEntropyLoss, AdamW, gradient clip(max_norm=5.0), early stopping(patience=7)
  7. 추론: softmax(logits / T), T=2.0 (temperature scaling for calibration)

=== 데이터 누수 방지 ===
  - Scaler(z-score mean/std)를 train fold 전용으로 fit → valid에 transform만 적용
  - 그래프 자체는 Phase 4.5에서 rolling-window(match_time < cur_match_time) 필터로 생성됨
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
# 기능: gnn_feature_scaler 모듈을 import한다. 동일 디렉토리에 있을 경우 직접 import, 없으면 파일 경로로 동적 로드한다.
# 동작/맥락: train_gnn_phase5.py 단독 실행 또는 다른 디렉토리에서 호출될 때 모두 스케일러 모듈을 안정적으로 불러오기 위해 필요하다.
# 데이터 입출력:
#   - Input: 없음
#   - Output: gnn_feature_scaler 모듈 객체
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
# 기능: train fold 그래프들로부터 node/edge 특징의 전역 z-score 통계(mean, std)를 계산한다.
# 동작/맥락: train fold 데이터만으로 scaler를 fit하여 valid/test에 동일 통계를 적용함으로써
#            데이터 누수(data leakage)를 차단한다. valid 그래프에 fit하면 정보 누수가 발생한다.
# 데이터 입출력:
#   - Input: train_graphs: List[HeteroData]
#   - Output: Dict[str, torch.Tensor] — {'node_mean':24D, 'node_std':24D, 'passes_mean':12D, ...}
def fit_train_fold_scaler(train_graphs: List) -> Dict[str, torch.Tensor]:
    """Fit scaler stats from train fold graphs only."""
    return _SCALER_MOD.fit_feature_scaler(train_graphs)
# 기능: 이미 fit된 scaler 통계로 그래프 리스트의 node_x와 edge_attr를 제자리(in-place) z-score 변환한다.
# 동작/맥락: train/valid 분할 후 train에서 fit한 scaler를 valid에도 transform-only로 적용한다.
#            (x - mean) / std 연산이 각 node/edge 타입 별로 독립적으로 수행된다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData], scaler_stats: Dict[str, torch.Tensor]
#   - Output: None (그래프 객체 내부 텐서를 직접 수정)
def transform_graphs_with_scaler_inplace(graphs: List, scaler_stats: Dict[str, torch.Tensor]) -> None:
    """Apply previously fitted scaler to any graph split (train/valid/inference)."""
    _SCALER_MOD.apply_global_zscore_to_graphs_inplace(graphs, scaler_stats)
# 기능: 내부 scaler_stats 딕셔너리를 모델 체크포인트에 함께 저장 가능한 직렬화 포맷으로 변환한다.
# 동작/맥락: 학습 완료 후 모델 .pt 파일 안에 scaler를 함께 저장해 추론 시 별도 scaler 파일 없이도 복원 가능하게 한다.
# 데이터 입출력:
#   - Input: scaler_stats: Dict[str, torch.Tensor]
#   - Output: Dict[str, object] — {'node': {'mean':..., 'std':...}, 'passes_to':..., 'defends_against':...}
def export_scaler_payload(scaler_stats: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return _SCALER_MOD.scaler_stats_to_payload(scaler_stats)
# 기능: scaler 통계를 .pt 파일로 저장한다 (모델과 별도로 독립 파일이 필요한 경우에 사용).
# 동작/맥락: 선택적으로 --output-scaler 경로를 지정할 때만 호출되며, 일반적으로는 모델 체크포인트 내부에 포함된다.
# 데이터 입출력:
#   - Input: scaler_stats: Dict[str, torch.Tensor], output_path: Path
#   - Output: None
def save_scaler_payload(scaler_stats: Dict[str, torch.Tensor], output_path: Path) -> None:
    _SCALER_MOD.save_feature_scaler(scaler_stats, output_path)


# ----------------------------
# Core utilities
# ----------------------------
# 기능: 관계 튜플 (src_type, relation, dst_type)을 nn.ModuleDict 키 문자열로 변환한다.
# 동작/맥락: PyG의 HeteroConv와 달리 여기서는 각 관계별 GATConv를 nn.ModuleDict에 직접 저장하므로
#            튜플을 "home_team__passes_to__home_team" 형태의 문자열 키로 변환해야 한다.
# 데이터 입출력:
#   - Input: rel: Tuple[str, str, str]  예) ("home_team", "passes_to", "home_team")
#   - Output: str  예) "home_team__passes_to__home_team"
def _rel_key(rel: Tuple[str, str, str]) -> str:
    return "__".join(rel)
# 기능: Python random / NumPy / PyTorch 모두의 랜덤 시드를 동일 값으로 고정한다.
# 동작/맥락: 실험 재현성을 보장하기 위해 학습 전 반드시 호출한다. CUDA 멀티GPU 환경에서도 시드가 적용된다.
# 데이터 입출력:
#   - Input: seed: int  예) 42
#   - Output: None
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SemanticLevelAttention(nn.Module):
    """HAN(Heterogeneous Attention Network) 방식의 의미 수준(Semantic-Level) 어텐션.

    각 GATConv 인코더가 생성한 관계별 노드 임베딩 {h_r : r=1..R}를 받아
    관계 중요도 가중치 β를 계산하고 가중합으로 최종 노드 임베딩을 출력한다.

    수식 (HAN Eq. 5-6 변형):
      e_r   = (1/|V|) Σ_v tanh(W · h_r_v) · q     [그래프 레벨 의미 점수, 스칼라]
      β     = softmax([e_1, ..., e_R])              [관계 중요도 가중치, R차원]
      h_out_v = Σ_r β_r · h_r_v                    [관계 임베딩 가중합, hidden_channels차원]

    파라미터:
      proj (nn.Linear): 노드 임베딩을 tanh 공간으로 투영하는 가중치 W ∈ R^{d×d}
      query (nn.Parameter): 의미 공간에서 중요도를 측정하는 쿼리 벡터 q ∈ R^d
    """
    # 기능: SemanticLevelAttention의 투영 레이어(proj)와 쿼리 벡터(query)를 초기화한다.
    # 동작/맥락: proj는 Xavier 초기화, query는 uniform(-0.1, 0.1)로 초기화하여 초기 학습 안정성을 확보한다.
    # 데이터 입출력:
    #   - Input: hidden_channels: int — GATConv 출력 임베딩 차원 d
    #            num_relations: int   — 융합할 관계 수 R (home=2: IO+ID, away=2: IO+ID)
    #            dropout: float       — β 가중치에 적용할 드롭아웃 비율 (학습 중에만 적용)
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
    # 기능: 노드별 스칼라 점수를 그래프 단위로 평균화하여 그래프 레벨 의미 점수 e_r을 산출한다.
    # 동작/맥락: HAN의 e_r = (1/|V|) Σ_v score_v 계산에 해당한다.
    #            미니배치 내 여러 그래프가 혼합되어 있으므로 batch 벡터로 그래프별 분리 후 평균한다.
    # 데이터 입출력:
    #   - Input: node_values: torch.Tensor [N] — 노드별 스칼라 점수 (tanh(Wh)·q 결과)
    #            batch: torch.Tensor [N]        — 각 노드가 속한 그래프 인덱스 (0~B-1)
    #            n_graphs: int                  — 배치 내 전체 그래프 수 B
    #   - Output: torch.Tensor [B] — 그래프별 평균 의미 점수
    @staticmethod
    def _graph_mean_scalar(node_values: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        summed = node_values.new_zeros((n_graphs,))
        count = node_values.new_zeros((n_graphs,))
        ones = node_values.new_ones((node_values.size(0),))
        summed.index_add_(0, batch, node_values)
        count.index_add_(0, batch, ones)
        return summed / count.clamp_min(1.0)
    # 기능: R개 관계별 노드 임베딩 리스트를 받아 의미 가중치 β로 가중합한 최종 임베딩을 반환한다.
    # 동작/맥락: 수식 흐름:
    #   for r in range(R):
    #     node_score[r] = tanh(proj(h_r)) · query          # [N] — 노드별 의미 점수
    #     graph_score[r] = mean(node_score[r], by batch)   # [B] — 그래프 레벨 의미 점수 e_r
    #   scores = stack([graph_score_1, ..., graph_score_R]) # [B, R]
    #   β = softmax(scores, dim=1)                          # [B, R] — 관계 중요도 가중치
    #   h_out = Σ_r β_r[batch] * h_r                       # [N, d] — 가중합 최종 임베딩
    # 데이터 입출력:
    #   - Input: relation_embeddings: List[Tensor[N, d]] — 각 GATConv 인코더의 출력 (R개)
    #            batch: Tensor[N] — 노드→그래프 인덱스 매핑
    #            return_weights: bool — True이면 β([B, R])도 함께 반환 (디버그/시각화용)
    #   - Output: Tensor[N, d] 또는 (Tensor[N, d], β[B, R])
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
    """엣지 특징을 활용하는 이종 그래프 GATConv + HAN 방식 의미 어텐션 기반 경기 결과 예측 모델.

    구조 요약:
      - 4개 독립 GATConv 인코더 (관계당 1개): EncoderHome_IO, EncoderAway_IO, EncoderHome_ID, EncoderAway_ID
      - 레이어마다 home/away 각각에 SemanticLevelAttention(R=2) 적용
      - 최종 Global Mean Pool → cat → MLP → 3-class logits

    어텐션 공식 (GATv1 with edge features):
      α_ij = softmax_j( LeakyReLU( a^T [ Wh_i ‖ Wh_j ‖ W^e·e_ij ] ) )
      h_i' = Σ_j α_ij · Wh_j
    여기서 e_ij는 12D 공간 시너지 벡터 (IO=패스 협력 강도, ID=수비 대결 강도)
    """
    # 기능: num_layers개의 (GATConv×4 + SemanticAttn×2) 레이어 블록과 MLP 헤드를 초기화한다.
    # 동작/맥락:
    #   - rel_convs: ModuleList[ModuleDict] — 레이어별로 4개 GATConv를 독립 파라미터로 관리
    #   - home_semantic, away_semantic: ModuleList[SemanticLevelAttention] — 레이어별 의미 어텐션
    #   - head: Linear(2*hidden) → ReLU → Dropout → Linear(num_classes) — 3-class 분류기
    #   - GATConv 입력 차원 (-1,-1): PyG lazy init → 첫 forward 시 실제 입력 차원으로 자동 결정
    #   - add_self_loops=False: 이종 그래프에서는 src/dst 타입이 다를 수 있어 self-loop 불필요
    # 데이터 입출력:
    #   - Input: hidden_channels: int — 각 GATConv 출력 차원 (멀티헤드 평균 후)
    #            num_layers: int      — GATConv + SemanticAttn 스택 반복 횟수
    #            heads: int           — GATConv 멀티헤드 수 (concat=False → 평균으로 합산)
    #            dropout: float       — GATConv 어텐션 드롭아웃 및 MLP 드롭아웃 비율
    #            num_classes: int     — 출력 클래스 수 (3: loss/draw/win)
    #   - Output: None
    def __init__(
        self,
        hidden_channels: int = 32,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.40,
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
                nn.Linear(2 * int(hidden_channels), int(hidden_channels)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
                nn.Linear(int(hidden_channels), self.num_classes),
        )
    # 기능: HeteroData 노드 스토어에서 batch 벡터를 꺼낸다. 단일 그래프(비배치)이면 0으로 채운 텐서를 반환한다.
    # 동작/맥락: DataLoader로 배치한 경우 node_store.batch가 자동 생성되지만,
    #            단일 HeteroData를 직접 forward에 넣을 때는 batch 속성이 없으므로 이 방어 코드가 필요하다.
    # 데이터 입출력:
    #   - Input: node_store — HeteroData의 노드 타입 스토어 (data["home_team"] 등)
    #   - Output: torch.Tensor [N] — 각 노드가 속한 그래프 인덱스 (모두 0이면 단일 그래프)
    @staticmethod
    def _node_batch(node_store) -> torch.Tensor:
        if hasattr(node_store, "batch") and node_store.batch is not None:
            return node_store.batch
        return torch.zeros(node_store.x.size(0), dtype=torch.long, device=node_store.x.device)
    # 기능: HeteroData를 입력받아 3-class 경기 결과 logits를 출력하는 전체 forward 패스를 수행한다.
    # 동작/맥락: 단계별 흐름:
    #   ① 입력 추출: x_home[N_h, 24], x_away[N_a, 24] 및 4종 edge_index/edge_attr 추출
    #   ② for 레이어 l=1..L:
    #      - home_io_out  = GATConv_HOME_IO(x_home, home→home IO 엣지, 엣지attr 12D)  → [N_h, d]
    #      - away_io_out  = GATConv_AWAY_IO(x_away, away→away IO 엣지, 엣지attr 12D)  → [N_a, d]
    #      - home_id_out  = GATConv_HOME_ID((x_home, x_away), home→away ID 엣지, ...)  → [N_a, d]
    #                       ※ src=home(수비측), dst=away(공격측) → away 관점에서 home 수비를 어텐션
    #      - away_id_out  = GATConv_AWAY_ID((x_away, x_home), away→home ID 엣지, ...)  → [N_h, d]
    #                       ※ src=away(수비측), dst=home(공격측) → home 관점에서 away 수비를 어텐션
    #      - x_home = SemanticAttn_home([home_io_out, away_id_out])  → [N_h, d]
    #                 R1=home_io_out: 홈팀 내 패스 협력 관계
    #                 R2=away_id_out: 어웨이팀이 홈을 수비한 결과(홈 공격 억제 정도)
    #      - x_away = SemanticAttn_away([away_io_out, home_id_out])  → [N_a, d]
    #                 R1=away_io_out: 어웨이팀 내 패스 협력 관계
    #                 R2=home_id_out: 홈팀이 어웨이를 수비한 결과(어웨이 공격 억제 정도)
    #      - ReLU + Dropout 적용
    #   ③ home_pool = global_mean_pool(x_home, batch) → [B, d]
    #      away_pool = global_mean_pool(x_away, batch) → [B, d]
    #   ④ match_repr = cat([home_pool, away_pool]) → [B, 2d]
    #   ⑤ logits = MLP(match_repr) → [B, 3]  (loss=0, draw=1, win=2)
    # 데이터 입출력:
    #   - Input: data: HeteroData (배치 또는 단일 그래프)
    #            return_debug: bool — True이면 각 레이어 shape 및 β 가중치 딕셔너리도 반환
    #   - Output: logits [B, 3] 또는 (logits, debug_dict)
    def forward(self, data, return_debug: bool = False):
        x_home = data["home_team"].x
        x_away = data["away_team"].x
        # print("== Forward pass ==")
        # print(data)
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
            # away_id_out = rel_layer[_rel_key(REL_HOME_ID)](
            #     (x_away, x_home),
            #     edge_index_away_id,
            #     edge_attr=edge_attr_away_id,
            # )
            # print(f"After GATConv layer {li}:")
            # print(f"  x_home: {x_home.shape}, x_away: {x_away.shape}, edge_index_home_io: {edge_index_home_io.shape}, edge_attr_home_io: {edge_attr_home_io.shape}")
            # print(f"Layer {li} - home_io_out: {home_io_out.shape}, away_io_out: {away_io_out.shape}, home_id_out: {home_id_out.shape}, away_id_out: {away_id_out.shape}")
            # exit()

            x_home, home_beta = self.home_semantic[li - 1]([home_io_out, away_id_out], home_batch, return_weights=True)
            x_away, away_beta = self.away_semantic[li - 1]([away_io_out, home_id_out], away_batch, return_weights=True)
            #x_away, away_beta = self.home_semantic[li - 1]([away_io_out, home_id_out], away_batch, return_weights=True)

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
        debug["home_pool"] = tuple(home_pool.shape)
        debug["away_pool"] = tuple(away_pool.shape)

        match_repr = torch.cat([home_pool, away_pool], dim=-1)
        debug["concat_match_repr"] = tuple(match_repr.shape)
        logits = self.head(match_repr)
        debug["logits"] = tuple(logits.shape)

        if return_debug:
            return logits, debug
        return logits


# ----------------------------
# Dataset / training utilities
# ----------------------------
# 기능: match_y가 있는 그래프만 남기고 CrossEntropyLoss에 필요한 정수형 레이블 target_result를 생성한다.
# 동작/맥락: Phase 4.5에서 스코어 정보가 없는 경기에는 match_y가 없으므로 레이블 없는 샘플을 제거한다.
#            레이블 인코딩: loss=0 (홈 패), draw=1 (무승부), win=2 (홈 승)
# 데이터 입출력:
#   - Input: graphs: List[HeteroData]
#   - Output: List[HeteroData] — match_y가 존재하고 0~2 범위인 그래프만 포함
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
# 기능: 전체 그래프를 seed 고정 셔플로 train/valid로 분할한다.
# 동작/맥락: valid_ratio=0.2이면 20%를 검증 세트로 분리. scaler는 반드시 train fold에서만 fit해야 하므로
#            이 분할 후 fit_train_fold_scaler(train_graphs)를 호출해야 한다.
# 데이터 입출력:
#   - Input: graphs: List[HeteroData], valid_ratio: float (0~1), seed: int
#   - Output: Tuple[train_graphs, valid_graphs]
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
# 기능: DataLoader 배치에서 CrossEntropyLoss에 입력할 1D long 텐서를 추출한다.
# 동작/맥락: DataLoader가 [B, 1] 형태로 묶기도 하므로 squeeze(-1)로 [B]로 변환.
#            CrossEntropyLoss는 logits[B, C]와 target[B]를 받는다.
# 데이터 입출력:
#   - Input: batch — DataLoader가 묶은 HeteroData 배치
#   - Output: torch.Tensor [B] (dtype=long) — loss=0, draw=1, win=2
def _batch_targets(batch) -> torch.Tensor:
    y = batch["target_result"]
    if y.dim() == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y.long()
# 기능: 3×3 혼동 행렬에서 클래스별 F1을 계산하고 매크로 평균을 반환한다.
# 동작/맥락: scikit-learn 없이 순수 텐서 연산으로 Macro-F1을 산출한다.
#            F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c), Macro-F1 = mean(F1_0, F1_1, F1_2)
#            클래스 불균형(패>>무승부>>홈승) 환경에서 accuracy보다 균형잡힌 지표를 제공한다.
# 데이터 입출력:
#   - Input: confusion: torch.Tensor [3, 3] — confusion[실제_클래스, 예측_클래스]
#   - Output: float — Macro-F1 (0~1)
def _macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    f1s = []
    for c in range(confusion.size(0)):
        tp = float(confusion[c, c].item())
        fp = float(confusion[:, c].sum().item() - confusion[c, c].item())
        fn = float(confusion[c, :].sum().item() - confusion[c, c].item())
        denom = (2.0 * tp) + fp + fn
        f1s.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
    return float(np.mean(f1s)) if f1s else float("nan")
# 기능: 검증 DataLoader 전체를 순회하며 평균 CrossEntropy loss, accuracy, Macro-F1을 계산한다.
# 동작/맥락: torch.no_grad()로 그래디언트를 비활성화하여 메모리를 절약하고 추론 속도를 높인다.
#            model.eval() 모드이므로 Dropout이 비활성화되고 BatchNorm은 running stats를 사용한다.
#            reduction="sum" CrossEntropy로 손실을 누적 후 n으로 나눠 배치 크기 불균형을 보정한다.
# 데이터 입출력:
#   - Input: model: nn.Module, loader: DataLoader, device: torch.device
#   - Output: Tuple[val_loss: float, val_acc: float, val_macro_f1: float]
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
# 기능: HeteroEdgeGATWinPredictor를 end-to-end 학습하고 best_val_loss 체크포인트를 .pt 파일로 저장한다.
# 동작/맥락: 전체 학습 흐름:
#   ① graph_pt 로드 → _prepare_graph_targets로 레이블 없는 그래프 제거
#   ② _split_dataset → train/valid 분리 (valid_ratio=20%)
#   ③ fit_train_fold_scaler(train_graphs) → train fold에서만 z-score 통계 계산
#   ④ transform_graphs_with_scaler_inplace(train + valid) → 동일 통계로 변환
#   ⑤ DataLoader(batch_size=1) → 경기당 1그래프 단위로 학습 (작은 데이터셋)
#   ⑥ AdamW + CrossEntropyLoss + gradient clip(max_norm=5.0) 으로 매 epoch 학습
#   ⑦ Early stopping: patience=7 (val_loss 개선 없는 epoch 7회 연속 시 중단)
#      - best_state: val_loss 최소 epoch의 model.state_dict()를 clone하여 보존
#      - 임계값 1e-8로 수치 노이즈에 의한 false-positive 개선 방지
#   ⑧ 학습 종료 후 best_state로 모델 복원 → CPU 이동 → state_dict 추출
#   ⑨ payload = {state_dict, model_config, feature_scaler, train_config} 를 torch.save
#      (추론 시 이 단일 파일만으로 모델 복원과 스케일러 적용이 가능하도록 설계)
# 데이터 입출력:
#   - Input: args: argparse.Namespace — CLI 인수 파싱 결과
#   - Output: None (.pt 파일로 체크포인트 저장)
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    graphs = torch.load(args.graph_pt, weights_only=False)
    graphs = _prepare_graph_targets(graphs)
    # print(f"[INFO] loaded {len(graphs)} graphs from: {args.graph_pt}")
    # print(graphs)

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
    patience = 7
    no_improve = 0

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

        if np.isfinite(val_loss) and val_loss < best_val - 1e-8:
            best_val = float(val_loss)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[EARLY STOP] patience={patience} exhausted at epoch {epoch}, best_val={best_val:.5f}")
            break

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure ALL lazy parameters are initialized by running full validation pass
    print("[INFO] Finalizing model: running validation pass to initialize all lazy parameters...")
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            _ = model(batch)
    print("[INFO] All lazy parameters now initialized")
    
    # Restore best checkpoint before saving
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Restored best model (val_loss={best_val:.5f})")

    # Move model to CPU and extract state dict safely
    model.cpu()
    final_state_dict = best_state if best_state is not None else dict(model.state_dict())
    
    payload = {
        "state_dict": final_state_dict,
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
    print(f"[OK] model saved: {args.output_model} with {len(final_state_dict)} state keys")
# 기능: CLI 인수 파서를 생성하고 모든 학습 하이퍼파라미터의 기본값을 설정한다.
# 동작/맥락: 기본값은 하이퍼파라미터 튜닝(train_gnn_phase5_tune.py)으로 결정된 최적 설정이다.
#   - hidden_channels=12, num_layers=1, heads=16, dropout=0.15, lr=1e-3
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
    parser.add_argument("--batch-size", type=int, default=1) # 0.16
    parser.add_argument("--hidden-channels", type=int, default=12) # 64
    parser.add_argument("--num-layers", type=int, default=1) # 1
    parser.add_argument("--heads", type=int, default=16) # 8
    parser.add_argument("--dropout", type=float, default=0.15) # 0.15
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser
# 기능: CLI 진입점. 인수를 파싱하여 train()을 호출한다.
# 데이터 입출력:
#   - Input: sys.argv — CLI 인수 (python train_gnn_phase5.py --graph-pt ... --output-model ...)
#   - Output: None
def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
