#!/usr/bin/env python3
from __future__ import annotations

"""
Step 3: Full-dataset training with Multi-GPU (A6000 x4) support via PyTorch DDP.

Features:
- Loads HeteroData list from Step 1 output (.pt)
- Uses HeteroEdgeGATWinPredictor from train_gnn_phase5.py
- DistributedDataParallel training (NCCL) with AMP
- Falls back to single-GPU/CPU if requested world size is unavailable
"""

import argparse
import importlib.util
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "team-builder"), Path(__file__).resolve().parents[1])
DATA_DIR = PROJECT_ROOT / "data"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_phase5_model_module(file_path: Path):
    spec = importlib.util.spec_from_file_location("train_gnn_phase5", file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def split_indices(n: int, valid_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_valid = max(1, int(n * valid_ratio))
    valid_idx = idx[:n_valid].tolist()
    train_idx = idx[n_valid:].tolist()
    return train_idx, valid_idx


def setup_process(rank: int, world_size: int, master_addr: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_process() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _batch_targets(batch) -> torch.Tensor:
    y = batch["target_result"]
    if y.dim() == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y.long()


def _macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    f1s = []
    for c in range(confusion.size(0)):
        tp = float(confusion[c, c].item())
        fp = float(confusion[:, c].sum().item() - confusion[c, c].item())
        fn = float(confusion[c, :].sum().item() - confusion[c, c].item())
        denom = (2.0 * tp) + fp + fn
        f1s.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    confusion = torch.zeros((3, 3), dtype=torch.long)
    criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")
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


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_process(rank, world_size, args.master_addr, args.master_port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    set_seed(args.seed + rank)

    phase5_mod = import_phase5_model_module(args.model_def)
    graphs = torch.load(args.graph_pt, weights_only=False)
    graphs = phase5_mod._prepare_graph_targets(graphs)

    if len(graphs) < 10:
        if rank == 0:
            raise ValueError(f"Too few labeled graphs: {len(graphs)}")
        cleanup_process()
        return

    train_idx, valid_idx = split_indices(len(graphs), args.valid_ratio, args.seed)
    train_graphs = [graphs[i] for i in train_idx]
    valid_graphs = [graphs[i] for i in valid_idx]

    scaler_stats = phase5_mod.fit_train_fold_scaler(train_graphs)
    phase5_mod.transform_graphs_with_scaler_inplace(train_graphs, scaler_stats)
    phase5_mod.transform_graphs_with_scaler_inplace(valid_graphs, scaler_stats)
    if rank == 0:
        print("[INFO] fitted scaler on train fold and transformed train/valid graphs")

    train_subset = Subset(graphs, train_idx)
    valid_subset = Subset(graphs, valid_idx)

    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_subset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = phase5_mod.HeteroEdgeGATWinPredictor(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        num_classes=args.num_classes,
    ).to(device)

    # Initialize lazy parameters (e.g., LazyLinear) before DDP wrapping.
    init_batch = next(iter(train_loader))
    init_batch = init_batch.to(device, non_blocking=True)
    with torch.no_grad():
        _ = model(init_batch)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        total_loss = 0.0
        n = 0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            y = _batch_targets(batch).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(batch)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            bs = int(y.numel())
            total_loss += float(loss.item()) * bs
            n += bs

        local_train_loss = total_loss / max(n, 1)
        local_val_loss, local_val_acc, local_val_f1 = evaluate(model, valid_loader, device)

        stats = torch.tensor([local_train_loss, local_val_loss, local_val_acc, local_val_f1], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats = stats / world_size

        train_loss = float(stats[0].item())
        val_loss = float(stats[1].item())
        val_acc = float(stats[2].item())
        val_f1 = float(stats[3].item())

        if rank == 0:
            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
            )
            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}

    if rank == 0:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": best_state if best_state is not None else model.module.state_dict(),
            "model_config": {
                "hidden_channels": args.hidden_channels,
                "num_layers": args.num_layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "num_classes": args.num_classes,
            },
            "feature_scaler": phase5_mod.export_scaler_payload(scaler_stats),
            "train_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "valid_ratio": args.valid_ratio,
                "seed": args.seed,
                "world_size": world_size,
                "amp": args.amp,
                "scaler_fit": "train_fold_only",
            },
            "best_val_loss": best_val,
            "n_graphs": len(graphs),
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
        }
        torch.save(payload, args.output_model)
        print(f"[OK] saved DDP model: {args.output_model}")
        if args.output_scaler is not None:
            phase5_mod.save_scaler_payload(scaler_stats, args.output_scaler)
            print(f"[OK] saved scaler: {args.output_scaler}")

    cleanup_process()


def single_process_train(args: argparse.Namespace) -> None:
    phase5_mod = import_phase5_model_module(args.model_def)
    graphs = torch.load(args.graph_pt, weights_only=False)
    graphs = phase5_mod._prepare_graph_targets(graphs)

    if len(graphs) < 10:
        raise ValueError(f"Too few labeled graphs: {len(graphs)}")

    train_idx, valid_idx = split_indices(len(graphs), args.valid_ratio, args.seed)
    train_graphs = [graphs[i] for i in train_idx]
    valid_graphs = [graphs[i] for i in valid_idx]

    scaler_stats = phase5_mod.fit_train_fold_scaler(train_graphs)
    phase5_mod.transform_graphs_with_scaler_inplace(train_graphs, scaler_stats)
    phase5_mod.transform_graphs_with_scaler_inplace(valid_graphs, scaler_stats)
    print("[INFO] fitted scaler on train fold and transformed train/valid graphs")

    train_subset = Subset(graphs, train_idx)
    valid_subset = Subset(graphs, valid_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = phase5_mod.HeteroEdgeGATWinPredictor(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        num_classes=args.num_classes,
    ).to(device)

    # Initialize lazy parameters (e.g., LazyLinear) before any state_dict save path.
    init_batch = next(iter(train_loader))
    init_batch = init_batch.to(device)
    with torch.no_grad():
        _ = model(init_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            batch = batch.to(device)
            y = _batch_targets(batch).to(device)

            logits = model(batch)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = int(y.numel())
            total_loss += float(loss.item()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)
        val_loss, val_acc, val_f1 = evaluate(model, valid_loader, device)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

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
        "feature_scaler": phase5_mod.export_scaler_payload(scaler_stats),
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "valid_ratio": args.valid_ratio,
            "seed": args.seed,
            "world_size": 1,
            "amp": False,
            "scaler_fit": "train_fold_only",
        },
        "best_val_loss": best_val,
        "n_graphs": len(graphs),
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
    }
    torch.save(payload, args.output_model)
    print(f"[OK] saved single-process model: {args.output_model}")
    if args.output_scaler is not None:
        phase5_mod.save_scaler_payload(scaler_stats, args.output_scaler)
        print(f"[OK] saved scaler: {args.output_scaler}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 3 Multi-GPU training for hetero GNN")
    p.add_argument(
        "--graph-pt",
        type=Path,
        default=DATA_DIR / "phase_4_synergy/data/gnn_phase4_5/hetero_graphs_non_england.pt",
    )
    p.add_argument(
        "--model-def",
        type=Path,
        default=PROJECT_ROOT / "proposed_spatial_gnn_ga/train_gnn_phase5.py",
        help="Path to model definition file containing HeteroEdgeGATWinPredictor",
    )
    p.add_argument(
        "--output-model",
        type=Path,
        default=DATA_DIR / "phase_5_lineup/data/gnn_phase5/hetero_edge_gat_win_ddp_full.pt",
    )
    p.add_argument(
        "--output-scaler",
        type=Path,
        default=None,
        help="Optional path to save train-fold fitted scaler payload",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-channels", type=int, default=96)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--valid-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpus", type=int, default=4, help="Requested number of GPUs (A6000 x4 => 4)")
    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29610")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_model.parent.mkdir(parents=True, exist_ok=True) # <-- 이 줄 추가!
    set_seed(args.seed)

    available = torch.cuda.device_count()
    requested = int(args.gpus)
    world_size = min(requested, available)

    if world_size >= 2:
        print(f"[INFO] Launching DDP with world_size={world_size} (requested={requested}, available={available})")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print(
            f"[WARN] Multi-GPU unavailable (requested={requested}, available={available}). "
            "Running single-process training."
        )
        single_process_train(args)


if __name__ == "__main__":
    main()
