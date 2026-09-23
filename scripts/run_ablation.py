"""Run a reproducible Top-K ablation on ETTh1 and save metrics/figure."""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloader
from src.forecasting_model import FANIntegratedModel
from src.trainer import evaluate_model, train_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    seq_len, pred_len, enc_in = 96, 24, 7
    data_path = "data/ETTh1.csv"
    train_loader = get_dataloader(data_path, "train", seq_len, pred_len, args.batch_size)
    val_loader = get_dataloader(data_path, "val", seq_len, pred_len, args.batch_size)
    test_loader = get_dataloader(data_path, "test", seq_len, pred_len, args.batch_size)
    rows = []

    for top_k in args.top_k:
        set_seed(args.seed)
        model = FANIntegratedModel(seq_len, pred_len, enc_in, freq_topk=top_k)
        model = train_model(
            model, train_loader, val_loader, num_epochs=args.epochs, device=device
        )
        mse, mae = evaluate_model(model, test_loader, device=device)
        checkpoint = f"outputs/models/ETTh1_FAN_topk{top_k}_24.pt"
        torch.save(model.state_dict(), checkpoint)
        rows.append({"top_k": top_k, "mse": round(mse, 8), "mae": round(mae, 8)})
        print(f"Top-K={top_k}: MSE={mse:.4f}, MAE={mae:.4f}")

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/ablation_topk.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["top_k", "mse", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    with open("outputs/results/ablation_topk.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "results": rows},
            handle,
            indent=2,
        )

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["top_k"] for row in rows], [row["mse"] for row in rows], marker="o", label="MSE")
    axis.plot([row["top_k"] for row in rows], [row["mae"] for row in rows], marker="s", label="MAE")
    axis.set_title("ETTh1 FAN Top-K ablation (horizon=24)")
    axis.set_xlabel("Top-K frequency components")
    axis.set_ylabel("Error")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig("outputs/figures/ablation_topk.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
