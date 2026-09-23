"""Train FAN and DLinear reproducibly and persist checkpoints plus metrics."""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_models import DLinear
from src.dataset import get_dataloader
from src.forecasting_model import FANIntegratedModel
from src.trainer import evaluate_model, train_model


DATASETS = {
    "ETTh1": "data/ETTh1.csv",
    "ETTh2": "data/ETTh2.csv",
    "ETTm1": "data/ETTm1.csv",
    "ETTm2": "data/ETTm2.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--pred-lens", nargs="+", type=int,
        default=[12, 24, 36, 48, 60, 72, 84, 96],
    )
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--freq-topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Explicit quick-run limit. Omit for complete training and evaluation.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(rows, configuration, result_dir="outputs/results"):
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "training_results.csv")
    json_path = os.path.join(result_dir, "training_results.json")

    fieldnames = [
        "dataset", "model", "seq_len", "pred_len", "epochs", "mse", "mae",
        "best_val_mse", "training_seconds", "checkpoint",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "results": rows,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be a positive integer")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)

    os.makedirs("outputs/models", exist_ok=True)
    set_seed(args.seed)
    print(f"Using device: {device}")
    print("Batch coverage:", "FULL" if args.max_batches is None else args.max_batches)

    rows = []
    configuration = vars(args).copy()
    configuration["device_used"] = str(device)

    for dataset_name in args.datasets:
        data_path = DATASETS[dataset_name]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Required dataset not found: {data_path}")

        print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
        for pred_len in args.pred_lens:
            train_loader = get_dataloader(
                data_path, "train", args.seq_len, pred_len, args.batch_size
            )
            val_loader = get_dataloader(
                data_path, "val", args.seq_len, pred_len, args.batch_size
            )
            test_loader = get_dataloader(
                data_path, "test", args.seq_len, pred_len, args.batch_size
            )

            for model_index, model_name in enumerate(("FAN", "DLinear")):
                set_seed(args.seed + pred_len + model_index)
                if model_name == "FAN":
                    model = FANIntegratedModel(
                        args.seq_len, pred_len, 7, freq_topk=args.freq_topk
                    )
                else:
                    model = DLinear(args.seq_len, pred_len, 7)
                print(f"\nTraining {model_name}: pred_len={pred_len}")
                started = time.perf_counter()
                model = train_model(
                    model,
                    train_loader,
                    val_loader,
                    num_epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    device=device,
                    max_batches=args.max_batches,
                )
                mse, mae = evaluate_model(
                    model, test_loader, device=device, max_batches=args.max_batches
                )
                elapsed = time.perf_counter() - started

                checkpoint = os.path.join(
                    "outputs/models", f"{dataset_name}_{model_name}_{pred_len}.pt"
                )
                torch.save(model.state_dict(), checkpoint)
                history = getattr(model, "training_history", [])
                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "seq_len": args.seq_len,
                    "pred_len": pred_len,
                    "epochs": args.epochs,
                    "mse": round(mse, 8),
                    "mae": round(mae, 8),
                    "best_val_mse": round(min(h["val_mse"] for h in history), 8),
                    "training_seconds": round(elapsed, 3),
                    "checkpoint": checkpoint.replace("\\", "/"),
                })
                save_results(rows, configuration)
                print(f"Checkpoint saved: {checkpoint}")

    print(f"\nCompleted {len(rows)} full model evaluations.")
    print("Results: outputs/results/training_results.csv")


if __name__ == "__main__":
    main()
