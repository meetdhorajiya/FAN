"""Experimental extension: Investigate impact of training duration (3, 10, 20 epochs)
on FAN and DLinear forecasting performance across ETT datasets.

Preserves:
- Chronological 70/20/10 split
- StandardScaler fitted strictly on training data
- Sequence length = 96
- Adam optimizer (lr = 0.001)
- Batch size = 32
- Model selection via minimum validation MSE with best state restoration
- Complete test split evaluation (element-weighted MSE and MAE)
- Deterministic seeding (seed + pred_len + model_index)

Saves all checkpoints to outputs/extension_models/ and results to outputs/extension_results/.
Original checkpoints and result files are never overwritten.
"""

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

DEFAULT_HORIZONS = {
    "ETTh1": [12, 24, 36, 48, 60, 72, 84, 96],
    "ETTh2": [12, 24, 48, 96],
    "ETTm1": [24, 48, 96],
    "ETTm2": [24, 48, 96],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS.keys()), default=["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    )
    parser.add_argument(
        "--epochs-list", nargs="+", type=int, default=[3, 10, 20],
        help="List of epoch budgets to compare (default: [3, 10, 20])"
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=None,
        help="Override prediction horizons for the run"
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Force re-run even if configuration is already recorded in results"
    )
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--freq-topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/extension_results"
    )
    parser.add_argument(
        "--model-dir", type=str, default="outputs/extension_models"
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Optional debug batch cap. Omit for full scientific runs."
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(rows, configuration, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "epoch_budget_results.csv")
    json_path = os.path.join(output_dir, "epoch_budget_results.json")

    fieldnames = [
        "dataset", "model", "seq_len", "pred_len", "epoch_budget",
        "best_val_epoch", "best_val_mse", "final_val_mse", "final_train_mse",
        "test_mse", "test_mae", "training_seconds", "seed", "checkpoint"
    ]
    csv_rows = [{k: v for k, v in r.items() if k != "training_history"} for r in rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "total_experiments": len(rows),
        "results": rows,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"=== FAN & DLinear Epoch Budget Extension ===")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Epoch budgets: {args.epochs_list}")
    print(f"Sequence length: {args.seq_len}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model dir: {args.model_dir}")

    config = vars(args).copy()
    config["device_used"] = str(device)

    rows = []
    json_path = os.path.join(args.output_dir, "epoch_budget_results.json")
    if os.path.exists(json_path) and not args.force:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                existing_data = json.load(handle)
                rows = existing_data.get("results", [])
                print(f"Loaded {len(rows)} previously recorded results.")
        except Exception as e:
            print(f"Warning: could not load existing results ({e}); starting fresh.")

    def is_already_run(ds, m, pl, ep):
        return any(
            r["dataset"] == ds and r["model"] == m and r["pred_len"] == pl and r["epoch_budget"] == ep
            for r in rows
        )

    total_start = time.perf_counter()

    for dataset_name in args.datasets:
        data_path = DATASETS[dataset_name]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        horizons = args.horizons if args.horizons is not None else DEFAULT_HORIZONS[dataset_name]
        print(f"\n{'='*70}\nDataset: {dataset_name} (Horizons: {horizons})\n{'='*70}")

        for pred_len in horizons:
            # Check if all runs for this pred_len are already done
            all_done = True
            for epoch_budget in args.epochs_list:
                for model_name in ("FAN", "DLinear"):
                    if not is_already_run(dataset_name, model_name, pred_len, epoch_budget):
                        all_done = False
                        break
                if not all_done:
                    break

            if all_done and not args.force:
                print(f"Skipping {dataset_name} H={pred_len}: all requested epoch budgets already evaluated.")
                continue

            # Load dataloaders once per dataset + horizon combination
            train_loader = get_dataloader(
                data_path, "train", args.seq_len, pred_len, args.batch_size
            )
            val_loader = get_dataloader(
                data_path, "val", args.seq_len, pred_len, args.batch_size
            )
            test_loader = get_dataloader(
                data_path, "test", args.seq_len, pred_len, args.batch_size
            )

            for epoch_budget in args.epochs_list:
                for model_index, model_name in enumerate(("FAN", "DLinear")):
                    if is_already_run(dataset_name, model_name, pred_len, epoch_budget) and not args.force:
                        print(f"Skipping {dataset_name} {model_name} H={pred_len} e={epoch_budget} (already done).")
                        continue

                    seed_val = args.seed + pred_len + model_index
                    set_seed(seed_val)

                    if model_name == "FAN":
                        model = FANIntegratedModel(
                            args.seq_len, pred_len, 7, freq_topk=args.freq_topk
                        )
                    else:
                        model = DLinear(args.seq_len, pred_len, 7)

                    print(f"\n--- {dataset_name} | {model_name} | H={pred_len} | Epochs={epoch_budget} (seed={seed_val}) ---")
                    t0 = time.perf_counter()
                    model = train_model(
                        model,
                        train_loader,
                        val_loader,
                        num_epochs=epoch_budget,
                        learning_rate=args.learning_rate,
                        device=device,
                        max_batches=args.max_batches,
                    )
                    test_mse, test_mae = evaluate_model(
                        model, test_loader, device=device, max_batches=args.max_batches
                    )
                    elapsed = time.perf_counter() - t0

                    history = getattr(model, "training_history", [])
                    best_entry = min(history, key=lambda h: h["val_mse"])
                    best_epoch = best_entry["epoch"]
                    best_val_mse = best_entry["val_mse"]
                    final_val_mse = history[-1]["val_mse"]
                    final_train_mse = history[-1]["train_mse"]

                    checkpoint_path = os.path.join(
                        args.model_dir,
                        f"{dataset_name}_{model_name}_{pred_len}_e{epoch_budget}.pt"
                    )
                    torch.save(model.state_dict(), checkpoint_path)

                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seq_len": args.seq_len,
                        "pred_len": pred_len,
                        "epoch_budget": epoch_budget,
                        "best_val_epoch": best_epoch,
                        "best_val_mse": round(best_val_mse, 8),
                        "final_val_mse": round(final_val_mse, 8),
                        "final_train_mse": round(final_train_mse, 8),
                        "test_mse": round(test_mse, 8),
                        "test_mae": round(test_mae, 8),
                        "training_seconds": round(elapsed, 3),
                        "seed": seed_val,
                        "checkpoint": checkpoint_path.replace("\\", "/"),
                        "training_history": history,
                    }
                    rows.append(row)

                    save_results(rows, config, args.output_dir)

                    print(
                        f"-> Done {dataset_name} {model_name} H={pred_len} e={epoch_budget}: "
                        f"Best Epoch={best_epoch}/{epoch_budget}, "
                        f"Val MSE={best_val_mse:.4f}, Test MSE={test_mse:.4f}, Test MAE={test_mae:.4f} "
                        f"in {elapsed:.2f}s"
                    )

    total_time = time.perf_counter() - total_start
    print(f"\n========================================================")
    print(f"Extension completed: {len(rows)} model training runs in {total_time/60:.2f} minutes.")
    print(f"Results saved to: {args.output_dir}/epoch_budget_results.csv")
    print(f"Checkpoints saved to: {args.model_dir}/")
    print(f"========================================================")


if __name__ == "__main__":
    main()
