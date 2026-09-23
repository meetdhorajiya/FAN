"""Evaluate saved checkpoints on complete test splits and generate report figures."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_models import DLinear
from src.dataset import ETTDataset, get_dataloader
from src.forecasting_model import FANIntegratedModel
from src.trainer import evaluate_model


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--freq-topk", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def build_model(model_name, seq_len, pred_len, freq_topk):
    if model_name == "FAN":
        return FANIntegratedModel(seq_len, pred_len, 7, freq_topk=freq_topk)
    return DLinear(seq_len, pred_len, 7)


def save_table(rows):
    os.makedirs("outputs/results", exist_ok=True)
    csv_path = "outputs/results/evaluation_results.csv"
    json_path = "outputs/results/evaluation_results.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "model", "pred_len", "mse", "mae", "samples"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "results": rows},
            handle,
            indent=2,
        )


def plot_metric(rows, metric):
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    horizons = sorted({row["pred_len"] for row in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for model_name, marker in [("FAN", "o"), ("DLinear", "s")]:
            subset = [
                row for row in rows
                if row["dataset"] == dataset and row["model"] == model_name
            ]
            values = {row["pred_len"]: row[metric] for row in subset}
            axis.plot(
                horizons,
                [values.get(horizon, np.nan) for horizon in horizons],
                marker=marker,
                label=model_name,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel(metric.upper())
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(f"Full test-set {metric.upper()} comparison")
    fig.tight_layout()
    path = f"outputs/figures/model_comparison_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_forecast(dataset_name, data_path, models, seq_len, pred_len):
    dataset = ETTDataset(data_path, "test", seq_len, pred_len)
    x, y_true = dataset[0]
    target_index = 6
    history = x[:, target_index].numpy()
    truth = y_true[:, target_index].numpy()

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(np.arange(seq_len), history, label="History", color="#1f77b4")
    future_axis = np.arange(seq_len, seq_len + pred_len)
    axis.plot(future_axis, truth, label="Ground truth", color="#2ca02c", linestyle="--")
    colors = {"FAN": "#d62728", "DLinear": "#9467bd"}
    with torch.no_grad():
        for model_name, model in models.items():
            prediction = model(x.unsqueeze(0)).squeeze(0)[:, target_index].numpy()
            axis.plot(future_axis, prediction, label=model_name, color=colors[model_name])
    axis.set_title(f"{dataset_name} forecast comparison (horizon={pred_len})")
    axis.set_xlabel("Time step")
    axis.set_ylabel("Normalized OT")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = f"outputs/figures/forecast_{dataset_name}_{pred_len}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)

    os.makedirs("outputs/figures", exist_ok=True)
    rows = []
    primary_horizon = 24 if 24 in args.pred_lens else args.pred_lens[0]

    for dataset_name in args.datasets:
        data_path = DATASETS[dataset_name]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Required dataset not found: {data_path}")

        primary_models = {}
        for pred_len in args.pred_lens:
            test_loader = get_dataloader(
                data_path, "test", args.seq_len, pred_len, args.batch_size
            )
            for model_name in ("FAN", "DLinear"):
                checkpoint = f"outputs/models/{dataset_name}_{model_name}_{pred_len}.pt"
                if not os.path.exists(checkpoint):
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
                model = build_model(model_name, args.seq_len, pred_len, args.freq_topk)
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.to(device)
                mse, mae = evaluate_model(model, test_loader, device=device)
                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "pred_len": pred_len,
                    "mse": round(mse, 8),
                    "mae": round(mae, 8),
                    "samples": len(test_loader.dataset),
                })
                if pred_len == primary_horizon:
                    primary_models[model_name] = model.cpu()
                print(f"{dataset_name} {model_name} h={pred_len}: MSE={mse:.4f}, MAE={mae:.4f}")

        if len(primary_models) == 2:
            plot_forecast(
                dataset_name, data_path, primary_models, args.seq_len, primary_horizon
            )

    save_table(rows)
    plot_metric(rows, "mse")
    plot_metric(rows, "mae")
    print("Saved complete metrics to outputs/results/evaluation_results.csv")


if __name__ == "__main__":
    main()
