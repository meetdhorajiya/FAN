"""Analyze experimental extension results on epoch budgets (3, 10, 20 epochs).
Generates publication-quality figures, comparative statistical summaries,
and LaTeX table snippets.
"""

import csv
import json
import os
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "outputs/extension_results"
FIGURES_DIR = "outputs/figures"
REPORT_FIG_DIR = "report/figures"
REPORT_EXT_FIG_DIR = "report/figures_extension"


def load_results():
    csv_path = os.path.join(OUTPUT_DIR, "epoch_budget_results.csv")
    json_path = os.path.join(OUTPUT_DIR, "epoch_budget_results.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            rows.append({
                "dataset": r["dataset"],
                "model": r["model"],
                "seq_len": int(r["seq_len"]),
                "pred_len": int(r["pred_len"]),
                "epoch_budget": int(r["epoch_budget"]),
                "best_val_epoch": int(r["best_val_epoch"]),
                "best_val_mse": float(r["best_val_mse"]),
                "final_val_mse": float(r["final_val_mse"]),
                "final_train_mse": float(r["final_train_mse"]),
                "test_mse": float(r["test_mse"]),
                "test_mae": float(r["test_mae"]),
                "training_seconds": float(r["training_seconds"]),
                "seed": int(r["seed"]),
                "checkpoint": r["checkpoint"],
            })

    json_data = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            json_data = json.load(handle)

    return rows, json_data


def generate_plot_etth1_horizons(rows):
    """Plot ETTh1 Test MSE across horizons for epochs 3, 10, 20."""
    subset = [r for r in rows if r["dataset"] == "ETTh1"]
    if not subset:
        return

    horizons = sorted(list(set(r["pred_len"] for r in subset)))
    budgets = sorted(list(set(r["epoch_budget"] for r in subset)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    models = ["FAN", "DLinear"]
    colors = {3: "#1f77b4", 10: "#ff7f0e", 20: "#2ca02c"}
    markers = {3: "o", 10: "s", 20: "^"}

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        for b in budgets:
            pts = [r for r in subset if r["model"] == model_name and r["epoch_budget"] == b]
            pts_dict = {r["pred_len"]: r["test_mse"] for r in pts}
            vals = [pts_dict.get(h, np.nan) for h in horizons]
            ax.plot(
                horizons, vals,
                label=f"{b} Epochs",
                color=colors.get(b, "black"),
                marker=markers.get(b, "o"),
                linewidth=1.8,
                markersize=6,
            )
        ax.set_title(f"ETTh1: {model_name} Test MSE vs Horizon", fontsize=12, fontweight="bold")
        ax.set_xlabel("Prediction Horizon (H)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Test MSE", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "extension_etth1_horizon_mse.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_plot_h24_comparison(rows):
    """Compare H=24 Test MSE across datasets for 3, 10, 20 epochs."""
    subset = [r for r in rows if r["pred_len"] == 24]
    if not subset:
        return

    datasets = sorted(list(set(r["dataset"] for r in subset)))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.5), squeeze=False)

    colors = {"FAN": "#d62728", "DLinear": "#1f77b4"}
    markers = {"FAN": "o", "DLinear": "s"}

    for idx, ds in enumerate(datasets):
        ax = axes[0][idx]
        for m in ["FAN", "DLinear"]:
            pts = [r for r in subset if r["dataset"] == ds and r["model"] == m]
            pts.sort(key=lambda x: x["epoch_budget"])
            budgets = [r["epoch_budget"] for r in pts]
            mses = [r["test_mse"] for r in pts]
            ax.plot(
                budgets, mses,
                label=m,
                color=colors[m],
                marker=markers[m],
                linewidth=2.0,
                markersize=7,
            )
        ax.set_title(f"{ds} (H=24)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Epochs", fontsize=11)
        ax.set_xticks([3, 10, 20])
        if idx == 0:
            ax.set_ylabel("Test MSE", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)

    fig.suptitle("Test MSE Trajectory across Training Budgets (Horizon 24)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "extension_epoch_comparison_h24.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_plot_learning_curves(json_data):
    """Plot epoch-by-epoch loss trajectories for 20-epoch models to observe convergence/overfitting."""
    results = json_data.get("results", [])
    twenty_ep = [r for r in results if r.get("epoch_budget") == 20 and "training_history" in r]
    if not twenty_ep:
        import torch
        from src.dataset import get_dataloader
        from src.forecasting_model import FANIntegratedModel
        from src.baseline_models import DLinear
        from src.trainer import train_model
        from scripts.run_epoch_extension import set_seed
        print("Generating 20-epoch loss histories on ETTh1 (H=24) for learning curves plot...")
        train_loader = get_dataloader("data/ETTh1.csv", "train", 96, 24, 32)
        val_loader = get_dataloader("data/ETTh1.csv", "val", 96, 24, 32)
        candidates = []
        for model_idx, (m_name, m_cls) in enumerate([
            ("FAN", lambda: FANIntegratedModel(96, 24, 7, freq_topk=5)),
            ("DLinear", lambda: DLinear(96, 24, 7)),
        ]):
            set_seed(42 + 24 + model_idx)
            m = m_cls()
            m = train_model(m, train_loader, val_loader, num_epochs=20)
            hist = m.training_history
            best_e = min(hist, key=lambda h: h["val_mse"])
            candidates.append({
                "dataset": "ETTh1",
                "model": m_name,
                "pred_len": 24,
                "epoch_budget": 20,
                "best_val_epoch": best_e["epoch"],
                "best_val_mse": best_e["val_mse"],
                "training_history": hist,
            })
    else:
        # Focus on ETTh1 H=24 and ETTh2 H=24 (or first available)
        candidates = [r for r in twenty_ep if r.get("pred_len") == 24 and r.get("dataset") in ("ETTh1", "ETTh2")]
        if not candidates:
            candidates = twenty_ep[:2]

    fig, axes = plt.subplots(1, len(candidates), figsize=(5.5 * len(candidates), 4.2), squeeze=False)

    for idx, item in enumerate(candidates):
        ax = axes[0][idx]
        history = item.get("training_history", [])
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_mse"] for h in history]
        val_loss = [h["val_mse"] for h in history]

        ax.plot(epochs, train_loss, label="Train MSE", color="#1f77b4", marker=".", linewidth=1.8)
        ax.plot(epochs, val_loss, label="Val MSE", color="#d62728", marker="x", linewidth=1.8)

        best_ep = item["best_val_epoch"]
        best_val = item["best_val_mse"]
        ax.axvline(best_ep, color="green", linestyle=":", label=f"Best Val (Epoch {best_ep})")
        ax.scatter([best_ep], [best_val], color="green", s=60, zorder=5)

        ax.set_title(f"{item['dataset']} {item['model']} (H={item['pred_len']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss (MSE)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Training vs Validation MSE Learning Curves (20-Epoch Runs)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "extension_learning_curves.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_plot_best_epoch_hist(rows):
    """Plot distribution of best validation epochs across all 10- and 20-epoch runs."""
    ten_and_twenty = [r for r in rows if r["epoch_budget"] in (10, 20)]
    if not ten_and_twenty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for idx, budget in enumerate([10, 20]):
        ax = axes[idx]
        subset = [r for r in ten_and_twenty if r["epoch_budget"] == budget]
        fan_epochs = [r["best_val_epoch"] for r in subset if r["model"] == "FAN"]
        dl_epochs = [r["best_val_epoch"] for r in subset if r["model"] == "DLinear"]

        bins = np.arange(1, budget + 2) - 0.5
        ax.hist(
            [fan_epochs, dl_epochs],
            bins=bins,
            label=["FAN", "DLinear"],
            color=["#d62728", "#1f77b4"],
            alpha=0.8,
            rwidth=0.8,
        )
        ax.set_title(f"Best Val Epoch Distribution (Budget = {budget})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch Reached Best Validation MSE", fontsize=10)
        ax.set_ylabel("Number of Runs", fontsize=10)
        ax.set_xticks(range(1, budget + 1))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "extension_best_epoch_distribution.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def copy_figures_to_report():
    os.makedirs(REPORT_FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_EXT_FIG_DIR, exist_ok=True)
    for fname in os.listdir(FIGURES_DIR):
        if fname.startswith("extension_") and fname.endswith(".png"):
            src = os.path.join(FIGURES_DIR, fname)
            shutil.copy2(src, os.path.join(REPORT_FIG_DIR, fname))
            shutil.copy2(src, os.path.join(REPORT_EXT_FIG_DIR, fname))
            print(f"Copied figure to report directories: {fname}")


def generate_latex_tables(rows):
    """Generate clean LaTeX table fragments for report inclusion."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    table_path = os.path.join(OUTPUT_DIR, "latex_tables.tex")

    with open(table_path, "w", encoding="utf-8") as handle:
        # Table 1: Horizon 24 cross-dataset comparison
        handle.write("% Table: Cross-Dataset Comparison at Horizon 24 across Epoch Budgets\n")
        handle.write("\\begin{table*}[t]\n")
        handle.write("\\caption{Test MSE and MAE at Horizon 24 across Training Budgets (3, 10, and 20 Epochs)}\n")
        handle.write("\\label{tab:extension_h24}\n")
        handle.write("\\centering\n\\footnotesize\n")
        handle.write("\\begin{tabular}{@{}llrrrrrr@{}}\n")
        handle.write("\\toprule\n")
        handle.write("Dataset & Model & 3-Ep MSE & 3-Ep MAE & 10-Ep MSE & 10-Ep MAE & 20-Ep MSE & 20-Ep MAE\\\\\n")
        handle.write("\\midrule\n")

        h24_rows = [r for r in rows if r["pred_len"] == 24]
        datasets = sorted(list(set(r["dataset"] for r in h24_rows)))

        for ds in datasets:
            for m in ["FAN", "DLinear"]:
                e3 = next((r for r in h24_rows if r["dataset"] == ds and r["model"] == m and r["epoch_budget"] == 3), None)
                e10 = next((r for r in h24_rows if r["dataset"] == ds and r["model"] == m and r["epoch_budget"] == 10), None)
                e20 = next((r for r in h24_rows if r["dataset"] == ds and r["model"] == m and r["epoch_budget"] == 20), None)

                e3_mse = f"{e3['test_mse']:.4f}" if e3 else "--"
                e3_mae = f"{e3['test_mae']:.4f}" if e3 else "--"
                e10_mse = f"{e10['test_mse']:.4f}" if e10 else "--"
                e10_mae = f"{e10['test_mae']:.4f}" if e10 else "--"
                e20_mse = f"{e20['test_mse']:.4f}" if e20 else "--"
                e20_mae = f"{e20['test_mae']:.4f}" if e20 else "--"

                handle.write(f"{ds} & {m} & {e3_mse} & {e3_mae} & {e10_mse} & {e10_mae} & {e20_mse} & {e20_mae}\\\\\n")
            handle.write("\\midrule\n")

        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

        # Table 2: ETTh1 complete horizon comparison
        handle.write("% Table: ETTh1 Full Horizon Comparison across Epoch Budgets\n")
        handle.write("\\begin{table*}[t]\n")
        handle.write("\\caption{ETTh1 Performance across Horizons ($H$) and Epoch Budgets}\n")
        handle.write("\\label{tab:extension_etth1}\n")
        handle.write("\\centering\n\\scriptsize\n")
        handle.write("\\begin{tabular}{@{}r|rrr|rrr|rrr|rrr@{}}\n")
        handle.write("\\toprule\n")
        handle.write(" & \\multicolumn{3}{c|}{FAN MSE} & \\multicolumn{3}{c|}{FAN MAE} & \\multicolumn{3}{c|}{DLinear MSE} & \\multicolumn{3}{c}{DLinear MAE}\\\\\n")
        handle.write("$H$ & 3 Ep & 10 Ep & 20 Ep & 3 Ep & 10 Ep & 20 Ep & 3 Ep & 10 Ep & 20 Ep & 3 Ep & 10 Ep & 20 Ep\\\\\n")
        handle.write("\\midrule\n")

        etth1_rows = [r for r in rows if r["dataset"] == "ETTh1"]
        horizons = sorted(list(set(r["pred_len"] for r in etth1_rows)))

        for h in horizons:
            f3 = next((r for r in etth1_rows if r["model"] == "FAN" and r["pred_len"] == h and r["epoch_budget"] == 3), None)
            f10 = next((r for r in etth1_rows if r["model"] == "FAN" and r["pred_len"] == h and r["epoch_budget"] == 10), None)
            f20 = next((r for r in etth1_rows if r["model"] == "FAN" and r["pred_len"] == h and r["epoch_budget"] == 20), None)

            d3 = next((r for r in etth1_rows if r["model"] == "DLinear" and r["pred_len"] == h and r["epoch_budget"] == 3), None)
            d10 = next((r for r in etth1_rows if r["model"] == "DLinear" and r["pred_len"] == h and r["epoch_budget"] == 10), None)
            d20 = next((r for r in etth1_rows if r["model"] == "DLinear" and r["pred_len"] == h and r["epoch_budget"] == 20), None)

            f3_mse = f"{f3['test_mse']:.4f}" if f3 else "--"
            f10_mse = f"{f10['test_mse']:.4f}" if f10 else "--"
            f20_mse = f"{f20['test_mse']:.4f}" if f20 else "--"

            f3_mae = f"{f3['test_mae']:.4f}" if f3 else "--"
            f10_mae = f"{f10['test_mae']:.4f}" if f10 else "--"
            f20_mae = f"{f20['test_mae']:.4f}" if f20 else "--"

            d3_mse = f"{d3['test_mse']:.4f}" if d3 else "--"
            d10_mse = f"{d10['test_mse']:.4f}" if d10 else "--"
            d20_mse = f"{d20['test_mse']:.4f}" if d20 else "--"

            d3_mae = f"{d3['test_mae']:.4f}" if d3 else "--"
            d10_mae = f"{d10['test_mae']:.4f}" if d10 else "--"
            d20_mae = f"{d20['test_mae']:.4f}" if d20 else "--"

            handle.write(
                f"{h} & {f3_mse} & {f10_mse} & {f20_mse} & {f3_mae} & {f10_mae} & {f20_mae} & "
                f"{d3_mse} & {d10_mse} & {d20_mse} & {d3_mae} & {d10_mae} & {d20_mae}\\\\\n"
            )

        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    print(f"Saved LaTeX tables to: {table_path}")


def main():
    rows, json_data = load_results()
    print(f"Loaded {len(rows)} records from extension results.")
    generate_plot_etth1_horizons(rows)
    generate_plot_h24_comparison(rows)
    generate_plot_learning_curves(json_data)
    generate_plot_best_epoch_hist(rows)
    copy_figures_to_report()
    generate_latex_tables(rows)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
