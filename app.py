"""
app.py — FAN Project Gradio UI
Loads pre-trained model weights and visualises forecasts on all 4 ETT datasets.
Run: python app.py
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

from src.dataset import ETTDataset
from src.forecasting_model import FANIntegratedModel
from src.baseline_models import DLinear

# ── Config ──────────────────────────────────────────────────────────────────
SEQ_LEN  = 96
PRED_LEN = 24
ENC_IN   = 7
MODEL_DIR = "outputs/models"

DATASETS = {
    "ETTh1": "data/ETTh1.csv",
    "ETTh2": "data/ETTh2.csv",
    "ETTm1": "data/ETTm1.csv",
    "ETTm2": "data/ETTm2.csv",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _build_model(model_name: str, pred_len: int):
    if model_name == "FAN-Integrated":
        return FANIntegratedModel(SEQ_LEN, pred_len, ENC_IN, freq_topk=5)
    return DLinear(SEQ_LEN, pred_len, ENC_IN)

def _weight_key(dataset: str, model_name: str) -> str:
    short = "FAN" if model_name == "FAN-Integrated" else "DLinear"
    return f"{dataset}_{short}"

def _load_model(dataset: str, model_name: str, pred_len: int):
    model = _build_model(model_name, pred_len)
    short = "FAN" if model_name == "FAN-Integrated" else "DLinear"
    weight_file = os.path.join(MODEL_DIR, f"{dataset}_{short}_{pred_len}.pt")
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location="cpu", weights_only=True))
        trained = True
    else:
        trained = False
    model.eval()
    return model, trained

# ── Main forecast function ────────────────────────────────────────────────────
def load_and_forecast(model_choice: str, dataset_choice: str, pred_len: int):
    pred_len = int(pred_len)
    data_path = DATASETS[dataset_choice]

    if not os.path.exists(data_path):
        raise gr.Error(f"Dataset file '{data_path}' not found. Please run 'python scripts/train_all.py' first.")

    # Load test split
    dataset = ETTDataset(data_path, flag='test', seq_len=SEQ_LEN, pred_len=pred_len)
    x, y_true = dataset[0]

    # Load model (with weights if available)
    model, trained = _load_model(dataset_choice, model_choice, pred_len)

    with torch.no_grad():
        y_pred = model(x.unsqueeze(0)).squeeze(0)

    # ── Plot (target feature: OT, index 6) ───────────────────────────────────
    target_idx = 6
    history    = x[:, target_idx].numpy()
    true_fut   = y_true[:, target_idx].numpy()
    pred_fut   = y_pred[:, target_idx].numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    t_hist = np.arange(SEQ_LEN)
    t_fut  = np.arange(SEQ_LEN, SEQ_LEN + pred_len)

    ax.plot(t_hist, history,  label="Historical (OT)",   color="#1f77b4", linewidth=1.5)
    ax.plot(t_fut,  true_fut, label="Ground Truth",      color="#2ca02c", linestyle="--", linewidth=1.5)
    ax.plot(t_fut,  pred_fut, label=f"{model_choice} Prediction", color="#d62728", linewidth=1.8)

    weight_note = "✅ Pre-trained weights loaded" if trained else "⚠️ Random weights (run train_all.py to train)"
    ax.set_title(f"[{dataset_choice}] {model_choice} — Horizon: {pred_len}   |   {weight_note}", fontsize=12)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalised Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ── Metrics ───────────────────────────────────────────────────────────────
    mse = float(np.mean((pred_fut - true_fut) ** 2))
    mae = float(np.mean(np.abs(pred_fut - true_fut)))
    metrics_text = f"**MSE:** {mse:.4f}   |   **MAE:** {mae:.4f}"
    if not trained:
        metrics_text += "\n\n> ⚠️ Weights not found. Run `python scripts/train_all.py` to train and save models."

    return fig, metrics_text

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="FAN Forecasting Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌊 FAN Project — Frequency Adaptive Normalization")
    gr.Markdown(
        "**NeurIPS 2024** | Select a dataset, model, and forecast horizon "
        "to visualise predictions on the held-out test set."
    )

    with gr.Row():
        dataset_dd   = gr.Dropdown(choices=list(DATASETS.keys()), value="ETTh1",
                                   label="Dataset")
        model_dd     = gr.Dropdown(choices=["FAN-Integrated", "DLinear Baseline"],
                                   value="FAN-Integrated", label="Model")
        pred_slider  = gr.Slider(minimum=12, maximum=96, step=12, value=24,
                                 label="Prediction Horizon (steps)")

    run_btn     = gr.Button("▶ Run Forecast", variant="primary")
    output_plot = gr.Plot(label="Forecast Visualisation")
    metrics_md  = gr.Markdown()

    run_btn.click(
        fn=load_and_forecast,
        inputs=[model_dd, dataset_dd, pred_slider],
        outputs=[output_plot, metrics_md],
    )

if __name__ == "__main__":
    print("Launching FAN Forecasting UI → http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
