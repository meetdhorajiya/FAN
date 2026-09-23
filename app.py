"""
app.py - FAN Project Gradio UI
Loads pre-trained model weights and visualises forecasts on all 4 ETT datasets.
Run: python app.py
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from pathlib import Path

from src.dataset import ETTDataset
from src.forecasting_model import FANIntegratedModel
from src.baseline_models import DLinear

# ── Config ──────────────────────────────────────────────────────────────────
SEQ_LEN  = 96
PRED_LEN = 24
ENC_IN   = 7
MODEL_DIR = "outputs/models"
EXPECTED_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

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
def load_and_forecast(
    model_choice: str,
    dataset_choice: str,
    pred_len: int,
    uploaded_file=None,
):
    pred_len = int(pred_len)
    if uploaded_file:
        data_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
        source_label = f"Uploaded: {Path(data_path).name}"
    else:
        data_path = DATASETS[dataset_choice]
        source_label = dataset_choice

    if not os.path.exists(data_path):
        raise gr.Error(f"Dataset file '{data_path}' was not found.")

    # Load test split
    try:
        dataset = ETTDataset(data_path, flag='test', seq_len=SEQ_LEN, pred_len=pred_len)
    except (ValueError, OSError) as exc:
        raise gr.Error(f"Could not load the CSV: {exc}") from exc
    if len(dataset) == 0:
        raise gr.Error("The CSV does not contain enough rows for this forecast horizon.")
    x, y_true = dataset[0]
    if x.shape[1] != ENC_IN:
        raise gr.Error(
            f"Expected {ENC_IN} numeric feature columns, but the CSV contains {x.shape[1]}."
        )
    if dataset.feature_names != EXPECTED_FEATURES:
        raise gr.Error(
            "Expected ETT columns in this order: " + ", ".join(EXPECTED_FEATURES)
        )

    # Load model (with weights if available)
    model, trained = _load_model(dataset_choice, model_choice, pred_len)

    with torch.no_grad():
        y_pred = model(x.unsqueeze(0)).squeeze(0)

    # ── Plot (target feature: OT, index 6) ───────────────────────────────────
    target_idx = dataset.feature_names.index("OT")
    history    = x[:, target_idx].numpy()
    true_fut   = y_true[:, target_idx].numpy()
    pred_fut   = y_pred[:, target_idx].numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    t_hist = np.arange(SEQ_LEN)
    t_fut  = np.arange(SEQ_LEN, SEQ_LEN + pred_len)

    ax.plot(t_hist, history,  label="Historical (OT)",   color="#1f77b4", linewidth=1.5)
    ax.plot(t_fut,  true_fut, label="Ground Truth",      color="#2ca02c", linestyle="--", linewidth=1.5)
    ax.plot(t_fut,  pred_fut, label=f"{model_choice} Prediction", color="#d62728", linewidth=1.8)

    weight_note = "Pre-trained weights loaded" if trained else "Random weights (run train_all.py to train)"
    ax.set_title(f"{source_label} | {model_choice} | Horizon: {pred_len} | {weight_note}", fontsize=12)
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
        metrics_text += "\n\nWeights not found. Run `python scripts/train_all.py` to train and save models."
    elif uploaded_file:
        metrics_text += f"\n\nUploaded CSV evaluated with **{dataset_choice}** pre-trained weights."

    return fig, metrics_text

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="FAN Forecasting Demo") as demo:
    gr.Markdown("# FAN Time-Series Forecasting")
    gr.Markdown(
        "Select a dataset, model, and forecast horizon to view predictions."
    )

    with gr.Row():
        dataset_dd   = gr.Dropdown(choices=list(DATASETS.keys()), value="ETTh1",
                                   label="Dataset")
        model_dd     = gr.Dropdown(choices=["FAN-Integrated", "DLinear Baseline"],
                                   value="FAN-Integrated", label="Model")
        pred_slider  = gr.Slider(minimum=12, maximum=96, step=12, value=24,
                                 label="Prediction Horizon (steps)")

    upload_csv = gr.File(
        label="Optional ETT-format CSV upload",
        file_types=[".csv"],
        type="filepath",
    )

    run_btn     = gr.Button("Run Forecast", variant="primary")
    output_plot = gr.Plot(label="Forecast Visualisation")
    metrics_md  = gr.Markdown()

    run_btn.click(
        fn=load_and_forecast,
        inputs=[model_dd, dataset_dd, pred_slider, upload_csv],
        outputs=[output_plot, metrics_md],
    )

if __name__ == "__main__":
    print("Launching FAN Forecasting UI: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
