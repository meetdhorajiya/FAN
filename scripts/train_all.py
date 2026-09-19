"""
train_all.py — Train FAN and DLinear on all 4 ETT datasets and save weights.
Run from project root: python scripts/train_all.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import get_dataloader
from src.forecasting_model import FANIntegratedModel
from src.baseline_models import DLinear
from src.trainer import train_model, evaluate_model

DATASETS = {
    "ETTh1": "data/ETTh1.csv",
    "ETTh2": "data/ETTh2.csv",
    "ETTm1": "data/ETTm1.csv",
    "ETTm2": "data/ETTm2.csv",
}

SEQ_LEN   = 96
PRED_LENS = [12, 24, 36, 48, 60, 72, 84, 96]   # all slider values in app.py
ENC_IN    = 7
BATCH     = 32
EPOCHS    = 3           # increase for better accuracy; 3 is fast & realistic
SAVE_DIR  = "outputs/models"

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

results = {}

for ds_name, ds_path in DATASETS.items():
    if not os.path.exists(ds_path):
        print(f"[SKIP] {ds_path} not found.\n")
        continue

    print(f"{'='*55}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*55}")

    for pred_len in PRED_LENS:
        print(f"\n  -- pred_len = {pred_len} --")

        train_loader = get_dataloader(ds_path, 'train', SEQ_LEN, pred_len, BATCH)
        val_loader   = get_dataloader(ds_path, 'val',   SEQ_LEN, pred_len, BATCH)
        test_loader  = get_dataloader(ds_path, 'test',  SEQ_LEN, pred_len, BATCH)

        for model_name, model in [
            ("FAN",     FANIntegratedModel(SEQ_LEN, pred_len, ENC_IN, freq_topk=5)),
            ("DLinear", DLinear(SEQ_LEN, pred_len, ENC_IN)),
        ]:
            print(f"\n--- Training {model_name} on {ds_name} (pred_len={pred_len}) ---")
            model = train_model(model, train_loader, val_loader,
                                num_epochs=EPOCHS, device=device)

            mse, mae = evaluate_model(model, test_loader, device=device)
            key = f"{ds_name}_{model_name}_{pred_len}"
            results[key] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}

            # Save weights — filename encodes dataset, model, and pred_len
            save_path = os.path.join(SAVE_DIR, f"{key}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"    Weights saved -> {save_path}")

print(f"\n{'='*55}")
print("  FINAL RESULTS SUMMARY")
print(f"{'='*55}")
print(f"{'Key':<30} {'MSE':>8} {'MAE':>8}")
print("-" * 48)
for k, v in results.items():
    print(f"{k:<30} {v['MSE']:>8.4f} {v['MAE']:>8.4f}")

print("\n[OK] All models trained and saved to outputs/models/")
