import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
from src.dataset import get_dataloader
from src.forecasting_model import FANIntegratedModel
from src.baseline_models import DLinear
from src.trainer import train_model, evaluate_model
from src.data_loader import download_etth1

def main():
    print("==================================================")
    print(" FAN Project - Phase 5 Demonstration")
    print(" Training & Evaluation")
    print("==================================================\n")
    
    # Check for dataset
    data_path = download_etth1("data")
    
    seq_len = 96
    pred_len = 24
    enc_in = 7
    batch_size = 32
    
    print("\n1. Loading Data...")
    train_loader = get_dataloader(data_path, flag='train', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    val_loader = get_dataloader(data_path, flag='val', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    test_loader = get_dataloader(data_path, flag='test', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n--- Training Baseline (DLinear) ---")
    baseline_model = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    # Train 1 epoch for demo
    trained_baseline = train_model(baseline_model, train_loader, val_loader, num_epochs=1, device=device)
    print("\nEvaluating Baseline...")
    evaluate_model(trained_baseline, test_loader, device=device)
    
    print("\n--- Training FAN-Integrated Model ---")
    fan_model = FANIntegratedModel(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    # Train 1 epoch for demo
    trained_fan = train_model(fan_model, train_loader, val_loader, num_epochs=1, device=device)
    print("\nEvaluating FAN Model...")
    evaluate_model(trained_fan, test_loader, device=device)
    
    print("\n==================================================")
    print(" Phase 5 Execution Completed Successfully!")
    print(" We can now quantitatively compare Baseline vs FAN.")
    print("==================================================")

if __name__ == "__main__":
    main()
