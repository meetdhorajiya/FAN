import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
from src.dataset import get_dataloader
from src.forecasting_model import FANIntegratedModel
from src.trainer import train_model, evaluate_model

def main():
    print("==================================================")
    print(" FAN Project - Phase 6 Demonstration")
    print(" Experiments & Analysis (Forecast Horizons & Ablation)")
    print("==================================================\n")
    
    data_path = "data/ETTh1.csv"
    seq_len = 96
    enc_in = 7
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n--- Experiment 1: Forecast Horizon Comparison (24 vs 48) ---")
    for pred_len in [24, 48]:
        print(f"\nEvaluating Horizon: {pred_len}")
        train_loader = get_dataloader(data_path, flag='train', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
        val_loader = get_dataloader(data_path, flag='val', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
        test_loader = get_dataloader(data_path, flag='test', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
        
        model = FANIntegratedModel(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        model = train_model(model, train_loader, val_loader, num_epochs=1, device=device)
        evaluate_model(model, test_loader, device=device)
        
    print("\n--- Experiment 2: Ablation Study (Top-K Frequencies: 5 vs 10) ---")
    pred_len = 24
    train_loader = get_dataloader(data_path, flag='train', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    val_loader = get_dataloader(data_path, flag='val', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    test_loader = get_dataloader(data_path, flag='test', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
    
    for topk in [5, 10]:
        print(f"\nEvaluating FAN with Top-K={topk}")
        model = FANIntegratedModel(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, freq_topk=topk)
        model = train_model(model, train_loader, val_loader, num_epochs=1, device=device)
        evaluate_model(model, test_loader, device=device)
        
    print("\n--- Experiment 3: Dataset Comparison ---")
    pred_len = 24
    datasets = ["data/ETTh1.csv", "data/ETTh2.csv", "data/ETTm1.csv", "data/ETTm2.csv"]
    for d_path in datasets:
        if not os.path.exists(d_path):
            print(f"\nDataset {d_path} not found. Skipping...")
            continue
            
        print(f"\nEvaluating Dataset: {os.path.basename(d_path)}")
        train_loader = get_dataloader(d_path, flag='train', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
        val_loader = get_dataloader(d_path, flag='val', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
        test_loader = get_dataloader(d_path, flag='test', seq_len=seq_len, pred_len=pred_len, batch_size=batch_size)
        
        model = FANIntegratedModel(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        model = train_model(model, train_loader, val_loader, num_epochs=1, device=device)
        evaluate_model(model, test_loader, device=device)

    print("\n==================================================")
    print(" Phase 6 Execution Completed Successfully!")
    print(" All experiments generated comparative results.")
    print("==================================================")

if __name__ == "__main__":
    main()
