import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.baseline_models import DLinear

def main():
    print("==================================================")
    print(" FAN Project - Phase 4 Demonstration")
    print(" Baseline Models (DLinear)")
    print("==================================================\n")
    
    # Hyperparameters
    batch_size = 16
    seq_len = 96
    pred_len = 24
    enc_in = 7
    
    # Dummy input
    dummy_input = torch.randn(batch_size, seq_len, enc_in)
    print(f"1. Input Data Shape: {dummy_input.shape}")
    
    # Initialize the Baseline Model
    print("\n2. Initializing DLinear Baseline Model...")
    baseline_model = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    
    # Forward Pass
    print("\n3. Performing Forward Pass...")
    baseline_prediction = baseline_model(dummy_input)
    
    print(f"\n4. Output Data Shape: {baseline_prediction.shape}")
    print("   -> DLinear successfully decomposed trend/seasonal and generated prediction!")
    
    print("\n==================================================")
    print(" Phase 4 Execution Completed Successfully!")
    print(" Baseline model is ready for evaluation comparison.")
    print("==================================================")

if __name__ == "__main__":
    main()
