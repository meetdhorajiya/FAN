import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.forecasting_model import FANIntegratedModel, SimpleLinearBackbone

def main():
    print("==================================================")
    print(" FAN Project - Phase 3 Demonstration")
    print(" Forecasting Model Integration")
    print("==================================================\n")
    
    # Hyperparameters
    batch_size = 16
    seq_len = 96
    pred_len = 24
    enc_in = 7 # Number of features (e.g., in ETTh1)
    
    # Dummy input
    dummy_input = torch.randn(batch_size, seq_len, enc_in)
    print(f"1. Input Data Shape (Historical): {dummy_input.shape}")
    
    # Initialize the Integrated Model
    print("\n2. Initializing FAN Integrated Model (FAN + Linear Backbone)...")
    model = FANIntegratedModel(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, freq_topk=5)
    
    # Forward Pass
    print("\n3. Performing End-to-End Forward Pass...")
    final_prediction = model(dummy_input)
    
    print(f"\n4. Final Output Data Shape (Prediction): {final_prediction.shape}")
    print("   -> Successfully transformed (Batch, Seq_Len, Features) to (Batch, Pred_Len, Features)!")
    
    print("\n==================================================")
    print(" Phase 3 Execution Completed Successfully!")
    print(" The FAN module is now fully integrated with a backbone.")
    print("==================================================")

if __name__ == "__main__":
    main()
