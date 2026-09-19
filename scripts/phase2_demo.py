import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.fan_core import FAN

def main():
    print("==================================================")
    print(" FAN Project - Phase 2 Demonstration")
    print(" Core FAN Implementation Testing")
    print("==================================================\n")
    
    # Hyperparameters
    batch_size = 32
    seq_len = 96
    pred_len = 24
    enc_in = 7 # Number of features (e.g., in ETTh1)
    freq_topk = 5
    
    # Initialize the FAN module
    print(f"Initializing FAN module with seq_len={seq_len}, pred_len={pred_len}, enc_in={enc_in}, top-k={freq_topk}")
    fan_module = FAN(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, freq_topk=freq_topk)
    
    # Create dummy input data simulating historical time series
    print("\nCreating dummy input data (Batch Size, Sequence Length, Features)...")
    dummy_input = torch.randn(batch_size, seq_len, enc_in)
    print(f"Input shape: {dummy_input.shape}")
    
    # Test normalization (extracting residual and non-stationary components)
    print("\nTesting Frequency-Based Normalization...")
    normalized_output = fan_module(dummy_input, mode='n')
    print(f"Normalized (Residual) Output shape: {normalized_output.shape}")
    print(f"Predicted Non-Stationary Pattern shape (internal): {fan_module.pred_main_freq_signal.shape}")
    
    # Create dummy prediction data from a forecasting backbone
    print("\nSimulating Forecasting Backbone prediction...")
    # The backbone will predict the future of the residual component
    dummy_backbone_pred = torch.randn(batch_size, pred_len, enc_in)
    print(f"Backbone Prediction shape: {dummy_backbone_pred.shape}")
    
    # Test denormalization (combining backbone prediction with FAN's non-stationary prediction)
    print("\nTesting Adaptive Denormalization (Combining Predictions)...")
    final_output = fan_module(dummy_backbone_pred, mode='d')
    print(f"Final Prediction Output shape: {final_output.shape}")
    
    print("\n==================================================")
    print(" Phase 2 Execution Completed Successfully!")
    print(" FAN Core components correctly process data tensors.")
    print("==================================================")

if __name__ == "__main__":
    main()
