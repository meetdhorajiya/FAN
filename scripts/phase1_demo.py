import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import matplotlib.pyplot as plt
from src.data_loader import download_etth1, load_data, inspect_data
from src.preprocessing import preprocess_data
from src.frequency_analysis import perform_fft_analysis, top_k_frequency_reconstruction

def main():
    print("==================================================")
    print(" FAN Project - Phase 1 Demonstration")
    print("==================================================\n")
    
    # Ensure directories exist
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Download and load data
    file_path = download_etth1(data_dir="data")
    df = load_data(file_path)
    
    # 2. Inspect dataset
    inspect_data(df)
    
    # Generate plot of raw ETTh1 data (first 1000 points of OT)
    print("\nGenerating raw data plot...")
    plt.figure(figsize=(12, 4))
    plt.plot(df['OT'].values[:1000])
    plt.title("Raw ETTh1 Data (Target Variable 'OT' - First 1000 points)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("outputs/figures/raw_data.png")
    plt.close()
    
    # 3. Preprocess and Split
    train_scaled, val_scaled, test_scaled, scaler = preprocess_data(df)
    
    # 4. Frequency Analysis Demo
    # We take the 'OT' (target) feature from the scaled training data
    if 'OT' in train_scaled.columns:
        signal = train_scaled['OT'].values
        
        # FFT Analysis
        perform_fft_analysis(signal, seq_len=96)
        
        # Reconstruction Demo
        top_k_frequency_reconstruction(signal, seq_len=96, k=3)
    else:
        print("'OT' feature not found for frequency analysis.")
        
    print("\n==================================================")
    print(" Phase 1 Execution Completed Successfully!")
    print(" Outputs saved in 'outputs/figures/'")
    print("==================================================")

if __name__ == "__main__":
    main()
