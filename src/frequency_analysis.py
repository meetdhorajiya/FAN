import numpy as np
import matplotlib.pyplot as plt
import os

def perform_fft_analysis(signal, seq_len=96, save_dir="outputs/figures"):
    """
    Performs FFT analysis on a given 1D signal window.
    Identifies dominant frequencies and plots the spectrum.
    """
    print("\n--- Frequency Analysis ---")
    os.makedirs(save_dir, exist_ok=True)
    
    # Take a sample window
    window = signal[:seq_len]
    
    # Apply FFT
    fft_result = np.fft.fft(window)
    
    # Calculate magnitude (amplitude)
    magnitude = np.abs(fft_result)
    
    # Frequencies
    freqs = np.fft.fftfreq(seq_len)
    
    # Identify dominant frequency components (excluding DC component at index 0)
    # We only look at positive frequencies (first half)
    half_len = seq_len // 2
    pos_freqs = freqs[1:half_len]
    pos_magnitude = magnitude[1:half_len]
    
    # Get indices of top frequencies
    top_indices = np.argsort(pos_magnitude)[::-1]
    top_k = min(5, len(pos_freqs))
    
    print(f"Top {top_k} dominant frequencies (normalized):")
    for i in range(top_k):
        idx = top_indices[i]
        print(f"Frequency: {pos_freqs[idx]:.4f}, Magnitude: {pos_magnitude[idx]:.4f}")
        
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Time Domain Window
    plt.subplot(1, 2, 1)
    plt.plot(window)
    plt.title("Time Domain Signal (Window)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    
    # Plot 2: Frequency Spectrum
    plt.subplot(1, 2, 2)
    plt.plot(pos_freqs, pos_magnitude)
    plt.title("Frequency Spectrum (Magnitude)")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "frequency_analysis.png"))
    print(f"Frequency analysis plot saved to {os.path.join(save_dir, 'frequency_analysis.png')}")
    plt.close()
    
    return fft_result, freqs

def top_k_frequency_reconstruction(signal, seq_len=96, k=5, save_dir="outputs/figures"):
    """
    Conceptual Demo: Selects top-K frequency components and reconstructs the signal.
    """
    print("\n--- Top-K Frequency Reconstruction Demo ---")
    window = signal[:seq_len]
    
    # FFT
    fft_result = np.fft.fft(window)
    magnitude = np.abs(fft_result)
    
    # Find top K indices
    # We must keep DC component (idx 0) and zero out others not in top K
    # Also need to preserve symmetry for real inverse FFT
    
    # Consider only first half to find top K
    half_len = seq_len // 2
    top_indices_half = np.argsort(magnitude[1:half_len])[::-1][:k] + 1
    
    # Create mask for top frequencies + DC component
    mask = np.zeros_like(fft_result, dtype=bool)
    mask[0] = True # Keep DC
    for idx in top_indices_half:
        mask[idx] = True
        mask[seq_len - idx] = True # Keep symmetric part for inverse FFT
        
    # Filtered FFT result
    filtered_fft = np.zeros_like(fft_result)
    filtered_fft[mask] = fft_result[mask]
    
    # Inverse FFT
    reconstructed_signal = np.fft.ifft(filtered_fft).real
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(window, label="Original Signal", alpha=0.7)
    plt.plot(reconstructed_signal, label=f"Reconstructed (Top {k} Freqs)", linestyle='--')
    plt.title(f"Signal Reconstruction using Top {k} Frequencies")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    
    save_path = os.path.join(save_dir, "reconstruction_demo.png")
    plt.savefig(save_path)
    print(f"Reconstruction demo plot saved to {save_path}")
    plt.close()
