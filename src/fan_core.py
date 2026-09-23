import torch
import torch.nn as nn

def main_freq_part(x, k, rfft=True):
    """
    DFT/FFT processing and Top-K frequency component selection.
    Reconstructs the non-stationary (dominant frequency) component and 
    separates the residual (stationary) component.
    """
    if x.ndim != 3:
        raise ValueError("Expected input shape (batch, sequence, features).")

    frequency_bins = x.size(1) // 2 + 1 if rfft else x.size(1)
    if not 1 <= k <= frequency_bins:
        raise ValueError(f"k must be between 1 and {frequency_bins}, received {k}.")

    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)
        
    # Top-K frequency component selection
    k_values = torch.topk(xf.abs(), k, dim=1)  
    indices = k_values.indices

    # Reconstruction of non-stationary component
    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        # Supplying n preserves odd as well as even input sequence lengths.
        x_filtered = torch.fft.irfft(xf_filtered, n=x.size(1), dim=1)
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real

    # Residual/stationary component
    norm_input = x - x_filtered
    
    return norm_input, x_filtered

class MLPfreq(nn.Module):
    """
    Non-stationary pattern prediction using MLP.
    Predicts the future values of the non-stationary component.
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        
        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )
        
        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, main_freq, x):
        # main_freq is the extracted non-stationary signal
        # x is the original input
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)

class FAN(nn.Module):
    """
    Frequency Adaptive Normalization (FAN) Core Implementation.
    """
    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.freq_topk = freq_topk
        self.rfft = rfft
        
        self._build_model()
        
    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)
        
    def loss(self, true):
        # Useful if we want to train the FAN module with its custom loss
        residual, pred_main = main_freq_part(true, self.freq_topk, self.rfft)
        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual) 
        
    def normalize(self, input):
        """
        Frequency-based normalization.
        """
        bs, length, dim = input.shape
        
        # Extract residual (norm_input) and non-stationary component (x_filtered)
        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)
        
        # Predict the future non-stationary pattern
        self.pred_main_freq_signal = self.model_freq(
            x_filtered.transpose(1, 2), 
            input.transpose(1, 2)
        ).transpose(1, 2)
        
        return norm_input.reshape(bs, length, dim)

    def denormalize(self, input_norm):
        """
        Adaptive denormalization.
        Combines the predicted residual and predicted non-stationary components.
        """
        if not hasattr(self, 'pred_main_freq_signal'):
            raise RuntimeError("normalize must be called before denormalize.")
        bs, length, dim = input_norm.shape
        if input_norm.shape != self.pred_main_freq_signal.shape:
            raise ValueError(
                "Residual and frequency predictions must have identical shapes; "
                f"received {tuple(input_norm.shape)} and "
                f"{tuple(self.pred_main_freq_signal.shape)}."
            )
        
        self.pred_residual = input_norm
        # Combine predictions
        output = self.pred_residual + self.pred_main_freq_signal
        
        return output.reshape(bs, length, dim)
    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)
        raise ValueError("mode must be 'n' (normalize) or 'd' (denormalize)")
