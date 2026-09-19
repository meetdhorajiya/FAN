import torch
import torch.nn as nn
from src.fan_core import FAN

class SimpleLinearBackbone(nn.Module):
    """
    A simple linear backbone for time-series forecasting.
    Takes input of shape (Batch, Seq_Len, Features)
    Outputs prediction of shape (Batch, Pred_Len, Features)
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(SimpleLinearBackbone, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # A simple linear layer applied to each feature independently
        self.linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        # We want to apply linear on the time dimension.
        # Transpose to (Batch, Features, Seq_Len) -> Linear -> Transpose back
        x = x.transpose(1, 2)
        out = self.linear(x)
        return out.transpose(1, 2)


class FANIntegratedModel(nn.Module):
    """
    Integrates the FAN core with a given forecasting backbone.
    Data flow: Input -> FAN Normalize -> Backbone -> FAN Denormalize -> Final Prediction
    """
    def __init__(self, seq_len, pred_len, enc_in, backbone_model=None, freq_topk=5, rfft=True):
        super(FANIntegratedModel, self).__init__()
        
        self.fan_module = FAN(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, freq_topk=freq_topk, rfft=rfft)
        
        if backbone_model is None:
            self.backbone = SimpleLinearBackbone(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        else:
            self.backbone = backbone_model
            
    def forward(self, x):
        # 1. FAN Normalization
        # Extracts non-stationary component and passes residual out
        # It also computes internal non-stationary prediction
        norm_x = self.fan_module(x, mode='n')
        
        # 2. Forecasting Backbone
        # Forecasts the future of the residual component
        pred_res = self.backbone(norm_x)
        
        # 3. FAN Denormalization
        # Combines the backbone's residual prediction with FAN's non-stationary prediction
        final_pred = self.fan_module(pred_res, mode='d')
        
        return final_pred
