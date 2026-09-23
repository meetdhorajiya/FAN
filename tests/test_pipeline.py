import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baseline_models import DLinear
from src.dataset import ETTDataset
from src.fan_core import FAN, main_freq_part
from src.forecasting_model import FANIntegratedModel
from src.trainer import evaluate_model, train_model


class FANPipelineTests(unittest.TestCase):
    def test_frequency_decomposition_reconstructs_input(self):
        x = torch.randn(3, 95, 7)
        residual, dominant = main_freq_part(x, k=5)
        self.assertEqual(residual.shape, x.shape)
        self.assertEqual(dominant.shape, x.shape)
        self.assertTrue(torch.allclose(residual + dominant, x, atol=1e-6))

    def test_fan_and_dlinear_output_shapes(self):
        x = torch.randn(2, 96, 7)
        expected = (2, 24, 7)
        self.assertEqual(FANIntegratedModel(96, 24, 7)(x).shape, expected)
        self.assertEqual(DLinear(96, 24, 7)(x).shape, expected)

    def test_fan_rejects_invalid_mode_and_topk(self):
        fan = FAN(96, 24, 7, freq_topk=5)
        with self.assertRaises(ValueError):
            fan(torch.randn(1, 96, 7), mode="invalid")
        with self.assertRaises(ValueError):
            main_freq_part(torch.randn(1, 10, 2), k=7)

    def test_dataset_uses_training_statistics_for_test_data(self):
        path = Path(__file__).parents[1] / "data" / "ETTh1.csv"
        frame = pd.read_csv(path)
        test_data = ETTDataset(path, "test", seq_len=12, pred_len=6)
        train_values = frame.drop(columns="date").values[: int(len(frame) * 0.7)]
        np.testing.assert_allclose(test_data.scaler.mean_, train_values.mean(axis=0))

    def test_training_and_full_evaluation(self):
        x = torch.randn(12, 8, 2)
        y = x[:, -2:, :]
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
        model = DLinear(seq_len=8, pred_len=2, enc_in=2)
        trained = train_model(model, loader, loader, num_epochs=1)
        mse, mae = evaluate_model(trained, loader)
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(np.isfinite(mae))
        self.assertEqual(len(trained.training_history), 1)

    def test_all_ui_checkpoints_reload(self):
        project_root = Path(__file__).parents[1]
        for dataset_name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
            for pred_len in (12, 24, 36, 48, 60, 72, 84, 96):
                for model_name, model_class in (
                    ("FAN", FANIntegratedModel),
                    ("DLinear", DLinear),
                ):
                    checkpoint = (
                        project_root / "outputs" / "models"
                        / f"{dataset_name}_{model_name}_{pred_len}.pt"
                    )
                    self.assertTrue(checkpoint.exists(), checkpoint)
                    model = model_class(96, pred_len, 7)
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    model.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
