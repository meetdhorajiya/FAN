"""Automated unit tests for the epoch budget experimental extension."""

import csv
import json
import os
import unittest
from pathlib import Path

import torch

from src.baseline_models import DLinear
from src.forecasting_model import FANIntegratedModel


class ExtensionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).parents[1]
        self.results_csv = self.project_root / "outputs" / "extension_results" / "epoch_budget_results.csv"
        self.results_json = self.project_root / "outputs" / "extension_results" / "epoch_budget_results.json"
        self.models_dir = self.project_root / "outputs" / "extension_models"

    def test_extension_results_schema(self):
        """Ensure extension results CSV matches expected academic schema."""
        if not self.results_csv.exists():
            self.skipTest("Extension results CSV not yet generated.")

        required_fields = {
            "dataset", "model", "seq_len", "pred_len", "epoch_budget",
            "best_val_epoch", "best_val_mse", "final_val_mse", "final_train_mse",
            "test_mse", "test_mae", "training_seconds", "seed", "checkpoint"
        }
        with open(self.results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertTrue(required_fields.issubset(set(reader.fieldnames)))
            rows = list(reader)
            self.assertGreater(len(rows), 0, "Extension results should have at least 1 record.")
            for r in rows:
                self.assertIn(r["epoch_budget"], ["3", "10", "20"])
                self.assertGreaterEqual(int(r["best_val_epoch"]), 1)
                self.assertLessEqual(int(r["best_val_epoch"]), int(r["epoch_budget"]))
                self.assertGreater(float(r["test_mse"]), 0.0)
                self.assertGreater(float(r["test_mae"]), 0.0)

    def test_extension_checkpoints_reloadable(self):
        """Verify saved extension checkpoints exist and reload properly into models."""
        if not self.models_dir.exists():
            self.skipTest("Extension models directory not yet generated.")

        checkpoints = list(self.models_dir.glob("*.pt"))
        if not checkpoints:
            self.skipTest("No extension checkpoints saved yet.")

        for cp in checkpoints[:10]:  # Test sample of up to 10 checkpoints
            parts = cp.stem.split("_")
            dataset, model_type, pred_len = parts[0], parts[1], int(parts[2])
            if model_type == "FAN":
                model = FANIntegratedModel(96, pred_len, 7)
            else:
                model = DLinear(96, pred_len, 7)
            state = torch.load(cp, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            x = torch.randn(2, 96, 7)
            out = model(x)
            self.assertEqual(out.shape, (2, pred_len, 7))

    def test_model_selection_prevents_val_loss_regression(self):
        """Verify best_val_mse <= final_val_mse for all recorded models."""
        if not self.results_csv.exists():
            self.skipTest("Extension results CSV not yet generated.")

        with open(self.results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                best_val = float(r["best_val_mse"])
                final_val = float(r["final_val_mse"])
                self.assertLessEqual(
                    best_val, final_val + 1e-6,
                    f"Best validation MSE ({best_val}) cannot exceed final validation MSE ({final_val})"
                )


if __name__ == "__main__":
    unittest.main()
