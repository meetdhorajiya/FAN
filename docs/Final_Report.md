# FAN Project: Final Analysis and Documentation

## 1. Implemented System

The repository implements an end-to-end multivariate forecasting pipeline:

1. Chronological loading and a 70/20/10 train/validation/test split.
2. Standard scaling fitted only on the training split.
3. FFT-based extraction of the Top-K dominant components.
4. Residual forecasting with a linear backbone.
5. MLP forecasting of the extracted frequency component.
6. Combination of both forecasts through FAN denormalization.
7. Comparison against a DLinear baseline using MSE and MAE.

All four bundled ETT datasets, eight forecast horizons, saved checkpoints, complete-test evaluation, plots, and the Gradio interface are present in the repository.

## 2. Experimental Setup

- Datasets: ETTh1, ETTh2, ETTm1, and ETTm2.
- Input sequence length: 96.
- Forecast horizons: 12, 24, 36, 48, 60, 72, 84, and 96.
- Split: chronological 70% training, 20% validation, and 10% testing.
- Scaling: `StandardScaler` fitted on training rows only.
- Training: 3 epochs, Adam, learning rate 0.001, batch size 32, seed 42, CPU.
- FAN Top-K for the main comparison: 5.
- Metrics: element-weighted MSE and MAE over every sample and all seven normalized output channels.
- UI display: OT (channel index 6); its displayed metrics are OT-only for the selected example.

The best validation state from the three epochs is restored before test evaluation. Machine-readable records are stored in `outputs/results/training_results.*` and independently reloaded results are stored in `outputs/results/evaluation_results.*`.

## 3. Full Test Results at Horizon 24

| Dataset | Model | Test MSE | Test MAE | Lower MSE |
|---|---|---:|---:|---|
| ETTh1 | FAN | 0.2995 | 0.3739 | **FAN** |
| ETTh1 | DLinear | 0.3058 | 0.3716 | |
| ETTh2 | FAN | 0.0831 | 0.2061 | |
| ETTh2 | DLinear | 0.0816 | 0.2026 | **DLinear** |
| ETTm1 | FAN | 0.2736 | 0.3403 | |
| ETTm1 | DLinear | 0.2507 | 0.3314 | **DLinear** |
| ETTm2 | FAN | 0.0577 | 0.1687 | **FAN** |
| ETTm2 | DLinear | 0.0600 | 0.1713 | |

These are project results, not numbers copied from the FAN paper. FAN has the lower MSE on ETTh1 and ETTm2 at horizon 24; DLinear has the lower MSE on ETTh2 and ETTm1.

## 4. ETTh1 Forecast-Horizon Results

| Horizon | FAN MSE | FAN MAE | DLinear MSE | DLinear MAE |
|---:|---:|---:|---:|---:|
| 12 | 0.2751 | 0.3578 | 0.2918 | 0.3656 |
| 24 | 0.2995 | 0.3739 | 0.3058 | 0.3716 |
| 36 | 0.3288 | 0.3924 | 0.3360 | 0.3936 |
| 48 | 0.3428 | 0.4019 | 0.3502 | 0.4016 |
| 60 | 0.3552 | 0.4105 | 0.3684 | 0.4149 |
| 72 | 0.3768 | 0.4269 | 0.3854 | 0.4266 |
| 84 | 0.3842 | 0.4325 | 0.3942 | 0.4326 |
| 96 | 0.3875 | 0.4354 | 0.3987 | 0.4353 |

In this run, error generally increases with forecast horizon. FAN has the lower ETTh1 MSE at every tested horizon, while MAE is close between the two models.

## 5. Top-K Ablation on ETTh1, Horizon 24

| Top-K | Test MSE | Test MAE |
|---:|---:|---:|
| 1 | 0.3002 | 0.3736 |
| 3 | 0.2972 | 0.3738 |
| 5 | 0.2981 | 0.3739 |
| 10 | 0.2958 | 0.3720 |
| 20 | **0.2948** | **0.3718** |

The tested range shows a small improvement as more dominant components are retained, with Top-K=20 producing the lowest MSE and MAE in this ablation. The differences are modest and apply only to this recorded ETTh1 run.

## 6. Generated Evidence

- `outputs/results/evaluation_results.csv`: full results for 64 dataset/model/horizon combinations.
- `outputs/results/ablation_topk.csv`: five Top-K ablation runs.
- `outputs/figures/model_comparison_mse.png` and `model_comparison_mae.png`: horizon comparisons.
- `outputs/figures/forecast_*_24.png`: saved FAN/DLinear forecast examples for all four datasets.
- `outputs/figures/ablation_topk.png`: ablation visualization.
- `outputs/models/`: trained checkpoints used by evaluation and the UI.

## 7. Limitations

- Training is intentionally limited to three CPU epochs; the results are a project-scale comparison rather than a claim of convergence or state-of-the-art performance.
- The integrated FAN model uses the repository's `SimpleLinearBackbone`; other backbones are not evaluated here.
- The main metrics average all seven normalized channels and are not inverse-transformed into physical units.
- Each configuration is represented by one seeded run, so run-to-run variance is not measured.
- Uploaded UI data must have the same seven numeric feature columns and should match the selected checkpoint's dataset distribution.

## 8. Reproduction Commands

```powershell
python scripts/train_all.py
python scripts/evaluate_all.py
python scripts/run_ablation.py
python -m unittest discover -s tests -v
python app.py
```

## 9. Conclusion

The project provides a complete, executable FAN-based forecasting prototype with leakage-safe scaling, an implemented baseline, full-loader training and evaluation, reproducible records, checkpoint reload verification, ablation analysis, report-ready figures, automated tests, and an interactive UI. The recorded results show that FAN is competitive with DLinear, but the winner depends on the dataset and metric.
