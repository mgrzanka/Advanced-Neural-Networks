# 01 — Bike Rental Regression

Predict how many bikes/scooters are rented in a given city in a given hour, from weather and calendar data.

## Approach

- **Target:** `cnt` (total rentals). `casual` and `registered` are dropped — they sum to the target, so keeping them would leak it.
- **Metric = loss.** The task is scored with RMSLE, so RMSLE is used directly as the training objective rather than MSE. `torch.clamp(y_pred, min=0)` keeps the logarithm defined when the network predicts negative counts.
- **Preprocessing** (`DataPreparator` in the notebook, fitted on train only):
  - IQR-based outlier clipping on `temp`, `hum`, `windspeed`,
  - log transform of the skewed `windspeed`,
  - `atemp` dropped (correlates ~1.0 with `temp`), `mnth` dropped (redundant with `season`),
  - standardisation with train statistics applied to validation and test.
- **Model:** small MLP (1 hidden layer, tanh), SGD, 1000 epochs.

## Result

Final training RMSLE ≈ **1.40**. Convergence plot in [`results/training.png`](results/training.png), predictions on the evaluation set in [`results/results.csv`](results/results.csv).

## Files

| File | Role |
|---|---|
| `bike-rental-regression.ipynb` | EDA, feature engineering, training, evaluation |
| `model.py` | `Network`, `torch_rmsle`, training loop |
| `data/data.csv` | training data (hourly records with weather + calendar) |
| `data/evaluation_data.csv` | held-out set, no target column |

## Run

```bash
jupyter notebook bike-rental-regression.ipynb
```
