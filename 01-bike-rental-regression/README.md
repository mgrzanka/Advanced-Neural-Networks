# 01 — Bike Rental Regression

Predict hourly bike/scooter rentals in a city from weather and calendar data. Scored with **RMSLE**.

The model itself is a small MLP — the substance here is the **training setup**: Hydra-configured, Accelerate-backed, EMA-averaged, checkpointed and resumable, tracked in W&B.

## Result

| | RMSLE |
|---|---|
| Train | 0.530 |
| **Validation** | **0.534** |

200 epochs, EMA weights. The near-identical train and validation figures indicate the model is not overfitting — the remaining error is what a 2-layer MLP can extract from these features.

## Why RMSLE is the loss, not just the metric

The task is graded on RMSLE, so the network optimises RMSLE directly rather than MSE and hoping it transfers:

```python
def torch_rmsle(y_pred, y_true):
    msle = torch.mean(torch.square(torch.log(torch.clamp(y_pred, 0) + 1) - torch.log(y_true + 1)))
    return torch.sqrt(msle)
```

The `clamp` is load-bearing and it is applied to **`y_pred`**: an unconstrained linear output layer can emit negative counts, and `log(negative + 1)` is NaN. Clamping the wrong argument makes the loss *look* correct — the squared difference is symmetric, so the value is unchanged — while quietly removing the only guard against NaN.

## Training setup

| Concern | How |
|---|---|
| Configuration | **Hydra** — `config/train.yaml`, `config/sample.yaml`; no constants buried in code |
| Distribution | **HF Accelerate** — `gather_for_metrics` so metrics are correct across processes |
| Weight averaging | **EMA** (decay 0.99), stored and restored around validation so EMA weights are what gets scored |
| Schedule | cosine with 100 warmup steps |
| Stability | gradient clipping at 1.0, gradient accumulation (2 steps) |
| Checkpointing | save/load hooks covering model **and** EMA state, step-frequency checkpoints, **mid-epoch resume** (`resume_step`) |
| Tracking | W&B, with run-ID resumption so a restarted job continues one chart rather than starting a second |
| Debugging | `run_debug()` in both entry points — tiny batches, 2 epochs, offline W&B |

## Preprocessing

An sklearn `ColumnTransformer`, fitted on train only, persisted with joblib and **reloaded at inference** — so serving cannot silently disagree with training:

- `mnth`, `hr` → median impute → standard scale
- `weathersit`, `weekday` → one-hot (`handle_unknown='ignore'`)
- `yr`, `holiday`, `workingday`, `temp`, `hum`, `windspeed` → passthrough
- everything else dropped: `instant` and `dteday` are identifiers, `casual` + `registered` sum to the target and would leak it, `atemp` correlates ~1.0 with `temp` (see the EDA heatmap)

## Files

| File | Role |
|---|---|
| `train.py` | training loop: Accelerate, EMA, scheduler, checkpoint/resume, W&B |
| `sample.py` | inference from EMA weights → `results/results.csv` |
| `model.py` | `NeuralNetwork` — MLP, one hidden layer |
| `data_preprocessing.py` | `DataTransformer` — the fitted, persisted sklearn pipeline |
| `config/` | Hydra configs for training and sampling |
| `notebooks/eda.ipynb` | correlations, distributions, the `atemp`/`temp` redundancy |
| `outputs/epoch-200/` | final weights (`model.pt`, `ema_model.pt`) and the fitted transformer |

## Run

```bash
pip install torch hydra-core accelerate diffusers wandb scikit-learn pandas

python train.py                          # uses config/train.yaml
python train.py epochs=50 lr=3e-4        # Hydra overrides
python sample.py                         # predictions from the EMA checkpoint
```
