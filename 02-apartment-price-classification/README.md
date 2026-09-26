# 02 — Apartment Price Classification

Classify an apartment into one of three price brackets from mixed numeric and categorical features.

The point of this project is not the classifier — it is the **controlled ablation**: one baseline, then one regularisation knob changed at a time, each run logged to Weights & Biases.

## Approach

- **Categorical features via entity embeddings** rather than one-hot: each category gets a learned dense vector, concatenated with the continuous features before the MLP. One-hot is kept as an ablation arm to test whether the embeddings actually earn their place.
- **Class-weighted cross-entropy** — the brackets are imbalanced, and accuracy is measured per class and averaged, so predicting the majority class is not rewarded.
- **Configurable architecture:** layer sizes, dropout, batch-norm and weight decay are all parameters, which is what makes the sweep possible.

## Experiments

| # | Experiment | What it tests |
|---|---|---|
| 00 | Baseline | `[128, 64]`, dropout 0.3, weight decay 1e-4, batch-norm, weighted loss |
| 01 | BatchNorm off | contribution of normalisation |
| 02 | Dropout 0.0 / 0.5 | under- vs over-regularisation |
| 03 | Weight decay 0.0 / 1e-2 | L2 strength |
| 04 | Size `[32,16]` / `[1024,512,256]` | deliberate underfit vs overfit |
| 05 | One-hot encoding | embeddings vs one-hot |
| 06 | Unweighted loss | effect of class weighting under imbalance |

Trained weights for every arm are in `models/`. Full write-up with curves: [W&B report](report-link.txt).

## Files

| File | Role |
|---|---|
| `experiments.py` | defines and runs the 8-experiment grid |
| `train.py` | training loop, checkpointing, per-class accuracy |
| `model.py` | MLP with entity embeddings, configurable depth/regularisation |
| `data_preprocessing.py` | encoding, scaling, the fitted transformer saved per experiment |
| `sample.py` | predictions on the test set → `pred.csv` |
| `evaluation.py` | mean per-class accuracy |
| `apartment-price-eda.ipynb` | exploratory analysis |

## Run

The course dataset is not redistributed here. Place `train_data.csv` and `test_data.csv` in `data/`, then:

```bash
python experiments.py     # runs all 8 arms
python sample.py          # predictions from the chosen model
```
