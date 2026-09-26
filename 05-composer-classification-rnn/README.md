# 05 — Composer Classification (RNN)

Identify the composer of a classical piece from its chord sequence. Five classes: Bach, Beethoven, Debussy, Scarlatti, Victoria. Chords are normalised to C major / A minor, so the signal is in *progression and phrasing*, not key.

## The sequence problem

Pieces vary enormously in length (mean ≈ 431 chords). The pipeline handles this properly rather than truncating:

- `pad_sequence` builds the batch, and each sample's true length travels with it,
- `pack_padded_sequence` feeds the LSTM, so **padding never reaches the recurrent computation**,
- `pad_packed_sequence` unpacks for pooling, and the mean-pooling arm divides by real lengths, not padded ones.

Getting this wrong silently averages in padding and quietly costs accuracy — hence the ablation below.

## Architecture ablation

Same data, same training loop, one change at a time:

| Variant | Validation accuracy |
|---|---|
| Unidirectional LSTM, last hidden state | 0.7925 |
| **Bidirectional** LSTM | 0.8118 |
| **Bidirectional + mean pooling** over outputs | **0.8288** |

Reading it: seeing the sequence in both directions helps, and pooling over *all* timesteps helps again — the final hidden state alone discards most of a 400-chord piece.

The model also supports two input representations, switched by `USE_EMBEDDING`: chords as scalar values, or as learned embeddings from a 194-token vocabulary (192 chords + pause + padding).

## Training

Adam with weight decay, `ReduceLROnPlateau`, gradient clipping at 1.0, best-on-validation checkpointing, 100 epochs.

## Files

| File | Role |
|---|---|
| `models.py` | `LSTM_Seq_Regressor`, `MusicAuthorClassifier` (pooling + embedding options) |
| `dataset.py` | pickle loading, padding, length-aware collate |
| `train.py` | training loop, scheduler, checkpointing |
| `sample.py` | predictions → `pred.csv` |
| `experiments.py` | entry point; hyperparameters at the top |
| `composer-classification.ipynb` | final configuration, end to end |
| `architecture-ablations.ipynb` | the four-variant comparison above |

## Run

```bash
python experiments.py
```

Data (`data/train.pkl`, `data/test_no_target.pkl`) is included — it is small.
