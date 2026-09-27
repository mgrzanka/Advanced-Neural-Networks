# 05 - Composer Classification (RNN)

Identify the composer of a classical piece from its chord sequence. Five classes: Bach, Beethoven, Debussy, Scarlatti, Victoria. Chords are normalised to C major / A minor, so the signal is in _progression and phrasing_, not key.

## The sequence problem

Pieces vary enormously in length (mean ≈ 431 chords). The pipeline handles this properly rather than truncating:

- `pad_sequence` builds the batch, and each sample's true length travels with it,
- `pack_padded_sequence` feeds the LSTM, so **padding never reaches the recurrent computation**,
- `pad_packed_sequence` unpacks for pooling, and mean pooling divides by **real** lengths, not padded ones.

Get this wrong and padding is silently averaged into every sequence representation - which is why the pooling variants are ablated below rather than assumed.

## Chords as embeddings

The final configuration (`experiments.py`) treats a chord as a **token, not a number**. Feeding the raw index as a scalar implies chord 180 is "larger than" chord 90, which is meaningless; an embedding table lets the model learn its own geometry over the chord space.

The vocabulary is derived from the data rather than guessed - the highest chord index present is 191, so:

```python
VOCAB_SIZE = 194   # chords 0-191, pause 192, padding 193
```

`nn.Embedding(..., padding_idx=193)` keeps the padding vector pinned at zero and excluded from gradient updates. Preprocessing maps NaNs to padding, negative values to the pause token, and clips anything above the vocabulary - so no index can escape the table.

Both representations remain available through `USE_EMBEDDING`, since the scalar variant is what the ablations below were run with.

## Architecture ablation

Same data, same training loop, one change at a time (scalar input, hidden size 128):

| Variant                                       | Validation accuracy |
| --------------------------------------------- | ------------------- |
| Unidirectional LSTM, last hidden state        | 0.7925              |
| **Bidirectional** LSTM                        | 0.8118              |
| **Bidirectional + mean pooling** over outputs | **0.8288**          |

Reading it: seeing the sequence in both directions helps, and pooling over _all_ timesteps helps again - the final hidden state alone discards most of a 400-chord piece. Both findings carried into the final configuration.

> The embedding model was trained last and its weights are shipped (`models/best_model_embedding.pt`, `pred_embedding.csv` - 24% of test predictions differ from the scalar model's), but that run's validation accuracy was not captured in a saved notebook, so it is not quoted here. `python experiments.py` reproduces it.

## Training

Adam with weight decay, `ReduceLROnPlateau`, gradient clipping at 1.0, best-on-validation checkpointing, 100 epochs. The classifier head is BatchNorm → Dropout(0.5) → Linear → GELU → BatchNorm → Dropout(0.5) → Linear.

## Files

| File                            | Role                                                                        |
| ------------------------------- | --------------------------------------------------------------------------- |
| `experiments.py`                | entry point and final configuration (embeddings on, vocab 194)              |
| `models.py`                     | `LSTM_Seq_Regressor`, `MusicAuthorClassifier` (pooling + embedding options) |
| `dataset.py`                    | pickle loading, token mapping, padding, length-aware collate                |
| `train.py`                      | training loop, scheduler, checkpointing                                     |
| `sample.py`                     | predictions → `pred.csv` (no header, no index, as the task requires)        |
| `composer-classification.ipynb` | end-to-end run, scalar input - the one with full training logs              |
| `embeddings.ipynb`              | embedding variant and the vocabulary-size derivation                        |
| `architecture-ablations.ipynb`  | the three-variant comparison above                                          |
| `models/`                       | trained weights: embedding and scalar variants                              |

## Run

```bash
python experiments.py
```

Data (`data/train.pkl`, `data/test_no_target.pkl`) is included - it is small.
