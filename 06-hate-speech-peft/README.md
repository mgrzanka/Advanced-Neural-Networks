# 06 - Hate Speech Detection & Parameter-Efficient Fine-Tuning

Binary hate-speech classification for Polish comments, and - the real subject - **a controlled comparison of four ways to adapt a pretrained transformer** to a small, heavily imbalanced dataset.

Base model: **HerBERT** (`allegro/herbert-base-cased`), 124M parameters, trained for informal Polish.

## The comparison

All four arms share one training loop, one metric and one seed. Only the adaptation strategy changes.

| Method                            | Trainable params | Share  | Val macro-F1 |
| --------------------------------- | ---------------- | ------ | ------------ |
| Frozen backbone + head            | 1,538            | 0.001% | 0.6076       |
| **Full fine-tuning**              | 124,740,868      | 100%   | **0.7773**   |
| LoRA (r=8, query/value)           | 296,450          | 0.24%  | 0.7321       |
| Prompt tuning (20 virtual tokens) | 16,898           | 0.014% | 0.6949       |

Each method gets its own learning rate and epoch budget - PEFT methods need substantially higher LR than full fine-tuning (1e-4 and 1e-2 vs 2e-5), and comparing them at a single LR would be a rigged test.

**What it shows:** with only 10k training examples, full fine-tuning wins outright. But LoRA reaches **94% of its macro-F1 while training 0.24% of the weights**, and prompt tuning gets to 0.69 by learning 17K parameters - the entire backbone untouched. On a larger dataset, or with several tasks to serve from one base model, that trade changes sign.

## The rest of the pipeline

- **Imbalance (91.5% / 8.5%)** - macro-F1 as the metric throughout, and class-weighted cross-entropy in the loss. Accuracy would read 0.92 for a model that never predicts hate.
- **`MAX_LENGTH` from the data** - 95th percentile of token length (38) rounded to 40, instead of a default 128 that would triple the compute on padding.
- **Text normalisation** - HTML entities unescaped, URLs → `[URL]`, @mentions → `[USER]`, so the model learns language rather than specific handles.
- **Threshold tuning** - the decision threshold is swept on validation only, then applied to test. 0.5 → 0.39 lifts macro-F1 from 0.7773 to **0.7817**.
- Best-epoch weights restored after training, not last-epoch.

## Final result

|               | Precision | Recall   | F1       | Support |
| ------------- | --------- | -------- | -------- | ------- |
| Non-hate      | 0.96      | 0.96     | 0.96     | 1839    |
| Hate          | 0.60      | 0.61     | 0.60     | 170     |
| **macro avg** | **0.78**  | **0.78** | **0.78** | 2009    |

Accuracy 0.93, macro-F1 0.7817 at the tuned threshold.

## Files

| File                     | Role                                                                  |
| ------------------------ | --------------------------------------------------------------------- |
| `model.py`               | `build_model(method)` - the four adaptation strategies                |
| `train.py`               | shared training loop, evaluation, method comparison, threshold tuning |
| `data.py`                | cleaning, tokenisation, length selection, class weights               |
| `config.py`              | hyperparameters, including per-method LR/epochs                       |
| `predict.py`             | test predictions at the tuned threshold → `pred.csv`                  |
| `hate-speech-peft.ipynb` | the Colab run with all logs and plots                                 |

## Run

```bash
pip install transformers peft beautifulsoup4 scikit-learn
python train.py     # trains all four arms, picks the best, writes pred.csv
```
