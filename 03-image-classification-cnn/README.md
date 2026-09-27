# 03 - Image Classification (CNN from scratch)

50-class image classification at 64×64, with a residual CNN written from scratch - no `torchvision.models`, no pretrained weights.

## Architecture

`CNNImageClassifier` stacks configurable `ConvBlock`s, each a two-convolution residual unit (conv → BN → ReLU → conv → BN, plus the skip):

```
initial conv (3 → 32)
  ├─ 2 × ConvBlock(32)    stride 1
  ├─ 2 × ConvBlock(64)    stride 2 on the first  ─┐ 1×1 projection on the skip
  ├─ 2 × ConvBlock(128)   stride 2 on the first   │ whenever shape changes
  └─ 2 × ConvBlock(256)   stride 2 on the first  ─┘
global average pool → FC 256 → dropout 0.5 → FC 50
```

The residual connection is a flag (`use_res_connection`), so the plain-CNN variant is one argument away for comparison.

## Training choices

- **Class weighting** - the classes are imbalanced (confirmed in EDA), so the loss is weighted by inverse class frequency.
- **Heavy augmentation** - 88k images across 50 classes is not much: random crop with padding, horizontal flip, ±15° rotation, colour jitter, and `RandomErasing(0.6)`.
- **Normalisation constants** (`MEAN`, `STD`) computed from the training split only.
- Checkpoint saved on **validation** accuracy improvement, not training accuracy.

## Result

**72% validation accuracy** over 50 classes (random ≈ 2%), AdamW, 50 epochs.

The notebook also contains an exploratory arm using **CLIP zero-shot** as a pseudo-ground-truth generator for the unlabelled test set - a check on how far a general-purpose vision-language model gets on the same task without training.

## Files

| File                         | Role                                                                |
| ---------------------------- | ------------------------------------------------------------------- |
| `model.py`                   | `ConvBlock`, `CNNImageClassifier`                                   |
| `dataset.py`                 | ImageFolder split, class weights, train/val transforms, test loader |
| `train.py`                   | training loop with weighted loss and best-checkpoint saving         |
| `predict.py`                 | predictions preserving original filenames → `pred.csv`              |
| `image-classification.ipynb` | full Colab run including the CLIP experiment                        |

## Run

Place the dataset as `data/train/<class>/*.JPEG` and `data/test/*.JPEG`, then:

```bash
python train.py
python predict.py
```
