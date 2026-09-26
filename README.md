# Advanced Neural Networks

**Six self-contained PyTorch projects**, built over one semester of *Sieci Neuronowe* (SSNE) at Warsaw University of Technology. Each one takes a different data modality and a different class of architecture, from a tabular MLP through a from-scratch denoising diffusion model to parameter-efficient fine-tuning of a Polish BERT.

Every project directory holds both **runnable modules** (`.py`) and the **Colab notebook** that produced the results, with training logs and plots preserved in the outputs.

---

| # | Project | Task | Architecture | Result |
|---|---|---|---|---|
| [01](01-bike-rental-regression) | Bike rental regression | Predict hourly rental counts from weather/calendar features | MLP, RMSLE optimised directly as the loss | RMSLE ≈ 1.40 |
| [02](02-apartment-price-classification) | Apartment price class | 3-class price bucket from mixed tabular features | MLP with entity embeddings for categoricals | 8-experiment regularisation ablation |
| [03](03-image-classification-cnn) | Image classification | 50-class image classification at 64×64 | Custom ResNet-style CNN, built from scratch | 72% validation accuracy |
| [04](04-vae-and-diffusion) | **VAE & diffusion** | Generate synthetic road-sign images | **DDPM implemented from scratch** + convolutional VAE | 1000 samples per model, 3 model scales |
| [05](05-composer-classification-rnn) | Composer classification | Identify the composer from a chord sequence | Bi-LSTM over packed variable-length sequences | 82.9% validation accuracy |
| [06](06-hate-speech-peft) | **Hate speech & PEFT** | Binary hate-speech detection in Polish | **HerBERT + LoRA / prompt tuning / full FT** | macro-F1 0.782 |

## Two worth looking at first

**[04 — VAE & Diffusion](04-vae-and-diffusion)** implements DDPM from first principles: the noise schedule, the closed-form forward process, a U-Net with sinusoidal timestep embeddings and self-attention at the bottleneck, and the iterative reverse sampler. No `diffusers`, no pretrained weights — the whole loop is in [`diffusion_model.py`](04-vae-and-diffusion/diffusion_model.py). A VAE on the same data serves as the comparison baseline.

**[06 — Hate speech & PEFT](06-hate-speech-peft)** compares four ways of adapting a 124M-parameter transformer to a small, heavily imbalanced dataset — frozen-backbone transfer, full fine-tuning, LoRA adapters (0.24% of weights) and prompt tuning (0.014%) — under one training loop and one metric, so the comparison is actually fair.

## Layout

```
NN-project/
├── *.py                 modules: dataset / model / train / sample
├── *.ipynb              the Colab run, with outputs and plots
├── task.md              the original assignment
└── README.md            problem, approach, results
```

Each project runs from its own directory (`cd 04-vae-and-diffusion && python train.py`).

## Stack

PyTorch · PyTorch Lightning · Hugging Face Transformers + PEFT · torchvision · scikit-learn · Weights & Biases · pandas

## A note on data and weights

Course datasets (image folders, pickles, apartment CSVs) are distributed by the university and are **not** committed here — each project README says what to place in its `data/` directory. Trained checkpoints are kept only where small enough to be useful; the diffusion checkpoints (up to 484 MB) are excluded, and `train.py` reproduces them.
