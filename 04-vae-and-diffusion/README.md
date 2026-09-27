# 04 - VAE & Diffusion (DDPM from scratch)

Generative models for synthetic road-sign images (GTSRB, 32×32). A **denoising diffusion probabilistic model implemented from first principles** alongside a convolutional VAE as a baseline.

No `diffusers`, no pretrained weights - the noise schedule, forward process, denoising network and reverse sampler are all in [`diffusion_model.py`](diffusion_model.py).

## The diffusion model

**Forward process** - closed form, so any timestep is reachable in one step:

```python
betas  = torch.linspace(1e-4, 0.02, 1000)       # linear schedule
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=-1)

# q(x_t | x_0) sampled directly:
noisy_x = sqrt(alphas_cum) * x + sqrt(1 - alphas_cum) * noise
```

Training samples a random `t` per image and regresses the network onto the noise that was added - MSE between predicted and true ε.

**Denoising network** - a U-Net built from:

- `SinusoidalPositionEmbeddings` - transformer-style timestep encoding, projected into every residual block so the network knows _how noisy_ its input is,
- `DiffusionResBlock` - residual blocks with the timestep embedding added to the feature maps,
- `SelfAttention` at the bottleneck - global context at the lowest resolution, where it is affordable,
- skip connections from encoder to decoder.

**Sampling** - the iterative reverse chain, 1000 steps from pure Gaussian noise back to an image.

## The VAE baseline

Convolutional encoder → `(mu, log_var)` → reparameterisation → decoder. Loss is reconstruction MSE (summed, per-sample) plus KL divergence. Reconstructions are logged to W&B each validation epoch, which makes posterior collapse immediately visible.

## Experiments

Three diffusion scales were trained to see what capacity buys at 32×32:

| Config  | Channels       | Time emb. | Attention | Trained to |
| ------- | -------------- | --------- | --------- | ---------- |
| small   | (32, 64, 128)  | 128       | off       | epoch 254  |
| big     | (64, 128, 256) | 256       | on        | epoch 343  |
| biggest | (64, 128, 256) | 256       | on        | epoch 407  |

Both model families share one training entry point and one sampler, selected by config dict - so the comparison is like-for-like. Each produces 1000 samples via `sample.py`.

## Files

| File                      | Role                                                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `diffusion_model.py`      | `LiDiffusion` (Lightning loop), `SinusoidalPositionEmbeddings`, `SelfAttention`, `DiffusionResBlock`, `DiffusionModel` |
| `vae_model.py`            | `LitVAE` + convolutional `VAEModel`                                                                                    |
| `config.py`               | model/transform configs for VAE and both diffusion scales                                                              |
| `dataset.py`              | ImageFolder loading, 80/20 split                                                                                       |
| `train.py`                | Lightning `Trainer`, W&B logging, mixed precision, checkpointing                                                       |
| `sample.py`               | generates 1000 samples from a checkpoint                                                                               |
| `vae-and-diffusion.ipynb` | the Colab run                                                                                                          |

## Run

Place the dataset as `data/trafic_32/<class>/*.jpg`, then:

```bash
python train.py     # trains VAE, then diffusion
python sample.py    # 1000 samples from a checkpoint
```

> Only the VAE checkpoint is committed (9 MB). The diffusion checkpoints are excluded - `train.py` reproduces them.
