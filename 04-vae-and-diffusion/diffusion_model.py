import wandb
import torch
import torchvision
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch.nn as nn
import math


class LiDiffusion(L.LightningModule):
    def __init__(self, diffusion_model: nn.Module, num_epochs, diffusion_steps=1000) -> None:
        super().__init__()

        self.model = diffusion_model
        self.criterion = nn.MSELoss()

        self.num_epochs = num_epochs
        self.diffusion_steps = diffusion_steps

        self.register_buffer("betas", torch.linspace(1e-4, 0.02, diffusion_steps))
        self.register_buffer("alphas", 1. - self.betas) # type: ignore
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=-1)) # type: ignore

    def _calculate_diffusion_loss(self, x, device):
        timestamps = torch.randint(0, self.diffusion_steps, size=(x.shape[0],), device=device)
        alphas_cum = self.alphas_cumprod[timestamps].view(-1, 1, 1, 1) # type: ignore

        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alphas_cum) * x + torch.sqrt(1. - alphas_cum) * noise

        predicted_noise = self.model(noisy_x, timestamps)
        loss = self.criterion(predicted_noise, noise)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        device = x.device

        loss = self._calculate_diffusion_loss(x, device)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        device = x.device

        loss = self._calculate_diffusion_loss(x, device)
        self.log("val/loss", loss, prog_bar=True)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            n = min(x.size(0), 16)

            reconstructed_images = self.generate_batch(n, device)

            grid = torchvision.utils.make_grid(
                reconstructed_images, nrow=4, normalize=True, value_range=(-1, 1)
            )
            self.logger.experiment.log({"val/reconstructions": [wandb.Image(grid)]})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return [optimizer], [lr_scheduler]

    def generate_batch(self, batch_size, device):
        reconstructed_images = torch.randn((batch_size, 3, 32, 32), device=device)

        for t in reversed(range(self.diffusion_steps)):
            timestamps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                # self.model to teraz referencja do U-Netu wewnątrz LiDiffusion
                predicted_noise = self.model(reconstructed_images, timestamps)

            alpha_t = self.alphas[t]                # type: ignore
            alpha_cum_t = self.alphas_cumprod[t]    # type: ignore
            beta_t = self.betas[t]                  # type: ignore

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cum_t)

            mu = coef1 * (reconstructed_images - coef2 * predicted_noise)

            if t > 0:
                z = torch.randn_like(reconstructed_images)
                sigma = torch.sqrt(beta_t)
                reconstructed_images = mu + sigma * z
            else:
                reconstructed_images = mu

        return torch.clamp(reconstructed_images, -1.0, 1.0)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, timestamp_indxes):
        device = timestamp_indxes.device

        half_dim = self.size // 2
        embeddings = torch.exp(-1 * torch.arange(half_dim, device=device) * math.log(10000)/(half_dim-1))
        embeddings = timestamp_indxes[:, None] * embeddings[None, :]

        return torch.concatenate((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1) # all in one conv
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        norm_x = self.group_norm(x)

        qkv = self.qkv(norm_x)
        qkv = qkv.view(batch_size, 3, channels, height * width) # (Batch, 3, Channels, N_pixels)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.transpose(-2, -1) # (Batch, N_pixels, Channels)
        attention_scores = torch.bmm(q, k) * (channels ** -0.5) # (Batch, N_pixels, N_pixels)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        v_transpose = v.transpose(-2, -1) # (Batch, N_pixels, Channels)
        out = torch.bmm(attention_weights, v_transpose) # (Batch, N_pixels, Channels)

        out = out.transpose(-2, -1).contiguous().view(batch_size, channels, height, width)

        return x + self.output_proj(out)


class DiffusionResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, timestamp_emb_dim) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(timestamp_emb_dim, out_ch)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.bnorm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, timestamps):
        out = self.bnorm1(x)
        out = self.act(out)
        out = self.conv1(out)

        timestamps = self.time_mlp(self.act(timestamps))
        timestamps_matrix = timestamps[:, :, None, None]
        out = out + timestamps_matrix

        out = self.bnorm2(out)
        out = self.act(out)
        out = self.conv2(out)

        return out + self.res_conv(x)


class DiffusionModel(nn.Module):
    def __init__(self, image_channels=3, block_out_channels=(32, 64, 128), time_emb_dim=128, use_attention=True):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.initial_conv = nn.Conv2d(image_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Unet ENCODER
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        for i in range(len(block_out_channels)-1):
            res_block = DiffusionResBlock(block_out_channels[i], block_out_channels[i], time_emb_dim)
            downsample_block = nn.Conv2d(block_out_channels[i], block_out_channels[i+1], 4, 2, padding=1)

            self.encoder_blocks.append(res_block)
            self.downsample_blocks.append(downsample_block)

        # Unet Bottleneck
        self.bottleneck_layer1 = DiffusionResBlock(block_out_channels[-1], block_out_channels[-1], time_emb_dim)
        self.bottleneck_attention = SelfAttention(block_out_channels[-1]) if use_attention else None
        self.bottleneck_layer2 = DiffusionResBlock(block_out_channels[-1], block_out_channels[-1], time_emb_dim)

        # Unet DECODER
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(1, len(block_out_channels)):
            unsample_block = nn.ConvTranspose2d(block_out_channels[-i], block_out_channels[-(i+1)], 4, 2, padding=1)
            res_block = DiffusionResBlock(2 * block_out_channels[-(i+1)], block_out_channels[-(i+1)], time_emb_dim)

            self.decoder_blocks.append(res_block)
            self.upsamples.append(unsample_block)

        self.output_conv = nn.Conv2d(block_out_channels[0], image_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        out = self.initial_conv(x)

        residuals = []

        for conv_block, downsample_block in zip(self.encoder_blocks, self.downsample_blocks):
            out = conv_block(out, t_emb)
            residuals.append(out)
            out = downsample_block(out)

        out = self.bottleneck_layer1(out, t_emb)
        if self.bottleneck_attention is not None:
            out = self.bottleneck_attention(out)
            out = self.bottleneck_layer2(out, t_emb)

        for conv_block, upsample_block in zip(self.decoder_blocks, self.upsamples):
            out = upsample_block(out)
            residual_x = residuals.pop()
            out = conv_block(torch.concat([out, residual_x], dim=1), t_emb)

        out = self.output_conv(out)
        return out
