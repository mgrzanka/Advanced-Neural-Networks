import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn as nn
import torchvision
import wandb


class LitVAE(L.LightningModule):
    def __init__(self, vae_model: nn.Module, num_epochs) -> None:
        super().__init__()
        self.num_epochs = num_epochs
        self.model = vae_model
        self.reconstruction_criterion = nn.MSELoss(reduction='sum')

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed_image, mu, var_log = self.model(x)

        recon_loss = self.reconstruction_criterion(reconstructed_image, x) / x.size(0)
        kl_loss = self._calculate_kl_loss(mu, var_log)
        loss = recon_loss + kl_loss

        self.log("train/recon_loss", recon_loss, prog_bar=False)
        self.log("train/kl_loss", kl_loss, prog_bar=False)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed_image, mu, var_log = self.model(x)

        recon_loss = self.reconstruction_criterion(reconstructed_image, x) / x.size(0)
        kl_loss = self._calculate_kl_loss(mu, var_log)
        loss = recon_loss + kl_loss

        self.log("val/loss", loss, prog_bar=True)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n], reconstructed_image[:n]])
            grid = torchvision.utils.make_grid(comparison, nrow=n, normalize=False)
            self.logger.experiment.log({"val/reconstructions": [wandb.Image(grid)]})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return [optimizer], [lr_scheduler]

    def _calculate_kl_loss(self, mu, var_log):
        kl_loss = -0.5 * torch.sum(1 + var_log - mu.pow(2) - var_log.exp(), dim=[1, 2, 3])
        return kl_loss.mean()


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.relu = nn.ReLU()

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.bnorm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        out = self.bnorm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bnorm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + self.res_conv(x)


class VAEModel(nn.Module):
    def __init__(self, image_channels=3, block_out_channels=(32, 64, 128)) -> None:
        super().__init__()

        self.initial_conv = nn.Conv2d(image_channels, block_out_channels[0], 3, padding=1)

        # ENCODER
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        for i in range(len(block_out_channels)-1):
            res_block = ResBlock(block_out_channels[i], block_out_channels[i])
            downsample_block = nn.Conv2d(block_out_channels[i], block_out_channels[i+1], 4, 2, padding=1)

            self.encoder_blocks.append(res_block)
            self.downsample_blocks.append(downsample_block)

        self.encoder_output = nn.Conv2d(block_out_channels[-1], 2*block_out_channels[-1], 3, padding=1) # mu and log(var)

        # DECODER
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i in range(1, len(block_out_channels)):
            upsample_block = nn.ConvTranspose2d(block_out_channels[-i], block_out_channels[-(i+1)], 4, 2, padding=1)
            res_block = ResBlock(block_out_channels[-(i+1)], block_out_channels[-(i+1)])

            self.decoder_blocks.append(res_block)
            self.upsample_blocks.append(upsample_block)

        self.decoder_output = nn.Conv2d(block_out_channels[0], image_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, var_log):
        std = torch.exp(0.5 * var_log)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = z
        for conv_block, upsample_block in zip(self.decoder_blocks, self.upsample_blocks):
            out = conv_block(upsample_block(out))

        recostructed_image = self.decoder_output(out)
        return self.sigmoid(recostructed_image)

    def forward(self, x):
        out = self.initial_conv(x)

        for conv_block, downsample_block in zip(self.encoder_blocks, self.downsample_blocks):
            out = downsample_block(conv_block(out))

        encoder_output = self.encoder_output(out)
        mu, var_log = torch.chunk(encoder_output, 2, dim=1)

        z = self.reparameterize(mu, var_log)    # sample from latent space

        reconstructed_scaled_image = self.decode(z)
        return reconstructed_scaled_image, mu, var_log
