import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import load_data
from config import diffusion_config_small_model, diffusion_config_big_model, vae_config


def train(config):
    train_transform = config["train_transform"]
    test_transform = config["test_transform"]

    train_loader, val_loader, class_to_idx = load_data(train_transform, test_transform, batch_size=64)
    num_classes=len(class_to_idx.keys())

    raw_model = config["model_class"](**config["model_args"])
    model = config["lightning_model_class"](raw_model, **config["lightning_model_class_args"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config['name']}",
        filename="ckp-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    wandb_logger = WandbLogger(project="MyExperiments")
    trainer = L.Trainer(
        max_epochs=config["lightning_model_class_args"]["num_epochs"],
        accelerator='gpu',
        devices=1,
        precision="16-mixed",   # Faster training
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    wandb.finish()


if __name__ == '__main__':
    train(vae_config)
    train(diffusion_config_big_model)
    train(diffusion_config_small_model)
