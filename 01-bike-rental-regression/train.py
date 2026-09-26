import os
import joblib
import pandas as pd
import torch
import math
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
import wandb
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from model import NeuralNetwork
from data_preprocessing import DataTransformer


logger = get_logger(__name__, log_level="INFO")


def load_data(data_path, transformer_path='outputs/models/transformer.pt'):
    df = pd.read_csv(data_path)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    if os.path.exists(transformer_path):
        transformer = joblib.load(transformer_path)
        X_train = transformer.transform(train_df).astype(np.float32)
    else:
        to_scale_cols = ["mnth", "hr"]
        discrete_cols = ['weathersit', 'weekday']
        other_cols = ["yr", "holiday", "workingday", "temp", "hum", "windspeed"]

        transformer = DataTransformer(to_scale_cols, discrete_cols, other_cols)
        X_train = transformer.fit_transform(train_df, save_path=transformer_path).astype(np.float32)

    y_train = train_df["cnt"].to_numpy(dtype=np.float32)
    X_val = transformer.transform(val_df).astype(np.float32)
    y_val = val_df["cnt"].to_numpy(dtype=np.float32)

    return X_train, y_train, X_val, y_val


def get_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def torch_rmsle(y_pred, y_true):
    msle = torch.mean(torch.square(torch.log(torch.clamp(y_pred, 0) + 1) - torch.log(y_true + 1)))
    return torch.sqrt(msle)


def train(data_path, transformer_path, batch_size, hidden_size, lr, epochs, use_ema, grad_acc_steps, output_dir, checkpoint_path, save_checkpoint_steps_freq, save_model_freq_epochs, config_dict):
    X_train, y_train, X_val, y_val = load_data(data_path, transformer_path)
    train_dataloader, val_dataloader = get_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    input_size = X_train.shape[1]
    max_training_steps = epochs * len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_acc_steps)

    model = NeuralNetwork(input_size, hidden_size, 1)
    total_params = sum(param.numel() for param in model.parameters())
    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=max_training_steps
    )
    ema_model = None
    if use_ema:
        ema_model = EMAModel(
            parameters=model.parameters(),
            decay=0.99,
            model_cls=type(model),
            model_config=None
        )

    accelerator = Accelerator(
        project_dir=output_dir,
        gradient_accumulation_steps=grad_acc_steps,
        log_with="wandb"
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config_dict.update({
            "total_params": total_params,
            "total_trainable_params": total_trainable_params
        })

        wandb_init_kwargs = {"config": config_dict}

        run_id = config_dict.get("wandb_run_id")
        if run_id is not None and checkpoint_path is not None:
            wandb_init_kwargs["id"] = run_id
            wandb_init_kwargs["resume"] = "must"

        accelerator.init_trackers(
            project_name="bikes-regression",
            init_kwargs={"wandb": wandb_init_kwargs}
        )

    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )
    if ema_model:
        ema_model.to(accelerator.device)

    def save_models_hook(models, weights, output_dir):
        if ema_model:
            ema_path = os.path.join(output_dir, "ema_model.pt")
            torch.save(ema_model.state_dict(), ema_path)

        for indx, model in enumerate(models):
            model_path = os.path.join(output_dir, f"model_{indx}.pt")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), model_path)
            weights.pop()

    def load_models_hook(models, input_dir):
        if ema_model:
            ema_path = os.path.join(input_dir, "ema_model.pt")
            ema_state_dict = torch.load(ema_path, map_location="cpu", weights_only=True)
            ema_model.load_state_dict(ema_state_dict)
            ema_model.to(accelerator.device)

        for indx in range(len(models)):
            model_path = os.path.join(input_dir, f"model_{indx}.pt")
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            m_to_load = models.pop()
            m_to_load.load_state_dict(state_dict)

    accelerator.register_save_state_pre_hook(save_models_hook)
    accelerator.register_load_state_pre_hook(load_models_hook)

    global_step = first_epoch = resume_step = 0
    if checkpoint_path is not None:
        accelerator.load_state(checkpoint_path)
        global_step = int(checkpoint_path.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % (num_update_steps_per_epoch * grad_acc_steps)

    logger.info("Starting training")

    for epoch in range(first_epoch, epochs):
        # TRAIN
        model.train()
        train_pbar = tqdm.tqdm(desc=f"Train epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process, total=num_update_steps_per_epoch)

        running_loss = 0.0
        total_train_samples = 0

        for step, (X, y) in enumerate(train_dataloader):
            if checkpoint_path and epoch == first_epoch and step < resume_step:
                if step % grad_acc_steps == 0:
                    train_pbar.update(1)
                continue

            # training logic
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                out = model(X)
                loss = torch_rmsle(out, y)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            out_gathered, y_gathered = accelerator.gather_for_metrics((out, y))
            global_loss = torch_rmsle(out_gathered, y_gathered)
            running_loss += global_loss.item() * y_gathered.size(0)
            total_train_samples += y_gathered.size(0)

            if accelerator.sync_gradients:
                if ema_model:
                    ema_model.step(model.parameters())
                global_step += 1
                train_pbar.update(1)

                curr_avg_loss = running_loss / total_train_samples
                accelerator.log({"train/loss": curr_avg_loss}, step=global_step)
                train_pbar.set_postfix(loss=f"{curr_avg_loss:.4f}")

                if accelerator.is_main_process and global_step % save_checkpoint_steps_freq == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Checkpoint saved to {save_path}")

        train_pbar.close()

        # VALIDATE
        model.eval()
        val_pbar = tqdm.tqdm(
            desc=f"Eval epoch {epoch+1}/{epochs}", disable=not accelerator.is_main_process, total=len(val_dataloader)
        )
        running_val_loss = 0.0
        total_val_samples = 0

        if ema_model:
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

        for step, (X, y) in enumerate(val_dataloader):
            with torch.no_grad():
                out = model(X)

            out_gathered, y_gathered = accelerator.gather_for_metrics((out, y))
            loss_gathered = torch_rmsle(out_gathered, y_gathered).item()
            running_val_loss += loss_gathered * y_gathered.size(0)
            total_val_samples += y_gathered.size(0)

            val_pbar.update(1)

        if ema_model:
            ema_model.restore(model.parameters())

        val_pbar.close()

        validation_loss = running_val_loss / total_val_samples
        accelerator.log({"val/loss": validation_loss}, step=global_step)

        if accelerator.is_main_process and (epoch % save_model_freq_epochs == 0 or epoch == epochs-1):
            save_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "model.pt"))

            if ema_model:
                torch.save(ema_model.state_dict(), os.path.join(save_dir, "ema_model.pt"))

            logger.info(f"Saved models for epoch {epoch} in {save_dir}")


@hydra.main(version_base=None, config_path="config", config_name="train")
def run_with_hydra(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    train(
        data_path=cfg.data_path,
        transformer_path=cfg.transformer_path,
        batch_size=cfg.batch_size,
        hidden_size=cfg.hidden_size,
        lr=cfg.lr,
        epochs=cfg.epochs,
        use_ema=cfg.use_ema,
        grad_acc_steps=cfg.grad_acc_steps,
        output_dir=cfg.output_dir,
        checkpoint_path=cfg.checkpoint_path,
        save_checkpoint_steps_freq=cfg.save_checkpoint_steps_freq,
        save_model_freq_epochs=cfg.save_model_freq_epochs,
        config_dict=config_dict
    )

def run_debug():
    os.environ["WANDB_MODE"] = "offline"    # to not include testing runs

    debug_config = {
        "data_path": "data/data.csv",
        "transformer_path": "outputs_debug/models/transformer_debug.pt",
        "batch_size": 4, # Smaller batch for fast testing
        "hidden_size": 64,
        "lr": 0.001,
        "epochs": 2, # Only few epochs to test
        "use_ema": True,
        "grad_acc_steps": 1,
        "output_dir": "outputs_debug",
        "checkpoint_path": "outputs_debug/checkpoint-3800",
        "save_checkpoint_steps_freq": 200,
        "save_model_freq_epochs": 1,
    }

    print("=== RUNNING IN DEBUG MODE ===")
    train(**debug_config, config_dict=debug_config.copy())


if __name__ == '__main__':
    DEBUG_MODE = False

    if DEBUG_MODE:
        run_debug()
    else:
        run_with_hydra()
