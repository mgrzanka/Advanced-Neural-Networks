import torch
import pandas as pd
import os
import numpy as np
import joblib
from torch.utils.data import TensorDataset, DataLoader
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from accelerate.logging import get_logger
import hydra
from omegaconf import DictConfig

from model import NeuralNetwork


logger = get_logger(__name__, "INFO")


def load_data(test_data_path, transformer_path):
    test_df = pd.read_csv(test_data_path)
    transformer = joblib.load(transformer_path)
    X_test = transformer.transform(test_df).astype(np.float32)
    return X_test


def get_dataloader(X_train, batch_size):
    dataset = TensorDataset(torch.tensor(X_train))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def sample(batch_size, test_data_path, transformer_path, model_path, hidden_size, output_dir):
    X_test = load_data(test_data_path, transformer_path)
    input_size = X_test.shape[1]
    dataloader = get_dataloader(X_test, batch_size)

    model_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = NeuralNetwork(input_size, hidden_size, 1)
    ema_model = EMAModel(
        parameters=model.parameters(),
        decay=0.99,
        model_cls=type(model)
    )
    ema_model.load_state_dict(model_state_dict)

    model.eval()
    ema_model.copy_to(model.parameters())

    accelerator = Accelerator(project_dir=output_dir)
    model, dataloader = accelerator.prepare(model, dataloader)

    all_preds = []

    for batch in dataloader:
        X = batch[0]

        with torch.no_grad():
            out = model(X)

        out_gathered = accelerator.gather_for_metrics((out))
        all_preds.extend(out_gathered.cpu().numpy().flatten())

    if accelerator.is_main_process:
        results_save_path = os.path.join(output_dir, "results.csv")
        os.makedirs(output_dir, exist_ok=True)

        results_df = pd.DataFrame(all_preds, columns=["prediction"])
        results_df.to_csv(results_save_path, index=False, header=False)
        logger.info(f"Results saved in {results_save_path}")


@hydra.main(version_base=None, config_path="config", config_name="sample")
def run_with_hydra(cfg: DictConfig):
    sample(
        batch_size=cfg.batch_size,
        test_data_path=cfg.test_data_path,
        transformer_path=cfg.transformer_path,
        model_path=cfg.model_path,
        hidden_size=cfg.hidden_size,
        output_dir=cfg.output_dir
    )


def run_debug():
    debug_config = {
        "batch_size": 32,
        "test_data_path": "data/evaluation_data.csv",
        "transformer_path": "outputs/models/transformer.pt",
        "model_path": "outputs/epoch-2/ema_model.pt",
        "hidden_size": 64,
        "output_dir": "outputs_test_debug"
    }

    print("=== RUNNING SAMPLING IN DEBUG MODE ===")
    sample(**debug_config)


if __name__ == '__main__':
    DEBUG_MODE = False

    if DEBUG_MODE:
        run_debug()
    else:
        run_with_hydra()
