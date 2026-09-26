import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from datetime import date

from model import NeuralNetwork


BATCH_SIZE = 32
TEST_PATH = 'data/test_data.csv'
OUTPUT_PATH = "pred.csv"

CONFIG = {
    "experiment_name": f"00_Baseline",
    "dropout_p": 0.3,
    "network_sizes": [128, 64],
    "use_batch_norm": True,
}

def load_data(transformator_path):
    df = pd.read_csv(TEST_PATH)
    df["Age"] = date.today().year - df["YearBuilt"]
    transformator = joblib.load(transformator_path)
    X_test_cont, X_test_cat = transformator.transform(df)

    test_dataset = TensorDataset(
        torch.tensor(X_test_cont), torch.tensor(X_test_cat)
    )
    dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    emb_dims = []
    if X_test_cat is not None and X_test_cat.shape[1] > 0:
        for encoder in transformator.discrete_transformer.transformers_[1][1].categories_:
            num_classes = len(encoder)
            emb_dim = max(1, min(50, num_classes // 2))
            emb_dims.append((num_classes, emb_dim))
    else:
        emb_dims = None

    input_size_cont = X_test_cont.shape[1]

    return dataloader, emb_dims, input_size_cont


def sample(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"models/{config['experiment_name']}/model.pt"
    transformator_path = f"models/{config['experiment_name']}/transformator.pt"

    dataloader, emb_dims, input_size_cont = load_data(transformator_path)

    model = NeuralNetwork(
        output_size=3,
        network_sizes=config["network_sizes"],
        input_size_cont=input_size_cont,
        emb_dims=emb_dims,
        p=config["dropout_p"],
        use_batch_norm=config["use_batch_norm"]
    ).to(device)
    model_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)

    all_predictions = []

    with torch.no_grad():
        model.eval()

        for batch_cont, batch_cat in dataloader:
            X_test_cont, X_test_cat = batch_cont.to(device), batch_cat.to(device)

            out = model(X_test_cont, X_test_cat)
            predictions = torch.argmax(out, dim=1).cpu().numpy()
            all_predictions.extend(predictions)

    final_output = np.array(all_predictions)
    pd.DataFrame(final_output).to_csv(OUTPUT_PATH, header=False, index=False)
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    sample(CONFIG)
