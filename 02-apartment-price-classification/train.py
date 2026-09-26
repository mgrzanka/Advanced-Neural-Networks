import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch import nn
import numpy as np
import pandas as pd
from datetime import date
import wandb
import os
from datetime import datetime

from data_preprocessing import DataTransformator
from model import NeuralNetwork
from evaluation import calc_accuracy_training


DATA_PATH = 'data/train_data.csv'
BATCH_SIZE=32
LR = 0.001
EPOCHS=100


def load_data(data_path, transformator_path, use_one_hot):
    df = pd.read_csv(data_path)
    df["Age"] = date.today().year - df["YearBuilt"]

    scale_cols = ["Age", "Size(sqf)", 'Floor', 'N_Parkinglot(Ground)', 'N_Parkinglot(Basement)', 'N_elevators', 'N_FacilitiesNearBy(Total)', 'N_SchoolNearBy(Total)', 'N_manager', 'N_FacilitiesInApt']
    log_transform_cols = ["Size(sqf)"]
    one_hot_cols = [] if not use_one_hot else ['HallwayType', 'TimeToBusStop', 'TimeToSubway', 'SubwayStation']
    label_cols = ['HeatingType', 'AptManageType'] if use_one_hot else ['HeatingType', 'AptManageType', 'HallwayType', 'TimeToBusStop', 'TimeToSubway', 'SubwayStation']

    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df.drop(df_train.index)

    transformator = DataTransformator(log_transform_cols, scale_cols, one_hot_cols, label_cols)
    X_train_cont, X_train_cat = transformator.fit_transform(df_train, transformator_path)

    X_val_cont, X_val_cat = transformator.transform(df_val)

    def transform_target(df):
        conditions = [
            (df["SalePrice"] <= 100000),
            (df["SalePrice"] > 100000) & (df["SalePrice"] <= 350000),
            (df["SalePrice"] > 350000)
        ]
        choices = [0, 1, 2]
        return np.select(conditions, choices)

    y_train = transform_target(df_train)
    y_val = transform_target(df_val)

    return (X_train_cont, X_train_cat, y_train), (X_val_cont, X_val_cat, y_val)


def get_dataset_objects(data_path, transformator_path, use_one_hot, use_weighted_criterion):
    (X_train_cont, X_train_cat, y_train), (X_val_cont, X_val_cat, y_val) = \
        load_data(data_path, transformator_path, use_one_hot)

    emb_dims = []
    if X_train_cat is not None and X_train_cat.shape[1] > 0:
        for i in range(X_train_cat.shape[1]):
            num_classes = int(np.max(X_train_cat[:, i])) + 1
            emb_dim = max(1, min(50, num_classes // 2))
            emb_dims.append((num_classes, emb_dim))
    else:
        emb_dims = None

    criterion_weight = None
    if use_weighted_criterion:
        classes, counts = np.unique(y_train, return_counts=True)
        criterion_weight = torch.tensor(
            [len(y_train) / (len(classes) * c) for c in counts],
            dtype=torch.float32
        )

    train_dataset = TensorDataset(
        torch.tensor(X_train_cont, dtype=torch.float32),
        torch.tensor(X_train_cat, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val_cont, dtype=torch.float32),
        torch.tensor(X_val_cat, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    input_size_cont = X_train_cont.shape[1]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, input_size_cont, emb_dims, criterion_weight


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"models/{config['experiment_name']}/model.pt"
    transformator_path = f"models/{config['experiment_name']}/transformator.pt"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_loader, val_loader, input_size_cont, emb_dims, criterion_weight = get_dataset_objects(
        data_path=DATA_PATH,
        transformator_path=transformator_path,
        use_one_hot=config["use_one_hot"],
        use_weighted_criterion=config["use_weighted_criterion"]
    )

    model = NeuralNetwork(
        output_size=3,
        network_sizes=config["network_sizes"],
        input_size_cont=input_size_cont,
        emb_dims=emb_dims,
        p=config["dropout_p"],
        use_batch_norm=config["use_batch_norm"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config["weight_decay"])

    if criterion_weight is not None:
        criterion_weight = criterion_weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)

    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []

    time_now = datetime.now()
    wandb.init(
        project="Apartment-prices-classification",
        name=f"{config['experiment_name']}-{time_now}",
        config=config,
        reinit=True
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        running_training_loss = 0.0
        running_training_acc = 0.0
        training_samples = 0

        for x, cat_x, cls in train_loader:
            x, cat_x, cls = x.to(device), cat_x.to(device), cls.to(device)

            out = model(x, cat_x)

            loss = criterion(out, cls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calc_accuracy_training(out, cls)

            running_training_loss += loss.item() * cls.size(0)
            running_training_acc += acc * cls.size(0)
            training_samples += cls.size(0)

        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0
        val_samples = 0

        with torch.no_grad():
            for x, cat_x, cls in val_loader:
                x, cat_x, cls = x.to(device), cat_x.to(device), cls.to(device)

                out = model(x, cat_x)
                loss = criterion(out, cls)

                acc = calc_accuracy_training(out, cls)
                running_val_loss += loss.item() * cls.size(0)
                running_val_acc += acc * cls.size(0)
                val_samples += cls.size(0)

        training_loss_mean = running_training_loss / training_samples
        training_acc_mean = running_training_acc / training_samples
        val_loss_mean = running_val_loss / val_samples
        val_acc_mean = running_val_acc / val_samples

        train_losses.append(training_loss_mean)
        train_acc.append(training_acc_mean)
        val_losses.append(val_loss_mean)
        val_acc.append(val_acc_mean)

        print(f"Epoch {epoch} | Training loss {training_loss_mean:.3f} | Val acc: {val_acc_mean:.3f}")

        wandb.log({
            "train/loss": training_loss_mean,
            "train/acc": training_acc_mean,
            "val/loss": val_loss_mean,
            "val/acc": val_acc_mean
        }, step=epoch)

        if val_acc_mean > best_val_acc:
            best_val_acc = val_acc_mean
            torch.save(model.state_dict(), model_path)
            print(f"Saved new, the best model. Acc: {val_acc_mean}")

    wandb.log_model(model_path, config['experiment_name'])

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

    wandb.finish()



if __name__ == '__main__':
    baseline = {
        "dropout_p": 0.3,
        "weight_decay": 1e-4,
        "network_sizes": [128, 64],
        "use_batch_norm": True,
        "use_one_hot": False,
        "use_weighted_criterion": True,
        "experiment_name": "00_Baseline"
    }
    train(baseline)
