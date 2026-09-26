"""Trening klasyfikatora obrazów (50 klas, 64x64).

Uruchomienie:
    python train.py

Dane oczekiwane są w `data/train/` w układzie ImageFolder (katalog na klasę).
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

from dataset import get_dataloaders
from model import CNNImageClassifier


BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001  # AdamW
N_CLASSES = 50
# mean, std i rozmiar obrazu wyznaczone w EDA
MEAN = [0.5209, 0.4955, 0.4384]
STD = [0.2111, 0.2103, 0.2101]
IMAGE_SIZE = 64

CONV_CHANNELS = [32, 64, 128, 256]
USE_RES_CONNECTION = True

# Mocna augmentacja - zbiór jest mały w stosunku do 50 klas.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(0.6),
])


def train(train_transform=TRAIN_TRANSFORM, conv_channels=CONV_CHANNELS,
          use_res_connection=USE_RES_CONNECTION, data_dir="data/train/"):
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join("models", f"model-{timestamp}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_weights, class_to_idx = get_dataloaders(
        train_transform, MEAN, STD, IMAGE_SIZE, data_dir=data_dir, batch_size=BATCH_SIZE
    )
    class_weights = class_weights.to(device)

    model = CNNImageClassifier(
        in_channels=3,
        n_classes=N_CLASSES,
        conv_channels=conv_channels,
        use_res_connection=use_res_connection,
    ).to(device)

    # Klasy są niezbalansowane (widać to w EDA), więc strata jest ważona.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0

    print("Training started")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        train_loss_avg = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val
        val_loss_avg = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
              f"Train Loss: {train_loss_avg:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} - Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print("Accuracy improved, saved new model")

    print(f"\nTraining finished. Best val acc: {best_val_acc:.4f} -> {model_save_path}")
    return model, class_to_idx


if __name__ == "__main__":
    train()
