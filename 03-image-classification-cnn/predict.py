import torch
import os
import csv
from torch.utils.data import DataLoader
from torchvision import transforms

from model import CNNImageClassifier
from dataset import TestDataset


MEAN = [0.5209, 0.4955, 0.4384]
STD = [0.2111, 0.2103, 0.2101]
IMAGE_SIZE = 64
CONV_CHANNELS = [32, 64, 128, 256]
USE_RES_CONNECTION = True
N_CLASSES = 50


def generate_predictions(model_path, test_dir, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    test_dataset = TestDataset(test_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNNImageClassifier(
        in_channels=3,
        n_classes=N_CLASSES,
        conv_channels=CONV_CHANNELS,
        use_res_connection=USE_RES_CONNECTION
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            outputs = model(images)

            _, predicted_classes = torch.max(outputs, 1)

            for filename, predicted_class in zip(filenames, predicted_classes):
                predictions.append([filename, predicted_class.item()])

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} to file: {output_csv}")


if __name__ == '__main__':
    MODEL_PATH = "models/model-20260415_181017.pt"

    TEST_DIR = "test/"
    OUTPUT_CSV = "pred.csv"

    if not os.path.exists(MODEL_PATH):
        print(f"File'{MODEL_PATH}' not existing")
    else:
        generate_predictions(MODEL_PATH, TEST_DIR, OUTPUT_CSV)
