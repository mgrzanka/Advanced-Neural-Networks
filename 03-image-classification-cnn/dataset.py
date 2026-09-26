import torch
import os
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from collections import Counter
from PIL import Image


class ImageClassifierDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(test_dir)
            if f.endswith('.JPEG') or f.endswith('.jpg')
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_dataloaders(train_transform, mean, std, image_size, data_dir="data/train/", batch_size=32, seed=42):
    initial_dataset = torchvision.datasets.ImageFolder(data_dir)

    # EDA showed unbalanced classes
    class_counts = Counter(initial_dataset.targets)
    num_classes = len(initial_dataset.classes)
    total_samples = sum(class_counts.values())

    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Transformations, datasets and dataloaders
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_size = int(0.8 * len(initial_dataset))
    val_size = len(initial_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(initial_dataset, [train_size, val_size], generator=generator)

    train_dataset = ImageClassifierDataset(train_subset, train_transform)
    val_dataset = ImageClassifierDataset(val_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_weights_tensor, initial_dataset.class_to_idx
