import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter


class RoadSignsDataset(Dataset):
    def __init__(self, subset, transform) -> None:
        super().__init__()
        self.dataset = subset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_data(train_transform, val_transform, batch_size=64, seed=42):
    dataset = ImageFolder("data/trafic_32/")

    class_counts = Counter(dataset.targets)
    total_samples = sum(class_counts.values())

    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_dataset = RoadSignsDataset(train_subset, train_transform)
    val_dataset = RoadSignsDataset(val_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, dataset.class_to_idx
