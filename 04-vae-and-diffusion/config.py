import torchvision.transforms as transforms

from vae_model import LitVAE, VAEModel
from diffusion_model import DiffusionModel, LiDiffusion


shared_train_transforms = [
    transforms.RandomCrop(32, padding=2, padding_mode='edge'),
    transforms.ToTensor(),
]
shared_test_transforms = [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
]

vae_config = {
    "name": "vae",
    "model_class": VAEModel,
    "lightning_model_class": LitVAE,
    "latent_dim": (128, 8, 8),
    "model_args": {
        "image_channels": 3,
        "block_out_channels": (32, 64, 128)
    },
    "lightning_model_class_args": {
        "num_epochs": 500,
    },
    "train_transform": transforms.Compose(shared_train_transforms),
    "test_transform": transforms.Compose(shared_test_transforms)
}

diffusion_config_big_model = {
    "name": "diffusion",
    "model_class": DiffusionModel,
    "lightning_model_class": LiDiffusion,
    "model_args": {
        "image_channels": 3,
        "block_out_channels": (64, 128, 256),
        "time_emb_dim": 256,
        "use_attention": True,
    },
    "lightning_model_class_args": {
        "num_epochs": 500,
        "diffusion_steps": 1000
    },
    "train_transform": transforms.Compose([
        *shared_train_transforms,
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "test_transform": transforms.Compose([
        *shared_test_transforms,
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
}

diffusion_config_small_model = {
    "name": "diffusion",
    "model_class": DiffusionModel,
    "lightning_model_class": LiDiffusion,
    "model_args": {
        "image_channels": 3,
        "block_out_channels": (32, 64, 128),
        "time_emb_dim": 128,
        "use_attention": False
    },
    "lightning_model_class_args": {
        "num_epochs": 500,
        "diffusion_steps": 1000
    },
    "train_transform": transforms.Compose([
        *shared_train_transforms,
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "test_transform": transforms.Compose([
        *shared_test_transforms,
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
}
