import os
from torchvision.utils import save_image
import torch

from config import diffusion_config_small_model, diffusion_config_big_model, vae_config


def generate_samples(config, checkpoint_path: str, num_samples=1000, batch_size=100):
    output_dir=f"generated_samples/{config['name']}"
    os.makedirs(output_dir, exist_ok=True)

    raw_model = config["model_class"](**config["model_args"])

    kwargs = {"vae_model": raw_model} if config["name"] == "vae" else {"diffusion_model": raw_model}
    kwargs.update(config["lightning_model_class_args"])
    lit_model = config["lightning_model_class"].load_from_checkpoint(
        checkpoint_path,
        **kwargs
    )
    model = lit_model.model   # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    generated_count = 0
    with torch.no_grad():

        while generated_count < num_samples:
            current_batch_size = min(batch_size, num_samples - generated_count)

            if config["name"] == "vae":
                latent_dim = config["latent_dim"]

                z = torch.randn(current_batch_size, *latent_dim).to(device)
                generated_images = model.decode(z)

                for i in range(current_batch_size):
                    filename = os.path.join(output_dir, f"sample_{generated_count + i:04d}.png")
                    save_image(generated_images[i], filename)

                generated_count += current_batch_size
                print(f"Generation for ({config['name']}): {generated_count}/{num_samples}")

            elif config["name"] == "diffusion":
                generated_images = lit_model.generate_batch(
                    batch_size=current_batch_size,
                    device=device
                )
                for i in range(current_batch_size):
                    filename = os.path.join(output_dir, f"sample_{generated_count + i:04d}.png")
                    save_image(
                        generated_images[i],
                        filename,
                        normalize=True,
                        value_range=(-1.0, 1.0)
                    )

                generated_count += current_batch_size
                print(f"Generation for ({config['name']}): {generated_count}/{num_samples}")

    print(f"Generated and saved {num_samples} images in directory: {output_dir}")


if __name__ == '__main__':
    checkpoint_path = 'checkpoints/diffusion/epoch=499-val_loss=120.45.ckpt'
    generate_samples(
        diffusion_config,
        checkpoint_path,
        num_samples=1000,
        batch_size=100,
    )

    checkpoint_path = 'checkpoints/vae/epoch=499-val_loss=120.45.ckpt'
    generate_samples(
        vae_config,
        checkpoint_path,
        num_samples=1000,
        batch_size=100,
    )
