import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from diffusion import DiffusionProcess


def get_cifar10_dataloader(batch_size=128, image_size=32):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # map [0,1] -> [-1,1] per channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    return loader


def train_diffusion_cifar(
    num_epochs=40,
    batch_size=128,
    noise_steps=1000,
    device=None,
    sample_every=5,
    num_sample_images=25,
    out_dir="outputs_cifar",
    beta_schedule="cosine",
):
    """
    Train the diffusion model on CIFAR-10 in RGB (3x32x32).
    Returns:
        diffusion (DiffusionProcess)
        loss_history (list of float)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(out_dir, exist_ok=True)

    dataloader = get_cifar10_dataloader(batch_size=batch_size, image_size=32)

    diffusion = DiffusionProcess(
        image_size=32,
        channels=3,
        hidden_dims=[64, 128, 256],   # deeper U-Net for CIFAR
        noise_steps=noise_steps,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule=beta_schedule,
        device=device,
    )

    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for x, _ in dataloader:
            loss = diffusion.train_step(x)
            running_loss += loss

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[CIFAR Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.4f}")

        if (epoch + 1) % sample_every == 0:
            diffusion.model.eval()
            with torch.no_grad():
                samples = diffusion.sample(num_samples=num_sample_images)
                # map [-1,1] -> [0,1]
                samples = (samples.clamp(-1, 1) + 1) / 2.0
                save_path = os.path.join(out_dir, f"samples_epoch_{epoch+1}.png")
                save_image(samples, save_path, nrow=int(num_sample_images ** 0.5))
                print(f"Saved CIFAR samples to {save_path}")

    model_path = os.path.join(out_dir, "diffusion_cifar10.pth")
    torch.save(diffusion.model.state_dict(), model_path)
    print(f"Saved CIFAR diffusion model to {model_path}")

    return diffusion, loss_history