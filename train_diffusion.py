import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from diffusion import DiffusionProcess


def get_mnist_dataloader(batch_size=128, image_size=28):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Map pixels from [0,1] -> [-1,1]
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    return dataloader


def train_diffusion(
    num_epochs=5,
    batch_size=128,
    noise_steps=1000,
    device=None,
    sample_every=1,
    num_sample_images=16,
    out_dir="outputs",
    beta_schedule="linear",
):
    """
    Train the diffusion model on MNIST.
    Returns:
        diffusion (DiffusionProcess)
        loss_history (list of float)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(out_dir, exist_ok=True)

    dataloader = get_mnist_dataloader(batch_size=batch_size, image_size=28)

    diffusion = DiffusionProcess(
        image_size=28,
        channels=1,
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
        print(f"[Epoch {epoch+1}/{num_epochs}] loss={avg_loss:.4f}")

        if (epoch + 1) % sample_every == 0:
            diffusion.model.eval()
            with torch.no_grad():
                samples = diffusion.sample(num_samples=num_sample_images)
                samples = (samples.clamp(-1, 1) + 1) / 2.0
                save_path = os.path.join(out_dir, f"samples_epoch_{epoch+1}.png")
                save_image(samples, save_path, nrow=int(num_sample_images ** 0.5))
                print(f"Saved samples to {save_path}")

    model_path = os.path.join(out_dir, "diffusion_mnist.pth")
    torch.save(diffusion.model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    return diffusion, loss_history