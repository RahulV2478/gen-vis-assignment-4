import torch

@torch.no_grad()
def ddim_sample(
    diffusion,
    num_samples=16,
    num_steps=50,
):
    """
    Deterministic DDIM sampling (eta = 0) using the trained diffusion model.

    Args:
        diffusion: a trained DiffusionProcess instance (with .model in eval mode).
        num_samples: how many images to generate.
        num_steps: number of DDIM steps (e.g., 50, 100, etc.).

    Returns:
        x_0 samples: (num_samples, C, H, W) in the same scale as diffusion.sample().
    """
    device = diffusion.device
    model = diffusion.model.eval()
    T = diffusion.noise_steps

    # Start from pure noise
    x = torch.randn(
        num_samples,
        diffusion.channels,
        diffusion.image_size,
        diffusion.image_size,
        device=device,
    )

    # Choose a sequence of timesteps from T-1 down to 0 with num_steps points
    t_seq = torch.linspace(T - 1, 0, steps=num_steps, dtype=torch.long, device=device)

    for i, t in enumerate(t_seq):
        t_int = int(t.item())
        t_batch = torch.full((num_samples,), t_int, device=device, dtype=torch.long)

        # Predict noise at this step
        eps_theta = model(x, t_batch)

        # Get alpha_hat for this and the next step
        alpha_hat_t = diffusion.alpha_cumprod[t_int]
        sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - alpha_hat_t)

        # Predict x0 from current xt and predicted noise
        x0_pred = (x - sqrt_one_minus_alpha_hat_t * eps_theta) / sqrt_alpha_hat_t

        # If this is the last step in the schedule, return x0_pred
        if i == num_steps - 1:
            x = x0_pred
            break

        # Otherwise, compute the next timestep (coarser step)
        t_next_int = int(t_seq[i + 1].item())
        alpha_hat_next = diffusion.alpha_cumprod[t_next_int]

        # Deterministic DDIM update (eta = 0)
        x = torch.sqrt(alpha_hat_next) * x0_pred + torch.sqrt(1.0 - alpha_hat_next) * eps_theta

    return x