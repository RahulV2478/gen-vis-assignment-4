import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple "double conv" building block used in the U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionProcess:
    def __init__(
        self,
        image_size,
        channels,
        hidden_dims=[32, 64, 128],
        beta_start=1e-4,
        beta_end=0.02,
        noise_steps=1000,
        device=torch.device("cpu"),
    ):
        """
        Initialize the diffusion process.
        Args:
            beta_start: Initial noise variance
            beta_end: Final noise variance
            noise_steps: Number of diffusion steps
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.hidden_dims = hidden_dims
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps

        # Choose device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # === Beta schedule and derived quantities ===
        # Linear schedule from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)  # \bar{alpha}_t

        # Precompute useful terms
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # === Model and optimizer ===
        self.model = DiffusionModel(
            image_size=image_size,
            channels=channels,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Adam optimizer with a reasonable default LR
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def add_noise(self, x, t):
        """
        Add noise to the input images according to the diffusion process.
        Args:
            x: Clean images tensor of shape [batch_size, channels, height, width]
            t: Timesteps tensor of shape [batch_size]
        Returns:
            Tuple of (noisy_images, noise)
        """
        x = x.to(self.device)
        t = t.to(self.device).long()

        # Gather the correct alpha_cumprod for each sample
        # shapes: (batch,) -> (batch, 1, 1, 1)
        sqrt_alpha_hat = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        # Gaussian noise
        noise = torch.randn_like(x)

        # Forward diffusion equation:
        # x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * eps
        noisy_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

        return noisy_x, noise

    @torch.no_grad()
    def sample(self, num_samples=16):
        """
        Generate new samples by reversing the diffusion process (DDPM sampling).
        Args:
            num_samples: Number of samples to generate
        Returns:
            Generated images tensor of shape [num_samples, channels, H, W]
        """
        self.model.eval()
        x = torch.randn(
            num_samples,
            self.channels,
            self.image_size,
            self.image_size,
            device=self.device,
        )

        for t in reversed(range(self.noise_steps)):
            t_batch = torch.full(
                (num_samples,),
                t,
                device=self.device,
                dtype=torch.long,
            )

            # Predict noise at this timestep
            eps_theta = self.model(x, t_batch)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_hat_t = self.alpha_cumprod[t]

            # DDPM update:
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_hat_t)*eps_theta) + sigma_t * z
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t)

            x = coef1 * (x - coef2 * eps_theta)

            if t > 0:
                # Add noise except at t=0
                sigma_t = torch.sqrt(beta_t)
                x = x + sigma_t * torch.randn_like(x)

        return x

    def train_step(self, x):
        """
        Perform one training step for the diffusion model.
        Args:
            x: Clean images tensor of shape [batch_size, channels, height, width]
        Returns:
            Loss value for the step (Python float)
        """
        self.model.train()
        x = x.to(self.device)
        batch_size = x.size(0)

        # 1. Sample random timesteps
        t = torch.randint(
            low=0,
            high=self.noise_steps,
            size=(batch_size,),
            device=self.device,
        )

        # 2. Add noise to images
        noisy_x, noise = self.add_noise(x, t)

        # 3. Predict the noise using the model
        pred_noise = self.model(noisy_x, t)

        # 4. MSE loss between predicted and actual noise
        loss = F.mse_loss(pred_noise, noise)

        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6. Return loss as a Python float
        return float(loss.item())


class DiffusionModel(nn.Module):
    def __init__(self, image_size, channles=None, channels=None, hidden_dims=[32, 64, 128]):
        """
        Initialize the diffusion model (U-Net style).
        NOTE: We accept `channles` to match the typo in the provided check function.
        Args:
            image_size: Height/width of the image (e.g., 28 for MNIST)
            channles/channels: Number of input channels (1 for MNIST)
            hidden_dims: List of feature map sizes for each level
        """
        super().__init__()

        # Handle the typo from the checker: `channles`
        if channels is None:
            channels = channles
        if channels is None:
            raise ValueError("You must provide `channels` or `channles`.")

        self.image_size = image_size
        self.in_channels = channels
        self.hidden_dims = hidden_dims

        # === Time embedding ===
        # Use a single embedding layer for timesteps, then an MLP
        self.max_time = 1000  # matches the assignment's noise_steps
        time_dim = hidden_dims[0]

        self.time_embed = nn.Embedding(self.max_time, time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # === Initial convolution ===
        self.init_conv = nn.Conv2d(self.in_channels, hidden_dims[0], kernel_size=3, padding=1)

        # === Encoder (downsampling path) ===
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = hidden_dims[0]
        for i, out_ch in enumerate(hidden_dims):
            self.down_blocks.append(DoubleConv(in_ch, out_ch))
            if i != len(hidden_dims) - 1:
                # Downsample between encoder blocks
                self.downsamples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        # === Bottleneck ===
        bottleneck_channels = hidden_dims[-1] * 2
        self.bottleneck = DoubleConv(hidden_dims[-1], bottleneck_channels)

        # === Decoder (upsampling path) ===
        self.up_trans = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.time_projs = nn.ModuleList()

        decoder_dims = list(reversed(hidden_dims))
        in_ch = bottleneck_channels

        for out_ch in decoder_dims:
            # Transposed conv for upsampling
            self.up_trans.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # After upsample, we concat skip connection -> channels double
            self.up_blocks.append(DoubleConv(out_ch * 2, out_ch))

            # Time projection for this decoder level
            self.time_projs.append(nn.Linear(time_dim, out_ch))

            in_ch = out_ch

        # === Final output layer ===
        self.final_conv = nn.Conv2d(hidden_dims[0], self.in_channels, kernel_size=1)

    def forward(self, x, t):
        """
        Forward pass through the U-Net model.
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            t: Timesteps tensor of shape [batch_size]
        Returns:
            Tensor of shape [batch_size, channels, height, width]
        """
        # 1. Embed the timestep
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.long()
        t = t.clamp(max=self.max_time - 1)  # safety, in case
        t_embed = self.time_embed(t)        # (B, time_dim)
        t_embed = self.time_mlp(t_embed)    # (B, time_dim)

        # 2. Initial convolution
        x = self.init_conv(x)

        # 3. Encoder with skip connections
        skips = []
        out = x
        for i, down_block in enumerate(self.down_blocks):
            out = down_block(out)
            if i != len(self.down_blocks) - 1:
                skips.append(out)
                out = self.downsamples[i](out)

        # 4. Bottleneck
        out = self.bottleneck(out)

        # 5. Decoder with time injection + skip connections
        for i, up in enumerate(self.up_trans):
            out = up(out)

            # Retrieve corresponding skip (reverse order)
            skip = skips[-(i + 1)]

            # Inject time information by addition
            time_feat = self.time_projs[i](t_embed)  # (B, C_out)
            time_feat = time_feat.view(time_feat.size(0), -1, 1, 1)
            out = out + time_feat

            # Concatenate skip and pass through conv block
            out = torch.cat([out, skip], dim=1)
            out = self.up_blocks[i](out)

        # 6. Final output conv
        out = self.final_conv(out)
        return out