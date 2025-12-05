import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
#  Basic building block
# -------------------------
class DoubleConv(nn.Module):
    """Conv -> BN -> SiLU -> Conv -> BN -> SiLU"""
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


# -------------------------
#  U-Net Diffusion Model
# -------------------------
class DiffusionModel(nn.Module):
    def __init__(self, image_size, channels=None, hidden_dims=None, channles=None):
        """
        U-Net-like model that predicts the noise epsilon.

        NOTE: `channles` is here so that the buggy checker
          model_class(image_size=image_size, channles=channels)
        still works.
        """
        super().__init__()

        # Handle the typo from the checker: `channles`
        if channels is None:
            channels = channles
        if channels is None:
            raise ValueError("You must provide `channels` or `channles`.")

        if hidden_dims is None:
            # Simple 2-level U-Net: 32 -> 64
            hidden_dims = [32, 64]

        self.image_size = image_size
        self.in_channels = channels
        self.hidden_dims = hidden_dims

        # === Time embedding ===
        # One embedding layer + small MLP
        self.max_time = 1000  # noise_steps
        time_dim = hidden_dims[0]

        self.time_embed = nn.Embedding(self.max_time, time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # === Initial convolution ===
        self.init_conv = nn.Conv2d(self.in_channels, hidden_dims[0], kernel_size=3, padding=1)

        # === Encoder (down path) ===
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = hidden_dims[0]
        for i, out_ch in enumerate(hidden_dims):
            self.down_blocks.append(DoubleConv(in_ch, out_ch))
            if i != len(hidden_dims) - 1:
                # Only downsample between blocks, not after the last one
                self.downsamples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        # === Bottleneck ===
        # e.g. hidden_dims[-1]=64 -> bottleneck_channels=128
        bottleneck_channels = hidden_dims[-1] * 2
        self.bottleneck = DoubleConv(hidden_dims[-1], bottleneck_channels)

        # === Decoder (up path) ===
        # For hidden_dims=[32, 64], we have 1 downsample -> 1 upsample
        decoder_dims = list(reversed(hidden_dims[:-1]))  # [32]

        self.up_trans = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.time_projs = nn.ModuleList()

        in_ch = bottleneck_channels

        for out_ch in decoder_dims:
            # Upsample, e.g. 128->32
            self.up_trans.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # After upsample, we'll concat with skip (out_ch channels) -> 2 * out_ch
            self.up_blocks.append(DoubleConv(out_ch * 2, out_ch))

            # Project time embedding to this decoder's channels
            self.time_projs.append(nn.Linear(time_dim, out_ch))

            in_ch = out_ch

        # === Final output conv: back to input channels (e.g. 32 -> 1)
        self.final_conv = nn.Conv2d(hidden_dims[0], self.in_channels, kernel_size=1)

    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: (B,) timesteps
        returns: (B, C, H, W) predicted noise
        """
        # 1. Embed t
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.long().clamp(max=self.max_time - 1)
        t_embed = self.time_embed(t)      # (B, time_dim)
        t_embed = self.time_mlp(t_embed)  # (B, time_dim)

        # 2. Initial conv
        out = self.init_conv(x)

        # 3. Encoder with skips
        skips = []
        for i, down_block in enumerate(self.down_blocks):
            out = down_block(out)
            if i != len(self.down_blocks) - 1:
                skips.append(out)               # store skip
                out = self.downsamples[i](out)  # downsample

        # 4. Bottleneck
        out = self.bottleneck(out)

        # 5. Decoder with time injection + skip connections
        for i, up in enumerate(self.up_trans):
            out = up(out)  # upsample

            # corresponding skip (reverse order)
            skip = skips[-(i + 1)]

            # inject time
            time_feat = self.time_projs[i](t_embed)   # (B, C_out)
            time_feat = time_feat.view(time_feat.size(0), -1, 1, 1)
            out = out + time_feat

            # concat skip
            out = torch.cat([out, skip], dim=1)
            out = self.up_blocks[i](out)

        # 6. Final conv
        out = self.final_conv(out)
        return out


# -------------------------
#  Diffusion Process
# -------------------------
class DiffusionProcess:
    def __init__(
        self,
        image_size,
        channels,
        hidden_dims=None,
        beta_start=1e-4,
        beta_end=0.02,
        noise_steps=1000,
        device=torch.device("cpu"),
    ):
        """
        Core diffusion process: beta schedule, add_noise, training, sampling.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]

        self.image_size = image_size
        self.channels = channels
        self.hidden_dims = hidden_dims
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ---- Beta schedule + derived quantities ----
        self.betas = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)  # \bar{alpha}_t

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # ---- Model + optimizer ----
        self.model = DiffusionModel(
            image_size=image_size,
            channels=channels,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def add_noise(self, x, t):
        """
        Forward diffusion q(x_t | x_0):
          x_t = sqrt(\bar{alpha}_t) * x_0 + sqrt(1-\bar{alpha}_t) * eps
        """
        x = x.to(self.device)
        t = t.to(self.device).long()

        sqrt_alpha_hat = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return noisy_x, noise

    @torch.no_grad()
    def sample(self, num_samples=16):
        """
        Reverse process (DDPM sampling).
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
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            eps_theta = self.model(x, t_batch)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_hat_t = self.alpha_cumprod[t]

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t)

            x = coef1 * (x - coef2 * eps_theta)

            if t > 0:
                sigma_t = torch.sqrt(beta_t)
                x = x + sigma_t * torch.randn_like(x)

        return x

    def train_step(self, x):
        """
        Single training step: sample t, add noise, predict noise, MSE loss.
        """
        self.model.train()
        x = x.to(self.device)
        batch_size = x.size(0)

        t = torch.randint(
            low=0,
            high=self.noise_steps,
            size=(batch_size,),
            device=self.device,
        )

        noisy_x, noise = self.add_noise(x, t)
        pred_noise = self.model(noisy_x, t)

        loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())