import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import dem_dataset


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim + cond_dim, out_channels),
        )

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t, c):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        tc = torch.cat([t, c], dim=1)
        time_cond = self.time_mlp(tc).unsqueeze(-1).unsqueeze(-1)
        h = h + time_cond

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.block1 = ResBlock(in_channels, out_channels, time_emb_dim, cond_dim)
        self.block2 = ResBlock(out_channels, out_channels, time_emb_dim, cond_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t, c):
        x = self.block1(x, t, c)
        x = self.block2(x, t, c)
        return self.pool(x), x


class UpBlock(nn.Module):
    def __init__(
        self, up_in_channels, skip_channels, out_channels, time_emb_dim, cond_dim
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(up_in_channels, out_channels, 2, stride=2)
        self.block1 = ResBlock(
            out_channels + skip_channels, out_channels, time_emb_dim, cond_dim
        )
        self.block2 = ResBlock(out_channels, out_channels, time_emb_dim, cond_dim)

    def forward(self, x, skip, t, c):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t, c)
        x = self.block2(x, t, c)
        return x


class DEMDiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        cond_dim=4,
        base_channels=64,
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.down1 = DownBlock(in_channels, base_channels, time_emb_dim, cond_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim, cond_dim)
        self.down3 = DownBlock(
            base_channels * 2, base_channels * 4, time_emb_dim, cond_dim
        )
        self.down4 = DownBlock(
            base_channels * 4, base_channels * 8, time_emb_dim, cond_dim
        )

        self.middle = ResBlock(
            base_channels * 8, base_channels * 8, time_emb_dim, cond_dim
        )

        self.up3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            time_emb_dim,
            cond_dim,
        )
        self.up2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            time_emb_dim,
            cond_dim,
        )
        self.up1 = UpBlock(
            base_channels * 2, base_channels, base_channels, time_emb_dim, cond_dim
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, 3),
        )

    def forward(self, x, t, c):
        t_emb = self.time_embedding(t)

        d1_out, skip1 = self.down1(x, t_emb, c)
        d2_out, skip2 = self.down2(d1_out, t_emb, c)
        d3_out, skip3 = self.down3(d2_out, t_emb, c)
        d4_out, _ = self.down4(d3_out, t_emb, c)

        mid = self.middle(d4_out, t_emb, c)

        up3_out = self.up3(mid, skip3, t_emb, c)
        up2_out = self.up2(up3_out, skip2, t_emb, c)
        up1_out = self.up1(up2_out, skip1, t_emb, c)

        denoised = self.final_conv(up1_out)
        aux_out = self.aux_head(up1_out)

        return denoised, aux_out


def compute_slope_map(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B, 1, H, W-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B, 1, H-1, W]
    dx = F.pad(dx, (0, 1, 0, 0))  # pad right
    dy = F.pad(dy, (0, 0, 0, 1))  # pad bottom
    slope = torch.sqrt(dx**2 + dy**2 + 1e-8)
    return slope


# Total Variation Loss
def total_variation(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def cosine_beta_schedule(timesteps, s=0.004):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def load_model(save_model, model, dataset, learning_rate, device, cond_dim):
    if isinstance(save_model, str) and os.path.exists(save_model):
        model_path = save_model
        save_model = True
    else:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join(
            "models", f"dem_diffusion-T{dataset.tile_size}-A{cond_dim}.pth"
        )
    if save_model:
        print(f"Model will be saved to {model_path}")

    if isinstance(model, nn.Module):
        model.to(device)
    elif isinstance(model, str) and os.path.exists(model):
        model = DEMDiffusionModel(cond_dim=cond_dim)
        model.load_state_dict(torch.load(model, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    elif isinstance(model, str) and model == "load":
        model = DEMDiffusionModel(cond_dim=cond_dim)
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Best model not found at {model_path}, starting from scratch")
    elif isinstance(model, str) and model == "new":
        model = DEMDiffusionModel(cond_dim=cond_dim)

    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimiser, model_path


def train_demdiffusionmodel(
    model,
    dataset,
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=20,
    subset_fraction=1.0,
    num_timesteps=100,
    save_model=True,
    device=(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    dem_loss_weight=1.0,
    stats_loss_weight=0.01,
    stats_dim=4,
    # data loader params
    num_workers=8,
    pin_memory=True,
):
    if subset_fraction < 1.0:
        num_epochs = round(round(num_epochs / subset_fraction))
        print(
            f"Using {subset_fraction:.0%} of the dataset, training extended to {num_epochs} epochs"
        )

    print(f"Using device: {device}")

    num_workers = 0 if dataset.preload_to_ram else num_workers
    persistent_workers = not dataset.preload_to_ram
    num_samples = max(int(len(dataset) * subset_fraction), 1)

    model, optimiser, model_path = load_model(
        save_model, model, dataset, learning_rate, device, cond_dim=stats_dim
    )

    loss_prev = np.inf

    betas = cosine_beta_schedule(num_timesteps)
    alphas = np.insert(np.cumprod(1.0 - betas), 0, 1.0)  # so index 1 corresponds to t=1
    alphas = torch.tensor(alphas, dtype=torch.float32, device=device)

    for epoch in range(num_epochs):
        indices = random.sample(range(len(dataset)), num_samples)
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model.train()
        epoch_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            dem, stats = batch
            dem = dem.to(device)
            stats = stats.to(device)

            # 1. Sample timestep
            timestep = torch.randint(1, num_timesteps, (dem.size(0),), device=device)
            alpha = alphas[timestep].view(-1, 1, 1, 1)

            # 2. Forward diffusion
            noise = torch.randn_like(dem)
            dem_noisy = alpha.sqrt() * dem + (1 - alpha).sqrt() * noise

            # 3. Predict noise + aux
            t = timestep.float() / num_timesteps
            noise_pred, stats_pred = model(dem_noisy, t, stats)

            # 4. Noise prediction loss (standard DDPM)
            noise_loss = F.mse_loss(noise_pred, noise).sqrt()

            # 5. Auxiliary constraint losses
            min_pred, mean_pred, max_pred = (
                stats_pred[:, 1],
                stats_pred[:, 0],
                stats_pred[:, 2],
            )

            min_pred = torch.clamp(min_pred, min=0.0)
            mean_pred = torch.clamp(mean_pred, min=min_pred, max=max_pred)
            stats_pred = torch.stack([mean_pred, min_pred, max_pred], dim=1)

            # 6. Auxiliary regression loss (match stats)
            stats_loss = F.mse_loss(stats_pred, stats[:, :3]).sqrt()

            # 7. Reconstruct x0 from predicted noise
            alpha_ref = 0.5
            dem_pred_ref = (dem_noisy - np.sqrt(alpha_ref) * noise_pred) / np.sqrt(
                alpha_ref
            )
            dem_pred_ref = torch.clamp(dem_pred_ref, -1.0, 1.0)  # optional safety

            dem_loss = F.mse_loss(dem_pred_ref, dem).sqrt()

            # 8. Total loss
            loss = (
                noise_loss
                + dem_loss_weight * dem_loss
                + stats_loss_weight * 1e-2 * stats_loss
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        if epoch_loss > loss_prev * 1000:
            print(
                f"Epoch {int(np.ceil((epoch+1)*subset_fraction)):04d}/{int(np.ceil(num_epochs*subset_fraction)):04d}  Loss: {epoch_loss:.3f}. Loss diverged, back tracking."
            )
            model = DEMDiffusionModel(cond_dim=stats_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        else:
            loss_prev = epoch_loss
            if save_model:
                torch.save(
                    model.state_dict(),
                    model_path,
                )
                print(
                    f"Epoch {int(np.ceil((epoch+1)*subset_fraction)):04d}/{int(np.ceil(num_epochs*subset_fraction)):04d}  Loss: {epoch_loss:.3f}. Model saved"
                )
            else:
                print(
                    f"Epoch {int(np.ceil((epoch+1)*subset_fraction)):04d}/{int(np.ceil(num_epochs*subset_fraction)):04d}  Loss: {epoch_loss:.3f}"
                )
    return model


def normalise_batch(x, mean, min_val, max_val, eps=1e-5):
    """
    Normalize batched DEM using per-sample mean/min/max from cond.
    x: [B, 1, H, W]
    mean, min_val, max_val: [B, 1, 1, 1]
    Returns: [B, 1, H, W]
    """
    denom = torch.clamp(max_val - min_val, min=eps)
    x_norm = (x - mean) / denom
    return x_norm


def denormalise_batch(x_norm, mean, min_val, max_val):
    denom = torch.clamp(max_val - min_val, min=1e-5)
    return x_norm * denom + mean
