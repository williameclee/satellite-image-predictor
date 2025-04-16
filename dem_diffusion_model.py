import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm


class DEMDiffusionModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128, cond_dim=4):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.down1 = DownBlock(in_channels, 64, time_emb_dim)
        self.down2 = DownBlock(64, 128, time_emb_dim)
        self.down3 = DownBlock(128, 256, time_emb_dim)
        self.down4 = DownBlock(256, 512, time_emb_dim)

        self.middle = ResBlock(512, 512, time_emb_dim)

        self.up3 = UpBlock(512, 256, 256, time_emb_dim)
        self.up2 = UpBlock(256, 128, 128, time_emb_dim)
        self.up1 = UpBlock(128, 64, 64, time_emb_dim)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, t, cond):
        t_emb = self.time_embedding(t)
        cond_emb = self.cond_mlp(cond)
        combined_emb = t_emb + cond_emb  # shape: [B, time_emb_dim]

        d1_out, skip1 = self.down1(x, combined_emb)
        d2_out, skip2 = self.down2(d1_out, combined_emb)
        d3_out, skip3 = self.down3(d2_out, combined_emb)
        d4_out, _ = self.down4(d3_out, combined_emb)

        mid = self.middle(d4_out, combined_emb)

        up3_out = self.up3(mid, skip3, combined_emb)
        up2_out = self.up2(up3_out, skip2, combined_emb)
        up1_out = self.up1(up2_out, skip1, combined_emb)

        return self.final_conv(up1_out)


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
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_out = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_out

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block1 = ResBlock(in_channels, out_channels, time_emb_dim)
        self.block2 = ResBlock(out_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return self.pool(x), x


class UpBlock(nn.Module):
    def __init__(self, up_in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(up_in_channels, out_channels, 2, stride=2)
        self.block1 = ResBlock(out_channels + skip_channels, out_channels, time_emb_dim)
        self.block2 = ResBlock(out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x


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
    l2loss_weight=1.0,
    tvloss_weight=0.01,
    slope_scale=5.0,
    cond_dim=4,
):
    if subset_fraction < 1.0:
        num_epochs = round(round(num_epochs / subset_fraction))
        print(
            f"Using {subset_fraction:.0%} of the dataset, training extended to {num_epochs} epochs"
        )

    print(f"Using device: {device}")

    model, optimiser, model_path = load_model(
        save_model, model, dataset, learning_rate, device, cond_dim=cond_dim
    )

    loss_prev = np.inf
    betas = cosine_beta_schedule(num_timesteps)
    alphas_cumprod = np.cumprod(1.0 - betas)

    for epoch in range(num_epochs):
        num_samples = int(len(dataset) * subset_fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
        )
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x0, cond = batch
            x0 = x0.to(device)
            cond = cond.to(device)

            t = torch.randint(1, num_timesteps, (x0.size(0),), device=device).long()
            alpha = torch.tensor(
                alphas_cumprod[t.cpu().numpy()], dtype=torch.float32, device=device
            ).view(-1, 1, 1, 1)

            noise = torch.randn_like(x0)
            xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise

            pred_noise = model(xt, t.float() / num_timesteps, cond)
            mse_loss = F.mse_loss(pred_noise, noise)

            with torch.no_grad():
                x0_pred = (xt - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
                slope = compute_slope_map(x0)
                slope_weight = 1.0 + slope_scale * slope
                weighted_l2 = ((x0_pred - x0) ** 2 * slope_weight).mean()

            tv_reg = total_variation(x0_pred)
            loss = mse_loss + l2loss_weight * weighted_l2 + tvloss_weight * tv_reg

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        if loss > loss_prev * 100:
            print(
                f"Epoch {round((epoch+1)*subset_fraction):04d}/{round(num_epochs*subset_fraction):04d}  Loss: {loss:.3f}. Loss diverged, back tracking."
            )
            model = DEMDiffusionModel(cond_dim=cond_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        else:
            loss_prev = loss
            if save_model:
                torch.save(
                    model.state_dict(),
                    model_path,
                )
                print(
                    f"Epoch {round((epoch+1)*subset_fraction):04d}/{round(num_epochs*subset_fraction):04d}  Loss: {loss:.3f}. Model saved"
                )
            else:
                print(
                    f"Epoch {round((epoch+1)*subset_fraction):04d}/{round(num_epochs*subset_fraction):04d}  Loss: {loss:.3f}"
                )
    return model
