import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import os


class SatelliteDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels=3 + 2 + 20,  # image + dem + cloud + land cover
        out_channels=3,
        base_channels=64,
        embed_dim=128,
        aux_input_dim=10,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(embed_dim)
        self.cond_encoder = ConditionEncoder(aux_input_dim, embed_dim)

        self.enc1 = DownBlock(in_channels, base_channels, embed_dim)
        self.enc2 = DownBlock(base_channels, base_channels * 2, embed_dim)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4, embed_dim)

        self.middle = ResidualBlock(base_channels * 4, base_channels * 4, embed_dim)

        self.dec3 = UpBlock(base_channels * 4, base_channels * 2, embed_dim)
        self.dec2 = UpBlock(base_channels * 2, base_channels, embed_dim)
        self.dec1 = UpBlock(base_channels, base_channels // 2, embed_dim)
        self.final = nn.Conv2d(base_channels // 2, out_channels, kernel_size=1)

    def forward(self, x, t, aux):
        t_emb = self.time_embed(t)
        aux_emb = self.cond_encoder(aux)

        x1 = self.enc1(x, t_emb, aux_emb)
        x2 = self.enc2(x1, t_emb, aux_emb)
        x3 = self.enc3(x2, t_emb, aux_emb)

        mid = self.middle(x3, t_emb, aux_emb)

        x = self.dec3(mid, t_emb, aux_emb) + x2
        x = self.dec2(x, t_emb, aux_emb) + x1
        x = self.dec1(x, t_emb, aux_emb)
        x = self.final(x)

        return x


class ConditionEncoder(nn.Module):
    def __init__(self, aux_input_dim=10, embed_dim=128):
        super().__init__()
        self.expected_aux_dim = aux_input_dim
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, aux_vector):
        if aux_vector.shape[1] < self.expected_aux_dim:
            pad_width = self.expected_aux_dim - aux_vector.shape[1]
            aux_vector = F.pad(aux_vector, (0, pad_width), value=0.0)
        elif aux_vector.shape[1] > self.expected_aux_dim:
            raise ValueError(
                f"Input has {aux_vector.shape[1]} dims, but expected only {self.expected_aux_dim}"
            )

        return self.aux_encoder(aux_vector)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t):
        return self.embed(t.unsqueeze(-1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_embed = nn.Linear(embed_dim, out_channels)
        self.aux_embed = nn.Linear(embed_dim, out_channels)
        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb, aux_emb):
        h = self.activation(self.conv1(x))
        B, C, H, W = h.shape
        t = self.time_embed(t_emb).view(B, C, 1, 1)
        a = self.aux_embed(aux_emb).view(B, C, 1, 1)
        h = h + t + a
        h = self.conv2(self.activation(h))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, embed_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, t_emb, aux_emb):
        x = self.res(x, t_emb, aux_emb)
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, embed_dim)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, t_emb, aux_emb):
        x = self.res(x, t_emb, aux_emb)
        return self.up(x)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.numpy()


def train_satellitediffusionmodel(
    model,
    dataset,
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=20,
    num_timesteps=100,
    subset_fraction=1.0,
    save_model=True,
    device=(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    num_workers=8,
    pin_memory=True,
    #
    rgb_loss_weight=0.0,
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
        save_model, model, dataset, learning_rate, device
    )

    betas = cosine_beta_schedule(num_timesteps)
    alphas = np.insert(np.cumprod(1.0 - betas), 0, 1.0)
    alphas = torch.tensor(alphas, dtype=torch.float32, device=device)

    for epoch in range(num_epochs):
        sampler = SubsetRandomSampler(random.sample(range(len(dataset)), num_samples))
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
            rgb = batch["target_image"].to(device)
            geoinfo_spatial = batch["geoinfo_spatial"].to(device)
            geoinfo_vector = batch["geoinfo_vector"].to(device)

            # 1. Sample timestep
            timestep = torch.randint(1, num_timesteps, (rgb.size(0),), device=device)
            alpha = alphas[timestep].view(-1, 1, 1, 1)

            noise = torch.randn_like(rgb)
            rgb_noisy = alpha.sqrt() * rgb + (1 - alpha).sqrt() * noise

            # 3. Predict noise
            t = timestep.float() / num_timesteps
            noise_pred = model(
                torch.cat([rgb_noisy, geoinfo_spatial], dim=1), t, geoinfo_vector
            )
            noise_loss = F.mse_loss(noise_pred, noise)

            alpha_ref = 0.5
            rgb_noisy_ref = alpha_ref * rgb + (1 - alpha_ref) * noise
            rgb_pred_ref = (rgb_noisy_ref - np.sqrt(1 - alpha_ref) * noise_pred) / np.sqrt(alpha_ref)
            rgb_pred_ref = torch.clamp(rgb_pred_ref, 0, 1.0)
            rgb_loss = F.mse_loss(rgb_pred_ref, rgb)
            loss = noise_loss + rgb_loss_weight * rgb_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        if save_model:
            torch.save(model.state_dict(), model_path)
        print(
            f"Epoch {int(np.ceil((epoch+1)*subset_fraction)):04d}/{int(np.ceil(num_epochs*subset_fraction)):04d} - Loss: {epoch_loss:.4f}"
        )

    return model


def load_model(save_model, model, dataset, learning_rate, device):
    if isinstance(save_model, str) and os.path.exists(save_model):
        model_path = save_model
        save_model = True
    else:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join(
            "models",
            f"satellite_diffusion-T{dataset.size[0]}-A{len(dataset.geoinfo_keys)}.pth",
        )
    if save_model:
        print(f"Model will be saved to {model_path}")

    in_channels = (
        dataset[0]["target_image"].shape[0] + dataset[0]["geoinfo_spatial"].shape[0]
    )
    out_channels = dataset[0]["target_image"].shape[0]
    aux_dim = dataset[0]["geoinfo_vector"].shape[0]

    if isinstance(model, nn.Module):
        model.to(device)
    elif isinstance(model, str) and os.path.exists(model):
        model = SatelliteDiffusionUNet(
            in_channels=in_channels, out_channels=out_channels, aux_input_dim=aux_dim
        )
        model.load_state_dict(torch.load(model, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    elif isinstance(model, str) and model == "load":
        model = SatelliteDiffusionUNet(
            in_channels=in_channels, out_channels=out_channels, aux_input_dim=aux_dim
        )
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Best model not found at {model_path}, starting from scratch")
    elif isinstance(model, str) and model == "new":
        model = SatelliteDiffusionUNet(
            in_channels=in_channels, out_channels=out_channels, aux_input_dim=aux_dim
        )

    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimiser, model_path
