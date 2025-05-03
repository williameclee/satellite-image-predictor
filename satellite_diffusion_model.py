import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import os

from dem_diffusion_model import SinusoidalPosEmb, cosine_beta_schedule

from dem_diffusion_model import ResidualBlock, DownBlock, UpBlock


class SatelliteDiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels=16,
        out_channels=3,
        base_channels=128,
        time_embed_dim=128,
        aux_input_dim=10,
        v_prediction=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_input_dim = aux_input_dim
        self.v_prediction = v_prediction

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.cond_encoder = ConditionEncoder(aux_input_dim, time_embed_dim)

        # Downsampling blocks
        self.down1 = DownBlock(in_channels, base_channels, time_embed_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_embed_dim)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_embed_dim)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8, time_embed_dim)

        # Bottleneck
        # self.middle = ResidualBlock(
        #     base_channels * 8, base_channels * 8, time_embed_dim
        # )
        self.middle = Bottleneck(base_channels * 8, time_embed_dim)

        # Upsampling blocks (with skip channels)
        self.up3 = UpBlock(
            base_channels * 8, base_channels * 4, base_channels * 4, time_embed_dim
        )
        self.up2 = UpBlock(
            base_channels * 4, base_channels * 2, base_channels * 2, time_embed_dim
        )
        self.up1 = UpBlock(
            base_channels * 2, base_channels, base_channels, time_embed_dim
        )

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t, aux):
        t_emb = self.time_embedding(t)
        aux_emb = self.cond_encoder(aux)

        down1, skip1 = self.down1(x, t_emb, aux_emb)
        down2, skip2 = self.down2(down1, t_emb, aux_emb)
        down3, skip3 = self.down3(down2, t_emb, aux_emb)
        down4, _ = self.down4(down3, t_emb, aux_emb)

        mid = self.middle(down4, t_emb, aux_emb)

        up3 = self.up3(mid, skip3, t_emb, aux_emb)
        up2 = self.up2(up3, skip2, t_emb, aux_emb)
        up1 = self.up1(up2, skip1, t_emb, aux_emb)

        out = self.final(up1)
        return out


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


class Bottleneck(nn.Module):
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_embed_dim)
        self.attn = AttentionBlock(channels)
        self.res2 = ResidualBlock(channels, channels, time_embed_dim)

    def forward(self, x, t_emb, aux_emb):
        x = self.res1(x, t_emb, aux_emb)
        x = self.attn(x)  # AttentionBlock only takes the tensor
        x = self.res2(x, t_emb, aux_emb)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # Reshape to (seq_len, batch, channels)
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = self.proj(x)
        x = x.permute(1, 2, 0).view(
            b, c, h, w
        )  # Reshape back to (batch, channels, h, w)
        return x


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
    v_prediction=False,
):
    if subset_fraction < 1.0:
        num_epochs = round(round(num_epochs / subset_fraction))
        print(
            f"Using {subset_fraction:.1%} of the dataset, training extended to {num_epochs} epochs"
        )

    print(f"Using device: {device}")

    num_workers = 0 if dataset.preload_to_ram else num_workers
    persistent_workers = not dataset.preload_to_ram
    num_samples = max(int(len(dataset) * subset_fraction), 1)

    model, optimiser, model_path = load_model(
        save_model,
        model,
        dataset,
        v_prediction,
        learning_rate,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )
    best_model_path = model_path.replace(".pth", "_best.pth")
    model.train()

    betas = cosine_beta_schedule(num_timesteps)
    alphas = np.insert(np.cumprod(1.0 - betas), 0, 1.0)
    alphas = torch.tensor(alphas, dtype=torch.float32, device=device)

    loss_prev = float("inf")
    lost_best = float("inf")

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

        epoch_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            rgb = batch["target_image"]
            geoinfo_spatial = batch["geoinfo_spatial"]
            geoinfo_vector = batch["geoinfo_vector"]

            # pad to the expected dimension
            if geoinfo_spatial.shape[1] < model.in_channels - rgb.shape[1]:
                pad_width = model.in_channels - geoinfo_spatial.shape[1] - rgb.shape[1]
                geoinfo_spatial = F.pad(geoinfo_spatial, (0, 0, 0, 0, 0, pad_width))
            if geoinfo_vector.shape[1] < model.aux_input_dim:
                pad_width = model.aux_input_dim - geoinfo_vector.shape[1]
                geoinfo_vector = F.pad(geoinfo_vector, (0, pad_width))
            rgb = rgb.to(device)
            geoinfo_spatial = geoinfo_spatial.to(device)
            geoinfo_vector = geoinfo_vector.to(device)

            # Sample timestep
            timestep = torch.randint(1, num_timesteps, (rgb.size(0),), device=device)
            alpha = alphas[timestep].view(-1, 1, 1, 1)

            noise = torch.randn_like(rgb)
            rgb_noisy = alpha.sqrt() * rgb + (1 - alpha).sqrt() * noise

            # Predict noise
            t = timestep.float() / num_timesteps

            if v_prediction:
                v = alpha.sqrt() * noise - (1 - alpha).sqrt() * rgb
                v_pred = model(
                    torch.cat([rgb_noisy, geoinfo_spatial], dim=1), t, geoinfo_vector
                )
                loss = F.mse_loss(v_pred, v)
            else:
                noise_pred = model(
                    torch.cat([rgb_noisy, geoinfo_spatial], dim=1), t, geoinfo_vector
                )
                loss = F.mse_loss(noise_pred, noise)

            optimiser.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)

        if (epoch_loss > loss_prev * 1e2) or (loss_prev < 0.5 and epoch_loss > 5):
            print(
                f"Loss increased from {loss_prev:.4f} to {epoch_loss:.4f}, backtracking..."
            )
            model, optimiser, _ = load_model(
                save_model, model, dataset, learning_rate, device
            )
            continue

        print(
            f"Epoch {int(np.ceil((epoch+1)*subset_fraction)):04d}/{int(np.ceil(num_epochs*subset_fraction)):04d} - Loss: {epoch_loss:.4f}"
        )
        loss_prev = epoch_loss

        # Model saving
        if not save_model:
            continue

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
            },
            model_path,
        )

        # Tracking best model
        if epoch_loss >= lost_best:
            continue

        lost_best = epoch_loss

        if epoch < 10:  # Save the best model only after 10 epochs
            continue

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
            },
            best_model_path,
        )
        print(f"Best model saved to {best_model_path}")

    return model


def load_model(
    save_model,
    model,
    dataset,
    v_prediction=False,
    learning_rate=1e-4,
    device=(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
):
    # Calculate channel size first, as it is needed for the model path name
    in_channels = max(
        dataset[0]["target_image"].shape[0] + dataset[0]["geoinfo_spatial"].shape[0],
        16,
    )
    out_channels = dataset[0]["target_image"].shape[0]
    aux_dim = max(dataset[0]["geoinfo_vector"].shape[0], 16)

    if isinstance(save_model, str) and os.path.exists(save_model):
        model_path = save_model
        save_model = True
    else:
        os.makedirs("models", exist_ok=True)
        if v_prediction:
            v_prediction_str = "-V"
        else:
            v_prediction_str = ""
        model_path = os.path.join(
            "models",
            f"satellite_diffusion{v_prediction_str}-T{dataset.size[0]:04d}-L{dataset.lct_classes}-N{int(dataset.normalise)}-I{in_channels}_{aux_dim}.pth",
        )

    load_model_from_path = False
    if isinstance(model, nn.Module):
        model.to(device)
    elif isinstance(model, str) and os.path.exists(model):
        model_path = model
        load_model_from_path = True
    elif isinstance(model, str) and model == "load":
        if os.path.exists(model_path):
            load_model_from_path = True
        else:
            print(f"Best model not found at {model_path}, starting from scratch")

    model = SatelliteDiffusionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        aux_input_dim=aux_dim,
        v_prediction=v_prediction,
    )
    if load_model_from_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    if load_model_from_path:
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        if not save_model:
            print(f"Loaded model from {model_path}")
        else:
            print(f"Loaded model from and will be saved to {model_path}")

    return model, optimiser, model_path
