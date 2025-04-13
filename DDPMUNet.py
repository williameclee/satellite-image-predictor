import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
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
            # nn.Sigmoid(), # no avtivation since the noise (torch.randn) is unbounded
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        d1_out, skip1 = self.down1(x, t_emb)
        d2_out, skip2 = self.down2(d1_out, t_emb)
        d3_out, skip3 = self.down3(d2_out, t_emb)
        d4_out, _ = self.down4(d3_out, t_emb)

        mid = self.middle(d4_out, t_emb)

        up3_out = self.up3(mid, skip3, t_emb)
        up2_out = self.up2(up3_out, skip2, t_emb)
        up1_out = self.up1(up2_out, skip1, t_emb)

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

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

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

    def forward(self, x, t):
        x = self.block1(x, t)
        x = self.block2(x, t)
        return self.pool(x), x


class UpBlock(nn.Module):
    def __init__(self, up_in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(up_in_channels, out_channels, 2, stride=2)
        self.block1 = ResBlock(out_channels + skip_channels, out_channels, time_emb_dim)
        self.block2 = ResBlock(out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t)
        x = self.block2(x, t)
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
