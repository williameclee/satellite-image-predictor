import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
from rasterio.windows import Window


class SatelliteDataset(Dataset):
    def __init__(
        self,
        input_path,
        target_path,
        tile_size=256,
        in_channels=2,
        rotate=True,
        forest_gamma=None,
        force_reload=False,
    ):
        self.tile_size = tile_size
        self.in_channels = in_channels
        self.use_rotation = rotate
        self.forest_gamma = forest_gamma
        self.force_reload = force_reload

        input_dir, target_dir = create_tiles(
            input_path, target_path, tile_size, in_channels, force_reload
        )

        self.input_dir = input_dir
        self.target_dir = target_dir

        self.base_filenames = sorted(
            [f for f in os.listdir(input_dir) if f.endswith(".npy")]
        )

        if self.use_rotation:
            self.filenames = [
                (fname, k) for fname in self.base_filenames for k in range(4)
            ]
        else:
            self.filenames = [(fname, 0) for fname in self.base_filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname, rot_k = self.filenames[idx]
        input_path = os.path.join(self.input_dir, fname)
        target_name = fname.replace("input_", "target_").replace(".npy", ".png")
        target_path = os.path.join(self.target_dir, target_name)

        if not os.path.exists(target_path):
            print(
                f"Warning: Target file {target_path} does not exist. Skipping this item."
            )
            return None

        input = np.load(input_path)  # [C, H, W]
        target = np.array(Image.open(target_path)).transpose(2, 0, 1)  # [C, H, W]

        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32) / 255.0

        # Add gamma correction to the forrest if needed
        if self.forest_gamma is not None:
            lc = input[1]  # assume channel 1 is land cover
            forest_mask = (lc == 4).unsqueeze(0).expand_as(target).float()

            gamma = self.forest_gamma
            target = torch.clamp(target, 1e-4, 1.0)  # avoid zeros before gamma
            corrected = torch.pow(target, 1.0 / gamma)
            target = forest_mask * corrected + (1 - forest_mask) * target

        if self.use_rotation and rot_k > 0:
            input = torch.rot90(input, k=rot_k, dims=[1, 2])
            target = torch.rot90(target, k=rot_k, dims=[1, 2])

        if torch.isnan(input).any() or torch.isinf(input).any():
            print(f"NaN/Inf found in input {fname}")
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"NaN/Inf found in target {target_name}")

        return input, target


def create_tiles(
    input_raw_path, target_raw_path, tile_size=256, in_channels=2, force_reload=False
):
    input_raw_folder = os.path.dirname(input_raw_path)
    input_raw_filename = os.path.basename(input_raw_path)
    target_raw_folder = os.path.dirname(target_raw_path)
    target_raw_filename = os.path.basename(target_raw_path)
    input_file = f"{input_raw_filename.replace('_stack', '').replace('.tif', '')}-T{tile_size}C{in_channels}"
    target_file = (
        f"{target_raw_filename.replace('_stack', '').replace('.tif', '')}-T{tile_size}"
    )
    input_folder = os.path.join(input_raw_folder, f"{input_file}/")
    target_folder = os.path.join(target_raw_folder, f"{target_file}/")

    if (
        not force_reload
        and os.path.exists(input_folder)
        and os.path.exists(target_folder)
    ):
        # If the folders already exist and force_reload is False, return existing folders
        print(f"Using existing tiles in {input_folder} and {target_folder}")
        return input_folder, target_folder

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)

    with rasterio.open(input_raw_path) as input_src, rasterio.open(
        target_raw_path
    ) as target_src:
        width, height = input_src.width, input_src.height

        patch_id = 0
        for top in range(0, height, round(tile_size / 1.5)):
            for left in range(0, width, round(tile_size / 1.5)):
                window = Window(left, top, tile_size, tile_size)

                # Skip tiles that don't fully fit
                if top + tile_size > height or left + tile_size > width:
                    continue

                # Read input (DEM + LC)
                input_tile = input_src.read(window=window)  # shape: [5, 256, 256]

                # Replace NaN values in DEM and LC
                dem = input_tile[0]
                lc = input_tile[1]
                dem[np.isnan(dem)] = 0
                lc[np.isnan(lc)] = 1

                if in_channels == 2:
                    input_tile = np.stack([dem, lc], axis=0)
                elif in_channels == 3:
                    hs = input_tile[4]
                    hs[np.isnan(hs)] = 1 / np.sqrt(2)
                    input_tile = np.stack([dem, lc, hs], axis=0)
                else:
                    raise ValueError(
                        f"Unsupported number of input channels: {in_channels}. Use 2 or 3."
                    )

                if np.isnan(input_tile).any():
                    # print(f"Skipping patch {patch_id}: contains NaN values in input.")
                    continue

                # Skip if too much water
                water_ratio = np.mean(lc == 1)
                is_flat = np.all(dem == 0)

                if water_ratio > 0.9 or is_flat:
                    continue

                # Read target (RGB)
                target_tile = target_src.read(window=window)  # shape: [3, 256, 256]
                target_tile = np.transpose(target_tile, (1, 2, 0))  # [H, W, C]

                # Skip if target has too much black (0) pixels
                too_much_black = np.mean(target_tile == 0) > 0.2
                if too_much_black:
                    # print(f"Skipping patch {patch_id}: too much black in target image")
                    continue

                # Normalize input if needed (assuming already normalized, skip this)
                np.save(
                    os.path.join(input_folder, f"input_{patch_id+1:04d}.npy"),
                    input_tile,
                )

                # Save target as PNG
                img = Image.fromarray(target_tile.astype(np.uint8))
                img.save(os.path.join(target_folder, f"target_{patch_id+1:04d}.png"))

                patch_id += 1

    print(f"Saved {patch_id} training samples (256x256)")

    return input_folder, target_folder
