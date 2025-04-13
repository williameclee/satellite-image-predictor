import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
import rasterio
from rasterio.windows import Window


class DEMDataset(Dataset):
    def __init__(
        self,
        dem_src_path,
        dem_out_dir=None,
        tile_size=256,
        rotate=True,
        normalise_factor=1e4,  # Make sure in no case the DEM values exceed 1 after normalisation (i.e. divide by 10000)
        log_transform=False,
    ):
        if dem_out_dir is None:
            dem_out_dir = dem2tiles(dem_src_path, tile_size=tile_size, be_quiet=True)
        elif os.path.exists(dem_out_dir) and not os.path.isdir(dem_out_dir):
            raise ValueError(f"{dem_out_dir} is not a directory")

        self.dem_out_dir = dem_out_dir
        self.transform = transforms.ToTensor()
        self.tile_size = tile_size
        self.use_rotation = rotate
        self.normalise_factor = normalise_factor
        self.log_transform = log_transform
        # self.files = sorted([f for f in os.listdir(dem_out_dir) if f.endswith(".npy")])
        self.files = sorted([f for f in os.listdir(dem_out_dir) if f.endswith(".npy")])
        if self.use_rotation:
            self.files = [(fname, k) for fname in self.files for k in range(4)]
        else:
            self.files = [(fname, 0) for fname in self.files]
        print(
            f"Loaded {len(self.files)} DEM tiles from {dem_out_dir} with size {tile_size}x{tile_size}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, id):
        filename, rot_k = self.files[id]
        dem = np.load(os.path.join(self.dem_out_dir, filename))
        dem = np.expand_dims(dem, 0)  # [1, H, W]
        dem = torch.tensor(dem, dtype=torch.float32)

        if self.log_transform:
            dem = torch.clamp(dem, min=0)
            dem = torch.log10(dem + 1) / np.log10(self.normalise_factor + 1)
        else:
            dem = dem / self.normalise_factor

        if self.use_rotation and rot_k > 0:
            dem = torch.rot90(dem, k=rot_k, dims=[1, 2])

        return dem


def dem2tiles(
    dem_src_path,
    dem_out_dir=None,
    tile_size=256,
    overlap=0.2,
    zero_threshold=0.9,
    force_reload=False,
    save_tiff=False,
    be_quiet=False,
):

    # Find output directory
    if dem_out_dir is None:
        dem_source_filename = os.path.splitext(os.path.basename(dem_src_path))[0]
        dem_out_dir = os.path.join(
            os.path.dirname(dem_src_path), f"{dem_source_filename}-T{tile_size}"
        )

    # Check if output directory exists (and is not empty)
    if not force_reload and os.path.exists(dem_out_dir) and os.listdir(dem_out_dir):
        if not be_quiet:
            print(f"Tiles exist in {dem_out_dir}")
        return dem_out_dir

    # Otherwise, create the output directory
    os.makedirs(dem_out_dir, exist_ok=True)

    # Open all input rasters
    with rasterio.open(dem_src_path) as dem_src:

        # Check dimensions
        width, height = dem_src.width, dem_src.height

        if overlap <= 1:
            tile_spacing = round(tile_size * (1 - overlap))

        i_patch = 0

        for top in range(0, height, tile_spacing):
            for left in range(0, width, tile_spacing):
                window = Window(left, top, tile_size, tile_size)

                # Skip tiles that don't fully fit
                if top + tile_size > height or left + tile_size > width:
                    continue

                # Read and clean input channels
                dem = dem_src.read(1, window=window).astype(np.float32)
                # Check for NaN values and throw an error if any are found
                if np.isnan(dem).any():
                    raise ValueError("DEM contains NaN values, which are not allowed.")

                # 0 = water body
                dem[np.isnan(dem)] = 0

                # Skip uninformative patches
                zero_ratio = np.mean(dem == 0)
                if zero_ratio > zero_threshold:
                    continue

                # Save input and target
                np.save(
                    os.path.join(
                        dem_out_dir, f"input_{i_patch:05d}.npy"
                    ),  # don't start from 1 so it's easier to look them up in the dataset
                    dem,
                )

                if save_tiff:
                    dem_out_path_tiff = os.path.join(
                        dem_out_dir, f"input_{i_patch:05d}.tif"
                    )
                    with rasterio.open(
                        dem_out_path_tiff,
                        "w",
                        driver="GTiff",
                        height=tile_size,
                        width=tile_size,
                        count=1,
                        dtype=dem.dtype,
                        crs=dem_src.crs,
                        transform=dem_src.window_transform(window),
                    ) as dst:
                        dst.write(dem, 1)

                i_patch += 1

    if not be_quiet:
        print(
            f"Saved {i_patch} training samples ({tile_size}x{tile_size}) to {dem_out_dir}"
        )
    return dem_out_dir
