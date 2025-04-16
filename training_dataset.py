import torch
from torch.utils.data import Dataset
import numpy as np
import os

# from torchvision import transforms
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling


# class DEMDataset(Dataset):
#     def __init__(
#         self,
#         dem_src_path,
#         dem_out_dir=None,
#         tile_size=256,
#         rotate=True,
#         normalise_factor=1e4,
#         log_transform=False,
#     ):
#         if dem_out_dir is None:
#             dem_out_dir = dem2tiles(dem_src_path, tile_size=tile_size, be_quiet=True)
#         elif os.path.exists(dem_out_dir) and not os.path.isdir(dem_out_dir):
#             raise ValueError(f"{dem_out_dir} is not a directory")

#         self.dem_out_dir = dem_out_dir
#         self.tile_size = tile_size
#         self.use_rotation = rotate
#         self.normalise_factor = normalise_factor
#         self.log_transform = log_transform

#         # Load all files into memory
#         self.data_cache = {}
#         self.files = sorted([f for f in os.listdir(dem_out_dir) if f.endswith(".npy")])
#         for fname in self.files:
#             file_path = os.path.join(self.dem_out_dir, fname)
#             self.data_cache[fname] = np.load(file_path)

#         if self.use_rotation:
#             self.files = [(fname, k) for fname in self.files for k in range(4)]
#         else:
#             self.files = [(fname, 0) for fname in self.files]

#         print(
#             f"Loaded {len(self.files)} DEM tiles from {dem_out_dir} with size {tile_size}x{tile_size}"
#         )

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, id):
#         filename, rot_k = self.files[id]
#         dem = self.data_cache[filename]  # Use preloaded data
#         dem = np.expand_dims(dem, 0)  # [1, H, W]
#         dem = torch.tensor(dem, dtype=torch.float32)

#         if self.log_transform:
#             dem = torch.clamp(dem, min=0)
#             dem = torch.log10(dem + 1) / np.log10(self.normalise_factor + 1)
#         else:
#             dem = dem / self.normalise_factor

#         if self.use_rotation and rot_k > 0:
#             dem = torch.rot90(dem, k=rot_k, dims=[1, 2])

#         return dem

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import shutil


class DEMDataset(Dataset):
    def __init__(
        self,
        dem_src,
        tile_size=256,
        coarsen=1,
        rotate=True,
        force_reload=False,
        be_quiet=False,
        # arguments for dem2tiles
        overlap=0.2,
    ):

        # Find data directory
        if coarsen == 1:
            dem_output_dir = f"training-data/{dem_src}-T{tile_size}"
        else:
            dem_output_dir = f"training-data/{dem_src}-T{tile_size}-C{coarsen}"
        dem_input_dir = f"training-data/{dem_src}"
        dem_input_file = f"training-data/{dem_src}.tif"

        if force_reload:
            if not be_quiet:
                print("Force reloading data...")
            if os.path.exists(dem_output_dir):
                shutil.rmtree(dem_output_dir)

        if os.path.exists(dem_output_dir):
            if not be_quiet:
                print("Source DEM is a directory.")
        elif os.path.exists(dem_input_dir):
            if not be_quiet:
                print("Converting DEM to tiles...")
            for dem_input_file in [
                f for f in os.listdir(dem_input_dir) if f.endswith(".tif")
            ]:
                dem2tiles(
                    os.path.join(dem_input_dir, dem_input_file),
                    dem_output_dir=dem_output_dir,
                    tile_size=tile_size,
                    coarsen=coarsen,
                    force_reload=force_reload,
                    be_quiet=be_quiet,
                    overlap=overlap,
                )
        elif os.path.exists(dem_input_file):
            if not be_quiet:
                print("Converting DEM to tiles...")
            dem2tiles(
                dem_input_file,
                dem_output_dir=dem_output_dir,
                tile_size=tile_size,
                coarsen=coarsen,
                force_reload=force_reload,
                be_quiet=be_quiet,
                overlap=overlap,
            )
        else:
            raise ValueError(
                f"DEM source {dem_src} not found. Place it in training-data/{dem_src} or training-data/{dem_src}.tif"
            )

        self.dem_output_dir = dem_output_dir
        self.tile_size = tile_size
        self.mesh_size = 50 * coarsen
        self.use_rotation = rotate
        self.files = sorted(
            [f for f in os.listdir(dem_output_dir) if f.endswith(".npy")]
        )

        if self.use_rotation:
            self.files = [(fname, k) for fname in self.files for k in range(4)]
        else:
            self.files = [(fname, 0) for fname in self.files]
        print(
            f"Loaded {len(self.files)} DEM tiles from {self.dem_output_dir} with size {tile_size}x{tile_size}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname, rot_k = self.files[idx]
        dem_path = os.path.join(self.dem_output_dir, fname)
        dem = np.load(dem_path).astype(np.float32)

        # Compute stats
        min_elev = np.min(dem)
        max_elev = np.max(dem)
        mean_elev = np.mean(dem)

        # Normalize tile by (dem - mean) / (max - min)
        denom = max(max_elev - min_elev, 1e-5)  # prevent division by 0
        dem_norm = (dem - mean_elev) / denom  # Bound to [-1, 1]

        dem = torch.tensor(dem_norm, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        cond = torch.tensor(
            [mean_elev, min_elev, max_elev, self.mesh_size], dtype=torch.float32
        )

        if self.use_rotation and rot_k > 0:
            dem = torch.rot90(dem, k=rot_k, dims=[1, 2])

        return dem, cond


class CachedDEMDataset(Dataset):
    def __init__(self, file_list, use_cache=True):
        self.file_list = file_list
        self.use_cache = use_cache
        self.cache = {}

        if self.use_cache:
            print("Caching data to memory...")
            for fname in tqdm(self.file_list):
                arr = np.load(fname, mmap_mode=None)  # fully load
                self.cache[fname] = arr

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        if self.use_cache:
            arr = self.cache[fname]
        else:
            arr = np.load(fname, mmap_mode="r")
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # adjust as needed

    def __len__(self):
        return len(self.file_list)


def dem2tiles(
    dem_input_path,
    dem_output_dir=None,
    tile_size=256,
    coarsen=1,
    overlap=0.2,
    zero_threshold=0.9,
    min_height_diff=40,
    force_reload=False,
    save_tiff=False,
    be_quiet=False,
):
    import os
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    import torch.nn.functional as F

    # Determine output directory
    if dem_output_dir is None:
        dem_source_filename = os.path.splitext(os.path.basename(dem_input_path))[0]
        dem_output_dir = os.path.join(
            os.path.dirname(dem_input_path),
            f"{dem_source_filename}-T{tile_size}-C{coarsen}",
        )

    os.makedirs(dem_output_dir, exist_ok=True)

    if not force_reload and os.listdir(dem_output_dir):
        if not be_quiet:
            print(f"Tiles exist in {dem_output_dir}")
        return dem_output_dir

    tile_spacing = round(tile_size * (1 - overlap))

    i_patch = len([f for f in os.listdir(dem_output_dir) if f.endswith(".npy")])

    with rasterio.open(dem_input_path) as src:
        dem = src.read(1).astype(np.float32)
        dem[np.isnan(dem)] = 0

        height, width = dem.shape
        patch_size = tile_size * coarsen

        # Pad DEM if smaller than one tile
        pad_bottom = max(0, patch_size - height)
        pad_right = max(0, patch_size - width)

        if pad_bottom > 0 or pad_right > 0:
            dem = np.pad(
                dem,
                ((0, pad_bottom), (0, pad_right)),
                mode="edge",
            )
            height, width = dem.shape

        for top in range(0, height - patch_size + 1, tile_spacing * coarsen):
            for left in range(0, width - patch_size + 1, tile_spacing * coarsen):
                patch = dem[top : top + patch_size, left : left + patch_size]

                if patch.shape != (patch_size, patch_size):
                    continue  # Safety check

                # Downsample
                if coarsen > 1:
                    patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0)
                    patch_tensor = (
                        F.avg_pool2d(patch_tensor, kernel_size=coarsen)
                        .squeeze()
                        .numpy()
                    )
                else:
                    patch_tensor = patch

                # Filtering conditions
                if np.mean(patch_tensor == 0) > zero_threshold:
                    continue
                if patch_tensor.max() - patch_tensor.min() < min_height_diff:
                    continue

                # Save
                np.save(
                    os.path.join(dem_output_dir, f"input_{i_patch:06d}.npy"),
                    patch_tensor,
                )
                i_patch += 1

                # if not be_quiet and i_patch % 1000 == 0:
                #     print(f"Saved {i_patch} tiles to {dem_output_dir}")

    if not be_quiet:
        print(f"Saved {i_patch} tiles to {dem_output_dir}")

    return
