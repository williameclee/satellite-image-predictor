import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import h5py


class DEMDataset(Dataset):
    def __init__(
        self,
        dem_src,
        tile_size=256,
        coarsen=1,
        rotate=True,
        force_reload=False,
        be_quiet=False,
        overlap=0.2,
        preload_to_ram=False,
    ):
        self.tile_size = tile_size
        self.mesh_size = 50 * coarsen
        self.use_rotation = rotate
        self.preload_to_ram = preload_to_ram

        base_dir = "training-data"
        h5_name = f"{dem_src}-T{tile_size}"
        if coarsen > 1:
            h5_name += f"-C{coarsen}"
        h5_path = os.path.join(base_dir, f"{h5_name}-N1.h5")

        if not os.path.exists(h5_path) or force_reload:
            dem_input_dir = os.path.join(base_dir, dem_src)
            dem_input_file = os.path.join(base_dir, f"{dem_src}.tif")
            dem_output_dir = h5_path.replace("-N1.h5", "")

            if force_reload and os.path.exists(dem_output_dir):
                shutil.rmtree(dem_output_dir)

            if os.path.exists(dem_output_dir):
                if not be_quiet:
                    print(f"Tiles already exist in {dem_output_dir}.")
            elif os.path.exists(dem_input_dir):
                if not be_quiet:
                    print("Converting folder of DEMs to tiles...")
                for tif in [f for f in os.listdir(dem_input_dir) if f.endswith(".tif")]:
                    dem2tiles(
                        os.path.join(dem_input_dir, tif),
                        dem_output_dir=dem_output_dir,
                        tile_size=tile_size,
                        coarsen=coarsen,
                        force_reload=force_reload,
                        be_quiet=be_quiet,
                        overlap=overlap,
                    )
            elif os.path.exists(dem_input_file):
                if not be_quiet:
                    print("Converting single DEM to tiles...")
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
                raise FileNotFoundError(
                    f"{dem_input_dir} or {dem_input_file} not found."
                )

            convert_npy_to_h5(
                dem_output_dir,
                output_path=h5_path,
                normalise=True,
                mesh_size=self.mesh_size,
            )
        else:
            if not be_quiet:
                print(f"Using existing HDF5 file at {h5_path}...")

        h5_content = h5py.File(h5_path, "r")
        self.dem_data = h5_content["data"]
        self.cond_data = h5_content["cond"]
        self.num_tiles = self.dem_data.shape[0]

        self.indices = (
            [(i, k) for i in range(self.num_tiles) for k in range(4)]
            if rotate
            else [(i, 0) for i in range(self.num_tiles)]
        )

        if preload_to_ram:
            with h5py.File(h5_path, "r") as f:
                self.data = f["data"][:]
                self.cond = f["cond"][:]
        else:
            h5_content = h5py.File(h5_path, "r")
            self.data = h5_content["data"]
            self.cond = h5_content["cond"]

        if not be_quiet:
            print(
                f"Loaded {len(self.indices)} tiles from {h5_path} with shape {self.dem_data.shape[1:]}"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.preload_to_ram:
            dem = torch.tensor(self.data[idx], dtype=torch.float32)
            # cond = torch.tensor(self.cond[idx], dtype=torch.float32)
            # dem = self.data[idx]
            cond = self.cond[idx]
        else:
            i, rot_k = self.indices[idx]
            dem = torch.tensor(self.dem_data[i], dtype=torch.float32)  # [1, H, W]
            cond = torch.tensor(self.cond_data[i], dtype=torch.float32)  # [4]

        if self.use_rotation and rot_k > 0:
            dem = torch.rot90(dem, k=rot_k, dims=[1, 2])

        return dem, cond


def normalise_dem(dem):
    # Compute stats
    min_elev = np.min(dem)
    max_elev = np.max(dem)
    mean_elev = np.mean(dem)

    # Normalize tile by (dem - mean) / (max - min)
    denom = max(max_elev - min_elev, 1e-5)  # prevent division by 0
    dem_norm = ((dem - mean_elev) / denom)[
        None, ...
    ]  # Bound to [-1, 1], with mean at 0

    return dem_norm, mean_elev, min_elev, max_elev


def denormalise_dem(dem, mean_elev, min_elev, max_elev):
    # Denormalize tile by dem * (max - min) + mean
    denom = max(max_elev - min_elev, 1e-5)  # prevent division by 0
    dem_denorm = dem * denom + mean_elev

    return dem_denorm


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

    if not be_quiet:
        print(f"Saved {i_patch} tiles to {dem_output_dir}")

    return


def convert_npy_to_h5(
    input_dir, output_path=None, dtype=np.float32, normalise=True, mesh_size=50
):
    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    if len(npy_files) == 0:
        raise ValueError("No .npy files found in input directory")

    # Default output path
    if output_path is None:
        base = os.path.basename(os.path.normpath(input_dir))
        output_path = os.path.join(
            os.path.dirname(input_dir),
            f"{base}-N{int(normalise)}.h5",
        )
        print(f"Default output path: {output_path}")

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Overwriting...")
        os.remove(output_path)

    all_data = []
    all_cond = []

    for fname in tqdm(npy_files, desc="Loading and computing stats"):
        dem = np.load(os.path.join(input_dir, fname)).astype(np.float32)

        # Replace NaNs and make sure it's float32
        dem = np.nan_to_num(dem, nan=0.0).astype(np.float32)

        if normalise:
            dem, mean_elev, min_elev, max_elev = normalise_dem(dem)
        else:
            _, mean_elev, min_elev, max_elev = normalise_dem(dem)

        all_data.append(dem[None, ...])  # add channel dimension [1, H, W]
        all_cond.append([mean_elev, min_elev, max_elev, mesh_size])

    all_data = np.stack(all_data, axis=0)  # [N, 1, H, W]
    all_cond = np.array(all_cond, dtype=np.float32)  # [N, 4]

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset(
            "data", data=all_data, dtype=dtype, compression="gzip", chunks=True
        )
        h5f.create_dataset("cond", data=all_cond, dtype=np.float32)

    print(f"HDF5 file saved to: {output_path}")
    print(f"  - DEM shape: {all_data.shape}")
    print(f"  - cond shape: {all_cond.shape}")
