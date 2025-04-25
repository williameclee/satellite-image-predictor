import os
import json
import torch
import numpy as np
import rasterio
import torch.nn.functional as F
from torch.utils.data import Dataset
from dem_dataset import normalise_dem
import h5py
from tqdm import tqdm


class SatelliteDataset(Dataset):
    def __init__(
        self,
        data_dir,
        geoinfo_keys=None,
        size=(256, 256),
        rotate=False,
        preload_to_ram=False,
        lct_classes=20,
    ):
        self.size = size
        self.lct_classes = lct_classes
        self.preload_to_ram = preload_to_ram

        self.geoinfo_keys = geoinfo_keys or [
            "dem_mean",
            "dem_min",
            "dem_max",
            "tile_scale",
            "month",
            "solar_zenith",
            "solar_azimuth",
        ]

        if not os.path.exists(data_dir):
            data_dir = os.path.join(
                "training-data", f"{data_dir}-T{size[0]:04d}-R{int(rotate)}"
            )
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Dataset directory {data_dir} not found in training-data."
            )

        self.data_dir = data_dir

        if preload_to_ram:
            self.h5_path = save_dataset_to_h5(
                self.data_dir,
                geoinfo_keys=self.geoinfo_keys,
                lct_classes=self.lct_classes,
            )
            h5_content = h5py.File(self.h5_path, "r")
            self.geoinfo_vector_data = torch.from_numpy(h5_content["geoinfo_vector"][:])
            print(f"  - geoinfo_vector shape: {self.geoinfo_vector_data.shape}")
            self.target_image_data = torch.from_numpy(h5_content["target_image"][:])
            print(f"  - target_image shape: {self.target_image_data.shape}")
            self.geoinfo_spatial_data = torch.from_numpy(
                h5_content["geoinfo_spatial"][:]
            )
            print(f"  - geoinfo_spatial shape: {self.geoinfo_spatial_data.shape}")
        else:
            self.tiles = sorted(
                [f for f in os.listdir(self.data_dir) if f.endswith("_met.json")]
            )

    def __len__(self):
        if self.preload_to_ram:
            return self.target_image_data.shape[0]
        else:
            return len(self.tiles)

    def __getitem__(self, idx):
        if self.preload_to_ram:
            return {
                "target_image": self.target_image_data[idx],
                "geoinfo_spatial": self.geoinfo_spatial_data[idx],
                "geoinfo_vector": self.geoinfo_vector_data[idx],
            }
        else:
            meta_file = self.tiles[idx]
            return load_data_from_metadata(
                self.data_dir,
                meta_file,
                size=self.size,
                geoinfo_keys=self.geoinfo_keys,
                lct_classes=self.lct_classes,
            )


def load_data_from_metadata(
    input_dir, meta_file, size, geoinfo_keys=None, lct_classes=20
):
    with open(os.path.join(input_dir, meta_file)) as f:
        meta = json.load(f)

    # --- Load satellite image ---
    rgb_path = os.path.join(input_dir, meta["tiles"]["rgb"])
    with rasterio.open(rgb_path) as src:
        rgb = src.read(out_shape=(3, *size)).astype(np.float32) / 255.0

    # --- Load DEM ---
    dem_path = os.path.join(input_dir, meta["tiles"]["dem"])
    with rasterio.open(dem_path) as src:
        dem = src.read(1, out_shape=(1, *size)).astype(np.float32)
        dem, meta["dem_mean"], meta["dem_min"], meta["dem_max"] = normalise_dem(dem)

    # --- Load land cover type ---
    lct_path = os.path.join(input_dir, meta["tiles"]["lct"])
    with rasterio.open(lct_path) as src:
        lc = src.read(1, out_shape=size).astype(np.float32)[None, ...]
        lc = lc / lct_classes

    # --- Load cloud mask ---
    cld_path = os.path.join(input_dir, meta["tiles"]["cld"])
    with rasterio.open(cld_path) as src:
        cloud_mask = src.read(1, out_shape=(1, *size))[None, ...]

    # --- Auxiliary scalar vector ---
    if geoinfo_keys is not None:
        geoinfo_vector = [meta[k] for k in geoinfo_keys]
        geoinfo_vector_tensor = torch.tensor(geoinfo_vector, dtype=torch.float32)
    else:
        geoinfo_vector_tensor = torch.tensor([0.0])

    # --- Prepare tensors ---
    rgb_tensor = torch.from_numpy(rgb).float()
    dem_tensor = torch.from_numpy(dem).float()
    lc_tensor = torch.from_numpy(lc).float()
    cloud_tensor = torch.from_numpy(cloud_mask)

    geoinfo_spatial_tensor = torch.cat([dem_tensor, lc_tensor, cloud_tensor], dim=0)

    return {
        "target_image": rgb_tensor,
        "geoinfo_spatial": geoinfo_spatial_tensor,
        "geoinfo_vector": geoinfo_vector_tensor,
    }


def save_dataset_to_h5(
    input_dir,
    output_path=None,
    geoinfo_keys=None,
    lct_classes=20,
    overwrite=False,
    be_quiet=False,
):
    if output_path is None:
        base = os.path.basename(os.path.normpath(input_dir))
        if geoinfo_keys is not None:
            geoinfo_key_name = "-K" + "_".join(geoinfo_keys)
        else:
            geoinfo_key_name = ""
        output_name = f"{base}-L{lct_classes}{geoinfo_key_name}.h5"
        output_path = os.path.join(os.path.dirname(input_dir), output_name)
    if not be_quiet:
        if not overwrite and os.path.exists(output_path):
            print(f"Compressed dataset found at {output_path}")
        elif overwrite and os.path.exists(output_path):
            os.remove(output_path)
            print(
                f"Compressed dataset found at {output_path}, removed and overwriting..."
            )
        else:
            print(f"Compressed dataset will be saved to {output_path}")
    if os.path.exists(output_path) and not overwrite:
        return output_path

    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_met.json")])
    if len(json_files) == 0:
        raise ValueError("No .json metadata files found in input directory")

    all_target_images = []
    all_geoinfo_spatial = []
    all_geoinfo_vector = []

    for meta_file in tqdm(json_files, desc="Processing tiles"):
        tile = load_data_from_metadata(
            input_dir,
            meta_file,
            size=(256, 256),
            geoinfo_keys=geoinfo_keys,
        )

        all_target_images.append(tile["target_image"].numpy())
        all_geoinfo_spatial.append(tile["geoinfo_spatial"].numpy())
        all_geoinfo_vector.append(tile["geoinfo_vector"].numpy())

    all_target_images = np.stack(all_target_images, axis=0)
    all_geoinfo_spatial = np.stack(all_geoinfo_spatial, axis=0)
    all_geoinfo_vector = np.stack(all_geoinfo_vector, axis=0)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset(
            "target_image",
            data=all_target_images,
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )
        h5f.create_dataset(
            "geoinfo_spatial",
            data=all_geoinfo_spatial,
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )
        h5f.create_dataset(
            "geoinfo_vector",
            data=all_geoinfo_vector,
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )
    if not be_quiet:
        print(f"Compressed dataset saved to {output_path}")
        print(f"  - target_image shape: {all_target_images.shape}")
        print(f"  - geoinfo_spatial shape: {all_geoinfo_spatial.shape}")
        print(f"  - geoinfo_vector shape: {all_geoinfo_vector.shape}")

    return output_path


# def save_dataset_to_h5(dataset, output_path=None, be_quiet=False, only_check=False):
#     if output_path is None:
#         output_name = (
#             f"{os.path.basename(dataset.data_dir)}-K{'_'.join(dataset.geoinfo_keys)}.h5"
#         )
#         output_path = os.path.join(
#             os.path.dirname(dataset.data_dir), output_name
#         )  # Default path in the parent directory
#     if os.path.exists(output_path):
#         if not be_quiet:
#             print(f"Output file {output_path} already exists.")
#         return output_path
#     elif only_check:
#         return None

#     with h5py.File(output_path, "w") as h5f:
#         for i in range(len(dataset)):
#             sample = dataset[i]
#             tid = dataset.tiles[i].split("_")[1]
#             g = h5f.create_group(f"tile_{tid}")
#             g.create_dataset(
#                 "target_image",
#                 data=sample["target_image"].numpy(),
#                 compression="gzip",
#                 chunks=True,
#             )
#             g.create_dataset(
#                 "geoinfo_spatial",
#                 data=sample["geoinfo_spatial"].numpy(),
#                 compression="gzip",
#                 chunks=True,
#             )
#             g.create_dataset(
#                 "geoinfo_vector",
#                 data=sample["geoinfo_vector"].numpy(),
#                 compression="gzip",
#                 chunks=True,
#             )
#     if not be_quiet:
#         print(f"Saved dataset to {output_path}")
#     return output_path
