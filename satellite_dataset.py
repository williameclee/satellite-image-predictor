import os
import json
import torch
import numpy as np
import rasterio
import torch.nn.functional as F
from torch.utils.data import Dataset
from dem_dataset import normalise_dem
import h5py


class SatelliteDataset(Dataset):
    def __init__(
        self,
        data_dir,
        geoinfo_keys=None,
        size=(256, 256),
        lc_classes=20,
        preload_to_ram=False,
    ):
        self.size = size
        self.lc_classes = lc_classes
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
            try:
                data_dir = os.path.join("training-data", data_dir)
            except FileNotFoundError:
                raise FileNotFoundError(f"Dataset directory {data_dir} not found.")
        self.data_dir = data_dir

        self.tiles = sorted(
            [f for f in os.listdir(data_dir) if f.endswith("_met.json")]
        )

        self.file_loaded = False
        if preload_to_ram:
            self.h5_path = save_dataset_to_h5(self, be_quiet=False)
            self.h5_content = h5py.File(self.h5_path, "r")
            self.file_loaded = True

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        meta_file = self.tiles[idx]
        if self.preload_to_ram and self.file_loaded:
            key = meta_file.replace("_met.json", "")
            return {
                "target_image": torch.from_numpy(self.h5_content[key]["target_image"][:]),
                "geoinfo_spatial": torch.from_numpy(
                    self.h5_content[key]["geoinfo_spatial"][:]
                ),
                "geoinfo_vector": torch.from_numpy(
                    self.h5_content[key]["geoinfo_vector"][:]
                ),
            }
        else:
            # Load metadata
            with open(os.path.join(self.data_dir, meta_file)) as f:
                meta = json.load(f)

            # --- Load satellite image ---
            rgb_path = os.path.join(self.data_dir, meta["tiles"]["rgb"])
            with rasterio.open(rgb_path) as src:
                rgb = src.read(out_shape=(3, *self.size)).astype(np.float32) / 255.0

            # --- Load DEM ---
            dem_path = os.path.join(self.data_dir, meta["tiles"]["dem"])
            with rasterio.open(dem_path) as src:
                dem = src.read(1, out_shape=(1, *self.size)).astype(np.float32)
                dem, meta["dem_mean"], meta["dem_min"], meta["dem_max"] = normalise_dem(
                    dem
                )

            # --- Load land cover type ---
            lct_path = os.path.join(self.data_dir, meta["tiles"]["lct"])
            with rasterio.open(lct_path) as src:
                lc = src.read(1, out_shape=self.size)

            # --- Load cloud mask ---
            cld_path = os.path.join(self.data_dir, meta["tiles"]["cld"])
            with rasterio.open(cld_path) as src:
                cloud_mask = src.read(1, out_shape=(1, *self.size))[None, ...]

            # --- Auxiliary scalar vector ---
            geoinfo_vector = [meta[k] for k in self.geoinfo_keys]
            geoinfo_vector_tensor = torch.tensor(geoinfo_vector, dtype=torch.float32)

            # --- Prepare tensors ---
            rgb_tensor = torch.from_numpy(rgb).float()
            dem_tensor = torch.from_numpy(dem).float()
            lc_tensor = torch.from_numpy(lc).long()
            lc_tensor = (
                F.one_hot(lc_tensor, num_classes=self.lc_classes)
                .permute(2, 0, 1)
                .float()
            )
            cloud_tensor = torch.from_numpy(cloud_mask)

            geoinfo_spatial_tensor = torch.cat(
                [dem_tensor, cloud_tensor, lc_tensor], dim=0
            )

            return {
                "target_image": rgb_tensor,
                "geoinfo_spatial": geoinfo_spatial_tensor,
                "geoinfo_vector": geoinfo_vector_tensor,
            }


def save_dataset_to_h5(dataset, output_path=None, be_quiet=False):
    if output_path is None:
        output_name = (
            f"{os.path.basename(dataset.data_dir)}-K{'_'.join(dataset.geoinfo_keys)}.h5"
        )
        output_path = os.path.join(
            os.path.dirname(dataset.data_dir), output_name
        )  # Default path in the parent directory
    if os.path.exists(output_path):
        if not be_quiet:
            print(f"Output file {output_path} already exists.")
        return output_path

    with h5py.File(output_path, "w") as h5f:
        for i in range(len(dataset)):
            sample = dataset[i]
            tid = dataset.tiles[i].split("_")[1]
            g = h5f.create_group(f"tile_{tid}")
            g.create_dataset(
                "target_image",
                data=sample["target_image"].numpy(),
                compression="gzip",
                chunks=True,
            )
            g.create_dataset(
                "geoinfo_spatial",
                data=sample["geoinfo_spatial"].numpy(),
                compression="gzip",
                chunks=True,
            )
            g.create_dataset(
                "geoinfo_vector",
                data=sample["geoinfo_vector"].numpy(),
                compression="gzip",
                chunks=True,
            )
    if not be_quiet:
        print(f"Saved dataset to {output_path}")
    return output_path
