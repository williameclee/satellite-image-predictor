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
        rotate=0,
        preload_to_ram=False,
        normalise=True,
        lct_classes=20,
    ):
        self.size = size
        self.lct_classes = lct_classes
        self.preload_to_ram = preload_to_ram
        self.normalise = normalise

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
                "training-data", f"{data_dir}-T{size[0]:04d}-R{int(rotate):03d}"
            )
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Dataset directory {data_dir} not found in training-data."
            )

        self.data_dir = data_dir

        self.h5_path = save_dataset_to_h5(
            self.data_dir,
            geoinfo_keys=self.geoinfo_keys,
            lct_classes=self.lct_classes,
            normalise=self.normalise,
        )
        if preload_to_ram:
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
        # Else, load data from metadata
        # h5_content = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        # return {
        #     "target_image": torch.from_numpy(h5_content["target_image"][idx]),
        #     "geoinfo_spatial": torch.from_numpy(h5_content["geoinfo_spatial"][idx]),
        #     "geoinfo_vector": torch.from_numpy(h5_content["geoinfo_vector"][idx]),
        #     "tile_id": h5_content["tile_id"][idx].decode("utf-8"),
        # }
        meta_file = self.tiles[idx]
        data = load_data_from_metadata(
            self.data_dir,
            meta_file,
            size=self.size,
            geoinfo_keys=self.geoinfo_keys,
            lct_classes=self.lct_classes,
            normalise=self.normalise,
        )
        data["tile_id"] = meta_file.replace("_met.json", "")
        return data


def load_data_from_metadata(
    input_dir, meta_file, size, geoinfo_keys=None, lct_classes=20, normalise=True
):
    with open(os.path.join(input_dir, meta_file)) as f:
        meta = json.load(f)

    # RGB
    rgb_path = os.path.join(input_dir, meta["tiles"]["rgb"])
    with rasterio.open(rgb_path) as src:
        rgb = src.read(out_shape=(3, *size)).astype(np.float32) / 255.0

    # DEM
    dem_path = os.path.join(input_dir, meta["tiles"]["dem"])
    with rasterio.open(dem_path) as src:
        dem = src.read(1, out_shape=(1, *size)).astype(np.float32)
        dem, meta["dem_mean"], meta["dem_min"], meta["dem_max"] = normalise_dem(dem)

    # Land cover type
    lct_path = os.path.join(input_dir, meta["tiles"]["lct"])
    with rasterio.open(lct_path) as src:
        lc = src.read(1, out_shape=size).astype(np.float32)[None, ...]
        lc = lc / lct_classes

    # Cloud mask
    cld_path = os.path.join(input_dir, meta["tiles"]["cld"])
    with rasterio.open(cld_path) as src:
        cloud_mask = src.read(1, out_shape=(1, *size))[None, ...]

    # Scalar information
    if geoinfo_keys is not None:
        geoinfo_vector = [meta[k] for k in geoinfo_keys]
        if normalise:
            # normalise month to [0, 1]
            if "month" in geoinfo_keys:
                lat = meta["centre_lat"]
                month = geoinfo_vector[geoinfo_keys.index("month")] + 6 * int(lat < 0)
                month = np.mod(month, 12)
                month = month + 12 * int(month <= 0)
                month = month / 12.0
                geoinfo_vector[geoinfo_keys.index("month")] = month
            # normalise solar azimuth to [0, 1]
            if "solar_azimuth" in geoinfo_keys:
                geoinfo_vector[geoinfo_keys.index("solar_azimuth")] = (
                    np.mod(
                        geoinfo_vector[geoinfo_keys.index("solar_azimuth")]
                        + meta["north_dir"],
                        360,
                    )
                    / 360.0
                )
            # normalise solar zenith to [0, 1]
            if "solar_zenith" in geoinfo_keys:
                geoinfo_vector[geoinfo_keys.index("solar_zenith")] = (
                    geoinfo_vector[geoinfo_keys.index("solar_zenith")] / 90.0
                )
            # normalise solar zenith to [0, 1]
            if "north_dir" in geoinfo_keys:
                geoinfo_vector[geoinfo_keys.index("north_dir")] = (
                    geoinfo_vector[geoinfo_keys.index("north_dir")] / 360.0
                )

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
        "tile_id": meta_file.replace("_met.json", ""),
    }


def save_dataset_to_h5(
    input_dir,
    output_path=None,
    geoinfo_keys=None,
    lct_classes=20,
    normalise=True,
    overwrite=False,
    be_quiet=False,
):
    if output_path is None:
        base = os.path.basename(os.path.normpath(input_dir))
        if geoinfo_keys is not None:
            geoinfo_key_name = "-K" + "_".join(geoinfo_keys)
        else:
            geoinfo_key_name = ""
        if normalise:
            normalise_key_name = "-N"
        else:
            normalise_key_name = ""

        output_name = f"{base}-L{lct_classes}{normalise_key_name}{geoinfo_key_name}.h5"
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
    num_samples = len(json_files)
    if num_samples == 0:
        raise ValueError("No .json metadata files found in input directory")

    dummy_tile = load_data_from_metadata(
        input_dir,
        json_files[0],
        size=(256, 256),
        geoinfo_keys=geoinfo_keys,
        normalise=normalise,
    )

    geoinfo_spatial_dim = dummy_tile["geoinfo_spatial"].numpy().shape[0]
    geoinfo_vector_dim = len(geoinfo_keys) if geoinfo_keys is not None else 1

    # Open HDF5 file for incremental writing
    with h5py.File(output_path, "w") as h5f:
        # Create empty datasets with appropriate shapes
        h5f.create_dataset(
            "target_image",
            shape=(num_samples, 3, 256, 256),  # Adjust shape as needed
            dtype=np.float32,
            compression="gzip",
            chunks=(1, 3, 256, 256),
        )
        h5f.create_dataset(
            "geoinfo_spatial",
            shape=(num_samples, geoinfo_spatial_dim, 256, 256),
            dtype=np.float32,
            compression="gzip",
            chunks=(1, geoinfo_spatial_dim, 256, 256),
        )
        h5f.create_dataset(
            "geoinfo_vector",
            shape=(num_samples, geoinfo_vector_dim),
            dtype=np.float32,
            compression="gzip",
            chunks=(1, geoinfo_vector_dim),
        )

        dt = h5py.string_dtype(encoding="utf-8", length=10)
        h5f.create_dataset(
            "tile_id",
            shape=(num_samples,),
            dtype=dt,
        )

        # Process and write data incrementally
        for i, meta_file in enumerate(tqdm(json_files, desc="Processing tiles")):
            tile = load_data_from_metadata(
                input_dir,
                meta_file,
                size=(256, 256),
                geoinfo_keys=geoinfo_keys,
                normalise=normalise,
            )

            # Write each sample to the HDF5 file
            h5f["target_image"][i] = tile["target_image"].numpy()
            h5f["geoinfo_spatial"][i] = tile["geoinfo_spatial"].numpy()
            h5f["geoinfo_vector"][i] = tile["geoinfo_vector"].numpy()
            h5f["tile_id"][i] = meta_file.replace("_met.json", "")

    if not be_quiet:
        print(f"Compressed dataset saved to {output_path}")

    return output_path
