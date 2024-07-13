import json
from pathlib import Path
from typing import Any

import modal

from . import common
from .common import COLLECTION


DEFAULT_CLAY_VERSION = "v0.5.7"
DEFAULT_CLAY_CKPT = "mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"

BANDS = ["blue", "green", "red", "nir"]

BATCH_SIZES = {"h100": 768, "a100-80gb": 768, "a100": 384, "a10g": 192}
GPU = "a10g"
BATCH_SIZE = BATCH_SIZES[GPU]


def download_model(clay_version=DEFAULT_CLAY_VERSION, clay_ckpt=DEFAULT_CLAY_CKPT):
    from src.model import ClayMAEModule

    model_url = f"https://clay-model-ckpt.s3.amazonaws.com/{clay_version}/{clay_ckpt}"
    ClayMAEModule.load_from_checkpoint(
        model_url,
        metadata_path="/model/configs/metadata.yaml",
        shuffle=False,
        mask_ratio=0,
    )


image = common.image.run_function(download_model).copy_local_file(
    Path(__file__).parent.parent / "data" / "stac.json",
    remote_path="/root/data/stac.json",
)


app = modal.App(common.EMBED_APP, image=image)


STAC_API = "https://earth-search.aws.element84.com/v1"


def prep_stac(input_stac: dict[str, Any]):
    """Handles raw stac input and loads pixel array."""
    import pystac
    import stackstac
    from rasterio.enums import Resampling

    if isinstance(input_stac, dict):
        input_stac = pystac.Item.from_dict(input_stac)

    stack = stackstac.stack(
        [input_stac],
        assets=BANDS,
        dtype="float32",
        rescale=False,
        fill_value=0,
        resampling=Resampling.nearest,
    )

    array = stack.compute().to_numpy()

    return array[0], input_stac


@app.cls(gpu=GPU)
class ClayEmbeddings:
    @modal.enter()
    def load(self):
        import torch

        from src.model import ClayMAEModule

        print("🌐 loading model")
        torch.set_default_device("cuda")
        model = ClayMAEModule.load_from_checkpoint(
            torch.hub.get_dir() + "/checkpoints/" + DEFAULT_CLAY_CKPT,
            metadata_path="/model/configs/metadata.yaml",
            shuffle=False,
            mask_ratio=0,
        )
        model.eval()
        model.half()  # lower precision

        self.model = model.to("cuda")

    @modal.method()
    def embed(self, input_stac):
        print("🌎 loading input data")
        array, input_stac = prep_stac(input_stac)
        print("🌎 prepping inputs for model")
        inputs_datacube = prep_datacube(array, input_stac)
        print("🌎 running model")
        embeddings = self.embed_datacube(inputs_datacube)
        print("🌎 done")
        return embeddings.tolist(), input_stac.to_dict()

    def embed_datacube(self, datacube):
        """Calculates the average embedding of a cube of satellite data.

        Uses Welford's online algorithm across batches to compute the average."""
        import torch

        average = torch.zeros(768)  # EMBEDDING_DIM  # TODO: common
        ct = 0

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                for ii in range(datacube["pixels"].shape[0] // BATCH_SIZE + 1):
                    print(f"running batch {ii}")
                    start = ii * BATCH_SIZE
                    end = start + BATCH_SIZE
                    batch = batch_datacube(datacube, start, end)
                    actual_batch_sz = batch["pixels"].shape[0]
                    unmsk_patch, *_ = self.model.model.encoder(batch)
                    ct += actual_batch_sz
                    delta_average = unmsk_patch[:, 0, :].mean(dim=0) - average
                    average += delta_average * actual_batch_sz / ct

        # return all of batch, first (class) token, all dimensions
        return average


def prep_datacube(array, input_stac):
    import yaml

    from box import Box
    import numpy as np
    import torch
    from torchvision.transforms import v2 as ttv2

    gsd = input_stac.assets["blue"].extra_fields["gsd"]  # all sensors have the same GSD

    metadata = Box(yaml.safe_load(open("/model/configs/metadata.yaml")))

    mean, std, waves = [], [], []
    for band in BANDS:
        mean.append(metadata[COLLECTION].bands.mean[band])
        std.append(metadata[COLLECTION].bands.std[band])
        waves.append(metadata[COLLECTION].bands.wavelength[band])

    transform = ttv2.Compose([ttv2.Normalize(mean=mean, std=std)])

    pixels = torch.tensor(reshape_to_tiles(array, tile_size=224), dtype=torch.float16)
    pixels = transform(pixels)

    zero = np.zeros(shape=(pixels.shape[0], 4), dtype=np.float16)

    datacube = {
        "platform": COLLECTION,
        "time": torch.tensor(zero, dtype=torch.float16, device="cuda"),
        "latlon": torch.tensor(zero, dtype=torch.float16, device="cuda"),
        "pixels": pixels.to("cuda"),
        "gsd": torch.tensor(gsd, dtype=torch.float16, device="cuda"),
        "waves": torch.tensor(waves, dtype=torch.float16, device="cuda"),
    }

    return datacube


def batch_datacube(datacube, start, end):
    import torch

    batch = {
        key: (
            value[start:end]
            if isinstance(value, torch.Tensor) and value.ndim > 1
            else value
        )
        for key, value in datacube.items()
    }
    return batch


def reshape_to_tiles(array, tile_size):
    """
    Reshapes a 3D array (C, H, W) into tiles of given size and returns a new array (B, C, tile_size, tile_size).

    Parameters:
        array (np.ndarray): Input array of shape (C, H, W).
        tile_size (int): Size of the tiles to extract.

    Returns:
        np.ndarray: Reshaped array of shape (B, C, tile_size, tile_size).
    """
    import numpy as np

    if len(array.shape) != 3:
        raise ValueError("Input array must be 3-dimensional")

    C, H, W = array.shape
    num_tiles_x = H // tile_size
    num_tiles_y = W // tile_size

    tiles = []

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            tile = array[
                :,
                i * tile_size : (i + 1) * tile_size,
                j * tile_size : (j + 1) * tile_size,
            ]
            tiles.append(tile)

    tiles_array = np.array(tiles)
    B = tiles_array.shape[0]
    reshaped_array = tiles_array.reshape(B, C, tile_size, tile_size)

    return reshaped_array


# TODO: test run on mock data?
@app.local_entrypoint()
def main():
    embedding_engine = ClayEmbeddings()

    input_stac = json.loads(
        (Path(__file__).parent.parent / "data" / "stac.json").read_text()
    )

    print(embedding_engine.embed.remote(input_stac)[0])
