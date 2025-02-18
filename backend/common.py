from datetime import date

import modal
import pydantic

GIT_SHA = "0a0b337a2309bd05803737b9970f688d754482f2"

image = (
    modal.Image.micromamba(python_version="3.11")
    .apt_install("git")
    .run_commands(
        [
            "git clone --branch main https://github.com/Clay-foundation/model",
            f"cd model && git checkout {GIT_SHA}",
            "micromamba install -n base -y --file model/environment.yml",
        ],
        gpu="a10g",
    )
    .env({"PYTHONPATH": "/model/:/pkg/:/root/"})
    .pip_install(
        "pystac==1.12.1", "pystac-client==0.8.6", "stackstac==0.5.1", "rasterio==1.4.3", "pydantic >= 2"
    )
)

COLLECTION = "sentinel-2-l2a"
EMBED_APP = f"clay-{COLLECTION}-embed"
EXTRACT_APP = f"clay-{COLLECTION}-extract"
CLIENT_APP = "clay-mongo-client"


class GeographicCoordinates(pydantic.BaseModel):
    latitude: float = pydantic.Field(..., ge=-90, le=90)
    longitude: float = pydantic.Field(..., ge=-180, le=180)


class TimeRange(pydantic.BaseModel):
    start: date
    end: date
