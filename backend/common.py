from datetime import date

import modal
import pydantic

image = (
    modal.Image.micromamba(python_version="3.11")
    .apt_install("git")
    .run_commands(
        [
            "git clone --depth=1 --branch v1.0 https://github.com/Clay-foundation/model",
            "micromamba install -n base -y --file model/environment.yml",
        ],
        gpu="a10g",
    )
    .env({"PYTHONPATH": "/model/:/pkg/:/root/"})
    .pip_install("pydantic >= 2")
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
