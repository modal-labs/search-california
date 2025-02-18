import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.8")
    .copy_local_dir(Path(__file__).parent / "static", remote_path="/root/static")
)


app = modal.App("clay-frontend-server", image=image)

web_app = FastAPI()


@web_app.get("/", response_class=HTMLResponse)
async def read_index():
    file_path = os.path.join("static", "index.html")
    with open(file_path, "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)


@app.function(keep_warm=1)
@modal.asgi_app(label="clay-hybrid-search")
def serve():
    web_app.mount("/static", StaticFiles(directory="/root/static"), name="static")

    return web_app
