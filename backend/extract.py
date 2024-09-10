"""Extracts data from the STAC API to store in the database and enriches database contents with embeddings.

Runs regularly if deployed and can be triggered manually with `modal run`.
"""
from datetime import timedelta
from datetime import date
from typing import Any, Optional

import modal

from . import common
from .common import COLLECTION, CLIENT_APP, EMBED_APP, EXTRACT_APP, TimeRange


app = modal.App(EXTRACT_APP, image=common.image)


STAC_API_URL = "https://earth-search.aws.element84.com/v1"
TARGET_AOI = "california"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.function()
def query_stac_aoi(
    aoi: dict[str, Any],
    dates: TimeRange,
    max_items: Optional[int] = None,
):
    f"""Queries the STAC API at {STAC_API_URL} for data inside a spatiotemporal box.

    That box is defined spatially by an area-of-interest (`aoi`) in GeoJSON format.
    It is defined temporally by a `TimeRange` with a start and an end date.

    The maximum number of items to return can be set by the `max_items` argument.
    """
    import pystac_client

    catalog = pystac_client.Client.open(STAC_API_URL)
    query = catalog.search(
        collections=[COLLECTION],
        datetime=f"{dates.start}/{dates.end}",
        intersects=aoi,
        max_items=max_items,
        limit=100,  # page size
    )

    ct = 0
    results = query.pages()
    for ii, page in enumerate(results):
        print(f"🌐 returning results from page {ii}")
        for row in page:
            if row.datetime.microsecond == 0:
                print(f"skipping row {row}")
                continue
            ct += 1
            yield row.to_dict()
    print(f"🌐 done fetching {ct} results")


@app.function(schedule=modal.Period(days=1))
def refresh_aoi(aoi=TARGET_AOI, since_days: int = 1):
    """Pulls data for an AOI since a certain number of days ago."""
    end = date.today()
    start = end - timedelta(days=since_days)

    main(aoi, start, end)


@app.function(schedule=modal.Period(days=1), timeout=2 * HOURS)
def enrich_vectors():
    """Adds embedding vectors in the field `"vector"` to any data that doesn't have it."""
    MongoClient = modal.Cls.lookup(CLIENT_APP, "MongoClient")
    client = MongoClient()

    ClayEmbeddings = modal.Cls.lookup(EMBED_APP, "ClayEmbeddings")
    model = ClayEmbeddings()

    has_no_vector = {"vector": {"$exists": False}}

    ct = 0
    handles = []
    for resps in batch(
        model.embed.map(
            client.find.remote_gen(filter=has_no_vector),
            order_outputs=False,
            return_exceptions=True,
        ),
    ):
        print("🌐 sending embedding batch to MongoDB")
        updates = []
        for resp in resps:
            if isinstance(resp, Exception):
                print(f"Exception: {resp}")
                continue
            embedding, doc = resp
            ct += 1

            updates.append(({"id": doc["id"]}, {"$set": {"vector": embedding}}))

        if updates:
            handles.append(client.bulk_update.spawn(updates))
            print(f"🌐 sent a total of {ct} embeddings to MongoDB")

    print("🌐 embeddings finished! confirming database ingestion")
    for handle in handles:
        handle.get()

    print(f"🌐 finished! updated {ct} documents with vectors")


@app.local_entrypoint()
def main(aoi: str = None, start: str = None, end: str = None):
    f"""Loads satellite data from {COLLECTION} inside the area-of-interest between start and end dates.

    If not provided, the area of interest is {TARGET_AOI}.

    If no end date is provided, the end date is set to the current date and the start date is set to one week before.

    If an end date is provided, a start date must also be provided.
    """
    if aoi is None:
        aoi = TARGET_AOI

    if end is None:
        end = date.today()
        start = end - timedelta(days=7)
    else:
        assert (
            start is not None
        ), f"if end date is provided, start date must also be provided. end date was {end}"

    print(f"🌐 searching for satellite data from {start} to {end}")
    dates = TimeRange(start=start, end=end)

    MongoClient = modal.Cls.lookup(CLIENT_APP, "MongoClient")
    client = MongoClient()

    try:
        aoi = next(client.get_aois.remote_gen(query={"_id": aoi}))
    except StopIteration as e:
        raise ValueError(f"Area-of-interest with id {aoi} not found") from e
    print(f"🌐 looking inside area-of-interest with id {aoi['_id']}")

    ct, bct, bsz = 0, 0, 100
    batch, handles = [], []
    for resp in query_stac_aoi.remote_gen(aoi, dates):
        batch.append(resp)
        ct += 1
        bct += 1
        if ct >= bsz:
            print("🌐 inserting batch")
            handle = client.add_stacs.spawn(batch)
            handles.append(handle)
            ct, batch = 0, []

    if batch:
        print("🌐 inserting batch")
        handle = client.add_stacs.spawn(batch)
        handles.append(handle)

    for handle in handles:
        handle.get()

    print("🌐 finished ingestion")


def batch(iterator, batch_sz=32, transform=None):
    """Chunks an iterator into batches, optionally applying an elementwise transformation."""
    batch = []
    for elem in iterator:
        elem = transform(elem) if transform else elem
        if len(batch) < batch_sz:
            batch.append(elem)
        else:
            yield batch
            batch = [elem]
    if batch:
        yield batch
