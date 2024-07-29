import json
from datetime import date
from typing import Any, Optional

import modal
from pydantic import BaseModel

from .common import CLIENT_APP

mongo_client_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pymongo[srv]"
)

app = modal.App(
    CLIENT_APP,
    secrets=[modal.Secret.from_name("my-mongodb-secret")],
    image=mongo_client_image,
)

with mongo_client_image.imports():
    import pymongo


@app.cls()
class MongoClient:
    @modal.enter()
    def connect(self):
        import os
        import urllib

        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        user, password = map(
            urllib.parse.quote_plus,
            (os.environ["MONGO_USER"], os.environ["MONGO_PASSWORD"]),
        )
        host = os.environ["MONGO_HOST"]

        uri = f"mongodb+srv://{user}:{password}@{host}/"

        self.client = MongoClient(uri, server_api=ServerApi("1"))

        try:
            self.client.admin.command("ping")
            print("🍃 connected to MongoDB")
        except Exception as e:
            print(e)

    @modal.exit()
    def disconnect(self):
        self.client.close()

    @modal.method()
    def insert_many(
        self,
        documents: list[dict[str, Any]],
        collection="test-clay",
        db="modal-examples",
    ):
        self.client[db][collection].insert_many(documents)

    @modal.method()
    def add_stacs(self, stac_documents, collection="test-clay", db="modal-examples"):
        filtered_documents = []
        for document in stac_documents:
            if self.client[db][collection].find_one({"id": document["id"]}):
                print(f"🍃 skipping duplicate document with id {document['id']}")
                continue
            filtered_documents.append(document)
        if filtered_documents:
            self.insert_many.local(filtered_documents, collection, db)

    @modal.method()
    def add_aoi(
        self, aoi_document_json, collection="test-clay-aoi", db="modal-examples"
    ):
        self.client[db][collection].insert_one(aoi_document_json)

    @modal.method()
    def get_aois(self, query=None, collection="test-clay-aoi", db="modal-examples"):
        import bson

        if query is None:
            query = {}
        for row in self.client[db][collection].find(query):
            yield json.loads(bson.json_util.dumps(row))

    @modal.method()
    def add_index(
        self, index_property, index_type, collection="test-clay", db="modal-examples"
    ):
        kwargs = {}
        if index_type == "geosphere":
            index = (index_property, pymongo.GEOSPHERE)
        elif index_type == "unique":
            index = (index_property, pymongo.ASCENDING)
            kwargs |= {"unique": True}
        else:
            raise ValueError(f"unknown index_type {index_type}")
        self.client[db][collection].create_index([index], **kwargs)

    @modal.method()
    def find(self, filter=None, collection="test-clay", db="modal-examples"):
        import bson

        for row in self.client[db][collection].find(filter=filter):
            print(f"found row {row['id']}")
            yield json.loads(bson.json_util.dumps(row))

    @modal.method()
    def bulk_update(
        self,
        updates: list[tuple[dict, dict]],
        collection="test-clay",
        db="modal-examples",
    ):
        updates = [pymongo.UpdateOne(*update) for update in updates]
        result = self.client[db][collection].bulk_write(updates)
        print(
            f"🍃 matched {result.matched_count} documents and modified {result.modified_count} documents."
        )

    @modal.method()
    def search(self, pipeline, collection="test-clay", db="modal-examples"):
        import bson

        results = self.client[db][collection].aggregate(pipeline)
        for result in results:
            yield json.loads(bson.json_util.dumps(result))


class GeoSearchRequest(BaseModel):
    lon: float
    lat: float


class VectorSearchRequest(BaseModel):
    vector: list[float]


@app.function(keep_warm=1)
@modal.web_endpoint(method="POST", docs=True)
def geo_search(
    request: GeoSearchRequest,
    since: Optional[date] = None,
    before: Optional[date] = None,
    limit: Optional[int] = None,
):
    lon, lat = request.lon, request.lat
    print(f"🍃 running geographic search near {lat}, {lon}")
    if since is None:
        since = "0000-01-01"
    if before is None:
        before = "9999-12-31"
    print(f"🍃 running search across period {since} - {before}")
    if limit is None:
        limit = 32
    limit = max(min(limit, 32), 1)
    pipeline = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [lon, lat]},
                "distanceField": "distance",
            }
        },
        {"$match": {"properties.created": {"$gte": str(since), "$lte": str(before)}}},
        {"$match": {"vector": {"$ne": None}}},
        {"$limit": limit},
        {
            "$project": {
                "id": 1,
                "bbox": 1,
                "vector": 1,
                "link": {
                    "$arrayElemAt": [
                        {
                            "$filter": {
                                "input": "$links",
                                "as": "link",
                                "cond": {"$eq": ["$$link.rel", "thumbnail"]},
                            }
                        },
                        0,
                    ]
                },
                "distance": 1,
            }
        },
        {
            "$project": {
                "id": 1,
                "bbox": 1,
                "vector": 1,
                "url": "$link.href",
                "distance": 1,
                "lon": {"$arrayElemAt": ["$bbox", 0]},
                "lat": {"$arrayElemAt": ["$bbox", 1]},
            }
        },
    ]

    results = list(MongoClient().search.local(pipeline))

    return {"results": results}


@app.function(keep_warm=1)
@modal.web_endpoint(method="POST", docs=True)
def vector_search(
    request: VectorSearchRequest,
    since: Optional[date] = None,
    before: Optional[date] = None,
    limit: Optional[int] = None,
):
    vector = request.vector
    print(f"🍃 running vector search with vector {vector[:10]}")
    if since is None:
        since = "0000-01-01"
    if before is None:
        before = "9999-12-31"
    print(f"🍃 running search across period {since} - {before}")
    if limit is None:
        limit = 32
    limit = max(min(limit, 32), 1)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector",
                "queryVector": vector,
                "exact": True,
                "limit": limit,
            }
        },
        {"$match": {"properties.created": {"$gte": str(since), "$lte": str(before)}}},
        {"$match": {"vector": {"$ne": None}}},
        {
            "$project": {
                "id": 1,
                "bbox": 1,
                "vector": 1,
                "link": {
                    "$arrayElemAt": [
                        {
                            "$filter": {
                                "input": "$links",
                                "as": "link",
                                "cond": {"$eq": ["$$link.rel", "thumbnail"]},
                            }
                        },
                        0,
                    ]
                },
                "score": {"$meta": "vectorSearchScore"},
            }
        },
        {
            "$project": {
                "id": 1,
                "bbox": 1,
                "vector": 1,
                "url": "$link.href",
                "score": 1,
                "lon": {"$arrayElemAt": ["$bbox", 0]},
                "lat": {"$arrayElemAt": ["$bbox", 1]},
            }
        },
    ]

    results = list(MongoClient().search.local(pipeline))

    return {"results": results}


@app.local_entrypoint()
def main(action: str, target: str = None):
    import json
    from pathlib import Path

    if action == "add_aoi":
        if target is None:
            target = Path(__file__).parent.parent / "data" / "california.geojson"

        aoi_str = Path(target).read_text()
        aoi_document_json = json.loads(aoi_str)
        MongoClient.add_aoi.remote(aoi_document_json)
    elif action == "add_index":
        property, type = target.split("$")
        MongoClient.add_index.remote(property, type)
    elif action == "find":
        for element in MongoClient.find.remote_gen(filter=json.loads(target)):
            print(element)
    elif action == "search":
        target_path = Path(target)
        pipeline = json.loads(target_path.read_text())
        with open(Path("tmp") / f"result-{target_path.stem}.jsonl", "w") as f:
            for element in MongoClient.search.remote_gen(pipeline=pipeline):
                print(element)
                f.write(json.dumps(element) + "\n")
            print(f"saved result to {f.name}")
    else:
        print(f"unknown action {action}")
