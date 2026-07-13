from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, FilterSelector, Filter, FieldCondition, MatchValue

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs_nvidia"):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

    def upsert(self, ids, vectors, payloads):
        if not vectors:
            return
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        if not self.client.collection_exists(self.collection):
            return {"contexts": [], "sources": []}
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        ).points  # <-- .points to get the list

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def delete_source(self, source: str):
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        ),
                    ],
                )
            ),
        )

    def get_all_sources(self):
        # We can use the scroll API to fetch all payloads and aggregate them
        if not self.client.collection_exists(self.collection):
            return {}
        sources = {}
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for r in records:
                source = r.payload.get("source") if r.payload else None
                if source:
                    sources[source] = sources.get(source, 0) + 1
            
            if next_offset is None:
                break
            offset = next_offset
            
        return sources
