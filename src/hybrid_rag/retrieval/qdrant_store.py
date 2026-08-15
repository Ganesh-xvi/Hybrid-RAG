import uuid

from langchain_core.documents import Document
from qdrant_client.http import models as qmodels

from hybrid_rag.config.settings import Settings, get_settings
from hybrid_rag.utils.embeddings import embed_documents
from hybrid_rag.utils.logging import logger
from hybrid_rag.utils.qdrant import encode_sparse_documents, get_qdrant_client


class QdrantStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = get_qdrant_client(self.settings.qdrant_url)
        self.collection = self.settings.qdrant_collection
        self.dense_name = self.settings.qdrant_dense_vector_name
        self.sparse_name = self.settings.qdrant_sparse_vector_name
        self._dense_dim: int | None = None

    def ensure_collection(self, dense_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.dense_name: qmodels.VectorParams(
                    size=dense_dim,
                    distance=qmodels.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.sparse_name: qmodels.SparseVectorParams()
            },
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="company",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection %s (dense_dim=%s)", self.collection, dense_dim)

    def upsert_chunks(self, chunks: list[Document], batch_size: int | None = None) -> int:
        if not chunks:
            return 0

        batch = batch_size or self.settings.qdrant_upsert_batch_size
        texts = [chunk.page_content for chunk in chunks]
        dense_vectors = embed_documents(texts, self.settings)
        if not self._dense_dim:
            self._dense_dim = len(dense_vectors[0])
        self.ensure_collection(self._dense_dim)

        total = 0
        for start in range(0, len(chunks), batch):
            batch_chunks = chunks[start : start + batch]
            batch_dense = dense_vectors[start : start + batch]
            batch_sparse = encode_sparse_documents(
                [c.page_content for c in batch_chunks], self.settings
            )
            points = []
            for chunk, dense, sparse in zip(batch_chunks, batch_dense, batch_sparse, strict=True):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["chunk_id"]))
                payload = {
                    "text": chunk.page_content,
                    **chunk.metadata,
                }
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector={
                            self.dense_name: dense,
                            self.sparse_name: sparse,
                        },
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    def point_count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        info = self.client.get_collection(self.collection)
        return int(info.points_count or 0)

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def _build_filter(self, company: str | None) -> qmodels.Filter | None:
        if not company:
            return None
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="company",
                    match=qmodels.MatchValue(value=company),
                )
            ]
        )

    def _points_to_documents(self, points: list) -> list[Document]:
        documents: list[Document] = []
        for point in points:
            payload = point.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("text", ""),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )
        return documents

    def dense_search(
        self,
        query: str,
        k: int | None = None,
        company: str | None = None,
    ) -> list[Document]:
        limit = k or self.settings.dense_k
        dense_vector = embed_documents([query], self.settings)[0]
        results = self.client.query_points(
            collection_name=self.collection,
            query=dense_vector,
            using=self.dense_name,
            query_filter=self._build_filter(company),
            limit=limit,
            with_payload=True,
        )
        return self._points_to_documents(results.points)

    def sparse_search(
        self,
        query: str,
        k: int | None = None,
        company: str | None = None,
    ) -> list[Document]:
        from hybrid_rag.utils.qdrant import encode_sparse_query

        limit = k or self.settings.sparse_k
        sparse_vector = encode_sparse_query(query, self.settings)
        results = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using=self.sparse_name,
            query_filter=self._build_filter(company),
            limit=limit,
            with_payload=True,
        )
        return self._points_to_documents(results.points)

    def hybrid_search(
        self,
        query: str,
        k: int | None = None,
        company: str | None = None,
    ) -> list[Document]:
        from hybrid_rag.utils.qdrant import encode_sparse_query

        limit = k or self.settings.hybrid_k
        dense_vector = embed_documents([query], self.settings)[0]
        sparse_vector = encode_sparse_query(query, self.settings)
        payload_filter = self._build_filter(company)

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_vector,
                    using=self.dense_name,
                    filter=payload_filter,
                    limit=limit,
                ),
                qmodels.Prefetch(
                    query=sparse_vector,
                    using=self.sparse_name,
                    filter=payload_filter,
                    limit=limit,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return self._dedupe_documents(self._points_to_documents(results.points))[:limit]

    @staticmethod
    def _dedupe_documents(documents: list[Document]) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique.append(doc)
        return unique
