from __future__ import annotations
from typing import Any, Callable
from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document

class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except ImportError:
            self._use_chroma = False

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": self._embedding_fn(doc.content)
        }

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            if self._use_chroma:
                self._collection.add(
                    ids=[record["id"]],
                    documents=[record["content"]],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"]]
                )
            else:
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self.search_with_filter(query, top_k=top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        query_vec = self._embedding_fn(query)
        
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=metadata_filter
            )
            formatted = []
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i]
                })
            return formatted

        candidates = self._store
        if metadata_filter:
            candidates = [
                r for r in self._store 
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
            
        scored = []
        for rec in candidates:
            score = compute_similarity(query_vec, rec["embedding"])
            scored.append({**rec, "score": score})
            
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            count_before = self.get_collection_size()
            self._collection.delete(ids=[doc_id])
            return self.get_collection_size() < count_before
            
        initial_len = len(self._store)
        self._store = [r for r in self._store if r["id"] != doc_id]
        return len(self._store) < initial_len