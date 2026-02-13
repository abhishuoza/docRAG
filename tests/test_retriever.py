"""Tests for the DocRetriever module."""

import uuid

import chromadb
from langchain_core.documents import Document

from docrag.retriever import DocRetriever


def _make_retriever() -> DocRetriever:
    """Create a retriever with an in-memory ChromaDB client and unique collection."""
    client = chromadb.Client()
    return DocRetriever(
        chroma_client=client,
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        chunk_size=200,
        chunk_overlap=50,
    )


def _sample_docs() -> list[Document]:
    return [
        Document(
            page_content=(
                "FastAPI is a modern web framework for building APIs with Python. "
                "It is based on standard Python type hints and provides automatic "
                "interactive API documentation."
            ),
            metadata={"source": "https://fastapi.tiangolo.com", "title": "FastAPI"},
        ),
        Document(
            page_content=(
                "PyTorch is an open source machine learning framework. "
                "It provides tensor computation with strong GPU acceleration "
                "and deep neural networks built on autograd."
            ),
            metadata={"source": "https://pytorch.org/docs", "title": "PyTorch"},
        ),
    ]


class TestDocRetriever:
    def test_index_documents(self):
        retriever = _make_retriever()
        count = retriever.index_documents(_sample_docs())
        assert count > 0

    def test_search_returns_relevant_results(self):
        retriever = _make_retriever()
        retriever.index_documents(_sample_docs())

        results = retriever.search("web framework for APIs", top_k=2)
        assert len(results) > 0
        assert any("FastAPI" in r.text or "API" in r.text for r in results)

    def test_search_empty_store(self):
        retriever = _make_retriever()
        results = retriever.search("anything", top_k=3)
        assert results == []

    def test_get_stats(self):
        retriever = _make_retriever()
        retriever.index_documents(_sample_docs())

        stats = retriever.get_stats()
        assert stats["total_chunks"] > 0
        assert stats["collection_name"].startswith("test_")

    def test_as_retriever(self):
        retriever = _make_retriever()
        retriever.index_documents(_sample_docs())

        lc_retriever = retriever.as_retriever(search_kwargs={"k": 1})
        docs = lc_retriever.invoke("machine learning")
        assert len(docs) >= 1

    def test_delete_documents(self):
        retriever = _make_retriever()
        retriever.index_documents(_sample_docs())

        before = retriever.get_stats()["total_chunks"]
        retriever.delete_documents("https://fastapi.tiangolo.com")
        after = retriever.get_stats()["total_chunks"]

        assert after < before
