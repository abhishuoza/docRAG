"""Tests for the FastAPI API module."""

import uuid

import chromadb
from fastapi.testclient import TestClient
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docrag.config import Settings
from docrag.pipeline import RAGPipeline


def _make_test_client(tmp_path) -> TestClient:
    """Create a TestClient with in-memory ChromaDB and fake embeddings."""
    import docrag.api as api_module

    settings = Settings(
        cache_dir=tmp_path / "cache",
        chroma_collection=f"test_{uuid.uuid4().hex[:8]}",
    )
    fake_llm = FakeListChatModel(responses=["print('hello')"] * 10)
    fake_embeddings = FakeEmbeddings(size=384)
    client = chromadb.Client()
    api_module.pipeline = RAGPipeline(
        settings=settings,
        llm=fake_llm,
        chroma_client=client,
        embeddings=fake_embeddings,
    )

    return TestClient(api_module.app, raise_server_exceptions=True)


class TestAPI:
    def test_health(self, tmp_path):
        client = _make_test_client(tmp_path)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_stats(self, tmp_path):
        client = _make_test_client(tmp_path)
        response = client.get("/stats")
        assert response.status_code == 200
        assert "total_chunks" in response.json()

    def test_generate(self, tmp_path):
        client = _make_test_client(tmp_path)
        response = client.post("/generate", json={
            "query": "hello world",
            "doc_urls": [],
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "hello world"
        assert "generated_code" in data

    def test_clear_cache(self, tmp_path):
        client = _make_test_client(tmp_path)
        response = client.delete("/cache")
        assert response.status_code == 200
        assert "cleared" in response.json()
