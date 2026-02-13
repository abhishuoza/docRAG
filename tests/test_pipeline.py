"""Tests for the RAGPipeline module."""

import uuid

import chromadb
import responses
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docrag.config import Settings
from docrag.pipeline import RAGPipeline

SAMPLE_HTML = """
<html><head><title>Test Doc</title></head>
<body><main><p>FastAPI is a web framework for building APIs.</p></main></body>
</html>
"""


def _make_pipeline(tmp_path) -> RAGPipeline:
    """Create a pipeline with in-memory ChromaDB and fake embeddings for fast tests."""
    settings = Settings(
        cache_dir=tmp_path / "cache",
        chroma_collection=f"test_{uuid.uuid4().hex[:8]}",
    )
    fake_llm = FakeListChatModel(responses=["Generated: `app = FastAPI()`"] * 10)
    fake_embeddings = FakeEmbeddings(size=384)
    client = chromadb.Client()
    return RAGPipeline(
        settings=settings,
        llm=fake_llm,
        chroma_client=client,
        embeddings=fake_embeddings,
    )


class TestRAGPipeline:
    @responses.activate
    def test_index_urls(self, tmp_path):
        responses.add(responses.GET, "https://example.com/docs", body=SAMPLE_HTML, status=200)

        pipeline = _make_pipeline(tmp_path)
        count = pipeline.index_urls(["https://example.com/docs"])

        assert count > 0

    @responses.activate
    def test_search_only(self, tmp_path):
        responses.add(responses.GET, "https://example.com/docs", body=SAMPLE_HTML, status=200)

        pipeline = _make_pipeline(tmp_path)
        pipeline.index_urls(["https://example.com/docs"])

        results = pipeline.search_only("web framework")
        assert len(results) > 0

    @responses.activate
    def test_run_full_pipeline(self, tmp_path):
        responses.add(responses.GET, "https://example.com/docs", body=SAMPLE_HTML, status=200)

        pipeline = _make_pipeline(tmp_path)
        response = pipeline.run(
            query="create an API",
            urls=["https://example.com/docs"],
        )

        assert response.query == "create an API"
        assert response.generated_code is not None
        assert len(response.generated_code) > 0

    def test_run_without_urls(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        response = pipeline.run(query="anything", urls=[])

        assert response.query == "anything"
        # No chunks at all → low_relevance should be True
        assert response.low_relevance is True

    def test_low_relevance_flag(self, tmp_path):
        from docrag.models import RetrievedChunk

        pipeline = _make_pipeline(tmp_path)

        # All scores above threshold (1.0) → low relevance
        bad_chunks = [
            RetrievedChunk(text="x", source_url="http://a.com", score=1.5, chunk_index=0),
            RetrievedChunk(text="y", source_url="http://b.com", score=1.6, chunk_index=1),
        ]
        assert pipeline._check_relevance(bad_chunks) is True

        # At least one score below threshold → relevant
        good_chunks = [
            RetrievedChunk(text="x", source_url="http://a.com", score=0.3, chunk_index=0),
            RetrievedChunk(text="y", source_url="http://b.com", score=1.2, chunk_index=1),
        ]
        assert pipeline._check_relevance(good_chunks) is False
