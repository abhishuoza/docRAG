"""Integration tests — require network access and real embeddings.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import uuid

import chromadb
import pytest

from docrag.cache import DocCache
from docrag.retriever import DocRetriever
from docrag.scraper import DocScraper


@pytest.mark.integration
class TestIntegration:
    def test_scrape_and_retrieve(self, tmp_path):
        """Scrape a real doc page, embed it, and verify retrieval works."""
        # Setup
        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        scraper = DocScraper(cache=cache)
        client = chromadb.Client()
        retriever = DocRetriever(
            chroma_client=client,
            collection_name=f"integration_{uuid.uuid4().hex[:8]}",
            chunk_size=500,
            chunk_overlap=100,
        )

        # Scrape a small, stable documentation page
        url = "https://docs.python.org/3/library/json.html"
        doc = scraper.scrape_url(url)

        assert doc.url == url
        assert len(doc.content) > 100

        # Convert and index
        lc_docs = scraper.to_langchain_docs([doc])
        count = retriever.index_documents(lc_docs)
        assert count > 0

        # Search for something we know is in the JSON docs
        results = retriever.search("parse JSON string", top_k=3)
        assert len(results) > 0
        assert any("json" in r.text.lower() for r in results)

        # Verify cache hit
        cached = cache.get(url)
        assert cached is not None
        assert cached.url == url
