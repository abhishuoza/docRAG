"""Tests for the DocCache module."""

from datetime import datetime, timedelta

from docrag.cache import DocCache
from docrag.models import ScrapedDocument


def _make_doc(url="https://example.com/docs", content="Hello world"):
    return ScrapedDocument(url=url, title="Test", content=content)


class TestDocCache:
    def test_set_and_get(self, tmp_path):
        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        doc = _make_doc()
        cache.set(doc.url, doc)

        result = cache.get(doc.url)
        assert result is not None
        assert result.url == doc.url
        assert result.content == doc.content

    def test_get_missing_returns_none(self, tmp_path):
        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        assert cache.get("https://nonexistent.com") is None

    def test_ttl_expiry(self, tmp_path):
        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        doc = ScrapedDocument(
            url="https://example.com",
            title="Old",
            content="stale",
            scraped_at=datetime.now() - timedelta(hours=2),
        )
        cache.set(doc.url, doc)

        assert cache.get(doc.url) is None

    def test_invalidate(self, tmp_path):
        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        doc = _make_doc()
        cache.set(doc.url, doc)

        assert cache.invalidate(doc.url) is True
        assert cache.get(doc.url) is None

    def test_invalidate_missing(self, tmp_path):
        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        assert cache.invalidate("https://nope.com") is False

    def test_clear(self, tmp_path):
        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        cache.set("https://a.com", _make_doc(url="https://a.com"))
        cache.set("https://b.com", _make_doc(url="https://b.com"))

        assert cache.clear() == 2
        assert cache.get("https://a.com") is None

    def test_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache"
        DocCache(cache_dir, ttl_hours=1)
        assert cache_dir.exists()
