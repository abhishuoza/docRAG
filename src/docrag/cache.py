"""JSON file-based cache with TTL for scraped documents."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from docrag.models import ScrapedDocument


class DocCache:
    """File-system cache keyed by SHA256(url) with configurable TTL."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def _path(self, url: str) -> Path:
        return self.cache_dir / f"{self._key(url)}.json"

    def get(self, url: str) -> ScrapedDocument | None:
        """Return cached document if it exists and hasn't expired."""
        path = self._path(url)
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        doc = ScrapedDocument(**data)

        if datetime.now() - doc.scraped_at > self.ttl:
            path.unlink()
            return None

        return doc

    def set(self, url: str, document: ScrapedDocument) -> None:
        """Write a document to the cache."""
        path = self._path(url)
        path.write_text(document.model_dump_json(indent=2))

    def invalidate(self, url: str) -> bool:
        """Remove a single URL from cache. Returns True if it existed."""
        path = self._path(url)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Remove all cached files. Returns count of files removed."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count
