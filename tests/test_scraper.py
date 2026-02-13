"""Tests for the DocScraper module."""

import responses

from docrag.cache import DocCache
from docrag.models import ScrapedDocument
from docrag.scraper import DocScraper

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
  <nav>Navigation links</nav>
  <header>Site header</header>
  <main>
    <h1>Documentation Title</h1>
    <p>This is the main documentation content.</p>
    <pre><code>print("hello")</code></pre>
  </main>
  <footer>Site footer</footer>
  <script>var x = 1;</script>
</body>
</html>
"""

MINIMAL_HTML = """
<html><head><title>Minimal</title></head>
<body><p>Just body content here.</p></body>
</html>
"""


class TestDocScraperExtract:
    """Tests for the _extract_content static method."""

    def test_extracts_main_content(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
        content = DocScraper._extract_content(soup)

        assert "Documentation Title" in content
        assert "main documentation content" in content

    def test_strips_nav_footer_script(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
        content = DocScraper._extract_content(soup)

        assert "Navigation links" not in content
        assert "Site footer" not in content
        assert "var x = 1" not in content

    def test_falls_back_to_body(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(MINIMAL_HTML, "html.parser")
        content = DocScraper._extract_content(soup)

        assert "Just body content here" in content


class TestDocScraperScrape:
    """Tests for scraping URLs (HTTP mocked with responses library)."""

    @responses.activate
    def test_scrape_url(self):
        responses.add(
            responses.GET,
            "https://example.com/docs",
            body=SAMPLE_HTML,
            status=200,
        )

        scraper = DocScraper()
        doc = scraper.scrape_url("https://example.com/docs")

        assert doc.url == "https://example.com/docs"
        assert doc.title == "Test Page"
        assert isinstance(doc, ScrapedDocument)

    @responses.activate
    def test_scrape_url_uses_cache(self, tmp_path):
        responses.add(
            responses.GET,
            "https://example.com/docs",
            body=SAMPLE_HTML,
            status=200,
        )

        cache = DocCache(tmp_path / "cache", ttl_hours=1)
        scraper = DocScraper(cache=cache)

        doc1 = scraper.scrape_url("https://example.com/docs")
        doc2 = scraper.scrape_url("https://example.com/docs")

        assert doc1.content == doc2.content
        # Only one HTTP call should have been made
        assert len(responses.calls) == 1

    @responses.activate
    def test_scrape_docs_skips_failures(self):
        responses.add(responses.GET, "https://good.com", body=MINIMAL_HTML, status=200)
        responses.add(responses.GET, "https://bad.com", body="error", status=500)

        scraper = DocScraper()
        docs = scraper.scrape_docs(["https://good.com", "https://bad.com"])

        assert len(docs) == 1
        assert docs[0].url == "https://good.com"


class TestDocScraperConvert:
    """Tests for converting to LangChain Documents."""

    def test_to_langchain_docs(self):
        scraper = DocScraper()
        scraped = [
            ScrapedDocument(url="https://a.com", title="A", content="Content A"),
            ScrapedDocument(url="https://b.com", title="B", content="Content B"),
        ]

        lc_docs = scraper.to_langchain_docs(scraped)

        assert len(lc_docs) == 2
        assert lc_docs[0].page_content == "Content A"
        assert lc_docs[0].metadata["source"] == "https://a.com"
        assert lc_docs[1].metadata["title"] == "B"
