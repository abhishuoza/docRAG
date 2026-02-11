"""Custom documentation scraper that outputs LangChain Document objects."""

from datetime import datetime

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from docrag.cache import DocCache
from docrag.models import ScrapedDocument


class DocScraper:
    """Scrapes documentation URLs and converts them to LangChain Documents."""

    def __init__(self, cache: DocCache | None = None, timeout: int = 30):
        self.cache = cache
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "DocRAG/0.1 (documentation scraper)"}
        )

    def scrape_url(self, url: str) -> ScrapedDocument:
        """Fetch a URL and extract its documentation content.

        Checks cache first if available. Caches the result after scraping.
        """
        if self.cache:
            cached = self.cache.get(url)
            if cached is not None:
                return cached

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        content = self._extract_content(soup)

        doc = ScrapedDocument(
            url=url,
            title=title,
            content=content,
            scraped_at=datetime.now(),
        )

        if self.cache:
            self.cache.set(url, doc)

        return doc

    def scrape_docs(self, urls: list[str]) -> list[ScrapedDocument]:
        """Scrape multiple URLs, returning all successfully scraped documents."""
        documents = []
        for url in urls:
            try:
                documents.append(self.scrape_url(url))
            except requests.RequestException:
                continue
        return documents

    def to_langchain_docs(self, documents: list[ScrapedDocument]) -> list[Document]:
        """Convert ScrapedDocuments to LangChain Document format."""
        return [
            Document(
                page_content=doc.content,
                metadata={
                    "source": doc.url,
                    "title": doc.title,
                    "scraped_at": doc.scraped_at.isoformat(),
                },
            )
            for doc in documents
        ]

    @staticmethod
    def _extract_content(soup: BeautifulSoup) -> str:
        """Extract the main documentation content from parsed HTML.

        Finds the best content container (article > main > body),
        strips noise elements, and returns clean text.
        """
        article = soup.find("article")
        main = soup.find("main")
        body = soup.find("body")

        if article:
            content = article
        elif main:
            content = main
        else:
            content = body

        for nav in content.find_all(["nav", "footer", "script", "style", "header"]):
            nav.decompose()

        return content.get_text("\n", strip=True)

