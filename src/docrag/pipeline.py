"""Orchestrates the full RAG pipeline: scrape -> cache -> retrieve -> generate."""

from langchain_core.language_models import BaseChatModel

from docrag.cache import DocCache
from docrag.config import Settings, get_settings
from docrag.models import GenerationResponse, RetrievedChunk
from docrag.scraper import DocScraper


class RAGPipeline:
    """Wires together scraper, cache, retriever, and generator.

    Components are lazily initialized — the LLM only loads when generation
    is requested, and embeddings only load when indexing/searching.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        chroma_client=None,
        embeddings=None,
    ):
        self.settings = settings or get_settings()
        self.cache = DocCache(self.settings.cache_dir, self.settings.cache_ttl_hours)
        self.scraper = DocScraper(cache=self.cache)

        # Store init args for lazy creation
        self._llm = llm
        self._chroma_client = chroma_client
        self._embeddings = embeddings
        self._retriever = None
        self._generator = None

    @property
    def retriever(self):
        """Lazily initialize the retriever (loads embedding model)."""
        if self._retriever is None:
            from docrag.retriever import DocRetriever

            self._retriever = DocRetriever(
                persist_dir=None if self._chroma_client else str(self.settings.chroma_dir),
                collection_name=self.settings.chroma_collection,
                embedding_model=self.settings.embedding_model,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                chroma_client=self._chroma_client,
                embeddings=self._embeddings,
            )
        return self._retriever

    @property
    def generator(self):
        """Lazily initialize the generator (loads LLM)."""
        if self._generator is None:
            from docrag.generator import CodeGenerator

            self._generator = CodeGenerator(settings=self.settings, llm=self._llm)
        return self._generator

    def index_urls(self, urls: list[str]) -> int:
        """Scrape URLs and index their content into the vector store.

        Returns the number of chunks indexed.
        """
        scraped = self.scraper.scrape_docs(urls)
        if not scraped:
            return 0
        lc_docs = self.scraper.to_langchain_docs(scraped)
        return self.retriever.index_documents(lc_docs)

    def search_only(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search the vector store without generating code."""
        return self.retriever.search(query, top_k=top_k)

    def _check_relevance(self, chunks: list[RetrievedChunk]) -> bool:
        """Return True if ALL chunks score above the relevance threshold (i.e. low relevance)."""
        if not chunks:
            return True
        threshold = self.settings.relevance_threshold
        return all(c.score > threshold for c in chunks)

    def run(
        self, query: str, urls: list[str], top_k: int = 5, search_query: str | None = None,
    ) -> GenerationResponse:
        """Full end-to-end pipeline: scrape, index, retrieve, generate.

        Args:
            query: The instruction sent to the LLM for code generation.
            urls: Documentation URLs to scrape and index.
            top_k: Number of context chunks to retrieve.
            search_query: If provided, used for vector similarity search instead
                          of ``query``. Useful when retrieval keywords differ from
                          the generation prompt.
        """
        if urls:
            self.index_urls(urls)
        chunks = self.retriever.search(search_query or query, top_k=top_k)
        response = self.generator.generate_code(query, chunks)
        response.low_relevance = self._check_relevance(chunks)
        return response
