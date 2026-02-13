"""LangChain-based retriever wrapping ChromaDB for document search."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docrag.models import RetrievedChunk


class DocRetriever:
    """Wraps LangChain's text splitter, embeddings, and Chroma vector store."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "docrag_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chroma_client=None,
        embeddings=None,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._embeddings = embeddings or HuggingFaceEmbeddings(model_name=embedding_model)

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
            client=chroma_client,
        )

    def index_documents(self, documents: list[Document]) -> int:
        """Split documents into chunks and add them to the vector store.

        Returns the number of chunks indexed.
        """
        chunks = self._splitter.split_documents(documents)
        if not chunks:
            return 0
        self._store.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search for the most relevant chunks matching a query."""
        results = self._store.similarity_search_with_score(query, k=top_k)
        return [
            RetrievedChunk(
                text=doc.page_content,
                source_url=doc.metadata.get("source", ""),
                score=score,
                chunk_index=doc.metadata.get("chunk_index", 0),
            )
            for doc, score in results
        ]

    def delete_documents(self, url: str) -> None:
        """Remove all chunks from a specific source URL."""
        self._store.delete(where={"source": url})

    def get_stats(self) -> dict:
        """Return basic stats about the vector store."""
        collection = self._store._collection
        total = collection.count()

        # Extract unique source URLs from metadata
        sources: list[str] = []
        if total > 0:
            result = collection.get(include=["metadatas"])
            metadatas = result["metadatas"] or []
            sources = sorted({
                str(m.get("source", "unknown"))
                for m in metadatas
                if m
            })

        return {
            "total_chunks": total,
            "collection_name": collection.name,
            "sources": sources,
        }

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        """Expose as a LangChain retriever for use in chains."""
        return self._store.as_retriever(**kwargs)
