"""FastAPI REST API for DocRAG."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from docrag.config import get_settings
from docrag.models import GenerationRequest, GenerationResponse, RetrievedChunk
from docrag.pipeline import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline once at startup."""
    global pipeline
    pipeline = RAGPipeline()
    yield
    pipeline = None


app = FastAPI(
    title="DocRAG API",
    description="Generate code from live documentation using RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return pipeline.retriever.get_stats()


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    return pipeline.run(
        query=request.query,
        urls=request.doc_urls,
        top_k=request.top_k,
        search_query=request.search_query,
    )


@app.post("/index")
async def index_docs(urls: list[str]):
    count = pipeline.index_urls(urls)
    return {"chunks_indexed": count, "urls_processed": len(urls)}


@app.post("/search")
async def search(query: str, top_k: int = 5):
    results = pipeline.search_only(query, top_k=top_k)
    return {"results": [r.model_dump() for r in results]}


@app.delete("/cache")
async def clear_cache():
    count = pipeline.cache.clear()
    return {"cleared": count}
