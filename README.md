# DocRAG

Generate code from live documentation using RAG (Retrieval-Augmented Generation).

## Overview

DocRAG is a CLI tool and REST API that generates code grounded in up-to-date documentation rather than stale training data. Point it at any docs URL: it scrapes, indexes, retrieves relevant sections, and feeds them as context to your LLM of choice.

```bash
docrag generate "async HTTP client with retry logic" \
  --url https://www.python-httpx.org/async/
```

### Architecture

```
User ──> CLI (typer) ──┬──> Local RAGPipeline
                       │      Scraper ──> Cache
                       │      Retriever (ChromaDB + embeddings)
                       │      Generator (LLM)
                       │
                       └──> Remote server (httpx POST)
                              FastAPI ──> RAGPipeline (same as above)
```

## Features

DocRAG works both as a CLI tool and a REST API powered by FastAPI, so you can use it from the terminal or integrate it into other tools. The packaging follows a thin-client / fat-server model: `pip install docrag` gives you a lightweight CLI (~50 MB) while heavy ML dependencies remain optional. If you'd rather not run the pipeline locally, the `--remote` flag lets you point the CLI at a remote DocRAG server instead. On the LLM side, DocRAG supports multiple backends including OpenAI, Anthropic, and local models such as Qwen2.5-Coder via HuggingFace Transformers. A TTL-based document cache avoids re-scraping pages you've already indexed, and built-in low-relevance warnings flag when retrieved context scores poorly so you know the output is falling back on general knowledge. For deployment, a single `docker compose up` command gets the full stack running.

## Installation

```bash
# Thin CLI client (remote mode only)
pip install docrag 

# OR

# All dependencies to run locally or host API
pip install docrag[server] 
```
### Dependency groups

| Install command          | What you get                                               |
|--------------------------|------------------------------------------------------------|
| `pip install docrag`     | Installs only `typer`, `rich`, `pydantic`, and `httpx`. Use with `--remote` to talk to a DocRAG server. |
| `pip install docrag[server]` | Includes FastAPI, LangChain, ChromaDB, sentence-transformers, and all scraping/retrieval dependencies. Everything you need to run the pipeline locally or host the API.                                 |
| `pip install docrag[local]`  | Extends `server` with `transformers` and `torch` for running LLMs on your own hardware (no API keys needed).                     |
| `pip install docrag[dev]`    | Testing and linting tools                                  |

## Quick Start

### Local usage (requires `server` or `local` install)

```bash
# Generate code from a docs URL
docrag generate "parse HTML tables" --url https://www.crummy.com/software/BeautifulSoup/bs4/doc/

# Pre-index docs, then generate without re-scraping
docrag index --url https://fastapi.tiangolo.com/tutorial/
docrag generate "CRUD API with path parameters"

# Search indexed docs directly
docrag search "dependency injection"

# View what's in the vector store
docrag stats

# Clear the document cache
docrag cache-clear
```

### Remote usage (works with thin install)

```bash
# Point at a running DocRAG server
docrag index --url https://fastapi.tiangolo.com/tutorial/ --remote http://my-server:8000
docrag generate "CRUD API with path parameters" --remote http://my-server:8000
```

### REST API

```bash
# Start the server
uvicorn docrag.api:app --host 0.0.0.0 --port 8000

# Or with Docker
docker compose up
```

API endpoints:

| Method | Endpoint    | Description                       |
|--------|-------------|-----------------------------------|
| GET    | `/health`   | Health check                      |
| GET    | `/stats`    | Vector store statistics            |
| POST   | `/generate` | Generate code from docs            |
| POST   | `/index`    | Index documentation URLs           |
| POST   | `/search`   | Search indexed documentation       |
| DELETE | `/cache`    | Clear the document cache           |

## Configuration

All settings are controlled via environment variables (prefix `DOCRAG_`) or a `.env` file:

| Variable                | Default                          | Description                        |
|-------------------------|----------------------------------|------------------------------------|
| `DOCRAG_LLM_BACKEND`   | `openai`                         | `openai`, `anthropic`, or `local`  |
| `OPENAI_API_KEY`        | -                                | OpenAI API key                     |
| `ANTHROPIC_API_KEY`     | -                                | Anthropic API key                  |
| `DOCRAG_OPENAI_MODEL`  | `gpt-5-mini-2025-08-07`          | OpenAI model name                  |
| `DOCRAG_ANTHROPIC_MODEL`| `claude-sonnet-4-5-20250929`     | Anthropic model name               |
| `DOCRAG_LOCAL_MODEL`   | `Qwen/Qwen2.5-Coder-7B-Instruct` | Local HuggingFace model            |
| `DOCRAG_EMBEDDING_MODEL`| `all-MiniLM-L6-v2`               | Sentence-transformers model        |
| `DOCRAG_CHUNK_SIZE`    | `1000`                           | Text splitting chunk size          |
| `DOCRAG_CHUNK_OVERLAP` | `200`                            | Overlap between chunks             |
| `DOCRAG_RELEVANCE_THRESHOLD`| `1.0`                            | ChromaDB distance threshold        |
| `DOCRAG_CACHE_TTL_HOURS`| `24`                             | Document cache time-to-live        |


## Testing

```bash
# Run all unit tests
pytest tests/ -v -m "not integration"

# Run with coverage
pytest tests/ --cov=docrag --cov-report=term-missing

# Run integration tests (requires API keys)
pytest tests/ -v -m integration
```

## Tech Stack

- CLI: Typer + Rich
- API: FastAPI + Pydantic
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: ChromaDB (via LangChain)
- LLM: OpenAI / Anthropic / HuggingFace Transformers
- Scraping: BeautifulSoup4 + Requests
- HTTP client: httpx (for remote mode)
- Containerization: Docker + Docker Compose

## Roadmap

### v0.1.0 (current)

- [x] Documentation scraping with smart caching
- [x] Vector storage and retrieval with ChromaDB
- [x] LLM code generation (OpenAI, Anthropic, local models)
- [x] CLI with `generate`, `index`, `search`, `stats`, `cache-clear`
- [x] REST API with FastAPI
- [x] Docker and Docker Compose deployment
- [x] Low-relevance detection and warnings
- [x] Dependency groups: thin CLI client vs full server install
- [x] Remote mode (`--remote` flag) for CLI commands
- [x] Comprehensive test suite

### v0.2.0 (planned)

- [ ] RAG pipeline improvements
- [ ] Multi-user data management
- [ ] AWS deployment
- [ ] Streaming responses
- [ ] Support for more documentation formats
- [ ] Web UI


