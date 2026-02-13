# DocRAG

Generate code from live documentation using RAG (Retrieval-Augmented Generation).

## Overview

DocRAG is a CLI tool and REST API that generates code grounded in **up-to-date documentation** rather than stale training data. Point it at any docs URL — it scrapes, indexes, retrieves relevant sections, and feeds them as context to an LLM.

```bash
docrag generate "async HTTP client with retry logic" \
  --url https://www.python-httpx.org/async/
```

## Features

- **CLI + REST API** — Use from the terminal or integrate into other tools via FastAPI
- **Thin client / fat server** — `pip install docrag` gives you a lightweight CLI (~50 MB); heavy ML deps are optional
- **Remote mode** — Point the CLI at a remote DocRAG server with `--remote` instead of running the pipeline locally
- **Multi-backend LLM** — OpenAI, Anthropic, or local models (Qwen2.5-Coder via HuggingFace Transformers)
- **Smart caching** — TTL-based document cache avoids re-scraping the same pages
- **Low-relevance warnings** — Flags when retrieved context scores poorly, so you know the output is based on general knowledge
- **Docker-ready** — One-command deployment with `docker compose up`

## Installation

### Thin CLI client (remote mode only)

```bash
pip install docrag
```

Installs only `typer`, `rich`, `pydantic`, and `httpx`. Use with `--remote` to talk to a DocRAG server.

### Full server

```bash
pip install docrag[server]
```

Includes FastAPI, LangChain, ChromaDB, sentence-transformers, and all scraping/retrieval dependencies. Everything you need to run the pipeline locally or host the API.

### Local model inference

```bash
pip install docrag[local]
```

Extends `server` with `transformers` and `torch` for running LLMs on your own hardware (no API keys needed).

### Development

```bash
pip install -e ".[server,dev]"
```

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
| `OPENAI_API_KEY`        | —                                | OpenAI API key                     |
| `ANTHROPIC_API_KEY`     | —                                | Anthropic API key                  |
| `DOCRAG_OPENAI_MODEL`  | `gpt-5-mini-2025-08-07`         | OpenAI model name                  |
| `DOCRAG_ANTHROPIC_MODEL`| `claude-sonnet-4-5-20250929`    | Anthropic model name               |
| `DOCRAG_LOCAL_MODEL`   | `Qwen/Qwen2.5-Coder-7B-Instruct`| Local HuggingFace model            |
| `DOCRAG_EMBEDDING_MODEL`| `all-MiniLM-L6-v2`             | Sentence-transformers model        |
| `DOCRAG_CHUNK_SIZE`    | `1000`                           | Text splitting chunk size          |
| `DOCRAG_CHUNK_OVERLAP` | `200`                            | Overlap between chunks             |
| `DOCRAG_RELEVANCE_THRESHOLD`| `1.0`                       | ChromaDB distance threshold        |
| `DOCRAG_CACHE_TTL_HOURS`| `24`                            | Document cache time-to-live        |

## Architecture

```
User ──> CLI (typer) ──┬──> Local RAGPipeline
                       │      Scraper ──> Cache
                       │      Retriever (ChromaDB + embeddings)
                       │      Generator (LLM)
                       │
                       └──> Remote server (httpx POST)
                              FastAPI ──> RAGPipeline (same as above)
```

### Dependency groups

| Install command          | What you get                            |
|--------------------------|-----------------------------------------|
| `pip install docrag`     | Thin CLI — `version`, `generate --remote`, `index --remote` |
| `pip install docrag[server]` | Full pipeline + API server          |
| `pip install docrag[local]`  | Server + local model inference (torch) |
| `pip install docrag[dev]`    | Testing and linting tools           |

## Tech Stack

- **CLI**: Typer + Rich
- **API**: FastAPI + Pydantic
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (via LangChain)
- **LLM**: OpenAI / Anthropic / HuggingFace Transformers
- **Scraping**: BeautifulSoup4 + Requests
- **HTTP client**: httpx (for remote mode)
- **Containerization**: Docker + Docker Compose

## Testing

```bash
# Run all unit tests
pytest tests/ -v -m "not integration"

# Run with coverage
pytest tests/ --cov=docrag --cov-report=term-missing

# Run integration tests (requires API keys)
pytest tests/ -v -m integration
```

## Roadmap

### v0.1.0 (current)

- [x] Documentation scraping with smart caching
- [x] Vector storage and retrieval with ChromaDB
- [x] LLM code generation (OpenAI, Anthropic, local models)
- [x] CLI with `generate`, `index`, `search`, `stats`, `cache-clear`
- [x] REST API with FastAPI
- [x] Docker and Docker Compose deployment
- [x] Low-relevance detection and warnings
- [x] Dependency groups — thin CLI client vs full server install
- [x] Remote mode (`--remote` flag) for CLI commands
- [x] Comprehensive test suite

### v0.2.0 (planned)

- [ ] **RAG pipeline improvements** — Iterate on the retrieval pipeline and system prompt; add advanced reranking and hybrid search (keyword + semantic) for more robust results
- [ ] **Multi-user data management** — Per-user vector store collections and cache isolation so multiple users can share a single server without data leakage
- [ ] **AWS deployment** — Terraform/CDK infrastructure for deploying DocRAG on AWS (ECS/Fargate, EFS for persistent ChromaDB, ALB, secrets via SSM)
- [ ] **Streaming responses** — Stream LLM output token-by-token to the CLI and API for faster perceived latency
- [ ] **More documentation formats** — Support Markdown files, Sphinx docs, and PDFs as input sources beyond HTML scraping
- [ ] **Web UI** — Browser-based interface for code generation and doc management
