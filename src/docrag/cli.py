"""Typer CLI for DocRAG."""

import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

app = typer.Typer(name="docrag", help="Generate code from live documentation using RAG.")
console = Console()


@app.command()
def version():
    """Show the DocRAG version."""
    from docrag import __version__

    typer.echo(f"docrag {__version__}")


def _extract_code_blocks(text: str) -> str:
    """Extract code from markdown fenced code blocks, or return the full text.

    Handles both complete blocks (```...```) and truncated blocks where
    the model output was cut off before the closing fence.
    """
    # Match complete blocks AND unclosed trailing blocks
    blocks = re.findall(r"```(?:\w*)\n(.*?)(?:```|$)", text, re.DOTALL)
    if blocks:
        # Filter out trivially short blocks (e.g. one-liner install commands)
        code_blocks = [b.strip() for b in blocks if b.strip().count("\n") >= 2]
        if code_blocks:
            return "\n\n".join(code_blocks)
        # Fall back to all blocks if none passed the filter
        return "\n\n".join(b.strip() for b in blocks)
    return text


@app.command()
def generate(
    query: str = typer.Argument(help="What code to generate"),
    url: list[str] = typer.Option([], "--url", "-u", help="Documentation URL(s) to use"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of context chunks"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save code to this file"),
    show_context: bool = typer.Option(False, "--show-context", "-c", help="Show full chunk text in references"),
    search_query: Optional[str] = typer.Option(None, "--search-query", "-s", help="Custom query for chunk retrieval (defaults to main query)"),
    remote: Optional[str] = typer.Option(None, "--remote", "-r", help="Remote DocRAG API base URL (e.g. http://localhost:8000)"),
):
    """Generate code from documentation."""
    if remote:
        import httpx

        from docrag.models import GenerationRequest, GenerationResponse

        request = GenerationRequest(
            query=query,
            doc_urls=url,
            top_k=top_k,
            search_query=search_query,
        )
        with console.status("[bold green]Calling remote API..."):
            resp = httpx.post(
                f"{remote.rstrip('/')}/generate",
                json=request.model_dump(),
                timeout=120.0,
            )
        if resp.status_code != 200:
            console.print(f"[red]Remote API error ({resp.status_code}): {resp.text}[/red]")
            raise typer.Exit(code=1)
        response = GenerationResponse.model_validate(resp.json())
    else:
        try:
            from docrag.pipeline import RAGPipeline
        except ImportError:
            console.print(
                "[red]This command requires server dependencies. "
                "Install them with:[/red] pip install docrag[server]"
            )
            raise typer.Exit(code=1)

        pipeline = RAGPipeline()
        with console.status("[bold green]Generating code..."):
            response = pipeline.run(query=query, urls=url, top_k=top_k, search_query=search_query)

    if response.low_relevance:
        console.print(Panel(
            "The indexed documentation did not closely match your query.\n"
            "The code below is based on the model's general knowledge, NOT the linked docs.",
            title="[bold]Low Relevance Warning[/bold]",
            border_style="yellow",
        ))

    console.print(Panel(
        Markdown(response.generated_code),
        title=f"[bold]Generated Code[/bold] (model: {response.model_used})",
        border_style="green",
    ))

    if response.references:
        console.print("\n[bold]References:[/bold]")
        for i, ref in enumerate(response.references, 1):
            if show_context:
                console.print(Panel(
                    ref.text,
                    title=f"[bold]Chunk {i}[/bold] - {ref.source_url}",
                    subtitle=f"score: {ref.score:.3f}",
                    border_style="dim",
                ))
            else:
                console.print(f"  - {ref.source_url} (score: {ref.score:.3f})")

    if output:
        code = _extract_code_blocks(response.generated_code)
        Path(output).write_text(code + "\n")
        console.print(f"\n[green]Code saved to {output}[/green]")


@app.command()
def index(
    url: list[str] = typer.Option(..., "--url", "-u", help="Documentation URL(s) to index"),
    remote: Optional[str] = typer.Option(None, "--remote", "-r", help="Remote DocRAG API base URL (e.g. http://localhost:8000)"),
):
    """Index documentation URLs into the vector store."""
    if remote:
        import httpx

        with console.status("[bold green]Indexing on remote server..."):
            resp = httpx.post(
                f"{remote.rstrip('/')}/index",
                json=url,
                timeout=120.0,
            )
        if resp.status_code != 200:
            console.print(f"[red]Remote API error ({resp.status_code}): {resp.text}[/red]")
            raise typer.Exit(code=1)
        data = resp.json()
        count = data["chunks_indexed"]
    else:
        try:
            from docrag.pipeline import RAGPipeline
        except ImportError:
            console.print(
                "[red]This command requires server dependencies. "
                "Install them with:[/red] pip install docrag[server]"
            )
            raise typer.Exit(code=1)

        pipeline = RAGPipeline()
        with console.status("[bold green]Indexing documentation..."):
            count = pipeline.index_urls(url)

    console.print(f"[green]Indexed {count} chunks from {len(url)} URL(s).[/green]")


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
):
    """Search indexed documentation."""
    try:
        from docrag.pipeline import RAGPipeline
    except ImportError:
        console.print(
            "[red]This command requires server dependencies. "
            "Install them with:[/red] pip install docrag[server]"
        )
        raise typer.Exit(code=1)

    pipeline = RAGPipeline()
    results = pipeline.search_only(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found. Try indexing some docs first.[/yellow]")
        return

    for i, chunk in enumerate(results, 1):
        console.print(Panel(
            chunk.text,
            title=f"[bold]Result {i}[/bold] - {chunk.source_url}",
            subtitle=f"score: {chunk.score:.3f}",
            border_style="blue",
        ))


@app.command(name="cache-clear")
def cache_clear():
    """Clear the document cache."""
    try:
        from docrag.cache import DocCache
        from docrag.config import get_settings
    except ImportError:
        console.print(
            "[red]This command requires server dependencies. "
            "Install them with:[/red] pip install docrag[server]"
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    cache = DocCache(settings.cache_dir, settings.cache_ttl_hours)
    count = cache.clear()
    console.print(f"[green]Cleared {count} cached document(s).[/green]")


@app.command()
def stats():
    """Show vector store statistics."""
    try:
        from docrag.pipeline import RAGPipeline
    except ImportError:
        console.print(
            "[red]This command requires server dependencies. "
            "Install them with:[/red] pip install docrag[server]"
        )
        raise typer.Exit(code=1)

    pipeline = RAGPipeline()
    info = pipeline.retriever.get_stats()

    summary = f"Collection: {info['collection_name']}\nTotal chunks: {info['total_chunks']}"
    sources = info.get("sources", [])
    if sources:
        summary += f"\nIndexed URLs ({len(sources)}):"
        for url in sources:
            summary += f"\n  - {url}"

    console.print(Panel(
        summary,
        title="[bold]Vector Store Stats[/bold]",
        border_style="cyan",
    ))
