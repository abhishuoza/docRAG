"""Typer CLI for DocRAG."""

import typer

app = typer.Typer(name="docrag", help="Generate code from live documentation using RAG.")


@app.command()
def version():
    """Show the DocRAG version."""
    from docrag import __version__

    typer.echo(f"docrag {__version__}")