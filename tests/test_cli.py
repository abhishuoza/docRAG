"""Tests for the CLI module."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from docrag.cli import app
from docrag.models import GenerationResponse

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "docrag" in result.output


@patch("docrag.pipeline.RAGPipeline")
def test_generate(mock_pipeline_cls):
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.run.return_value = GenerationResponse(
        query="test query",
        generated_code="print('hello')",
        references=[],
        model_used="openai",
    )

    result = runner.invoke(app, ["generate", "test query"])
    assert result.exit_code == 0
    assert "Generated Code" in result.output


@patch("httpx.post")
def test_generate_remote(mock_post):
    """Test generate with --remote flag calls the API and renders the response."""
    # Build the JSON payload the remote API would return
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = GenerationResponse(
        query="test query",
        generated_code="print('hello from remote')",
        references=[],
        model_used="remote-openai",
    ).model_dump()
    mock_post.return_value = mock_response

    result = runner.invoke(app, ["generate", "test query", "--remote", "http://localhost:8000"])
    assert result.exit_code == 0
    assert "Generated Code" in result.output
    assert "remote-openai" in result.output

    # Verify the HTTP call was made to the correct endpoint
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "http://localhost:8000/generate"
    assert call_args[1]["json"]["query"] == "test query"


@patch("httpx.post")
def test_index_remote(mock_post):
    """Test index with --remote flag calls the API and reports chunk count."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"chunks_indexed": 42, "urls_processed": 2}
    mock_post.return_value = mock_response

    result = runner.invoke(app, [
        "index",
        "--url", "https://docs.example.com",
        "--url", "https://docs.other.com",
        "--remote", "http://localhost:8000",
    ])
    assert result.exit_code == 0
    assert "Indexed 42 chunks" in result.output

    # Verify correct endpoint and payload
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "http://localhost:8000/index"
    assert call_args[1]["json"] == ["https://docs.example.com", "https://docs.other.com"]


@patch("docrag.cache.DocCache")
def test_cache_clear(mock_cache_cls):
    mock_cache = MagicMock()
    mock_cache_cls.return_value = mock_cache
    mock_cache.clear.return_value = 3

    result = runner.invoke(app, ["cache-clear"])
    assert result.exit_code == 0
