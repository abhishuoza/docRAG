"""Tests for the CodeGenerator module."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docrag.config import Settings
from docrag.generator import CodeGenerator
from docrag.models import RetrievedChunk

FAKE_CODE = '''```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

Sources: [Source 1: https://fastapi.tiangolo.com]'''


def _make_generator(responses: list[str] | None = None) -> CodeGenerator:
    fake_llm = FakeListChatModel(responses=responses or [FAKE_CODE])
    settings = Settings(llm_backend="openai")
    return CodeGenerator(settings=settings, llm=fake_llm)


def _sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text="FastAPI is a modern web framework. Use FastAPI() to create an app.",
            source_url="https://fastapi.tiangolo.com",
            score=0.15,
            chunk_index=0,
        ),
        RetrievedChunk(
            text="Define routes with @app.get('/path') decorator.",
            source_url="https://fastapi.tiangolo.com/tutorial/first-steps",
            score=0.22,
            chunk_index=1,
        ),
    ]


class TestCodeGenerator:
    def test_format_context(self):
        chunks = _sample_chunks()
        context = CodeGenerator.format_context(chunks)

        assert "[Source 1: https://fastapi.tiangolo.com]" in context
        assert "[Source 2:" in context
        assert "FastAPI is a modern web framework" in context
        assert "---" in context

    def test_generate_code(self):
        gen = _make_generator()
        chunks = _sample_chunks()

        response = gen.generate_code("create a basic FastAPI app", chunks)

        assert response.query == "create a basic FastAPI app"
        assert "FastAPI" in response.generated_code
        assert len(response.references) == 2
        assert response.model_used == "openai"

    def test_build_prompt_returns_template(self):
        gen = _make_generator()
        prompt = gen._build_prompt()

        assert prompt is not None
        input_vars = prompt.input_variables
        assert "context" in input_vars
        assert "query" in input_vars

    def test_generate_code_with_empty_chunks(self):
        gen = _make_generator(responses=["No documentation context provided."])

        response = gen.generate_code("do something", [])

        assert response.generated_code == "No documentation context provided."
        assert response.references == []
