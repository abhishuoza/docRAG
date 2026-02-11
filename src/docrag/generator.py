"""LangChain LLM wrappers and RAG chain for code generation."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from docrag.config import LLMBackend, Settings
from docrag.models import GenerationResponse, RetrievedChunk


class CodeGenerator:
    """Generates code from documentation context using a LangChain RAG chain."""

    def __init__(self, settings: Settings, llm: BaseChatModel | None = None):
        self._settings = settings
        self._llm = llm or self._create_llm()
        self._chain = self._build_chain()

    def _create_llm(self) -> BaseChatModel:
        """Factory that returns the right LangChain LLM based on settings."""
        backend = self._settings.llm_backend

        if backend == LLMBackend.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=self._settings.openai_model)

        elif backend == LLMBackend.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=self._settings.anthropic_model)

        elif backend == LLMBackend.LOCAL:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

            pipe = HuggingFacePipeline.from_model_id(
                model_id=self._settings.local_model,
                task="text-generation",
                device=0,
            )
            return ChatHuggingFace(llm=pipe)

        raise ValueError(f"Unknown LLM backend: {backend}")

    def _build_prompt(self) -> ChatPromptTemplate:
        """Build the ChatPromptTemplate for code generation with source citations."""
        system_message = ("Generate working code based ONLY on the retrieved documentation.\n"
                          "Include source references to show which doc URLs informed the code.\n"
                          "Add brief comments explaining key parts.\n"
                          "Admit when the documentation doesn't cover something, and then suggest code according to your best understanding, while explicitly flagging that the response code is not based on the linked documentation.\n ")
        return ChatPromptTemplate.from_messages([
                ("system", f"{system_message}"),
                ("human", "Retrieval: {context}\n\nRequest: {query}"),
            ])

    def _build_chain(self):
        """Create the LangChain chain: prompt -> LLM -> parse output."""
        prompt = self._build_prompt()
        return prompt | self._llm | StrOutputParser()

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string for the prompt."""
        sections = []
        for i, chunk in enumerate(chunks, 1):
            sections.append(
                f"[Source {i}: {chunk.source_url}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(sections)

    def generate_code(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> GenerationResponse:
        """Run the RAG chain to generate code from documentation context."""
        context = self.format_context(chunks)
        result = self._chain.invoke({"context": context, "query": query})

        return GenerationResponse(
            query=query,
            generated_code=result,
            references=chunks,
            model_used=str(self._settings.llm_backend.value),
        )
