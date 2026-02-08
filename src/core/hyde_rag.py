import asyncio
import logging

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ..config import Config
from .vector_search import VectorSearch

load_dotenv()
logger = logging.getLogger(__name__)


class HyDERAGSystem:
    """HyDE (Hypothetical Document Embeddings) RAG."""

    def __init__(self, model: str | None = None, table_name: str | None = None):
        if not Config.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        self.async_llm = AsyncOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.model = model or Config.DEEPSEEK_MODEL
        if table_name:
            from .db import VectorDatabase
            db = VectorDatabase(table_name=table_name)
            db.connect()
            self.vector_search = VectorSearch(db=db)
        else:
            self.vector_search = VectorSearch()

    def setup(self):
        self.vector_search.setup()

    def add_documents(self, documents: list[tuple[str, dict]]) -> list[int]:
        ids = []
        for text, metadata in documents:
            doc_ids = self.vector_search.add_text(text, metadata or {})
            if doc_ids:
                ids.extend(doc_ids)
        return ids

    def build_hypothesis_messages(self, query: str) -> list[dict]:
        system_prompt = """You generate hypothetical document passages that would answer a question.
        Write a detailed, factual-sounding passage as it would appear in a knowledge base or encyclopedia.
        Be specific with names, dates, and details. Write directly without preamble."""

        user_prompt = f"""Question: {query}

        Passage:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def build_answer_messages(self, query: str, context: list[str]) -> list[dict]:
        context_text = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context)])

        system_prompt = """You are a precise question-answering system. Answer questions using ONLY the provided documents.

            Instructions:
            - Extract the exact answer from the documents
            - Be concise: use the minimum words necessary
            - If the answer is a name, date, number, or short phrase, respond with just that
            - If multiple documents contain relevant info, synthesize into one brief answer
            - If the documents don't contain the answer, respond with "Unknown"
            - Never explain your reasoning or add context"""

        user_prompt = f"""Documents:
        {context_text}

        Question: {query}

        Answer:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def async_generate_hypothesis(self, query: str) -> str:
        try:
            response = await self.async_llm.chat.completions.create(
                messages=self.build_hypothesis_messages(query),
                model=self.model,
                temperature=0.2,
                max_tokens=350
            )
            return response.choices[0].message.content or query
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return query

    def generate_hypothesis(self, query: str) -> str:
        return asyncio.run(self.async_generate_hypothesis(query))

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        hypothesis = self.generate_hypothesis(query)
        results = self.vector_search.search(hypothesis, limit=limit)
        return [text for (id, text, similarity, metadata) in results]

    async def async_generate(self, query: str, context: list[str]) -> str:
        try:
            response = await self.async_llm.chat.completions.create(
                messages=self.build_answer_messages(query, context),
                model=self.model,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {e}"

    def generate(self, query: str, context: list[str]) -> str:
        return asyncio.run(self.async_generate(query, context))

    async def async_ask(self, query: str, limit: int = 3) -> dict:
        hypothesis = await self.async_generate_hypothesis(query)
        results = self.vector_search.search(hypothesis, limit=limit)
        documents = [text for (id, text, similarity, metadata) in results]
        answer = await self.async_generate(query, documents)
        return {"query": query, "answer": answer, "sources": documents}

    def ask(self, query: str, limit: int = 3) -> dict:
        return asyncio.run(self.async_ask(query, limit))

    def close(self):
        self.vector_search.close()
