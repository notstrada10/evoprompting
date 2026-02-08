import asyncio
import logging

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ..config import Config
from .vector_search import VectorSearch

load_dotenv()
logger = logging.getLogger(__name__)


class RAGSystem:
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

    def add_document(self, text: str, metadata: dict | None = None) -> list[int]:
        return self.vector_search.add_text(text, metadata or {})

    def add_documents(self, documents: list[tuple[str, dict]]) -> list[int]:
        ids = []
        for text, metadata in documents:
            doc_id = self.add_document(text, metadata)
            if doc_id:
                ids.append(doc_id)
        return ids

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        results = self.vector_search.search(query, limit=limit)
        return [text for (id, text, similarity, metadata) in results]

    def build_messages(self, query: str, context: list[str]) -> list[dict]:
        context_text = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context)])

        system_prompt = """You are a precise question-answering system. Answer questions using ONLY the provided documents.

            Instructions:
            - Extract the exact answer from the documents
            - Be concise: use the minimum words necessary
            - If the answer is a name, date, number, or short phrase, respond with just that
            - If multiple documents contain relevant info, synthesize into one brief answer
            - If the documents don't contain the answer, say so
            - Never explain your reasoning or add context"""

        user_prompt = f"""Documents:
        {context_text}

        Question: {query}

        Answer:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def async_generate(self, query: str, context: list[str]) -> str:
        try:
            response = await self.async_llm.chat.completions.create(
                messages=self.build_messages(query, context),
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
        documents = self.retrieve(query, limit=limit)
        answer = await self.async_generate(query, documents)
        return {"query": query, "answer": answer, "sources": documents}

    def ask(self, query: str, limit: int = 3) -> dict:
        return asyncio.run(self.async_ask(query, limit))

    def close(self):
        self.vector_search.close()
