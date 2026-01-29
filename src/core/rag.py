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
        """
        Initialize the RAG system.

        Args:
            model: Model to use for generation.
            table_name: Database table name for embeddings.
        """
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
        """Setup the database."""
        self.vector_search.setup()

    def add_document(self, text: str, metadata: dict | None = None) -> list[int]:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text.
            metadata: Optional metadata.

        Returns:
            List of chunk IDs.
        """
        return self.vector_search.add_text(text, metadata or {})

    def add_documents(self, documents: list[tuple[str, dict]]) -> list[int]:
        """
        Add multiple documents to the knowledge base.

        Args:
            documents: List of (text, metadata) tuples.

        Returns:
            List of document IDs.
        """
        ids = []
        for text, metadata in documents:
            doc_id = self.add_document(text, metadata)
            if doc_id:
                ids.append(doc_id)
                #logger.info(f"Added: {text[:50]}...")
        return ids

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Retrieve the most relevant documents for the query.

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            List of relevant texts.
        """
        results = self.vector_search.search(query, limit=limit)
        return [text for (id, text, similarity, metadata) in results]

    def build_messages(self, query: str, context: list[str]) -> list[dict]:
        """Build the messages for the LLM call."""
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
        """
        Generate a response using the LLM.

        Args:
            query: User query.
            context: List of relevant documents.

        Returns:
            Generated response.
        """
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
        """Sync wrapper for async_generate."""
        return asyncio.run(self.async_generate(query, context))


    # MAIN Function
    async def async_ask(self, query: str, limit: int = 3) -> dict:
        """
        Complete RAG pipeline: Retrieve + Generate.

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            Dict with answer and documents used.
        """
        documents = self.retrieve(query, limit=limit)
        answer = await self.async_generate(query, documents)
        return {"query": query, "answer": answer, "sources": documents}

    def ask(self, query: str, limit: int = 3) -> dict:
        """Sync wrapper for async_ask."""
        return asyncio.run(self.async_ask(query, limit))

    def close(self):
        """Close connections."""
        self.vector_search.close()


# Alternative prompt - move smw else..-
system_prompt2 = """
    You are a helpful assistant that answers questions based on the provided documents.

    Rules:
    1. Base your answer on the documents. If they don't contain the answer, say so.
    2. Synthesize information from multiple documents when relevant.
    3. Be concise but complete.
    4. Do not make up information not present in the documents.
    5. Answer in a complete sentence that restates the question.
    6. Include all the informations gathered from the documents, when relevant.
    7. You don't need to say "Based on the provided documents" or similar phrases.
"""
