import logging

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from ..config import Config
from .vector_search import VectorSearch

load_dotenv()
logger = logging.getLogger(__name__)


class HyDERAGSystem:
    """
    HyDE (Hypothetical Document Embeddings) RAG.
    """

    def __init__(self, model: str | None = None):
        """
        Initialize the HyDE RAG system.

        Args:
            model: Model to use for generation.
        """
        if not Config.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        self.llm = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.async_llm = AsyncOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.model = model or Config.DEEPSEEK_MODEL
        self.vector_search = VectorSearch()

    def setup(self):
        """Setup the database."""
        self.vector_search.setup()

    def add_documents(self, documents: list[tuple[str, dict]]) -> list[int]:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of (text, metadata) tuples.

        Returns:
            List of document IDs.
        """
        ids = []
        for text, metadata in documents:
            doc_ids = self.vector_search.add_text(text, metadata or {})
            if doc_ids:
                ids.extend(doc_ids)
        return ids

    def _build_hypothesis_messages(self, query: str) -> list[dict]:
        """Build the messages for hypothesis generation."""
        system_prompt = """You generate hypothetical document passages that would answer a question.
Write a detailed, factual-sounding passage as it would appear in a knowledge base or encyclopedia.
Be specific with names, dates, and details. Write directly without preamble."""

        user_prompt = f"""Question: {query}

Passage:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_answer_messages(self, query: str, context: list[str]) -> list[dict]:
        """Build the messages for answer generation."""
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

    def generate_hypothesis(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: User query.

        Returns:
            Hypothetical answer text.
        """
        try:
            response = self.llm.chat.completions.create(
                messages=self._build_hypothesis_messages(query),
                model=self.model,
                temperature=0.2,
                max_tokens=350
            )
            return response.choices[0].message.content or query
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return query

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Retrieve using hypothetical document embedding.

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            List of relevant texts.
        """
        hypothesis = self.generate_hypothesis(query)
        results = self.vector_search.search(hypothesis, limit=limit)
        return [text for (id, text, similarity, metadata) in results]

    def generate(self, query: str, context: list[str]) -> str:
        """
        Generate final response from retrieved documents.

        Args:
            query: User query.
            context: List of relevant documents.

        Returns:
            Generated response.
        """
        try:
            response = self.llm.chat.completions.create(
                messages=self._build_answer_messages(query, context),
                model=self.model,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {e}"

    def ask(self, query: str, limit: int = 3) -> dict:
        """
        Complete HyDE RAG pipeline:

        1. Query Understanding: Parse user intent
        2. Hypothesis Generation: Create hypothetical answer
        3. Enhanced Retrieval: Search using hypothesis embedding
        4. Response Generation: Generate answer from real documents

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            Dict with answer and documents used.
        """
        # Retrieve using HyDE approach (hypothesis-based)
        documents = self.retrieve(query, limit=limit)

        # Generate final answer from real documents
        answer = self.generate(query, documents)

        return {"query": query, "answer": answer, "sources": documents}

    async def async_generate_hypothesis(self, query: str) -> str:
        """Async version of generate_hypothesis."""
        try:
            response = await self.async_llm.chat.completions.create(
                messages=self._build_hypothesis_messages(query),
                model=self.model,
                temperature=0.2,
                max_tokens=350
            )
            return response.choices[0].message.content or query
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return query

    async def async_generate(self, query: str, context: list[str]) -> str:
        """Async version of generate."""
        try:
            response = await self.async_llm.chat.completions.create(
                messages=self._build_answer_messages(query, context),
                model=self.model,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {e}"

    async def async_ask(self, query: str, limit: int = 3) -> dict:
        """Async version of ask for concurrent processing."""
        # Generate hypothesis async
        hypothesis = await self.async_generate_hypothesis(query)

        # Retrieve using hypothesis (sync - DB operation)
        results = self.vector_search.search(hypothesis, limit=limit)
        documents = [text for (id, text, similarity, metadata) in results]

        # Generate final answer async
        answer = await self.async_generate(query, documents)

        return {"query": query, "answer": answer, "sources": documents}

    def close(self):
        """Close connections."""
        self.vector_search.close()


system_prompt = """
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
