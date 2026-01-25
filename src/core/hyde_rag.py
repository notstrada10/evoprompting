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

    def generate_hypothesis(self, query: str) -> str:
        """
        Layer 2: Generate a hypothetical document that would answer the query.
        This bridges the semantic gap between short queries and longer documents.

        Args:
            query: User query.

        Returns:
            Hypothetical answer text.
        """

        system_prompt = """You are an expert assistant generating a hypothetical answer to a question.

            Your task is to write a detailed, factual-sounding answer that would be typical in a knowledge base document.
            Write as if you are answering from a reference document, not conversationally.
            Be specific and include details that would typically appear in such documents.

            Do not mention that this is hypothetical. Just write the answer directly."""

        user_prompt = f"""Generate a detailed answer to this question as it would appear in a reference document:

Question: {query}

Answer:"""

        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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

        context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])

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

        user_prompt = f"""Documents:

            {context_text}

            Question: {query}

            Provide a direct answer based on the documents above."""

        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
        system_prompt = """You are an expert assistant generating a hypothetical answer to a question.

Your task is to write a detailed, factual-sounding answer that would be typical in a knowledge base document.
Write as if you are answering from a reference document, not conversationally.
Be specific and include details that would typically appear in such documents.

Do not mention that this is hypothetical. Just write the answer directly."""

        user_prompt = f"""Generate a detailed answer to this question as it would appear in a reference document:

Question: {query}

Answer:"""

        try:
            response = await self.async_llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
        context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])

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

        system_prompt2 = """
            You are a helpful assistant that answers questions based on the provided documents.

            Rules:
            1. Base your answer on the documents. If they don't contain the answer, say so.
            3. Respond only with the information needed, with 1, 2 words.
        """

        user_prompt = f"""Documents:

            {context_text}

            Question: {query}

            Provide a direct answer based on the documents above."""

        try:
            response = await self.async_llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt2},
                    {"role": "user", "content": user_prompt},
                ],
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
