import logging

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from ..config import Config
from .vector_search import VectorSearch

load_dotenv()
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, model: str | None = None):
        """
        Initialize the RAG system.

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
                logger.info(f"Added: {text[:50]}...")
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

    def generate(self, query: str, context: list[str]) -> str:
        """
        Generate a response using the LLM.

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

        system_prompt2 = """
            You are a helpful assistant that answers questions based on the provided documents.

            Rules:
            1. Base your answer on the documents. If they don't contain the answer, say so.
            3. Respond only with the information needed, with 1, 2 words.
        """

        user_prompt = f"""

        Documents: {context_text}

        Question: {query}

        Provide a direct answer based on the documents above."""

        try:
            response = self.llm.chat.completions.create(
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

    def ask(self, query: str, limit: int = 3) -> dict:
        """
        Complete RAG pipeline: Retrieve + Generate.

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            Dict with answer and documents used.
        """
        documents = self.retrieve(query, limit=limit)

        answer = self.generate(query, documents)

        return {"query": query, "answer": answer, "sources": documents}

    async def async_generate(self, query: str, context: list[str]) -> str:
        """
        Async version of generate for concurrent LLM calls.

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

        system_prompt21 = """
            You are a helpful assistant that answers questions based on the provided documents.

            Rules:
            1. Base your answer on the documents. If they don't contain the answer, say so.
            3. Respond only with the information needed, with 1, 2 words.
        """

        user_prompt = f"""

        Documents: {context_text}

        Question: {query}

        Provide a direct answer based on the documents above."""

        try:
            response = await self.async_llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt21},
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
        """
        Async version of ask for concurrent processing.

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            Dict with answer and documents used.
        """
        documents = self.retrieve(query, limit=limit)
        answer = await self.async_generate(query, documents)
        return {"query": query, "answer": answer, "sources": documents}

    def close(self):
        """Close connections."""
        self.vector_search.close()
