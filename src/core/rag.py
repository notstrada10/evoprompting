import logging

from dotenv import load_dotenv
from openai import OpenAI

from ..config import Config
from .vector_search import VectorSearch

load_dotenv()
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, model: str = None, use_reranking: bool = False):
        """
        Initialize the RAG system.

        Args:
            model: Model to use for generation.
            use_reranking: Whether to use cross-encoder reranking. Defaults to False.
        """
        if Config.LLM_PROVIDER == "deepseek":
            if not Config.DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            self.llm = OpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url=Config.DEEPSEEK_BASE_URL
            )
            self.model = model or Config.DEEPSEEK_MODEL
        else:
            from groq import Groq
            if not Config.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            self.llm = Groq(api_key=Config.GROQ_API_KEY)
            self.model = model or Config.GROQ_MODEL

        self.vector_search = VectorSearch(use_reranking=use_reranking)

    def setup(self):
        """Setup the database."""
        self.vector_search.setup()

    def add_document(self, text: str, metadata: dict | None = None) -> int:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text.
            metadata: Optional metadata.

        Returns:
            Document ID.
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

    def retrieve(self, query: str, limit: int = 3) -> list[str]:
        """
        Retrieve the most relevant documents for the query.
        Applies diversity to avoid retrieving multiple chunks from the same document.

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            List of relevant texts.
        """
        results = self.vector_search.search(query, limit=limit * 3)

        documents = []
        seen_titles = {}
        max_chunks_per_doc = 2

        for (id, text, similarity, metadata) in results:
            title = text.split('\n')[0] if '\n' in text else text[:100]

            if title not in seen_titles:
                seen_titles[title] = 0

            if seen_titles[title] < max_chunks_per_doc:
                documents.append(text)
                seen_titles[title] += 1

            if len(documents) >= limit:
                break

        return documents

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
        5. Answer in a complete sentence that restates the question
        6. Include all the informations gathered from the documents, when relevant.
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

            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and hasattr(message, 'content') and message.content:
                    return message.content
                else:
                    return "Error: No content in response message"
            else:
                return "Error: No choices in response"

        except Exception as e:
            return f"Error generating response: {str(e)}"

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

    def close(self):
        """Close connections."""
        self.vector_search.close()
