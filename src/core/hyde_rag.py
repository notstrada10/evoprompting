import logging

from dotenv import load_dotenv
from openai import OpenAI

from ..config import Config
from .vector_search import VectorSearch

load_dotenv()
logger = logging.getLogger(__name__)


class HyDERAGSystem:
    """
    HyDE (Hypothetical Document Embeddings) RAG implementation.

    Architecture:
    1. Query Understanding Layer: Analyze user intent
    2. Hypothetical Document Layer: Generate hypothetical answer, embed it
    3. Enhanced Retrieval Layer: Search using hypothesis embedding
    4. Processing Layer: Generate final answer from real documents
    """

    def __init__(self, model: str = None, use_reranking: bool = False):
        """
        Initialize the HyDE RAG system.

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
                temperature=0.3,  # Lower temperature for more focused hypotheses
                max_tokens=300  # Keep hypothesis reasonably sized
            )

            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and hasattr(message, 'content') and message.content:
                    return message.content
                else:
                    logger.warning("No content in hypothesis generation response")
                    return query  # Fallback to original query
            else:
                logger.warning("No choices in hypothesis generation response")
                return query

        except Exception as e:
            logger.error(f"Error generating hypothesis: {str(e)}")
            return query  # Fallback to original query

    def retrieve(self, query: str, limit: int = 3) -> list[str]:
        """
        Layer 3: Enhanced retrieval using hypothetical document embedding.

        Instead of embedding the query directly, we:
        1. Generate a hypothetical answer
        2. Embed that answer
        3. Search for documents similar to the hypothesis

        This improves retrieval by matching document-to-document semantics
        rather than query-to-document.

        Args:
            query: User query.
            limit: Number of documents to retrieve.

        Returns:
            List of relevant texts.
        """
        # Generate hypothetical document
        hypothesis = self.generate_hypothesis(query)
        logger.debug(f"Hypothesis: {hypothesis[:100]}...")

        # Search using hypothesis instead of query
        results = self.vector_search.search(hypothesis, limit=limit * 3)

        # Apply diversity filter to avoid redundant chunks
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
        Layer 4: Generate final response from retrieved documents.

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

    def close(self):
        """Close connections."""
        self.vector_search.close()
