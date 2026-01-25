import json
import logging
from typing import Any, Optional

import numpy as np
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as Connection

from ..config import Config

load_dotenv()
logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self, connection_string: Optional[str] = None, table_name: str = "embeddings"):
        """
        Initialize the vector database.

        Args:
            connection_string: PostgreSQL connection string. Defaults to Config.DATABASE_URL.
            table_name: Name of the table to use for embeddings. Defaults to "embeddings".
        """
        self.connection_string = connection_string or Config.DATABASE_URL
        self.table_name = table_name
        self.conn: Optional[Connection] = None

    def connect(self) -> None:
        """Connect to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def _ensure_connection(self) -> Connection:
        """
        Ensure connection exists and return it.

        Raises:
            RuntimeError: If database is not connected.

        Returns:
            Active database connection.
        """
        if self.conn is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self.conn

    def setup_database(self) -> None:
        """Setup the database schema and indexes."""
        if not self.conn:
            self.connect()

        conn = self._ensure_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector({Config.EMBEDDING_DIM}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_vector_idx
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops);
            """)

            conn.commit()
            logger.info(f"Database setup complete for table '{self.table_name}'")

    def insert_embedding(
        self, text: str, embedding: list[float], metadata: Optional[dict[str, Any]] = None
    ) -> int:
        """
        Insert a text embedding into the database.

        Args:
            text: The text content.
            embedding: The embedding vector.
            metadata: Optional metadata to store with the embedding.

        Returns:
            The ID of the inserted record.
        """
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            embedding_array = np.array(embedding)

            cur.execute(
                f"""
                INSERT INTO {self.table_name} (text, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (text, embedding_array, json.dumps(metadata) if metadata else None),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Failed to insert embedding")

            inserted_id = result[0]
            conn.commit()
            return inserted_id

    def insert_embeddings_batch(
        self, items: list[tuple[str, list[float], Optional[dict[str, Any]]]]
    ) -> list[int]:
        """
        Insert multiple text embeddings into the database in a single transaction.

        Args:
            items: List of tuples (text, embedding, metadata).

        Returns:
            List of IDs for inserted records.
        """
        if not items:
            return []

        conn = self._ensure_connection()
        ids = []

        with conn.cursor() as cur:
            from psycopg2.extras import execute_values

            # Prepare data for batch insert
            values = [
                (text, np.array(embedding), json.dumps(metadata) if metadata else None)
                for text, embedding, metadata in items
            ]

            # Use execute_values for efficient batch insert
            execute_values(
                cur,
                f"""
                INSERT INTO {self.table_name} (text, embedding, metadata)
                VALUES %s
                RETURNING id;
                """,
                values,
                template="(%s, %s, %s)",
                fetch=True
            )

            ids = [row[0] for row in cur.fetchall()]
            conn.commit()

        return ids

    def search_similar(
        self, query_embedding: list[float], limit: int = 5
    ) -> list[tuple[Any, ...]]:
        """
        Search for similar embeddings using cosine similarity.

        Args:
            query_embedding: The query embedding vector.
            limit: Maximum number of results to return.

        Returns:
            List of tuples (id, text, similarity, metadata).
        """
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            query_array = np.array(query_embedding)

            cur.execute(
                f"""
                SELECT
                    id,
                    text,
                    1 - (embedding <=> %s) as similarity,
                    metadata
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """,
                (query_array, query_array, limit),
            )

            return cur.fetchall()

    def delete_all(self) -> None:
        """Delete all embeddings from the database."""
        if self.conn is None:
            logger.warning("DB is not connected")
            return

        with self.conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name};")
            self.conn.commit()
            logger.info(f"All embeddings deleted from table '{self.table_name}'")

    def count(self) -> int:
        """
        Count the total number of embeddings in the database.

        Returns:
            Number of embeddings.
        """
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            result = cur.fetchone()
            if result is None:
                return 0
            return result[0]

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            self.conn = None
