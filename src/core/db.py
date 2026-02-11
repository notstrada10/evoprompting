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
        self.connection_string = connection_string or Config.DATABASE_URL
        self.table_name = table_name
        self.conn: Optional[Connection] = None

    def connect(self) -> None:
        try:
            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def ensure_connection(self) -> Connection:
        if self.conn is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self.conn

    def setup_database(self) -> None:
        if not self.conn:
            self.connect()

        conn = self.ensure_connection()
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

    def insert_embedding(self, text: str, embedding: list[float], metadata: Optional[dict[str, Any]] = None) -> int:
        conn = self.ensure_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table_name} (text, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (text, np.array(embedding), json.dumps(metadata) if metadata else None),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Failed to insert embedding")
            inserted_id = result[0]
            conn.commit()
            return inserted_id

    def insert_embeddings_batch(self, items: list[tuple[str, list[float], Optional[dict[str, Any]]]]) -> list[int]:
        if not items:
            return []

        conn = self.ensure_connection()
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values

            values = [
                (text, np.array(embedding), json.dumps(metadata) if metadata else None)
                for text, embedding, metadata in items
            ]
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

    def search_similar(self, query_embedding: list[float], limit: int = 5) -> list[tuple[Any, ...]]:
        conn = self.ensure_connection()
        with conn.cursor() as cur:
            query_array = np.array(query_embedding)
            if limit > 40:
                cur.execute(f"SET hnsw.ef_search = {limit + 50};")
            cur.execute(
                f"""
                SELECT id, text, 1 - (embedding <=> %s) as similarity, metadata
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_array, query_array, limit),
            )
            return cur.fetchall()

    def delete_all(self) -> None:
        if self.conn is None:
            logger.warning("DB is not connected")
            return
        with self.conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name};")
            self.conn.commit()
            logger.info(f"All embeddings deleted from table '{self.table_name}'")

    def count(self) -> int:
        conn = self.ensure_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            result = cur.fetchone()
            return result[0] if result else 0

    def get_all_chunks(self) -> list[tuple[int, str]]:
        conn = self.ensure_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, text FROM {self.table_name} ORDER BY id;")
            return cur.fetchall()

    def get_all_chunks_with_embeddings(self) -> list[tuple[int, str, Any]]:
        """Fetch all (id, text, embedding) rows. Embedding is a numpy array."""
        conn = self.ensure_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, text, embedding FROM {self.table_name} ORDER BY id;")
            return cur.fetchall()

    def get_random_chunks(self, limit: int) -> list[tuple[int, str]]:
        conn = self.ensure_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text FROM {self.table_name} ORDER BY RANDOM() LIMIT %s;",
                (limit,)
            )
            return cur.fetchall()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            self.conn = None
