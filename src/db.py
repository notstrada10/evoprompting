import json
import os

import numpy as np
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()

class VectorDB:
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.environ.get("DATABASE_URL")
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
            print("✅ Connected to PostgreSQL")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            raise

    def setup_database(self):
        if not self.conn:
            self.connect()

        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_vector_idx
                ON embeddings
                USING hnsw (embedding vector_cosine_ops);
            """)

            self.conn.commit()
            print("✅ Database setup complete")

    def insert_embedding(self, text: str, embedding: list, metadata: dict = None):
        with self.conn.cursor() as cur:
            # Converti in numpy array per pgvector
            embedding_array = np.array(embedding)

            cur.execute(
                """
                INSERT INTO embeddings (text, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (text, embedding_array, json.dumps(metadata) if metadata else None)
            )
            inserted_id = cur.fetchone()[0]
            self.conn.commit()
            return inserted_id

    def search_similar(self, query_embedding: list, limit: int = 5):
        with self.conn.cursor() as cur:
            # Converti in numpy array per pgvector
            query_array = np.array(query_embedding)

            cur.execute("""
                SELECT
                    id,
                    text,
                    1 - (embedding <=> %s) as similarity,
                    metadata
                FROM embeddings
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (query_array, query_array, limit))

            return cur.fetchall()

    def delete_all(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE embeddings;")
            self.conn.commit()
            print("✅ All embeddings deleted")

    def count(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings;")
            return cur.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
