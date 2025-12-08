import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
    print("✅ Connection successful!")

    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"PostgreSQL version: {version[0]}")

    conn.close()
    print("✅ Test passed!")
except Exception as e:
    print(f"❌ Error: {e}")
