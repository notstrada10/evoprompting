import logging
import sys

sys.path.insert(0, '/Users/marcostrada/Desktop/evoprompting')

import psycopg2
from dotenv import load_dotenv

from src.config import Config

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(Config.DATABASE_URL)
    logger.info("Connection successful!")

    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    logger.info(f"PostgreSQL version: {version[0]}")

    conn.close()
    logger.info("Test passed!")
except Exception as e:
    logger.error(f"Error: {e}")
