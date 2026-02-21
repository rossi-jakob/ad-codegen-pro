import os
import logging
from typing import List, Optional

import numpy as np
import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_PATH,
    LANCE_DB_PATH,
    RAG_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    KNOWLEDGE_DIR,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Singleton DB connection (DIRECTORY ONLY)
# ─────────────────────────────────────────────
_db_connection: Optional[lancedb.DBConnection] = None


def get_db_connection() -> lancedb.DBConnection:
    global _db_connection
    if _db_connection is None:
        os.makedirs(LANCE_DB_PATH, exist_ok=True)
        _db_connection = lancedb.connect(LANCE_DB_PATH)
    return _db_connection


# ─────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────
class RAGEngine:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.db: Optional[lancedb.DBConnection] = None
        self.collection: Optional[lancedb.table.Table] = None
        self.embedding_dim: Optional[int] = None

    # ─────────────────────────────────────────
    # Initialize
    # ─────────────────────────────────────────
    def initialize(self):
        logger.info("[*] Initializing RAG engine...")

        # Load embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"[✓] Embedding dimension: {self.embedding_dim}")

        # Connect DB
        self.db = get_db_connection()
        table_name = "code_knowledge"

        # Always ensure correct schema
        class CodeChunk(LanceModel):
            text: str
            source: str
            vector: Vector(self.embedding_dim)

        # If table exists → try opening
        if table_name in self.db.table_names():
            try:
                self.collection = self.db.open_table(table_name)
                logger.info(f"[✓] Opened existing table '{table_name}'")
            except Exception:
                logger.warning("⚠ Table corrupted or schema mismatch. Recreating...")
                self.db.drop_table(table_name)
                self.collection = self.db.create_table(
                    table_name,
                    schema=CodeChunk
                )
        else:
            logger.info(f"[!] Creating new table '{table_name}'")
            self.collection = self.db.create_table(
                table_name,
                schema=CodeChunk
            )

        # Ingest knowledge
        self._ingest_knowledge_base()

        logger.info("[✓] RAG engine ready")

    # ─────────────────────────────────────────
    # Knowledge ingestion
    # ─────────────────────────────────────────
    def _ingest_knowledge_base(self):
        if not os.path.isdir(KNOWLEDGE_DIR):
            return

        logger.info(f"[*] Ingesting from {KNOWLEDGE_DIR}")

        for root, _, files in os.walk(KNOWLEDGE_DIR):
            for filename in files:
                if not filename.endswith(
                    (".py", ".txt", ".md", ".json", ".js", ".ts")
                ):
                    continue

                filepath = os.path.join(root, filename)

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        self.add_code_file(filepath, content)
                except Exception as e:
                    logger.warning(f"Failed reading {filepath}: {e}")

    # ─────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + CHUNK_SIZE, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    # ─────────────────────────────────────────
    # Add code file
    # ─────────────────────────────────────────
    def add_code_file(self, filepath: str, content: str):
        if not self.collection or not self.model:
            return

        chunks = self._chunk_text(content)
        records = []

        for chunk in chunks:
            vector = (
                self.model.encode(chunk)
                .astype(np.float32)
                .tolist()
            )

            records.append({
                "text": chunk,
                "source": filepath,
                "vector": vector,
            })

        if records:
            self.collection.add(records)

    # ─────────────────────────────────────────
    # Add conversation
    # ─────────────────────────────────────────
    def add_conversation(self, user_msg: str, assistant_msg: str):
        if not self.collection or not self.model:
            return

        combined = f"User: {user_msg}\nAssistant: {assistant_msg}"
        chunks = self._chunk_text(combined)
        records = []

        for chunk in chunks:
            vector = (
                self.model.encode(chunk)
                .astype(np.float32)
                .tolist()
            )

            records.append({
                "text": chunk,
                "source": "conversation",
                "vector": vector,
            })

        if records:
            self.collection.add(records)

    # ─────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────
    def query(self, query_text: str, top_k: int = RAG_TOP_K) -> str:
        if not self.collection or not self.model:
            return ""

        query_vector = (
            self.model.encode(query_text)
            .astype(np.float32)
        )

        results = (
            self.collection
            .search(query_vector)
            .limit(top_k)
            .to_list()
        )

        return "\n".join(r["text"] for r in results)