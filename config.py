"""
Configuration for Offline AI Code Generator (LanceDB version).
All paths and settings are configured here.
"""
import os

# ── Model Settings ──────────────────────────────────────────────
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "CodeLlama-7b-Instruct-hf")

# Quantization: set to True to use 4-bit quantization (reduces VRAM to ~4GB)
USE_4BIT_QUANTIZATION = True

# Device: "cuda" for GPU, "cpu" for CPU-only (much slower)
DEVICE = "cpu"

# ── Generation Settings ─────────────────────────────────────────
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.95
REPETITION_PENALTY = 1.15

# Token budget per chunk when generating large projects
CHUNK_TOKEN_LIMIT = 1800

# ── RAG Settings (LanceDB Offline) ─────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # offline SentenceTransformer model
EMBEDDING_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "embeddings")

# LanceDB local storage path
LANCE_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "lancedb")

# RAG retrieval parameters
RAG_TOP_K = 5
CHUNK_SIZE = 512        # number of characters per chunk
CHUNK_OVERLAP = 50      # overlapping characters between chunks

# ── Knowledge Base ──────────────────────────────────────────────
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

# ── Project Generation ──────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_projects")