# 🤖 Offline AI Code Generator

A fully offline AI code generation tool powered by **CodeLlama-7B** with **RAG** (Retrieval-Augmented Generation) for context-aware code generation.

## Features

- **Fully Offline** — After initial model download, works without internet
- **CodeLlama-7B** — Meta's code-specialized LLM with 4-bit quantization support
- **RAG-Enhanced** — ChromaDB + local embeddings for context-aware generation
- **Chat Mode** — Conversational interface like ChatGPT (remembers context)
- **Project Generation** — Generate complete project structures with all files
- **Token Management** — Automatic chunked generation for large files
- **Knowledge Base** — Add your own code/docs for better context

## Requirements

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU with 6GB+ VRAM (optional, CPU works but slower)
- ~15GB disk space for model

## Quick Start

```bash
# 1. Run setup (requires internet ONCE)
chmod +x setup.sh
./setup.sh

# 2. Start the generator (fully offline)
source venv/bin/activate
python main.py
```

## Usage

### Interactive Chat
```
💬 You: Write a Python function to merge two sorted lists
🤖 Assistant: [generates code]

💬 You: Now add type hints and docstring
🤖 Assistant: [generates improved code with context]
```

### Generate Full Projects
```
💬 You: /project A Flask REST API for a todo app with SQLite database
```

### Add Knowledge
```
💬 You: /add /path/to/your/codebase/utils.py
```

## Project Structure

```
offline-code-generator/
├── main.py              # Entry point & CLI interface
├── config.py            # All configuration settings
├── model_loader.py      # CodeLlama model loading & quantization
├── rag_engine.py        # RAG with ChromaDB & local embeddings
├── code_generator.py    # Code generation & token management
├── project_generator.py # Project file writer
├── setup.sh             # One-time setup script
├── requirements.txt     # Python dependencies
├── models/              # Downloaded models (created by setup)
├── knowledge_base/      # Your reference code/docs
├── data/                # ChromaDB vector store
└── generated_projects/  # Output directory
```

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_4BIT_QUANTIZATION` | `True` | Use 4-bit quantization (saves VRAM) |
| `DEVICE` | `cuda` | `cuda` for GPU, `cpu` for CPU-only |
| `MAX_NEW_TOKENS` | `2048` | Max tokens per generation |
| `TEMPERATURE` | `0.2` | Lower = more deterministic |
| `RAG_TOP_K` | `5` | Number of context chunks to retrieve |

## How It Works

1. **Model**: CodeLlama-7B-Instruct runs locally with optional 4-bit quantization
2. **RAG**: Your prompts are matched against a local ChromaDB vector store containing code patterns, project structures, and conversation history
3. **Token Management**: Long files are generated in chunks; the system detects incomplete code and continues automatically
4. **Memory**: Conversation history is stored both in-memory and in the vector store for long-term recall


# 