"""
Upgraded Code Generator — ChatGPT-style Conversational Coding
Project-aware + RAG-enhanced iterative workflow.
"""

import re
from typing import List, Dict

from transformers import TextIteratorStreamer
import threading
import torch

from config import (
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    CHUNK_TOKEN_LIMIT,
)
from model_loader import ModelLoader
from rag_engine import RAGEngine


class CodeGenerator:
    """
    Conversational, project-aware AI coding assistant.
    Works like ChatGPT for coding workflows.
    """

    SYSTEM_PROMPT = (
        "You are a senior software engineer.\n"
        "You maintain and improve existing projects.\n"
        "If user provides fixes or errors, update the existing code.\n"
        "Always return full updated file content when modifying files.\n"
        "Be precise and production-ready."
    )

    def __init__(self, model_loader: ModelLoader, rag_engine: RAGEngine):
        self.loader = model_loader
        self.rag = rag_engine
        self.conversation_history: List[dict] = []
        self.active_project_files: Dict[str, str] = {}

    # ─────────────────────────────────────────────
    # MAIN GENERATE (ChatGPT-like behavior)
    # ─────────────────────────────────────────────

    def generate(self, user_prompt: str) -> str:
        """
        Conversational generation.
        Retrieves previous project + conversation context automatically.
        """

        # 1️⃣ Retrieve relevant previous code (project memory)
        project_context = self._retrieve_project_context(user_prompt)

        # 2️⃣ Retrieve semantic RAG matches
        rag_context = self.rag.query(user_prompt)

        # 3️⃣ Build prompt
        full_prompt = self._build_prompt(
            user_prompt,
            rag_context=rag_context,
            project_context=project_context,
        )

        # 4️⃣ Generate (blocking)
        response = "".join(self.stream_inference(full_prompt))

        # 5️⃣ Save to conversation memory
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.conversation_history.append({"role": "assistant", "content": response})

        # 6️⃣ Save entire exchange to RAG
        self.rag.add_conversation(user_prompt, response)

        return response

    # ─────────────────────────────────────────────
    # PROJECT GENERATION
    # ─────────────────────────────────────────────

    def generate_project(self, project_description: str) -> dict:
        """
        Generate new project and activate it as working memory.
        """

        structure_prompt = (
            f"Generate ONLY the directory/file structure for this project:\n"
            f"{project_description}\n\n"
            f"One path per line."
        )

        structure = self.generate(structure_prompt)
        filepaths = self._parse_structure(structure)

        if not filepaths:
            return {"main.py": structure}

        project_files = {}

        for filepath in filepaths:
            file_prompt = (
                f"Project: {project_description}\n\n"
                f"Structure:\n{structure}\n\n"
                f"Generate COMPLETE code for file: {filepath}\n"
                f"Return only code."
            )

            code = self._generate_chunked(file_prompt, filepath)
            project_files[filepath] = code

            # Index file in RAG
            self.rag.add_code_file(filepath, code)

        # Activate project memory
        self.active_project_files = project_files

        return project_files

    # ─────────────────────────────────────────────
    # PROJECT CONTEXT RETRIEVAL
    # ─────────────────────────────────────────────

    def _retrieve_project_context(self, user_prompt: str) -> str:
        """
        Retrieve relevant files from active project.
        """

        if not self.active_project_files:
            return ""

        context_blocks = []

        for filename, content in self.active_project_files.items():
            if any(keyword in user_prompt.lower() for keyword in filename.lower().split(".")):
                context_blocks.append(f"File: {filename}\n{content}")

        # If nothing matched specifically, include top 2 files
        if not context_blocks:
            for i, (fname, content) in enumerate(self.active_project_files.items()):
                if i >= 2:
                    break
                context_blocks.append(f"File: {fname}\n{content}")

        return "\n\n".join(context_blocks)

    # ─────────────────────────────────────────────
    # PROMPT BUILDER
    # ─────────────────────────────────────────────

    def _build_prompt(self, user_msg: str, rag_context: str, project_context: str) -> str:

        parts = [f"[INST] <<SYS>>\n{self.SYSTEM_PROMPT}\n<</SYS>>\n\n"]

        # Include project memory
        if project_context:
            parts.append(
                f"[INST] Current project files:\n{project_context}\n[/INST]\n"
            )

        # Include semantic RAG
        if rag_context:
            parts.append(
                f"[INST] Related reference material:\n{rag_context}\n[/INST]\n"
            )

        # Include conversation history
        for turn in self.conversation_history[-6:]:
            if turn["role"] == "user":
                parts.append(f"[INST] {turn['content']} [/INST]\n")
            else:
                parts.append(f"{turn['content']}\n")

        parts.append(f"[INST] {user_msg} [/INST]\n")

        return "".join(parts)

    # ─────────────────────────────────────────────
    # STREAMING INFERENCE
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def stream_inference(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS):

        inputs = self.loader.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.loader.model.device)

        streamer = TextIteratorStreamer(
            self.loader.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            streamer=streamer,
            pad_token_id=self.loader.tokenizer.eos_token_id,
        )

        thread = threading.Thread(
            target=self.loader.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()

    # ─────────────────────────────────────────────
    # CHUNKED GENERATION
    # ─────────────────────────────────────────────

    def _generate_chunked(self, prompt: str, filepath: str) -> str:

        full_code = ""
        continuation_count = 0

        while continuation_count < 5:

            chunk = "".join(
                self.stream_inference(prompt, max_tokens=CHUNK_TOKEN_LIMIT)
            )

            full_code += chunk

            if self._is_code_complete(full_code, filepath):
                break

            continuation_count += 1
            prompt = (
                f"Continue file {filepath}.\n"
                f"Current content:\n{full_code[-800:]}\n"
                f"Return only remaining code."
            )

        return self._clean_code(full_code)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _parse_structure(structure_text: str) -> List[str]:
        lines = structure_text.strip().split("\n")
        filepaths = []

        for line in lines:
            cleaned = re.sub(r"[│├└─┬┤┐┘┌┴ ]", "", line).strip()
            if cleaned and "." in cleaned:
                filepaths.append(cleaned)

        return filepaths

    @staticmethod
    def _is_code_complete(code: str, filepath: str) -> bool:
        if not code.strip():
            return False
        if code.count("{") != code.count("}"):
            return False
        if code.count("(") != code.count(")"):
            return False
        if code.count("[") != code.count("]"):
            return False
        return True

    @staticmethod
    def _clean_code(code: str) -> str:
        code = re.sub(r"^```[\w]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
        return code.strip()