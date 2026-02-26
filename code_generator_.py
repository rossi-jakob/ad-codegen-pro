"""
Code Generator — Orchestrates prompt building, token management, and inference.
Supports chunked generation to bypass token limits.
"""
import re
from typing import List, Optional

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
    """Generate code using CodeLlama with RAG-augmented context."""

    SYSTEM_PROMPT = (
        "You are an expert software engineer. You write clean, well-documented, "
        "production-ready code. When asked to generate a project, first produce "
        "the directory structure, then generate each file completely. "
        "Always include necessary imports and error handling."
    )

    def __init__(self, model_loader: ModelLoader, rag_engine: RAGEngine):
        self.loader = model_loader
        self.rag = rag_engine
        self.conversation_history: List[dict] = []

    # ── Public API ──────────────────────────────────────────────
    def generate(self, user_prompt: str) -> str:
        """
        Generate code for a given prompt.
        Automatically retrieves relevant context from RAG and manages tokens.
        """
        # 1. Retrieve relevant context
        rag_context = self.rag.query(user_prompt)

        # 2. Build the full prompt with conversation history
        full_prompt = self._build_prompt(user_prompt, rag_context)

        # 3. Check if we need chunked generation
        prompt_tokens = self.loader.count_tokens(full_prompt)
        available_tokens = MAX_NEW_TOKENS

        if prompt_tokens + available_tokens > 3800:
            # Truncate conversation history to fit
            self.conversation_history = self.conversation_history[-4:]
            full_prompt = self._build_prompt(user_prompt, rag_context)

        # 4. Generate
        response = self.stream_inference(full_prompt)

        # 5. Store in conversation history & RAG
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.conversation_history.append({"role": "assistant", "content": response})
        self.rag.add_conversation(user_prompt, response)

        return response

    def generate_project(self, project_description: str) -> dict:
        """
        Generate a complete project: first the structure, then each file.
        Returns {filepath: content} dict.
        """
        # Step 1 — Generate project structure
        structure_prompt = (
            f"Generate ONLY the directory/file structure (no code) for this project:\n"
            f"{project_description}\n\n"
            f"Output as a tree listing, one path per line. Include all necessary files."
        )
        structure = self.generate(structure_prompt)
        filepaths = self._parse_structure(structure)

        if not filepaths:
            # Fallback: treat entire response as a single file
            return {"main.py": structure}

        # Step 2 — Generate each file using chunked generation
        project_files = {}
        for filepath in filepaths:
            # print(f"  Generating {filepath} …")
            file_prompt = (
                f"You are generating code for the project: {project_description}\n\n"
                f"Project structure:\n{structure}\n\n"
                f"Now generate the COMPLETE code for the file: {filepath}\n"
                f"Output ONLY the code, no explanations."
            )
            code = self._generate_chunked(file_prompt, filepath)
            project_files[filepath] = code

            # Index generated code for future context
            self.rag.add_code_file(filepath, code)

        return project_files

    # ── Chunked generation ──────────────────────────────────────
    def _generate_chunked(self, prompt: str, filepath: str) -> str:
        """
        Generate long files in chunks to overcome token limits.
        If the model stops mid-code, we continue generation automatically.
        """
        full_code = ""
        continuation_count = 0
        max_continuations = 5

        current_prompt = prompt

        while continuation_count < max_continuations:
            chunk = self.stream_inference(
                self._build_prompt(current_prompt, ""),
                max_tokens=CHUNK_TOKEN_LIMIT,
            )

            full_code += chunk

            # Check if the code looks complete
            if self._is_code_complete(full_code, filepath):
                break

            # Continue generation
            continuation_count += 1
            current_prompt = (
                f"Continue generating the code for {filepath}. "
                f"Here is what you have so far:\n```\n{full_code[-800:]}\n```\n"
                f"Continue from where you left off. Output ONLY code."
            )

        return self._clean_code(full_code)

    # ── Prompt construction ─────────────────────────────────────
    def _build_prompt(self, user_msg: str, context: str) -> str:
        """Build a CodeLlama-Instruct formatted prompt."""
        parts = [f"[INST] <<SYS>>\n{self.SYSTEM_PROMPT}\n<</SYS>>\n\n"]

        # Add conversation history (last 6 turns max)
        for turn in self.conversation_history[-6:]:
            if turn["role"] == "user":
                parts.append(f"[INST] {turn['content']} [/INST]\n")
            else:
                parts.append(f"{turn['content']}\n")

        # Add RAG context
        if context:
            parts.append(
                f"[INST] Relevant reference material:\n{context}\n\n"
                f"User request: {user_msg} [/INST]\n"
            )
        else:
            parts.append(f"[INST] {user_msg} [/INST]\n")

        return "".join(parts)

    # ── Inference ───────────────────────────────────────────────
    @torch.no_grad()
    def _inference(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
        inputs = self.loader.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self.loader.model.device) for k, v in inputs.items()}

        outputs = self.loader.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=self.loader.tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.loader.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    @torch.no_grad()
    def stream_inference(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS):

        inputs = self.loader.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
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
        )

        thread = threading.Thread(
            target=self.loader.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        for new_text in streamer:
            yield new_text

    # ── Helpers ─────────────────────────────────────────────────
    @staticmethod
    def _parse_structure(structure_text: str) -> List[str]:
        """Extract file paths from a tree-style structure output."""
        lines = structure_text.strip().split("\n")
        filepaths = []
        for line in lines:
            # Remove tree characters
            cleaned = re.sub(r"[│├└─┬┤┐┘┌┴ ]", "", line).strip()
            # Keep lines that look like file paths
            if cleaned and "." in cleaned and not cleaned.startswith("#"):
                filepaths.append(cleaned)
        return filepaths

    @staticmethod
    def _is_code_complete(code: str, filepath: str) -> bool:
        """Heuristic check if generated code looks complete."""
        code = code.strip()
        if not code:
            return False

        # Check balanced brackets/parens
        open_braces = code.count("{") - code.count("}")
        open_parens = code.count("(") - code.count(")")
        open_brackets = code.count("[") - code.count("]")

        if open_braces > 0 or open_parens > 0 or open_brackets > 0:
            return False

        # Python files: check if last line is properly indented
        if filepath.endswith(".py"):
            last_lines = code.split("\n")[-3:]
            # If last line is a hanging string or comment, probably incomplete
            for ln in last_lines:
                if ln.strip().endswith(("\\", ",")):
                    return False

        return True

    @staticmethod
    def _clean_code(code: str) -> str:
        """Remove markdown fences and trailing artifacts."""
        code = re.sub(r"^```\w*\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
        return code.strip()
