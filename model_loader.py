"""
Model Loader — Downloads (once) and loads CodeLlama-7b-Instruct locally.
Supports 4-bit quantization for low-VRAM GPUs.
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from config import (
    MODEL_NAME,
    MODEL_PATH,
    USE_4BIT_QUANTIZATION,
    DEVICE,
)


class ModelLoader:
    """Handles loading and caching the CodeLlama model and tokenizer."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ── public ──────────────────────────────────────────────────
    def load(self):
        """Load model + tokenizer. Downloads to MODEL_PATH on first run."""
        if self._loaded:
            return

        print("[*] Loading tokenizer …")
        self.tokenizer = self._load_tokenizer()

        print("[*] Loading model …")
        self.model = self._load_model()

        self._loaded = True
        print("[✓] Model ready.")

    # ── private ─────────────────────────────────────────────────
    def _resolve_path(self):
        """Return local path if it exists, otherwise the HF hub name."""
        if os.path.isdir(MODEL_PATH):
            return MODEL_PATH
        return MODEL_NAME

    def _load_tokenizer(self):
        print(MODEL_PATH)
        source = self._resolve_path()
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            cache_dir=MODEL_PATH,
            local_files_only=os.path.isdir(MODEL_PATH),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        source = self._resolve_path()
        kwargs = {
            "cache_dir": MODEL_PATH,
            "local_files_only": os.path.isdir(MODEL_PATH),
            "device_map": "auto" if DEVICE == "cuda" else None,
            "torch_dtype": torch.float16,
        }

        if USE_4BIT_QUANTIZATION and DEVICE == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)

        if DEVICE == "cpu":
            model = model.to("cpu")

        model.eval()
        return model

    def save_locally(self):
        """Persist the model + tokenizer to MODEL_PATH for true offline use."""
        os.makedirs(MODEL_PATH, exist_ok=True)
        self.tokenizer.save_pretrained(MODEL_PATH)
        self.model.save_pretrained(MODEL_PATH)
        print(f"[✓] Model saved to {MODEL_PATH}")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
