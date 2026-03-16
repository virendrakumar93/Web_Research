"""
LLM Model Loader
=================

Purpose:
    Loads and manages language models for text generation. Supports two
    backends:

    1. **transformers** — Full HuggingFace models with optional 4-bit
       quantization via bitsandbytes.
    2. **llama_cpp** — Lightweight GGUF quantized models via
       llama-cpp-python (10x faster startup, 4-5x lower RAM).

Inputs:
    - config: Dictionary with model_name, temperature, max_tokens,
      quantize, device_map, and backend settings from config.yaml.

Outputs:
    - A ModelLoader instance that exposes a `generate(prompt)` method
      returning the model's text response.

Role in Architecture:
    Central LLM provider — every agent delegates text generation to this
    module, ensuring consistent model usage and resource sharing.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.logger import get_logger

logger = get_logger(__name__)

# Supported HuggingFace models (transformers backend)
SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]

# Suggested GGUF models for the llama_cpp backend
SUGGESTED_GGUF_MODELS = {
    "phi-3": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    "mistral-7b": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "llama-3-8b": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
}


def _fix_rope_scaling(model_name: str) -> dict | None:
    """Return safe rope_scaling overrides for models with config mismatches.

    Phi-3 publishes rope_scaling with key "rope_type" but older
    transformers code looks for "type". Transformers >=4.43 handles
    this, but we also patch it defensively here.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Dict of extra config kwargs, or None.
    """
    if "phi-3" in model_name.lower() or "phi3" in model_name.lower():
        return {"rope_scaling": None}  # let the model re-derive it
    return None


class ModelLoader:
    """Loads a language model and provides text generation.

    Supports transformers (HuggingFace) and llama_cpp (GGUF) backends.
    """

    def __init__(self, config: dict):
        """Initialize the model loader.

        Args:
            config: LLM configuration dictionary with keys:
                model_name, temperature, max_tokens, quantize,
                device_map, backend, gguf_model_path.
        """
        self.backend = config.get("backend", "transformers")
        self.model_name = config.get("model_name", SUPPORTED_MODELS[2])
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.quantize = config.get("quantize", False)
        self.device_map = config.get("device_map", "auto")
        self.gguf_model_path = config.get("gguf_model_path", "")
        self.n_ctx = config.get("n_ctx", 4096)

        if self.backend == "transformers" and self.model_name not in SUPPORTED_MODELS:
            logger.warning(
                "Model '%s' not in supported list. Supported: %s",
                self.model_name,
                SUPPORTED_MODELS,
            )

        self.tokenizer = None
        self.model = None
        self._llm_cpp = None  # llama_cpp.Llama instance

    def load(self) -> None:
        """Download and load the model into memory.

        Dispatches to the correct backend based on self.backend.
        """
        if self.backend == "llama_cpp":
            self._load_gguf()
        else:
            self._load_transformers()

    # ------------------------------------------------------------------
    # Transformers backend
    # ------------------------------------------------------------------

    def _load_transformers(self) -> None:
        """Load a HuggingFace model via transformers + accelerate."""
        logger.info(
            "Loading model (transformers): %s (quantize=%s)",
            self.model_name,
            self.quantize,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {
            "device_map": self.device_map,
            "trust_remote_code": True,
            # Use eager attention to avoid Flash Attention / SDPA issues
            # on systems without CUDA or with driver incompatibilities.
            "attn_implementation": "eager",
        }

        # Determine dtype — use float32 on CPU, float16 on CUDA
        if torch.cuda.is_available():
            compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32
            # CPU cannot use 'auto' device_map with accelerate in all cases
            model_kwargs["device_map"] = None

        if self.quantize:
            if not torch.cuda.is_available():
                logger.warning(
                    "4-bit quantization requires CUDA. Falling back to full precision."
                )
                model_kwargs["torch_dtype"] = compute_dtype
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = compute_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        logger.info("Model loaded successfully (transformers).")

    # ------------------------------------------------------------------
    # llama.cpp / GGUF backend
    # ------------------------------------------------------------------

    def _load_gguf(self) -> None:
        """Load a GGUF model via llama-cpp-python.

        The model path must be set in config.yaml under
        llm.gguf_model_path. If the file does not exist, a helpful
        error message is raised listing download suggestions.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Install it with:\n"
                "  pip install llama-cpp-python\n"
                "For GPU acceleration, see: "
                "https://github.com/abetlen/llama-cpp-python#installation"
            )

        if not self.gguf_model_path or not os.path.isfile(self.gguf_model_path):
            raise FileNotFoundError(
                f"GGUF model file not found: '{self.gguf_model_path}'\n"
                f"Set llm.gguf_model_path in config.yaml to a valid .gguf file.\n"
                f"Suggested downloads:\n"
                + "\n".join(
                    f"  {name}: {url}"
                    for name, url in SUGGESTED_GGUF_MODELS.items()
                )
            )

        logger.info("Loading GGUF model: %s", self.gguf_model_path)

        n_gpu_layers = -1 if torch.cuda.is_available() else 0
        self._llm_cpp = Llama(
            model_path=self.gguf_model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("GGUF model loaded successfully (llama_cpp).")

    # ------------------------------------------------------------------
    # Unified generation interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt.

        Dispatches to the correct backend automatically.

        Args:
            prompt: The input text prompt.

        Returns:
            The generated text response (model output only, prompt stripped).
        """
        if self.backend == "llama_cpp":
            return self._generate_gguf(prompt)
        return self._generate_transformers(prompt)

    def _generate_transformers(self, prompt: str) -> str:
        """Generate text using the HuggingFace transformers model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3500
        )
        # Move to model device (handles both CPU and CUDA)
        device = self.model.device if hasattr(self.model, "device") else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def _generate_gguf(self, prompt: str) -> str:
        """Generate text using the llama-cpp-python model."""
        if self._llm_cpp is None:
            raise RuntimeError("GGUF model not loaded. Call load() first.")

        output = self._llm_cpp(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            stop=["\n\n\n"],
        )
        text = output["choices"][0]["text"]
        return text.strip()
