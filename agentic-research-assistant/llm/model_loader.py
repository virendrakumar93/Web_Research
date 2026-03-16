"""
LLM Model Loader
=================

Purpose:
    Loads and manages open-source HuggingFace language models with optional
    4-bit quantization. Provides a unified interface for text generation
    used by all agents in the system.

Inputs:
    - config: Dictionary with model_name, temperature, max_tokens, quantize,
      and device_map settings from config.yaml.

Outputs:
    - A ModelLoader instance that exposes a `generate(prompt)` method
      returning the model's text response.

Role in Architecture:
    Central LLM provider — every agent delegates text generation to this
    module, ensuring consistent model usage and resource sharing.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.logger import get_logger

logger = get_logger(__name__)

# Supported models
SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]


class ModelLoader:
    """Loads a HuggingFace model and tokenizer, with optional quantization."""

    def __init__(self, config: dict):
        """Initialize the model loader.

        Args:
            config: LLM configuration dictionary with keys:
                model_name, temperature, max_tokens, quantize, device_map.
        """
        self.model_name = config.get("model_name", SUPPORTED_MODELS[2])
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.quantize = config.get("quantize", False)
        self.device_map = config.get("device_map", "auto")

        if self.model_name not in SUPPORTED_MODELS:
            logger.warning(
                "Model '%s' not in supported list. Supported: %s",
                self.model_name,
                SUPPORTED_MODELS,
            )

        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Download and load the model and tokenizer into memory."""
        logger.info("Loading model: %s (quantize=%s)", self.model_name, self.quantize)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        logger.info("Model loaded successfully.")

    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt.

        Args:
            prompt: The input text prompt.

        Returns:
            The generated text response (model output only, prompt stripped).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3500
        ).to(self.model.device)

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
