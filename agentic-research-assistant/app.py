"""
Multi-Agent Research Assistant — Application Entry Point
========================================================

Purpose:
    Main entry point that loads configuration, initializes the LLM,
    and launches the Gradio web interface.

Usage:
    python app.py
    python app.py --model meta-llama/Meta-Llama-3-8B-Instruct
    python app.py --quantize
    python app.py --share
"""

import argparse
import os
import sys

import yaml

# Add project root to path so imports work regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.model_loader import ModelLoader, SUPPORTED_MODELS
from ui.gradio_app import create_app
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to config/config.yaml
                     relative to this file.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help="HuggingFace model to use (overrides config.yaml)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable 4-bit quantization (reduces VRAM usage)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (overrides config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main():
    """Initialize and launch the research assistant."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply CLI overrides
    llm_config = config.get("llm", {})
    if args.model:
        llm_config["model_name"] = args.model
    if args.quantize:
        llm_config["quantize"] = True

    ui_config = config.get("ui", {})
    if args.share:
        ui_config["share"] = True
    if args.port:
        ui_config["server_port"] = args.port

    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.warning(
            "No HuggingFace token found. Set HF_TOKEN environment variable "
            "if accessing gated models (e.g., Llama 3)."
        )

    # Load the LLM
    logger.info("Initializing LLM...")
    llm = ModelLoader(llm_config)
    llm.load()

    # Create and launch the Gradio app
    logger.info("Starting Gradio interface...")
    app = create_app(config, llm)
    app.launch(
        server_name=ui_config.get("server_name", "0.0.0.0"),
        server_port=ui_config.get("server_port", 7860),
        share=ui_config.get("share", False),
    )


if __name__ == "__main__":
    main()
