"""
Model registry: resolve model names to paths and load configs.

The models/ directory at the project root contains one subdirectory per model:
    models/
    └── GrassLM-10M/
        ├── config.json
        ├── checkpoints/
        │   ├── best_model.pt
        │   └── latest.pt
        └── exported/
"""

import json
import os
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (parent of python/)."""
    return Path(__file__).resolve().parent.parent.parent


def models_dir() -> Path:
    """Return the absolute path to the models/ directory."""
    return _project_root() / "models"


def get_model_dir(model_name: str) -> Path:
    """Return the directory for a named model."""
    d = models_dir() / model_name
    if not d.is_dir():
        available = list_models()
        raise FileNotFoundError(
            f"Model '{model_name}' not found in {models_dir()}. "
            f"Available models: {available}"
        )
    return d


def list_models() -> list[str]:
    """List all available model names (directories with a config.json)."""
    root = models_dir()
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def load_config(model_name: str) -> dict:
    """Load and return a model's config.json as a dict."""
    config_path = get_model_dir(model_name) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def save_config(model_name: str, config: dict) -> None:
    """Write a model's config.json."""
    model_dir = models_dir() / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def resolve_checkpoint(model_name: str, which: str = "best") -> str:
    """Resolve a model name to a checkpoint path.

    Args:
        model_name: Name of the model (e.g. "GrassLM-10M").
        which: "best" or "latest".

    Returns:
        Absolute path to the checkpoint file.
    """
    model_dir = get_model_dir(model_name)
    filename = "best_model.pt" if which == "best" else "latest.pt"
    ckpt_path = model_dir / "checkpoints" / filename
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return str(ckpt_path)
