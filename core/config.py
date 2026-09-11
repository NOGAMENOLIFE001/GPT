"""
Core Configuration Loader

Loads YAML configuration files with validation and merging support.
All project hyperparameters are defined in configs/ YAML files.
"""

import os
import copy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def load_config(name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file by name (without extension).

    Args:
        name: Config file name without extension (e.g., "model")
        config_dir: Override config directory; defaults to configs/

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigError: If file not found or YAML parse error.
    """
    if config_dir is None:
        config_dir = CONFIG_DIR
    else:
        config_dir = Path(config_dir)

    filepath = config_dir / f"{name}.yaml"
    if not filepath.exists():
        raise ConfigError(f"Config file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {filepath}: {exc}") from exc

    if config is None:
        config = {}

    return config


def get_variant_config(
    config: dict[str, Any], variant: str = "default"
) -> dict[str, Any]:
    """Extract a named variant from a config, merged with 'default'.

    The 'default' section provides base values. A variant (e.g., 'tiny')
    overrides only the keys it specifies.

    Args:
        config: Full config dict with 'default' and optional variant keys.
        variant: Variant name to merge (e.g., 'tiny', 'small').

    Returns:
        Merged configuration for the requested variant.
    """
    base = copy.deepcopy(config.get("default", {}))

    if variant != "default" and variant in config:
        overrides = copy.deepcopy(config[variant])
        _deep_merge(base, overrides)

    return base


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (in-place on base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def validate_config(config: dict[str, Any], required_keys: list[str]) -> None:
    """Validate that required keys exist in config.

    Args:
        config: Configuration dictionary.
        required_keys: List of required top-level keys.

    Raises:
        ConfigError: If any required key is missing.
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")


def load_model_config(variant: str = "tiny") -> dict[str, Any]:
    """Convenience: load model config for a given variant."""
    config = load_config("model")
    return get_variant_config(config, variant)


def load_training_config() -> dict[str, Any]:
    """Convenience: load training config."""
    config = load_config("training")
    return get_variant_config(config, "default")


def load_dataset_config(name: str = "d1") -> dict[str, Any]:
    """Convenience: load dataset config."""
    config = load_config("dataset")
    return get_variant_config(config, "default")


def resolve_path(path: str) -> Path:
    """Resolve a path relative to project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
