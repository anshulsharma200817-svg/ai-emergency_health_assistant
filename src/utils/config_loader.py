"""Configuration loader utility for reading config.yaml.

Educational simulation system.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Dict containing configuration key-value pairs.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        yaml.YAMLError: If YAML file parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
