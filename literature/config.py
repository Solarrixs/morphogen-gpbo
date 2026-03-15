"""Configuration loader for the literature scraping pipeline."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = str(Path(__file__).parent / "config.yaml")
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ``${ENV_VAR}`` references in string values."""
    if isinstance(obj, str):
        def _replace(match: re.Match) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        return _ENV_VAR_RE.sub(_replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration, resolving ``${ENV_VAR}`` references.

    Args:
        path: Path to the YAML config file.  Defaults to
            ``literature/config.yaml`` next to this module.

    Returns:
        A dictionary with all ``${ENV_VAR}`` placeholders replaced by the
        corresponding environment variable values (left as-is when the
        variable is not set).
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh)
    return _resolve_env_vars(raw or {})
