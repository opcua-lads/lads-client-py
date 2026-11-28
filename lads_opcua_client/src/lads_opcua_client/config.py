"""Helpers for reading connection configurations (config.json)."""

from __future__ import annotations

import os
import re


def get_value(data: dict, key: str) -> any:
    if key in data:
        return data[key]
    else:
        return None


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: any) -> any:
    """
    Replaces ${VAR} in strings with the value of the environment variable VAR.
    Non-string values (e.g. None) are returned unchanged.

    Raises:
        KeyError: If a referenced environment variable is not set.
    """
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"Environment variable '{name}' is not set")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)
