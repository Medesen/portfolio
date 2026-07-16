"""Configuration loading and management utilities.

``load_config`` validates the YAML at load time — structure, unknown keys,
and types/ranges of the load-bearing values — so a typo like ``chunck_size``
or ``top_k: -3`` fails immediately with the offending key path, instead of
surfacing later as an opaque error deep inside ChromaDB or the fusion math.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml

# Known keys per top-level section. An unknown section or key is an error:
# a misspelled key silently falling back to a default is worse than a crash.
_ALLOWED_KEYS: Dict[str, set] = {
    "paths": {"corpus_root", "processed_dir", "state_dir", "logs_dir", "vector_store_dir"},
    "preprocessing": {"input_name", "force_reprocess"},
    "logging": {"level", "format", "console_output", "file_output"},
    "chunking": {"strategies"},
    "embeddings": {"model", "batch_size", "device"},
    "indexing": {"force_reindex"},
    "retrieval": {
        "vector_db", "top_k", "min_similarity", "strategy", "result_format",
        "search_mode", "hybrid_alpha", "rrf_k", "overfetch_factor", "max_query_length",
    },
    "query_rewriting": {"enabled", "temperature", "max_tokens", "timeout", "cache_size"},
    "reranking": {"enabled", "model", "overfetch_k", "final_top_k", "batch_size", "device"},
    "generation": {
        "ollama_base_url", "model", "timeout", "temperature", "max_tokens",
        "top_p", "max_context_length", "prompt_template", "include_sources",
    },
    "evaluation": {
        "test_set_path", "results_dir", "strategies", "top_k_values",
        "judge_answers", "judge_criteria",
    },
}

_CHUNKING_STRATEGY_KEYS: Dict[str, set] = {
    "fixed": {"enabled", "chunk_size", "overlap"},
    "semantic": {"enabled", "max_chunk_size", "method"},
    "hierarchical": {"enabled", "max_chunk_size"},
}

#: (dotted key, checker, human-readable requirement)
_VALUE_CHECKS = [
    ("embeddings.batch_size", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("retrieval.top_k", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("retrieval.rrf_k", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("retrieval.overfetch_factor", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("retrieval.hybrid_alpha", lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0, "a number in [0, 1]"),
    ("retrieval.min_similarity", lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0, "a number in [0, 1]"),
    ("retrieval.search_mode", lambda v: v in ("semantic", "keyword", "hybrid"), "one of: semantic, keyword, hybrid"),
    ("retrieval.strategy", lambda v: v in ("fixed", "semantic", "hierarchical"), "one of: fixed, semantic, hierarchical"),
    ("reranking.overfetch_k", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("reranking.final_top_k", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("reranking.batch_size", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("generation.timeout", lambda v: isinstance(v, (int, float)) and v > 0, "a positive number"),
    ("generation.max_tokens", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("generation.temperature", lambda v: isinstance(v, (int, float)) and v >= 0.0, "a non-negative number"),
    ("query_rewriting.cache_size", lambda v: isinstance(v, int) and v >= 1, "a positive integer"),
    ("logging.level", lambda v: v in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
     "one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"),
]


def _dotted_get(config_dict: Dict[str, Any], dotted: str):
    value: Any = config_dict
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def validate_config(config_dict: Any) -> None:
    """Validate a parsed configuration mapping; raise ValueError on problems."""
    if config_dict is None:
        raise ValueError("Configuration file is empty")
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Configuration root must be a mapping, got {type(config_dict).__name__}"
        )

    for section, content in config_dict.items():
        if section not in _ALLOWED_KEYS:
            raise ValueError(
                f"Unknown configuration section {section!r}. "
                f"Known sections: {', '.join(sorted(_ALLOWED_KEYS))}"
            )
        if content is None:
            continue
        if not isinstance(content, dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping")
        unknown = set(content) - _ALLOWED_KEYS[section]
        if unknown:
            raise ValueError(
                f"Unknown key(s) in section {section!r}: {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_KEYS[section]))}"
            )

    strategies = _dotted_get(config_dict, "chunking.strategies")
    if strategies is not None:
        if not isinstance(strategies, dict):
            raise ValueError("chunking.strategies must be a mapping")
        for name, strat_cfg in strategies.items():
            if name not in _CHUNKING_STRATEGY_KEYS:
                raise ValueError(
                    f"Unknown chunking strategy {name!r}. "
                    f"Known: {', '.join(sorted(_CHUNKING_STRATEGY_KEYS))}"
                )
            if strat_cfg is None:
                continue
            if not isinstance(strat_cfg, dict):
                raise ValueError(f"chunking.strategies.{name} must be a mapping")
            unknown = set(strat_cfg) - _CHUNKING_STRATEGY_KEYS[name]
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in chunking.strategies.{name}: "
                    f"{', '.join(sorted(unknown))}"
                )

    for dotted, ok, requirement in _VALUE_CHECKS:
        value = _dotted_get(config_dict, dotted)
        if value is not None and not ok(value):
            raise ValueError(f"Config {dotted} must be {requirement}, got {value!r}")


class Config:
    """Configuration manager for the RAG pipeline."""

    def __init__(self, config_dict: Dict[str, Any], base_path: Path):
        """
        Initialize configuration.

        Args:
            config_dict: Dictionary containing configuration values
            base_path: Base path for resolving relative paths
        """
        self._config = config_dict
        self._base_path = base_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'paths.corpus_root')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_path(self, key: str, create: bool = False) -> Path:
        """
        Get a path from configuration and resolve it relative to base path.

        Args:
            key: Configuration key for the path
            create: Whether to create the directory if it doesn't exist

        Returns:
            Resolved Path object
        """
        path_str = self.get(key)
        if path_str is None:
            raise ValueError(f"Path configuration '{key}' not found")

        path = Path(path_str)
        if not path.is_absolute():
            path = self._base_path / path

        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found")
        return value

    @property
    def base_path(self) -> Path:
        """Get the base path for this configuration."""
        return self._base_path


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, looks for config.yaml
                    in default locations.

    Returns:
        Config object

    Raises:
        FileNotFoundError: If configuration file not found
        yaml.YAMLError: If configuration file is invalid
    """
    if config_path is None:
        # Try to find config in default locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path.cwd() / "config/config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                "Configuration file not found. Searched: " + 
                ", ".join(str(p) for p in possible_paths)
            )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Fail fast on structural problems, unknown keys, and invalid values —
    # see validate_config above.
    validate_config(config_dict)

    # Base path is the parent of the config directory
    base_path = config_path.parent.parent

    return Config(config_dict, base_path)

