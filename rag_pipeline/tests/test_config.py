"""
Test configuration loading and management.

These tests verify that:
- Configuration can be loaded from YAML
- Paths are resolved correctly relative to base
- Dot-notation access works
- Missing keys return defaults
"""

import pytest
from pathlib import Path
from src.utils.config import load_config


def test_config_loads_from_yaml():
    """Test that configuration can be loaded from YAML file."""
    config = load_config(Path("config/config.yaml"))
    
    # Verify key settings are loaded
    assert config.get("embeddings.model") == "all-MiniLM-L6-v2"
    assert config.get("retrieval.top_k") == 20
    assert config.get("chunking.strategies.fixed.chunk_size") == 512


def test_config_path_resolution():
    """Test that paths are resolved relative to project base."""
    config = load_config(Path("config/config.yaml"))
    
    # Get a path and verify it resolves correctly
    corpus_path = config.get_path("paths.corpus_root")
    # Path should contain the corpus directory name
    assert "data/corpus" in str(corpus_path) or "scikit-learn" in str(corpus_path)


def test_config_get_with_default():
    """Test that config.get() returns default for missing keys."""
    config = load_config(Path("config/config.yaml"))
    
    # Get non-existent key with default
    value = config.get("nonexistent.key", "default_value")
    assert value == "default_value"


def test_config_nested_key_access():
    """Test accessing nested configuration values."""
    config = load_config(Path("config/config.yaml"))
    
    # Access nested keys (note: config structure is chunking.strategies.fixed)
    assert config.get("chunking.strategies.fixed.enabled") is True
    assert config.get("chunking.strategies.semantic.max_chunk_size") == 1000
    assert config.get("generation.temperature") == 0.3



def _write_config(tmp_path, text):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "config.yaml"
    path.write_text(text)
    return path


def test_load_config_rejects_empty_file(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_config(_write_config(tmp_path, ""))


def test_load_config_rejects_non_mapping_root(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write_config(tmp_path, "- just\n- a\n- list\n"))


def test_load_config_rejects_unknown_section(tmp_path):
    with pytest.raises(ValueError, match="Unknown configuration section"):
        load_config(_write_config(tmp_path, "retreival:\n  top_k: 20\n"))


def test_load_config_rejects_misspelled_key(tmp_path):
    with pytest.raises(ValueError, match="chunck_size"):
        load_config(_write_config(
            tmp_path,
            "chunking:\n  strategies:\n    fixed:\n      enabled: true\n      chunck_size: 512\n",
        ))


def test_load_config_rejects_out_of_range_value(tmp_path):
    with pytest.raises(ValueError, match="retrieval.top_k"):
        load_config(_write_config(tmp_path, "retrieval:\n  top_k: -3\n"))
    with pytest.raises(ValueError, match="hybrid_alpha"):
        load_config(_write_config(tmp_path, "retrieval:\n  hybrid_alpha: 1.5\n"))


def test_shipped_config_passes_validation():
    """The repository's own config.yaml must satisfy the schema."""
    config = load_config(Path("config/config.yaml"))
    assert config.get("retrieval.top_k") == 20
