"""Tests for generation-layer failure visibility.

- A reachable-but-unhealthy Ollama (non-200) must raise, not return None.
- A caught generation error must be flagged so the CLI can exit non-zero.
"""

from unittest.mock import Mock, patch

import pytest

from src.generation.answer_generator import AnswerGenerator
from src.generation.llm_client import OllamaClient


# --- finding 12: Ollama non-200 must raise ----------------------------------


@patch("src.generation.llm_client.requests.get")
def test_check_connection_raises_on_non_200(mock_get):
    """A non-200 from /api/tags raises ConnectionError instead of returning None."""
    response = Mock()
    response.status_code = 500
    mock_get.return_value = response

    # OllamaClient.__init__ runs _check_connection(), so construction raises.
    with pytest.raises(ConnectionError, match="500"):
        OllamaClient(base_url="http://ollama:11434", model="llama3.2:3b")


@patch("src.generation.llm_client.requests.get")
def test_check_connection_succeeds_on_200(mock_get):
    """A healthy 200 response connects normally."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
    mock_get.return_value = response

    client = OllamaClient(base_url="http://ollama:11434", model="llama3.2:3b")
    assert client.model == "llama3.2:3b"


# --- finding 13: caught generation errors are flagged -----------------------


def test_answer_generator_flags_failure():
    """When generation fails, the result carries generation_failed=True so the
    CLI can exit non-zero instead of reporting success."""
    config = Mock()
    config.get.side_effect = lambda key, default=None: default

    llm_client = Mock()
    llm_client.generate.side_effect = RuntimeError("model exploded")

    generator = AnswerGenerator(config=config, llm_client=llm_client)
    result = generator.generate_answer(
        query="What is X?",
        retrieved_results=[
            {
                "chunk_id": "1",
                "doc_id": "d1",
                "content": "some content",
                "similarity_score": 0.9,
                "metadata": {},
            }
        ],
    )

    assert result.get("generation_failed") is True
    assert "error" in result
