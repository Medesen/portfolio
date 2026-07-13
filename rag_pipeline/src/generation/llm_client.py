"""Ollama LLM client implementation."""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import json
import requests
from requests.exceptions import RequestException, Timeout

from ..utils.logger import get_logger


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Provides a clean interface for generating text completions with local LLMs.
    Designed to be modular - easy to swap models or add other LLM providers.
    """
    
    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "llama3.2:3b",
        timeout: int = 60,
        logger_name: str = "ollama_client"
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL. Default assumes Docker Compose setup
                      where the Ollama service is named 'ollama'. For local development
                      outside Docker, use 'http://localhost:11434'.
            model: Model name (e.g., "llama3.2:3b", "llama3.1:8b", "mistral")
            timeout: Request timeout in seconds
            logger_name: Logger name
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.logger = get_logger(logger_name)
        
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
        self.logger.info(f"Initialized Ollama client: {self.base_url}, model: {self.model}")
        
        # Check if Ollama is available
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """
        Check if Ollama is available.
        
        Returns:
            True if connected successfully
            
        Raises:
            ConnectionError if Ollama is not reachable
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("Successfully connected to Ollama")
                
                # Check if our model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if self.model in model_names:
                    self.logger.info(f"Model '{self.model}' is available")
                else:
                    self.logger.warning(
                        f"Model '{self.model}' not found. Available models: {model_names}"
                    )
                    self.logger.warning(f"Run: ollama pull {self.model}")

                return True
            else:
                # A reachable-but-unhealthy Ollama (non-200) previously fell through
                # and returned None; fail loudly instead.
                raise ConnectionError(
                    f"Ollama at {self.base_url} returned HTTP {response.status_code} "
                    f"from /api/tags (expected 200)."
                )
        except RequestException as e:
            error_msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: 'ollama serve' or 'systemctl start ollama'"
            )
            self.logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text completion from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (None for model default)
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Whether to stream response (not implemented yet)
            
        Returns:
            Dictionary with 'response' (generated text) and metadata
            
        Raises:
            RequestException: If API call fails
        """
        self.logger.info(f"Generating completion with {self.model}")
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Non-streaming for now
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response text
            generated_text = result.get("response", "")
            
            self.logger.info(f"Generated {len(generated_text)} characters")
            
            return {
                "response": generated_text,
                "model": result.get("model", self.model),
                "done": result.get("done", True),
                "context": result.get("context"),
                "total_duration": result.get("total_duration", 0),
                "eval_count": result.get("eval_count", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
            }
            
        except Timeout:
            self.logger.error(f"Request timed out after {self.timeout}s")
            raise
        except RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generate chat completion from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary with 'response' and metadata
        """
        self.logger.info(f"Generating chat completion with {self.model}")
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract message content
            message = result.get("message", {})
            generated_text = message.get("content", "")
            
            self.logger.info(f"Generated {len(generated_text)} characters")
            
            return {
                "response": generated_text,
                "model": result.get("model", self.model),
                "done": result.get("done", True),
                "total_duration": result.get("total_duration", 0),
                "eval_count": result.get("eval_count", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
            }
            
        except RequestException as e:
            self.logger.error(f"Chat API request failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of recommended models.
        
        Returns:
            List of model names
        """
        return [
            "llama3.2:3b",      # Fast, good quality (recommended)
            "llama3.2:1b",      # Very fast, lighter
            "llama3.1:8b",      # Better quality, slower
            "mistral:7b",       # Good alternative
            "phi3:3.8b",        # Microsoft's model
        ]

