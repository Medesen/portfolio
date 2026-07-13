"""RAG answer generation orchestration."""

from __future__ import annotations
from typing import Dict, Any, Optional
import time

from .llm_client import OllamaClient
from .prompt_builder import PromptBuilder
from ..utils.logger import get_logger


class AnswerGenerator:
    """
    Orchestrates answer generation for RAG pipeline.
    
    Combines retrieval results with LLM generation to produce
    natural language answers with citations.
    """
    
    def __init__(
        self,
        config,
        llm_client: OllamaClient,
        prompt_builder: Optional[PromptBuilder] = None,
        logger_name: str = "answer_generator"
    ):
        """
        Initialize answer generator.
        
        Args:
            config: Configuration object
            llm_client: LLM client for generation
            prompt_builder: Prompt builder (uses default if None)
            logger_name: Logger name
        """
        self.config = config
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.logger = get_logger(logger_name)
        
        # Load generation configuration
        self.temperature = config.get("generation.temperature", 0.7)
        self.max_tokens = config.get("generation.max_tokens", 512)
        self.top_p = config.get("generation.top_p", 0.9)
        self.max_context_length = config.get("generation.max_context_length", 4000)
        self.include_sources = config.get("generation.include_sources", True)
        
        self.logger.info("Answer generator initialized")
    
    def generate_answer(
        self,
        query: str,
        retrieved_results: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_sources: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer from query and retrieved results.
        
        Args:
            query: User's question
            retrieved_results: List of retrieved chunk dictionaries
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            include_sources: Whether to include sources section
            
        Returns:
            Dictionary with generated answer, citations, timing, etc.
        """
        start_time = time.time()
        
        self.logger.info(f"Generating answer for query: '{query}'")
        self.logger.info(f"Using {len(retrieved_results)} retrieved chunks")
        
        # Use overrides or defaults
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        include_sources = include_sources if include_sources is not None else self.include_sources
        
        # Build prompt
        self.logger.info("Building RAG prompt...")
        prompt_build_start = time.time()
        prompt_data = self.prompt_builder.build_prompt(
            query=query,
            context_chunks=retrieved_results,
            max_context_length=self.max_context_length,
            include_metadata=True
        )
        prompt_build_time = time.time() - prompt_build_start
        
        prompt = prompt_data["prompt"]
        citations_map = prompt_data["citations_map"]
        num_chunks_used = prompt_data["num_chunks_used"]
        
        self.logger.info(
            f"Prompt built: {len(prompt)} chars, "
            f"{num_chunks_used}/{len(retrieved_results)} chunks used"
        )
        
        # Generate answer
        self.logger.info(f"Generating with LLM (temperature={temperature})...")
        generation_start = time.time()
        
        try:
            llm_response = self.llm_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.top_p
            )
            
            generated_text = llm_response["response"]
            generation_time = time.time() - generation_start
            
            self.logger.info(
                f"Answer generated: {len(generated_text)} chars "
                f"in {generation_time:.2f}s"
            )
            
            # Extract citations used
            used_citations = self.prompt_builder.extract_citations(generated_text)
            self.logger.info(f"Citations found in answer: {used_citations}")
            
            # Format with sources if requested
            if include_sources and citations_map:
                formatted_answer = self.prompt_builder.format_answer_with_sources(
                    generated_text=generated_text,
                    citations_map=citations_map,
                    include_full_sources=True
                )
            else:
                formatted_answer = generated_text
            
            total_time = time.time() - start_time
            
            # Prepare response
            response = {
                "answer": formatted_answer,
                "raw_answer": generated_text,  # Without sources section
                "query": query,
                "citations_used": used_citations,
                "citations_map": citations_map,
                "num_chunks_retrieved": len(retrieved_results),
                "num_chunks_used_in_prompt": num_chunks_used,
                "metadata": {
                    "model": llm_response.get("model", "unknown"),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timing": {
                        "prompt_build_time": round(prompt_build_time, 3),
                        "generation_time": round(generation_time, 3),
                        "total_time": round(total_time, 3)
                    },
                    "llm_stats": {
                        "prompt_tokens": llm_response.get("prompt_eval_count", 0),
                        "generated_tokens": llm_response.get("eval_count", 0),
                        "total_duration_ns": llm_response.get("total_duration", 0)
                    }
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}", exc_info=True)
            
            # Return error response
            return {
                "answer": f"Error: Failed to generate answer. {str(e)}",
                "raw_answer": "",
                "query": query,
                "citations_used": [],
                "citations_map": {},
                "num_chunks_retrieved": len(retrieved_results),
                "num_chunks_used_in_prompt": 0,
                "error": str(e),
                "generation_failed": True,
                "metadata": {
                    "timing": {
                        "total_time": round(time.time() - start_time, 3)
                    }
                }
            }
    
    def format_console_output(
        self,
        generation_result: Dict[str, Any],
        show_metadata: bool = True
    ) -> str:
        """
        Format generation result for console display.
        
        Args:
            generation_result: Result from generate_answer()
            show_metadata: Whether to show metadata (timing, tokens, etc.)
            
        Returns:
            Formatted string for console output
        """
        lines = []
        lines.append("=" * 80)
        lines.append("GENERATED ANSWER")
        lines.append("=" * 80)
        lines.append(f"Query: \"{generation_result['query']}\"")
        lines.append("-" * 80)
        
        # Answer
        lines.append("\n" + generation_result["answer"])
        
        # Metadata
        if show_metadata:
            lines.append("\n" + "-" * 80)
            lines.append("Generation Metadata:")
            
            metadata = generation_result.get("metadata", {})
            
            # Model info
            model = metadata.get("model", "unknown")
            temperature = metadata.get("temperature", 0.7)
            lines.append(f"  Model: {model}")
            lines.append(f"  Temperature: {temperature}")
            
            # Chunks
            num_retrieved = generation_result.get("num_chunks_retrieved", 0)
            num_used = generation_result.get("num_chunks_used_in_prompt", 0)
            lines.append(f"  Chunks: {num_used}/{num_retrieved} used in prompt")
            
            # Citations
            citations = generation_result.get("citations_used", [])
            lines.append(f"  Citations in answer: {len(citations)}")
            
            # Timing
            timing = metadata.get("timing", {})
            if timing:
                prompt_time = timing.get("prompt_build_time", 0)
                gen_time = timing.get("generation_time", 0)
                total_time = timing.get("total_time", 0)
                lines.append(
                    f"  Time: {total_time}s "
                    f"(prompt: {prompt_time}s, generation: {gen_time}s)"
                )
            
            # Token stats
            llm_stats = metadata.get("llm_stats", {})
            if llm_stats:
                prompt_tokens = llm_stats.get("prompt_tokens", 0)
                gen_tokens = llm_stats.get("generated_tokens", 0)
                lines.append(
                    f"  Tokens: {prompt_tokens} prompt + {gen_tokens} generated "
                    f"= {prompt_tokens + gen_tokens} total"
                )
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

