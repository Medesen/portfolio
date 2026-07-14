"""Prompt builder for RAG with citation support."""

from __future__ import annotations
from typing import List, Dict, Any


class PromptBuilder:
    """
    Builds prompts for RAG (Retrieval-Augmented Generation).
    
    Handles context injection, citation formatting, and instruction templates.
    """
    
    # Default RAG prompt template
    DEFAULT_TEMPLATE = """You are a helpful assistant answering questions about scikit-learn, a machine learning library in Python.

Use the following context from the scikit-learn documentation to answer the question. If the answer is not in the context, say so.

When you reference information from the context, cite the source using [1], [2], etc. corresponding to the context chunks below.

Context:
{context}

Question: {query}

Answer (with citations):"""

    STRUCTURED_TEMPLATE = """
    You are an expert on scikit-learn. Follow these rules:
    1. Only answer based on the provided context
    2. If the context doesn't contain the answer, say so clearly
    3. Cite sources using [1], [2] format
    4. Be concise but complete
    5. Include code examples when relevant
    Context from scikit-learn documentation:

    {context}

    Question: {query}

    Provide a clear, well-cited answer:"""

    def __init__(self, template: str = None):
        """
        Initialize prompt builder.
        
        Args:
            template: Custom prompt template (uses DEFAULT_TEMPLATE if None)
        """
        self.template = template or self.DEFAULT_TEMPLATE
    
    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_context_length: int = 4000,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Build a RAG prompt from query and retrieved chunks.
        
        Args:
            query: User's question
            context_chunks: List of retrieved chunk dictionaries
            max_context_length: Maximum characters for context
            include_metadata: Whether to include doc_id in citations
            
        Returns:
            Dictionary with 'prompt', 'context_chunks_used', and 'citations_map'
        """
        # Format context with citations. Chunks are visited in rank order; a
        # chunk that doesn't fit the remaining budget is SKIPPED (not a hard
        # stop) so one oversized top-ranked chunk can't wipe out the entire
        # context. If even the first chunk exceeds the whole budget, it is
        # truncated instead — the LLM must never be left with zero context
        # while relevant chunks were retrieved. Citation numbers are assigned
        # to the chunks actually used, in order.
        context_parts = []
        citations_map = {}
        total_length = 0
        chunks_used = []

        for chunk in context_chunks:
            content = chunk.get("content", "")
            doc_id = chunk.get("doc_id", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")

            i = len(chunks_used) + 1  # citation number for the next used chunk
            if include_metadata:
                context_entry = f"[{i}] (Source: {doc_id})\n{content}"
            else:
                context_entry = f"[{i}] {content}"

            if total_length + len(context_entry) > max_context_length:
                if not chunks_used:
                    # First chunk alone exceeds the budget: truncate to fit
                    overshoot = total_length + len(context_entry) - max_context_length
                    content = content[: max(0, len(content) - overshoot)]
                    if include_metadata:
                        context_entry = f"[{i}] (Source: {doc_id})\n{content}"
                    else:
                        context_entry = f"[{i}] {content}"
                else:
                    # Doesn't fit alongside what's already included; try the
                    # next (smaller) chunk instead of stopping outright
                    continue

            context_parts.append(context_entry)
            total_length += len(context_entry)
            chunks_used.append(chunk)

            # Store citation mapping
            citations_map[i] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "content": content,
                "similarity_score": chunk.get("similarity_score", 0.0),
                "strategy": chunk.get("strategy", "unknown")
            }
        
        # Join context parts
        context_text = "\n\n".join(context_parts)
        
        # Build final prompt
        prompt = self.template.format(
            context=context_text,
            query=query
        )
        
        return {
            "prompt": prompt,
            "context_chunks_used": chunks_used,
            "citations_map": citations_map,
            "num_chunks_used": len(chunks_used),
            "context_length": total_length
        }
    
    def extract_citations(self, generated_text: str) -> List[int]:
        """
        Extract citation numbers from generated text.
        
        Args:
            generated_text: Text with citations like [1], [2]
            
        Returns:
            List of citation numbers found in the text
        """
        import re
        
        # Find all [N] patterns where N is a number
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, generated_text)
        
        # Convert to integers and remove duplicates while preserving order
        citations = []
        seen = set()
        for match in matches:
            num = int(match)
            if num not in seen:
                citations.append(num)
                seen.add(num)
        
        return citations
    
    def format_answer_with_sources(
        self,
        generated_text: str,
        citations_map: Dict[int, Dict[str, Any]],
        include_full_sources: bool = True
    ) -> str:
        """
        Format the generated answer with a sources section.
        
        Args:
            generated_text: Generated answer text
            citations_map: Mapping of citation numbers to chunk info
            include_full_sources: Whether to include detailed sources section
            
        Returns:
            Formatted text with answer and sources
        """
        if not include_full_sources:
            return generated_text
        
        # Extract citations used in the answer
        used_citations = self.extract_citations(generated_text)
        
        if not used_citations:
            return generated_text
        
        # Build sources section
        sources_lines = ["\n\nSources:"]
        for citation_num in used_citations:
            if citation_num in citations_map:
                info = citations_map[citation_num]
                doc_id = info["doc_id"]
                strategy = info.get("strategy", "unknown")
                score = info.get("similarity_score", 0.0)
                
                sources_lines.append(
                    f"[{citation_num}] {doc_id} "
                    f"(strategy: {strategy}, relevance: {score:.2f})"
                )
        
        sources_text = "\n".join(sources_lines)
        
        return generated_text + sources_text
    
    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        """
        Get available prompt templates.
        
        Returns:
            Dictionary of template names to template strings
        """
        return {
            "default": PromptBuilder.DEFAULT_TEMPLATE,
            "structured": PromptBuilder.STRUCTURED_TEMPLATE,
        }

