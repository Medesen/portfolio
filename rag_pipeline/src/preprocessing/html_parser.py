"""HTML parser for extracting clean text from scikit-learn documentation."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag

from ..utils.logger import get_logger


class HTMLParser:
    """Parser for extracting structured content from scikit-learn HTML docs."""

    # CSS selectors for content areas in scikit-learn docs
    CONTENT_SELECTORS = [
        "div.body",
        "div.document",
        "main",
        "article",
    ]

    # Elements to remove (navigation, footers, etc.)
    REMOVE_SELECTORS = [
        "nav",
        "header.page-header",
        "footer",
        "div.sidebar",
        "div.sphinxsidebar",
        "div.related",
        "div.navigation",
        "script",
        "style",
        "noscript",
        ".headerlink",
        ".viewcode-link",
        "div.admonition.sphx-glr-download-link-note",
    ]

    def __init__(self):
        """Initialize the HTML parser."""
        self.logger = get_logger("html_parser")

    def parse_file(self, file_path: Path, corpus_root: Path) -> Optional[Dict]:
        """
        Parse an HTML file and extract structured content.

        Args:
            file_path: Path to the HTML file
            corpus_root: Root path of the corpus (for relative path calculation)

        Returns:
            Dictionary with parsed content and metadata, or None if parsing fails
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "lxml")

            # Extract metadata
            title = self._extract_title(soup)
            doc_type = self._infer_doc_type(file_path, corpus_root)
            
            # Extract main content
            content = self._extract_content(soup)
            
            if not content or len(content.strip()) < 50:
                # Skip files with insufficient content
                return None

            # Calculate relative path from corpus root
            try:
                relative_path = file_path.relative_to(corpus_root)
            except ValueError:
                relative_path = file_path

            # Generate document ID
            doc_id = self._generate_doc_id(relative_path)

            return {
                "doc_id": doc_id,
                "source_path": str(relative_path),
                "title": title,
                "content": content,
                "doc_type": doc_type,
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "sections": self._extract_sections(soup),
                },
            }

        except Exception as e:
            # Log error but don't crash
            self.logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract document title from HTML."""
        # Try various title sources in order of preference
        # 1. h1 in main content
        h1 = soup.find("h1")
        if h1:
            return self._clean_text(h1.get_text())

        # 2. title tag
        title = soup.find("title")
        if title:
            text = title.get_text()
            # Remove common suffixes
            text = re.sub(r"\s*—\s*scikit-learn.*$", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*\|\s*scikit-learn.*$", "", text, flags=re.IGNORECASE)
            return self._clean_text(text)

        return "Untitled Document"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML, removing navigation and boilerplate."""
        # Remove unwanted elements
        for selector in self.REMOVE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()

        # Find main content area
        content_element = None
        for selector in self.CONTENT_SELECTORS:
            content_element = soup.select_one(selector)
            if content_element:
                break

        # Fallback to body if no content area found
        if content_element is None:
            content_element = soup.find("body")

        if content_element is None:
            return ""

        # Extract text while preserving some structure
        text = self._extract_text_with_structure(content_element)
        return self._clean_text(text)

    def _extract_text_with_structure(self, element: Tag) -> str:
        """
        Flatten an element to plain, space-separated text.

        This intentionally does NOT preserve heading or paragraph structure:
        an earlier "structure-preserving" implementation built a structured
        version and then discarded it, so the flat text below is what every
        published evaluation ran on. Downstream consumers — in particular the
        hierarchical chunker, which locates heading strings inside this text —
        rely on exactly this flattened form.
        """
        return element.get_text(separator=" ", strip=True)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove multiple spaces
        text = re.sub(r" +", " ", text)
        
        # Remove multiple newlines (keep at most 2)
        text = re.sub(r"\n\n+", "\n\n", text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def _extract_sections(self, soup: BeautifulSoup) -> List[str]:
        """Extract section headings from the document."""
        sections = []
        for heading in soup.find_all(["h1", "h2", "h3"]):
            text = self._clean_text(heading.get_text())
            if text and len(text) < 200:  # Sanity check
                sections.append(text)
        return sections

    def _infer_doc_type(self, file_path: Path, corpus_root: Path) -> str:
        """
        Infer document type based on path.

        Args:
            file_path: Path to the file
            corpus_root: Root path of the corpus

        Returns:
            Document type: 'api', 'guide', 'example', or 'other'
        """
        try:
            relative = file_path.relative_to(corpus_root)
            parts = relative.parts

            if "api" in parts:
                return "api"
            elif "modules" in parts:
                if "generated" in parts:
                    return "api"
                return "guide"
            elif "auto_examples" in parts:
                return "example"
            else:
                return "other"

        except ValueError:
            return "other"

    def _generate_doc_id(self, relative_path: Path) -> str:
        """
        Generate a unique document ID from the relative path.

        Args:
            relative_path: Relative path from corpus root

        Returns:
            Document ID (path with / replaced by __ and .html removed)
        """
        # Convert path to string and normalize
        path_str = str(relative_path).replace("\\", "/")
        
        # Remove .html extension
        if path_str.endswith(".html"):
            path_str = path_str[:-5]
        
        # Replace separators with double underscore
        doc_id = path_str.replace("/", "__")
        
        return doc_id

