"""Corpus processor for orchestrating document preprocessing."""

from __future__ import annotations
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config import Config
from ..utils.logger import get_logger
from .html_parser import HTMLParser


class StateTracker:
    """Track preprocessing state to avoid redundant operations."""

    def __init__(self, state_path: Path, processed_dir: Optional[Path] = None):
        """
        Initialize state tracker.

        Args:
            state_path: Path to state JSON file
            processed_dir: Path to processed directory for validation
        """
        self.state_path = state_path
        self.processed_dir = processed_dir
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from file or return empty state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._empty_state()
        return self._empty_state()

    def _empty_state(self) -> Dict:
        """Return empty state structure."""
        return {
            "pruning_completed": False,
            "pruning_timestamp": None,
            "processing_completed": False,
            "processing_timestamp": None,
            "processed_files": [],
            "file_count": 0,
        }

    def save_state(self):
        """Save current state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def is_pruning_completed(self) -> bool:
        """Check if pruning has been completed."""
        return self.state.get("pruning_completed", False)

    def mark_pruning_completed(self):
        """Mark pruning as completed."""
        self.state["pruning_completed"] = True
        self.state["pruning_timestamp"] = datetime.now(timezone.utc).isoformat()
        self.save_state()

    def is_processing_completed(self) -> bool:
        """
        Check if processing has been completed.
        
        Validates that the state file says it's complete AND that
        the processed files actually exist.
        
        Returns:
            True if processing is completed and files exist, False otherwise
        """
        if not self.state.get("processing_completed", False):
            return False
        
        # If we have a processed_dir, validate that files actually exist
        if self.processed_dir is not None:
            if not self._validate_processed_files():
                # State says complete but files don't exist - mark as incomplete
                self.state["processing_completed"] = False
                self.save_state()
                return False
        
        return True
    
    def _validate_processed_files(self) -> bool:
        """
        Validate that processed files actually exist.
        
        Returns:
            True if processed directory exists and has files, False otherwise
        """
        if not self.processed_dir.exists():
            return False
        
        # Count JSON files in processed directory
        json_files = list(self.processed_dir.rglob("*.json"))
        actual_count = len(json_files)
        expected_count = self.state.get("file_count", 0)
        
        # Require exact match for data integrity
        if expected_count > 0 and actual_count != expected_count:
            return False
        
        return actual_count > 0

    def mark_processing_completed(self, file_count: int, processed_files: List[str]):
        """Mark processing as completed."""
        self.state["processing_completed"] = True
        self.state["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
        self.state["file_count"] = file_count
        self.state["processed_files"] = processed_files[:100]  # Store sample
        self.save_state()

    def reset(self):
        """Reset state (force reprocessing)."""
        self.state = self._empty_state()
        self.save_state()


class CorpusProcessor:
    """Orchestrate corpus preprocessing: pruning and parsing."""

    def __init__(self, config: Config, logger_name: str = "corpus_processor"):
        """
        Initialize corpus processor.

        Args:
            config: Configuration object
            logger_name: Name for the logger
        """
        self.config = config
        self.logger = get_logger(logger_name)
        self.parser = HTMLParser()

        # Setup paths
        self.corpus_root = config.get_path("paths.corpus_root")
        self.processed_dir = config.get_path("paths.processed_dir", create=True)
        self.state_dir = config.get_path("paths.state_dir", create=True)
        self.state_file = self.state_dir / "preprocessing_state.json"

        # Initialize state tracker with processed_dir for validation
        self.state_tracker = StateTracker(self.state_file, self.processed_dir)

    def run(self, force_reprocess: bool = False):
        """
        Run the full preprocessing pipeline.

        Args:
            force_reprocess: If True, force reprocessing even if already done
        """
        self.logger.info("Starting corpus preprocessing pipeline")

        if force_reprocess:
            self.logger.info("Force reprocessing enabled - resetting state")
            self.state_tracker.reset()

        # Step 1: Prune corpus if needed
        if not self.state_tracker.is_pruning_completed() or force_reprocess:
            self.logger.info("Pruning corpus...")
            self._run_pruning_script()
            self.state_tracker.mark_pruning_completed()
            self.logger.info("Pruning completed")
        else:
            self.logger.info("Pruning already completed (skipping)")

        # Step 2: Parse and process HTML files
        if not self.state_tracker.is_processing_completed() or force_reprocess:
            self.logger.info("Processing HTML files...")
            processed_files = self._process_html_files()
            self.state_tracker.mark_processing_completed(
                len(processed_files), processed_files
            )
            self.logger.info(f"Processing completed: {len(processed_files)} files")
        else:
            self.logger.info("Processing already completed (skipping)")

        self.logger.info("Preprocessing pipeline finished")

    def _find_sklearn_corpus_dir(self, corpus_parent: Path) -> Optional[str]:
        """
        Find scikit-learn corpus directory matching the pattern.
        
        Supports patterns like:
        - "scikit-learn-docs" (no version)
        - "scikit-learn-1.7.2-docs" (with version)
        
        Args:
            corpus_parent: Parent directory containing the corpus
            
        Returns:
            Name of the matching directory, or None if not found
        """
        # Pattern: scikit-learn[-X.Y.Z]-docs where version is optional
        pattern = re.compile(r'^scikit-learn(-\d+\.\d+\.\d+)?-docs$')
        
        for entry in corpus_parent.iterdir():
            if entry.is_dir() and pattern.match(entry.name):
                self.logger.info(f"Found matching corpus directory: {entry.name}")
                return entry.name
        
        return None
    
    def _run_pruning_script(self):
        """Run the pruning script to prepare the corpus."""
        # Find the pruning script
        script_path = self.config.base_path / "scripts" / "prune_sklearn_corpus.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Pruning script not found: {script_path}")

        # Get configuration for pruning
        input_name_pattern = self.config.get("preprocessing.input_name", "scikit-learn-*-docs")
        corpus_parent = self.corpus_root.parent
        
        # Find actual directory matching the pattern
        input_name = self._find_sklearn_corpus_dir(corpus_parent)
        
        if input_name is None:
            # Fall back to the configured value if it's not a pattern
            input_name = input_name_pattern
            self.logger.warning(
                f"No directory matching scikit-learn pattern found, "
                f"using configured value: {input_name}"
            )

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--root", str(corpus_parent),
            "--input-name", input_name,
        ]

        self.logger.info(f"Running pruning script: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            self.logger.debug(f"Pruning output: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Pruning stderr: {result.stderr}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Pruning script failed: {e}")
            self.logger.error(f"Stdout: {e.stdout}")
            self.logger.error(f"Stderr: {e.stderr}")
            raise

    def _process_html_files(self) -> List[str]:
        """
        Process all HTML files in the corpus.

        Returns:
            List of processed file paths
        """
        # Find all HTML files
        html_files = list(self.corpus_root.rglob("*.html"))
        self.logger.info(f"Found {len(html_files)} HTML files to process")

        processed_files = []
        skipped_files = 0

        for i, html_file in enumerate(html_files, 1):
            if i % 50 == 0:
                self.logger.info(f"Processing file {i}/{len(html_files)}...")

            # Parse the file
            parsed_doc = self.parser.parse_file(html_file, self.corpus_root)

            if parsed_doc is None:
                skipped_files += 1
                continue

            # Save to processed directory
            self._save_processed_doc(parsed_doc)
            processed_files.append(parsed_doc["doc_id"])

        self.logger.info(
            f"Processed {len(processed_files)} files, skipped {skipped_files}"
        )
        return processed_files

    def _save_processed_doc(self, doc: Dict):
        """
        Save processed document as JSON.

        Args:
            doc: Document dictionary
        """
        # Create subdirectory based on doc_type for organization
        doc_type = doc.get("doc_type", "other")
        type_dir = self.processed_dir / doc_type
        type_dir.mkdir(exist_ok=True)

        # Save as JSON
        output_file = type_dir / f"{doc['doc_id']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    def get_processing_stats(self) -> Dict:
        """
        Get statistics about processed corpus.

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "pruning_completed": self.state_tracker.is_pruning_completed(),
            "processing_completed": self.state_tracker.is_processing_completed(),
            "file_count": self.state_tracker.state.get("file_count", 0),
        }

        # Add timestamps if available
        if self.state_tracker.state.get("pruning_timestamp"):
            stats["pruning_timestamp"] = self.state_tracker.state["pruning_timestamp"]
        if self.state_tracker.state.get("processing_timestamp"):
            stats["processing_timestamp"] = self.state_tracker.state[
                "processing_timestamp"
            ]

        # Count files by type
        if self.processed_dir.exists():
            stats["files_by_type"] = {}
            for doc_type_dir in self.processed_dir.iterdir():
                if doc_type_dir.is_dir():
                    count = len(list(doc_type_dir.glob("*.json")))
                    stats["files_by_type"][doc_type_dir.name] = count

        return stats

