#!/usr/bin/env python3
"""
Main entry point for the RAG Pipeline system.

This script provides a CLI interface for running various pipeline operations:
- preprocess: Prepare and process the document corpus
- index: Build embeddings and vector index (future)
- query: Query the RAG system (future)
- evaluate: Run evaluation framework (future)
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.preprocessing.corpus_processor import CorpusProcessor
from src.retrieval.indexer import Indexer
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.query_rewriter import QueryRewriter
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator


def setup_logging(config, log_file_name: str = "rag_pipeline.log") -> logging.Logger:
    """
    Set up logging based on configuration.

    Args:
        config: Configuration object
        log_file_name: Name of the log file

    Returns:
        Logger instance
    """
    log_level = config.get("logging.level", "INFO")
    log_format = config.get("logging.format")
    console_output = config.get("logging.console_output", True)
    file_output = config.get("logging.file_output", True)

    # Get or create logs directory
    logs_dir = config.get_path("paths.logs_dir", create=True)
    log_file = logs_dir / log_file_name if file_output else None

    return setup_logger(
        name="rag_pipeline",
        level=log_level,
        log_file=log_file,
        console_output=console_output,
        file_output=file_output,
        log_format=log_format,
    )


def cmd_preprocess(args, config) -> None:
    """
    Run corpus preprocessing.

    Args:
        args: Command-line arguments
        config: Configuration object
    """
    logger = setup_logging(config, "preprocessing.log")
    logger.info("=" * 60)
    logger.info("Starting PREPROCESS command")
    logger.info("=" * 60)

    # Override force_reprocess from config if provided via CLI
    force_reprocess = args.force or config.get("preprocessing.force_reprocess", False)
    logger.debug(f"Force reprocess: {force_reprocess}")

    # Create and run processor
    logger.debug("Initializing corpus processor...")
    processor = CorpusProcessor(config, logger_name="corpus_processor")
    
    try:
        logger.debug("Starting preprocessing pipeline...")
        processor.run(force_reprocess=force_reprocess)
        
        # Print statistics
        stats = processor.get_processing_stats()
        logger.info("\nProcessing Statistics:")
        logger.info(f"  Pruning completed: {stats['pruning_completed']}")
        logger.info(f"  Processing completed: {stats['processing_completed']}")
        logger.info(f"  Total files processed: {stats['file_count']}")
        
        if "files_by_type" in stats:
            logger.info("\n  Files by type:")
            for doc_type, count in stats["files_by_type"].items():
                logger.info(f"    {doc_type}: {count}")
        
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESS command completed successfully")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Corpus not found: {e}")
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the corpus is present at:")
        print("  rag_pipeline/data/corpus/scikit-learn-1.7.2-docs/")
        print("\nThe corpus should contain HTML documentation files.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        print(f"\n❌ Preprocessing failed: {e}")
        print("\nFor detailed error information, check:")
        print("  logs/preprocessing.log")
        sys.exit(1)


def cmd_index(args, config) -> None:
    """
    Build embeddings and vector index.

    Args:
        args: Command-line arguments
        config: Configuration object
    """
    logger = setup_logging(config, "indexing.log")
    logger.info("=" * 60)
    logger.info("Starting INDEX command")
    logger.info("=" * 60)
    
    # Override force_reindex from config if provided via CLI
    force_reindex = args.force or config.get("indexing.force_reindex", False)
    logger.debug(f"Force reindex: {force_reindex}")
    logger.debug(f"Strategy: {args.strategy}")
    
    # Create indexer
    logger.debug("Initializing indexer...")
    indexer = Indexer(config, logger_name="indexer")
    
    try:
        # Run indexing
        logger.debug("Starting indexing pipeline...")
        indexer.index(strategy=args.strategy, force_reindex=force_reindex)
        
        # Print statistics
        stats = indexer.get_stats()
        logger.info("\nIndexing Statistics:")
        logger.info(f"  Total collections: {stats['vector_store']['total_collections']}")
        logger.info(f"  Total chunks: {stats['vector_store']['total_chunks']}")
        
        if stats["strategies"]:
            logger.info("\n  By strategy:")
            for strategy, info in stats["strategies"].items():
                status = "✓ indexed" if info["indexed"] else "✗ not indexed"
                logger.info(
                    f"    {strategy}: {status} "
                    f"({info['chunk_count']} chunks, {info['doc_count']} docs)"
                )
        
        logger.info("\n" + "=" * 60)
        logger.info("INDEX command completed successfully")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        print(f"\n❌ Error: {e}")
        print("\nLikely cause: Corpus not preprocessed yet.")
        print("\nPlease run preprocessing first:")
        print("  make preprocess")
        print("  OR: docker compose run --rm rag-pipeline preprocess")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        print(f"\n❌ Indexing failed: {e}")
        print("\nFor detailed error information, check:")
        print("  logs/indexing.log")
        sys.exit(1)


def cmd_query(args, config) -> None:
    """
    Query the RAG system.

    Args:
        args: Command-line arguments
        config: Configuration object
    """
    import json
    
    logger = setup_logging(config, "query.log")
    logger.info("=" * 60)
    logger.info("Starting QUERY command")
    logger.info("=" * 60)
    
    # Validate query text
    if not args.query_text or not args.query_text.strip():
        logger.error("Query text is required")
        print("\n❌ Error: Query text cannot be empty")
        print("\nUsage:")
        print("  docker compose run --rm rag-pipeline query \"your question here\"")
        print("  OR: make query Q=\"your question here\"")
        print("\nExample:")
        print("  make query Q=\"How do I use StandardScaler?\"")
        sys.exit(1)
    
    try:
        # Initialize components
        logger.info("Initializing retrieval components...")
        logger.debug(f"Query text: {args.query_text}")
        logger.debug(f"Strategy: {args.strategy}")
        logger.debug(f"Top-k: {args.top_k}")
        logger.debug(f"Generate answer: {args.generate}")
        
        # Get paths from config
        vector_store_dir = config.get_path("paths.vector_store_dir")
        logger.debug(f"Vector store directory: {vector_store_dir}")
        
        # Initialize embedder
        model_name = config.get("embeddings.model", "all-MiniLM-L6-v2")
        device = config.get("embeddings.device", "cpu")
        batch_size = config.get("embeddings.batch_size", 32)
        
        logger.debug(f"Loading embedding model: {model_name} on {device}")
        embedder = Embedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            logger_name="query_embedder"
        )
        logger.debug("Embedder initialized successfully")
        
        # Initialize vector store
        logger.debug("Initializing vector store...")
        vector_store = VectorStore(
            persist_directory=vector_store_dir,
            logger_name="query_vector_store"
        )
        logger.debug("Vector store initialized successfully")
        
        # Initialize BM25 index for hybrid search
        bm25_index = BM25Index(
            persist_directory=vector_store_dir / "bm25",
            logger_name="query_bm25_index"
        )
        logger.debug("BM25 index initialized")
        
        # Initialize query rewriter if enabled
        query_rewriter = None
        if config.get("query_rewriting.enabled", False):
            # Initialize LLM client for query rewriting
            ollama_url = config.get("generation.ollama_base_url", "http://ollama:11434")
            model_name = config.get("generation.model", "llama3.2:3b")
            timeout = config.get("query_rewriting.timeout", 30)
            
            rewrite_llm_client = OllamaClient(
                base_url=ollama_url,
                model=model_name,
                timeout=timeout,
                logger_name="query_rewrite_llm"
            )
            
            query_rewriter = QueryRewriter(
                llm_client=rewrite_llm_client,
                config=config,
                logger_name="query_rewriter"
            )
            logger.debug("Query rewriter initialized")
        
        # Initialize query processor with BM25 index and query rewriter
        query_processor = QueryProcessor(
            config=config,
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=bm25_index,
            query_rewriter=query_rewriter,
            logger_name="query_processor"
        )
        
        # Determine search mode
        search_mode = args.search_mode or config.get("retrieval.search_mode", "hybrid")
        use_hybrid = search_mode in ("hybrid", "keyword")
        
        # Get strategy from args or config
        strategy = args.strategy or config.get("retrieval.strategy", "fixed")
        
        # Process query
        logger.debug(f"Processing query (mode={search_mode})...")
        if use_hybrid:
            # Load BM25 index for the strategy
            if not bm25_index.load_index(strategy):
                logger.warning(f"BM25 index not found for '{strategy}', falling back to semantic search")
                search_mode = "semantic"
                results = query_processor.process_query(
                    query_text=args.query_text,
                    strategy=strategy,
                    top_k=args.top_k,
                    show_full_content=args.show_content
                )
            else:
                results = query_processor.process_query_hybrid(
                    query_text=args.query_text,
                    strategy=strategy,
                    top_k=args.top_k,
                    alpha=args.alpha,
                    search_mode=search_mode
                )
        else:
            results = query_processor.process_query(
                query_text=args.query_text,
                strategy=strategy,
                top_k=args.top_k,
                show_full_content=args.show_content
            )
        logger.debug(f"Query returned {len(results.get('results', []))} results")
        
        # Generate answer if requested
        if args.generate:
            logger.info("Generating answer with LLM...")
            
            # First, display retrieval results (same as query-retrieve)
            formatted_output = query_processor.format_console_output(
                results,
                show_full_content=args.show_content
            )
            print("\n" + formatted_output)
            
            try:
                # Initialize LLM components
                ollama_url = config.get("generation.ollama_base_url", "http://ollama:11434")
                model_name = config.get("generation.model", "llama3.2:3b")
                timeout = config.get("generation.timeout", 60)
                
                llm_client = OllamaClient(
                    base_url=ollama_url,
                    model=model_name,
                    timeout=timeout,
                    logger_name="query_llm_client"
                )
                
                # Get prompt template
                template_name = config.get("generation.prompt_template", "default")
                templates = PromptBuilder.get_available_templates()
                template = templates.get(template_name, templates["default"])
                prompt_builder = PromptBuilder(template=template)
                
                answer_generator = AnswerGenerator(
                    config=config,
                    llm_client=llm_client,
                    prompt_builder=prompt_builder,
                    logger_name="query_answer_generator"
                )
                
                # Get the query that was used for retrieval (may have been rewritten)
                generation_query = results.get('query', args.query_text)
                
                # Generate answer from retrieval results
                generation_result = answer_generator.generate_answer(
                    query=generation_query,
                    retrieved_results=results["results"]
                )
                
                # Display generated answer (or the error the generator returned)
                formatted_answer = answer_generator.format_console_output(
                    generation_result,
                    show_metadata=True
                )
                print("\n" + formatted_answer)

                # Save generation result if output file requested
                if args.output:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Combine retrieval and generation results
                    combined_results = {
                        "retrieval": results,
                        "generation": generation_result
                    }
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(combined_results, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Results saved to: {output_path}")
                    print(f"\n💾 Results saved to: {output_path}")

                # The generator catches errors internally and returns an
                # error-shaped answer; surface that as a non-zero exit (after
                # saving, if requested) instead of reporting success.
                if generation_result.get("generation_failed"):
                    logger.error(
                        f"Answer generation failed: "
                        f"{generation_result.get('error', 'unknown error')}"
                    )
                    print("\n❌ Answer generation failed (see logs)")
                    sys.exit(1)

            except ConnectionError as e:
                logger.error(f"Cannot connect to Ollama: {e}")
                print("\n❌ Error: Cannot connect to Ollama service")
                print(f"\nExpected URL: {ollama_url}")
                print("\nTroubleshooting:")
                print("  1. Check if Ollama service is running:")
                print("     docker compose ps")
                print("  2. Check Ollama logs:")
                print("     docker compose logs ollama")
                print("  3. Restart Ollama service:")
                print("     docker compose restart ollama")
                print("  4. Verify Ollama is healthy:")
                print("     docker compose exec ollama ollama list")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Answer generation failed: {e}", exc_info=True)
                print(f"\n❌ Answer generation failed: {e}")
                sys.exit(1)
        else:
            # Just display retrieval results (no generation)
            formatted_output = query_processor.format_console_output(
                results,
                show_full_content=args.show_content
            )
            print("\n" + formatted_output)
            
            # Save to file if requested
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Results saved to: {output_path}")
                print(f"\n💾 Results saved to: {output_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("QUERY command completed successfully")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        print(f"\n❌ Error: {e}")
        print("\nLikely causes:")
        print("  1. Vector index not built yet. Run:")
        print("     make index")
        print("  2. Corpus not preprocessed yet. Run:")
        print("     make preprocess")
        print("  3. Run complete setup:")
        print("     make setup")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        print(f"\n❌ Query failed: {e}")
        print("\nFor detailed error information, check:")
        print("  logs/query.log")
        sys.exit(1)


def cmd_evaluate(args, config) -> None:
    """
    Run evaluation framework.

    Args:
        args: Command-line arguments
        config: Configuration object
    """
    from src.evaluation import RAGEvaluator, ResultsAnalyzer
    from datetime import datetime
    
    logger = setup_logging(config, "evaluation.log")
    logger.info("=" * 70)
    logger.info("Starting EVALUATE command")
    logger.info("=" * 70)
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator(config)
        
        # Run evaluation
        # Note: judge_answers=None uses config default (currently false)
        # --no-judge flag explicitly disables judging
        results = evaluator.run_evaluation(
            strategy=args.strategy,
            max_questions=args.max_questions,
            judge_answers=False if args.no_judge else None
        )
        
        # Print summary
        analyzer = ResultsAnalyzer()
        analyzer.print_summary(results)
        
        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
        else:
            # Default output path
            results_dir = config.base_path / config.get("evaluation.results_dir", "data/evaluation/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"evaluation_{timestamp}.json"
        
        evaluator.save_results(results, output_path)
        
        # Generate report if requested
        if args.report:
            report_path = output_path.with_suffix('.md')
            report_content = analyzer.generate_comparison_report(results)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Report saved to: {report_path}")
            print(f"\n📄 Report saved to: {report_path}")
        
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATE command completed successfully")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - A sophisticated retrieval-augmented generation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file (default: config/config.yaml)",
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Prepare and process the document corpus",
    )
    preprocess_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if already completed",
    )

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Build embeddings and vector index",
    )
    index_parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "hierarchical"],
        default=None,
        help="Chunking strategy to index (default: all enabled strategies)",
    )
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if already completed",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG system",
    )
    query_parser.add_argument(
        "query_text",
        help="Natural language query text",
    )
    query_parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "hierarchical"],
        default=None,
        help="Chunking strategy to query (default: from config)",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        dest="top_k",
        default=None,
        help="Number of results to return (default: from config)",
    )
    query_parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    query_parser.add_argument(
        "--show-content",
        action="store_true",
        dest="show_content",
        help="Display full chunk content (default: show excerpts)",
    )
    query_parser.add_argument(
        "--search-mode",
        choices=["semantic", "keyword", "hybrid"],
        dest="search_mode",
        default=None,
        help="Search mode (default: from config, typically 'hybrid')",
    )
    query_parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Hybrid search alpha weight for semantic (0.0-1.0). Only used with hybrid mode.",
    )
    query_parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate answer using LLM (requires Ollama)",
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation framework on test set",
    )
    evaluate_parser.add_argument(
        "--strategy",
        type=str,
        choices=["fixed", "semantic", "hierarchical"],
        default=None,
        help="Evaluate specific strategy (default: all strategies)",
    )
    evaluate_parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (default: all)",
    )
    evaluate_parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-based answer quality judging (faster)",
    )
    evaluate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for results JSON (default: auto-generated in data/evaluation/results/)",
    )
    evaluate_parser.add_argument(
        "--report",
        action="store_true",
        help="Generate markdown comparison report",
    )

    args = parser.parse_args()

    # If no command provided, print help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Route to appropriate command handler
    commands = {
        "preprocess": cmd_preprocess,
        "index": cmd_index,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, config)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

