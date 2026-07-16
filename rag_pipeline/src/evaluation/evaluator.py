"""RAG system evaluation orchestration."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json
from datetime import datetime

from .test_loader import TestLoader, TestQuestion
from .metrics import RetrievalMetrics, extract_doc_ids_from_chunks
from .llm_judge import LLMJudge
from ..retrieval.query_processor import QueryProcessor
from ..retrieval.embedder import Embedder
from ..retrieval.vector_store import VectorStore
from ..generation.llm_client import OllamaClient
from ..generation.answer_generator import AnswerGenerator
from ..utils.logger import get_logger


class RAGEvaluator:
    """
    Main evaluator for RAG pipeline.

    Orchestrates end-to-end evaluation including:
    - Loading test set
    - Running retrieval for each question
    - Generating answers
    - Computing retrieval metrics
    - LLM-based answer quality scoring
    - Aggregating results

    Evaluation scope: retrieval runs through QueryProcessor.process_query with
    no BM25 index, query rewriter, or reranker — i.e. plain semantic-only
    retrieval — to isolate the chunking-strategy comparison. This deliberately
    differs from the shipped default query path (hybrid + rewriting +
    reranking); results must be labeled accordingly wherever they are reported.
    """
    
    def __init__(
        self,
        config,
        query_processor: Optional[QueryProcessor] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        logger_name: str = "evaluator"
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
            query_processor: Query processor (creates default if None)
            answer_generator: Answer generator (creates default if None)
            logger_name: Logger name
        """
        self.config = config
        self.logger = get_logger(logger_name)
        
        # Initialize components
        self.test_loader = TestLoader(config)
        self.metrics_calculator = RetrievalMetrics()
        
        # Query processor (with embedder and vector store)
        if query_processor is None:
            # Initialize embedder with config values
            model_name = config.get("embeddings.model", "all-MiniLM-L6-v2")
            device = config.get("embeddings.device", "cpu")
            batch_size = config.get("embeddings.batch_size", 32)
            embedder = Embedder(
                model_name=model_name,
                device=device,
                batch_size=batch_size
            )
            
            # Initialize vector store with config values
            vector_store_dir = config.get_path("paths.vector_store_dir")
            vector_store = VectorStore(
                persist_directory=vector_store_dir,
                logger_name="evaluator_vector_store"
            )
            
            self.query_processor = QueryProcessor(config, embedder, vector_store)
        else:
            self.query_processor = query_processor
        
        # Answer generator (with LLM client)
        if answer_generator is None:
            # Initialize LLM client with config values
            ollama_url = config.get("generation.ollama_base_url", "http://ollama:11434")
            model_name = config.get("generation.model", "llama3.2:3b")
            timeout = config.get("generation.timeout", 60)
            llm_client = OllamaClient(
                base_url=ollama_url,
                model=model_name,
                timeout=timeout,
                logger_name="evaluator_llm_client"
            )
            self.answer_generator = AnswerGenerator(config, llm_client)
        else:
            self.answer_generator = answer_generator
            llm_client = self.answer_generator.llm_client
        
        # LLM judge (share the same LLM client for efficiency)
        self.llm_judge = LLMJudge(config, llm_client=llm_client)
        
        # Evaluation settings
        self.top_k_values = config.get("evaluation.top_k_values", [5, 10, 20])
        self.strategies = config.get("evaluation.strategies", ["fixed", "semantic", "hierarchical"])
        self.judge_answers = config.get("evaluation.judge_answers", True)
        
        self.logger.info("RAG evaluator initialized")
    
    def run_evaluation(
        self,
        strategy: Optional[str] = None,
        max_questions: Optional[int] = None,
        judge_answers: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            strategy: Specific strategy to evaluate (None for all)
            max_questions: Maximum questions to evaluate (None for all)
            judge_answers: Whether to judge answer quality (None uses config)
            
        Returns:
            Complete evaluation results
        """
        start_time = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("Starting RAG Pipeline Evaluation")
        self.logger.info("=" * 70)
        
        # Load test set
        self.logger.info("Loading test set...")
        questions = self.test_loader.load_test_set()
        
        if max_questions:
            questions = questions[:max_questions]
            self.logger.info(f"Limited to first {max_questions} questions")
        
        # Determine strategies to evaluate
        strategies = [strategy] if strategy else self.strategies
        self.logger.info(f"Evaluating strategies: {strategies}")
        
        # Run evaluation for each strategy
        results_by_strategy = {}
        for strat in strategies:
            self.logger.info(f"\n{'=' * 70}")
            self.logger.info(f"Evaluating strategy: {strat}")
            self.logger.info(f"{'=' * 70}")
            
            strategy_results = self._evaluate_strategy(
                questions, strat, judge_answers
            )
            results_by_strategy[strat] = strategy_results
        
        # Aggregate results
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info("Evaluation Complete")
        self.logger.info(f"Total time: {elapsed_time:.2f}s")
        self.logger.info(f"{'=' * 70}")
        
        # Compile final results
        final_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "strategies_evaluated": strategies,
                "total_time": elapsed_time
            },
            "results_by_strategy": results_by_strategy,
            "comparison": self._compare_strategies(results_by_strategy)
        }
        
        return final_results
    
    def _evaluate_strategy(
        self,
        questions: List[TestQuestion],
        strategy: str,
        judge_answers: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single strategy on all test questions.
        
        Args:
            questions: List of test questions
            strategy: Strategy to evaluate
            judge_answers: Whether to judge answer quality
            
        Returns:
            Strategy evaluation results
        """
        judge_answers = judge_answers if judge_answers is not None else self.judge_answers
        
        per_question_results = []
        
        for i, question in enumerate(questions, 1):
            # Print to console AND log
            progress_msg = f"\n{'='*70}\n[{i}/{len(questions)}] {strategy.upper()}: {question.question}\n{'='*70}"
            self.logger.info(progress_msg)
            print(progress_msg, flush=True)
            
            try:
                result = self._evaluate_single_question(
                    question, strategy, judge_answers
                )
                per_question_results.append(result)
                
                # Show completion status
                status = "✅ SUCCESS" if result.get("success") else "❌ FAILED"
                completion_msg = f"{status} - Question {i}/{len(questions)} complete"
                self.logger.info(completion_msg)
                print(completion_msg, flush=True)
                
            except Exception as e:
                error_msg = f"❌ ERROR on question {i}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                print(error_msg, flush=True)
                
                # Add failed result instead of crashing
                per_question_results.append({
                    "question_id": question.id,
                    "question_text": question.question,
                    "category": question.category,
                    "difficulty": question.difficulty,
                    "success": False,
                    "error": str(e)
                })
                
                # Continue with next question
                continue
        
        # Aggregate metrics
        aggregated = self._aggregate_results(per_question_results)
        
        return {
            "strategy": strategy,
            "per_question": per_question_results,
            "aggregated": aggregated
        }
    
    def _evaluate_single_question(
        self,
        question: TestQuestion,
        strategy: str,
        judge_answers: bool
    ) -> Dict[str, Any]:
        """
        Evaluate a single question.
        
        Args:
            question: Test question
            strategy: Strategy to use
            judge_answers: Whether to judge answer quality
            
        Returns:
            Question evaluation results
        """
        result = {
            "question_id": question.id,
            "question_text": question.question,
            "category": question.category,
            "difficulty": question.difficulty
        }
        
        try:
            # 1. Run retrieval
            print("  → Retrieving chunks...", flush=True)
            retrieval_start = time.time()
            query_results = self.query_processor.process_query(
                query_text=question.question,
                strategy=strategy,
                top_k=max(self.top_k_values),
                show_full_content=True
            )
            retrieval_time = time.time() - retrieval_start
            print(f"  ✓ Retrieved {len(query_results.get('results', []))} chunks in {retrieval_time:.2f}s", flush=True)
            
            retrieved_chunks = query_results.get("results", [])
            
            # Extract doc IDs from chunks
            retrieved_doc_ids = extract_doc_ids_from_chunks(retrieved_chunks)
            
            # 2. Calculate retrieval metrics
            retrieval_metrics = self.metrics_calculator.calculate_all_metrics(
                retrieved_doc_ids,
                question.relevant_doc_ids,
                self.top_k_values
            )
            
            # Also calculate topic coverage
            topic_coverage = self.metrics_calculator.topic_coverage(
                retrieved_chunks,
                question.expected_topics
            )
            retrieval_metrics["topic_coverage"] = topic_coverage
            
            result["retrieval"] = {
                "retrieved_count": len(retrieved_chunks),
                "retrieved_doc_ids": retrieved_doc_ids,
                "metrics": retrieval_metrics,
                "time": retrieval_time
            }
            
            # 3. Generate answer
            if retrieved_chunks:
                print("  → Generating answer...", flush=True)
                generation_start = time.time()
                answer_result = self.answer_generator.generate_answer(
                    query=question.question,
                    retrieved_results=retrieved_chunks
                )
                generation_time = time.time() - generation_start
                print(f"  ✓ Answer generated in {generation_time:.2f}s", flush=True)
                
                generated_answer = answer_result.get("answer", "")
                
                result["generation"] = {
                    "answer": generated_answer,
                    "raw_answer": answer_result.get("raw_answer", ""),
                    "citations_used": answer_result.get("citations_used", []),
                    "chunks_used": answer_result.get("num_chunks_used_in_prompt", 0),
                    "time": generation_time
                }
                
                # 4. Judge answer quality (if enabled)
                if judge_answers and generated_answer:
                    print("  → Judging answer quality (4 criteria)...", flush=True)
                    judge_start = time.time()
                    
                    context = "\n\n".join([
                        chunk.get("content", "")
                        for chunk in retrieved_chunks[:10]  # Use top 10 for context
                    ])
                    
                    judgment = self.llm_judge.judge_answer(
                        question=question.question,
                        answer=generated_answer,
                        context=context
                    )
                    
                    judge_time = time.time() - judge_start
                    
                    # Display all criterion scores
                    scores = judgment.get("scores", {})
                    score_str = ", ".join([f"{k}: {v:.1f}" for k, v in scores.items()])
                    avg_score = judgment.get("average_score")
                    avg_str = f"{avg_score:.2f}" if avg_score is not None else "n/a (all criteria failed)"
                    print(f"  ✓ Judging complete in {judge_time:.2f}s", flush=True)
                    print(f"    Scores: [{score_str}] → avg: {avg_str}", flush=True)
                    
                    result["judgment"] = judgment
            else:
                result["generation"] = None
                result["judgment"] = None
                self.logger.warning("No chunks retrieved, skipping generation")
            
            result["success"] = True
            
        except Exception as e:
            self.logger.error(f"Error evaluating question {question.id}: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def _aggregate_results(
        self,
        per_question_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across all questions.
        
        Args:
            per_question_results: List of per-question results
            
        Returns:
            Aggregated metrics
        """
        # Extract retrieval metrics
        retrieval_metrics_list = []
        for result in per_question_results:
            if result.get("success") and result.get("retrieval"):
                retrieval_metrics_list.append(result["retrieval"]["metrics"])
        
        # Average retrieval metrics
        avg_retrieval_metrics = self.metrics_calculator.average_metrics(retrieval_metrics_list)
        
        # Extract judgment scores
        judgment_scores = []
        for result in per_question_results:
            if result.get("success") and result.get("judgment"):
                judgment_scores.append(result["judgment"])
        
        # Average judgment scores
        avg_judgment = self.llm_judge.summarize_judgments(judgment_scores) if judgment_scores else {}
        
        # Calculate success rate
        success_count = sum(1 for r in per_question_results if r.get("success"))
        success_rate = success_count / len(per_question_results) if per_question_results else 0.0
        
        # Aggregate by category and difficulty
        by_category = self._aggregate_by_attribute(per_question_results, "category")
        by_difficulty = self._aggregate_by_attribute(per_question_results, "difficulty")
        
        return {
            "num_questions": len(per_question_results),
            "success_rate": success_rate,
            "retrieval_metrics": avg_retrieval_metrics,
            "answer_quality": avg_judgment,
            "by_category": by_category,
            "by_difficulty": by_difficulty
        }
    
    def _aggregate_by_attribute(
        self,
        per_question_results: List[Dict[str, Any]],
        attribute: str
    ) -> Dict[str, Any]:
        """Aggregate results by a specific attribute (category or difficulty)."""
        
        grouped = {}
        for result in per_question_results:
            if not result.get("success"):
                continue
            
            attr_value = result.get(attribute)
            if attr_value not in grouped:
                grouped[attr_value] = []
            
            if result.get("retrieval"):
                grouped[attr_value].append(result["retrieval"]["metrics"])
        
        # Average metrics for each group
        aggregated = {}
        for attr_value, metrics_list in grouped.items():
            aggregated[attr_value] = self.metrics_calculator.average_metrics(metrics_list)
        
        return aggregated
    
    def _compare_strategies(
        self,
        results_by_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare strategies and identify best performers.
        
        Args:
            results_by_strategy: Results for each strategy
            
        Returns:
            Strategy comparison
        """
        if len(results_by_strategy) < 2:
            return {}
        
        comparison = {}
        
        # Compare on key metrics
        key_metrics = ["recall@10", "mrr", "ndcg@10", "topic_coverage"]
        
        for metric in key_metrics:
            metric_values = {}
            for strategy, results in results_by_strategy.items():
                value = results.get("aggregated", {}).get("retrieval_metrics", {}).get(metric)
                if value is not None:
                    metric_values[strategy] = value
            
            if metric_values:
                best_strategy = max(metric_values, key=metric_values.get)
                comparison[metric] = {
                    "values": metric_values,
                    "best_strategy": best_strategy,
                    "best_value": metric_values[best_strategy]
                }
        
        return comparison
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_path}")

