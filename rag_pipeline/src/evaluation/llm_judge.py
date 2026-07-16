"""LLM-based judge for answer quality assessment."""

from __future__ import annotations
from typing import Dict, Any, Optional
import re
import time

from ..generation.llm_client import OllamaClient
from ..utils.logger import get_logger


class LLMJudge:
    """
    Uses an LLM to evaluate answer quality.
    
    Scores answers on multiple dimensions using structured prompts.
    """
    
    def __init__(
        self,
        config,
        llm_client: Optional[OllamaClient] = None,
        logger_name: str = "llm_judge"
    ):
        """
        Initialize LLM judge.
        
        Args:
            config: Configuration object
            llm_client: LLM client (creates default if None)
            logger_name: Logger name
        """
        self.config = config
        
        # Initialize LLM client if not provided
        if llm_client is None:
            ollama_url = config.get("generation.ollama_base_url", "http://ollama:11434")
            model_name = config.get("generation.model", "llama3.2:3b")
            timeout = config.get("generation.timeout", 60)
            self.llm_client = OllamaClient(
                base_url=ollama_url,
                model=model_name,
                timeout=timeout,
                logger_name="llm_judge_client"
            )
        else:
            self.llm_client = llm_client
        
        self.logger = get_logger(logger_name)
        
        # Use lower temperature for more consistent judgments
        self.judge_temperature = 0.3
        self.judge_max_tokens = 500  # Increased for longer, more detailed explanations
        
        self.logger.info("LLM judge initialized")
    
    def judge_answer(
        self,
        question: str,
        answer: str,
        context: str,
        criteria: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Judge answer quality across multiple criteria.
        
        Args:
            question: Original question
            answer: Generated answer to evaluate
            context: Context/chunks used to generate answer
            criteria: List of criteria to evaluate (uses defaults if None)
            
        Returns:
            Dictionary with scores and explanations
        """
        if criteria is None:
            # Use all criteria for comprehensive evaluation
            criteria = ["faithfulness", "relevance", "completeness", "overall"]
        
        start_time = time.time()
        
        scores = {}
        explanations = {}
        
        failed_criteria = []
        for criterion in criteria:
            self.logger.info(f"Judging '{criterion}' for question: {question[:50]}...")

            score, explanation = self._judge_criterion(
                question, answer, context, criterion
            )

            explanations[criterion] = explanation
            if score is None:
                # A failed LLM call is missing data, not a neutral 3/5 —
                # excluding it keeps the averages honest.
                failed_criteria.append(criterion)
            else:
                scores[criterion] = score

        elapsed_time = time.time() - start_time

        return {
            "scores": scores,
            "explanations": explanations,
            "failed_criteria": failed_criteria,
            "average_score": sum(scores.values()) / len(scores) if scores else None,
            "judging_time": elapsed_time
        }
    
    def _judge_criterion(
        self,
        question: str,
        answer: str,
        context: str,
        criterion: str
    ) -> tuple[Optional[float], str]:
        """
        Judge answer on a specific criterion.

        Args:
            question: Original question
            answer: Generated answer
            context: Context used
            criterion: Criterion to evaluate

        Returns:
            Tuple of (score, explanation). The score is None when the LLM call
            failed or returned nothing — callers must exclude it from averages
            rather than treat the failure as a neutral 3/5.
        """
        # Build prompt
        prompt = self._build_judging_prompt(question, answer, context, criterion)
        
        # Log the prompt being sent (debug level only to reduce noise)
        self.logger.debug(f"\n{'='*70}")
        self.logger.debug(f"Judging criterion: {criterion}")
        self.logger.debug(f"Question: {question[:100]}...")
        self.logger.debug(f"Answer length: {len(answer)} chars")
        self.logger.debug(f"Context length: {len(context)} chars")
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        try:
            # Make LLM call
            self.logger.debug("Calling LLM...")
            start_time = time.time()
            
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.judge_temperature,
                max_tokens=self.judge_max_tokens
            )
            
            elapsed = time.time() - start_time
            self.logger.debug(f"LLM responded in {elapsed:.2f}s")
            
            # Extract response text (Ollama returns it in "response" field, not "text")
            response_text = response.get("response", "")
            
            # Log the complete response (debug level to reduce noise)
            self.logger.debug(f"Response length: {len(response_text)} chars")
            self.logger.debug(f"Full LLM response:\n{'-'*70}\n{response_text}\n{'-'*70}")
            
            # Check for empty response
            if not response_text or response_text.strip() == "":
                self.logger.error(f"❌ Empty response from LLM for criterion: {criterion}")
                self.logger.error(f"Full response dict: {response}")
                return None, "Empty LLM response"
            
            # Parse the response
            self.logger.debug("Parsing response...")
            score, explanation = self._parse_judgment(response_text)
            
            self.logger.info(f"✅ {criterion}: {score}/5")
            self.logger.debug(f"Explanation: {explanation[:200]}...")
            
            return score, explanation
            
        except Exception as e:
            self.logger.error(f"❌ Error judging {criterion}: {e}", exc_info=True)
            return None, f"Error during judgment: {str(e)}"
    
    def _build_judging_prompt(
        self,
        question: str,
        answer: str,
        context: str,
        criterion: str
    ) -> str:
        """Build judging prompt for a specific criterion with few-shot examples."""
        
        # Few-shot examples for calibration
        few_shot_examples = {
            "faithfulness": """
CALIBRATION EXAMPLES:

Example 1 - Score: 5
Question: "What does StandardScaler do?"
Context: "StandardScaler standardizes features by removing the mean and scaling to unit variance."
Answer: "StandardScaler standardizes features by removing the mean and scaling to unit variance."
Reasoning: Answer uses only information from context, no hallucinations.

Example 2 - Score: 3
Question: "What does StandardScaler do?"
Context: "StandardScaler standardizes features by removing the mean and scaling to unit variance."
Answer: "StandardScaler normalizes your data by scaling features. It's commonly used for preprocessing."
Reasoning: Partially correct but uses term "normalizes" which isn't in context, and adds vague claims.

Example 3 - Score: 1
Question: "What does StandardScaler do?"
Context: "StandardScaler standardizes features by removing the mean and scaling to unit variance."
Answer: "StandardScaler is used for encoding categorical variables into numerical format."
Reasoning: Completely contradicts the context - this describes encoding, not scaling.
""",
            "relevance": """
CALIBRATION EXAMPLES:

Example 1 - Score: 5
Question: "How do I use StandardScaler?"
Answer: "Import StandardScaler from sklearn.preprocessing, create an instance with scaler = StandardScaler(), then use scaler.fit_transform(X) on your training data."
Reasoning: Directly answers the how-to question with specific steps.

Example 2 - Score: 3
Question: "How do I use StandardScaler?"
Answer: "StandardScaler is a preprocessing tool that removes the mean and scales to unit variance. It's important for many machine learning algorithms."
Reasoning: Explains what it does (relevant) but doesn't answer how to use it (the actual question).

Example 3 - Score: 1
Question: "How do I use StandardScaler?"
Answer: "Principal Component Analysis (PCA) is used for dimensionality reduction in machine learning."
Reasoning: Completely off-topic, answers about PCA instead of StandardScaler.
""",
            "completeness": """
CALIBRATION EXAMPLES:

Example 1 - Score: 5
Question: "What is cross-validation?"
Answer: "Cross-validation is a technique for assessing model performance by splitting data into k folds, training on k-1 folds, and testing on the remaining fold. This process repeats k times with each fold used once as test set. It helps detect overfitting and provides more reliable performance estimates than a single train-test split."
Reasoning: Covers definition, process, and purpose comprehensively.

Example 2 - Score: 3
Question: "What is cross-validation?"
Answer: "Cross-validation splits your data into parts and tests your model multiple times to get better performance estimates."
Reasoning: Basic definition present but lacks details about k-folds, rotation process, and specific benefits.

Example 3 - Score: 1
Question: "What is cross-validation?"
Answer: "It's a technique used in machine learning."
Reasoning: Extremely vague, missing all key information about what it does and how it works.
""",
            "overall": """
CALIBRATION EXAMPLES:

Example 1 - Score: 5
Question: "How do I handle missing values?"
Answer: "To handle missing values in scikit-learn, use SimpleImputer from sklearn.impute. For basic strategies: imputer = SimpleImputer(strategy='mean') replaces with column means. You can also use strategy='median' or 'most_frequent'. For more sophisticated imputation, KNNImputer uses k-nearest neighbors to estimate missing values."
Reasoning: Comprehensive, accurate, includes code, covers multiple approaches - truly excellent.

Example 2 - Score: 3
Question: "How do I handle missing values?"
Answer: "You can use an imputer to fill in missing values with the mean or median."
Reasoning: Correct but minimal - lacks implementation details, class names, or examples.

Example 3 - Score: 1
Question: "How do I handle missing values?"
Answer: "Just remove all rows with missing data from your dataset."
Reasoning: Poor advice (data loss), doesn't mention any scikit-learn tools, unhelpful.
"""
        }
        
        # Critical evaluation instructions
        criteria_prompts = {
            "faithfulness": """BE STRICT: Evaluate if the answer is faithful to the provided context.

Score 1-5 where:
1 = Answer contradicts context or invents information not present
2 = Answer has multiple unsupported claims or misrepresents context
3 = Answer generally follows context but includes some unverified information
4 = Answer accurately uses context with only one minor unsupported detail
5 = Answer is PERFECTLY faithful - every claim is directly supported by context

CRITICAL MINDSET: Look for any information that wasn't explicitly in the context. Even small additions should lower the score. Only perfect faithfulness deserves a 5.""",
            
            "relevance": """BE STRICT: Evaluate if the answer directly addresses what was asked.

Score 1-5 where:
1 = Completely off-topic, answers wrong question
2 = Tangentially related but misses the main question
3 = Addresses question but with unnecessary tangents or lacks focus
4 = Relevant with one minor off-topic element
5 = PERFECTLY relevant - every sentence directly addresses the question

CRITICAL MINDSET: If the answer explains related concepts but doesn't answer the actual question, it's not a 5. Don't be generous.""",
            
            "completeness": """BE STRICT: Evaluate if the answer covers all important aspects.

Score 1-5 where:
1 = Missing most key information, essentially useless
2 = Covers less than half of what should be included
3 = Covers basic points but missing important details or examples
4 = Fairly complete, missing only one minor detail
5 = PERFECTLY complete - covers definition, usage, examples, edge cases as needed

CRITICAL MINDSET: Think about what a thorough answer should include. Missing code examples, use cases, or key details should lower the score.""",
            
            "overall": """BE STRICT: Evaluate overall answer quality critically.

Score 1-5 where:
1 = Incorrect, misleading, or useless
2 = Partially correct but has significant problems (accuracy, clarity, or completeness)
3 = Acceptable minimum - correct but basic, lacking detail or examples
4 = Good answer - accurate, reasonably complete, helpful
5 = EXCELLENT - accurate, comprehensive, well-explained with examples, truly outstanding

CRITICAL MINDSET: Reserve 5 for truly exceptional answers. Most good answers are 3-4. Don't inflate scores - be a tough grader. What issues or improvements can you identify?"""
        }
        
        criterion_instructions = criteria_prompts.get(
            criterion,
            "Evaluate the answer quality on a scale of 1-5."
        )
        
        calibration_examples = few_shot_examples.get(criterion, "")
        
        prompt = f"""You are a STRICT expert evaluator assessing AI-generated answers. Be critical and look for flaws.

{calibration_examples}

NOW EVALUATE THIS ANSWER:

Question: {question}

Context provided to the AI:
{context[:1500]}...

Generated Answer:
{answer}

{criterion_instructions}

Remember: Be strict. Look for problems. Only exceptional answers deserve 5. Most answers have room for improvement.

Provide your evaluation in this EXACT format:
Score: [1-5]
Explanation: [Your critical reasoning in 2-3 sentences, focusing on what's missing or problematic]"""
        
        return prompt
    
    def _parse_judgment(self, response_text: str) -> tuple[float, str]:
        """
        Parse LLM judgment response to extract score and explanation.
        
        Tries multiple patterns to be robust to various LLM output formats.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Tuple of (score, explanation)
        """
        score = None
        parsing_log = []
        
        # Pattern 1: "Score: X" or "Score X"
        parsing_log.append("Trying pattern 1: 'Score: X' or 'Score X'")
        score_match = re.search(r'Score[:\s]+(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                parsing_log.append(f"✓ Pattern 1 matched: {score}")
            except ValueError:
                parsing_log.append("✗ Pattern 1 matched but couldn't convert to float")
        else:
            parsing_log.append("✗ Pattern 1 not found")
        
        # Pattern 2: "X/5" or "X out of 5"
        if score is None:
            parsing_log.append("Trying pattern 2: 'X/5' or 'X out of 5'")
            score_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/|out of)\s*5', response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    parsing_log.append(f"✓ Pattern 2 matched: {score}")
                except ValueError:
                    parsing_log.append("✗ Pattern 2 matched but couldn't convert to float")
            else:
                parsing_log.append("✗ Pattern 2 not found")
        
        # Pattern 3: "Rating: X" or "Rate: X"
        if score is None:
            parsing_log.append("Trying pattern 3: 'Rating: X' or 'Rate: X'")
            score_match = re.search(r'Rat(?:e|ing)[:\s]+(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    parsing_log.append(f"✓ Pattern 3 matched: {score}")
                except ValueError:
                    parsing_log.append("✗ Pattern 3 matched but couldn't convert to float")
            else:
                parsing_log.append("✗ Pattern 3 not found")
        
        # Pattern 4: First number 1-5 in the response
        if score is None:
            parsing_log.append("Trying pattern 4: First number 1-5 anywhere")
            number_match = re.search(r'\b([1-5])\b', response_text)
            if number_match:
                score = float(number_match.group(1))
                parsing_log.append(f"✓ Pattern 4 matched: {score}")
            else:
                parsing_log.append("✗ Pattern 4 not found")
        
        # Log parsing attempts (debug level)
        self.logger.debug("Score parsing attempts:")
        for log_line in parsing_log:
            self.logger.debug(f"  {log_line}")
        
        # Default if nothing found
        if score is None:
            self.logger.warning("❌ Could not extract score from any pattern!")
            self.logger.warning(f"Response text: {response_text}")
            score = 3.0
        else:
            # Clamp to 1-5 range
            score = max(1.0, min(5.0, score))
        
        # Extract explanation - try multiple patterns
        explanation = None
        
        # Pattern 1: "Explanation: ..."
        explanation_match = re.search(
            r'Explanation[:\s]+(.+?)(?:\n\n|\Z)',
            response_text,
            re.IGNORECASE | re.DOTALL
        )
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            self.logger.debug("Explanation extracted using pattern 'Explanation: ...'")
        
        # Pattern 2: Text after score
        if explanation is None:
            # Find text after the score pattern
            parts = re.split(r'Score[:\s]+\d+', response_text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                explanation = parts[1].strip()
                self.logger.debug("Explanation extracted from text after score")
        
        # Fallback: use entire response
        if explanation is None or len(explanation) < 10:
            explanation = response_text.strip()
            self.logger.debug("Using entire response as explanation (fallback)")
        
        return score, explanation
    
    def batch_judge_answers(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[str],
        criteria: Optional[list] = None
    ) -> list[Dict[str, Any]]:
        """
        Judge multiple answers in batch.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of contexts used
            criteria: Criteria to evaluate
            
        Returns:
            List of judgment dictionaries
        """
        if not (len(questions) == len(answers) == len(contexts)):
            raise ValueError("questions, answers, and contexts must have same length")
        
        self.logger.info(f"Batch judging {len(questions)} answers...")
        
        results = []
        for i, (question, answer, context) in enumerate(zip(questions, answers, contexts)):
            self.logger.info(f"Judging answer {i+1}/{len(questions)}")
            judgment = self.judge_answer(question, answer, context, criteria)
            results.append(judgment)
        
        self.logger.info("Batch judging complete")
        return results
    
    def summarize_judgments(
        self,
        judgments: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Summarize judgments across multiple answers.
        
        Args:
            judgments: List of judgment dictionaries
            
        Returns:
            Summary statistics
        """
        if not judgments:
            return {}
        
        # Extract all criteria
        all_criteria = set()
        for judgment in judgments:
            all_criteria.update(judgment.get("scores", {}).keys())
        
        summary = {}
        
        # Calculate averages for each criterion, skipping judgments where the
        # LLM call failed (score absent) instead of counting them as 0.
        for criterion in all_criteria:
            scores = [
                s
                for s in (j.get("scores", {}).get(criterion) for j in judgments)
                if s is not None
            ]
            if scores:
                summary[f"{criterion}_mean"] = sum(scores) / len(scores)
                summary[f"{criterion}_min"] = min(scores)
                summary[f"{criterion}_max"] = max(scores)
        
        # Overall average
        avg_scores = [j.get("average_score", 0.0) for j in judgments]
        summary["overall_mean"] = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        
        return summary

