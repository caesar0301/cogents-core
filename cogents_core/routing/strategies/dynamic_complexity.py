"""Dynamic complexity routing strategy implementation."""

import math
import re

from cogents_core.routing.base import BaseRoutingStrategy
from cogents_core.routing.types import ComplexityScore, ModelTier, RoutingResult
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


class DynamicComplexityStrategy(BaseRoutingStrategy):
    """
    Routing strategy based on dynamic complexity index calculation.

    This strategy computes a complexity score from multiple signals:
    - Linguistic complexity (sentence structure, vocabulary)
    - Reasoning depth (assessed by lite model)
    - Knowledge uncertainty (perplexity/confidence analysis)
    """

    def __init__(self, lite_client=None, config=None):
        """
        Initialize the dynamic complexity strategy.

        Args:
            lite_client: LLM client for the lite model
            config: Configuration dict with optional parameters:
                - alpha: Weight for linguistic score (default: 0.4)
                - beta: Weight for reasoning score (default: 0.4)
                - gamma: Weight for uncertainty score (default: 0.2)
                - lite_threshold: Max score for lite routing (default: 0.3)
                - fast_threshold: Max score for fast routing (default: 0.65)
                - temperature: Temperature for reasoning assessment (default: 0.1)
                - reasoning_max_tokens: Max tokens for reasoning assessment (default: 3)
                - uncertainty_max_tokens: Max tokens for uncertainty analysis (default: 64)
        """
        super().__init__(lite_client, config)

        # Weighting parameters
        self.alpha = self.config.get("alpha", 0.4)  # Linguistic weight
        self.beta = self.config.get("beta", 0.4)  # Reasoning weight
        self.gamma = self.config.get("gamma", 0.2)  # Uncertainty weight

        # Threshold parameters
        self.lite_threshold = self.config.get("lite_threshold", 0.3)
        self.fast_threshold = self.config.get("fast_threshold", 0.65)

        # LLM parameters
        self.temperature = self.config.get("temperature", 0.1)
        self.reasoning_max_tokens = self.config.get("reasoning_max_tokens", 3)
        self.uncertainty_max_tokens = self.config.get("uncertainty_max_tokens", 64)

        # Validate weights sum to 1.0
        total_weight = self.alpha + self.beta + self.gamma
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0 (sum={total_weight}). Normalizing...")
            self.alpha /= total_weight
            self.beta /= total_weight
            self.gamma /= total_weight

    def route(self, query: str) -> RoutingResult:
        """
        Route query based on dynamic complexity index.

        Args:
            query: Input query to assess

        Returns:
            RoutingResult with tier recommendation and detailed analysis
        """
        try:
            # Calculate individual complexity components
            linguistic_score = self._calculate_linguistic_score(query)
            reasoning_score = self._calculate_reasoning_score(query)
            uncertainty_score = self._calculate_uncertainty_score(query)

            # Compute weighted complexity index
            complexity_index = (
                self.alpha * linguistic_score + self.beta * reasoning_score + self.gamma * uncertainty_score
            )

            # Determine tier based on complexity index
            tier = self._index_to_tier(complexity_index)

            # Calculate confidence based on component consistency
            confidence = self._calculate_confidence(linguistic_score, reasoning_score, uncertainty_score)

            # Create detailed complexity score
            complexity_score_obj = ComplexityScore(
                total=complexity_index,
                linguistic=linguistic_score,
                reasoning=reasoning_score,
                uncertainty=uncertainty_score,
                metadata={
                    "weights": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma},
                    "components": {
                        "linguistic": linguistic_score,
                        "reasoning": reasoning_score,
                        "uncertainty": uncertainty_score,
                    },
                },
            )

            return self._create_result(
                tier=tier,
                confidence=confidence,
                complexity_score=complexity_score_obj,
                metadata={"thresholds": {"lite": self.lite_threshold, "fast": self.fast_threshold}},
            )

        except Exception as e:
            logger.error(f"Error in dynamic complexity routing: {e}")
            # Fallback to fast tier on error
            return self._create_result(
                tier=ModelTier.FAST,
                confidence=0.0,
                complexity_score=ComplexityScore(total=0.5),
                metadata={"error": str(e), "fallback": True},
            )

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "dynamic_complexity"

    def _calculate_linguistic_score(self, query: str) -> float:
        """
        Calculate linguistic complexity based on sentence structure and vocabulary.

        Args:
            query: Input query

        Returns:
            Linguistic complexity score (0.0-1.0)
        """
        try:
            # Handle empty query case
            if not query.strip():
                return 0.0

            # Count tokens (approximate)
            tokens = re.findall(r"\w+", query)
            token_count = len(tokens)

            # Count structural complexity indicators
            clauses = (
                query.count(",")
                + query.count(";")
                + query.count(" and ")
                + query.count(" or ")
                + query.count(" but ")
                + query.count(" because ")
                + query.count(" if ")
                + query.count(" when ")
                + query.count(" while ")
            )

            # Count sentences
            sentences = len(re.split(r"[.!?]+", query.strip()))

            # Calculate complexity factors
            token_factor = min(1.0, token_count / 50.0)  # Normalize around 50 tokens
            clause_factor = min(1.0, clauses / 5.0)  # Normalize around 5 clauses
            sentence_factor = min(1.0, sentences / 3.0)  # Normalize around 3 sentences

            # Count complex words (>6 characters as simple heuristic)
            complex_words = sum(1 for token in tokens if len(token) > 6)
            vocab_factor = min(1.0, complex_words / max(1, token_count) * 2)

            # Weighted combination
            linguistic_score = 0.3 * token_factor + 0.3 * clause_factor + 0.2 * sentence_factor + 0.2 * vocab_factor

            return min(1.0, max(0.0, linguistic_score))

        except Exception as e:
            logger.warning(f"Error calculating linguistic score: {e}")
            return 0.5

    def _calculate_reasoning_score(self, query: str) -> float:
        """
        Calculate reasoning depth using lite model assessment.

        Args:
            query: Input query

        Returns:
            Reasoning complexity score (0.0-1.0)
        """
        if not self.lite_client:
            # Fallback: simple keyword-based reasoning detection
            return self._fallback_reasoning_score(query)

        try:
            prompt = f"""Classify reasoning depth of request:
- 0 = factual recall
- 1 = some reasoning/planning
- 2 = multi-step or abstract reasoning

Request: "{query}"
Output: number only"""

            messages = [{"role": "user", "content": prompt}]
            response = self.lite_client.completion(
                messages=messages, temperature=self.temperature, max_tokens=self.reasoning_max_tokens
            )

            # Parse response
            response_clean = response.strip()
            for char in response_clean:
                if char.isdigit():
                    score = int(char)
                    if 0 <= score <= 2:
                        return score / 2.0  # Normalize to 0-1 range

            # Fallback if parsing fails
            logger.warning(f"Could not parse reasoning response: {response}")
            return self._fallback_reasoning_score(query)

        except Exception as e:
            logger.warning(f"Error calculating reasoning score with LLM: {e}")
            return self._fallback_reasoning_score(query)

    def _fallback_reasoning_score(self, query: str) -> float:
        """Fallback reasoning score based on keywords."""
        reasoning_keywords = [
            "analyze",
            "compare",
            "evaluate",
            "explain",
            "why",
            "how",
            "cause",
            "effect",
            "relationship",
            "implication",
            "conclusion",
            "strategy",
            "plan",
            "design",
            "create",
            "develop",
            "solve",
        ]

        query_lower = query.lower()
        keyword_count = sum(1 for keyword in reasoning_keywords if keyword in query_lower)

        return min(1.0, keyword_count / 3.0)  # Normalize around 3 keywords

    def _calculate_uncertainty_score(self, query: str) -> float:
        """
        Calculate knowledge uncertainty using perplexity analysis.

        Args:
            query: Input query

        Returns:
            Uncertainty score (0.0-1.0)
        """
        if not self.lite_client:
            # Fallback: domain-based uncertainty estimation
            return self._fallback_uncertainty_score(query)

        try:
            # Check if we can get logprobs (depends on the LLM client implementation)
            messages = [{"role": "user", "content": query}]

            # Try to get response with some generation to assess uncertainty
            response = self.lite_client.completion(
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=self.uncertainty_max_tokens,
            )

            # For now, use response length and coherence as uncertainty proxy
            # A very short or very long response might indicate uncertainty
            response_tokens = len(response.split())

            if response_tokens < 5:  # Very short response
                uncertainty = 0.7
            elif response_tokens > 40:  # Very long response
                uncertainty = 0.6
            else:
                uncertainty = 0.3  # Normal length suggests confidence

            # Adjust based on hedging words
            hedging_words = ["maybe", "perhaps", "possibly", "might", "could", "uncertain"]
            hedging_count = sum(1 for word in hedging_words if word in response.lower())
            uncertainty += min(0.3, hedging_count * 0.1)

            return min(1.0, max(0.0, uncertainty))

        except Exception as e:
            logger.warning(f"Error calculating uncertainty score with LLM: {e}")
            return self._fallback_uncertainty_score(query)

    def _fallback_uncertainty_score(self, query: str) -> float:
        """Fallback uncertainty score based on domain heuristics."""
        # Questions tend to have higher uncertainty
        question_count = query.count("?")

        # Specific vs general queries
        specific_indicators = ["specific", "exact", "precise", "particular"]
        general_indicators = ["general", "overview", "broad", "overall"]

        query_lower = query.lower()
        specific_score = sum(1 for word in specific_indicators if word in query_lower)
        general_score = sum(1 for word in general_indicators if word in query_lower)

        base_uncertainty = 0.4
        uncertainty_adjustment = question_count * 0.1 + general_score * 0.1 - specific_score * 0.1

        return min(1.0, max(0.0, base_uncertainty + uncertainty_adjustment))

    def _index_to_tier(self, complexity_index: float) -> ModelTier:
        """
        Convert complexity index to model tier.

        Args:
            complexity_index: Overall complexity score (0.0-1.0)

        Returns:
            Appropriate ModelTier
        """
        if complexity_index < self.lite_threshold:
            return ModelTier.LITE
        elif complexity_index < self.fast_threshold:
            return ModelTier.FAST
        else:
            return ModelTier.POWER

    def _calculate_confidence(self, linguistic: float, reasoning: float, uncertainty: float) -> float:
        """
        Calculate confidence based on consistency of component scores.

        Args:
            linguistic: Linguistic complexity score
            reasoning: Reasoning complexity score
            uncertainty: Uncertainty score

        Returns:
            Confidence score (0.0-1.0)
        """
        scores = [linguistic, reasoning, uncertainty]

        # Calculate standard deviation as measure of consistency
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = math.sqrt(variance)

        # Higher consistency (lower std_dev) = higher confidence
        # Max std_dev is ~0.58 (when scores are maximally spread, e.g., 0, 1, 0.5)
        # Make it more sensitive to inconsistency
        consistency = 1.0 - min(1.0, std_dev * 2.5)

        # Base confidence adjusted by consistency
        base_confidence = 0.6
        confidence = base_confidence + (consistency * 0.4)

        return min(1.0, max(0.0, confidence))
