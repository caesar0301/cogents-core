"""Self-assessment routing strategy implementation."""

from cogents_core.routing.base import BaseRoutingStrategy
from cogents_core.routing.types import ComplexityScore, ModelTier, RoutingResult
from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)


class SelfAssessmentStrategy(BaseRoutingStrategy):
    """
    Routing strategy where the lite model assesses query complexity itself.

    This strategy asks the lite model to rate the complexity of a query
    and routes based on that self-assessment.
    """

    def __init__(self, lite_client=None, config=None):
        """
        Initialize the self-assessment strategy.

        Args:
            lite_client: LLM client for the lite model
            config: Configuration dict with optional parameters:
                - temperature: Temperature for lite model (default: 0.1)
                - max_tokens: Max tokens for assessment (default: 5)
                - lite_threshold: Max score for lite routing (default: 1)
                - fast_threshold: Max score for fast routing (default: 3)
        """
        super().__init__(lite_client, config)

        if not self.lite_client:
            raise ValueError("SelfAssessmentStrategy requires a lite_client")

        # Configuration parameters
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 5)
        self.lite_threshold = self.config.get("lite_threshold", 1)
        self.fast_threshold = self.config.get("fast_threshold", 3)

    def route(self, query: str) -> RoutingResult:
        """
        Route query based on lite model's self-assessment.

        Args:
            query: Input query to assess

        Returns:
            RoutingResult with tier recommendation
        """
        try:
            # Create assessment prompt
            prompt = self._create_assessment_prompt(query)
            messages = [{"role": "user", "content": prompt}]

            # Get assessment from lite model
            response = self.lite_client.completion(
                messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Parse complexity score
            complexity_score, confidence = self._parse_assessment_response(response)

            # Determine tier based on score
            tier = self._score_to_tier(complexity_score)

            # Create complexity score object
            score_obj = ComplexityScore(
                total=complexity_score / 5.0,  # Normalize to 0-1 range
                metadata={"raw_score": complexity_score, "raw_response": response},
            )

            return self._create_result(
                tier=tier, confidence=confidence, complexity_score=score_obj, metadata={"raw_assessment": response}
            )

        except Exception as e:
            logger.error(f"Error in self-assessment routing: {e}")
            # Fallback to fast tier on error
            return self._create_result(
                tier=ModelTier.FAST,
                confidence=0.0,
                complexity_score=ComplexityScore(total=0.5),
                metadata={"error": str(e), "fallback": True},
            )

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "self_assessment"

    def _create_assessment_prompt(self, query: str) -> str:
        """Create the assessment prompt for the lite model."""
        return f"""You are a request classifier. Rate the complexity of the following request:
- 1 = simple fact or direct instruction (lite model can handle)
- 2-3 = reasoning or multi-sentence but not too complex (fast model recommended)
- 4-5 = deep reasoning, creativity, or novel synthesis (power model required)

Request: "{query}"

Output ONLY a number from 1 to 5:"""

    def _parse_assessment_response(self, response: str) -> tuple[int, float]:
        """
        Parse the assessment response to extract complexity score.

        Args:
            response: Raw response from lite model

        Returns:
            Tuple of (complexity_score, confidence)
        """
        try:
            # Extract first digit from response
            response_clean = response.strip()
            for char in response_clean:
                if char.isdigit():
                    score = int(char)
                    if 1 <= score <= 5:
                        # Higher confidence if response is clean (just the number with no extra text)
                        is_clean = response_clean.strip() == str(score)
                        confidence = 0.9 if is_clean else 0.7
                        return score, confidence

            # Fallback if no valid digit found
            logger.warning(f"Could not parse assessment response: {response}")
            return 3, 0.3  # Default to medium complexity with low confidence

        except Exception as e:
            logger.error(f"Error parsing assessment response: {e}")
            return 3, 0.1

    def _score_to_tier(self, score: int) -> ModelTier:
        """
        Convert complexity score to model tier.

        Args:
            score: Complexity score from 1-5

        Returns:
            Appropriate ModelTier
        """
        if score <= self.lite_threshold:
            return ModelTier.LITE
        elif score <= self.fast_threshold:
            return ModelTier.FAST
        else:
            return ModelTier.POWER
