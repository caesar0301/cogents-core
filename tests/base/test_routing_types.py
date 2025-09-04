"""
Unit tests for routing types module.

Tests cover ModelTier enum, ComplexityScore and RoutingResult Pydantic models.
"""

import pytest
from pydantic import ValidationError

from cogents_core.routing.types import ComplexityScore, ModelTier, RoutingResult


class TestModelTier:
    """Test cases for ModelTier enum."""

    def test_model_tier_values(self):
        """Test that ModelTier has correct values."""
        assert ModelTier.LITE == "lite"
        assert ModelTier.FAST == "fast"
        assert ModelTier.POWER == "power"

    def test_model_tier_string_inheritance(self):
        """Test that ModelTier inherits from str."""
        assert isinstance(ModelTier.LITE, str)
        assert isinstance(ModelTier.FAST, str)
        assert isinstance(ModelTier.POWER, str)

    def test_model_tier_comparison(self):
        """Test ModelTier comparison with strings."""
        assert ModelTier.LITE == "lite"
        assert ModelTier.FAST == "fast"
        assert ModelTier.POWER == "power"

    def test_model_tier_iteration(self):
        """Test iterating over ModelTier values."""
        tiers = list(ModelTier)
        expected = [ModelTier.LITE, ModelTier.FAST, ModelTier.POWER]
        assert tiers == expected


class TestComplexityScore:
    """Test cases for ComplexityScore model."""

    def test_valid_complexity_score_creation(self):
        """Test creating ComplexityScore with valid data."""
        score = ComplexityScore(total=0.5, linguistic=0.4, reasoning=0.6, uncertainty=0.3, metadata={"test": "value"})

        assert score.total == 0.5
        assert score.linguistic == 0.4
        assert score.reasoning == 0.6
        assert score.uncertainty == 0.3
        assert score.metadata == {"test": "value"}

    def test_complexity_score_with_minimal_data(self):
        """Test creating ComplexityScore with only required fields."""
        score = ComplexityScore(total=0.75)

        assert score.total == 0.75
        assert score.linguistic is None
        assert score.reasoning is None
        assert score.uncertainty is None
        assert score.metadata is None

    def test_complexity_score_validation_total_range(self):
        """Test that total score must be in range 0.0-1.0."""
        # Valid boundary values
        ComplexityScore(total=0.0)
        ComplexityScore(total=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            ComplexityScore(total=-0.1)

        with pytest.raises(ValidationError):
            ComplexityScore(total=1.1)

    def test_complexity_score_validation_component_ranges(self):
        """Test that component scores must be in range 0.0-1.0."""
        # Valid boundary values
        ComplexityScore(total=0.5, linguistic=0.0, reasoning=1.0, uncertainty=0.5)

        # Invalid linguistic
        with pytest.raises(ValidationError):
            ComplexityScore(total=0.5, linguistic=-0.1)

        with pytest.raises(ValidationError):
            ComplexityScore(total=0.5, linguistic=1.1)

        # Invalid reasoning
        with pytest.raises(ValidationError):
            ComplexityScore(total=0.5, reasoning=-0.1)

        # Invalid uncertainty
        with pytest.raises(ValidationError):
            ComplexityScore(total=0.5, uncertainty=1.1)

    def test_complexity_score_serialization(self):
        """Test ComplexityScore serialization."""
        score = ComplexityScore(total=0.7, linguistic=0.6, reasoning=0.8, uncertainty=0.4, metadata={"source": "test"})

        data = score.model_dump()
        expected = {
            "total": 0.7,
            "linguistic": 0.6,
            "reasoning": 0.8,
            "uncertainty": 0.4,
            "metadata": {"source": "test"},
        }
        assert data == expected

    def test_complexity_score_from_dict(self):
        """Test creating ComplexityScore from dictionary."""
        data = {"total": 0.65, "linguistic": 0.7, "reasoning": 0.5, "uncertainty": 0.75}

        score = ComplexityScore(**data)
        assert score.total == 0.65
        assert score.linguistic == 0.7
        assert score.reasoning == 0.5
        assert score.uncertainty == 0.75


class TestRoutingResult:
    """Test cases for RoutingResult model."""

    @pytest.fixture
    def sample_complexity_score(self):
        """Provide a sample ComplexityScore for testing."""
        return ComplexityScore(total=0.6, linguistic=0.5, reasoning=0.7, uncertainty=0.4)

    def test_valid_routing_result_creation(self, sample_complexity_score):
        """Test creating RoutingResult with valid data."""
        result = RoutingResult(
            tier=ModelTier.FAST,
            confidence=0.85,
            complexity_score=sample_complexity_score,
            strategy="test_strategy",
            metadata={"test": "value"},
        )

        assert result.tier == ModelTier.FAST
        assert result.confidence == 0.85
        assert result.complexity_score == sample_complexity_score
        assert result.strategy == "test_strategy"
        assert result.metadata == {"test": "value"}

    def test_routing_result_with_minimal_data(self, sample_complexity_score):
        """Test creating RoutingResult with only required fields."""
        result = RoutingResult(
            tier=ModelTier.LITE, confidence=0.9, complexity_score=sample_complexity_score, strategy="minimal_strategy"
        )

        assert result.tier == ModelTier.LITE
        assert result.confidence == 0.9
        assert result.complexity_score == sample_complexity_score
        assert result.strategy == "minimal_strategy"
        assert result.metadata is None

    def test_routing_result_confidence_validation(self, sample_complexity_score):
        """Test that confidence must be in range 0.0-1.0."""
        # Valid boundary values
        RoutingResult(tier=ModelTier.FAST, confidence=0.0, complexity_score=sample_complexity_score, strategy="test")

        RoutingResult(tier=ModelTier.FAST, confidence=1.0, complexity_score=sample_complexity_score, strategy="test")

        # Invalid values
        with pytest.raises(ValidationError):
            RoutingResult(
                tier=ModelTier.FAST, confidence=-0.1, complexity_score=sample_complexity_score, strategy="test"
            )

        with pytest.raises(ValidationError):
            RoutingResult(
                tier=ModelTier.FAST, confidence=1.1, complexity_score=sample_complexity_score, strategy="test"
            )

    def test_routing_result_tier_validation(self, sample_complexity_score):
        """Test that tier must be a valid ModelTier."""
        # Valid ModelTier values
        for tier in ModelTier:
            RoutingResult(tier=tier, confidence=0.8, complexity_score=sample_complexity_score, strategy="test")

    def test_routing_result_serialization(self, sample_complexity_score):
        """Test RoutingResult serialization."""
        result = RoutingResult(
            tier=ModelTier.POWER,
            confidence=0.95,
            complexity_score=sample_complexity_score,
            strategy="dynamic_complexity",
            metadata={"threshold": 0.8},
        )

        data = result.model_dump()
        assert data["tier"] == "power"
        assert data["confidence"] == 0.95
        assert data["strategy"] == "dynamic_complexity"
        assert data["metadata"] == {"threshold": 0.8}
        assert "complexity_score" in data

    def test_routing_result_from_dict(self):
        """Test creating RoutingResult from dictionary."""
        data = {
            "tier": "fast",
            "confidence": 0.75,
            "complexity_score": {"total": 0.5},
            "strategy": "self_assessment",
            "metadata": {"source": "test"},
        }

        result = RoutingResult(**data)
        assert result.tier == ModelTier.FAST
        assert result.confidence == 0.75
        assert result.complexity_score.total == 0.5
        assert result.strategy == "self_assessment"
        assert result.metadata == {"source": "test"}

    def test_routing_result_extra_fields_forbidden(self, sample_complexity_score):
        """Test that extra fields are forbidden in RoutingResult."""
        with pytest.raises(ValidationError):
            RoutingResult(
                tier=ModelTier.LITE,
                confidence=0.8,
                complexity_score=sample_complexity_score,
                strategy="test",
                extra_field="not_allowed",  # This should raise ValidationError
            )
