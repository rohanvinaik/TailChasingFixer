"""
Tests for LLM adapters and manager.

Tests the LLM abstraction layer with mocking to avoid requiring actual API keys.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from tailchasing.llm.adapters.base import (
    BaseLLMAdapter, LLMResponse, ModelTier, CostTracker, 
    RetryConfig, get_prompt_template
)
from tailchasing.llm.adapters.openai_adapter import OpenAIAdapter
from tailchasing.llm.adapters.anthropic_adapter import AnthropicAdapter
from tailchasing.llm.adapters.ollama_adapter import OllamaAdapter
from tailchasing.llm.manager import LLMManager, TaskComplexity
from tailchasing.llm.prompts import TailChasingPrompts, PromptContext, create_context_aware_prompt
from tailchasing.core.issues import Issue


class MockAdapter(BaseLLMAdapter):
    """Mock adapter for testing base functionality."""
    
    def __init__(self, model_tier=ModelTier.STANDARD, fail_count=0):
        super().__init__()
        self._model_tier = model_tier
        self._fail_count = fail_count
        self._call_count = 0
    
    @property
    def model_tier(self) -> ModelTier:
        return self._model_tier
    
    def _make_api_call(self, prompt: str, **kwargs) -> LLMResponse:
        self._call_count += 1
        
        if self._call_count <= self._fail_count:
            raise Exception("Simulated API failure")
        
        return LLMResponse(
            content='{"content": "Fixed code here", "confidence": 0.8, "rationale": "Test fix"}',
            model_used="mock",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01
        )


@pytest.fixture
def sample_issue():
    """Create a sample issue for testing."""
    return Issue(
        kind="phantom_function",
        message="Function has no implementation",
        severity=3,
        file="test.py",
        line=10,
        symbol="empty_function",
        evidence={"stub_type": "pass"}
    )


@pytest.fixture
def sample_source_code():
    """Sample source code for testing."""
    return """def empty_function():
    pass  # TODO: implement later

def working_function():
    return "this works"
"""


class TestCostTracker:
    """Test cost tracking functionality."""
    
    def test_cost_tracker_initialization(self):
        """Test cost tracker starts with zero values."""
        tracker = CostTracker()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.call_count == 0
    
    def test_add_usage(self):
        """Test adding usage statistics."""
        tracker = CostTracker()
        tracker.add_usage(100, 50, 0.05)
        
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.total_cost_usd == 0.05
        assert tracker.call_count == 1
        
        # Add more usage
        tracker.add_usage(200, 75, 0.10)
        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 125
        assert abs(tracker.total_cost_usd - 0.15) < 1e-10  # Handle floating point precision
        assert tracker.call_count == 2
    
    def test_get_summary(self):
        """Test cost summary generation."""
        tracker = CostTracker()
        tracker.add_usage(100, 50, 0.05)
        tracker.cache_hits = 2
        tracker.failed_calls = 1
        
        summary = tracker.get_summary()
        
        assert summary["total_tokens"] == 150
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 50
        assert summary["total_cost_usd"] == 0.05
        assert summary["calls"] == 1
        assert summary["cache_hits"] == 2
        assert summary["failed_calls"] == 1
        assert summary["avg_cost_per_call"] == 0.05


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_default_retry_config(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.backoff_multiplier == 2.0
        assert config.retry_on_rate_limit is True
    
    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            fallback_to_simpler_model=False
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.fallback_to_simpler_model is False


class TestLLMResponse:
    """Test LLM response handling."""
    
    def test_valid_fix_detection(self):
        """Test valid fix detection logic."""
        # Valid fix
        valid_response = LLMResponse(
            content="def fixed_function(): return 42",
            confidence=0.8
        )
        assert valid_response.is_valid_fix() is True
        
        # Invalid - too short
        short_response = LLMResponse(content="ok")
        assert short_response.is_valid_fix() is False
        
        # Invalid - low confidence
        low_conf_response = LLMResponse(
            content="def fixed_function(): return 42",
            confidence=0.2
        )
        assert low_conf_response.is_valid_fix() is False
        
        # Invalid - empty
        empty_response = LLMResponse(content="")
        assert empty_response.is_valid_fix() is False


class TestBaseLLMAdapter:
    """Test base adapter functionality."""
    
    def test_token_counting(self):
        """Test token counting functionality."""
        adapter = MockAdapter()
        
        text = "def hello(): return 'world'"
        tokens = adapter.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_successful_api_call(self):
        """Test successful API call with retry logic."""
        adapter = MockAdapter()
        
        response = adapter.generate_fix("Test prompt")
        
        assert response.content is not None
        assert response.model_used == "mock"
        assert response.input_tokens > 0
        assert response.output_tokens > 0
    
    def test_retry_logic(self):
        """Test retry logic on failures.""" 
        # Adapter that fails twice then succeeds
        adapter = MockAdapter(fail_count=2)
        
        # Override _should_retry to always retry for test exceptions  
        original_should_retry = adapter._should_retry
        def always_retry(e):
            return "Simulated API failure" in str(e) or original_should_retry(e)
        adapter._should_retry = always_retry
        
        # Use unique prompt to avoid caching
        import time
        unique_prompt = f"Test prompt for retry {time.time()}"
        response = adapter.generate_fix(unique_prompt)
        
        # Should eventually succeed
        assert response.content is not None
        assert adapter._call_count == 3  # Failed twice, succeeded on third
    
    def test_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""
        # Adapter that always fails
        adapter = MockAdapter(fail_count=10)
        
        # Override _should_retry to always retry for test exceptions
        original_should_retry = adapter._should_retry
        def always_retry(e):
            return "Simulated API failure" in str(e) or original_should_retry(e)
        adapter._should_retry = always_retry
        
        # Use unique prompt to avoid caching
        import time
        unique_prompt = f"Test prompt for max retries {time.time()}"
        response = adapter.generate_fix(unique_prompt)
        
        # Should return empty response after retries
        assert response.content == ""
        assert adapter._call_count == adapter.retry_config.max_retries + 1
    
    def test_caching(self):
        """Test response caching."""
        adapter = MockAdapter()
        
        # First call
        response1 = adapter.generate_fix("Test prompt")
        call_count_after_first = adapter._call_count
        
        # Second call with same prompt - should be cached
        response2 = adapter.generate_fix("Test prompt")
        call_count_after_second = adapter._call_count
        
        # Should not make additional API calls
        assert call_count_after_second == call_count_after_first
        assert response2.cached is True


class TestOpenAIAdapter:
    """Test OpenAI adapter functionality."""
    
    @patch('tailchasing.llm.adapters.openai_adapter.OpenAI')
    def test_openai_adapter_initialization(self, mock_openai_class):
        """Test OpenAI adapter initialization."""
        adapter = OpenAIAdapter("gpt-4o-mini", api_key="test-key")
        
        assert adapter.model == "gpt-4o-mini"
        assert adapter.model_tier == ModelTier.STANDARD
    
    def test_invalid_model(self):
        """Test error on invalid model."""
        with pytest.raises(ValueError):
            OpenAIAdapter("invalid-model")
    
    def test_cost_estimation(self):
        """Test cost estimation for different models."""
        adapter = OpenAIAdapter("gpt-4o-mini")
        
        cost = adapter.estimate_cost(1000, 500)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Premium model should cost more
        premium_adapter = OpenAIAdapter("gpt-4o")
        premium_cost = premium_adapter.estimate_cost(1000, 500)
        assert premium_cost > cost
    
    @patch('tailchasing.llm.adapters.openai_adapter.OpenAI')
    def test_json_response_parsing(self, mock_openai_class):
        """Test JSON response parsing."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content": "test", "confidence": 0.9}'
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        
        mock_openai_class.return_value = mock_client
        
        adapter = OpenAIAdapter("gpt-4o-mini", api_key="test")
        adapter._client = mock_client
        
        response = adapter.generate_fix("test prompt requiring json")
        
        # Should parse JSON response
        assert response.content == "test"
        assert response.confidence == 0.9


class TestAnthropicAdapter:
    """Test Anthropic adapter functionality."""
    
    @patch('tailchasing.llm.adapters.anthropic_adapter.Anthropic')
    def test_anthropic_adapter_initialization(self, mock_anthropic_class):
        """Test Anthropic adapter initialization."""
        adapter = AnthropicAdapter("claude-3-haiku-20240307", api_key="test-key")
        
        assert adapter.model == "claude-3-haiku-20240307"
        assert adapter.model_tier == ModelTier.STANDARD
    
    def test_model_tiers(self):
        """Test model tier assignment."""
        haiku_adapter = AnthropicAdapter("claude-3-haiku-20240307")
        assert haiku_adapter.model_tier == ModelTier.STANDARD
        
        sonnet_adapter = AnthropicAdapter("claude-3-sonnet-20240229")
        assert sonnet_adapter.model_tier == ModelTier.PREMIUM


class TestOllamaAdapter:
    """Test Ollama adapter functionality."""
    
    @patch('tailchasing.llm.adapters.ollama_adapter.requests')
    def test_ollama_adapter_initialization(self, mock_requests):
        """Test Ollama adapter initialization."""
        # Mock successful availability check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "codellama:7b-code"}]}
        mock_requests.get.return_value = mock_response
        
        adapter = OllamaAdapter("codellama:7b-code")
        assert adapter.model == "codellama:7b-code"
        assert adapter.model_tier == ModelTier.LOCAL
    
    @patch('tailchasing.llm.adapters.ollama_adapter.requests')
    def test_unavailable_ollama(self, mock_requests):
        """Test handling of unavailable Ollama."""
        # Mock connection failure
        mock_requests.get.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError):
            OllamaAdapter("codellama:7b-code")
    
    def test_zero_cost(self):
        """Test that local models have zero cost."""
        with patch('tailchasing.llm.adapters.ollama_adapter.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": [{"name": "codellama:7b-code"}]}
            mock_requests.get.return_value = mock_response
            
            adapter = OllamaAdapter("codellama:7b-code")
            cost = adapter.estimate_cost(1000, 500)
            assert cost == 0.0


class TestTailChasingPrompts:
    """Test intelligent prompting system."""
    
    def test_prompt_generation(self, sample_issue, sample_source_code):
        """Test context-aware prompt generation."""
        prompts = TailChasingPrompts()
        
        context = PromptContext(
            current_issue=sample_issue,
            source_code=sample_source_code
        )
        
        prompt = prompts.generate_fix_prompt(context)
        
        assert "phantom_function" in prompt
        assert "TODO: implement later" in prompt
        assert "RESPONSE FORMAT" in prompt
        assert "confidence" in prompt
    
    def test_validation_prompt(self, sample_issue):
        """Test validation prompt generation."""
        prompts = TailChasingPrompts()
        
        original = "def func(): pass"
        fixed = "def func(): return 42"
        
        prompt = prompts.generate_validation_prompt(original, fixed, sample_issue)
        
        assert "VALIDATION CRITERIA" in prompt
        assert original in prompt
        assert fixed in prompt
        assert "json" in prompt.lower()
    
    def test_explanation_prompt(self, sample_issue):
        """Test explanation prompt generation."""
        prompts = TailChasingPrompts()
        
        prompt = prompts.generate_explanation_prompt(sample_issue)
        
        assert "phantom_function" in prompt
        assert "Root Cause Analysis" in prompt
        assert "Prevention Strategy" in prompt
    
    def test_few_shot_examples(self):
        """Test few-shot example loading."""
        prompts = TailChasingPrompts()
        
        # Should have examples for common patterns
        assert "phantom_function" in prompts.few_shot_examples
        assert "duplicate_function" in prompts.few_shot_examples
        
        phantom_examples = prompts.few_shot_examples["phantom_function"]
        assert len(phantom_examples) > 0
        assert "before" in phantom_examples[0]
        assert "after" in phantom_examples[0]
        assert "approach" in phantom_examples[0]


class TestLLMManager:
    """Test LLM manager functionality."""
    
    def test_manager_initialization(self):
        """Test LLM manager initialization."""
        adapters = {
            "mock-standard": MockAdapter(ModelTier.STANDARD),
            "mock-premium": MockAdapter(ModelTier.PREMIUM)
        }
        
        manager = LLMManager(adapters=adapters)
        
        assert len(manager.adapters) == 2
        assert "mock-standard" in manager.adapters
        assert "mock-premium" in manager.adapters
    
    def test_model_selection_by_complexity(self):
        """Test intelligent model selection based on task complexity."""
        adapters = {
            "local": MockAdapter(ModelTier.LOCAL),
            "standard": MockAdapter(ModelTier.STANDARD), 
            "premium": MockAdapter(ModelTier.PREMIUM)
        }
        
        manager = LLMManager(adapters=adapters)
        
        # Complex task should prefer premium
        complex_model = manager._select_model_for_task(TaskComplexity.COMPLEX)
        assert complex_model == "premium"
        
        # Moderate task should prefer standard
        moderate_model = manager._select_model_for_task(TaskComplexity.MODERATE)
        assert moderate_model == "standard"
        
        # Simple task should prefer local (cheapest)
        simple_model = manager._select_model_for_task(TaskComplexity.SIMPLE)
        assert simple_model == "local"
    
    def test_budget_constraints(self):
        """Test model selection with budget constraints."""
        adapters = {
            "expensive": MockAdapter(ModelTier.PREMIUM),
            "cheap": MockAdapter(ModelTier.LOCAL)
        }
        
        # Mock cost estimation
        adapters["expensive"].estimate_cost = Mock(return_value=2.0)
        adapters["cheap"].estimate_cost = Mock(return_value=0.0)
        
        manager = LLMManager(adapters=adapters)
        
        # With low budget, should select cheap model
        model = manager._select_model_for_task(TaskComplexity.COMPLEX, max_budget_usd=1.0)
        assert model == "cheap"
        
        # With high budget, should select expensive model
        model = manager._select_model_for_task(TaskComplexity.COMPLEX, max_budget_usd=5.0)
        assert model == "expensive"
    
    def test_fix_generation_with_context(self, sample_issue, sample_source_code):
        """Test fix generation with context."""
        adapters = {"mock": MockAdapter()}
        manager = LLMManager(adapters=adapters)
        
        response = manager.generate_fix(
            sample_issue,
            source_code=sample_source_code,
            complexity=TaskComplexity.SIMPLE
        )
        
        assert response is not None
        assert response.model_used == "mock"
    
    def test_caching_integration(self, sample_issue):
        """Test response caching in manager."""
        adapters = {"mock": MockAdapter()}
        manager = LLMManager(adapters=adapters, enable_caching=True)
        
        # First call
        response1 = manager.generate_fix(sample_issue)
        
        # Second identical call should be cached
        response2 = manager.generate_fix(sample_issue)
        
        # Mock adapter should only be called once
        assert adapters["mock"]._call_count == 1
        assert response2.cached is True
    
    def test_cost_tracking(self, sample_issue):
        """Test global cost tracking."""
        adapters = {"mock": MockAdapter()}
        manager = LLMManager(adapters=adapters)
        
        # Generate some responses
        manager.generate_fix(sample_issue)
        manager.generate_fix(sample_issue, complexity=TaskComplexity.COMPLEX)
        
        cost_summary = manager.get_cost_summary()
        
        assert "global" in cost_summary
        assert "by_adapter" in cost_summary
        assert cost_summary["global"]["calls"] > 0
    
    def test_fallback_logic(self, sample_issue):
        """Test fallback to simpler models on failure."""
        # Create adapters where premium fails but standard works
        premium_adapter = MockAdapter(ModelTier.PREMIUM, fail_count=10)  # Always fails
        standard_adapter = MockAdapter(ModelTier.STANDARD)  # Always works
        
        adapters = {
            "premium": premium_adapter,
            "standard": standard_adapter
        }
        
        manager = LLMManager(adapters=adapters)
        
        # Should fallback to standard model when premium fails
        response = manager.generate_fix(sample_issue, complexity=TaskComplexity.COMPLEX)
        
        # Should get a response despite premium model failing
        assert response.content != ""
        assert not response.content.startswith("All models failed")


class TestPromptTemplates:
    """Test prompt template functionality."""
    
    def test_template_retrieval(self):
        """Test getting predefined templates."""
        template = get_prompt_template("phantom_function_fix")
        assert template is not None
        assert "phantom" in template.template.lower()
    
    def test_invalid_template(self):
        """Test error on invalid template name."""
        with pytest.raises(ValueError):
            get_prompt_template("nonexistent_template")
    
    def test_template_formatting(self):
        """Test template variable substitution."""
        template = get_prompt_template("phantom_function_fix")
        
        formatted = template.format(
            code="def test(): pass",
            context="Test context"
        )
        
        assert "def test(): pass" in formatted
        assert "Test context" in formatted


class TestIntegration:
    """Integration tests for the complete LLM system."""
    
    def test_end_to_end_fix_generation(self, sample_issue, sample_source_code):
        """Test complete fix generation workflow."""
        # Create mock adapters
        adapters = {"mock": MockAdapter()}
        manager = LLMManager(adapters=adapters, enable_caching=True)
        
        # Generate fix
        response = manager.generate_fix(
            sample_issue,
            source_code=sample_source_code,
            complexity=TaskComplexity.MODERATE
        )
        
        # Validate fix
        validation_response = manager.validate_fix(
            sample_source_code,
            response.content,
            sample_issue
        )
        
        # Get explanation
        explanation_response = manager.explain_issue(sample_issue)
        
        # All should succeed
        assert response.content != ""
        assert validation_response.content != ""
        assert explanation_response.content != ""
        
        # Check cost tracking
        cost_summary = manager.get_cost_summary()
        assert cost_summary["global"]["calls"] >= 3
    
    def test_batch_processing(self):
        """Test batch fix generation."""
        adapters = {"mock": MockAdapter()}
        manager = LLMManager(adapters=adapters)
        
        issues = [
            Issue(kind="phantom_function", message="Test 1", severity=2),
            Issue(kind="duplicate_function", message="Test 2", severity=3)
        ]
        
        responses = manager.batch_generate_fixes(issues, total_budget_usd=1.0)
        
        assert len(responses) == 2
        assert all(r.content != "" for r in responses)
    
    def test_successful_pattern_tracking(self, sample_issue):
        """Test tracking of successful fix patterns."""
        adapters = {"mock": MockAdapter()}
        manager = LLMManager(adapters=adapters)
        
        # Generate some fixes
        manager.generate_fix(sample_issue)
        
        # Should track successful patterns
        patterns = manager.get_successful_patterns("phantom_function")
        assert isinstance(patterns, list)


@pytest.mark.integration
def test_context_aware_prompt_creation():
    """Integration test for context-aware prompt creation."""
    issue = Issue(
        kind="phantom_function",
        message="Empty function needs implementation",
        severity=3,
        file="test.py",
        line=5
    )
    
    source_code = "def empty_func(): pass"
    
    prompt = create_context_aware_prompt(
        issue,
        source_code=source_code,
        previous_attempts=[{
            "approach": "simple return",
            "result": "failed", 
            "failure_reason": "insufficient implementation"
        }]
    )
    
    # Should include all context
    assert "phantom_function" in prompt
    assert "def empty_func(): pass" in prompt
    assert "PREVIOUS ATTEMPTS" in prompt
    assert "simple return" in prompt
    assert "RESPONSE FORMAT" in prompt