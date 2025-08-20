"""
LLM Manager with intelligent routing, retry logic, and fallbacks.

Manages multiple LLM adapters with smart model selection, caching,
and graceful degradation for reliable code generation.
"""

import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .adapters.base import LLMAdapter, LLMResponse, ModelTier, CostTracker
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.ollama_adapter import OllamaAdapter
from .prompts import TailChasingPrompts, PromptContext
from ..core.issues import Issue
from ..utils.logging_setup import get_logger


class TaskComplexity(Enum):
    """Complexity levels for task routing."""
    SIMPLE = "simple"      # Simple fixes, explanations
    MODERATE = "moderate"  # Multi-step fixes, validation
    COMPLEX = "complex"    # Complex refactoring, analysis


@dataclass
class FixAttempt:
    """Record of a fix attempt for learning and retry logic."""
    timestamp: float
    model_used: str
    approach: str
    result: str
    success: bool
    failure_reason: Optional[str] = None
    confidence: Optional[float] = None
    cost_usd: float = 0.0


@dataclass
class CachedResponse:
    """Cached LLM response with metadata."""
    response: LLMResponse
    timestamp: float
    hit_count: int = 0
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if cached response is expired."""
        age_hours = (time.time() - self.timestamp) / 3600
        return age_hours > max_age_hours


class LLMManager:
    """
    Manages multiple LLM adapters with intelligent routing and fallbacks.
    
    Features:
    - Smart model selection based on task complexity and cost
    - Response caching with TTL
    - Retry logic with exponential backoff 
    - Fallback to simpler/cheaper models on failure
    - Cost tracking and budgeting
    - Pattern caching for successful fixes
    """
    
    def __init__(self,
                 adapters: Optional[Dict[str, LLMAdapter]] = None,
                 default_budget_usd: float = 10.0,
                 enable_caching: bool = True,
                 cache_ttl_hours: int = 24):
        
        self.adapters = adapters or {}
        self.default_budget_usd = default_budget_usd
        self.enable_caching = enable_caching
        self.cache_ttl_hours = cache_ttl_hours
        
        self.logger = get_logger(__name__)
        self.prompts = TailChasingPrompts()
        
        # Tracking and caching
        self.global_cost_tracker = CostTracker()
        self._response_cache: Dict[str, CachedResponse] = {}
        self._fix_attempts: List[FixAttempt] = []
        self._successful_patterns: Dict[str, List[str]] = {}  # issue_type -> successful approaches
        
        # Auto-configure adapters if none provided
        if not self.adapters:
            self._auto_configure_adapters()
    
    def _auto_configure_adapters(self) -> None:
        """Auto-configure available adapters based on environment."""
        import os
        
        # Try OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.adapters["gpt-4o-mini"] = OpenAIAdapter("gpt-4o-mini")
                self.adapters["gpt-4o"] = OpenAIAdapter("gpt-4o")
                self.logger.info("Configured OpenAI adapters")
            except Exception as e:
                self.logger.warning(f"Failed to configure OpenAI: {e}")
        
        # Try Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.adapters["claude-3-haiku"] = AnthropicAdapter("claude-3-haiku-20240307")
                self.adapters["claude-3-sonnet"] = AnthropicAdapter("claude-3-sonnet-20240229")
                self.logger.info("Configured Anthropic adapters")
            except Exception as e:
                self.logger.warning(f"Failed to configure Anthropic: {e}")
        
        # Try Ollama (local)
        try:
            ollama_adapter = OllamaAdapter("codellama:7b-code")
            # Test if Ollama is available
            ollama_adapter._check_ollama_availability()
            self.adapters["codellama"] = ollama_adapter
            self.logger.info("Configured Ollama adapter")
        except Exception as e:
            self.logger.debug(f"Ollama not available: {e}")
        
        if not self.adapters:
            self.logger.warning("No LLM adapters configured. Set API keys or install Ollama.")
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        models = {}
        for name, adapter in self.adapters.items():
            models[name] = {
                "tier": adapter.model_tier.value,
                "cost_tracker": adapter.cost_tracker.get_summary(),
                "available": True
            }
        return models
    
    def generate_fix(self,
                    issue: Issue,
                    source_code: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None,
                    complexity: TaskComplexity = TaskComplexity.MODERATE,
                    max_budget_usd: Optional[float] = None) -> LLMResponse:
        """
        Generate a fix for the given issue with intelligent model selection.
        
        Args:
            issue: The tail-chasing issue to fix
            source_code: The problematic source code
            context: Additional context information
            complexity: Task complexity for model selection
            max_budget_usd: Maximum budget for this request
            
        Returns:
            LLMResponse with the generated fix
        """
        # Build context-aware prompt
        previous_attempts = [
            {
                "approach": attempt.approach,
                "result": attempt.result,
                "failure_reason": attempt.failure_reason
            }
            for attempt in self._get_previous_attempts(issue.kind)
        ]
        
        prompt_context = PromptContext(
            current_issue=issue,
            source_code=source_code,
            previous_attempts=previous_attempts,
            codebase_patterns=context
        )
        
        prompt = self.prompts.generate_fix_prompt(prompt_context)
        
        # Check cache first
        if self.enable_caching:
            cached = self._get_cached_response(prompt, issue.kind)
            if cached:
                return cached
        
        # Select appropriate model
        selected_model = self._select_model_for_task(complexity, max_budget_usd)
        if not selected_model:
            return LLMResponse(
                content="No suitable model available within budget",
                confidence=0.0
            )
        
        # Generate with retry and fallback logic
        response = self._generate_with_fallbacks(
            prompt, issue, selected_model, complexity, max_budget_usd
        )
        
        # Cache successful responses
        if self.enable_caching and response.is_valid_fix():
            self._cache_response(prompt, issue.kind, response)
        
        # Record attempt for learning
        self._record_attempt(response, issue, selected_model)
        
        # Update global cost tracking
        self.global_cost_tracker.add_usage(
            response.input_tokens,
            response.output_tokens, 
            response.cost_usd
        )
        
        return response
    
    def validate_fix(self,
                    original_code: str,
                    proposed_fix: str,
                    issue: Issue,
                    complexity: TaskComplexity = TaskComplexity.SIMPLE) -> LLMResponse:
        """Validate a proposed fix using appropriate model."""
        
        validation_prompt = self.prompts.generate_validation_prompt(
            original_code, proposed_fix, issue
        )
        
        # Use simpler model for validation
        model_name = self._select_model_for_task(complexity, max_budget_usd=1.0)
        if not model_name:
            return LLMResponse(content="No model available for validation", confidence=0.0)
        
        adapter = self.adapters[model_name]
        return adapter.validate_fix(original_code, proposed_fix, issue.message)
    
    def explain_issue(self,
                     issue: Issue,
                     context: Optional[PromptContext] = None) -> LLMResponse:
        """Generate explanation for why an issue occurred."""
        
        explanation_prompt = self.prompts.generate_explanation_prompt(issue, context)
        
        # Use premium model for analysis
        model_name = self._select_model_for_task(TaskComplexity.MODERATE)
        if not model_name:
            return LLMResponse(content="No model available for explanation", confidence=0.0)
        
        adapter = self.adapters[model_name]
        return adapter.generate_explanation(issue, context.__dict__ if context else None)
    
    def batch_generate_fixes(self,
                           issues: List[Issue],
                           source_codes: Optional[List[str]] = None,
                           max_concurrent: int = 3,
                           total_budget_usd: Optional[float] = None) -> List[LLMResponse]:
        """Generate fixes for multiple issues efficiently."""
        
        if total_budget_usd is None:
            total_budget_usd = self.default_budget_usd
        
        budget_per_issue = total_budget_usd / len(issues)
        responses = []
        
        for i, issue in enumerate(issues):
            source_code = source_codes[i] if source_codes and i < len(source_codes) else None
            
            # Adjust complexity based on issue type
            complexity = self._infer_complexity(issue)
            
            try:
                response = self.generate_fix(
                    issue, 
                    source_code,
                    complexity=complexity,
                    max_budget_usd=budget_per_issue
                )
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Failed to generate fix for issue {i}: {e}")
                responses.append(LLMResponse(
                    content="",
                    confidence=0.0,
                    model_used="error"
                ))
            
            # Stop if we've exceeded budget
            if self.global_cost_tracker.total_cost_usd > total_budget_usd:
                self.logger.warning("Budget exceeded, stopping batch generation")
                break
        
        return responses
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary across all adapters."""
        summary = {
            "global": self.global_cost_tracker.get_summary(),
            "by_adapter": {}
        }
        
        for name, adapter in self.adapters.items():
            summary["by_adapter"][name] = adapter.cost_tracker.get_summary()
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()
        self.logger.info("Response cache cleared")
    
    def get_successful_patterns(self, issue_type: str) -> List[str]:
        """Get successful fix patterns for an issue type."""
        return self._successful_patterns.get(issue_type, [])
    
    def _select_model_for_task(self,
                              complexity: TaskComplexity,
                              max_budget_usd: Optional[float] = None) -> Optional[str]:
        """Select the best model for a task based on complexity and budget."""
        
        if not self.adapters:
            return None
        
        # Filter by budget if specified
        available_models = []
        for name, adapter in self.adapters.items():
            if max_budget_usd is None:
                available_models.append((name, adapter))
            else:
                # Estimate if this model fits budget (rough estimation)
                estimated_cost = adapter.estimate_cost(1000, 500)  # Typical request
                if estimated_cost <= max_budget_usd:
                    available_models.append((name, adapter))
        
        if not available_models:
            return None
        
        # Select based on complexity and tier
        if complexity == TaskComplexity.COMPLEX:
            # Prefer premium models
            premium_models = [(n, a) for n, a in available_models if a.model_tier == ModelTier.PREMIUM]
            if premium_models:
                return premium_models[0][0]  # First premium model
        
        elif complexity == TaskComplexity.MODERATE:
            # Prefer standard or premium models
            suitable_models = [(n, a) for n, a in available_models 
                             if a.model_tier in [ModelTier.STANDARD, ModelTier.PREMIUM]]
            if suitable_models:
                # Choose standard first for cost efficiency
                standard_models = [(n, a) for n, a in suitable_models if a.model_tier == ModelTier.STANDARD]
                if standard_models:
                    return standard_models[0][0]
                return suitable_models[0][0]
        
        # For simple tasks or fallback, use any available model (prefer cheaper)
        local_models = [(n, a) for n, a in available_models if a.model_tier == ModelTier.LOCAL]
        if local_models:
            return local_models[0][0]
        
        # Return first available model
        return available_models[0][0]
    
    def _generate_with_fallbacks(self,
                                prompt: str,
                                issue: Issue,
                                primary_model: str,
                                complexity: TaskComplexity,
                                max_budget_usd: Optional[float]) -> LLMResponse:
        """Generate response with fallback logic."""
        
        models_to_try = [primary_model]
        
        # Add fallback models
        if complexity != TaskComplexity.SIMPLE:
            # Add simpler models as fallbacks
            fallback_model = self._select_model_for_task(TaskComplexity.SIMPLE, max_budget_usd)
            if fallback_model and fallback_model != primary_model:
                models_to_try.append(fallback_model)
        
        last_error = None
        
        for model_name in models_to_try:
            try:
                adapter = self.adapters[model_name]
                response = adapter.generate_fix(prompt)
                
                if response.is_valid_fix():
                    return response
                else:
                    self.logger.warning(f"Model {model_name} produced invalid fix")
                    last_error = "Invalid fix generated"
                    
            except Exception as e:
                self.logger.error(f"Model {model_name} failed: {e}")
                last_error = str(e)
                continue
        
        # All models failed
        return LLMResponse(
            content=f"All models failed. Last error: {last_error}",
            confidence=0.0,
            model_used="fallback_failed"
        )
    
    def _get_cached_response(self, prompt: str, issue_type: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired."""
        cache_key = self._get_cache_key(prompt, issue_type)
        
        if cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            
            if not cached.is_expired(self.cache_ttl_hours):
                cached.hit_count += 1
                cached_response = cached.response
                cached_response.cached = True
                return cached_response
            else:
                # Remove expired cache entry
                del self._response_cache[cache_key]
        
        return None
    
    def _cache_response(self, prompt: str, issue_type: str, response: LLMResponse) -> None:
        """Cache a successful response."""
        cache_key = self._get_cache_key(prompt, issue_type)
        self._response_cache[cache_key] = CachedResponse(
            response=response,
            timestamp=time.time()
        )
    
    def _get_cache_key(self, prompt: str, issue_type: str) -> str:
        """Generate cache key for prompt and issue type."""
        combined = f"{issue_type}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _record_attempt(self, response: LLMResponse, issue: Issue, model_name: str) -> None:
        """Record fix attempt for learning."""
        attempt = FixAttempt(
            timestamp=time.time(),
            model_used=model_name,
            approach=response.rationale or "Unknown",
            result=response.content[:200],  # Truncate for storage
            success=response.is_valid_fix(),
            failure_reason=None if response.is_valid_fix() else "Invalid fix",
            confidence=response.confidence,
            cost_usd=response.cost_usd
        )
        
        self._fix_attempts.append(attempt)
        
        # Track successful patterns
        if attempt.success and response.rationale:
            issue_type = issue.kind
            if issue_type not in self._successful_patterns:
                self._successful_patterns[issue_type] = []
            
            # Keep only unique approaches
            if response.rationale not in self._successful_patterns[issue_type]:
                self._successful_patterns[issue_type].append(response.rationale)
                
            # Limit to most recent successful patterns
            self._successful_patterns[issue_type] = self._successful_patterns[issue_type][-10:]
    
    def _get_previous_attempts(self, issue_type: str) -> List[FixAttempt]:
        """Get previous attempts for this issue type."""
        return [
            attempt for attempt in self._fix_attempts[-20:]  # Last 20 attempts
            if not attempt.success  # Only failed attempts for learning
        ]
    
    def _infer_complexity(self, issue: Issue) -> TaskComplexity:
        """Infer task complexity from issue characteristics."""
        complex_patterns = {
            "hallucination_cascade",
            "context_window_thrashing", 
            "prototype_fragmentation"
        }
        
        moderate_patterns = {
            "semantic_duplicate_function",
            "circular_import", 
            "missing_symbol"
        }
        
        if issue.kind in complex_patterns:
            return TaskComplexity.COMPLEX
        elif issue.kind in moderate_patterns:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE


# Convenience function for easy setup
def create_llm_manager(budget_usd: float = 10.0,
                      enable_caching: bool = True) -> LLMManager:
    """Create LLM manager with automatic adapter configuration."""
    return LLMManager(
        default_budget_usd=budget_usd,
        enable_caching=enable_caching
    )