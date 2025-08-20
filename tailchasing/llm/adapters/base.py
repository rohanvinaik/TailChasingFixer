"""
Base LLM adapter protocol and common components.

Defines the interface for LLM providers with token counting, cost tracking,
and intelligent prompting capabilities for tail-chasing scenarios.
"""

import time
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum
import json

from ...utils.logging_setup import get_logger
from ...core.issues import Issue


class ModelTier(Enum):
    """Model capability tiers for intelligent fallbacks."""
    PREMIUM = "premium"      # GPT-4o, Claude-3-Sonnet
    STANDARD = "standard"    # GPT-4o-mini, Claude-3-Haiku  
    LOCAL = "local"         # Ollama models


@dataclass
class CostTracker:
    """Tracks token usage and costs across LLM calls."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    cache_hits: int = 0
    failed_calls: int = 0
    
    def add_usage(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Add usage statistics for a single call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.call_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics."""
        return {
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "calls": self.call_count,
            "cache_hits": self.cache_hits,
            "failed_calls": self.failed_calls,
            "avg_cost_per_call": round(self.total_cost_usd / max(1, self.call_count), 4)
        }


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True
    fallback_to_simpler_model: bool = True


@dataclass
class LLMResponse:
    """Response from an LLM call with metadata."""
    content: str
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    model_used: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    retry_count: int = 0
    duration_seconds: float = 0.0
    
    def is_valid_fix(self) -> bool:
        """Check if response appears to be a valid fix."""
        return (
            bool(self.content.strip()) and
            len(self.content) > 20 and  # Not too short
            (self.confidence is None or self.confidence > 0.3)  # Reasonable confidence
        )


class PromptTemplate:
    """Template for constructing prompts with variable substitution."""
    
    def __init__(self, template: str, variables: Optional[List[str]] = None):
        self.template = template
        self.variables = variables or []
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        return self.template.format(**kwargs)
    
    def get_hash(self, **kwargs) -> str:
        """Get hash of formatted prompt for caching."""
        formatted = self.format(**kwargs)
        return hashlib.md5(formatted.encode()).hexdigest()


class LLMAdapter(Protocol):
    """Protocol defining the interface for LLM adapters."""
    
    @property
    def model_tier(self) -> ModelTier:
        """Get the tier/capability level of this model."""
        ...
    
    @property 
    def cost_tracker(self) -> CostTracker:
        """Get cost tracking information."""
        ...
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model-specific tokenizer."""
        ...
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts in USD."""
        ...
    
    def generate_fix(self, 
                    prompt: str,
                    context: Optional[Dict[str, Any]] = None,
                    max_tokens: int = 2000,
                    temperature: float = 0.1) -> LLMResponse:
        """Generate a code fix based on the prompt."""
        ...
    
    def generate_explanation(self,
                           issue: Issue,
                           context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate an explanation for why an issue occurred."""
        ...
    
    def validate_fix(self,
                    original_code: str,
                    fixed_code: str,
                    issue_description: str) -> LLMResponse:
        """Validate that a fix actually addresses the issue."""
        ...


class BaseLLMAdapter(ABC):
    """Base implementation of LLMAdapter with common functionality."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 retry_config: Optional[RetryConfig] = None):
        self.api_key = api_key
        self.retry_config = retry_config or RetryConfig()
        self.cost_tracker = CostTracker()
        self.logger = get_logger(__name__)
        self._fix_cache: Dict[str, LLMResponse] = {}
        
    @property
    @abstractmethod
    def model_tier(self) -> ModelTier:
        """Get the tier/capability level of this model."""
        pass
    
    @abstractmethod
    def _make_api_call(self, prompt: str, **kwargs) -> LLMResponse:
        """Make actual API call to the LLM provider."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (approximate for non-OpenAI models)."""
        try:
            import tiktoken
            # Use cl100k_base tokenizer as reasonable approximation
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough approximation when tiktoken not available
            return int(len(text.split()) * 1.3)
        except Exception:
            # Fallback: rough approximation
            return int(len(text.split()) * 1.3)
    
    def generate_fix(self, 
                    prompt: str,
                    context: Optional[Dict[str, Any]] = None,
                    max_tokens: int = 2000,
                    temperature: float = 0.1) -> LLMResponse:
        """Generate a code fix with caching and retry logic."""
        # Check cache first
        cache_key = hashlib.md5(f"{prompt}{context}".encode()).hexdigest()
        if cache_key in self._fix_cache:
            cached_response = self._fix_cache[cache_key]
            cached_response.cached = True
            self.cost_tracker.cache_hits += 1
            return cached_response
        
        # Make call with retry logic
        response = self._call_with_retry(
            prompt, 
            context=context,
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        # Cache successful responses
        if response.is_valid_fix():
            self._fix_cache[cache_key] = response
        
        return response
    
    def generate_explanation(self,
                           issue: Issue,
                           context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate explanation for an issue."""
        explanation_prompt = self._build_explanation_prompt(issue, context)
        return self.generate_fix(
            explanation_prompt,
            context=context,
            max_tokens=1000,
            temperature=0.3
        )
    
    def validate_fix(self,
                    original_code: str,
                    fixed_code: str, 
                    issue_description: str) -> LLMResponse:
        """Validate that a fix addresses the issue."""
        validation_prompt = self._build_validation_prompt(
            original_code, fixed_code, issue_description
        )
        return self.generate_fix(
            validation_prompt,
            max_tokens=500,
            temperature=0.0
        )
    
    def _call_with_retry(self, prompt: str, **kwargs) -> LLMResponse:
        """Make API call with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                start_time = time.time()
                response = self._make_api_call(prompt, **kwargs)
                response.duration_seconds = time.time() - start_time
                response.retry_count = attempt
                
                # Track successful call
                self.cost_tracker.add_usage(
                    response.input_tokens,
                    response.output_tokens,
                    response.cost_usd
                )
                
                return response
                
            except Exception as e:
                last_exception = e
                self.cost_tracker.failed_calls += 1
                
                # Check if we should retry
                if attempt >= self.retry_config.max_retries:
                    break
                
                if not self._should_retry(e):
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.retry_config.base_delay * (
                        self.retry_config.backoff_multiplier ** attempt
                    ),
                    self.retry_config.max_delay
                )
                
                if self.retry_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                self.logger.warning(
                    f"LLM call failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
        
        # All retries failed
        self.logger.error(f"LLM call failed after {self.retry_config.max_retries} retries: {last_exception}")
        return LLMResponse(
            content="",
            model_used=self.__class__.__name__,
            retry_count=self.retry_config.max_retries
        )
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        error_str = str(exception).lower()
        
        # Rate limiting
        if self.retry_config.retry_on_rate_limit:
            if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429']):
                return True
        
        # Server errors
        if self.retry_config.retry_on_server_error:
            if any(keyword in error_str for keyword in ['500', '502', '503', '504', 'server error']):
                return True
        
        return False
    
    def _build_explanation_prompt(self, issue: Issue, context: Optional[Dict[str, Any]]) -> str:
        """Build prompt for explaining an issue."""
        context = context or {}
        
        prompt_parts = [
            "You are an expert Python developer analyzing tail-chasing patterns in code.",
            f"\nISSUE: {issue.kind}",
            f"DESCRIPTION: {issue.message}",
            f"SEVERITY: {issue.severity}",
        ]
        
        if issue.file and issue.line:
            prompt_parts.append(f"LOCATION: {issue.file}:{issue.line}")
        
        if issue.evidence:
            prompt_parts.append(f"EVIDENCE: {json.dumps(issue.evidence, indent=2)}")
        
        if context.get('source_code'):
            prompt_parts.append(f"\nSOURCE CODE:\n```python\n{context['source_code']}\n```")
        
        prompt_parts.extend([
            "\nPlease explain:",
            "1. WHY this tail-chasing pattern occurred",
            "2. What likely caused the developer/AI to create this pattern", 
            "3. How to prevent this pattern in the future",
            "\nProvide a clear, concise explanation suitable for a developer."
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_validation_prompt(self, original: str, fixed: str, issue: str) -> str:
        """Build prompt for validating a fix."""
        return f"""You are validating a code fix for tail-chasing patterns.

ORIGINAL ISSUE: {issue}

ORIGINAL CODE:
```python
{original}
```

PROPOSED FIX:
```python
{fixed}
```

Please validate this fix and respond with:
1. VALID: true/false
2. CONFIDENCE: 0.0-1.0 confidence score  
3. EXPLANATION: Brief explanation of your assessment

Focus on:
- Does the fix actually address the stated issue?
- Is the fix syntactically correct?
- Are there any new issues introduced?
- Is this an AST-safe transformation?

Respond in JSON format:
{{"valid": boolean, "confidence": float, "explanation": "string"}}"""


# Common prompt templates for tail-chasing scenarios
TAIL_CHASING_TEMPLATES = {
    "phantom_function_fix": PromptTemplate("""
You are fixing a phantom function that has no real implementation.

PHANTOM FUNCTION:
{code}

CONTEXT: {context}

Requirements:
1. Either provide a COMPLETE implementation or remove the function entirely
2. Do NOT create stubs with pass/NotImplementedError
3. Ensure the implementation matches the function signature and docstring
4. Use existing patterns from the codebase when possible

RESPOND WITH:
- ACTION: "implement" or "remove"
- CODE: The complete fixed code  
- CONFIDENCE: Your confidence (0.0-1.0) in this fix
- RATIONALE: Brief explanation of your approach
"""),
    
    "duplicate_consolidation": PromptTemplate("""
You are consolidating duplicate functions with identical behavior.

DUPLICATE FUNCTIONS:
{duplicates}

CONTEXT: {context}

Requirements:
1. Choose the BEST implementation (most complete, best named, well-tested)
2. Update all callers to use the chosen implementation
3. Remove the redundant implementations
4. Ensure no functionality is lost

RESPOND WITH:
- CANONICAL: Name of the function to keep
- REMOVALS: List of functions to remove
- UPDATES: List of files/lines to update with new calls
- CONFIDENCE: Your confidence (0.0-1.0) in this consolidation
- RATIONALE: Brief explanation of your choice
"""),
    
    "import_fix": PromptTemplate("""
You are fixing import issues in Python code.

IMPORT ISSUE: {issue_type}
PROBLEMATIC CODE:
{code}

AVAILABLE MODULES: {available_modules}
CONTEXT: {context}

Requirements:
1. Fix circular imports by restructuring if needed
2. Remove unused imports
3. Use correct import paths based on available modules
4. Maintain code functionality

RESPOND WITH:
- FIXED_IMPORTS: The corrected import statements
- MOVED_CODE: Any code that needs to be moved to resolve circularity  
- CONFIDENCE: Your confidence (0.0-1.0) in this fix
- RATIONALE: Brief explanation of changes made
""")
}


def get_prompt_template(template_name: str) -> PromptTemplate:
    """Get a predefined prompt template for tail-chasing scenarios."""
    if template_name not in TAIL_CHASING_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    return TAIL_CHASING_TEMPLATES[template_name]