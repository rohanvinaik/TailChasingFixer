"""
Anthropic adapter for Claude-3-Haiku and Claude-3-Sonnet models.

Provides integration with Anthropic's API for code generation and fix suggestions
with proper cost tracking and Claude-specific optimizations.
"""

import json
import os
from typing import Dict, Optional, Any, Union, List

from .base import BaseLLMAdapter, LLMResponse, ModelTier, RetryConfig


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude models."""
    
    # Model configurations
    MODELS = {
        "claude-3-haiku-20240307": {
            "tier": ModelTier.STANDARD,
            "input_cost_per_1k": 0.00025,   # $0.25 per 1M input tokens
            "output_cost_per_1k": 0.00125,  # $1.25 per 1M output tokens
            "max_tokens": 200000,
            "display_name": "claude-3-haiku"
        },
        "claude-3-sonnet-20240229": {
            "tier": ModelTier.PREMIUM,
            "input_cost_per_1k": 0.003,     # $3 per 1M input tokens
            "output_cost_per_1k": 0.015,    # $15 per 1M output tokens 
            "max_tokens": 200000,
            "display_name": "claude-3-sonnet"
        }
    }
    
    def __init__(self,
                 model: str = "claude-3-haiku-20240307", 
                 api_key: Optional[str] = None,
                 retry_config: Optional[RetryConfig] = None):
        
        if model not in self.MODELS:
            raise ValueError(f"Unsupported model: {model}. Available: {list(self.MODELS.keys())}")
        
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), retry_config)
        self.model = model
        self.model_config = self.MODELS[model]
        
        # Initialize Anthropic client (lazy import)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client
    
    @property
    def model_tier(self) -> ModelTier:
        return self.model_config["tier"]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Claude's tokenizer (fallback to approximation)."""
        try:
            # Use Anthropic's token counting if available
            response = self.client.count_tokens(text)
            return response.token_count
        except Exception:
            # Fallback to tiktoken approximation
            return super().count_tokens(text)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for given token counts."""
        input_cost = (input_tokens / 1000) * self.model_config["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * self.model_config["output_cost_per_1k"]
        return input_cost + output_cost
    
    def _make_api_call(self, prompt: str, **kwargs) -> LLMResponse:
        """Make API call to Anthropic."""
        max_tokens = kwargs.get("max_tokens", 2000)
        temperature = kwargs.get("temperature", 0.1)
        
        # Build Claude-specific prompt format
        full_prompt = f"{self._get_system_prompt()}\n\nHuman: {prompt}\n\nAssistant:"
        
        # Count input tokens
        input_tokens = self.count_tokens(full_prompt)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._get_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            output_tokens = response.usage.output_tokens
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            # Parse structured response if requested
            parsed_content = self._parse_response(content, prompt)
            
            return LLMResponse(
                content=parsed_content["content"] if isinstance(parsed_content, dict) else parsed_content,
                confidence=parsed_content.get("confidence") if isinstance(parsed_content, dict) else None,
                rationale=parsed_content.get("rationale") if isinstance(parsed_content, dict) else None,
                model_used=self.model_config["display_name"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt optimized for Claude."""
        return """You are an expert Python developer specialized in detecting and fixing tail-chasing patterns in code. You excel at providing complete, working solutions that eliminate problematic patterns while preserving functionality.

Core principles for tail-chasing fixes:
1. Always provide COMPLETE implementations - never stubs or placeholders
2. Maintain AST-safe transformations - all code must be syntactically correct
3. Consolidate duplicate functionality by choosing the best implementation
4. Address root causes rather than applying superficial fixes
5. Preserve all existing functionality while eliminating the problematic pattern

Response format guidelines:
- For structured requests, respond in JSON with "content", "confidence" (0.0-1.0), and "rationale" fields
- For code fixes, provide complete, runnable code
- Include brief but clear explanations of your approach
- Rate your confidence honestly based on available context

You are particularly skilled at:
- Identifying and removing phantom/stub functions
- Consolidating semantically duplicate code
- Resolving circular import dependencies  
- Fixing missing symbol references
- Preventing rename cascades"""
    
    def _parse_response(self, content: str, original_prompt: str) -> Union[str, Dict[str, Any]]:
        """Parse Claude's response, handling structured formats."""
        if not self._requires_structured_response(original_prompt):
            return content
        
        try:
            # Try to parse as JSON first
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            # Claude often uses more natural structured responses
            # Try to extract structured information
            import re
            
            result = {"content": content}
            
            # Look for confidence indicators
            confidence_patterns = [
                r"confidence[:\s]+([\d.]+)",
                r"confidence[:\s]+(\d+)%",
                r"i'm ([\d.]+) confident",
                r"about ([\d.]+) confident"
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, content.lower())
                if match:
                    conf_val = float(match.group(1))
                    if conf_val > 1.0:  # Percentage format
                        conf_val /= 100.0
                    result["confidence"] = conf_val
                    break
            
            # Look for rationale/explanation
            rationale_patterns = [
                r"rationale[:\s]+([^\.]+\.)",
                r"explanation[:\s]+([^\.]+\.)",
                r"because[:\s]+([^\.]+\.)",
                r"approach[:\s]+([^\.]+\.)"
            ]
            
            for pattern in rationale_patterns:
                match = re.search(pattern, content.lower())
                if match:
                    result["rationale"] = match.group(1).strip()
                    break
            
            return result
    
    def _requires_structured_response(self, prompt: str) -> bool:
        """Check if prompt expects structured response from Claude."""
        structured_indicators = [
            "json format",
            "confidence",
            "rationale", 
            "respond with:",
            "provide:",
            "include:"
        ]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in structured_indicators)
    
    def generate_fix_with_reasoning(self,
                                  issue_code: str,
                                  issue_type: str,
                                  context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate fix with Claude's reasoning capabilities."""
        
        prompt = f"""I need you to fix a tail-chasing pattern in Python code.

ISSUE TYPE: {issue_type}
PROBLEMATIC CODE:
```python
{issue_code}
```

ADDITIONAL CONTEXT:
{json.dumps(context or {}, indent=2)}

Please analyze this step-by-step:

1. First, identify exactly what makes this a tail-chasing pattern
2. Determine the root cause that led to this pattern
3. Choose the best approach to eliminate the pattern completely
4. Provide the complete fixed code

Respond with:
- **Analysis**: What tail-chasing pattern you identified and why
- **Root Cause**: What likely caused this pattern to emerge  
- **Solution**: Your approach to fixing it
- **Code**: Complete, working fixed code
- **Confidence**: Your confidence level (0.0-1.0) in this fix

Focus on providing a complete solution that eliminates the pattern while preserving all intended functionality."""

        return self.generate_fix(prompt, context, temperature=0.0)
    
    def explain_pattern_prevention(self,
                                 issue_type: str,
                                 examples: Optional[List[str]] = None) -> LLMResponse:
        """Generate explanation of how to prevent specific tail-chasing patterns."""
        
        examples_text = ""
        if examples:
            examples_text = "EXAMPLES OF THIS PATTERN:\n" + "\n\n".join(
                f"Example {i+1}:\n```python\n{ex}\n```" 
                for i, ex in enumerate(examples[:3])
            )
        
        prompt = f"""You are teaching developers how to prevent {issue_type} tail-chasing patterns.

{examples_text}

Please provide a comprehensive guide on preventing this pattern:

1. **Pattern Recognition**: How to identify when this pattern is emerging
2. **Root Causes**: What typically leads developers/AI to create this pattern
3. **Prevention Strategies**: Specific techniques to avoid the pattern
4. **Best Practices**: General principles that help prevent this and similar patterns
5. **Early Warning Signs**: Code smells that indicate this pattern may be developing

Make this practical and actionable for both human developers and AI code generation systems."""

        return self.generate_fix(prompt, temperature=0.2)
    
    def validate_fix_thoroughly(self,
                              original_code: str,
                              fixed_code: str,
                              issue_description: str,
                              test_cases: Optional[List[str]] = None) -> LLMResponse:
        """Thorough validation of fixes using Claude's analytical capabilities."""
        
        test_cases_text = ""
        if test_cases:
            test_cases_text = "\nTEST CASES TO VERIFY:\n" + "\n".join(
                f"- {test}" for test in test_cases
            )
        
        prompt = f"""Please thoroughly validate this tail-chasing pattern fix.

ORIGINAL ISSUE: {issue_description}

ORIGINAL CODE:
```python
{original_code}
```

PROPOSED FIX:
```python
{fixed_code}
```
{test_cases_text}

Please analyze:

1. **Issue Resolution**: Does the fix actually eliminate the stated tail-chasing pattern?
2. **Completeness**: Is the implementation complete and functional?
3. **Syntax Correctness**: Is the code syntactically valid Python?
4. **Functionality Preservation**: Does it maintain all original intended behavior?
5. **New Issues**: Are any new problems introduced?
6. **AST Safety**: Are the transformations AST-safe?

Provide your assessment with:
- **Valid**: true/false - whether this is a good fix
- **Confidence**: 0.0-1.0 confidence in your assessment
- **Issues Found**: Any problems you identified
- **Recommendations**: Suggestions for improvement if needed

Be thorough but practical in your analysis."""

        return self.generate_fix(prompt, temperature=0.0)