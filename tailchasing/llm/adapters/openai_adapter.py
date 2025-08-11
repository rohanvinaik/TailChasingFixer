"""
OpenAI adapter for GPT-4o and GPT-4o-mini models.

Provides integration with OpenAI's API for code generation and fix suggestions
with proper cost tracking and model-specific optimizations.
"""

import json
import os
from typing import Dict, Optional, Any, Union, List

from .base import BaseLLMAdapter, LLMResponse, ModelTier, RetryConfig


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI GPT models."""
    
    # Model configurations
    MODELS = {
        "gpt-4o": {
            "tier": ModelTier.PREMIUM,
            "input_cost_per_1k": 0.005,   # $5 per 1M input tokens
            "output_cost_per_1k": 0.015,  # $15 per 1M output tokens
            "max_tokens": 128000,
            "encoding": "cl100k_base"
        },
        "gpt-4o-mini": {
            "tier": ModelTier.STANDARD, 
            "input_cost_per_1k": 0.00015,  # $0.15 per 1M input tokens
            "output_cost_per_1k": 0.0006,  # $0.60 per 1M output tokens
            "max_tokens": 128000,
            "encoding": "cl100k_base"
        }
    }
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 retry_config: Optional[RetryConfig] = None):
        
        if model not in self.MODELS:
            raise ValueError(f"Unsupported model: {model}. Available: {list(self.MODELS.keys())}")
        
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"), retry_config)
        self.model = model
        self.model_config = self.MODELS[model]
        
        # Initialize tokenizer
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(self.model_config["encoding"])
        except ImportError:
            self.tokenizer = None  # Will fallback to parent class method
        except Exception:
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self.tokenizer = None
        
        # Initialize OpenAI client (lazy import to avoid dependency issues)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client
    
    @property
    def model_tier(self) -> ModelTier:
        return self.model_config["tier"]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tiktoken."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to parent class method
            return super().count_tokens(text)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for given token counts."""
        input_cost = (input_tokens / 1000) * self.model_config["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * self.model_config["output_cost_per_1k"]
        return input_cost + output_cost
    
    def _make_api_call(self, prompt: str, **kwargs) -> LLMResponse:
        """Make API call to OpenAI."""
        max_tokens = kwargs.get("max_tokens", 2000)
        temperature = kwargs.get("temperature", 0.1)
        
        # Determine mode from kwargs or context
        mode = kwargs.get("mode", "refactor")
        settings = kwargs.get("settings", None)
        context = kwargs.get("context", None)
        
        # Build messages for chat completion
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(mode, settings=settings, context=context)
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Count input tokens
        input_tokens = sum(self.count_tokens(msg["content"]) for msg in messages)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"} if self._requires_json(prompt) else None
            )
            
            content = response.choices[0].message.content
            output_tokens = response.usage.completion_tokens
            cost = self.estimate_cost(input_tokens, output_tokens)
            
            # Parse JSON response if requested
            parsed_content = self._parse_response(content, prompt)
            
            return LLMResponse(
                content=parsed_content["content"] if isinstance(parsed_content, dict) else parsed_content,
                confidence=parsed_content.get("confidence") if isinstance(parsed_content, dict) else None,
                rationale=parsed_content.get("rationale") if isinstance(parsed_content, dict) else None,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _get_system_prompt(self, mode: str = "refactor", *, settings=None, context=None) -> str:
        """Get system prompt tuned for specific refactor/verification tasks."""
        base = [
            "You are a meticulous senior software engineer.",
            "Follow these rules strictly:",
            "1) Prefer AST-safe, minimal diffs.",
            "2) Preserve public API and behavior.",
            "3) Do not invent symbols, imports, or files.",
            "4) If uncertain, explain and stop rather than hallucinate.",
        ]
        
        if mode == "lint_fix":
            base.append("Scope: style-only changes (format, import order). No behavior changes.")
        elif mode == "test_fix":
            base.append("Scope: make the smallest change to satisfy failing tests in the provided logs.")
        elif mode == "codegen":
            base.append("Scope: generate new code only when explicitly instructed.")
        else:
            base.append("Scope: refactor existing code with minimal, safe edits.")
        
        # Optional toggles from settings/context
        pyver = getattr(settings, "python_version", "3.11")
        base.append(f"Environment: Python {pyver}; tools available: ruff, mypy, pytest.")
        base.append("Output format: unified diff or patched function body only, no commentary unless asked.")
        base.append("PROMPT_VERSION=openai.v1")
        
        return "\n".join(base)
    
    def _requires_json(self, prompt: str) -> bool:
        """Check if prompt requires JSON response format."""
        json_indicators = [
            "respond in json",
            "json format",
            "confidence",
            "rationale",
            "{\"",
            "RESPOND WITH:"
        ]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in json_indicators)
    
    def _parse_response(self, content: str, original_prompt: str) -> Union[str, Dict[str, Any]]:
        """Parse response content, handling JSON when appropriate."""
        if not self._requires_json(original_prompt):
            return content
        
        try:
            # Try to parse as JSON
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            # Fallback: extract JSON-like content with regex
            import re
            
            # Look for confidence and rationale patterns
            confidence_match = re.search(r'"confidence":\s*([\d.]+)', content)
            rationale_match = re.search(r'"rationale":\s*"([^"]+)"', content)
            
            result = {"content": content}
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))
            if rationale_match:
                result["rationale"] = rationale_match.group(1)
            
            return result
    
    def generate_fix_with_examples(self,
                                  issue_code: str,
                                  issue_type: str,
                                  context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Generate fix with few-shot examples for the specific issue type."""
        examples = self._get_few_shot_examples(issue_type)
        
        prompt = f"""Here are examples of fixing {issue_type} patterns:

{examples}

Now fix this similar issue:

ISSUE TYPE: {issue_type}
CODE TO FIX:
```python
{issue_code}
```

CONTEXT: {json.dumps(context or {}, indent=2)}

Provide your fix in JSON format:
{{
  "content": "complete fixed code",
  "confidence": 0.0-1.0,
  "rationale": "brief explanation"
}}"""
        
        return self.generate_fix(prompt, context, temperature=0.0)
    
    def _get_few_shot_examples(self, issue_type: str) -> str:
        """Get few-shot examples for specific issue types."""
        examples = {
            "phantom_function": """
EXAMPLE 1 - Phantom Function Fix:
BEFORE:
```python
def process_data(data):
    pass  # TODO: implement
```

AFTER:
```python
def process_data(data):
    return [item.upper() for item in data if item]
```
RATIONALE: Implemented complete functionality based on function name and typical use patterns.

EXAMPLE 2 - Remove Phantom:
BEFORE:
```python
def unused_helper():
    raise NotImplementedError("To be implemented")
```

AFTER:
(function removed completely - not referenced anywhere)
RATIONALE: Function was never called, so removal is safest approach.
""",
            
            "duplicate_function": """
EXAMPLE - Duplicate Function Consolidation:
BEFORE:
```python
def validate_email(email):
    return "@" in email and "." in email

def check_email_format(email):
    return "@" in email and "." in email.split("@")[1]

def email_is_valid(email_addr):
    return bool(re.match(r'^[^@]+@[^@]+\.[^@]+$', email_addr))
```

AFTER:
```python
def validate_email(email):
    return bool(re.match(r'^[^@]+@[^@]+\.[^@]+$', email))

# All calls updated to use validate_email()
```
RATIONALE: Chose the most robust implementation with regex validation, updated all callers.
""",
            
            "circular_import": """
EXAMPLE - Circular Import Fix:
BEFORE:
```python
# file: models.py
from utils import format_data
class User:
    def display(self): return format_data(self.name)

# file: utils.py  
from models import User
def format_data(data): return f"User: {data}"
```

AFTER:
```python
# file: models.py
class User:
    def display(self): 
        from utils import format_data  # Local import
        return format_data(self.name)

# file: utils.py
def format_data(data): 
    return f"User: {data}"
```
RATIONALE: Moved import inside method to break circular dependency while preserving functionality.
"""
        }
        
        return examples.get(issue_type, "No specific examples available for this issue type.")
    
    def batch_generate_fixes(self, 
                           issues: List[Dict[str, Any]], 
                           max_concurrent: int = 3) -> List[LLMResponse]:
        """Generate fixes for multiple issues with rate limiting."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def generate_single_fix(issue_data):
            return self.generate_fix_with_examples(
                issue_data["code"],
                issue_data["type"], 
                issue_data.get("context")
            )
        
        responses = []
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(generate_single_fix, issue) for issue in issues]
            for future in futures:
                try:
                    response = future.result(timeout=30)
                    responses.append(response)
                except Exception as e:
                    self.logger.error(f"Batch fix generation failed: {e}")
                    responses.append(LLMResponse(
                        content="", 
                        model_used=self.model,
                        confidence=0.0
                    ))
        
        return responses