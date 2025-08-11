"""
Ollama adapter for local LLM models.

Provides integration with Ollama for running local models like CodeLlama, Mistral,
and other open-source models for code generation and fix suggestions.
"""

import json
import logging
import requests
import time
from typing import Dict, Optional, Any, Union, List

from .base import BaseLLMAdapter, LLMResponse, ModelTier, RetryConfig


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for local Ollama models."""
    
    # Common Ollama models suitable for code tasks
    RECOMMENDED_MODELS = {
        "codellama:7b-code": {
            "tier": ModelTier.LOCAL,
            "suitable_for": ["code_generation", "simple_fixes"],
            "max_context": 4096
        },
        "codellama:13b-code": {
            "tier": ModelTier.LOCAL, 
            "suitable_for": ["code_generation", "complex_fixes", "explanations"],
            "max_context": 4096
        },
        "mistral:7b": {
            "tier": ModelTier.LOCAL,
            "suitable_for": ["explanations", "simple_fixes"],
            "max_context": 8192
        },
        "deepseek-coder:6.7b": {
            "tier": ModelTier.LOCAL,
            "suitable_for": ["code_generation", "complex_fixes"],
            "max_context": 16384
        },
        "codeqwen:7b": {
            "tier": ModelTier.LOCAL,
            "suitable_for": ["code_generation", "bug_fixes"],
            "max_context": 8192
        }
    }
    
    def __init__(self,
                 model: str = "codellama:7b-code",
                 ollama_host: str = "http://localhost:11434",
                 retry_config: Optional[RetryConfig] = None):
        
        super().__init__(api_key=None, retry_config=retry_config)  # No API key needed for local
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.model_config = self.RECOMMENDED_MODELS.get(model, {
            "tier": ModelTier.LOCAL,
            "suitable_for": ["general"],
            "max_context": 4096
        })
        
        # Check if Ollama is running and model is available
        self._check_ollama_availability()
    
    @property
    def model_tier(self) -> ModelTier:
        return ModelTier.LOCAL
    
    def count_tokens(self, text: str) -> int:
        """Approximate token counting for local models."""
        # Ollama doesn't provide exact token counting, use approximation
        return super().count_tokens(text)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """
        Estimate equivalent cost for local models.
        
        While local models run for free, we track an equivalent cost for:
        - Resource accounting and comparison
        - Understanding relative computational expense
        - Capacity planning for local vs cloud models
        """
        pt = max(0, int(input_tokens or 0))
        ct = max(0, int(output_tokens or 0))
        total = pt + ct
        
        # Equivalent pricing map: per-1k tokens (input, output)
        # Based on relative compute requirements vs cloud models
        PRICING = {
            # Llama models
            "llama3:8b": (0.05, 0.10),
            "llama3:70b": (0.60, 0.80),
            "llama2:7b": (0.04, 0.08),
            "llama2:13b": (0.08, 0.15),
            "llama2:70b": (0.55, 0.75),
            
            # CodeLlama models
            "codellama:7b": (0.04, 0.08),
            "codellama:7b-code": (0.04, 0.08),
            "codellama:13b": (0.08, 0.15),
            "codellama:13b-code": (0.08, 0.15),
            "codellama:34b": (0.20, 0.35),
            
            # Mistral models
            "mistral:7b": (0.03, 0.06),
            "mixtral:8x7b": (0.25, 0.40),
            "mistral-nemo:12b": (0.10, 0.18),
            
            # DeepSeek models
            "deepseek-coder:6.7b": (0.03, 0.06),
            "deepseek-coder:33b": (0.18, 0.30),
            
            # Qwen models
            "codeqwen:7b": (0.04, 0.08),
            "qwen2:7b": (0.04, 0.08),
            "qwen2:72b": (0.65, 0.85),
            
            # Phi models
            "phi3:mini": (0.02, 0.04),
            "phi3:medium": (0.08, 0.15),
            
            # StarCoder models
            "starcoder2:3b": (0.02, 0.04),
            "starcoder2:7b": (0.04, 0.08),
            "starcoder2:15b": (0.12, 0.20),
            
            # Default for unknown models (conservative estimate)
            "default": (0.05, 0.10)
        }
        
        def get_model_family(model_name: str) -> str:
            """Extract model family for pricing fallback."""
            # Try exact match first
            if model_name in PRICING:
                return model_name
            
            # Try family prefix match
            for key in PRICING:
                if model_name.startswith(key.split(":")[0]):
                    return key
            
            # Try to match by parameter count
            import re
            param_match = re.search(r'(\d+)b', model_name.lower())
            if param_match:
                param_size = int(param_match.group(1))
                if param_size <= 7:
                    return "default"
                elif param_size <= 13:
                    return "codellama:13b"  # Medium model equivalent
                elif param_size <= 34:
                    return "codellama:34b"  # Large model equivalent
                else:
                    return "llama3:70b"  # Very large model equivalent
            
            return "default"
        
        # Get pricing for this model
        pricing_key = get_model_family(self.model)
        in_rate, out_rate = PRICING.get(pricing_key, PRICING["default"])
        
        # Calculate equivalent cost
        equivalent_cost = (pt / 1000.0) * in_rate + (ct / 1000.0) * out_rate
        
        # Build comprehensive info
        info = {
            "model": self.model,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": total,
            "cost_usd": round(equivalent_cost, 6),
            "actual_cost_usd": 0.0,  # Always 0 for local models
            "equivalent_cost_usd": round(equivalent_cost, 6),
            "pricing_model": pricing_key,
            "is_local": True,
            "rates": {
                "input_per_1k": in_rate,
                "output_per_1k": out_rate
            }
        }
        
        logging.debug("Ollama cost estimation: %s", info)
        return info
    
    def _check_ollama_availability(self) -> None:
        """Check if Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama server not responding")
            
            # Check if model is available
            available_models = [model["name"] for model in response.json().get("models", [])]
            if self.model not in available_models:
                self.logger.warning(
                    f"Model {self.model} not found in Ollama. Available: {available_models}. "
                    f"You may need to run: ollama pull {self.model}"
                )
            
        except (requests.RequestException, ConnectionError) as e:
            self.logger.error(
                f"Cannot connect to Ollama at {self.ollama_host}. "
                f"Make sure Ollama is installed and running. Error: {e}"
            )
            raise ConnectionError(f"Ollama not available: {e}")
    
    def _make_api_call(self, prompt: str, **kwargs) -> LLMResponse:
        """Make API call to local Ollama instance."""
        max_tokens = kwargs.get("max_tokens", 2000)
        temperature = kwargs.get("temperature", 0.1)
        
        # Determine mode from kwargs or context
        mode = kwargs.get("mode", "refactor")
        settings = kwargs.get("settings", None)
        context = kwargs.get("context", None)
        
        # Build prompt with system context
        full_prompt = f"{self._get_system_prompt(mode, settings=settings, context=context)}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "stop": ["Human:", "User:", "Assistant:", "\n\n---"]
            }
        }
        
        input_tokens = self.count_tokens(full_prompt)
        
        try:
            # Track timing
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=120  # Local models can be slow
            )
            
            # Calculate duration
            duration_seconds = time.time() - start_time
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            content = result.get("response", "")
            
            # Approximate output token count
            output_tokens = self.count_tokens(content)
            
            # Get cost estimation with all details
            cost_info = self.estimate_cost(input_tokens, output_tokens)
            
            # Parse structured response if needed
            parsed_content = self._parse_response(content, prompt)
            
            # Create response with full accounting
            llm_response = LLMResponse(
                content=parsed_content["content"] if isinstance(parsed_content, dict) else parsed_content,
                confidence=parsed_content.get("confidence") if isinstance(parsed_content, dict) else None,
                rationale=parsed_content.get("rationale") if isinstance(parsed_content, dict) else None,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_info["equivalent_cost_usd"],  # Use equivalent cost for accounting
                duration_seconds=duration_seconds
            )
            
            # Log comprehensive metrics
            self.logger.debug(
                "Ollama call completed: model=%s, tokens=%d (in=%d, out=%d), "
                "time=%.2fs, equivalent_cost=$%.6f",
                self.model, cost_info["total_tokens"], input_tokens, output_tokens,
                duration_seconds, cost_info["equivalent_cost_usd"]
            )
            
            return llm_response
            
        except requests.RequestException as e:
            self.logger.error(f"Ollama API call failed: {e}")
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
        base.append("PROMPT_VERSION=ollama.v1")
        
        return "\n".join(base)
    
    def _parse_response(self, content: str, original_prompt: str) -> Union[str, Dict[str, Any]]:
        """Parse response from local model, handling various formats."""
        if not self._needs_structured_response(original_prompt):
            return self._clean_content(content)
        
        # Try JSON parsing first
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*"content"[^{}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract information with patterns
        cleaned_content = self._clean_content(content)
        result = {"content": cleaned_content}
        
        # Extract confidence if mentioned
        import re
        confidence_patterns = [
            r"confidence[:\s]+([\d.]+)",
            r"confident[:\s]+([\d.]+)",
            r"certainty[:\s]+([\d.]+)"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content.lower())
            if match:
                conf = float(match.group(1))
                if conf > 1.0:  # Handle percentage
                    conf /= 100.0
                result["confidence"] = min(1.0, max(0.0, conf))
                break
        
        return result
    
    def _needs_structured_response(self, prompt: str) -> bool:
        """Check if prompt expects JSON or structured response."""
        indicators = ["json", "confidence", "rationale", "format:", "respond with:"]
        return any(indicator in prompt.lower() for indicator in indicators)
    
    def _clean_content(self, content: str) -> str:
        """Clean up local model response content."""
        # Remove common artifacts from local models
        content = content.strip()
        
        # Remove repeated phrases local models sometimes generate
        lines = content.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            # Skip duplicate consecutive lines
            if line.strip() != prev_line.strip() or not line.strip():
                cleaned_lines.append(line)
            prev_line = line
        
        return '\n'.join(cleaned_lines)
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List models available in Ollama."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [
                    {
                        "name": model["name"],
                        "size": model.get("size", 0),
                        "modified": model.get("modified_at"),
                        "suitable_for": self.RECOMMENDED_MODELS.get(
                            model["name"], {}
                        ).get("suitable_for", ["general"])
                    }
                    for model in models
                ]
        except Exception as e:
            self.logger.error(f"Failed to list Ollama models: {e}")
        
        return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull/download a model to Ollama."""
        try:
            self.logger.info(f"Pulling model {model_name} via Ollama...")
            
            payload = {"name": model_name}
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json=payload,
                stream=True,
                timeout=600  # 10 minutes for model download
            )
            
            if response.status_code == 200:
                # Stream the download progress
                for line in response.iter_lines():
                    if line:
                        try:
                            status = json.loads(line.decode('utf-8'))
                            if status.get("status"):
                                self.logger.info(f"Pull status: {status['status']}")
                        except json.JSONDecodeError:
                            continue
                
                self.logger.info(f"Successfully pulled model {model_name}")
                return True
            else:
                self.logger.error(f"Failed to pull model: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def generate_with_context_limit(self,
                                  prompt: str, 
                                  context: Optional[Dict[str, Any]] = None,
                                  **kwargs) -> LLMResponse:
        """Generate response while respecting model's context limitations."""
        max_context = self.model_config.get("max_context", 4096)
        
        # Estimate tokens and truncate if needed
        full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
        prompt_tokens = self.count_tokens(full_prompt)
        
        if prompt_tokens > max_context * 0.8:  # Leave room for response
            # Truncate the prompt to fit context
            self.logger.warning(f"Prompt too long ({prompt_tokens} tokens), truncating for {self.model}")
            
            # Keep system prompt and truncate user prompt
            system_prompt = self._get_system_prompt()
            available_tokens = int(max_context * 0.6)  # 60% for prompt, 40% for response
            
            # Truncate from the middle to preserve start and end
            prompt_lines = prompt.split('\n')
            if len(prompt_lines) > 20:
                keep_start = prompt_lines[:10]
                keep_end = prompt_lines[-10:]
                truncated_prompt = '\n'.join(keep_start + ["\n... [content truncated] ...\n"] + keep_end)
            else:
                # Simple truncation
                truncated_prompt = prompt[:available_tokens * 4]  # ~4 chars per token
            
            return self.generate_fix(truncated_prompt, context, **kwargs)
        
        return self.generate_fix(prompt, context, **kwargs)