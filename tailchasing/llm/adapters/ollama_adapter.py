"""
Ollama adapter for local LLM models.

Provides integration with Ollama for running local models like CodeLlama, Mistral,
and other open-source models for code generation and fix suggestions.
"""

import json
import requests
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
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local models have no monetary cost."""
        return 0.0
    
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
        
        # Build prompt with system context
        full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
        
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
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=120  # Local models can be slow
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            content = result.get("response", "")
            
            # Approximate output token count
            output_tokens = self.count_tokens(content)
            
            # Parse structured response if needed
            parsed_content = self._parse_response(content, prompt)
            
            return LLMResponse(
                content=parsed_content["content"] if isinstance(parsed_content, dict) else parsed_content,
                confidence=parsed_content.get("confidence") if isinstance(parsed_content, dict) else None,
                rationale=parsed_content.get("rationale") if isinstance(parsed_content, dict) else None,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0  # Local models are free
            )
            
        except requests.RequestException as e:
            self.logger.error(f"Ollama API call failed: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt optimized for local code models."""
        return """You are a Python expert focused on fixing tail-chasing patterns in code. Provide complete, working solutions.

Key rules:
1. Never create incomplete functions or stubs - always implement fully
2. Fix the root cause, not just symptoms  
3. Consolidate duplicate code by choosing the best implementation
4. Ensure all code is syntactically correct
5. Preserve existing functionality while eliminating problems

For structured responses, use this exact JSON format:
{
  "content": "your complete code or explanation here",
  "confidence": 0.8,
  "rationale": "brief explanation of your approach"
}

Focus on practical, working solutions that eliminate tail-chasing patterns completely."""
    
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