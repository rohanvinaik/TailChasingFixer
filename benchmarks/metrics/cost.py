"""
Cost metrics tracking token usage and estimated costs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModelPricing(Enum):
    """Token pricing for different models (per 1K tokens)."""
    
    # OpenAI pricing (as of 2024)
    GPT4_TURBO_INPUT = 0.01
    GPT4_TURBO_OUTPUT = 0.03
    GPT4_INPUT = 0.03
    GPT4_OUTPUT = 0.06
    GPT35_TURBO_INPUT = 0.0005
    GPT35_TURBO_OUTPUT = 0.0015
    
    # Anthropic pricing
    CLAUDE3_OPUS_INPUT = 0.015
    CLAUDE3_OPUS_OUTPUT = 0.075
    CLAUDE3_SONNET_INPUT = 0.003
    CLAUDE3_SONNET_OUTPUT = 0.015
    CLAUDE3_HAIKU_INPUT = 0.00025
    CLAUDE3_HAIKU_OUTPUT = 0.00125
    
    # Local models (compute cost estimate)
    LOCAL_LLAMA_INPUT = 0.0001
    LOCAL_LLAMA_OUTPUT = 0.0001
    LOCAL_MISTRAL_INPUT = 0.0001
    LOCAL_MISTRAL_OUTPUT = 0.0001


@dataclass
class CostMetrics:
    """Metrics related to token usage and costs."""
    
    # Token usage
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Per-step token usage
    step_tokens: List[int] = field(default_factory=list)
    max_tokens_per_step: int = 0
    average_tokens_per_step: float = 0.0
    
    # Cost calculation
    total_cost: float = 0.0  # USD
    input_cost: float = 0.0
    output_cost: float = 0.0
    
    # Model-specific tracking
    model_name: str = ""
    pricing_model: Optional[Dict[str, float]] = None
    
    # Efficiency metrics
    tokens_per_fix: float = 0.0
    cost_per_fix: float = 0.0
    wasted_tokens: int = 0  # Tokens used in failed attempts
    
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.pricing_model = self._get_pricing_model(model_name)
        self.step_tokens = []
    
    def _get_pricing_model(self, model_name: str) -> Dict[str, float]:
        """Get pricing model based on model name."""
        model_lower = model_name.lower()
        
        if "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower:
            return {
                "input": ModelPricing.GPT4_TURBO_INPUT.value,
                "output": ModelPricing.GPT4_TURBO_OUTPUT.value
            }
        elif "gpt-4" in model_lower:
            return {
                "input": ModelPricing.GPT4_INPUT.value,
                "output": ModelPricing.GPT4_OUTPUT.value
            }
        elif "gpt-3.5" in model_lower:
            return {
                "input": ModelPricing.GPT35_TURBO_INPUT.value,
                "output": ModelPricing.GPT35_TURBO_OUTPUT.value
            }
        elif "claude-3-opus" in model_lower:
            return {
                "input": ModelPricing.CLAUDE3_OPUS_INPUT.value,
                "output": ModelPricing.CLAUDE3_OPUS_OUTPUT.value
            }
        elif "claude-3-sonnet" in model_lower:
            return {
                "input": ModelPricing.CLAUDE3_SONNET_INPUT.value,
                "output": ModelPricing.CLAUDE3_SONNET_OUTPUT.value
            }
        elif "claude-3-haiku" in model_lower:
            return {
                "input": ModelPricing.CLAUDE3_HAIKU_INPUT.value,
                "output": ModelPricing.CLAUDE3_HAIKU_OUTPUT.value
            }
        elif "llama" in model_lower:
            return {
                "input": ModelPricing.LOCAL_LLAMA_INPUT.value,
                "output": ModelPricing.LOCAL_LLAMA_OUTPUT.value
            }
        elif "mistral" in model_lower:
            return {
                "input": ModelPricing.LOCAL_MISTRAL_INPUT.value,
                "output": ModelPricing.LOCAL_MISTRAL_OUTPUT.value
            }
        else:
            # Default/unknown model - use conservative estimate
            return {
                "input": 0.001,
                "output": 0.002
            }
    
    def record_tokens(self, tokens: int, input_tokens: Optional[int] = None, 
                     output_tokens: Optional[int] = None, failed: bool = False):
        """Record token usage for a step."""
        self.total_tokens += tokens
        self.step_tokens.append(tokens)
        
        # Update max
        self.max_tokens_per_step = max(self.max_tokens_per_step, tokens)
        
        # Track wasted tokens if step failed
        if failed:
            self.wasted_tokens += tokens
        
        # If we have input/output breakdown
        if input_tokens is not None:
            self.input_tokens += input_tokens
        if output_tokens is not None:
            self.output_tokens += output_tokens
        
        # If we don't have breakdown, estimate 30/70 split
        if input_tokens is None and output_tokens is None:
            estimated_input = int(tokens * 0.3)
            estimated_output = tokens - estimated_input
            self.input_tokens += estimated_input
            self.output_tokens += estimated_output
    
    def calculate_total_cost(self):
        """Calculate total cost based on token usage and pricing model."""
        if not self.pricing_model:
            return
        
        # Calculate costs (pricing is per 1K tokens)
        self.input_cost = (self.input_tokens / 1000) * self.pricing_model["input"]
        self.output_cost = (self.output_tokens / 1000) * self.pricing_model["output"]
        self.total_cost = self.input_cost + self.output_cost
        
        # Calculate averages
        if self.step_tokens:
            self.average_tokens_per_step = sum(self.step_tokens) / len(self.step_tokens)
    
    def calculate_efficiency_metrics(self, num_fixes: int):
        """Calculate efficiency metrics based on number of fixes."""
        if num_fixes > 0:
            self.tokens_per_fix = self.total_tokens / num_fixes
            self.cost_per_fix = self.total_cost / num_fixes
    
    def get_efficiency_score(self) -> float:
        """Calculate cost efficiency score (0-100)."""
        base_score = 100.0
        
        # Penalize for wasted tokens
        if self.total_tokens > 0:
            waste_ratio = self.wasted_tokens / self.total_tokens
            base_score -= waste_ratio * 30
        
        # Penalize for excessive token usage (>10K tokens is excessive for our benchmarks)
        if self.total_tokens > 10000:
            excess_ratio = (self.total_tokens - 10000) / 10000
            base_score -= min(40, excess_ratio * 40)
        
        # Penalize for high cost (>$0.50 is high for a single benchmark)
        if self.total_cost > 0.50:
            cost_penalty = min(30, (self.total_cost - 0.50) / 0.50 * 30)
            base_score -= cost_penalty
        
        return max(0.0, base_score)
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown."""
        return {
            "total_cost": self.total_cost,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "average_cost_per_step": self.total_cost / len(self.step_tokens) if self.step_tokens else 0,
            "wasted_cost": (self.wasted_tokens / self.total_tokens * self.total_cost) if self.total_tokens > 0 else 0,
            "tokens_per_dollar": self.total_tokens / self.total_cost if self.total_cost > 0 else 0
        }