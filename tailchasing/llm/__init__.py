"""
Clean LLM abstraction layer for tail-chasing fix generation.

Provides unified interface to multiple LLM providers with intelligent prompting,
retry logic, and cost tracking for code generation and fix suggestions.
"""

from .adapters.base import (
    LLMAdapter,
    LLMResponse,
    PromptTemplate,
    CostTracker,
    RetryConfig
)
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.anthropic_adapter import AnthropicAdapter  
from .adapters.ollama_adapter import OllamaAdapter
from .manager import LLMManager, create_llm_manager
from .prompts import TailChasingPrompts

__all__ = [
    "LLMAdapter",
    "LLMResponse", 
    "PromptTemplate",
    "CostTracker",
    "RetryConfig",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter", 
    "LLMManager",
    "create_llm_manager",
    "TailChasingPrompts"
]