"""
Format feedback for different LLM APIs.

Provides adapters to format tail-chasing prevention feedback
for various LLM API formats including OpenAI, Anthropic, and local models.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

from .feedback_generator import LLMFeedback, FeedbackGenerator
from ..core.issues import Issue

logger = logging.getLogger(__name__)


@dataclass
class FormattedPrompt:
    """A formatted prompt for a specific LLM API."""
    
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    assistant_prompt: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_format: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system_prompt': self.system_prompt,
            'user_prompt': self.user_prompt,
            'assistant_prompt': self.assistant_prompt,
            'messages': self.messages,
            'metadata': self.metadata
        }


class BaseLLMAdapter:
    """
    Base adapter for formatting feedback for LLM APIs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.include_examples = self.config.get('include_examples', True)
        self.max_prompt_length = self.config.get('max_prompt_length', 4000)
        self.include_metadata = self.config.get('include_metadata', False)
    
    def format_feedback(
        self,
        feedback: LLMFeedback,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedPrompt:
        """
        Format feedback for the specific LLM API.
        
        Args:
            feedback: LLMFeedback object
            context: Optional context about current task
            
        Returns:
            FormattedPrompt object
        """
        raise NotImplementedError("Subclasses must implement format_feedback")
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length while preserving structure."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at a sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If we find a period in the last 20%
            return truncated[:last_period + 1]
        
        # Otherwise truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space] + "..."
        
        return truncated + "..."
    
    def _format_examples(
        self,
        negative_examples: List[Dict[str, Any]],
        positive_examples: List[Dict[str, Any]]
    ) -> str:
        """Format examples for inclusion in prompt."""
        sections = []
        
        if negative_examples:
            sections.append("### ❌ Patterns to Avoid:")
            for example in negative_examples[:3]:  # Limit to 3
                sections.append(f"\n{example.get('description', 'Issue detected')}")
                if 'why_bad' in example:
                    sections.append(f"**Why:** {example['why_bad']}")
                if 'evidence' in example and example['evidence']:
                    evidence_str = ', '.join(f"{k}: {v}" for k, v in example['evidence'].items())
                    sections.append(f"**Evidence:** {evidence_str}")
        
        if positive_examples:
            sections.append("\n### ✅ Better Approaches:")
            for example in positive_examples[:3]:  # Limit to 3
                sections.append(f"\n{example.get('description', 'Improvement')}")
                if 'example' in example:
                    # Format code example
                    code = example['example']
                    # Limit code length
                    if len(code) > 500:
                        code = code[:500] + "\n# ... (truncated)"
                    sections.append(f"```python\n{code}\n```")
                if 'principle' in example:
                    sections.append(f"**Principle:** {example['principle']}")
        
        return "\n".join(sections)
    
    def _format_rules(self, rules: List[str]) -> str:
        """Format rules as a numbered list."""
        if not rules:
            return ""
        
        formatted = ["### 📋 Key Rules to Follow:"]
        for i, rule in enumerate(rules[:10], 1):  # Limit to 10 rules
            formatted.append(f"{i}. {rule}")
        
        return "\n".join(formatted)


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI API format (ChatGPT, GPT-4, etc.).
    """
    
    def format_feedback(
        self,
        feedback: LLMFeedback,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedPrompt:
        """
        Format feedback for OpenAI API.
        
        Returns a FormattedPrompt with messages in OpenAI format.
        """
        messages = []
        
        # Build system message
        system_parts = []
        
        # Add base system prompt
        system_parts.append(
            "You are a helpful assistant that writes high-quality code while "
            "avoiding common anti-patterns and tail-chasing behaviors."
        )
        
        # Add system prompt additions from feedback
        if feedback.system_prompt_additions:
            system_parts.extend(feedback.system_prompt_additions)
        
        # Add rules
        if feedback.rules:
            system_parts.append(self._format_rules(feedback.rules))
        
        # Create system message
        system_content = "\n\n".join(system_parts)
        if self.max_prompt_length and len(system_content) > self.max_prompt_length:
            system_content = self._truncate_text(system_content, self.max_prompt_length)
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add context warnings as assistant message if present
        if feedback.context_warnings:
            warning_content = "\n\n".join(feedback.context_warnings)
            messages.append({
                "role": "assistant",
                "content": f"⚠️ **Important Context:**\n{warning_content}"
            })
        
        # Add examples if configured
        if self.include_examples and (feedback.negative_examples or feedback.positive_examples):
            examples_content = self._format_examples(
                feedback.negative_examples,
                feedback.positive_examples
            )
            if examples_content:
                messages.append({
                    "role": "assistant",
                    "content": examples_content
                })
        
        # Add user context if provided
        if context and context.get('current_task'):
            messages.append({
                "role": "user",
                "content": f"Current task: {context['current_task']}"
            })
        
        # Create formatted prompt
        prompt = FormattedPrompt(
            messages=messages,
            metadata=feedback.metadata if self.include_metadata else {},
            raw_format={
                "model": self.config.get('model', 'gpt-4'),
                "messages": messages,
                "temperature": self.config.get('temperature', 0.7),
                "max_tokens": self.config.get('max_tokens', None)
            }
        )
        
        return prompt
    
    def format_for_completion(
        self,
        feedback: LLMFeedback,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Format feedback for OpenAI completion endpoint.
        
        Args:
            feedback: LLMFeedback object
            prompt: User's prompt
            
        Returns:
            OpenAI API request format
        """
        formatted = self.format_feedback(feedback)
        
        # Add user prompt
        formatted.messages.append({
            "role": "user",
            "content": prompt
        })
        
        return formatted.raw_format


class AnthropicAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic API format (Claude).
    """
    
    def format_feedback(
        self,
        feedback: LLMFeedback,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedPrompt:
        """
        Format feedback for Anthropic API.
        
        Returns a FormattedPrompt with Anthropic-style formatting.
        """
        # Build system prompt
        system_parts = []
        
        # Add focused system instruction
        system_parts.append(
            "You are Claude, an AI assistant focused on writing clean, maintainable code. "
            "You actively avoid creating duplicate functionality, circular dependencies, "
            "and other anti-patterns that can arise from incremental development."
        )
        
        # Add specific guidance from feedback
        if feedback.system_prompt_additions:
            system_parts.extend(feedback.system_prompt_additions)
        
        system_prompt = "\n\n".join(system_parts)
        
        # Build user prompt with context
        user_parts = []
        
        # Add context warnings
        if feedback.context_warnings:
            user_parts.append("## Context Alerts\n")
            user_parts.extend(feedback.context_warnings)
            user_parts.append("")  # Empty line
        
        # Add rules in a more conversational style
        if feedback.rules:
            user_parts.append("## Guidelines\n")
            user_parts.append(
                "Please keep these guidelines in mind while working on the code:"
            )
            for rule in feedback.rules[:7]:  # Fewer rules for Claude
                user_parts.append(f"• {rule}")
            user_parts.append("")
        
        # Add examples if configured
        if self.include_examples:
            if feedback.negative_examples:
                user_parts.append("## Examples of Issues to Avoid\n")
                for example in feedback.negative_examples[:2]:  # Fewer examples
                    desc = example.get('description', 'Issue')
                    reason = example.get('why_bad', 'Problematic pattern')
                    user_parts.append(f"{desc}\n   → {reason}\n")
            
            if feedback.positive_examples:
                user_parts.append("## Recommended Patterns\n")
                for example in feedback.positive_examples[:2]:
                    desc = example.get('description', 'Pattern')
                    principle = example.get('principle', '')
                    user_parts.append(f"{desc}")
                    if principle:
                        user_parts.append(f"   Principle: {principle}\n")
        
        user_prompt = "\n".join(user_parts) if user_parts else None
        
        # Build messages list for Anthropic format
        messages = []
        
        if user_prompt:
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            messages.append({
                "role": "assistant",
                "content": "I understand these guidelines and will follow them carefully while working on the code. I'll actively avoid creating duplicate functions, circular dependencies, and other tail-chasing patterns."
            })
        
        # Create formatted prompt
        prompt = FormattedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            metadata=feedback.metadata if self.include_metadata else {},
            raw_format={
                "model": self.config.get('model', 'claude-3-opus-20240229'),
                "system": system_prompt,
                "messages": messages,
                "max_tokens": self.config.get('max_tokens', 4096),
                "temperature": self.config.get('temperature', 0.7)
            }
        )
        
        return prompt
    
    def format_for_messages(
        self,
        feedback: LLMFeedback,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Format feedback for Anthropic messages endpoint.
        
        Args:
            feedback: LLMFeedback object
            user_message: User's message
            
        Returns:
            Anthropic API request format
        """
        formatted = self.format_feedback(feedback)
        
        # Add user message
        formatted.raw_format["messages"].append({
            "role": "user",
            "content": user_message
        })
        
        return formatted.raw_format


class LocalLLMAdapter(BaseLLMAdapter):
    """
    Adapter for local LLM formats (llama.cpp, Ollama, etc.).
    """
    
    def format_feedback(
        self,
        feedback: LLMFeedback,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedPrompt:
        """
        Format feedback for local LLM.
        
        Returns a FormattedPrompt with simple text format suitable for local models.
        """
        # Build a single concatenated prompt
        prompt_parts = []
        
        # Add instructions
        prompt_parts.append("### Instructions")
        prompt_parts.append(
            "You are an AI coding assistant. Follow these guidelines to avoid "
            "common anti-patterns and write clean, maintainable code:"
        )
        prompt_parts.append("")
        
        # Add rules as simple bullet points
        if feedback.rules:
            prompt_parts.append("### Guidelines")
            for rule in feedback.rules[:5]:  # Fewer rules for local models
                prompt_parts.append(f"- {rule}")
            prompt_parts.append("")
        
        # Add simplified warnings
        if feedback.context_warnings:
            prompt_parts.append("### Warnings")
            for warning in feedback.context_warnings[:2]:  # Limit warnings
                # Simplify warning text
                simple_warning = re.sub(r'[*_`]', '', warning)  # Remove markdown
                simple_warning = simple_warning.replace('⚠️', 'WARNING:')
                prompt_parts.append(simple_warning)
            prompt_parts.append("")
        
        # Add very simple examples
        if self.include_examples and feedback.negative_examples:
            prompt_parts.append("### Avoid These Patterns")
            for example in feedback.negative_examples[:2]:
                pattern = example.get('pattern', 'unknown')
                prompt_parts.append(f"- {pattern.replace('_', ' ').title()}")
            prompt_parts.append("")
        
        # Combine into single prompt
        full_prompt = "\n".join(prompt_parts)
        
        # Apply token limit for local models
        max_tokens = self.config.get('max_prompt_tokens', 2000)
        if len(full_prompt) > max_tokens:
            full_prompt = self._truncate_text(full_prompt, max_tokens)
        
        # Create formatted prompt
        prompt = FormattedPrompt(
            system_prompt=full_prompt,
            metadata=feedback.metadata if self.include_metadata else {},
            raw_format={
                "prompt": full_prompt,
                "temperature": self.config.get('temperature', 0.7),
                "top_p": self.config.get('top_p', 0.9),
                "max_tokens": self.config.get('max_tokens', 1024)
            }
        )
        
        return prompt
    
    def format_for_completion(
        self,
        feedback: LLMFeedback,
        user_input: str
    ) -> str:
        """
        Format feedback and user input for local LLM completion.
        
        Args:
            feedback: LLMFeedback object
            user_input: User's input
            
        Returns:
            Complete prompt string
        """
        formatted = self.format_feedback(feedback)
        
        # Add user input with clear separation
        full_prompt = formatted.system_prompt + "\n\n### User Request\n" + user_input + "\n\n### Response\n"
        
        return full_prompt


class UniversalAdapter:
    """
    Universal adapter that can format for multiple LLM APIs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize universal adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize specific adapters
        self.adapters = {
            'openai': OpenAIAdapter(config),
            'anthropic': AnthropicAdapter(config),
            'local': LocalLLMAdapter(config)
        }
        
        self.default_format = self.config.get('default_format', 'openai')
    
    def format_feedback(
        self,
        feedback: LLMFeedback,
        format_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedPrompt:
        """
        Format feedback for specified LLM type.
        
        Args:
            feedback: LLMFeedback object
            format_type: Type of formatting ('openai', 'anthropic', 'local')
            context: Optional context
            
        Returns:
            FormattedPrompt object
        """
        format_type = format_type or self.default_format
        
        adapter = self.adapters.get(format_type)
        if not adapter:
            logger.warning(f"Unknown format type: {format_type}, using default")
            adapter = self.adapters[self.default_format]
        
        return adapter.format_feedback(feedback, context)
    
    def generate_and_format(
        self,
        issues: List[Issue],
        format_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedPrompt:
        """
        Generate feedback from issues and format for LLM.
        
        Args:
            issues: List of detected issues
            format_type: Type of formatting
            context: Optional context
            
        Returns:
            FormattedPrompt object
        """
        # Generate feedback
        generator = FeedbackGenerator(self.config)
        feedback = generator.generate_feedback(issues, context)
        
        # Format for specified LLM
        return self.format_feedback(feedback, format_type, context)
    
    def export_all_formats(
        self,
        feedback: LLMFeedback,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Export feedback in all available formats.
        
        Args:
            feedback: LLMFeedback object
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        output_dir = output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_name, adapter in self.adapters.items():
            formatted = adapter.format_feedback(feedback)
            
            # Save to file
            filename = f"feedback_{format_name}_{timestamp}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(formatted.to_dict(), f, indent=2, default=str)
            
            exported[format_name] = filepath
            logger.info(f"Exported {format_name} format to {filepath}")
        
        return exported


class StreamingAdapter:
    """
    Adapter for streaming LLM responses with real-time feedback injection.
    """
    
    def __init__(self, base_adapter: BaseLLMAdapter):
        """
        Initialize streaming adapter.
        
        Args:
            base_adapter: Base adapter to use for formatting
        """
        self.base_adapter = base_adapter
        self.active_patterns: Set[str] = set()
        self.pattern_counts: Dict[str, int] = defaultdict(int)
    
    def inject_warning(
        self,
        pattern_type: str,
        context: str
    ) -> Optional[str]:
        """
        Generate a warning to inject into the stream.
        
        Args:
            pattern_type: Type of pattern detected
            context: Current context
            
        Returns:
            Warning message to inject, or None
        """
        # Avoid warning about the same pattern repeatedly
        if pattern_type in self.active_patterns:
            self.pattern_counts[pattern_type] += 1
            if self.pattern_counts[pattern_type] % 5 != 0:  # Warn every 5th occurrence
                return None
        else:
            self.active_patterns.add(pattern_type)
            self.pattern_counts[pattern_type] = 1
        
        # Generate pattern-specific warning
        warnings = {
            'duplicate_function': (
                "\n⚠️ Similar function already exists. "
                "Consider reusing or extending it instead.\n"
            ),
            'circular_import': (
                "\n⚠️ This import may create a circular dependency. "
                "Check the import hierarchy.\n"
            ),
            'phantom_function': (
                "\n⚠️ Creating a stub function. "
                "Remember to implement it fully or remove if not needed.\n"
            ),
            'import_anxiety': (
                "\n⚠️ Many imports detected. "
                "Only import what you actually use.\n"
            )
        }
        
        return warnings.get(pattern_type)
    
    def create_checkpoint_feedback(
        self,
        issues_so_far: List[Issue]
    ) -> Optional[str]:
        """
        Create feedback at a checkpoint in the stream.
        
        Args:
            issues_so_far: Issues detected up to this point
            
        Returns:
            Checkpoint feedback message
        """
        if not issues_so_far:
            return None
        
        pattern_summary = defaultdict(int)
        for issue in issues_so_far:
            pattern = issue.kind.lower()
            pattern_summary[pattern] += 1
        
        message_parts = ["\n---\n📊 **Progress Check:**"]
        message_parts.append(f"Detected {len(issues_so_far)} potential issues:")
        
        for pattern, count in pattern_summary.items():
            message_parts.append(f"  • {pattern.replace('_', ' ').title()}: {count}")
        
        message_parts.append("Consider reviewing and addressing these before continuing.\n---\n")
        
        return "\n".join(message_parts)