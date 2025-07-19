"""
AST to hypervector encoder.

Extracts multi-channel features from Python functions and encodes them
as compositional hypervectors for semantic similarity detection.
"""

import ast
import re
from typing import List, Tuple, Set, Dict, Any, Optional
import numpy as np

from .hv_space import HVSpace


# Regex patterns for identifier splitting
CAMEL_CASE_PATTERN = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+')
SNAKE_CASE_PATTERN = re.compile(r'[a-z]+|[A-Z]+|\d+')
WORD_PATTERN = re.compile(r'\w+')

# Common Python keywords for control flow detection
CONTROL_KEYWORDS = {
    'if', 'elif', 'else', 'for', 'while', 'try', 'except',
    'finally', 'with', 'return', 'yield', 'raise', 'break',
    'continue', 'pass', 'assert'
}

# Threshold for considering a token meaningful
MIN_TOKEN_LENGTH = 2


def split_identifier(name: str) -> List[str]:
    """
    Split identifier into semantic tokens.
    
    Handles camelCase, snake_case, and CONSTANT_CASE.
    """
    if not name:
        return []
    
    # Try camelCase first
    parts = CAMEL_CASE_PATTERN.findall(name)
    if not parts:
        # Fall back to snake_case
        parts = name.split('_')
    
    # Clean and lowercase
    tokens = []
    for part in parts:
        part = part.strip().lower()
        if part and len(part) >= MIN_TOKEN_LENGTH:
            tokens.append(part)
    
    return tokens


def extract_docstring_tokens(docstring: str) -> List[str]:
    """Extract meaningful tokens from docstring."""
    if not docstring:
        return []
    
    # Extract all words
    words = WORD_PATTERN.findall(docstring.lower())
    
    # Filter short and common words
    tokens = [w for w in words if len(w) >= MIN_TOKEN_LENGTH]
    
    # Limit to prevent docstring domination
    return tokens[:50]


class FunctionFeatureExtractor(ast.NodeVisitor):
    """Extract multi-channel features from a function AST node."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset feature collections."""
        self.calls: Set[str] = set()
        self.control_flow: List[str] = []
        self.literals: List[str] = []
        self.imports: Set[str] = set()
        self.exceptions: Set[str] = set()
        self.decorators: List[str] = []
        self.comprehensions: int = 0
        self.nested_functions: int = 0
    
    def extract(self, node: ast.FunctionDef) -> Dict[str, List[str]]:
        """Extract all features from function."""
        self.reset()
        self.visit(node)
        
        features = {}
        
        # Name tokens
        name_tokens = split_identifier(node.name)
        if name_tokens:
            features['NAME_TOKENS'] = name_tokens
        
        # Arguments
        arg_tokens = []
        for arg in node.args.args:
            arg_tokens.extend(split_identifier(arg.arg))
        if arg_tokens:
            features['ARG_SIG'] = arg_tokens[:20]  # Limit
        
        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            doc_tokens = extract_docstring_tokens(docstring)
            if doc_tokens:
                features['DOC_TOKENS'] = doc_tokens
        
        # Calls
        if self.calls:
            features['CALLS'] = list(self.calls)[:50]
        
        # Control flow
        if self.control_flow:
            features['CONTROL'] = self.control_flow[:30]
        
        # Literals
        if self.literals:
            features['LITERALS'] = self.literals[:30]
        
        # Decorators
        if self.decorators:
            features['DECORATORS'] = self.decorators[:10]
        
        # Imports used
        if self.imports:
            features['IMPORTS'] = list(self.imports)[:20]
        
        # Exceptions
        if self.exceptions:
            features['EXCEPTIONS'] = list(self.exceptions)[:10]
        
        # Complexity indicators
        complexity_indicators = []
        if self.comprehensions > 0:
            complexity_indicators.append(f"comp_{self.comprehensions}")
        if self.nested_functions > 0:
            complexity_indicators.append(f"nested_{self.nested_functions}")
        if complexity_indicators:
            features['COMPLEXITY'] = complexity_indicators
        
        return features
    
    def visit_Call(self, node: ast.Call):
        """Extract function calls."""
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id.lower())
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr.lower())
            # Also track the object being called on
            if isinstance(node.func.value, ast.Name):
                self.calls.add(f"{node.func.value.id}.{node.func.attr}".lower())
        
        self.generic_visit(node)
    
    def visit_If(self, node: ast.If):
        self.control_flow.append('if')
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        self.control_flow.append('for')
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While):
        self.control_flow.append('while')
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try):
        self.control_flow.append('try')
        for handler in node.handlers:
            if handler.type:
                if isinstance(handler.type, ast.Name):
                    self.exceptions.add(handler.type.id)
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With):
        self.control_flow.append('with')
        self.generic_visit(node)
    
    def visit_Raise(self, node: ast.Raise):
        self.control_flow.append('raise')
        if node.exc and isinstance(node.exc, ast.Call):
            if isinstance(node.exc.func, ast.Name):
                self.exceptions.add(node.exc.func.id)
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return):
        self.control_flow.append('return')
        self.generic_visit(node)
    
    def visit_Yield(self, node: ast.Yield):
        self.control_flow.append('yield')
        self.generic_visit(node)
    
    def visit_YieldFrom(self, node: ast.YieldFrom):
        self.control_flow.append('yield_from')
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant):
        """Track literal types."""
        val = node.value
        if isinstance(val, bool):
            self.literals.append('bool')
        elif isinstance(val, int):
            self.literals.append('int')
        elif isinstance(val, float):
            self.literals.append('float')
        elif isinstance(val, str):
            if len(val) > 0:
                self.literals.append('str')
        elif val is None:
            self.literals.append('none')
        
        self.generic_visit(node)
    
    def visit_ListComp(self, node: ast.ListComp):
        self.comprehensions += 1
        self.generic_visit(node)
    
    def visit_SetComp(self, node: ast.SetComp):
        self.comprehensions += 1
        self.generic_visit(node)
    
    def visit_DictComp(self, node: ast.DictComp):
        self.comprehensions += 1
        self.generic_visit(node)
    
    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        self.comprehensions += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Don't recurse into nested functions, just count them
        self.nested_functions += 1
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.nested_functions += 1
    
    def visit_Lambda(self, node: ast.Lambda):
        self.nested_functions += 1
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)


def encode_function(
    fn: ast.FunctionDef,
    file: str,
    space: HVSpace,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Encode a function AST node as a hypervector.
    
    Args:
        fn: Function AST node
        file: Source file path
        space: Hypervector space
        config: Encoding configuration
    
    Returns:
        Compositional hypervector representing the function
    """
    # Extract features
    extractor = FunctionFeatureExtractor()
    features = extractor.extract(fn)
    
    # Get channel weights from config
    channel_weights = config.get('channel_weights', {})
    default_weight = 1.0
    
    # Encode each channel
    channel_vectors = []
    
    for channel, tokens in features.items():
        if not tokens:
            continue
        
        # Get weight for this channel
        weight = channel_weights.get(channel, default_weight)
        if weight <= 0:
            continue
        
        # Encode tokens
        token_hvs = []
        for token in tokens:
            token_hvs.append(space.token(token))
        
        # Bundle tokens for this channel
        channel_bundle = space.bundle(token_hvs)
        
        # Bind with role vector
        role_vec = space.role(channel)
        channel_vec = space.bind(role_vec, channel_bundle)
        
        # Apply weight by repeating (simple weighting scheme)
        repeat_count = max(1, int(weight))
        for _ in range(repeat_count):
            channel_vectors.append(channel_vec)
    
    # Handle empty function case
    if not channel_vectors:
        # Create a unique vector for empty function
        return space.token(f"EMPTY::{file}::{fn.name}")
    
    # Bundle all channel vectors
    return space.bundle(channel_vectors)


def extract_channels(fn: ast.FunctionDef, source: str) -> List[Tuple[str, List[str]]]:
    """
    Legacy interface for channel extraction.
    
    Returns list of (channel_name, tokens) tuples.
    """
    extractor = FunctionFeatureExtractor()
    features = extractor.extract(fn)
    return list(features.items())