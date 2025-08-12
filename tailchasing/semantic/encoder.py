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
    use_weighted_bundle = config.get('use_weighted_bundle', False)
    normalize_channels = config.get('normalize_channels', True)
    
    # Encode each channel
    channel_vectors = []
    channel_weight_pairs = []
    
    for channel, tokens in features.items():
        if not tokens:
            continue
        
        # Get weight for this channel
        weight = channel_weights.get(channel, default_weight)
        if weight <= 0:
            continue
        
        # Encode tokens with different strategies based on channel
        channel_vec = encode_channel(channel, tokens, space, config)
        
        if use_weighted_bundle:
            channel_weight_pairs.append((channel_vec, weight))
        else:
            # Apply weight by repeating (simple weighting scheme)
            repeat_count = max(1, int(weight * 5))  # Scale factor for weights
            for _ in range(repeat_count):
                channel_vectors.append(channel_vec)
    
    # Handle empty function case
    if not channel_vectors and not channel_weight_pairs:
        # Create a unique vector for empty function
        return space.token(f"EMPTY::{file}::{fn.name}")
    
    # Bundle all channel vectors
    if use_weighted_bundle and channel_weight_pairs:
        result = space.weighted_bundle(channel_weight_pairs)
    else:
        result = space.bundle(channel_vectors)
    
    # Optional normalization
    if normalize_channels:
        result = space.threshold_cleanup(result)
    
    return result


def encode_channel(
    channel: str, 
    tokens: List[str], 
    space: HVSpace, 
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Encode a specific channel with appropriate strategy.
    
    Args:
        channel: Channel name
        tokens: List of tokens for this channel
        space: Hypervector space
        config: Encoding configuration
    
    Returns:
        Encoded channel vector
    """
    if not tokens:
        return space._rand_vec()
    
    # Choose encoding strategy based on channel type
    if channel in ['NAME_TOKENS', 'ARG_SIG']:
        # Use sequence encoding for structured tokens
        use_sequence = config.get('use_sequence_encoding', False)
        if use_sequence:
            channel_vec = space.create_sequence_vector(tokens, use_permutation=True)
        else:
            token_hvs = [space.token(token) for token in tokens]
            channel_vec = space.bundle(token_hvs)
    
    elif channel == 'CONTROL':
        # For control flow, order matters
        channel_vec = space.create_sequence_vector(tokens, use_permutation=True)
    
    elif channel == 'DOC_TOKENS':
        # For documentation, use frequency weighting
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        weighted_pairs = []
        for token, count in token_counts.items():
            weight = min(3.0, count)  # Cap frequency weight
            weighted_pairs.append((space.token(token), weight))
        
        if weighted_pairs:
            channel_vec = space.weighted_bundle(weighted_pairs)
        else:
            channel_vec = space._rand_vec()
    
    elif channel in ['CALLS', 'IMPORTS', 'EXCEPTIONS']:
        # For these channels, treat as sets (no order, but unique items)
        unique_tokens = list(set(tokens))
        token_hvs = [space.token(token) for token in unique_tokens]
        channel_vec = space.bundle(token_hvs)
    
    elif channel == 'LITERALS':
        # For literals, group by type
        type_groups = {}
        for token in tokens:
            type_groups.setdefault(token, []).append(token)
        
        type_vecs = []
        for lit_type, instances in type_groups.items():
            type_vec = space.token(lit_type)
            count_vec = space.token(f"count_{len(instances)}")
            bound_vec = space.bind(type_vec, count_vec)
            type_vecs.append(bound_vec)
        
        channel_vec = space.bundle(type_vecs) if type_vecs else space._rand_vec()
    
    else:
        # Default encoding: simple bundle
        token_hvs = [space.token(token) for token in tokens]
        channel_vec = space.bundle(token_hvs)
    
    # Bind with role vector
    role_vec = space.role(channel)
    return space.bind(role_vec, channel_vec)


def encode_function_with_context(
    fn: ast.FunctionDef,
    file: str,
    space: HVSpace,
    config: Dict[str, Any],
    class_context: Optional[str] = None,
    module_imports: Optional[List[str]] = None
) -> np.ndarray:
    """
    Encode a function with additional context information.
    
    Args:
        fn: Function AST node
        file: Source file path
        space: Hypervector space
        config: Encoding configuration
        class_context: Name of containing class (if any)
        module_imports: List of module imports
    
    Returns:
        Contextual hypervector representing the function
    """
    # Get base function encoding
    base_vec = encode_function(fn, file, space, config)
    
    # Add context vectors
    context_vecs = [base_vec]
    
    # File context
    file_role = space.role('FILE_CONTEXT')
    file_token = space.token(file.split('/')[-1].replace('.py', ''))
    file_vec = space.bind(file_role, file_token)
    context_vecs.append(file_vec)
    
    # Class context
    if class_context:
        class_role = space.role('CLASS_CONTEXT')
        class_tokens = split_identifier(class_context)
        if class_tokens:
            class_bundle = space.bundle([space.token(token) for token in class_tokens])
            class_vec = space.bind(class_role, class_bundle)
            context_vecs.append(class_vec)
    
    # Import context
    if module_imports:
        import_role = space.role('IMPORT_CONTEXT')
        # Limit to most relevant imports
        relevant_imports = module_imports[:10]
        import_tokens = [space.token(imp) for imp in relevant_imports]
        import_bundle = space.bundle(import_tokens)
        import_vec = space.bind(import_role, import_bundle)
        context_vecs.append(import_vec)
    
    # Function signature context
    sig_context = encode_signature_context(fn, space)
    if sig_context is not None:
        context_vecs.append(sig_context)
    
    return space.bundle(context_vecs)


def encode_signature_context(fn: ast.FunctionDef, space: HVSpace) -> Optional[np.ndarray]:
    """
    Encode function signature as contextual information.
    
    Args:
        fn: Function AST node
        space: Hypervector space
    
    Returns:
        Signature context vector or None
    """
    sig_elements = []
    
    # Argument count
    arg_count = len(fn.args.args)
    sig_elements.append(f"args_{arg_count}")
    
    # Default arguments
    default_count = len(fn.args.defaults)
    if default_count > 0:
        sig_elements.append(f"defaults_{default_count}")
    
    # Varargs and kwargs
    if fn.args.vararg:
        sig_elements.append("varargs")
    if fn.args.kwarg:
        sig_elements.append("kwargs")
    
    # Return annotation
    if fn.returns:
        sig_elements.append("typed_return")
    
    # Decorators
    if fn.decorator_list:
        sig_elements.append(f"decorators_{len(fn.decorator_list)}")
    
    if not sig_elements:
        return None
    
    sig_role = space.role('SIGNATURE_CONTEXT')
    sig_tokens = [space.token(elem) for elem in sig_elements]
    sig_bundle = space.bundle(sig_tokens)
    
    return space.bind(sig_role, sig_bundle)


def batch_encode_functions(
    functions: List[Tuple[ast.FunctionDef, str]], 
    space: HVSpace,
    config: Dict[str, Any]
) -> List[np.ndarray]:
    """
    Efficiently encode a batch of functions.
    
    Args:
        functions: List of (function_node, file_path) tuples
        space: Hypervector space
        config: Encoding configuration
    
    Returns:
        List of encoded function vectors
    """
    encoded_functions = []
    
    # Pre-warm token cache with common tokens
    if config.get('prewarm_cache', False):
        prewarm_token_cache(functions, space)
    
    for fn, file_path in functions:
        try:
            encoded_vec = encode_function(fn, file_path, space, config)
            encoded_functions.append(encoded_vec)
        except Exception as e:
            # Create error vector for failed encodings
            error_vec = space.token(f"ERROR::{file_path}::{fn.name}")
            encoded_functions.append(error_vec)
    
    return encoded_functions


def prewarm_token_cache(
    functions: List[Tuple[ast.FunctionDef, str]], 
    space: HVSpace
):
    """
    Pre-warm the token cache with common tokens from functions.
    
    Args:
        functions: List of (function_node, file_path) tuples
        space: Hypervector space
    """
    token_counts = {}
    extractor = FunctionFeatureExtractor()
    
    # Sample a subset of functions for pre-warming
    sample_size = min(100, len(functions))
    sample_functions = functions[:sample_size]
    
    for fn, _ in sample_functions:
        features = extractor.extract(fn)
        for channel, tokens in features.items():
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
    
    # Pre-generate vectors for frequent tokens
    frequent_tokens = [
        token for token, count in token_counts.items() 
        if count >= 3  # Appear in at least 3 functions
    ]
    
    for token in frequent_tokens:
        space.token(token)  # This caches the vector


def extract_channels(fn: ast.FunctionDef, source: str) -> List[Tuple[str, List[str]]]:
    """
    Legacy interface for channel extraction.
    
    Returns list of (channel_name, tokens) tuples.
    """
    extractor = FunctionFeatureExtractor()
    features = extractor.extract(fn)
    return list(features.items())