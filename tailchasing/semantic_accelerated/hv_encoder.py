"""
AST to hypervector encoder using Vector Symbolic Architecture (VSA).

This module provides deterministic encoding of Python AST nodes into
high-dimensional ternary vectors for efficient similarity computation.
"""

import ast
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass

# Constants
HV_DIM = 8192  # Hypervector dimensionality
TERNARY_VALUES = np.array([-1, 0, 1], dtype=np.int8)
SEED = 42  # Fixed seed for determinism


@dataclass
class EncodingConfig:
    """Configuration for hypervector encoding."""
    dimension: int = HV_DIM
    sparsity: float = 0.33  # Proportion of zero elements
    max_depth: int = 10  # Maximum AST depth to encode
    position_weight: float = 0.8  # Weight for position encoding
    normalize_ast: bool = True  # Whether to normalize AST nodes


class ASTNormalizer(ast.NodeTransformer):
    """
    Normalizes AST nodes to canonical form for similarity comparison.
    
    Transforms:
    - Name nodes → ID (except builtins)
    - Constants → type-based placeholders (NUM, STR, BOOL, etc.)  
    - Attributes → ATTR
    - Function/class names → canonical forms
    """
    
    BUILTINS = {
        'None', 'True', 'False', 'int', 'float', 'str', 'bool',
        'list', 'dict', 'set', 'tuple', 'print', 'len', 'range',
        'enumerate', 'zip', 'map', 'filter', 'sum', 'min', 'max'
    }
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize variable names while preserving builtins."""
        if node.id in self.BUILTINS:
            return node
        return ast.copy_location(ast.Name(id='ID', ctx=node.ctx), node)
    
    def visit_Constant(self, node: ast.Constant) -> ast.Name:
        """Replace constants with type placeholders."""
        value = node.value
        if isinstance(value, bool):
            placeholder = 'BOOL'
        elif isinstance(value, (int, float, complex)):
            placeholder = 'NUM'
        elif isinstance(value, str):
            placeholder = 'STR'
        elif value is None:
            placeholder = 'NONE'
        else:
            placeholder = 'CONST'
        return ast.copy_location(ast.Name(id=placeholder, ctx=ast.Load()), node)
    
    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Normalize attribute access."""
        self.generic_visit(node)
        return ast.copy_location(
            ast.Attribute(value=node.value, attr='ATTR', ctx=node.ctx), 
            node
        )
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize function definitions."""
        # Remove decorators and docstrings for similarity
        body = node.body
        if body and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    body = body[1:]  # Skip docstring
        
        normalized = ast.FunctionDef(
            name='FUNC',
            args=node.args,
            body=body or [ast.Pass()],
            decorator_list=[],
            returns=None,
            type_comment=None
        )
        self.generic_visit(normalized)
        return normalized
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.FunctionDef:
        """Normalize async functions to regular functions."""
        return self.visit_FunctionDef(ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment
        ))
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Normalize class definitions."""
        normalized = ast.ClassDef(
            name='CLASS',
            bases=[],
            keywords=[],
            body=node.body,
            decorator_list=[]
        )
        self.generic_visit(normalized)
        return normalized


class HypervectorEncoder:
    """
    Encodes Python AST nodes into high-dimensional ternary vectors.
    
    Uses Vector Symbolic Architecture (VSA) operations:
    - Binding: Element-wise multiplication for role-filler pairs
    - Bundling: Normalized addition for superposition
    - Permutation: Circular shift for sequence encoding
    """
    
    def __init__(self, config: Optional[EncodingConfig] = None):
        """
        Initialize the encoder with configuration.
        
        Args:
            config: Encoding configuration (uses defaults if None)
        """
        self.config = config or EncodingConfig()
        self.dim = self.config.dimension
        self.normalizer = ASTNormalizer() if self.config.normalize_ast else None
        
        # Initialize random generator with fixed seed
        self.rng = np.random.RandomState(SEED)
        
        # Cache for base vectors
        self._vector_cache: Dict[str, np.ndarray] = {}
        self._init_base_vectors()
    
    def _init_base_vectors(self) -> None:
        """Initialize base vectors for common AST node types."""
        node_types = [
            'Module', 'FunctionDef', 'ClassDef', 'Return', 'Delete',
            'Assign', 'AugAssign', 'For', 'While', 'If', 'With',
            'Raise', 'Try', 'Import', 'ImportFrom', 'Expr', 'Pass',
            'Break', 'Continue', 'BinOp', 'UnaryOp', 'BoolOp',
            'Compare', 'Call', 'Attribute', 'Subscript', 'Name',
            'Constant', 'List', 'Tuple', 'Set', 'Dict'
        ]
        
        for node_type in node_types:
            self._vector_cache[f'NODE_{node_type}'] = self._create_ternary_vector()
        
        # Position/depth seed vectors
        for i in range(self.config.max_depth):
            self._vector_cache[f'DEPTH_{i}'] = self._create_ternary_vector()
            self._vector_cache[f'POS_{i}'] = self._create_ternary_vector()
    
    def _create_ternary_vector(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Create a random ternary vector {-1, 0, +1}.
        
        Args:
            seed: Optional seed for this specific vector
            
        Returns:
            Ternary vector with specified sparsity
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng
        
        # Create sparse ternary vector
        vec = np.zeros(self.dim, dtype=np.int8)
        n_nonzero = int(self.dim * (1 - self.config.sparsity))
        nonzero_idx = rng.choice(self.dim, n_nonzero, replace=False)
        vec[nonzero_idx] = rng.choice([-1, 1], n_nonzero)
        
        return vec
    
    def _get_or_create_vector(self, key: str) -> np.ndarray:
        """Get cached vector or create new one."""
        if key not in self._vector_cache:
            # Use hash of key as seed for determinism
            seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            self._vector_cache[key] = self._create_ternary_vector(seed)
        return self._vector_cache[key]
    
    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Bind two vectors using element-wise multiplication.
        
        For ternary vectors: maintains ternary values
        """
        return np.sign(vec1 * vec2).astype(np.int8)
    
    def bundle(self, vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Bundle multiple vectors using weighted superposition.
        
        Args:
            vectors: List of vectors to bundle
            weights: Optional weights for each vector
            
        Returns:
            Bundled ternary vector
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.int8)
        
        if weights is None:
            weights = [1.0] * len(vectors)
        
        # Weighted sum
        result = np.zeros(self.dim, dtype=np.float32)
        for vec, weight in zip(vectors, weights):
            result += vec * weight
        
        # Threshold to ternary
        result = result / len(vectors)
        ternary = np.zeros(self.dim, dtype=np.int8)
        ternary[result > 0.33] = 1
        ternary[result < -0.33] = -1
        
        return ternary
    
    def permute(self, vec: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute vector by circular shift.
        
        Args:
            vec: Vector to permute
            shift: Number of positions to shift
            
        Returns:
            Permuted vector
        """
        return np.roll(vec, shift)
    
    def encode_ast(self, node: ast.AST, depth: int = 0, position: int = 0) -> np.ndarray:
        """
        Recursively encode AST node to hypervector.
        
        Args:
            node: AST node to encode
            depth: Current depth in AST
            position: Position among siblings
            
        Returns:
            Hypervector encoding of the AST node
        """
        if depth > self.config.max_depth:
            return np.zeros(self.dim, dtype=np.int8)
        
        # Normalize if configured
        if self.normalizer:
            node = self.normalizer.visit(node)
        
        # Get base vector for node type
        node_type = type(node).__name__
        base_vec = self._get_or_create_vector(f'NODE_{node_type}')
        
        # Add position encoding
        depth_vec = self._get_or_create_vector(f'DEPTH_{min(depth, self.config.max_depth-1)}')
        pos_vec = self._get_or_create_vector(f'POS_{position % 10}')
        
        # Bind with position information
        positioned = self.bind(base_vec, self.bind(depth_vec, pos_vec))
        
        # Encode node-specific attributes
        attr_vecs = []
        
        if isinstance(node, ast.Name):
            attr_vecs.append(self._get_or_create_vector(f'NAME_{node.id}'))
        elif isinstance(node, ast.Attribute):
            attr_vecs.append(self._get_or_create_vector(f'ATTR_{node.attr}'))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Encode argument structure
            if node.args:
                arg_vec = self._encode_arguments(node.args)
                attr_vecs.append(arg_vec)
        
        # Recursively encode children
        child_vecs = []
        for i, child in enumerate(ast.iter_child_nodes(node)):
            child_vec = self.encode_ast(child, depth + 1, i)
            # Weight by position to preserve order information
            weight = self.config.position_weight ** i
            child_vecs.append(child_vec * weight)
        
        # Bundle all components
        components = [positioned] + attr_vecs + child_vecs
        return self.bundle(components)
    
    def _encode_arguments(self, args: ast.arguments) -> np.ndarray:
        """Encode function arguments structure."""
        components = []
        
        # Regular arguments
        for i, arg in enumerate(args.args):
            arg_vec = self._get_or_create_vector(f'ARG_{i}')
            if arg.annotation:
                type_vec = self._get_or_create_vector(f'TYPE_{ast.dump(arg.annotation)}')
                arg_vec = self.bind(arg_vec, type_vec)
            components.append(arg_vec)
        
        # Keyword-only arguments
        for kwarg in args.kwonlyargs:
            components.append(self._get_or_create_vector(f'KWARG_{kwarg.arg}'))
        
        # Varargs and kwargs
        if args.vararg:
            components.append(self._get_or_create_vector('VARARG'))
        if args.kwarg:
            components.append(self._get_or_create_vector('KWARG'))
        
        return self.bundle(components) if components else np.zeros(self.dim, dtype=np.int8)
    
    def encode_function(self, func_node: ast.FunctionDef, 
                       context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Encode a function definition to hypervector.
        
        Args:
            func_node: Function AST node
            context: Optional context (imports, class info, etc.)
            
        Returns:
            Hypervector encoding of the function
        """
        # Base function encoding
        func_vec = self.encode_ast(func_node)
        
        # Add context if provided
        if context:
            context_vecs = []
            
            # Encode imports
            if 'imports' in context:
                for imp in context['imports']:
                    imp_vec = self._get_or_create_vector(f'IMPORT_{imp}')
                    context_vecs.append(imp_vec)
            
            # Encode class context
            if 'class_name' in context:
                class_vec = self._get_or_create_vector(f'CLASS_{context["class_name"]}')
                context_vecs.append(class_vec)
            
            if context_vecs:
                context_bundle = self.bundle(context_vecs)
                func_vec = self.bundle([func_vec, context_bundle], [0.8, 0.2])
        
        return func_vec
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.
        
        Args:
            vec1: First hypervector
            vec2: Second hypervector
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def hamming_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute Hamming similarity between two ternary vectors.
        
        Args:
            vec1: First hypervector
            vec2: Second hypervector
            
        Returns:
            Hamming similarity in range [0, 1]
        """
        return np.mean(vec1 == vec2)