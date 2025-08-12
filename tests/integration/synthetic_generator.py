"""
Synthetic codebase generator for stress testing.

Generates large codebases with controlled amounts of tail-chasing patterns
for testing scalability and performance of the detection system.
"""

import ast
import random
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import string
import textwrap


class PatternType(Enum):
    """Types of tail-chasing patterns to inject."""
    DUPLICATE_FUNCTIONS = "duplicate_functions"
    PHANTOM_FUNCTIONS = "phantom_functions"
    CONTEXT_THRASHING = "context_thrashing"
    HALLUCINATION_CASCADE = "hallucination_cascade"
    CIRCULAR_IMPORTS = "circular_imports"
    CLEAN_CODE = "clean_code"  # No patterns


@dataclass
class GenerationConfig:
    """Configuration for synthetic codebase generation."""
    
    # Overall structure
    num_files: int = 50
    functions_per_file: int = 20
    classes_per_file: int = 3
    
    # Pattern injection rates (0.0 to 1.0)
    duplicate_function_rate: float = 0.1
    phantom_function_rate: float = 0.15
    context_thrashing_rate: float = 0.08
    hallucination_cascade_rate: float = 0.05
    circular_import_rate: float = 0.03
    
    # Complexity parameters
    max_function_length: int = 50
    max_nesting_depth: int = 4
    imports_per_file: int = 8
    
    # Naming conventions
    module_name_prefix: str = "synthetic"
    base_seed: int = 42


class NameGenerator:
    """Generates realistic but synthetic names for code elements."""
    
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        
        # Common programming concepts for name generation
        self.nouns = [
            'data', 'info', 'result', 'value', 'item', 'element', 'object',
            'record', 'entry', 'node', 'entity', 'model', 'service', 'handler',
            'manager', 'processor', 'analyzer', 'validator', 'generator',
            'calculator', 'formatter', 'parser', 'builder', 'factory',
            'strategy', 'policy', 'rule', 'config', 'setting', 'parameter'
        ]
        
        self.verbs = [
            'get', 'set', 'create', 'build', 'make', 'generate', 'produce',
            'process', 'handle', 'manage', 'analyze', 'validate', 'verify',
            'check', 'test', 'run', 'execute', 'perform', 'calculate',
            'compute', 'format', 'parse', 'convert', 'transform', 'filter',
            'sort', 'search', 'find', 'load', 'save', 'store', 'retrieve'
        ]
        
        self.adjectives = [
            'primary', 'secondary', 'main', 'base', 'core', 'basic', 'simple',
            'complex', 'advanced', 'enhanced', 'extended', 'improved',
            'optimized', 'efficient', 'fast', 'slow', 'safe', 'secure',
            'public', 'private', 'internal', 'external', 'local', 'global',
            'temporary', 'permanent', 'active', 'inactive', 'valid', 'invalid'
        ]
        
        self.domains = [
            'user', 'account', 'profile', 'session', 'auth', 'permission',
            'product', 'order', 'payment', 'shipping', 'inventory', 'catalog',
            'customer', 'vendor', 'supplier', 'partner', 'client', 'server',
            'database', 'cache', 'queue', 'job', 'task', 'event', 'message',
            'report', 'analytics', 'metrics', 'stats', 'log', 'audit',
            'config', 'setting', 'preference', 'option', 'feature', 'module'
        ]
    
    def function_name(self, context: str = "general") -> str:
        """Generate a realistic function name."""
        patterns = [
            lambda: f"{self.random.choice(self.verbs)}_{self.random.choice(self.nouns)}",
            lambda: f"{self.random.choice(self.verbs)}_{self.random.choice(self.adjectives)}_{self.random.choice(self.nouns)}",
            lambda: f"{self.random.choice(self.domains)}_{self.random.choice(self.verbs)}_{self.random.choice(self.nouns)}",
            lambda: f"is_{self.random.choice(self.adjectives)}_{self.random.choice(self.nouns)}",
            lambda: f"has_{self.random.choice(self.adjectives)}_{self.random.choice(self.nouns)}",
        ]
        
        return self.random.choice(patterns)()
    
    def class_name(self, context: str = "general") -> str:
        """Generate a realistic class name."""
        patterns = [
            lambda: f"{self.random.choice(self.adjectives).title()}{self.random.choice(self.nouns).title()}",
            lambda: f"{self.random.choice(self.domains).title()}{self.random.choice(self.nouns).title()}",
            lambda: f"{self.random.choice(self.nouns).title()}{self.random.choice(['Manager', 'Handler', 'Service', 'Processor'])}",
        ]
        
        return self.random.choice(patterns)()
    
    def variable_name(self) -> str:
        """Generate a realistic variable name."""
        patterns = [
            lambda: self.random.choice(self.nouns),
            lambda: f"{self.random.choice(self.adjectives)}_{self.random.choice(self.nouns)}",
            lambda: f"{self.random.choice(self.domains)}_{self.random.choice(self.nouns)}",
        ]
        
        return self.random.choice(patterns)()
    
    def module_name(self, prefix: str = "synthetic") -> str:
        """Generate a realistic module name."""
        suffix = self.random.choice(self.domains + self.nouns)
        return f"{prefix}_{suffix}"


class CodeTemplateGenerator:
    """Generates code templates with various patterns and structures."""
    
    def __init__(self, name_generator: NameGenerator):
        self.name_gen = name_generator
        self.random = name_generator.random
    
    def generate_function(self, pattern_type: PatternType = PatternType.CLEAN_CODE, 
                         complexity: int = 3) -> str:
        """Generate a function with specified pattern and complexity."""
        
        if pattern_type == PatternType.DUPLICATE_FUNCTIONS:
            return self._generate_duplicate_function_template()
        elif pattern_type == PatternType.PHANTOM_FUNCTIONS:
            return self._generate_phantom_function()
        elif pattern_type == PatternType.CONTEXT_THRASHING:
            return self._generate_context_thrashing_function()
        else:
            return self._generate_clean_function(complexity)
    
    def _generate_clean_function(self, complexity: int) -> str:
        """Generate a clean, well-implemented function."""
        func_name = self.name_gen.function_name()
        
        # Generate parameters
        param_count = min(complexity, 5)
        params = [self.name_gen.variable_name() for _ in range(param_count)]
        param_str = ", ".join(params)
        
        # Generate function body based on complexity
        if complexity <= 2:
            body = self._generate_simple_function_body(params)
        elif complexity <= 4:
            body = self._generate_medium_function_body(params)
        else:
            body = self._generate_complex_function_body(params)
        
        return f'''
def {func_name}({param_str}):
    """Generated function: {func_name}."""
{textwrap.indent(body, "    ")}
        '''.strip()
    
    def _generate_simple_function_body(self, params: List[str]) -> str:
        """Generate simple function body."""
        if not params:
            return "return True"
        
        templates = [
            f"return {params[0]}",
            f"return {params[0]} if {params[0]} else None",
            f"return len({params[0]}) if {params[0]} else 0",
            f"result = {params[0]}\nreturn result",
        ]
        
        return self.random.choice(templates)
    
    def _generate_medium_function_body(self, params: List[str]) -> str:
        """Generate medium complexity function body."""
        var_name = self.name_gen.variable_name()
        
        templates = [
            f"""
{var_name} = []
for item in {params[0] if params else 'items'}:
    if item:
        {var_name}.append(item)
return {var_name}
            """.strip(),
            
            f"""
if not {params[0] if params else 'data'}:
    return None

{var_name} = 0
for value in {params[0] if params else 'data'}:
    if isinstance(value, (int, float)):
        {var_name} += value

return {var_name}
            """.strip(),
            
            f"""
{var_name} = {{}}
for key, value in {params[0] if params else 'items'}.items():
    if value is not None:
        {var_name}[key] = str(value).upper()

return {var_name}
            """.strip()
        ]
        
        return self.random.choice(templates)
    
    def _generate_complex_function_body(self, params: List[str]) -> str:
        """Generate complex function body with multiple operations."""
        var1 = self.name_gen.variable_name()
        var2 = self.name_gen.variable_name()
        var3 = self.name_gen.variable_name()
        
        return f"""
{var1} = []
{var2} = {{}}
{var3} = 0

for item in {params[0] if params else 'items'}:
    if not item:
        continue
    
    key = str(item.get('key', 'default'))
    value = item.get('value', 0)
    
    if key not in {var2}:
        {var2}[key] = []
    
    {var2}[key].append(value)
    {var1}.append({{
        'processed_key': key,
        'processed_value': value * 2,
        'timestamp': '{var3}'
    }})
    
    {var3} += 1
    
    if {var3} % 10 == 0:
        # Process batch
        pass

return {{
    'results': {var1},
    'grouped': {var2},
    'count': {var3}
}}
        """.strip()
    
    def _generate_duplicate_function_template(self) -> str:
        """Generate a function template that will be duplicated."""
        func_name_base = self.name_gen.function_name()
        
        # Generate a function that will be copied with minor variations
        template = f'''
def {func_name_base}(input_data):
    """Process input data with standard algorithm."""
    if not input_data:
        return None
    
    processed_items = []
    total_count = 0
    
    for item in input_data:
        if item.get('active', False):
            processed_value = item.get('value', 0) * 2
            processed_items.append({{
                'id': item.get('id'),
                'processed_value': processed_value,
                'status': 'processed'
            }})
            total_count += 1
    
    return {{
        'items': processed_items,
        'total_count': total_count,
        'status': 'completed'
    }}
        '''
        
        return template.strip()
    
    def _generate_phantom_function(self) -> str:
        """Generate a phantom/placeholder function."""
        func_name = self.name_gen.function_name()
        
        phantom_types = [
            f'''
def {func_name}():
    """This function is not implemented yet."""
    pass
            '''.strip(),
            
            f'''
def {func_name}():
    """TODO: Implement this function."""
    # Implementation pending
    raise NotImplementedError("Function not implemented")
            '''.strip(),
            
            f'''
def {func_name}():
    """Placeholder function."""
    ...
            '''.strip(),
            
            f'''
def {func_name}():
    """Function stub - implement later."""
    # TODO: Add implementation
    return None
            '''.strip()
        ]
        
        return self.random.choice(phantom_types)
    
    def _generate_context_thrashing_function(self) -> str:
        """Generate function that will contribute to context thrashing."""
        func_name = self.name_gen.function_name()
        
        # These functions are similar but different enough to cause thrashing
        templates = [
            f'''
def {func_name}_version_a(data):
    """Process data using method A."""
    # Validation
    if not data:
        return {{'error': 'Data is required'}}
    
    # Processing logic
    results = []
    for item in data:
        if item.get('status') == 'active':
            processed = {{
                'id': item['id'],
                'value': item.get('value', 0) * 1.5,
                'method': 'A'
            }}
            results.append(processed)
    
    return {{'results': results, 'method': 'A'}}
            '''.strip(),
            
            f'''
def {func_name}_version_b(input_data):
    """Process input using method B."""
    # Input validation
    if not input_data:
        return {{'error': 'Input data is required'}}
    
    # Different processing approach
    output = []
    for record in input_data:
        if record.get('status') == 'active':
            transformed = {{
                'identifier': record['id'],
                'computed_value': record.get('value', 0) * 1.5,
                'processing_method': 'B'
            }}
            output.append(transformed)
    
    return {{'output': output, 'processing_method': 'B'}}
            '''.strip()
        ]
        
        return self.random.choice(templates)
    
    def generate_class(self, pattern_type: PatternType = PatternType.CLEAN_CODE) -> str:
        """Generate a class with specified pattern."""
        class_name = self.name_gen.class_name()
        
        if pattern_type == PatternType.HALLUCINATION_CASCADE:
            return self._generate_fictional_class(class_name)
        else:
            return self._generate_clean_class(class_name)
    
    def _generate_clean_class(self, class_name: str) -> str:
        """Generate a clean, well-implemented class."""
        return f'''
class {class_name}:
    """Generated class: {class_name}."""
    
    def __init__(self):
        self.{self.name_gen.variable_name()} = {{}}
        self.{self.name_gen.variable_name()} = []
        self._initialized = True
    
    def {self.name_gen.function_name()}(self, data):
        """Process data method."""
        if not data:
            return None
        
        return {{
            'processed': True,
            'data': data,
            'timestamp': time.time() if 'time' in globals() else 0
        }}
    
    def {self.name_gen.function_name()}(self):
        """Status check method."""
        return self._initialized
        '''.strip()
    
    def _generate_fictional_class(self, class_name: str) -> str:
        """Generate a fictional class that's part of a hallucination cascade."""
        dependency1 = self.name_gen.class_name()
        dependency2 = self.name_gen.class_name()
        
        return f'''
class {class_name}:
    """Advanced {class_name.lower()} with integrated subsystems."""
    
    def __init__(self):
        self.{dependency1.lower()}_manager = {dependency1}Manager()
        self.{dependency2.lower()}_processor = {dependency2}Processor()
        self.orchestration_engine = OrchestrationEngine()
        self.telemetry_collector = TelemetryCollector()
    
    def execute_advanced_processing(self, request):
        """Execute processing through the integrated subsystem."""
        # Route through orchestration engine
        orchestrated_request = self.orchestration_engine.prepare_request(request)
        
        # Process through manager
        managed_data = self.{dependency1.lower()}_manager.handle_advanced_operation(orchestrated_request)
        
        # Apply processor transformations
        processed_result = self.{dependency2.lower()}_processor.transform_with_analytics(managed_data)
        
        # Collect telemetry
        self.telemetry_collector.record_processing_metrics(processed_result)
        
        return processed_result
    
    def synchronize_subsystems(self):
        """Synchronize all integrated subsystems."""
        sync_state = self.orchestration_engine.get_synchronization_state()
        
        self.{dependency1.lower()}_manager.apply_sync_state(sync_state)
        self.{dependency2.lower()}_processor.align_with_orchestration(sync_state)
        
        return sync_state.is_synchronized
        '''.strip()


class SyntheticCodebaseGenerator:
    """Main generator for creating synthetic codebases."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.name_gen = NameGenerator(config.base_seed)
        self.template_gen = CodeTemplateGenerator(self.name_gen)
        
        # Track generated patterns for validation
        self.generated_patterns = {
            PatternType.DUPLICATE_FUNCTIONS: [],
            PatternType.PHANTOM_FUNCTIONS: [],
            PatternType.CONTEXT_THRASHING: [],
            PatternType.HALLUCINATION_CASCADE: [],
            PatternType.CIRCULAR_IMPORTS: []
        }
    
    def generate_codebase(self, output_dir: Optional[Path] = None) -> Path:
        """Generate complete synthetic codebase."""
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="synthetic_codebase_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating synthetic codebase in: {output_dir}")
        
        # Generate files
        for file_idx in range(self.config.num_files):
            file_path = output_dir / f"{self.config.module_name_prefix}_{file_idx:03d}.py"
            self._generate_file(file_path, file_idx)
        
        # Generate __init__.py
        init_file = output_dir / "__init__.py"
        with open(init_file, 'w') as f:
            f.write(f'"""Synthetic codebase generated for testing."""\n')
        
        # Generate summary
        self._generate_summary(output_dir)
        
        return output_dir
    
    def _generate_file(self, file_path: Path, file_idx: int):
        """Generate a single Python file."""
        imports = self._generate_imports()
        classes = []
        functions = []
        
        # Generate classes
        for class_idx in range(self.config.classes_per_file):
            pattern_type = self._select_pattern_type()
            class_code = self.template_gen.generate_class(pattern_type)
            classes.append(class_code)
            
            if pattern_type != PatternType.CLEAN_CODE:
                self.generated_patterns[pattern_type].append(
                    f"{file_path.name}:class_{class_idx}"
                )
        
        # Generate functions
        for func_idx in range(self.config.functions_per_file):
            pattern_type = self._select_pattern_type()
            complexity = self.name_gen.random.randint(1, 5)
            
            if pattern_type == PatternType.DUPLICATE_FUNCTIONS:
                # Generate original and duplicate
                original_func = self.template_gen.generate_function(pattern_type, complexity)
                duplicate_func = self._create_duplicate_variation(original_func)
                functions.extend([original_func, duplicate_func])
                
                self.generated_patterns[pattern_type].append(
                    f"{file_path.name}:func_{func_idx}"
                )
            else:
                func_code = self.template_gen.generate_function(pattern_type, complexity)
                functions.append(func_code)
                
                if pattern_type != PatternType.CLEAN_CODE:
                    self.generated_patterns[pattern_type].append(
                        f"{file_path.name}:func_{func_idx}"
                    )
        
        # Write file
        with open(file_path, 'w') as f:
            f.write(f'"""Synthetic module: {file_path.name}"""\n\n')
            f.write('\n'.join(imports))
            f.write('\n\n')
            f.write('\n\n\n'.join(classes))
            f.write('\n\n\n')
            f.write('\n\n\n'.join(functions))
            f.write('\n')
    
    def _generate_imports(self) -> List[str]:
        """Generate realistic import statements."""
        standard_imports = [
            'import os',
            'import sys', 
            'import time',
            'import json',
            'from typing import Dict, List, Optional, Any',
            'from datetime import datetime',
            'from dataclasses import dataclass',
        ]
        
        # Add some imports randomly
        selected_imports = self.name_gen.random.sample(
            standard_imports, 
            min(self.config.imports_per_file, len(standard_imports))
        )
        
        return selected_imports
    
    def _select_pattern_type(self) -> PatternType:
        """Select pattern type based on configured rates."""
        rand_val = self.name_gen.random.random()
        
        if rand_val < self.config.duplicate_function_rate:
            return PatternType.DUPLICATE_FUNCTIONS
        elif rand_val < self.config.duplicate_function_rate + self.config.phantom_function_rate:
            return PatternType.PHANTOM_FUNCTIONS
        elif rand_val < (self.config.duplicate_function_rate + 
                         self.config.phantom_function_rate + 
                         self.config.context_thrashing_rate):
            return PatternType.CONTEXT_THRASHING
        elif rand_val < (self.config.duplicate_function_rate + 
                         self.config.phantom_function_rate + 
                         self.config.context_thrashing_rate + 
                         self.config.hallucination_cascade_rate):
            return PatternType.HALLUCINATION_CASCADE
        else:
            return PatternType.CLEAN_CODE
    
    def _create_duplicate_variation(self, original_func: str) -> str:
        """Create a duplicate variation of a function."""
        # Simple variations: change variable names and comments
        variations = {
            'input_data': 'data_input',
            'processed_items': 'item_list',
            'total_count': 'count_total',
            'item': 'element',
            'processed_value': 'value_processed',
            'active': 'is_active',
            'standard algorithm': 'standard processing',
            'Process input data': 'Handle input data',
        }
        
        duplicate = original_func
        for old, new in variations.items():
            duplicate = duplicate.replace(old, new)
        
        # Change function name slightly
        if 'def ' in duplicate:
            func_name_start = duplicate.find('def ') + 4
            func_name_end = duplicate.find('(', func_name_start)
            original_name = duplicate[func_name_start:func_name_end]
            new_name = f"{original_name}_alt"
            duplicate = duplicate.replace(f"def {original_name}(", f"def {new_name}(")
        
        return duplicate
    
    def _generate_summary(self, output_dir: Path):
        """Generate summary of patterns injected."""
        summary = {
            'config': {
                'num_files': self.config.num_files,
                'functions_per_file': self.config.functions_per_file,
                'classes_per_file': self.config.classes_per_file,
                'pattern_rates': {
                    'duplicate_functions': self.config.duplicate_function_rate,
                    'phantom_functions': self.config.phantom_function_rate,
                    'context_thrashing': self.config.context_thrashing_rate,
                    'hallucination_cascade': self.config.hallucination_cascade_rate,
                }
            },
            'generated_patterns': {
                pattern_type.value: len(locations) 
                for pattern_type, locations in self.generated_patterns.items()
            },
            'pattern_locations': {
                pattern_type.value: locations
                for pattern_type, locations in self.generated_patterns.items()
            }
        }
        
        summary_file = output_dir / "generation_summary.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"Generated codebase summary:")
        print(f"  Files: {self.config.num_files}")
        print(f"  Total functions: ~{self.config.num_files * self.config.functions_per_file}")
        print(f"  Total classes: ~{self.config.num_files * self.config.classes_per_file}")
        for pattern_type, count in summary['generated_patterns'].items():
            if count > 0:
                print(f"  {pattern_type.replace('_', ' ').title()}: {count}")


def create_small_test_codebase() -> Path:
    """Create a small synthetic codebase for testing."""
    config = GenerationConfig(
        num_files=5,
        functions_per_file=10,
        classes_per_file=2,
        duplicate_function_rate=0.2,
        phantom_function_rate=0.3,
        context_thrashing_rate=0.1,
        hallucination_cascade_rate=0.1
    )
    
    generator = SyntheticCodebaseGenerator(config)
    return generator.generate_codebase()


def create_medium_test_codebase() -> Path:
    """Create a medium synthetic codebase for performance testing."""
    config = GenerationConfig(
        num_files=25,
        functions_per_file=20,
        classes_per_file=4,
        duplicate_function_rate=0.15,
        phantom_function_rate=0.2,
        context_thrashing_rate=0.1,
        hallucination_cascade_rate=0.05
    )
    
    generator = SyntheticCodebaseGenerator(config)
    return generator.generate_codebase()


def create_large_stress_test_codebase() -> Path:
    """Create a large synthetic codebase for stress testing."""
    config = GenerationConfig(
        num_files=100,
        functions_per_file=50,
        classes_per_file=8,
        duplicate_function_rate=0.12,
        phantom_function_rate=0.18,
        context_thrashing_rate=0.08,
        hallucination_cascade_rate=0.04
    )
    
    generator = SyntheticCodebaseGenerator(config)
    return generator.generate_codebase()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        size = sys.argv[1].lower()
        if size == "small":
            codebase_path = create_small_test_codebase()
        elif size == "medium":
            codebase_path = create_medium_test_codebase()
        elif size == "large":
            codebase_path = create_large_stress_test_codebase()
        else:
            print("Usage: python synthetic_generator.py [small|medium|large]")
            sys.exit(1)
    else:
        codebase_path = create_small_test_codebase()
    
    print(f"\nSynthetic codebase generated at: {codebase_path}")
    print("Use this for testing the tail-chasing detection system.")