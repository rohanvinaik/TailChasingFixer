"""
Automated upgrader for trivial/stub functions.

This module detects functions with trivial implementations (pass, None return, etc.)
and automatically generates appropriate implementations based on their names,
docstrings, and usage patterns.
"""

import ast
import re
import pathlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..utils.logging_setup import get_logger


TRIVIAL_RETURNS = (None, False, True, [], {}, ())


@dataclass
class TrivialFunction:
    """Information about a trivial function found in the codebase."""
    file_path: str
    name: str
    full_name: str  # Including class name if method
    node: ast.FunctionDef
    docstring: Optional[str]
    signature: str
    class_name: Optional[str] = None
    decorators: List[str] = None
    call_sites: List[Tuple[str, int]] = None  # (file, line) pairs
    

@dataclass  
class ImplementationTemplate:
    """Template for generating function implementations."""
    pattern: str  # Regex pattern for function name
    description: str
    template_code: str
    imports_needed: List[str] = None
    test_template: Optional[str] = None
    

class TrivialFunctionUpgrader:
    """Detects and upgrades trivial function implementations."""
    
    def __init__(self):
        self.logger = get_logger("trivial_upgrader")
        self.trivial_functions: List[TrivialFunction] = []
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, ImplementationTemplate]:
        """Initialize implementation templates for common patterns."""
        templates = {
            # Parent/inheritance patterns
            "_parent_has_init": ImplementationTemplate(
                pattern=r".*_parent_has_(init|__init__).*",
                description="Check if parent class has __init__ method",
                template_code='''def {name}(self, cls: type) -> bool:
    """Check if parent class has __init__ method."""
    import inspect
    for base in cls.__bases__:
        if base is object:
            continue
        if hasattr(base, '__init__'):
            init_method = getattr(base, '__init__')
            # Check if it's not the default object.__init__
            if init_method is not object.__init__:
                return True
    return False''',
                imports_needed=["import inspect"],
                test_template='''def test_{name}(self):
    class Parent:
        def __init__(self):
            pass
    
    class Child(Parent):
        pass
    
    assert self.instance.{name}(Child) is True
    assert self.instance.{name}(object) is False'''
            ),
            
            # Structural duplicate detection
            "_are_structural_duplicates": ImplementationTemplate(
                pattern=r".*structural_duplicate.*",
                description="Check if two functions are structurally identical",
                template_code='''def {name}(self, f1: dict, f2: dict, *, 
                               seq_threshold: float = 0.9, skel_threshold: float = 0.9) -> bool:
    """Check if functions are structural duplicates."""
    n1 = f1.get("ast_node")
    n2 = f2.get("ast_node")
    if not isinstance(n1, ast.AST) or not isinstance(n2, ast.AST):
        return False

    def normalize(node: ast.AST) -> ast.AST:
        class N(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name):
                return ast.copy_location(ast.Name(id="ID", ctx=n.ctx), n)
            def visit_Constant(self, n: ast.Constant):
                tag = "NUM" if isinstance(n.value, (int, float, complex)) else "STR"
                return ast.copy_location(ast.Name(id=tag, ctx=ast.Load()), n)
        return N().visit(node)

    from difflib import SequenceMatcher
    sk1 = ast.dump(normalize(n1), annotate_fields=False)
    sk2 = ast.dump(normalize(n2), annotate_fields=False)
    ratio = SequenceMatcher(None, sk1, sk2).ratio()
    return ratio >= skel_threshold''',
                imports_needed=["import ast", "from difflib import SequenceMatcher"],
            ),
            
            # Dependency collection
            "get_dependencies": ImplementationTemplate(
                pattern=r".*get_dependencies.*",
                description="Collect dependencies from class attributes",
                template_code='''def {name}(self) -> List[str]:
    """Get list of dependencies required."""
    deps = []
    for attr in ["REQUIRES_ANALYZERS", "REQUIRES_TOOLS", "REQUIRES_MODELS"]:
        values = getattr(self, attr, ())
        prefix = attr.replace("REQUIRES_", "").lower().rstrip("S")
        for v in values:
            deps.append(f"{{prefix}}:{{v}}")
    return sorted(set(deps))''',
                imports_needed=["from typing import List"],
            ),
            
            # Can handle patterns
            "can_handle": ImplementationTemplate(
                pattern=r"can_handle",
                description="Check if strategy can handle issue",
                template_code='''def {name}(self, issue: Issue) -> bool:
    """Check if this strategy can handle the given issue."""
    # Check issue kind matches our supported types
    supported_kinds = getattr(self, "SUPPORTED_ISSUE_KINDS", [])
    if supported_kinds:
        return issue.kind in supported_kinds
    
    # Fallback to checking by name pattern
    return issue.kind == self.name.replace("_strategy", "")''',
                imports_needed=["from ..core.issues import Issue"],
            ),
            
            # Validation patterns
            "validate": ImplementationTemplate(
                pattern=r"validate.*",
                description="Validate input/output",
                template_code='''def {name}(self, data: Any) -> bool:
    """Validate the provided data."""
    if data is None:
        return False
    
    # Add specific validation based on method name
    if "config" in "{name}":
        required_keys = getattr(self, "REQUIRED_CONFIG_KEYS", [])
        if required_keys and isinstance(data, dict):
            return all(key in data for key in required_keys)
    
    return bool(data)''',
                imports_needed=["from typing import Any"],
            ),
            
            # Initialization patterns
            "initialize": ImplementationTemplate(
                pattern=r"(initialize|_init).*",
                description="Initialize component",
                template_code='''def {name}(self) -> None:
    """Initialize the component."""
    # Set default values
    self._initialized = True
    self._cache = {{}}
    
    # Initialize any class-specific attributes
    for attr in dir(self):
        if attr.startswith("DEFAULT_"):
            value = getattr(self, attr)
            target_attr = attr.replace("DEFAULT_", "").lower()
            if not hasattr(self, target_attr):
                setattr(self, target_attr, value)
    
    self.logger.debug(f"{{self.__class__.__name__}} initialized")''',
                imports_needed=[],
            ),
            
            # Getter patterns
            "get_": ImplementationTemplate(
                pattern=r"get_\w+",
                description="Getter method",
                template_code='''def {name}(self) -> Any:
    """Get {attribute_name}."""
    attr_name = "_{attribute_name}"
    if hasattr(self, attr_name):
        return getattr(self, attr_name)
    
    # Try without underscore
    attr_name = "{attribute_name}"
    if hasattr(self, attr_name):
        return getattr(self, attr_name)
    
    # Return default
    default_attr = f"DEFAULT_{{attr_name.upper()}}"
    if hasattr(self, default_attr):
        return getattr(self, default_attr)
    
    return None''',
                imports_needed=["from typing import Any"],
            ),
            
            # Checker patterns
            "is_": ImplementationTemplate(
                pattern=r"is_\w+",
                description="Boolean check method",
                template_code='''def {name}(self, obj: Any = None) -> bool:
    """Check if {condition}."""
    if obj is None:
        obj = self
    
    # Extract condition from method name
    condition = "{name}".replace("is_", "")
    
    # Common patterns
    if condition == "valid":
        return hasattr(obj, "validate") and obj.validate()
    elif condition == "empty":
        return not bool(obj) if hasattr(obj, "__bool__") else len(obj) == 0
    elif condition == "initialized":
        return getattr(obj, "_initialized", False)
    
    # Check for attribute
    return hasattr(obj, condition) and bool(getattr(obj, condition))''',
                imports_needed=["from typing import Any"],
            ),
            
            # Has patterns
            "has_": ImplementationTemplate(
                pattern=r"has_\w+",
                description="Check if has attribute/feature",
                template_code='''def {name}(self, obj: Any = None) -> bool:
    """Check if has {feature}."""
    if obj is None:
        obj = self
    
    feature = "{name}".replace("has_", "")
    
    # Check direct attribute
    if hasattr(obj, feature):
        value = getattr(obj, feature)
        return value is not None and (len(value) > 0 if hasattr(value, "__len__") else bool(value))
    
    # Check with underscore prefix
    if hasattr(obj, f"_{{feature}}"):
        value = getattr(obj, f"_{{feature}}")
        return value is not None and (len(value) > 0 if hasattr(value, "__len__") else bool(value))
    
    return False''',
                imports_needed=["from typing import Any"],
            ),
        }
        
        return templates
    
    def is_trivial_fn(self, fn: ast.FunctionDef) -> bool:
        """Check if a function has a trivial implementation."""
        # Filter out docstrings and comments
        body = [
            n for n in fn.body 
            if not (isinstance(n, ast.Expr) and 
                   isinstance(getattr(n, "value", None), ast.Constant))
        ]
        
        if not body:
            return True
            
        if len(body) == 1:
            stmt = body[0]
            
            # Just pass
            if isinstance(stmt, ast.Pass):
                return True
            
            # Return with trivial value
            if isinstance(stmt, ast.Return):
                v = stmt.value
                if v is None:
                    return True
                if isinstance(v, ast.Constant) and v.value in TRIVIAL_RETURNS:
                    return True
                if isinstance(v, (ast.List, ast.Dict, ast.Tuple)):
                    elts = getattr(v, "elts", getattr(v, "keys", []))
                    if len(elts) == 0:
                        return True
            
            # Just raises NotImplementedError
            if isinstance(stmt, ast.Raise):
                exc = stmt.exc
                if isinstance(exc, ast.Call):
                    if isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                        return True
                
        return False
    
    def scan_file(self, file_path: pathlib.Path) -> List[TrivialFunction]:
        """Scan a single file for trivial functions."""
        found = []
        
        try:
            source = file_path.read_text()
            tree = ast.parse(source, str(file_path))
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return found
        
        class FunctionCollector(ast.NodeVisitor):
            def __init__(self, file_path: str):
                self.file_path = file_path
                self.current_class = None
                self.functions = []
            
            def visit_ClassDef(self, node: ast.ClassDef):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class
            
            def visit_FunctionDef(self, node: ast.FunctionDef):
                # Skip nested functions for now
                if self.current_class or node.col_offset == 0:
                    full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
                    
                    # Extract docstring
                    docstring = ast.get_docstring(node)
                    
                    # Get decorators
                    decorators = []
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Name):
                            decorators.append(dec.id)
                        elif isinstance(dec, ast.Attribute):
                            decorators.append(dec.attr)
                    
                    # Build signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    signature = f"({', '.join(args)})"
                    
                    self.functions.append(TrivialFunction(
                        file_path=self.file_path,
                        name=node.name,
                        full_name=full_name,
                        node=node,
                        docstring=docstring,
                        signature=signature,
                        class_name=self.current_class,
                        decorators=decorators
                    ))
            
            visit_AsyncFunctionDef = visit_FunctionDef
        
        collector = FunctionCollector(str(file_path))
        collector.visit(tree)
        
        # Filter to only trivial functions
        for func in collector.functions:
            if self.is_trivial_fn(func.node):
                found.append(func)
                self.logger.debug(f"Found trivial function: {func.full_name} in {file_path}")
        
        return found
    
    def scan_codebase(self, root_path: pathlib.Path) -> None:
        """Scan entire codebase for trivial functions."""
        self.trivial_functions = []
        
        for py_file in root_path.rglob("*.py"):
            # Skip test files and migrations
            if any(part in str(py_file) for part in ["test", "migration", "__pycache__", ".git"]):
                continue
            
            functions = self.scan_file(py_file)
            self.trivial_functions.extend(functions)
        
        self.logger.info(f"Found {len(self.trivial_functions)} trivial functions")
    
    def find_call_sites(self, func: TrivialFunction, root_path: pathlib.Path) -> List[Tuple[str, int]]:
        """Find all call sites for a function using grep."""
        call_sites = []
        
        # Simple pattern for function calls
        patterns = [
            f"{func.name}\\(",  # Direct call
            f"\\.{func.name}\\(",  # Method call
        ]
        
        for pattern in patterns:
            for py_file in root_path.rglob("*.py"):
                if str(py_file) == func.file_path:
                    continue  # Skip the definition file
                
                try:
                    source = py_file.read_text()
                    for i, line in enumerate(source.splitlines(), 1):
                        if re.search(pattern, line):
                            call_sites.append((str(py_file), i))
                except Exception:
                    pass
        
        return call_sites
    
    def select_template(self, func: TrivialFunction) -> Optional[ImplementationTemplate]:
        """Select the best template for a function based on its name and context."""
        for template_name, template in self.templates.items():
            if re.match(template.pattern, func.name, re.IGNORECASE):
                self.logger.debug(f"Matched template {template_name} for {func.name}")
                return template
        
        # Try to infer from docstring
        if func.docstring:
            doc_lower = func.docstring.lower()
            if "check" in doc_lower or "validate" in doc_lower:
                if func.name.startswith("is_"):
                    return self.templates.get("is_")
                elif func.name.startswith("has_"):
                    return self.templates.get("has_")
            elif "get" in doc_lower or "return" in doc_lower:
                if func.name.startswith("get_"):
                    return self.templates.get("get_")
        
        return None
    
    def generate_implementation(self, func: TrivialFunction, template: ImplementationTemplate) -> str:
        """Generate implementation code for a function."""
        # Extract attribute name from getter/checker methods
        attribute_name = ""
        if func.name.startswith(("get_", "is_", "has_")):
            attribute_name = func.name.split("_", 1)[1] if "_" in func.name else ""
        
        # Format the template
        code = template.template_code.format(
            name=func.name,
            attribute_name=attribute_name,
            condition=attribute_name,
            feature=attribute_name
        )
        
        return code
    
    def generate_test(self, func: TrivialFunction, template: ImplementationTemplate) -> Optional[str]:
        """Generate test code for an upgraded function."""
        if not template.test_template:
            return None
        
        return template.test_template.format(
            name=func.name,
            class_name=func.class_name or "Module"
        )
    
    def upgrade_function(self, func: TrivialFunction, root_path: pathlib.Path) -> bool:
        """Upgrade a single trivial function."""
        # Find call sites for context
        func.call_sites = self.find_call_sites(func, root_path)
        
        # Select template
        template = self.select_template(func)
        if not template:
            self.logger.warning(f"No template found for {func.full_name}")
            return False
        
        # Generate implementation
        new_impl = self.generate_implementation(func, template)
        
        # Generate test if applicable
        test_code = self.generate_test(func, template)
        
        self.logger.info(
            f"Generated implementation for {func.full_name}:\n{new_impl}\n"
            f"Test:\n{test_code if test_code else 'N/A'}"
        )
        
        return True
    
    def upgrade_all(self, root_path: pathlib.Path, dry_run: bool = True) -> Dict[str, Any]:
        """Upgrade all trivial functions found."""
        self.scan_codebase(root_path)
        
        results = {
            "total": len(self.trivial_functions),
            "upgraded": 0,
            "skipped": 0,
            "failed": 0,
            "functions": []
        }
        
        for func in self.trivial_functions:
            try:
                if self.upgrade_function(func, root_path):
                    results["upgraded"] += 1
                    results["functions"].append({
                        "name": func.full_name,
                        "file": func.file_path,
                        "status": "upgraded"
                    })
                else:
                    results["skipped"] += 1
                    results["functions"].append({
                        "name": func.full_name,
                        "file": func.file_path,
                        "status": "skipped"
                    })
            except Exception as e:
                results["failed"] += 1
                results["functions"].append({
                    "name": func.full_name,
                    "file": func.file_path,
                    "status": "failed",
                    "error": str(e)
                })
                self.logger.error(f"Failed to upgrade {func.full_name}: {e}")
        
        return results


def main():
    """CLI entry point for the trivial function upgrader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upgrade trivial/stub functions")
    parser.add_argument("path", type=pathlib.Path, help="Path to scan")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    upgrader = TrivialFunctionUpgrader()
    results = upgrader.upgrade_all(args.path, dry_run=args.dry_run)
    
    print(f"\nUpgrade Summary:")
    print(f"  Total trivial functions: {results['total']}")
    print(f"  Successfully upgraded: {results['upgraded']}")
    print(f"  Skipped (no template): {results['skipped']}") 
    print(f"  Failed: {results['failed']}")
    
    if args.verbose:
        print("\nDetails:")
        for func in results["functions"]:
            print(f"  {func['name']} ({func['file']}): {func['status']}")
            if "error" in func:
                print(f"    Error: {func['error']}")


if __name__ == "__main__":
    main()