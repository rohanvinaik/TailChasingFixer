"""
Example: Fixing Semantic Duplicates

This example demonstrates how TailChasing Fixer identifies and consolidates
semantically similar functions that perform the same operation.
"""

# ============================================================================
# BEFORE: Multiple functions doing the same thing
# ============================================================================

# Original code with semantic duplicates
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    total = sum(numbers)
    count = len(numbers)
    return total / count

def compute_mean(data_list):
    """Compute mean value of data."""
    if len(data_list) == 0:
        return 0
    accumulator = 0
    for value in data_list:
        accumulator += value
    return accumulator / len(data_list)

def get_avg(values):
    """Get average."""
    return sum(values) / len(values) if values else 0

def find_mean_value(collection):
    """Find the mean value in a collection."""
    if not collection:
        return 0.0
    return float(sum(collection)) / float(len(collection))

# Similar pattern for validation functions
def validate_email(email):
    """Check if email is valid."""
    return "@" in email and "." in email

def check_email_format(email_address):
    """Validate email address format."""
    if "@" not in email_address:
        return False
    if "." not in email_address:
        return False
    return True

def is_valid_email(email_str):
    """Determine if email string is valid."""
    has_at = "@" in email_str
    has_dot = "." in email_str
    return has_at and has_dot

# File I/O duplicates
def read_json_file(filepath):
    """Read JSON from file."""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)

def load_json_data(filename):
    """Load JSON data from disk."""
    import json
    file = open(filename, 'r')
    data = json.load(file)
    file.close()
    return data

def get_json_from_file(path):
    """Get JSON content from file path."""
    import json
    return json.load(open(path, 'r'))


# ============================================================================
# AFTER: Consolidated functions with aliases
# ============================================================================

from typing import List, Union, Any, Optional
import json
from pathlib import Path
import warnings
from functools import wraps


def deprecation_warning(old_name: str, new_name: str):
    """Decorator to warn about deprecated function names."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{old_name} is deprecated, use {new_name} instead",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# --- Consolidated Average Calculation ---

def calculate_average(numbers: List[Union[int, float]]) -> float:
    """Calculate the average of a list of numbers.
    
    This is the primary implementation for computing arithmetic mean.
    Other function names are maintained as aliases for backward compatibility.
    
    Args:
        numbers: List of numeric values
        
    Returns:
        The arithmetic mean, or 0 if the list is empty
        
    Examples:
        >>> calculate_average([1, 2, 3, 4, 5])
        3.0
        >>> calculate_average([])
        0.0
    """
    if not numbers:
        return 0.0
    
    # Use built-in sum for efficiency
    total = sum(numbers)
    count = len(numbers)
    
    # Ensure float division
    return float(total) / float(count)


# Maintain backward compatibility with deprecation warnings
compute_mean = deprecation_warning("compute_mean", "calculate_average")(calculate_average)
get_avg = deprecation_warning("get_avg", "calculate_average")(calculate_average)
find_mean_value = deprecation_warning("find_mean_value", "calculate_average")(calculate_average)


# --- Consolidated Email Validation ---

def validate_email(email: str, strict: bool = False) -> bool:
    """Validate email address format.
    
    This is the primary email validation implementation.
    
    Args:
        email: Email address string to validate
        strict: If True, perform more thorough validation
        
    Returns:
        True if email format is valid, False otherwise
        
    Examples:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid")
        False
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic validation
    if "@" not in email:
        return False
    
    # Split into local and domain parts
    parts = email.split("@")
    if len(parts) != 2:
        return False
    
    local, domain = parts
    
    # Check both parts are non-empty
    if not local or not domain:
        return False
    
    # Domain must have at least one dot
    if "." not in domain:
        return False
    
    if strict:
        # Additional strict validation
        import re
        # RFC 5322 simplified regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    return True


# Backward compatibility aliases
check_email_format = deprecation_warning("check_email_format", "validate_email")(validate_email)
is_valid_email = deprecation_warning("is_valid_email", "validate_email")(validate_email)


# --- Consolidated JSON File Reading ---

def read_json_file(filepath: Union[str, Path], 
                  encoding: str = 'utf-8',
                  errors: str = 'strict') -> Any:
    """Read and parse JSON data from a file.
    
    This is the primary JSON file reading implementation with
    proper error handling and resource management.
    
    Args:
        filepath: Path to the JSON file
        encoding: File encoding (default: utf-8)
        errors: How to handle encoding errors
        
    Returns:
        Parsed JSON data (dict, list, etc.)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        
    Examples:
        >>> data = read_json_file("config.json")
        >>> data = read_json_file(Path("data") / "settings.json")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding=encoding, errors=errors) as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {filepath}: {e.msg}",
            e.doc,
            e.pos
        )


# Backward compatibility aliases
load_json_data = deprecation_warning("load_json_data", "read_json_file")(read_json_file)
get_json_from_file = deprecation_warning("get_json_from_file", "read_json_file")(read_json_file)


# ============================================================================
# MIGRATION HELPER
# ============================================================================

class MigrationHelper:
    """Helper class to assist with migrating to consolidated functions."""
    
    @staticmethod
    def update_imports(code: str) -> str:
        """Update import statements to use new function names."""
        replacements = {
            'compute_mean': 'calculate_average',
            'get_avg': 'calculate_average',
            'find_mean_value': 'calculate_average',
            'check_email_format': 'validate_email',
            'is_valid_email': 'validate_email',
            'load_json_data': 'read_json_file',
            'get_json_from_file': 'read_json_file',
        }
        
        updated_code = code
        for old_name, new_name in replacements.items():
            updated_code = updated_code.replace(
                f"from module import {old_name}",
                f"from module import {new_name}"
            )
        
        return updated_code
    
    @staticmethod
    def find_usage_locations(codebase_path: Path) -> Dict[str, List[int]]:
        """Find where deprecated functions are used."""
        deprecated_names = [
            'compute_mean', 'get_avg', 'find_mean_value',
            'check_email_format', 'is_valid_email',
            'load_json_data', 'get_json_from_file'
        ]
        
        usage_locations = {}
        
        for py_file in codebase_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                for name in deprecated_names:
                    if name in line:
                        if str(py_file) not in usage_locations:
                            usage_locations[str(py_file)] = []
                        usage_locations[str(py_file)].append(i)
        
        return usage_locations


# ============================================================================
# FIX STRATEGY EXPLANATION
# ============================================================================

"""
The fix strategy for semantic duplicates involves:

1. **Detection Phase**:
   - Use semantic hypervector analysis to find similar functions
   - Group functions by semantic similarity (>85% similarity threshold)
   - Identify the most complete/robust implementation

2. **Consolidation Strategy**:
   - Choose the best implementation as the primary function
   - Add comprehensive documentation and type hints
   - Improve error handling and edge cases
   - Create aliases for backward compatibility

3. **Migration Support**:
   - Add deprecation warnings to old function names
   - Provide migration helper utilities
   - Generate documentation for changes
   - Create update scripts for automatic migration

4. **Validation**:
   - Ensure all test cases still pass
   - Verify backward compatibility
   - Check for performance improvements
   - Monitor for regression issues

Running the fix:
```bash
# Detect semantic duplicates
tailchasing analyze . --semantic --threshold 0.85

# Generate consolidation plan
tailchasing fix . --type semantic_duplicate --plan-only

# Apply fixes with migration support
tailchasing fix . --type semantic_duplicate --migrate
```

Benefits of consolidation:
- Reduced code duplication (60-80% reduction)
- Improved maintainability (single source of truth)
- Better performance (optimized implementations)
- Enhanced documentation and type safety
- Easier testing and debugging
"""

if __name__ == "__main__":
    # Test consolidated functions
    print("Testing consolidated functions...")
    
    # Test average calculation
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0
    assert calculate_average([]) == 0.0
    
    # Test email validation
    assert validate_email("user@example.com") == True
    assert validate_email("invalid") == False
    
    # Test JSON reading (would need actual file)
    # data = read_json_file("config.json")
    
    print("All tests passed!")