# Smart Semantic Filtering

## Overview

TailChasingFixer includes intelligent filtering to prevent false positives in semantic duplicate detection. The system recognizes legitimate patterns that are supposed to be similar and excludes them from tail-chasing warnings.

## Why This Matters

Without smart filtering, the semantic analyzer would flag every `__init__.py` file as containing "duplicate" functions, or mark legitimate test setUp/tearDown methods as tail-chasing patterns. This creates noise and reduces trust in the tool.

## Filtered Patterns

### 1. Init File Patterns (`__init__.py`)

**What gets filtered:**
- Functions in `__init__.py` files that handle module initialization
- Simple import/export functions
- Version, author, and package metadata functions

**Example that won't be flagged:**
```python
# package1/__init__.py
def setup_logging():
    logging.basicConfig(level=logging.INFO)

__version__ = "1.0.0"

# package2/__init__.py  
def setup_logging():
    logging.basicConfig(level=logging.INFO)

__version__ = "2.0.0"
```

### 2. Test Boilerplate

**What gets filtered:**
- `setUp` and `tearDown` methods
- Simple test fixtures
- Test utility functions with similar assert patterns

**Example that won't be flagged:**
```python
# test_module1.py
def setUp(self):
    self.client = TestClient()
    self.db = create_test_db()

# test_module2.py
def setUp(self):
    self.client = TestClient() 
    self.db = create_test_db()
```

### 3. Property Accessors

**What gets filtered:**
- Simple getter/setter methods
- Properties with `@property` decorator
- Cached properties

**Example that won't be flagged:**
```python
class User:
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

class Product:
    @property  
    def full_name(self):
        return f"{self.brand} {self.model}"
```

### 4. Protocol Implementations

**What gets filtered:**
- Dunder methods (`__str__`, `__repr__`, etc.)
- Interface method implementations with same name
- Abstract method implementations

**Example that won't be flagged:**
```python
class Rectangle:
    def __str__(self):
        return f"Rectangle({self.width}x{self.height})"

class Circle:
    def __str__(self):
        return f"Circle(radius={self.radius})"
```

### 5. Factory Methods

**What gets filtered:**
- Functions starting with `create_`, `make_`, `build_`, `from_`, `new_`
- Simple instantiation patterns

**Example that won't be flagged:**
```python
def create_user(name, email):
    return User(name=name, email=email)

def create_admin(name, email):
    return Admin(name=name, email=email)
```

### 6. Configuration Patterns

**What gets filtered:**
- Functions in files containing 'config', 'settings', 'options'
- Configuration method patterns

**Example that won't be flagged:**
```python
# config/database.py
def get_connection_string():
    return f"postgresql://{USER}:{PASS}@{HOST}:{PORT}/{DB}"

# config/cache.py
def get_connection_string():
    return f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
```

### 7. Exception Classes

**What gets filtered:**
- Exception class `__init__` methods
- Custom exception patterns

### 8. Migration Scripts

**What gets filtered:**
- Database migration functions
- Schema upgrade/downgrade methods

## Configuration

You can control smart filtering in your `.tailchasing.yml`:

```yaml
semantic:
  enable: true
  smart_filtering: true  # Enable smart filtering (default: true)
  
  # Customize which patterns to filter
  filter_config:
    enable_init_filter: true
    enable_test_filter: true
    enable_property_filter: true
    enable_protocol_filter: true
    
    # Add custom patterns
    custom_patterns:
      - pattern_name: "django_views"
        file_pattern: "views.py"
        function_pattern: ["get", "post", "put", "delete"]
      - pattern_name: "serializers"
        file_pattern: "serializers.py"
        function_pattern: ["to_representation", "validate_"]
```

## Transparency

When smart filtering removes pairs, it logs what was filtered:

```
🧹 Filtered out legitimate patterns:
  - Legitimate init_files pattern: 12 pairs
  - Legitimate test_boilerplate pattern: 8 pairs
  - Legitimate property_accessors pattern: 4 pairs
```

## Customization

You can add your own filtering patterns by extending the `SemanticDuplicateFilter` class:

```python
from tailchasing.semantic.smart_filter import SemanticDuplicateFilter

class CustomFilter(SemanticDuplicateFilter):
    def __init__(self):
        super().__init__()
        self.legitimate_patterns['my_pattern'] = self._is_my_pattern
    
    def _is_my_pattern(self, func1, func2, file1, file2):
        # Your custom logic here
        return False
```

## Performance Impact

Smart filtering adds minimal overhead:
- ~5ms per pair analysis on typical functions
- Runs only on pairs that already passed statistical significance
- Filters run after heavy hypervector computation

## False Negative Prevention

The filtering is conservative - it only filters patterns it's confident about. If there's any doubt, the pair is kept for manual review. This prevents missing actual tail-chasing that happens to occur in legitimate contexts.

## Examples of What Still Gets Caught

Even with smart filtering, real tail-chasing patterns are still detected:

```python
# __init__.py - This WOULD still be flagged
def process_user_data(data):
    # Complex business logic that got duplicated
    validated = validate_input(data)
    normalized = normalize_data(validated)
    enriched = enrich_with_metadata(normalized)
    return store_processed_data(enriched)

def handle_user_information(info):
    # Same logic, different name - likely tail-chasing
    validated = validate_input(info)
    normalized = normalize_data(validated)  
    enriched = enrich_with_metadata(normalized)
    return store_processed_data(enriched)
```

The smart filter recognizes that this isn't simple `__init__.py` boilerplate and flags it appropriately.
