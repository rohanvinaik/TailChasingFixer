# Fix Strategies Module

This module contains the refactored fix strategies, split from the monolithic `fix_strategies.py` file into smaller, focused modules.

## Structure

```
strategies/
├── __init__.py           # Package exports
├── base.py              # Base classes and common types
├── imports.py           # ImportResolutionStrategy
├── duplicates.py        # DuplicateMergeStrategy (TODO)
├── placeholders.py      # PlaceholderImplementationStrategy (TODO)
├── circular.py          # CircularDependencyBreaker (TODO)
├── async_sync.py        # AsyncSyncMismatchFixer (TODO)
└── selector.py          # StrategySelector (TODO)
```

## Migration Status

### Completed
- ✅ Base classes (RiskLevel, Action, Patch, FixStrategy, BaseFixStrategy)
- ✅ ImportResolutionStrategy

### TODO
- ⏳ DuplicateMergeStrategy
- ⏳ PlaceholderImplementationStrategy
- ⏳ CircularDependencyBreaker
- ⏳ AsyncSyncMismatchFixer
- ⏳ StrategySelector

## Benefits of Refactoring

1. **Modularity**: Each strategy is in its own file, making it easier to find and modify
2. **Maintainability**: Smaller files are easier to understand and maintain
3. **Testability**: Individual strategies can be tested in isolation
4. **Scalability**: New strategies can be added without making existing files larger
5. **Organization**: Related code is grouped together logically

## Usage

### Importing from new location

```python
from tailchasing.fixers.strategies import (
    RiskLevel,
    Action,
    Patch,
    ImportResolutionStrategy
)

# Create and use a strategy
strategy = ImportResolutionStrategy()
if strategy.can_handle(issue):
    patch = strategy.propose_fix(issue, context)
```

### Backward Compatibility

During migration, a compatibility layer is provided in `fix_strategies_compat.py` that re-exports from both old and new locations.

## Adding New Strategies

To add a new fix strategy:

1. Create a new file in `strategies/` directory
2. Inherit from `BaseFixStrategy`
3. Implement required methods:
   - `can_handle(issue)`: Determine if strategy applies
   - `propose_fix(issue, context)`: Generate fix patch
4. Export from `__init__.py`
5. Add tests in `tests/test_strategies/`

## Architecture

Each strategy follows the same pattern:

```python
from .base import BaseFixStrategy, Action, Patch, RiskLevel
from ...core.issues import Issue

class MyStrategy(BaseFixStrategy):
    def __init__(self):
        super().__init__("MyStrategy")
    
    def can_handle(self, issue: Issue) -> bool:
        # Check if this strategy can handle the issue
        return issue.kind in ["my_issue_type"]
    
    def propose_fix(self, issue: Issue, context=None) -> Optional[Patch]:
        # Generate and return a Patch with Actions
        actions = self._generate_actions(issue, context)
        return Patch(
            actions=actions,
            description="Fix description",
            confidence=0.9,
            risk_level=RiskLevel.LOW,
            estimated_time=1.0
        )
```