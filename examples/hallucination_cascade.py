"""
Example: Hallucination Cascade Anti-Pattern

This demonstrates over-engineered abstraction chains that solve
non-existent problems, often created when LLMs generate "enterprise-grade"
solutions for simple requirements.
"""

# BEFORE: Over-engineered abstraction cascade

from abc import ABC, abstractmethod
from typing import Protocol, Generic, TypeVar, Any

# Layer 1: Abstract interfaces
class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class DataValidator(ABC):
    @abstractmethod
    def validate(self, data: Any) -> bool:
        pass

class DataTransformer(ABC):
    @abstractmethod
    def transform(self, data: Any) -> Any:
        pass

# Layer 2: Generic base classes
class BaseProcessor(DataProcessor):
    @abstractmethod
    def pre_process(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def post_process(self, data: Any) -> Any:
        pass
    
    def process(self, data: Any) -> Any:
        data = self.pre_process(data)
        data = self._internal_process(data)
        return self.post_process(data)
    
    @abstractmethod
    def _internal_process(self, data: Any) -> Any:
        pass

# Layer 3: Specific abstract managers
class AbstractDataManager(BaseProcessor, DataValidator, DataTransformer):
    @abstractmethod
    def manage(self, data: Any) -> Any:
        pass
    
    def pre_process(self, data: Any) -> Any:
        if not self.validate(data):
            raise ValueError("Invalid data")
        return data
    
    def post_process(self, data: Any) -> Any:
        return self.transform(data)

# Layer 4: Concrete abstract implementations (?!)
class ConcreteAbstractManager(AbstractDataManager):
    @abstractmethod
    def get_configuration(self) -> dict:
        pass
    
    def manage(self, data: Any) -> Any:
        config = self.get_configuration()
        # Does nothing with config
        return self.process(data)

# Layer 5: Factory for creating factories
class ManagerFactory(ABC):
    @abstractmethod
    def create_manager(self) -> AbstractDataManager:
        pass

class ConcreteManagerFactory(ManagerFactory):
    def create_manager(self) -> AbstractDataManager:
        return SimpleDataManager()  # Finally, the actual implementation

# Layer 6: The actual simple implementation
class SimpleDataManager(ConcreteAbstractManager):
    def validate(self, data: Any) -> bool:
        return data is not None
    
    def transform(self, data: Any) -> Any:
        return str(data).upper()  # Just uppercases strings!
    
    def _internal_process(self, data: Any) -> Any:
        return data
    
    def get_configuration(self) -> dict:
        return {}

# Usage of this monstrosity:
factory = ConcreteManagerFactory()
manager = factory.create_manager()
result = manager.manage("hello")  # Just returns "HELLO"


# AFTER: Simple, direct implementation

def process_text(text: str) -> str:
    """
    Process text by converting to uppercase.
    
    Args:
        text: Input text
        
    Returns:
        Uppercase text
        
    Raises:
        ValueError: If text is None
    """
    if text is None:
        raise ValueError("Text cannot be None")
    return text.upper()


# How TailChasingFixer detects this:
#
# $ tailchasing analyze --deep
#
# Issue: hallucination_cascade
# Severity: 4 (High)
# File: examples/hallucination_cascade.py
# 
# Detected over-engineered abstraction cascade:
# - 6 levels of abstraction for simple string uppercasing
# - 7 abstract methods with only 1 concrete implementation
# - Factory pattern creating factories (meta-factory anti-pattern)
# - No external usage of intermediate abstractions
# - Abstract class (ConcreteAbstractManager) with "Concrete" and "Abstract" in name
#
# Recommendation:
# Replace entire hierarchy with simple function (3 lines of code vs 90+)
#
# Pattern characteristics:
# - Chain length: 6 (threshold: 3)
# - Abstract methods: 12
# - Concrete usage points: 1
# - External references: 0
# - Abstraction depth: 6 (threshold: 5)