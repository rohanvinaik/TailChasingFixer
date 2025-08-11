"""
Example: Fixing Circular Import Dependencies

This example demonstrates how TailChasing Fixer identifies and resolves
circular import dependencies using various strategies.
"""

# ============================================================================
# BEFORE: Circular dependency between modules
# ============================================================================

# File: models/user.py
"""
from models.order import Order  # Circular!

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.orders = []
    
    def add_order(self, order: Order):
        self.orders.append(order)
        order.set_user(self)
"""

# File: models/order.py
"""
from models.user import User  # Circular!

class Order:
    def __init__(self, order_id):
        self.order_id = order_id
        self.user = None
    
    def set_user(self, user: User):
        self.user = user
"""

# ============================================================================
# FIX STRATEGY 1: Using TYPE_CHECKING
# ============================================================================

# File: models/user_fixed_v1.py
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from models.order import Order  # Import only for type checking

class UserV1:
    """User model with TYPE_CHECKING fix."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.orders: List['Order'] = []  # Forward reference
    
    def add_order(self, order: 'Order'):
        """Add an order to user's history."""
        self.orders.append(order)
        # Late import to avoid circular dependency at runtime
        from models.order import Order
        if isinstance(order, Order):
            order.set_user(self)


# File: models/order_fixed_v1.py
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models.user import User

class OrderV1:
    """Order model with TYPE_CHECKING fix."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.user: Optional['User'] = None
    
    def set_user(self, user: 'User'):
        """Set the user for this order."""
        self.user = user


# ============================================================================
# FIX STRATEGY 2: Using Protocol/Interface
# ============================================================================

from typing import Protocol, List, Optional

# File: models/interfaces.py
class UserProtocol(Protocol):
    """Protocol defining user interface."""
    user_id: str
    orders: List['OrderProtocol']
    
    def add_order(self, order: 'OrderProtocol') -> None: ...

class OrderProtocol(Protocol):
    """Protocol defining order interface."""
    order_id: str
    user: Optional['UserProtocol']
    
    def set_user(self, user: 'UserProtocol') -> None: ...


# File: models/user_fixed_v2.py
class UserV2:
    """User model using Protocol."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.orders: List[OrderProtocol] = []
    
    def add_order(self, order: OrderProtocol):
        """Add an order to user's history."""
        self.orders.append(order)
        order.set_user(self)


# File: models/order_fixed_v2.py  
class OrderV2:
    """Order model using Protocol."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.user: Optional[UserProtocol] = None
    
    def set_user(self, user: UserProtocol):
        """Set the user for this order."""
        self.user = user


# ============================================================================
# FIX STRATEGY 3: Dependency Injection
# ============================================================================

# File: models/user_fixed_v3.py
class UserV3:
    """User model with dependency injection."""
    
    def __init__(self, user_id: str, order_factory=None):
        self.user_id = user_id
        self.orders = []
        self.order_factory = order_factory
    
    def create_order(self, order_id: str):
        """Create a new order for this user."""
        if self.order_factory:
            order = self.order_factory(order_id)
            self.add_order(order)
            return order
        return None
    
    def add_order(self, order):
        """Add an order to user's history."""
        self.orders.append(order)
        if hasattr(order, 'set_user'):
            order.set_user(self)


# File: models/order_fixed_v3.py
class OrderV3:
    """Order model with dependency injection."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.user = None
    
    def set_user(self, user):
        """Set the user for this order."""
        self.user = user


# File: models/factory.py
class ModelFactory:
    """Factory to create models without circular imports."""
    
    @staticmethod
    def create_user(user_id: str) -> UserV3:
        """Create a user with order factory."""
        user = UserV3(user_id, order_factory=ModelFactory.create_order)
        return user
    
    @staticmethod
    def create_order(order_id: str) -> OrderV3:
        """Create an order."""
        return OrderV3(order_id)


# ============================================================================
# FIX STRATEGY 4: Restructuring into Single Module
# ============================================================================

# File: models/entities.py
class UserV4:
    """User model in single module."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.orders: List[OrderV4] = []
    
    def add_order(self, order: 'OrderV4'):
        """Add an order to user's history."""
        self.orders.append(order)
        order.set_user(self)


class OrderV4:
    """Order model in single module."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.user: Optional[UserV4] = None
    
    def set_user(self, user: UserV4):
        """Set the user for this order."""
        self.user = user


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Demonstrate usage of fixed models."""
    
    # Using TYPE_CHECKING version
    user1 = UserV1("user_001")
    # order1 = OrderV1("order_001")
    # user1.add_order(order1)
    
    # Using Protocol version
    user2 = UserV2("user_002")
    order2 = OrderV2("order_002")
    user2.add_order(order2)
    
    # Using Dependency Injection
    factory = ModelFactory()
    user3 = factory.create_user("user_003")
    order3 = user3.create_order("order_003")
    
    # Using Single Module
    user4 = UserV4("user_004")
    order4 = OrderV4("order_004")
    user4.add_order(order4)
    
    print("All circular dependencies resolved!")


# ============================================================================
# FIX STRATEGY EXPLANATION
# ============================================================================

"""
The fix strategy for circular imports involves multiple approaches:

1. **TYPE_CHECKING Strategy**:
   - Use `if TYPE_CHECKING:` to import only during type checking
   - Use string literals for forward references
   - Pros: Maintains type hints, minimal changes
   - Cons: Requires Python 3.5.2+

2. **Protocol/Interface Strategy**:
   - Define protocols that both classes can reference
   - No direct imports between concrete classes
   - Pros: Clean separation, testable
   - Cons: More boilerplate code

3. **Dependency Injection Strategy**:
   - Pass dependencies as parameters
   - Use factories to manage creation
   - Pros: Very flexible, testable
   - Cons: More complex setup

4. **Module Restructuring Strategy**:
   - Combine related classes into single module
   - Eliminate cross-module imports
   - Pros: Simple, no runtime overhead
   - Cons: May create large modules

Running the fix:
```bash
tailchasing fix . --type circular_import --strategy type_checking
```

The fixer will:
- Detect circular import chains
- Analyze which strategy is most appropriate
- Apply the fix while preserving functionality
- Validate no runtime errors are introduced
"""

if __name__ == "__main__":
    example_usage()