"""
Circular dependency scenarios that should be fixed in 3-4 steps.
"""

from typing import Dict, List, Tuple
from .base import BenchmarkScenario


class CircularDependencyScenario(BenchmarkScenario):
    """Scenario with circular import dependencies requiring careful refactoring."""
    
    def __init__(self):
        super().__init__(
            name="circular_dependency",
            description="Circular import dependencies requiring module restructuring",
            expected_steps=(3, 4)
        )
    
    def setup(self) -> str:
        """Set up the circular dependency scenario."""
        temp_dir = self.create_temp_directory()
        self.write_files(self.get_initial_code())
        return temp_dir
    
    def get_initial_code(self) -> Dict[str, str]:
        """Get initial code with circular dependencies."""
        return {
            "models/user.py": '''"""User model with circular dependency."""

from .order import Order  # Circular: Order imports User

class User:
    """User model."""
    
    def __init__(self, user_id: str, name: str):
        self.user_id = user_id
        self.name = name
        self.orders = []
    
    def add_order(self, order: Order):
        """Add an order to user."""
        self.orders.append(order)
        order.set_user(self)
    
    def get_total_spent(self) -> float:
        """Calculate total amount spent."""
        return sum(order.total_amount for order in self.orders)
    
    def get_order_count(self) -> int:
        """Get number of orders."""
        return len(self.orders)
''',
            "models/order.py": '''"""Order model with circular dependency."""

from .user import User  # Circular: User imports Order
from .product import Product  # Circular: Product imports Order

class Order:
    """Order model."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.user = None
        self.products = []
        self.total_amount = 0.0
    
    def set_user(self, user: User):
        """Set the user for this order."""
        self.user = user
    
    def add_product(self, product: Product, quantity: int):
        """Add a product to the order."""
        self.products.append((product, quantity))
        self.total_amount += product.get_price() * quantity
        product.increment_sales(quantity)
    
    def get_user_name(self) -> str:
        """Get the user's name."""
        return self.user.name if self.user else "Unknown"
''',
            "models/product.py": '''"""Product model with circular dependency."""

from .order import Order  # Circular: Order imports Product
from .inventory import Inventory  # Circular: Inventory imports Product

class Product:
    """Product model."""
    
    def __init__(self, product_id: str, name: str, price: float):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.inventory = None
        self.sales_count = 0
    
    def get_price(self) -> float:
        """Get product price."""
        return self.price
    
    def set_inventory(self, inventory: Inventory):
        """Set inventory for this product."""
        self.inventory = inventory
    
    def increment_sales(self, quantity: int):
        """Increment sales count."""
        self.sales_count += quantity
        if self.inventory:
            self.inventory.reduce_stock(self.product_id, quantity)
    
    def check_availability(self, order: Order) -> bool:
        """Check if product is available for order."""
        return self.inventory.get_stock(self.product_id) > 0
''',
            "models/inventory.py": '''"""Inventory model with circular dependency."""

from .product import Product  # Circular: Product imports Inventory

class Inventory:
    """Inventory management."""
    
    def __init__(self):
        self.stock = {}
        self.products = {}
    
    def add_product(self, product: Product, initial_stock: int):
        """Add a product to inventory."""
        self.products[product.product_id] = product
        self.stock[product.product_id] = initial_stock
        product.set_inventory(self)
    
    def get_stock(self, product_id: str) -> int:
        """Get current stock level."""
        return self.stock.get(product_id, 0)
    
    def reduce_stock(self, product_id: str, quantity: int):
        """Reduce stock by quantity."""
        if product_id in self.stock:
            self.stock[product_id] = max(0, self.stock[product_id] - quantity)
''',
            "models/__init__.py": '''"""Models package initialization."""

from .user import User
from .order import Order
from .product import Product
from .inventory import Inventory

__all__ = ['User', 'Order', 'Product', 'Inventory']
'''
        }
    
    def get_expected_solution(self) -> Dict[str, str]:
        """Get expected solution with circular dependencies resolved."""
        return {
            "models/base.py": '''"""Base types to avoid circular dependencies."""

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
    from .order import Order
    from .product import Product
    from .inventory import Inventory

class UserProtocol(Protocol):
    """Protocol for User type."""
    user_id: str
    name: str
    orders: list

class OrderProtocol(Protocol):
    """Protocol for Order type."""
    order_id: str
    total_amount: float

class ProductProtocol(Protocol):
    """Protocol for Product type."""
    product_id: str
    name: str
    price: float

class InventoryProtocol(Protocol):
    """Protocol for Inventory type."""
    stock: dict
''',
            "models/user.py": '''"""User model without circular dependency."""

from typing import TYPE_CHECKING, List
from .base import OrderProtocol

if TYPE_CHECKING:
    from .order import Order

class User:
    """User model."""
    
    def __init__(self, user_id: str, name: str):
        self.user_id = user_id
        self.name = name
        self.orders: List[OrderProtocol] = []
    
    def add_order(self, order: OrderProtocol):
        """Add an order to user."""
        self.orders.append(order)
        if hasattr(order, 'set_user'):
            order.set_user(self)
    
    def get_total_spent(self) -> float:
        """Calculate total amount spent."""
        return sum(order.total_amount for order in self.orders)
    
    def get_order_count(self) -> int:
        """Get number of orders."""
        return len(self.orders)
''',
            "models/order.py": '''"""Order model without circular dependency."""

from typing import TYPE_CHECKING, List, Optional, Tuple
from .base import UserProtocol, ProductProtocol

if TYPE_CHECKING:
    from .user import User
    from .product import Product

class Order:
    """Order model."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.user: Optional[UserProtocol] = None
        self.products: List[Tuple[ProductProtocol, int]] = []
        self.total_amount = 0.0
    
    def set_user(self, user: UserProtocol):
        """Set the user for this order."""
        self.user = user
    
    def add_product(self, product: ProductProtocol, quantity: int):
        """Add a product to the order."""
        self.products.append((product, quantity))
        if hasattr(product, 'get_price'):
            self.total_amount += product.get_price() * quantity
        if hasattr(product, 'increment_sales'):
            product.increment_sales(quantity)
    
    def get_user_name(self) -> str:
        """Get the user's name."""
        return self.user.name if self.user else "Unknown"
''',
            "models/product.py": '''"""Product model without circular dependency."""

from typing import TYPE_CHECKING, Optional
from .base import InventoryProtocol, OrderProtocol

if TYPE_CHECKING:
    from .inventory import Inventory
    from .order import Order

class Product:
    """Product model."""
    
    def __init__(self, product_id: str, name: str, price: float):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.inventory: Optional[InventoryProtocol] = None
        self.sales_count = 0
    
    def get_price(self) -> float:
        """Get product price."""
        return self.price
    
    def set_inventory(self, inventory: InventoryProtocol):
        """Set inventory for this product."""
        self.inventory = inventory
    
    def increment_sales(self, quantity: int):
        """Increment sales count."""
        self.sales_count += quantity
        if self.inventory and hasattr(self.inventory, 'reduce_stock'):
            self.inventory.reduce_stock(self.product_id, quantity)
    
    def check_availability(self, order: OrderProtocol) -> bool:
        """Check if product is available for order."""
        if self.inventory and hasattr(self.inventory, 'get_stock'):
            return self.inventory.get_stock(self.product_id) > 0
        return False
''',
            "models/inventory.py": '''"""Inventory model without circular dependency."""

from typing import TYPE_CHECKING, Dict
from .base import ProductProtocol

if TYPE_CHECKING:
    from .product import Product

class Inventory:
    """Inventory management."""
    
    def __init__(self):
        self.stock: Dict[str, int] = {}
        self.products: Dict[str, ProductProtocol] = {}
    
    def add_product(self, product: ProductProtocol, initial_stock: int):
        """Add a product to inventory."""
        self.products[product.product_id] = product
        self.stock[product.product_id] = initial_stock
        if hasattr(product, 'set_inventory'):
            product.set_inventory(self)
    
    def get_stock(self, product_id: str) -> int:
        """Get current stock level."""
        return self.stock.get(product_id, 0)
    
    def reduce_stock(self, product_id: str, quantity: int):
        """Reduce stock by quantity."""
        if product_id in self.stock:
            self.stock[product_id] = max(0, self.stock[product_id] - quantity)
''',
            "models/__init__.py": '''"""Models package initialization."""

from .user import User
from .order import Order
from .product import Product
from .inventory import Inventory

__all__ = ['User', 'Order', 'Product', 'Inventory']
'''
        }
    
    def validate_solution(self, current_code: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate if circular dependencies are resolved."""
        errors = []
        
        # Check for TYPE_CHECKING usage
        for file_name in ["models/user.py", "models/order.py", "models/product.py", "models/inventory.py"]:
            if file_name in current_code:
                content = current_code[file_name]
                if "from typing import TYPE_CHECKING" not in content and "TYPE_CHECKING" in content:
                    errors.append(f"Missing TYPE_CHECKING import in {file_name}")
        
        # Check for direct circular imports
        import_map = {}
        for file_name, content in current_code.items():
            if file_name.endswith(".py") and "__init__" not in file_name:
                imports = []
                for line in content.split("\n"):
                    if line.strip().startswith("from .") and "TYPE_CHECKING" not in line:
                        # Extract module name from import
                        parts = line.split()
                        if len(parts) >= 2:
                            module = parts[1].replace(".", "")
                            imports.append(module)
                import_map[file_name] = imports
        
        # Detect circular dependencies
        def has_circular_dependency(module: str, target: str, visited: set) -> bool:
            if module == target and len(visited) > 0:
                return True
            if module in visited:
                return False
            visited.add(module)
            
            module_file = f"models/{module}.py"
            if module_file in import_map:
                for imported in import_map[module_file]:
                    if has_circular_dependency(imported, target, visited.copy()):
                        return True
            return False
        
        for module in ["user", "order", "product", "inventory"]:
            if has_circular_dependency(module, module, set()):
                errors.append(f"Circular dependency detected involving {module}")
        
        # Validate syntax
        import ast
        for file_path, content in current_code.items():
            if file_path.endswith(".py"):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"Syntax error in {file_path}: {e}")
        
        return len(errors) == 0, errors