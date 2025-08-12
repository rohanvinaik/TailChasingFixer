"""
Test fixtures for circular import detection.

Contains files that would create circular import dependencies
when used together.
"""

# File 1: models/user.py (simulated)
"""
This represents the content of models/user.py which imports from order.py
"""

from typing import List, Optional
# This would create a circular import:
# from .order import Order  # Circular import!

class User:
    """User model that references orders."""
    
    def __init__(self, user_id: str, email: str, name: str):
        self.id = user_id
        self.email = email
        self.name = name
        self.orders = []  # Should be List[Order] but creates circular dependency
        self.profile = None  # Should be UserProfile but creates circular dependency
    
    def add_order(self, order):  # Should be Order type
        """Add an order to this user."""
        self.orders.append(order)
        # This creates circular dependency:
        order.set_user(self)
    
    def get_total_spent(self) -> float:
        """Calculate total amount spent by user."""
        total = 0.0
        for order in self.orders:
            total += order.get_total_amount()
        return total
    
    def create_profile(self, profile_data: dict):
        """Create user profile - circular dependency."""
        # This would require importing UserProfile
        # from .profile import UserProfile
        # self.profile = UserProfile(self, profile_data)
        pass


# File 2: models/order.py (simulated)
"""
This represents the content of models/order.py which imports from user.py
"""

from typing import List, Dict, Any
from datetime import datetime
# This would create a circular import:
# from .user import User  # Circular import!
# from .product import Product  # Might be circular too

class Order:
    """Order model that references users and products."""
    
    def __init__(self, order_id: str):
        self.id = order_id
        self.user = None  # Should be User type but creates circular dependency
        self.items = []   # Should be List[OrderItem] 
        self.created_at = datetime.now()
        self.status = "pending"
        self.shipping_address = None
    
    def set_user(self, user):  # Should be User type
        """Set the user for this order."""
        self.user = user
        # Ensure user has this order (bidirectional relationship)
        if self not in user.orders:
            user.orders.append(self)
    
    def add_item(self, product, quantity: int):
        """Add product to order - creates dependency on Product."""
        # This creates circular dependency if Product imports Order
        item = OrderItem(product, quantity)
        self.items.append(item)
        
        # Update product's order history (circular dependency)
        product.add_to_order_history(self)
    
    def get_total_amount(self) -> float:
        """Calculate total order amount."""
        total = 0.0
        for item in self.items:
            total += item.get_subtotal()
        return total
    
    def set_shipping_address(self, address):
        """Set shipping address - might create dependency on Address."""
        # from .address import Address  # Potential circular import
        self.shipping_address = address


class OrderItem:
    """Individual item within an order."""
    
    def __init__(self, product, quantity: int):
        self.product = product  # Should be Product type
        self.quantity = quantity
        self.unit_price = product.price if hasattr(product, 'price') else 0
    
    def get_subtotal(self) -> float:
        """Get subtotal for this item."""
        return self.unit_price * self.quantity


# File 3: models/product.py (simulated)
"""
This represents the content of models/product.py which might import from order.py
"""

from typing import List, Optional
# Potential circular imports:
# from .order import Order  # If we want to track order history
# from .category import Category  # If Category imports Product

class Product:
    """Product model that might reference orders."""
    
    def __init__(self, product_id: str, name: str, price: float):
        self.id = product_id
        self.name = name
        self.price = price
        self.category = None  # Should be Category type
        self.order_history = []  # Should be List[Order] - circular dependency!
        self.related_products = []  # Self-referential relationship
    
    def add_to_order_history(self, order):  # Should be Order type
        """Add order to product's history - creates circular dependency."""
        if order not in self.order_history:
            self.order_history.append(order)
    
    def get_order_count(self) -> int:
        """Get number of times this product was ordered."""
        return len(self.order_history)
    
    def set_category(self, category):
        """Set product category - potential circular dependency."""
        # If Category class imports Product for product listings
        self.category = category
        category.add_product(self)  # This could create circular dependency
    
    def add_related_product(self, related_product: 'Product'):
        """Add related product - self-referential but not circular."""
        self.related_products.append(related_product)
        # Make relationship bidirectional
        if self not in related_product.related_products:
            related_product.related_products.append(self)


# File 4: models/category.py (simulated)
"""
This represents the content of models/category.py which imports from product.py
"""

from typing import List
# This creates circular import:
# from .product import Product  # Circular import!

class Category:
    """Category model that references products."""
    
    def __init__(self, category_id: str, name: str):
        self.id = category_id
        self.name = name
        self.products = []  # Should be List[Product] but creates circular dependency
        self.parent_category = None  # Self-referential
        self.subcategories = []  # Self-referential
    
    def add_product(self, product):  # Should be Product type
        """Add product to category - creates circular dependency."""
        if product not in self.products:
            self.products.append(product)
        
        # Ensure product knows its category
        if product.category != self:
            product.category = self
    
    def get_product_count(self) -> int:
        """Get number of products in category."""
        return len(self.products)
    
    def get_all_products(self) -> List:  # Should be List[Product]
        """Get all products including from subcategories."""
        all_products = self.products.copy()
        
        for subcategory in self.subcategories:
            all_products.extend(subcategory.get_all_products())
        
        return all_products


# File 5: models/profile.py (simulated)
"""
This represents the content of models/profile.py which imports from user.py
"""

from typing import Dict, Any, Optional
# This creates circular import:
# from .user import User  # Circular import!

class UserProfile:
    """User profile model that references user."""
    
    def __init__(self, user, profile_data: Dict[str, Any]):
        self.user = user  # Should be User type - circular dependency!
        self.bio = profile_data.get('bio', '')
        self.avatar_url = profile_data.get('avatar_url', '')
        self.preferences = profile_data.get('preferences', {})
        self.privacy_settings = profile_data.get('privacy_settings', {})
    
    def update_preferences(self, new_preferences: Dict[str, Any]):
        """Update user preferences."""
        self.preferences.update(new_preferences)
        # Notify user of preference changes (circular dependency)
        self.user.on_profile_updated()
    
    def get_public_info(self) -> Dict[str, Any]:
        """Get publicly visible profile information."""
        return {
            'user_id': self.user.id,
            'name': self.user.name,
            'bio': self.bio,
            'avatar_url': self.avatar_url
        }


# File 6: models/address.py (simulated)  
"""
This represents the content of models/address.py which might import from user.py or order.py
"""

from typing import List, Optional
# Potential circular imports:
# from .user import User  # If tracking address owners
# from .order import Order  # If tracking orders shipped to this address

class Address:
    """Address model that might reference users and orders."""
    
    def __init__(self, street: str, city: str, state: str, zip_code: str):
        self.street = street
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.users = []  # Should be List[User] - potential circular dependency
        self.orders = []  # Should be List[Order] - potential circular dependency
        self.is_verified = False
    
    def add_user(self, user):  # Should be User type
        """Associate user with this address."""
        if user not in self.users:
            self.users.append(user)
        
        # Add to user's addresses (bidirectional relationship)
        if hasattr(user, 'addresses') and self not in user.addresses:
            user.addresses.append(self)
    
    def add_order(self, order):  # Should be Order type
        """Associate order with this address."""
        if order not in self.orders:
            self.orders.append(order)


# Non-circular utility classes (these should NOT be detected as circular imports)

class ValidationUtils:
    """Utility class for validation - no circular dependencies."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format."""
        import re
        pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        return re.match(pattern, phone) is not None


class DateUtils:
    """Date utility class - no circular dependencies."""
    
    @staticmethod
    def format_date(date_obj) -> str:
        """Format date as string."""
        return date_obj.strftime('%Y-%m-%d')
    
    @staticmethod
    def parse_date(date_string: str):
        """Parse date string."""
        from datetime import datetime
        return datetime.strptime(date_string, '%Y-%m-%d')


class Constants:
    """Application constants - no dependencies."""
    
    DEFAULT_CURRENCY = 'USD'
    MAX_ORDER_ITEMS = 100
    DEFAULT_SHIPPING_COST = 9.99
    
    ORDER_STATUSES = [
        'pending',
        'confirmed', 
        'processing',
        'shipped',
        'delivered',
        'cancelled'
    ]
    
    USER_ROLES = [
        'customer',
        'admin',
        'moderator'
    ]