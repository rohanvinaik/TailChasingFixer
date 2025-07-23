"""
Demo: E-commerce order processing system with tail-chasing patterns.

This demo showcases various tail-chasing anti-patterns that often occur
when LLMs assist in development without full context.
"""

# ========== File: order_processor.py ==========

class OrderProcessor:
    """Main order processing logic."""
    
    def process_order(self, order_data):
        """Process an incoming order."""
        # Validate order
        if not self.validate_order(order_data):
            return False
        
        # Calculate total
        total = self.calculate_total(order_data)
        
        # Apply discounts
        final_price = self.apply_discounts(order_data, total)
        
        # Process payment
        return self.process_payment(order_data, final_price)
    
    def validate_order(self, order):
        """Validate order data."""
        # TODO: Implement validation
        pass
    
    def calculate_total(self, order):
        """Calculate order total."""
        total = 0
        for item in order.get('items', []):
            total += item['price'] * item['quantity']
        return total
    
    def apply_discounts(self, order, total):
        """Apply discounts to order."""
        # Phantom implementation
        raise NotImplementedError("Discount system not implemented")
    
    def process_payment(self, order, amount):
        """Process payment for order."""
        pass


# ========== Semantic Duplicate 1 ==========
def compute_order_total(order_data):
    """Compute the total for an order."""
    sum_total = 0
    items = order_data.get('items', [])
    for item in items:
        sum_total = sum_total + (item['price'] * item['quantity'])
    return sum_total


# ========== Semantic Duplicate 2 (Different implementation, same logic) ==========
def get_order_sum(order):
    """Get sum of all items in order."""
    return sum(it['price'] * it['quantity'] for it in order.get('items', []))


# ========== Wrapper Function (Trivial abstraction) ==========
def calculate_order_total_wrapper(order):
    """Wrapper for order total calculation."""
    return compute_order_total(order)


# ========== Another wrapper with slight modification ==========
def safe_calculate_total(order_data):
    """Safely calculate order total."""
    if not order_data:
        return 0
    return compute_order_total(order_data)


# ========== Rename cascade - same function, different names ==========
def process_order_items(items):
    """Process items in an order."""
    validated_items = []
    for item in items:
        if item.get('quantity', 0) > 0:
            validated_items.append(item)
    return validated_items

def handle_order_items(items):
    """Handle order items."""  # Just renamed version
    validated_items = []
    for item in items:
        if item.get('quantity', 0) > 0:
            validated_items.append(item)
    return validated_items

def validate_items_list(items):
    """Validate list of items."""  # Another rename
    validated_items = []
    for item in items:
        if item.get('quantity', 0) > 0:
            validated_items.append(item)
    return validated_items