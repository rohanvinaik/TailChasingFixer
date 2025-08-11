"""
Complex refactoring scenarios that should be fixed in 5-8 steps.
"""

from typing import Dict, List, Tuple
from .base import BenchmarkScenario


class ComplexRefactoringScenario(BenchmarkScenario):
    """Scenario requiring complex multi-file refactoring."""
    
    def __init__(self):
        super().__init__(
            name="complex_refactoring",
            description="Complex refactoring involving multiple patterns and files",
            expected_steps=(5, 8)
        )
    
    def setup(self) -> str:
        """Set up the complex refactoring scenario."""
        temp_dir = self.create_temp_directory()
        self.write_files(self.get_initial_code())
        return temp_dir
    
    def get_initial_code(self) -> Dict[str, str]:
        """Get initial code with multiple issues requiring complex refactoring."""
        return {
            "api/handlers.py": '''"""API handlers with multiple issues."""

# Missing imports
def handle_user_request(request):
    """Handle user API request."""
    try:
        # Parse JSON from request (missing json import)
        data = json.loads(request.body)
        
        # Validate data (missing validation)
        user_id = data.get("user_id")
        if not user_id:
            return {"error": "Missing user_id"}, 400
        
        # Get user from database (undefined function)
        user = fetch_user_from_db(user_id)
        
        if not user:
            return {"error": "User not found"}, 404
        
        # Process user data (duplicate logic)
        processed = {
            "id": user["id"],
            "name": user["name"].strip().title(),
            "email": user["email"].lower(),
            "status": "active" if user["active"] else "inactive"
        }
        
        return processed, 200
        
    except Exception as e:  # Bare except
        print(f"Error: {e}")  # Should use logging
        return {"error": "Internal error"}, 500

def handle_product_request(request):
    """Handle product API request."""
    try:
        # Duplicate JSON parsing logic
        data = json.loads(request.body)
        
        product_id = data.get("product_id")
        if not product_id:
            return {"error": "Missing product_id"}, 400
        
        # Get product (undefined function)
        product = fetch_product_from_db(product_id)
        
        if not product:
            return {"error": "Product not found"}, 404
        
        # Process product data (similar to user processing)
        processed = {
            "id": product["id"],
            "name": product["name"].strip().title(),
            "price": product["price"],
            "status": "available" if product["in_stock"] else "out_of_stock"
        }
        
        return processed, 200
        
    except Exception as e:  # Another bare except
        print(f"Error: {e}")
        return {"error": "Internal error"}, 500

def handle_order_request(request):
    """Handle order API request."""
    # Yet another duplicate implementation
    try:
        data = json.loads(request.body)
        
        order_id = data.get("order_id")
        if not order_id:
            return {"error": "Missing order_id"}, 400
        
        # Get order (undefined)
        order = fetch_order_from_db(order_id)
        
        if not order:
            return {"error": "Order not found"}, 404
        
        # Process order (similar pattern again)
        processed = {
            "id": order["id"],
            "customer": order["customer_name"].strip().title(),
            "total": order["total_amount"],
            "status": order["status"]
        }
        
        return processed, 200
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": "Internal error"}, 500
''',
            "database/queries.py": '''"""Database queries with issues."""

# Circular import potential
from ..api.handlers import handle_user_request

def get_user_by_id(user_id):
    """Get user from database."""
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    
    # Missing connection handling
    result = execute_query(query)
    
    if result:
        return result[0]
    return None

def get_product_by_id(product_id):
    """Get product from database."""
    # Another SQL injection
    query = f"SELECT * FROM products WHERE id = {product_id}"
    
    result = execute_query(query)
    
    if result:
        return result[0]
    return None

def get_order_by_id(order_id):
    """Get order from database."""
    # Yet another SQL injection
    query = f"SELECT * FROM orders WHERE id = {order_id}"
    
    result = execute_query(query)
    
    if result:
        return result[0]
    return None

# Duplicate function with slightly different name
def fetch_user_from_db(uid):
    """Fetch user data."""
    return get_user_by_id(uid)

def fetch_product_from_db(pid):
    """Fetch product data."""
    return get_product_by_id(pid)

def fetch_order_from_db(oid):
    """Fetch order data."""
    return get_order_by_id(oid)
''',
            "utils/validation.py": '''"""Validation utilities needed but missing proper implementation."""

def validate_request_data(data, required_fields):
    """Validate request data has required fields."""
    # This function should be used but isn't
    errors = []
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    return len(errors) == 0, errors

def validate_user_data(user_data):
    """Validate user data specifically."""
    # Duplicate validation logic
    if not user_data.get("user_id"):
        return False, ["Missing user_id"]
    
    if not user_data.get("email"):
        return False, ["Missing email"]
    
    # Email validation duplicated
    if "@" not in user_data.get("email", ""):
        return False, ["Invalid email"]
    
    return True, []

def validate_product_data(product_data):
    """Validate product data."""
    # More duplicate validation
    if not product_data.get("product_id"):
        return False, ["Missing product_id"]
    
    if not product_data.get("price"):
        return False, ["Missing price"]
    
    if product_data.get("price", 0) < 0:
        return False, ["Invalid price"]
    
    return True, []
''',
            "utils/formatting.py": '''"""Formatting utilities with duplication."""

def format_user_response(user):
    """Format user data for API response."""
    # Duplicate formatting logic from handlers
    return {
        "id": user["id"],
        "name": user["name"].strip().title(),
        "email": user["email"].lower(),
        "status": "active" if user["active"] else "inactive"
    }

def format_product_response(product):
    """Format product data for API response."""
    # More duplicate formatting
    return {
        "id": product["id"],
        "name": product["name"].strip().title(),
        "price": product["price"],
        "status": "available" if product["in_stock"] else "out_of_stock"
    }

def format_order_response(order):
    """Format order data for API response."""
    # Even more duplicate formatting
    return {
        "id": order["id"],
        "customer": order["customer_name"].strip().title(),
        "total": order["total_amount"],
        "status": order["status"]
    }

# Generic formatter that should be used
def format_response(data, field_mapping):
    """Generic response formatter."""
    result = {}
    
    for key, value in field_mapping.items():
        if callable(value):
            result[key] = value(data)
        else:
            result[key] = data.get(value)
    
    return result
'''
        }
    
    def get_expected_solution(self) -> Dict[str, str]:
        """Get expected solution after complex refactoring."""
        return {
            "api/handlers.py": '''"""API handlers refactored to use proper patterns."""

import json
import logging
from typing import Dict, Tuple, Any

from ..database.queries import get_user_by_id, get_product_by_id, get_order_by_id
from ..utils.validation import validate_request_data
from ..utils.formatting import format_user_response, format_product_response, format_order_response

logger = logging.getLogger(__name__)

def parse_request_body(request) -> Dict[str, Any]:
    """Parse JSON from request body."""
    try:
        return json.loads(request.body)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse request body: {e}")
        return {}

def handle_entity_request(request, entity_type: str, id_field: str, 
                         fetch_func, format_func) -> Tuple[Dict, int]:
    """Generic handler for entity requests."""
    try:
        data = parse_request_body(request)
        
        # Validate required field
        is_valid, errors = validate_request_data(data, [id_field])
        if not is_valid:
            return {"error": errors[0]}, 400
        
        entity_id = data[id_field]
        
        # Fetch entity from database
        entity = fetch_func(entity_id)
        
        if not entity:
            return {"error": f"{entity_type} not found"}, 404
        
        # Format response
        processed = format_func(entity)
        
        return processed, 200
        
    except Exception as e:
        logger.error(f"Error handling {entity_type} request: {e}")
        return {"error": "Internal error"}, 500

def handle_user_request(request) -> Tuple[Dict, int]:
    """Handle user API request."""
    return handle_entity_request(
        request, "User", "user_id",
        get_user_by_id, format_user_response
    )

def handle_product_request(request) -> Tuple[Dict, int]:
    """Handle product API request."""
    return handle_entity_request(
        request, "Product", "product_id",
        get_product_by_id, format_product_response
    )

def handle_order_request(request) -> Tuple[Dict, int]:
    """Handle order API request."""
    return handle_entity_request(
        request, "Order", "order_id",
        get_order_by_id, format_order_response
    )
''',
            "database/queries.py": '''"""Database queries with proper parameterization."""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def execute_safe_query(query: str, params: tuple) -> list:
    """Execute parameterized query safely."""
    # This would use actual database connection
    # For now, placeholder implementation
    logger.debug(f"Executing query: {query} with params: {params}")
    return []

def get_entity_by_id(table: str, entity_id: Any) -> Optional[Dict]:
    """Generic function to get entity by ID."""
    query = f"SELECT * FROM {table} WHERE id = %s"
    
    try:
        result = execute_safe_query(query, (entity_id,))
        if result:
            return result[0]
    except Exception as e:
        logger.error(f"Database query failed: {e}")
    
    return None

def get_user_by_id(user_id: Any) -> Optional[Dict]:
    """Get user from database."""
    return get_entity_by_id("users", user_id)

def get_product_by_id(product_id: Any) -> Optional[Dict]:
    """Get product from database."""
    return get_entity_by_id("products", product_id)

def get_order_by_id(order_id: Any) -> Optional[Dict]:
    """Get order from database."""
    return get_entity_by_id("orders", order_id)
''',
            "utils/validation.py": '''"""Validation utilities with consolidated logic."""

from typing import Tuple, List, Dict, Any

def validate_request_data(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Validate request data has required fields."""
    errors = []
    
    if not data:
        errors.append("No data provided")
        return False, errors
    
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    return len(errors) == 0, errors

def validate_email(email: str) -> bool:
    """Validate email format."""
    return email and "@" in email and "." in email.split("@")[1]

def validate_price(price: Any) -> bool:
    """Validate price is valid."""
    try:
        return float(price) >= 0
    except (TypeError, ValueError):
        return False

def validate_user_data(user_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate user data specifically."""
    is_valid, errors = validate_request_data(user_data, ["user_id"])
    
    if "email" in user_data and not validate_email(user_data["email"]):
        errors.append("Invalid email format")
        is_valid = False
    
    return is_valid, errors

def validate_product_data(product_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate product data."""
    is_valid, errors = validate_request_data(product_data, ["product_id"])
    
    if "price" in product_data and not validate_price(product_data["price"]):
        errors.append("Invalid price")
        is_valid = False
    
    return is_valid, errors
''',
            "utils/formatting.py": '''"""Formatting utilities with unified approach."""

from typing import Dict, Any, Callable

def safe_format_string(value: Any) -> str:
    """Safely format string value."""
    if not value:
        return ""
    return str(value).strip().title()

def format_response(data: Dict, field_mapping: Dict[str, Any]) -> Dict:
    """Generic response formatter."""
    result = {}
    
    for key, value in field_mapping.items():
        if callable(value):
            result[key] = value(data)
        elif isinstance(value, str):
            result[key] = data.get(value)
        else:
            result[key] = value
    
    return result

def format_user_response(user: Dict) -> Dict:
    """Format user data for API response."""
    return format_response(user, {
        "id": "id",
        "name": lambda u: safe_format_string(u.get("name")),
        "email": lambda u: u.get("email", "").lower(),
        "status": lambda u: "active" if u.get("active") else "inactive"
    })

def format_product_response(product: Dict) -> Dict:
    """Format product data for API response."""
    return format_response(product, {
        "id": "id",
        "name": lambda p: safe_format_string(p.get("name")),
        "price": "price",
        "status": lambda p: "available" if p.get("in_stock") else "out_of_stock"
    })

def format_order_response(order: Dict) -> Dict:
    """Format order data for API response."""
    return format_response(order, {
        "id": "id",
        "customer": lambda o: safe_format_string(o.get("customer_name")),
        "total": "total_amount",
        "status": "status"
    })
'''
        }
    
    def validate_solution(self, current_code: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate if complex refactoring is properly done."""
        errors = []
        
        # Check handlers.py
        if "api/handlers.py" in current_code:
            content = current_code["api/handlers.py"]
            
            # Should have proper imports
            if "import json" not in content:
                errors.append("api/handlers.py: Missing json import")
            
            if "import logging" not in content and "logger" in content:
                errors.append("api/handlers.py: Missing logging import")
            
            # Should not have bare excepts
            if "except:" in content or "except Exception:" in content:
                if "except Exception as" not in content:
                    errors.append("api/handlers.py: Contains bare except clauses")
            
            # Should not have print statements
            if "print(" in content:
                errors.append("api/handlers.py: Contains print statements instead of logging")
            
            # Should have consolidated handler logic
            if content.count("json.loads(request.body)") > 1:
                errors.append("api/handlers.py: Duplicate request parsing logic")
        
        # Check database/queries.py
        if "database/queries.py" in current_code:
            content = current_code["database/queries.py"]
            
            # Should not have SQL injection vulnerabilities
            if 'f"SELECT * FROM' in content and "%s" not in content:
                errors.append("database/queries.py: Contains SQL injection vulnerabilities")
            
            # Should not have circular imports
            if "from ..api.handlers import" in content:
                errors.append("database/queries.py: Contains circular import from api.handlers")
            
            # Should have parameterized queries
            if "execute_query(query)" in content and "%s" not in content:
                errors.append("database/queries.py: Missing parameterized queries")
        
        # Check utils/validation.py
        if "utils/validation.py" in current_code:
            content = current_code["utils/validation.py"]
            
            # Should have consolidated validation
            duplicate_validations = content.count('if not') + content.count('if field not in')
            if duplicate_validations > 10:
                errors.append("utils/validation.py: Too much duplicate validation logic")
        
        # Check utils/formatting.py
        if "utils/formatting.py" in current_code:
            content = current_code["utils/formatting.py"]
            
            # Should use generic formatter
            if "format_response" not in content:
                errors.append("utils/formatting.py: Missing generic format_response function")
            
            # Should not have too much duplication
            strip_count = content.count('.strip()')
            if strip_count > 5:
                errors.append("utils/formatting.py: Too much duplicate string formatting")
        
        # Validate syntax for all files
        import ast
        for file_path, content in current_code.items():
            if file_path.endswith(".py"):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"Syntax error in {file_path}: {e}")
        
        return len(errors) == 0, errors