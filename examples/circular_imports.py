"""
Example: Circular Import Anti-Pattern

This demonstrates circular dependencies that arise when LLMs add imports
reactively without considering module architecture.
"""

# FILE: user_service.py
"""
from database import get_user_db  # Imports from database
from validators import validate_user  # Circular!

class UserService:
    def __init__(self):
        self.db = get_user_db()
    
    def create_user(self, data):
        if validate_user(data):
            return self.db.save(data)
        raise ValueError("Invalid user data")
    
    def get_user_by_email(self, email):
        # Used by validators.py
        return self.db.find_one({"email": email})
"""

# FILE: validators.py
"""
from user_service import UserService  # Circular import!

def validate_user(data):
    # Check if email already exists
    service = UserService()
    existing = service.get_user_by_email(data["email"])
    
    if existing:
        return False
    
    # Other validation logic
    return True
"""

# FILE: database.py
"""
from user_service import UserService  # Another circular dependency!

def get_user_db():
    # Returns database connection
    return MongoDB("users")

def migrate_users():
    service = UserService()
    # Migration logic using service
    pass
"""

# FIXED VERSION:

# FILE: models.py (New shared module)
"""
# Shared data models
class User:
    def __init__(self, email, name):
        self.email = email
        self.name = name
"""

# FILE: database_fixed.py
"""
# No imports from user_service
def get_user_db():
    return MongoDB("users")

class UserRepository:
    def __init__(self):
        self.db = get_user_db()
    
    def find_by_email(self, email):
        return self.db.find_one({"email": email})
    
    def save(self, user_data):
        return self.db.save(user_data)
"""

# FILE: validators_fixed.py
"""
from database_fixed import UserRepository

def validate_user(data):
    # Use repository instead of service
    repo = UserRepository()
    existing = repo.find_by_email(data["email"])
    
    if existing:
        return False
    
    return True
"""

# FILE: user_service_fixed.py
"""
from database_fixed import UserRepository
from validators_fixed import validate_user

class UserService:
    def __init__(self):
        self.repo = UserRepository()
    
    def create_user(self, data):
        if validate_user(data):
            return self.repo.save(data)
        raise ValueError("Invalid user data")
"""

# How TailChasingFixer detects this:
#
# $ tailchasing analyze --deep
#
# Issue: circular_import
# Severity: 4 (High)
# Circular dependency chain detected:
#   user_service.py -> validators.py -> user_service.py
#   user_service.py -> database.py -> user_service.py
# 
# Suggested fix:
# 1. Extract shared functionality to a separate module
# 2. Use dependency injection or repository pattern
# 3. Move validation logic to a standalone module