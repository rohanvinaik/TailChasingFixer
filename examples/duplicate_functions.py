"""
Example: Duplicate Function Anti-Pattern

This file demonstrates the duplicate function pattern that commonly
occurs when LLMs recreate existing functionality with different names.
"""

# BEFORE: Multiple functions doing the same thing

def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total


def compute_total(values):
    """Compute the total of values."""
    result = 0
    for val in values:
        result = result + val
    return result


def add_all_numbers(number_list):
    """Add all numbers in the list."""
    sum_value = 0
    for n in number_list:
        sum_value += n
    return sum_value


# These are semantic duplicates with slight variations
def sum_with_initial(items, start=0):
    """Sum items starting from an initial value."""
    accumulator = start
    for item in items:
        accumulator += item
    return accumulator


def aggregate_values(data, initial=0):
    """Aggregate values with an initial value."""
    total = initial
    for d in data:
        total = total + d
    return total


# AFTER: Consolidated into a single flexible function

def calculate_sum_fixed(numbers, initial=0):
    """
    Calculate the sum of numbers with optional initial value.
    
    Args:
        numbers: Iterable of numbers to sum
        initial: Starting value (default: 0)
        
    Returns:
        Sum of all numbers plus initial value
    """
    return sum(numbers, initial)


# Example of how TailChasingFixer would detect this:
# 
# $ tailchasing analyze examples/duplicate_functions.py
# 
# Found 5 issue(s)
# 
# Issue #1 (Severity: 3)
# duplicate_function
# File: examples/duplicate_functions.py:9
# Message: Function 'calculate_sum' is a duplicate of 'compute_total'
# 
# Issue #2 (Severity: 3)
# semantic_duplicate
# File: examples/duplicate_functions.py:25
# Message: Function 'sum_with_initial' is semantically identical to 'aggregate_values'
# Similarity: 0.92