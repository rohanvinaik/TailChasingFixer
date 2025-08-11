"""
Import Anxiety Scenario - Defensive over-importing pattern.

This demonstrates the tail-chasing pattern where an LLM:
1. Gets an import error
2. Defensively imports everything it might need
3. Adds star imports "just in case"
4. Imports unused modules to prevent future errors
5. Creates import cycles through over-eager importing
"""

# LLM imports everything it has seen in similar code
import os
import sys
import json
import yaml
import csv
import xml.etree.ElementTree as ET
import pickle
import shelve
import sqlite3
import urllib
import requests
import socket
import threading
import multiprocessing
import asyncio
import time
import datetime
import calendar
import random
import math
import statistics
import decimal
import fractions
import itertools
import functools
import operator
import collections
import heapq
import bisect
import array
import weakref
import types
import copy
import pprint
import re
import difflib
import textwrap
import unicodedata
import stringprep
import logging
import warnings
import traceback
import inspect
import dis

# Star imports because LLM is unsure what's needed
from typing import *  # Import anxiety - imports everything
from collections import *  # Just in case we need any collection
from itertools import *  # Might need some iteration tools
from functools import *  # Could need functional tools
from pathlib import *  # Paths are always useful, right?
from dataclasses import *  # Modern Python uses these
from enum import *  # Enums might be needed
from abc import *  # Abstract classes just in case

# Import specific items that might be used
from typing import (
    Dict, List, Set, Tuple, Optional, Union, Any, 
    Callable, Iterable, Iterator, Generator, TypeVar,
    Generic, Protocol, ClassVar, Final, Literal,
    TypedDict, NamedTuple, cast, overload,
    get_type_hints, get_origin, get_args
)

# Defensive imports for error handling
try:
    import numpy as np
except ImportError:
    np = None  # LLM creates fallback

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

# Import from local modules defensively
try:
    from .utils import *  # Star import from local module
except ImportError:
    pass  # Ignore if module doesn't exist

try:
    from ..core import *  # Import from parent package
except ImportError:
    pass

try:
    from ...lib import *  # Even more parent imports
except ImportError:
    pass


class DataProcessor:
    """Class that only needs a fraction of the imports."""
    
    def __init__(self):
        # Only actually uses these imports
        self.data = []
        self.logger = logging.getLogger(__name__)
    
    def process(self, filename: str) -> List[Dict[str, Any]]:
        """Process a JSON file."""
        # Only needs json and typing, but imported 50+ modules
        with open(filename, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            # Simple processing that doesn't need all those imports
            if isinstance(item, dict):
                results.append(item)
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to file."""
        # Again, just needs json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


class ConfigManager:
    """Another class with import anxiety."""
    
    def __init__(self):
        # Imports that might be needed for config
        self.config = {}
        
        # Check which libraries are available (defensive programming)
        self.has_yaml = 'yaml' in sys.modules
        self.has_toml = 'toml' in sys.modules
        self.has_ini = 'configparser' in sys.modules
        
    def load_config(self, path: str) -> Dict:
        """Load configuration from file."""
        # Over-engineered to handle every possible format
        if path.endswith('.json'):
            with open(path) as f:
                return json.load(f)
        elif path.endswith('.yaml') and self.has_yaml:
            with open(path) as f:
                return yaml.safe_load(f)
        elif path.endswith('.py'):
            # Import Python file as config (dangerous!)
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            return vars(config)
        else:
            # Fall back to JSON
            with open(path) as f:
                return json.load(f)


# Circular import anxiety - each module imports the others defensively

class ServiceA:
    """Service that imports other services."""
    
    def __init__(self):
        # Circular import through defensive importing
        from . import ServiceB  # Import inside method
        from . import ServiceC
        
        self.service_b = ServiceB()
        self.service_c = ServiceC()


class ServiceB:
    """Service B with import anxiety."""
    
    def __init__(self):
        # More circular imports
        from . import ServiceA  # Circular!
        from . import ServiceC
        
        # Defensive type checking
        if TYPE_CHECKING:  # This isn't even defined properly
            from . import ServiceA as ServiceAType


class ServiceC:
    """Service C completing the circle."""
    
    def __init__(self):
        # Complete the circular dependency
        from . import ServiceA
        from . import ServiceB


# Utility functions that don't need most imports

def simple_add(a: int, b: int) -> int:
    """Function that doesn't need 50+ imports."""
    return a + b


def read_file(path: str) -> str:
    """Read file - only needs built-in open."""
    with open(path, 'r') as f:
        return f.read()


# Signs of import anxiety:
# 1. Massive import block at top of file
# 2. Star imports from multiple modules
# 3. Try/except blocks around imports with fallbacks
# 4. Imports inside functions/methods (circular import fear)
# 5. Conditional imports based on availability
# 6. Importing entire standard library "just in case"
# 7. Most imports are never actually used

def main():
    """Example usage showing only a few imports are needed."""
    processor = DataProcessor()
    
    # Only uses: json, open, List, Dict
    results = processor.process("data.json")
    processor.save_results(results, "output.json")
    
    # 95% of imports were unnecessary


if __name__ == "__main__":
    # Even imports modules for main block
    import argparse  # Imported here due to anxiety
    import sys  # Already imported but does it again
    
    main()