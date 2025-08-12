"""
Fixes needed for semantic_hv.py to work correctly with SemanticIndex
"""

# Issues found and fixes needed:

# 1. Line 169: self.index.add() should be self.index.add_function()
#    But add_function expects ast_node, not hypervector
#    Need to pass the node instead of pre-encoding

# 2. Line 152: encode_function() is being called with wrong parameters
#    Should check the actual signature

# 3. Line 179: self.index.find_all_similar_pairs() - need to verify this method exists

# 4. Lines 229-230: self.index.entries[self.index.id_to_index[id1]]
#    entries is Dict[str, FunctionEntry], not a list
#    Should be: self.index.entries[id1]

# 5. Line 313: for entry_id, hv, metadata in self.index.entries:
#    entries is a dict, need to iterate properly

# Let's create the fixed version:

FIXES = """
# Line 169 - Fix the add call
# OLD:
self.index.add(func_name, file, line, hv, {
    'features': features or {},
    'args': entry.get('args', [])
})

# NEW:
self.index.add_function(
    function_id=func_id,
    ast_node=node,
    file_path=file,
    line_number=line,
    metadata={
        'features': features or {},
        'args': entry.get('args', []),
        'name': func_name
    }
)

# Lines 145-172 - Remove pre-encoding since add_function will do it
# The entire encoding block can be simplified

# Line 179 - Check method name
# OLD: pairs = self.index.find_all_similar_pairs(limit=100)
# NEW: pairs = self.index.find_similar_pairs(top_k=100)

# Lines 229-231 - Fix entry access
# OLD:
self.index.entries[self.index.id_to_index[id1]][1],
self.index.entries[self.index.id_to_index[id2]][1],

# NEW:
self.index.entries[id1].file_path,
self.index.entries[id2].file_path,

# Line 313 - Fix iteration over entries
# OLD: for entry_id, hv, metadata in self.index.entries:
# NEW: for entry_id, entry in self.index.entries.items():
"""