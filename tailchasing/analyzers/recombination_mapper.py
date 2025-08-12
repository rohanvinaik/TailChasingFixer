"""
Recombination mapping for co-edit analysis using genetic linkage metaphors.

This module implements biologically-inspired algorithms to track co-evolution patterns
in codebases, mapping genetic recombination concepts to code file relationships:

- Co-edit frequency analysis (genetic linkage)
- Linkage disequilibrium computation (recombination mapping)
- Extraction ROI prediction using LD patterns
- Insulator boundary generation for module isolation

Based on population genetics principles applied to code evolution.
"""

import ast
import subprocess
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re

from .base import AnalysisContext
from ..utils.logging_setup import get_logger


@dataclass
class CoEditEvent:
    """Represents a co-edit event between files (genetic linkage)."""
    commit_hash: str
    timestamp: datetime
    files: Set[str]
    author: str
    message: str
    lines_changed: Dict[str, int]  # file -> lines changed


@dataclass
class LinkageGroup:
    """Group of files with high co-edit frequency (genetic linkage group)."""
    files: Set[str]
    coedit_frequency: float  # How often these files change together
    recombination_rate: float  # 1 - linkage strength
    extraction_candidate: bool = False
    shared_functionality: List[str] = field(default_factory=list)


@dataclass 
class ExtractionROI:
    """Region of Interest for helper extraction (recombination hotspot)."""
    target_files: Set[str]
    shared_symbols: Set[str]
    extraction_benefit: float  # Cost-benefit score for extraction
    proposed_helper_name: str
    insulation_score: float  # How well isolated the extraction would be
    boundary_suggestions: List[str] = field(default_factory=list)


@dataclass
class ModuleBoundary:
    """Module boundary with insulation properties."""
    module_path: str
    boundary_type: str  # 'export', 'facade', 'common'
    insulation_score: float  # How well this boundary isolates the module
    suggested_exports: Set[str] = field(default_factory=set)
    facade_content: Optional[str] = None
    common_extractions: Set[str] = field(default_factory=set)


class RecombinationMapper:
    """
    Maps genetic recombination patterns to code co-edit analysis.
    
    Uses population genetics concepts to analyze code evolution:
    - Linkage analysis: Files that change together frequently
    - Recombination mapping: Files that change independently
    - Linkage disequilibrium: Non-random association of changes
    - Extraction prediction: Where helper modules would be beneficial
    """
    
    def __init__(self, max_commits: int = 1000, time_window_days: int = 180):
        """
        Initialize recombination mapper.
        
        Args:
            max_commits: Maximum commits to analyze
            time_window_days: Time window for co-edit analysis
        """
        self.max_commits = max_commits
        self.time_window_days = time_window_days
        self.logger = get_logger(__name__)
        
        # Caches for expensive computations
        self._coedit_cache: Dict[str, np.ndarray] = {}
        self._ld_cache: Dict[str, np.ndarray] = {}
        self._git_history_cache: List[CoEditEvent] = []
        
    def build_coedit_matrix(self, git_history: List[CoEditEvent], 
                          file_list: List[str]) -> np.ndarray:
        """
        Build co-edit frequency matrix tracking which files change together.
        
        Implements genetic linkage analysis where high co-edit frequency
        indicates tight genetic linkage (low recombination).
        
        Args:
            git_history: List of co-edit events from git
            file_list: List of files to analyze
            
        Returns:
            Symmetric matrix where M[i,j] = co-edit frequency of files i,j
        """
        self.logger.info(f"Building co-edit matrix for {len(file_list)} files from {len(git_history)} commits")
        
        # Create file index mapping
        file_to_idx = {file: i for i, file in enumerate(file_list)}
        n_files = len(file_list)
        
        # Initialize co-edit matrix
        coedit_matrix = np.zeros((n_files, n_files))
        file_change_counts = np.zeros(n_files)
        
        # Process each co-edit event
        for event in git_history:
            changed_files = [f for f in event.files if f in file_to_idx]
            changed_indices = [file_to_idx[f] for f in changed_files]
            
            # Update individual file change counts
            for idx in changed_indices:
                file_change_counts[idx] += 1
            
            # Update co-edit counts for all pairs
            for i, idx_i in enumerate(changed_indices):
                for j, idx_j in enumerate(changed_indices):
                    if i != j:  # Don't count self-changes
                        coedit_matrix[idx_i, idx_j] += 1
        
        # Normalize to co-edit frequencies
        # Frequency = P(A and B change together | A changes)
        for i in range(n_files):
            for j in range(n_files):
                if i != j and file_change_counts[i] > 0:
                    coedit_matrix[i, j] = coedit_matrix[i, j] / file_change_counts[i]
        
        # Make matrix symmetric (max of both directions)
        for i in range(n_files):
            for j in range(i+1, n_files):
                max_freq = max(coedit_matrix[i, j], coedit_matrix[j, i])
                coedit_matrix[i, j] = max_freq
                coedit_matrix[j, i] = max_freq
        
        # Set diagonal to 1.0 (file always changes with itself)
        np.fill_diagonal(coedit_matrix, 1.0)
        
        self.logger.info(f"Co-edit matrix built: mean frequency = {np.mean(coedit_matrix):.3f}")
        return coedit_matrix
    
    def compute_linkage_disequilibrium(self, coedit_matrix: np.ndarray, 
                                     file_list: List[str]) -> np.ndarray:
        """
        Compute linkage disequilibrium matrix for recombination mapping.
        
        LD measures non-random association between file changes.
        High LD = low recombination = files strongly linked.
        Low LD = high recombination = files change independently.
        
        LD(A,B) = P(A∩B) - P(A)×P(B)
        Normalized LD = LD / sqrt(P(A)×(1-P(A))×P(B)×(1-P(B)))
        
        Args:
            coedit_matrix: Co-edit frequency matrix
            file_list: List of file names
            
        Returns:
            LD matrix where positive values = linkage, negative = repulsion
        """
        self.logger.info("Computing linkage disequilibrium matrix")
        
        n_files = len(file_list)
        ld_matrix = np.zeros((n_files, n_files))
        
        # Extract marginal probabilities (diagonal represents individual change freq)
        marginal_probs = np.diag(coedit_matrix.copy())
        
        # Compute pairwise LD values
        for i in range(n_files):
            for j in range(n_files):
                if i != j:
                    # P(A∩B) = observed co-edit frequency
                    p_joint = coedit_matrix[i, j]
                    
                    # P(A) × P(B) = expected under independence
                    p_expected = marginal_probs[i] * marginal_probs[j]
                    
                    # Raw LD = observed - expected
                    raw_ld = p_joint - p_expected
                    
                    # Normalize LD to [-1, 1] scale
                    if marginal_probs[i] > 0 and marginal_probs[j] > 0:
                        max_ld = min(
                            marginal_probs[i] * (1 - marginal_probs[j]),
                            marginal_probs[j] * (1 - marginal_probs[i])
                        )
                        min_ld = -marginal_probs[i] * marginal_probs[j]
                        
                        if raw_ld >= 0 and max_ld > 0:
                            ld_matrix[i, j] = raw_ld / max_ld
                        elif raw_ld < 0 and min_ld < 0:
                            ld_matrix[i, j] = raw_ld / abs(min_ld)
                        else:
                            ld_matrix[i, j] = 0.0
                    else:
                        ld_matrix[i, j] = 0.0
        
        # Set diagonal to 1.0 (perfect linkage with self)
        np.fill_diagonal(ld_matrix, 1.0)
        
        self.logger.info(f"LD matrix computed: mean LD = {np.mean(ld_matrix):.3f}")
        return ld_matrix
    
    def predict_extraction_roi(self, linkage_groups: List[LinkageGroup], 
                             ld_matrix: np.ndarray, 
                             file_list: List[str],
                             context: AnalysisContext) -> List[ExtractionROI]:
        """
        Predict extraction regions of interest using linkage disequilibrium.
        
        Identifies recombination hotspots where helper extraction would
        break up tight linkage groups and improve modularity.
        
        Args:
            linkage_groups: Groups of highly linked files
            ld_matrix: Linkage disequilibrium matrix
            file_list: List of file names
            context: Analysis context for symbol information
            
        Returns:
            List of extraction ROI recommendations
        """
        self.logger.info(f"Predicting extraction ROI from {len(linkage_groups)} linkage groups")
        
        extraction_rois = []
        file_to_idx = {file: i for i, file in enumerate(file_list)}
        
        for group in linkage_groups:
            if len(group.files) < 2:
                continue
                
            # Get indices for files in this linkage group
            group_indices = [file_to_idx[f] for f in group.files if f in file_to_idx]
            
            if len(group_indices) < 2:
                continue
            
            # Extract LD submatrix for this group
            group_ld = ld_matrix[np.ix_(group_indices, group_indices)]
            
            # Analyze shared symbols across files in group
            shared_symbols = self._find_shared_symbols(group.files, context)
            
            # Compute extraction benefit
            extraction_benefit = self._compute_extraction_benefit(
                group, group_ld, shared_symbols
            )
            
            # Only suggest extraction if benefit is significant
            if extraction_benefit > 0.3:  # Threshold for worthwhile extraction
                # Generate helper module name
                helper_name = self._generate_helper_name(group.files, shared_symbols)
                
                # Compute insulation score
                insulation_score = self._compute_group_insulation_score(
                    group_indices, ld_matrix
                )
                
                # Generate boundary suggestions
                boundary_suggestions = self._generate_boundary_suggestions(
                    group.files, shared_symbols
                )
                
                roi = ExtractionROI(
                    target_files=group.files,
                    shared_symbols=shared_symbols,
                    extraction_benefit=extraction_benefit,
                    proposed_helper_name=helper_name,
                    insulation_score=insulation_score,
                    boundary_suggestions=boundary_suggestions
                )
                
                extraction_rois.append(roi)
                self.logger.debug(f"ROI identified: {helper_name} "
                                f"(benefit: {extraction_benefit:.2f}, "
                                f"insulation: {insulation_score:.2f})")
        
        # Sort by extraction benefit (highest first)
        extraction_rois.sort(key=lambda roi: roi.extraction_benefit, reverse=True)
        
        self.logger.info(f"Identified {len(extraction_rois)} extraction ROIs")
        return extraction_rois
    
    def extract_git_history(self, working_directory: Path) -> List[CoEditEvent]:
        """
        Extract git history for co-edit analysis.
        
        Args:
            working_directory: Project root directory
            
        Returns:
            List of co-edit events from git history
        """
        if self._git_history_cache:
            return self._git_history_cache
        
        self.logger.info(f"Extracting git history from {working_directory}")
        
        try:
            # Get git log with file changes
            since_date = datetime.now() - timedelta(days=self.time_window_days)
            since_str = since_date.strftime('%Y-%m-%d')
            
            cmd = [
                'git', 'log',
                f'--since={since_str}',
                f'--max-count={self.max_commits}',
                '--name-status',
                '--format=%H|%ai|%an|%s',
                '--no-merges',  # Skip merge commits
                '--',
                '*.py'  # Only Python files
            ]
            
            result = subprocess.run(
                cmd, cwd=working_directory, 
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Git command failed: {result.stderr}")
                return []
            
            # Parse git log output
            events = self._parse_git_log(result.stdout)
            self._git_history_cache = events
            
            self.logger.info(f"Extracted {len(events)} co-edit events")
            return events
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            self.logger.error(f"Failed to extract git history: {e}")
            return []
    
    def identify_linkage_groups(self, coedit_matrix: np.ndarray, 
                              file_list: List[str],
                              threshold: float = 0.6) -> List[LinkageGroup]:
        """
        Identify linkage groups from co-edit matrix.
        
        Args:
            coedit_matrix: Co-edit frequency matrix
            file_list: List of file names
            threshold: Minimum co-edit frequency for linkage
            
        Returns:
            List of identified linkage groups
        """
        self.logger.info(f"Identifying linkage groups with threshold {threshold}")
        
        # Create graph of linked files
        n_files = len(file_list)
        linkage_graph = nx.Graph()
        
        # Add nodes
        for i, file in enumerate(file_list):
            linkage_graph.add_node(i, file=file)
        
        # Add edges for linked files
        for i in range(n_files):
            for j in range(i+1, n_files):
                if coedit_matrix[i, j] >= threshold:
                    linkage_graph.add_edge(i, j, weight=coedit_matrix[i, j])
        
        # Find connected components (linkage groups)
        components = list(nx.connected_components(linkage_graph))
        
        linkage_groups = []
        for component in components:
            if len(component) >= 2:  # Minimum group size
                files = {file_list[i] for i in component}
                
                # Compute average co-edit frequency for this group
                group_indices = list(component)
                frequencies = []
                for i in range(len(group_indices)):
                    for j in range(i+1, len(group_indices)):
                        idx_i, idx_j = group_indices[i], group_indices[j]
                        frequencies.append(coedit_matrix[idx_i, idx_j])
                
                avg_frequency = np.mean(frequencies) if frequencies else 0.0
                recombination_rate = 1.0 - avg_frequency  # High frequency = low recombination
                
                group = LinkageGroup(
                    files=files,
                    coedit_frequency=avg_frequency,
                    recombination_rate=recombination_rate
                )
                linkage_groups.append(group)
        
        self.logger.info(f"Identified {len(linkage_groups)} linkage groups")
        return linkage_groups
    
    # === Helper Methods ===
    
    def _parse_git_log(self, git_output: str) -> List[CoEditEvent]:
        """Parse git log output into co-edit events."""
        events = []
        lines = git_output.strip().split('\n')
        
        current_commit = None
        current_files = set()
        lines_changed = {}
        
        for line in lines:
            if not line:
                # Empty line - finalize current commit
                if current_commit and current_files:
                    events.append(CoEditEvent(
                        commit_hash=current_commit['hash'],
                        timestamp=current_commit['timestamp'],
                        files=current_files.copy(),
                        author=current_commit['author'],
                        message=current_commit['message'],
                        lines_changed=lines_changed.copy()
                    ))
                current_commit = None
                current_files = set()
                lines_changed = {}
                continue
            
            if '|' in line:
                # Commit header
                parts = line.split('|', 3)
                if len(parts) == 4:
                    hash_val, timestamp_str, author, message = parts
                    timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                    
                    current_commit = {
                        'hash': hash_val,
                        'timestamp': timestamp,
                        'author': author,
                        'message': message
                    }
            else:
                # File change line
                parts = line.split('\t')
                if len(parts) >= 2:
                    change_type = parts[0]
                    file_path = parts[-1]
                    
                    # Only include Python files
                    if file_path.endswith('.py'):
                        current_files.add(file_path)
                        
                        # Estimate lines changed based on change type
                        if change_type == 'A':  # Added
                            lines_changed[file_path] = 50  # Estimate
                        elif change_type == 'D':  # Deleted
                            lines_changed[file_path] = 0
                        elif change_type == 'M':  # Modified
                            lines_changed[file_path] = 10  # Estimate
                        else:
                            lines_changed[file_path] = 5   # Default
        
        # Handle last commit
        if current_commit and current_files:
            events.append(CoEditEvent(
                commit_hash=current_commit['hash'],
                timestamp=current_commit['timestamp'],
                files=current_files.copy(),
                author=current_commit['author'],
                message=current_commit['message'],
                lines_changed=lines_changed.copy()
            ))
        
        return events
    
    def _find_shared_symbols(self, files: Set[str], context: AnalysisContext) -> Set[str]:
        """Find symbols shared across files in a linkage group."""
        if not files:
            return set()
        
        # Get all symbols from each file
        file_symbols = {}
        for file_path in files:
            if file_path in context.ast_index:
                symbols = set()
                try:
                    tree = context.ast_index[file_path]
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            symbols.add(node.name)
                        elif isinstance(node, ast.ClassDef):
                            symbols.add(node.name)
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    symbols.add(target.id)
                    file_symbols[file_path] = symbols
                except Exception as e:
                    self.logger.warning(f"Could not analyze symbols in {file_path}: {e}")
                    file_symbols[file_path] = set()
            else:
                file_symbols[file_path] = set()
        
        # Find symbols that appear in multiple files
        if not file_symbols:
            return set()
        
        all_symbols = set.union(*file_symbols.values()) if file_symbols.values() else set()
        shared_symbols = set()
        
        for symbol in all_symbols:
            count = sum(1 for symbols in file_symbols.values() if symbol in symbols)
            if count >= 2:  # Symbol appears in at least 2 files
                shared_symbols.add(symbol)
        
        return shared_symbols
    
    def _compute_extraction_benefit(self, group: LinkageGroup, 
                                  group_ld: np.ndarray, 
                                  shared_symbols: Set[str]) -> float:
        """Compute benefit score for extracting shared functionality."""
        # Base benefit from number of shared symbols
        symbol_benefit = min(1.0, len(shared_symbols) / 10.0)
        
        # Benefit from co-edit frequency (high frequency = high benefit)
        frequency_benefit = group.coedit_frequency
        
        # Benefit from group size (more files = more benefit)
        size_benefit = min(1.0, len(group.files) / 5.0)
        
        # Penalty for very high LD (might be too tightly coupled)
        mean_ld = np.mean(group_ld[group_ld != 1.0])  # Exclude diagonal
        ld_penalty = 1.0 if mean_ld < 0.8 else (1.0 - mean_ld) * 2.0
        
        # Combined benefit score
        benefit = (
            0.4 * symbol_benefit +
            0.3 * frequency_benefit +
            0.2 * size_benefit +
            0.1 * ld_penalty
        )
        
        return max(0.0, min(1.0, benefit))
    
    def _compute_group_insulation_score(self, group_indices: List[int], 
                                      ld_matrix: np.ndarray) -> float:
        """Compute how well insulated a group is from other modules."""
        if len(group_indices) < 2:
            return 0.0
        
        # Internal linkage (high = good insulation)
        internal_ld_values = []
        for i in range(len(group_indices)):
            for j in range(i+1, len(group_indices)):
                idx_i, idx_j = group_indices[i], group_indices[j]
                internal_ld_values.append(ld_matrix[idx_i, idx_j])
        
        internal_ld = np.mean(internal_ld_values) if internal_ld_values else 0.0
        
        # External linkage (low = good insulation)
        external_ld_values = []
        n_total = ld_matrix.shape[0]
        external_indices = [i for i in range(n_total) if i not in group_indices]
        
        for group_idx in group_indices:
            for ext_idx in external_indices:
                external_ld_values.append(abs(ld_matrix[group_idx, ext_idx]))
        
        external_ld = np.mean(external_ld_values) if external_ld_values else 0.0
        
        # Insulation score = high internal linkage + low external linkage
        insulation = (internal_ld + (1.0 - external_ld)) / 2.0
        return max(0.0, min(1.0, insulation))
    
    def _generate_helper_name(self, files: Set[str], shared_symbols: Set[str]) -> str:
        """Generate a name for the helper module."""
        # Try to find common prefix from file names
        file_names = [Path(f).stem for f in files]
        
        if len(file_names) >= 2:
            # Find common prefix
            common_prefix = ""
            min_len = min(len(name) for name in file_names)
            
            for i in range(min_len):
                chars = [name[i] for name in file_names]
                if len(set(chars)) == 1:
                    common_prefix += chars[0]
                else:
                    break
            
            if len(common_prefix) >= 3:
                return f"{common_prefix}_common"
        
        # Fallback: use shared symbol names
        if shared_symbols:
            # Find most common word in shared symbols
            words = []
            for symbol in shared_symbols:
                # Split camelCase and snake_case
                words.extend(re.findall(r'[A-Z][a-z]*|[a-z]+', symbol))
            
            if words:
                word_counts = Counter(words)
                most_common = word_counts.most_common(1)[0][0].lower()
                return f"{most_common}_helpers"
        
        # Final fallback
        return "shared_helpers"
    
    def _generate_boundary_suggestions(self, files: Set[str], 
                                     shared_symbols: Set[str]) -> List[str]:
        """Generate suggestions for module boundary improvements."""
        suggestions = []
        
        if len(shared_symbols) > 0:
            suggestions.append(f"Extract {len(shared_symbols)} shared symbols to helper module")
        
        if len(files) > 3:
            suggestions.append("Consider splitting into smaller, focused modules")
        
        suggestions.append("Add __all__ exports to control public interface")
        suggestions.append("Create facade.py for unified external interface")
        
        return suggestions


class InsulatorGenerator:
    """
    Generates module boundary insulators using chromatin insulator metaphors.
    
    Creates boundaries that prevent unwanted interactions between TADs:
    - Export control via __all__ declarations
    - Facade pattern implementation
    - Common module extraction
    - Insulation scoring
    """
    
    def __init__(self):
        """Initialize insulator generator."""
        self.logger = get_logger(__name__)
        self._insulation_cache: Dict[str, float] = {}
    
    def add_module_boundaries(self, module_path: str, 
                            context: AnalysisContext,
                            extraction_roi: Optional[ExtractionROI] = None) -> ModuleBoundary:
        """
        Add insulator boundaries to a module.
        
        Implements chromatin insulator functionality to create clear
        module boundaries and prevent cross-TAD interference.
        
        Args:
            module_path: Path to the module
            context: Analysis context
            extraction_roi: Optional extraction ROI for guided boundary creation
            
        Returns:
            ModuleBoundary specification with insulator properties
        """
        self.logger.info(f"Adding module boundaries to {module_path}")
        
        # Analyze module structure
        module_symbols = self._analyze_module_symbols(module_path, context)
        
        # Generate __all__ exports
        suggested_exports = self._generate_exports(module_symbols, extraction_roi)
        
        # Generate facade content if beneficial
        facade_content = self._generate_facade_content(module_path, module_symbols)
        
        # Determine common extractions
        common_extractions = self._identify_common_extractions(
            module_symbols, extraction_roi
        )
        
        # Compute insulation score
        insulation_score = self.compute_insulation_score(module_path, context)
        
        # Determine boundary type
        boundary_type = self._determine_boundary_type(
            module_symbols, suggested_exports, facade_content, common_extractions
        )
        
        boundary = ModuleBoundary(
            module_path=module_path,
            boundary_type=boundary_type,
            insulation_score=insulation_score,
            suggested_exports=suggested_exports,
            facade_content=facade_content,
            common_extractions=common_extractions
        )
        
        self.logger.debug(f"Module boundary created: {boundary_type} "
                         f"(insulation: {insulation_score:.2f})")
        
        return boundary
    
    def compute_insulation_score(self, module_path: str, 
                               context: AnalysisContext) -> float:
        """
        Compute insulation score for a module.
        
        Measures how well isolated the module is from other TADs,
        similar to chromatin insulator strength.
        
        Args:
            module_path: Path to the module
            context: Analysis context
            
        Returns:
            Insulation score (0.0-1.0, higher = better insulated)
        """
        if module_path in self._insulation_cache:
            return self._insulation_cache[module_path]
        
        self.logger.debug(f"Computing insulation score for {module_path}")
        
        # Analyze imports and dependencies
        imports_score = self._compute_import_insulation(module_path, context)
        
        # Analyze symbol usage patterns
        symbol_score = self._compute_symbol_insulation(module_path, context)
        
        # Analyze structural complexity
        structure_score = self._compute_structural_insulation(module_path, context)
        
        # Combined insulation score
        insulation_score = (
            0.4 * imports_score +
            0.3 * symbol_score +
            0.3 * structure_score
        )
        
        self._insulation_cache[module_path] = insulation_score
        
        self.logger.debug(f"Insulation score for {module_path}: {insulation_score:.2f}")
        return insulation_score
    
    def generate_facade_file(self, module_path: str, boundary: ModuleBoundary) -> str:
        """
        Generate facade.py file content for module interface.
        
        Args:
            module_path: Path to the module
            boundary: Module boundary specification
            
        Returns:
            Generated facade.py content
        """
        self.logger.info(f"Generating facade for {module_path}")
        
        module_name = Path(module_path).stem
        
        # Generate facade content
        facade_lines = [
            '"""',
            f'Facade interface for {module_name} module.',
            '',
            f'This facade provides a clean, stable interface to the {module_name}',
            'module while hiding internal implementation details and preventing',
            'direct access to private components.',
            '',
            'Generated by TailChasingFixer InsulatorGenerator.',
            '"""',
            '',
        ]
        
        # Add imports
        if boundary.suggested_exports:
            facade_lines.extend([
                '# Import public interfaces',
                f'from .{module_name} import (',
            ])
            
            exports_list = sorted(boundary.suggested_exports)
            for i, export in enumerate(exports_list):
                comma = ',' if i < len(exports_list) - 1 else ''
                facade_lines.append(f'    {export}{comma}')
            
            facade_lines.extend([
                ')',
                '',
            ])
        
        # Add __all__ declaration
        if boundary.suggested_exports:
            facade_lines.extend([
                '# Public interface',
                '__all__ = [',
            ])
            
            for export in sorted(boundary.suggested_exports):
                facade_lines.append(f"    '{export}',")
            
            facade_lines.extend([
                ']',
                '',
            ])
        
        # Add convenience functions if applicable
        if boundary.boundary_type == 'facade':
            facade_lines.extend([
                '# Convenience interface functions',
                'def get_version() -> str:',
                f'    """Get {module_name} version."""',
                '    return "1.0.0"  # TODO: Implement version detection',
                '',
                'def get_status() -> dict:',
                f'    """Get {module_name} status information."""',
                '    return {',
                f'        "module": "{module_name}",',
                '        "exports": len(__all__),',
                '        "status": "active"',
                '    }',
                '',
            ])
        
        return '\n'.join(facade_lines)
    
    def generate_common_file(self, extraction_roi: ExtractionROI) -> str:
        """
        Generate *_common.py file for shared constants and utilities.
        
        Args:
            extraction_roi: Extraction ROI specification
            
        Returns:
            Generated common file content
        """
        self.logger.info(f"Generating common file for {extraction_roi.proposed_helper_name}")
        
        common_lines = [
            '"""',
            f'Common utilities and constants for {extraction_roi.proposed_helper_name}.',
            '',
            'This module contains shared functionality extracted from:',
        ]
        
        for file_path in sorted(extraction_roi.target_files):
            common_lines.append(f'- {file_path}')
        
        common_lines.extend([
            '',
            'Generated by TailChasingFixer RecombinationMapper.',
            '"""',
            '',
            '# Shared constants',
        ])
        
        # Add shared symbols as constants/functions
        for symbol in sorted(extraction_roi.shared_symbols):
            if symbol.isupper():
                # Constant
                common_lines.append(f'{symbol} = None  # TODO: Define value')
            else:
                # Function or class
                if symbol[0].isupper():
                    # Class
                    common_lines.extend([
                        '',
                        f'class {symbol}:',
                        f'    """Shared {symbol} class."""',
                        '    pass  # TODO: Implement class',
                    ])
                else:
                    # Function
                    common_lines.extend([
                        '',
                        f'def {symbol}(*args, **kwargs):',
                        f'    """Shared {symbol} function."""',
                        f'    raise NotImplementedError("TODO: Implement {symbol}")',
                    ])
        
        common_lines.extend([
            '',
            '# Export all shared symbols',
            '__all__ = [',
        ])
        
        for symbol in sorted(extraction_roi.shared_symbols):
            common_lines.append(f"    '{symbol}',")
        
        common_lines.extend([
            ']',
            '',
        ])
        
        return '\n'.join(common_lines)
    
    # === Helper Methods ===
    
    def _analyze_module_symbols(self, module_path: str, 
                              context: AnalysisContext) -> Dict[str, List[str]]:
        """Analyze symbols in a module."""
        symbols = {
            'functions': [],
            'classes': [],
            'constants': [],
            'imports': []
        }
        
        if module_path not in context.ast_index:
            return symbols
        
        try:
            tree = context.ast_index[module_path]
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    symbols['classes'].append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbols['constants'].append(target.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            symbols['imports'].append(alias.name)
                    else:
                        symbols['imports'].append(node.module or 'unknown')
        
        except Exception as e:
            self.logger.warning(f"Could not analyze symbols in {module_path}: {e}")
        
        return symbols
    
    def _generate_exports(self, module_symbols: Dict[str, List[str]], 
                        extraction_roi: Optional[ExtractionROI]) -> Set[str]:
        """Generate suggested __all__ exports."""
        exports = set()
        
        # Add public functions and classes (not starting with _)
        for func in module_symbols['functions']:
            if not func.startswith('_'):
                exports.add(func)
        
        for cls in module_symbols['classes']:
            if not cls.startswith('_'):
                exports.add(cls)
        
        # Add important constants
        for const in module_symbols['constants']:
            if const.isupper() and not const.startswith('_'):
                exports.add(const)
        
        # Add symbols from extraction ROI if provided
        if extraction_roi:
            exports.update(extraction_roi.shared_symbols)
        
        return exports
    
    def _generate_facade_content(self, module_path: str, 
                               module_symbols: Dict[str, List[str]]) -> Optional[str]:
        """Determine if facade is beneficial and generate content."""
        # Generate facade if module has many exports or complex structure
        total_symbols = (
            len(module_symbols['functions']) +
            len(module_symbols['classes']) +
            len(module_symbols['constants'])
        )
        
        if total_symbols > 10:  # Complex module benefits from facade
            return "facade"  # Marker for facade generation
        
        return None
    
    def _identify_common_extractions(self, module_symbols: Dict[str, List[str]],
                                   extraction_roi: Optional[ExtractionROI]) -> Set[str]:
        """Identify symbols that should be extracted to common module."""
        extractions = set()
        
        # Constants are good candidates for extraction
        for const in module_symbols['constants']:
            if const.isupper():
                extractions.add(const)
        
        # Add symbols from extraction ROI
        if extraction_roi:
            extractions.update(extraction_roi.shared_symbols)
        
        return extractions
    
    def _determine_boundary_type(self, module_symbols: Dict[str, List[str]],
                               suggested_exports: Set[str],
                               facade_content: Optional[str],
                               common_extractions: Set[str]) -> str:
        """Determine the type of boundary to create."""
        if facade_content:
            return "facade"
        elif common_extractions:
            return "common"
        elif suggested_exports:
            return "export"
        else:
            return "simple"
    
    def _compute_import_insulation(self, module_path: str, 
                                 context: AnalysisContext) -> float:
        """Compute insulation score based on import patterns."""
        if module_path not in context.ast_index:
            return 0.5  # Neutral score
        
        try:
            tree = context.ast_index[module_path]
            
            total_imports = 0
            external_imports = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    total_imports += 1
                    
                    # Check if import is external (not relative)
                    if isinstance(node, ast.ImportFrom):
                        if not node.module or not node.module.startswith('.'):
                            external_imports += 1
                    else:
                        external_imports += 1
            
            if total_imports == 0:
                return 1.0  # Perfect insulation (no imports)
            
            # Good insulation = fewer external imports
            insulation = 1.0 - (external_imports / total_imports)
            return max(0.0, min(1.0, insulation))
        
        except Exception:
            return 0.5
    
    def _compute_symbol_insulation(self, module_path: str, 
                                 context: AnalysisContext) -> float:
        """Compute insulation score based on symbol usage patterns."""
        # For now, return neutral score
        # TODO: Implement symbol usage analysis
        return 0.5
    
    def _compute_structural_insulation(self, module_path: str, 
                                     context: AnalysisContext) -> float:
        """Compute insulation score based on structural complexity."""
        if module_path not in context.ast_index:
            return 0.5
        
        try:
            tree = context.ast_index[module_path]
            
            # Count different types of nodes
            node_counts = defaultdict(int)
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1
            
            # Simple structure = better insulation
            total_nodes = sum(node_counts.values())
            if total_nodes == 0:
                return 1.0
            
            # Penalize complex structures
            complexity_penalty = min(1.0, total_nodes / 1000.0)
            insulation = 1.0 - complexity_penalty
            
            return max(0.0, min(1.0, insulation))
        
        except Exception:
            return 0.5


# Integration point with ChromatinContactAnalyzer
def enhance_chromatin_analyzer(chromatin_analyzer, recombination_mapper: RecombinationMapper):
    """
    Enhance ChromatinContactAnalyzer with recombination mapping data.
    
    Integrates co-edit analysis to improve distance calculations and
    generate insulator suggestions.
    
    Args:
        chromatin_analyzer: ChromatinContactAnalyzer instance
        recombination_mapper: RecombinationMapper instance
    """
    logger = get_logger(__name__)
    logger.info("Enhancing ChromatinContactAnalyzer with recombination mapping")
    
    # Store recombination mapper reference
    chromatin_analyzer._recombination_mapper = recombination_mapper
    
    # Enhanced distance calculation using co-edit data
    original_polymer_distance = chromatin_analyzer.polymer_distance
    
    def enhanced_polymer_distance(elem1, elem2):
        """Enhanced polymer distance using co-edit data."""
        base_distance = original_polymer_distance(elem1, elem2)
        
        # Apply co-edit correction if data available
        if hasattr(recombination_mapper, '_coedit_cache'):
            # TODO: Implement co-edit distance correction
            pass
        
        return base_distance
    
    chromatin_analyzer.polymer_distance = enhanced_polymer_distance
    
    logger.info("ChromatinContactAnalyzer enhancement complete")