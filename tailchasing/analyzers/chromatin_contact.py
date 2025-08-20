"""Chromatin Contact Analyzer.

This analyzer uses concepts from chromatin biology and polymer physics to detect
tightly coupled code clusters and organizational anti-patterns. The biological
metaphor is based on how DNA is organized in the cell nucleus:

- **Chromatin**: DNA wrapped around proteins, forming a flexible polymer chain
- **Contact Probability**: How likely two DNA regions are to be spatially close
- **Topological Domains**: Regions of DNA that preferentially contact each other
- **Loop Extrusion**: Process that brings distant DNA regions into contact

In code analysis, this translates to:
- **Code Chromatin**: Functions/classes as segments of a polymer chain
- **Contact Distance**: Coupling strength between code elements
- **TADs (Topologically Associating Domains)**: Modules that should be cohesive
- **Contact Violations**: Unexpected tight coupling across module boundaries

The analyzer identifies anti-patterns like:
1. Excessive cross-module coupling (violating expected polymer distance decay)
2. Missing intra-module cohesion (weak contacts within expected domains)  
3. Architectural boundary violations (contacts that cross design boundaries)
4. Polymer chain tangles (circular dependencies and complex coupling graphs)
"""

from __future__ import annotations

import ast
import math
import logging
from typing import Dict, List, Any, Optional, Iterable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum

from ..core.types import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue


logger = logging.getLogger(__name__)


class ContactType(Enum):
    """Types of contacts between code elements."""
    DIRECT_CALL = "direct_call"
    INHERITANCE = "inheritance"
    IMPORT = "import"
    SHARED_DATA = "shared_data"
    EXCEPTION_FLOW = "exception_flow"


class CouplingStrength(Enum):
    """Strength levels for code coupling."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class CodeSegment:
    """Represents a segment of code in the chromatin polymer model.
    
    Attributes:
        name: Name of the code segment (function, class, module)
        file_path: Path to the file containing this segment
        start_line: Starting line number
        end_line: Ending line number
        segment_type: Type of segment (function, class, module)
        polymer_position: Position along the conceptual polymer chain
        tad_domain: Topological domain this segment belongs to
    """
    name: str
    file_path: str
    start_line: int
    end_line: int
    segment_type: str
    polymer_position: float = 0.0
    tad_domain: Optional[str] = None


@dataclass
class ChromatinContact:
    """Represents a contact between two code segments.
    
    Attributes:
        segment1: First code segment
        segment2: Second code segment
        contact_type: Type of contact (call, import, inheritance, etc.)
        coupling_strength: Strength of the coupling
        polymer_distance: Distance along the polymer chain
        spatial_distance: Actual coupling distance (weighted by type)
        expected_probability: Expected contact probability from polymer model
        observed_frequency: Observed contact frequency
        violation_score: How much this contact violates expected patterns
    """
    segment1: CodeSegment
    segment2: CodeSegment
    contact_type: ContactType
    coupling_strength: CouplingStrength
    polymer_distance: float
    spatial_distance: float
    expected_probability: float = 0.0
    observed_frequency: float = 0.0
    violation_score: float = 0.0


@dataclass
class TADomain:
    """Topologically Associating Domain - a cohesive code region.
    
    Attributes:
        name: Name of the TAD (usually module or package name)
        segments: Code segments within this domain
        internal_contacts: Contacts within the domain
        external_contacts: Contacts crossing domain boundaries
        cohesion_score: How cohesive the domain is internally
        insulation_score: How well insulated from other domains
        boundary_violations: Contacts that shouldn't cross boundaries
    """
    name: str
    segments: List[CodeSegment] = field(default_factory=list)
    internal_contacts: List[ChromatinContact] = field(default_factory=list)
    external_contacts: List[ChromatinContact] = field(default_factory=list)
    cohesion_score: float = 0.0
    insulation_score: float = 0.0
    boundary_violations: List[ChromatinContact] = field(default_factory=list)


class ChromatinContactAnalyzer(BaseAnalyzer):
    """Analyzer for detecting tightly coupled code clusters using chromatin biology concepts.
    
    This analyzer models code organization as a polymer chain (like chromatin in cell nuclei)
    and identifies organizational anti-patterns by detecting violations of expected contact
    probabilities and domain boundaries.
    
    The analysis process:
    
    1. **Polymer Modeling**: Map code elements to positions on a conceptual polymer chain
       based on their logical organization (module hierarchy, call graphs, etc.)
    
    2. **Contact Detection**: Identify all contacts (dependencies) between code segments
       and classify their strength and type
    
    3. **Distance Calculation**: Compute both polymer distance (logical separation) and
       spatial distance (actual coupling strength)
    
    4. **Probability Modeling**: Calculate expected contact probabilities using polymer
       physics models (power-law decay with distance)
    
    5. **Violation Detection**: Find contacts that violate expected patterns:
       - Unexpectedly strong long-range contacts (tight coupling across modules)
       - Unexpectedly weak short-range contacts (low cohesion within modules)
       - Boundary violations (contacts that cross architectural boundaries)
    
    Biological Inspiration:
    - In cells, chromatin forms loops and domains that regulate gene expression
    - Contact probability decays as a power law with polymer distance  
    - TADs (Topologically Associating Domains) are regions of preferential contact
    - Loop extrusion processes bring distant regions into contact
    - Boundary elements insulate domains from each other
    
    Code Analysis Application:
    - Functions/classes are segments of the polymer chain
    - Dependencies are contacts with different strengths
    - Modules are TADs that should have internal cohesion
    - Architectural boundaries should insulate domains
    - Design patterns create expected contact structures
    """
    
    name = "chromatin_contact"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ChromatinContactAnalyzer.
        
        Args:
            config: Configuration options for the analyzer
                - contact_decay_exponent: Power law exponent for distance decay (default: 1.2)
                - violation_threshold: Threshold for flagging violations (default: 2.0)
                - tad_boundary_patterns: Patterns that define domain boundaries
                - polymer_weights: Weights for different distance types
                - coupling_thresholds: Thresholds for coupling strength classification
        """
        super().__init__(self.name)
        self.config = config or {}
        
        # Polymer physics parameters
        self.contact_decay_exponent = self.config.get('contact_decay_exponent', 1.2)
        self.violation_threshold = self.config.get('violation_threshold', 2.0)
        
        # Domain boundary patterns
        self.tad_boundary_patterns = self.config.get('tad_boundary_patterns', [
            r'.*\.api\..*',
            r'.*\.core\..*', 
            r'.*\.models\..*',
            r'.*\.utils\..*',
            r'.*\.tests\..*'
        ])
        
        # Distance calculation weights
        self.polymer_weights = self.config.get('polymer_weights', {
            'file': 1.0,      # File-level distance
            'module': 2.0,    # Module-level distance  
            'package': 3.0,   # Package-level distance
            'git': 0.5        # Git history distance (optional)
        })
        
        # Coupling strength thresholds
        self.coupling_thresholds = self.config.get('coupling_thresholds', {
            'weak': 0.2,
            'moderate': 0.5,
            'strong': 0.8,
            'very_strong': 1.0
        })
        
        # Internal state
        self.code_segments: List[CodeSegment] = []
        self.contacts: List[ChromatinContact] = []
        self.tad_domains: Dict[str, TADomain] = {}
        
        logger.debug(f"Initialized ChromatinContactAnalyzer with decay_exponent={self.contact_decay_exponent}")
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Run chromatin contact analysis on the provided context.
        
        Args:
            ctx: Analysis context containing files and configuration
            
        Yields:
            Issue: Issues found during chromatin contact analysis
        """
        # TODO: Implement full chromatin contact analysis
        logger.info(f"Starting chromatin contact analysis on {len(ctx.files)} files")
        
        try:
            # Step 1: Build polymer model
            self._build_polymer_model(ctx)
            
            # Step 2: Detect contacts
            self._detect_contacts(ctx)
            
            # Step 3: Calculate expected probabilities
            self._calculate_contact_probabilities()
            
            # Step 4: Identify TAD domains
            self._identify_tad_domains()
            
            # Step 5: Detect violations
            violations = self._detect_contact_violations()
            
            # Step 6: Generate issues
            for violation in violations:
                issue = self._create_violation_issue(violation)
                yield issue
                
        except Exception as e:
            logger.error(f"Error during chromatin contact analysis: {e}")
        
        logger.info("Chromatin contact analysis completed")
    
    def _build_polymer_model(self, ctx: AnalysisContext) -> None:
        """Build the polymer model of the codebase.
        
        Maps code elements to positions on a conceptual polymer chain
        based on their logical organization.
        
        Args:
            ctx: Analysis context
        """
        # TODO: Implement polymer model construction
        logger.debug("Building polymer model")
        
        self.code_segments = []
        position = 0.0
        
        for file_path in sorted(ctx.files):
            try:
                segments = self._extract_segments_from_file(file_path, position)
                self.code_segments.extend(segments)
                position += len(segments) * 10.0  # Space segments along polymer
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
        
        logger.debug(f"Built polymer model with {len(self.code_segments)} segments")
    
    def _extract_segments_from_file(self, file_path: Path, start_position: float) -> List[CodeSegment]:
        """Extract code segments from a single file.
        
        Args:
            file_path: Path to the file
            start_position: Starting position on the polymer chain
            
        Returns:
            List of code segments from the file
        """
        # TODO: Implement comprehensive segment extraction
        segments = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            position = start_position
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    segment = CodeSegment(
                        name=node.name,
                        file_path=str(file_path),
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        segment_type='function',
                        polymer_position=position
                    )
                    segments.append(segment)
                    position += 1.0
                
                elif isinstance(node, ast.ClassDef):
                    segment = CodeSegment(
                        name=node.name,
                        file_path=str(file_path),
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        segment_type='class',
                        polymer_position=position
                    )
                    segments.append(segment)
                    position += 2.0  # Classes take more "space" on polymer
                    
        except Exception as e:
            logger.error(f"Failed to extract segments from {file_path}: {e}")
        
        return segments
    
    def _detect_contacts(self, ctx: AnalysisContext) -> None:
        """Detect all contacts between code segments.
        
        Args:
            ctx: Analysis context
        """
        # TODO: Implement comprehensive contact detection
        logger.debug("Detecting contacts between segments")
        
        self.contacts = []
        
        # Build a lookup map for faster segment finding
        segment_map = {(seg.file_path, seg.name): seg for seg in self.code_segments}
        
        for file_path in ctx.files:
            try:
                file_contacts = self._detect_file_contacts(file_path, segment_map)
                self.contacts.extend(file_contacts)
            except Exception as e:
                logger.warning(f"Failed to detect contacts in {file_path}: {e}")
        
        logger.debug(f"Detected {len(self.contacts)} contacts")
    
    def _detect_file_contacts(self, file_path: Path, segment_map: Dict[Tuple[str, str], CodeSegment]) -> List[ChromatinContact]:
        """Detect contacts within a single file.
        
        Args:
            file_path: Path to the file
            segment_map: Map from (file, name) to CodeSegment
            
        Returns:
            List of detected contacts
        """
        # TODO: Implement sophisticated contact detection
        contacts = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Detect function calls
                    caller_segment = self._find_containing_segment(node.lineno, str(file_path))
                    callee_name = self._extract_call_name(node)
                    
                    if caller_segment and callee_name:
                        callee_segment = segment_map.get((str(file_path), callee_name))
                        if callee_segment:
                            contact = self._create_contact(
                                caller_segment, callee_segment, ContactType.DIRECT_CALL
                            )
                            contacts.append(contact)
                            
        except Exception as e:
            logger.error(f"Failed to detect contacts in {file_path}: {e}")
        
        return contacts
    
    def _find_containing_segment(self, line_num: int, file_path: str) -> Optional[CodeSegment]:
        """Find the code segment containing a given line number.
        
        Args:
            line_num: Line number
            file_path: File path
            
        Returns:
            CodeSegment containing the line, or None
        """
        # TODO: Implement efficient segment lookup
        for segment in self.code_segments:
            if (segment.file_path == file_path and
                segment.start_line <= line_num <= segment.end_line):
                return segment
        return None
    
    def _extract_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract function name from a call node.
        
        Args:
            node: AST call node
            
        Returns:
            Function name if extractable, None otherwise
        """
        # TODO: Implement more sophisticated name extraction
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
    
    def _create_contact(self, seg1: CodeSegment, seg2: CodeSegment, contact_type: ContactType) -> ChromatinContact:
        """Create a ChromatinContact between two segments.
        
        Args:
            seg1: First code segment
            seg2: Second code segment
            contact_type: Type of contact
            
        Returns:
            ChromatinContact object
        """
        # Calculate distances
        polymer_distance = abs(seg1.polymer_position - seg2.polymer_position)
        spatial_distance = self._calculate_spatial_distance(seg1, seg2, contact_type)
        
        # Determine coupling strength
        coupling_strength = self._classify_coupling_strength(spatial_distance)
        
        return ChromatinContact(
            segment1=seg1,
            segment2=seg2,
            contact_type=contact_type,
            coupling_strength=coupling_strength,
            polymer_distance=polymer_distance,
            spatial_distance=spatial_distance
        )
    
    def _calculate_spatial_distance(self, seg1: CodeSegment, seg2: CodeSegment, contact_type: ContactType) -> float:
        """Calculate spatial distance between two segments.
        
        Args:
            seg1: First segment
            seg2: Second segment
            contact_type: Type of contact
            
        Returns:
            Spatial distance (weighted by contact type and location)
        """
        # TODO: Implement comprehensive distance calculation
        
        # Base distance from file/module separation
        distance = 0.0
        
        # File-level distance
        if seg1.file_path != seg2.file_path:
            distance += self.polymer_weights['file']
        
        # Module-level distance
        mod1 = Path(seg1.file_path).stem
        mod2 = Path(seg2.file_path).stem
        if mod1 != mod2:
            distance += self.polymer_weights['module']
        
        # Package-level distance  
        pkg1 = Path(seg1.file_path).parent.name
        pkg2 = Path(seg2.file_path).parent.name
        if pkg1 != pkg2:
            distance += self.polymer_weights['package']
        
        # Weight by contact type
        type_weights = {
            ContactType.DIRECT_CALL: 1.0,
            ContactType.INHERITANCE: 0.8,
            ContactType.IMPORT: 0.6,
            ContactType.SHARED_DATA: 1.2,
            ContactType.EXCEPTION_FLOW: 0.4
        }
        distance *= type_weights.get(contact_type, 1.0)
        
        return distance
    
    def _classify_coupling_strength(self, spatial_distance: float) -> CouplingStrength:
        """Classify coupling strength based on spatial distance.
        
        Args:
            spatial_distance: Calculated spatial distance
            
        Returns:
            Coupling strength classification
        """
        # Invert distance to get strength (closer = stronger)
        strength = 1.0 / (1.0 + spatial_distance)
        
        if strength >= self.coupling_thresholds['very_strong']:
            return CouplingStrength.VERY_STRONG
        elif strength >= self.coupling_thresholds['strong']:
            return CouplingStrength.STRONG
        elif strength >= self.coupling_thresholds['moderate']:
            return CouplingStrength.MODERATE
        else:
            return CouplingStrength.WEAK
    
    def _calculate_contact_probabilities(self) -> None:
        """Calculate expected contact probabilities using polymer physics model.
        
        Uses power-law decay: P(contact) ~ distance^(-alpha)
        where alpha is the contact_decay_exponent.
        """
        # TODO: Implement polymer physics probability calculation
        logger.debug("Calculating contact probabilities")
        
        for contact in self.contacts:
            # Power-law decay model
            if contact.polymer_distance > 0:
                expected_prob = 1.0 / (contact.polymer_distance ** self.contact_decay_exponent)
            else:
                expected_prob = 1.0  # Same position = maximum probability
            
            contact.expected_probability = expected_prob
            
            # For stub implementation, assume observed frequency equals expected
            # TODO: Calculate actual observed frequency from contact patterns
            contact.observed_frequency = expected_prob
        
        logger.debug("Contact probabilities calculated")
    
    def _identify_tad_domains(self) -> None:
        """Identify Topologically Associating Domains (cohesive code regions).
        """
        # TODO: Implement TAD domain identification
        logger.debug("Identifying TAD domains")
        
        self.tad_domains = {}
        
        # Group segments by module for basic domain identification
        module_groups = defaultdict(list)
        for segment in self.code_segments:
            module = Path(segment.file_path).stem
            module_groups[module].append(segment)
        
        for module_name, segments in module_groups.items():
            domain = TADomain(name=module_name, segments=segments)
            
            # Classify contacts as internal or external
            for contact in self.contacts:
                seg1_module = Path(contact.segment1.file_path).stem
                seg2_module = Path(contact.segment2.file_path).stem
                
                if seg1_module == module_name or seg2_module == module_name:
                    if seg1_module == seg2_module:
                        domain.internal_contacts.append(contact)
                    else:
                        domain.external_contacts.append(contact)
            
            # Calculate basic cohesion score
            total_internal = len(domain.internal_contacts)
            total_external = len(domain.external_contacts)
            if total_internal + total_external > 0:
                domain.cohesion_score = total_internal / (total_internal + total_external)
            
            self.tad_domains[module_name] = domain
        
        logger.debug(f"Identified {len(self.tad_domains)} TAD domains")
    
    def _detect_contact_violations(self) -> List[ChromatinContact]:
        """Detect contacts that violate expected chromatin organization patterns.
        
        Returns:
            List of violating contacts
        """
        # TODO: Implement comprehensive violation detection
        logger.debug("Detecting contact violations")
        
        violations = []
        
        for contact in self.contacts:
            # Calculate violation score
            if contact.expected_probability > 0:
                # Ratio of observed to expected frequency
                obs_exp_ratio = contact.observed_frequency / contact.expected_probability
                
                # Violations are significant deviations from expected
                if abs(math.log(obs_exp_ratio)) > self.violation_threshold:
                    contact.violation_score = abs(math.log(obs_exp_ratio))
                    violations.append(contact)
        
        logger.debug(f"Detected {len(violations)} contact violations")
        return violations
    
    def _create_violation_issue(self, contact: ChromatinContact) -> Issue:
        """Create an Issue from a contact violation.
        
        Args:
            contact: ChromatinContact representing a violation
            
        Returns:
            Issue object for the violation
        """
        # TODO: Implement comprehensive issue creation with detailed messages
        
        violation_type = "unexpected_strong_contact" if contact.observed_frequency > contact.expected_probability else "unexpected_weak_contact"
        
        message = (f"Chromatin contact violation: {violation_type} between "
                  f"{contact.segment1.name} and {contact.segment2.name} "
                  f"(polymer distance: {contact.polymer_distance:.1f}, "
                  f"violation score: {contact.violation_score:.2f})")
        
        return Issue(
            kind="chromatin_contact_violation",
            message=message,
            file=contact.segment1.file_path,
            line=contact.segment1.start_line,
            confidence=min(contact.violation_score / 5.0, 1.0),  # Scale to 0-1
            severity="medium",
            metadata={
                'violation_type': violation_type,
                'contact_type': contact.contact_type.value,
                'coupling_strength': contact.coupling_strength.value,
                'polymer_distance': contact.polymer_distance,
                'spatial_distance': contact.spatial_distance,
                'violation_score': contact.violation_score,
                'expected_probability': contact.expected_probability,
                'observed_frequency': contact.observed_frequency,
                'analyzer': self.name
            }
        )
    
    def get_chromatin_statistics(self) -> Dict[str, Any]:
        """Get statistics about the chromatin contact analysis.
        
        Returns:
            Dictionary containing analysis statistics
        """
        # TODO: Implement comprehensive statistics
        return {
            'total_segments': len(self.code_segments),
            'total_contacts': len(self.contacts),
            'tad_domains': len(self.tad_domains),
            'contact_types': dict(Counter(c.contact_type.value for c in self.contacts)),
            'coupling_strengths': dict(Counter(c.coupling_strength.value for c in self.contacts)),
            'average_polymer_distance': sum(c.polymer_distance for c in self.contacts) / max(len(self.contacts), 1),
            'average_spatial_distance': sum(c.spatial_distance for c in self.contacts) / max(len(self.contacts), 1)
        }


# TODO: Implement additional chromatin analysis features:
# - Loop extrusion simulation for detecting design patterns
# - Insulator detection for architectural boundaries  
# - Contact enrichment analysis for hotspots
# - Temporal analysis of contact evolution
# - Integration with git history for chromatin dynamics
# - Machine learning for contact prediction
# - Visualization of chromatin contact maps
# - Export to polymer simulation formats

__all__ = [
    'ChromatinContactAnalyzer', 
    'CodeSegment', 
    'ChromatinContact', 
    'TADomain',
    'ContactType',
    'CouplingStrength'
]