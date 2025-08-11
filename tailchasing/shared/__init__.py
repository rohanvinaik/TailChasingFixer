"""Shared utilities and common functions for tail-chasing analysis."""

from .common_functions import (
    is_excluded,
    get_source_lines,
    should_ignore_issue,
    is_placeholder_allowed,
    get_analyzer_cache,
    get_file_metadata,
    get_confidence,
    filter_by_severity,
    group_by_file,
)

__all__ = [
    'is_excluded',
    'get_source_lines',
    'should_ignore_issue',
    'is_placeholder_allowed',
    'get_analyzer_cache',
    'get_file_metadata',
    'get_confidence',
    'filter_by_severity',
    'group_by_file',
]