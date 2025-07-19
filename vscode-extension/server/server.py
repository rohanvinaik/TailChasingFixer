#!/usr/bin/env python3
"""
Language Server Protocol implementation for Tail-Chasing Detector.

This server provides real-time analysis for the VS Code extension.
"""

import sys
import asyncio
import json
from typing import List, Dict, Optional, Any
from pathlib import Path

# Add parent directory to path to import tailchasing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pygls.server import LanguageServer
from pygls.lsp.methods import (
    COMPLETION,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    HOVER,
    CODE_ACTION,
    CODE_LENS,
    WORKSPACE_EXECUTE_COMMAND,
    INITIALIZE
)
from pygls.lsp.types import (
    CompletionItem,
    CompletionList,
    CompletionParams,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
    HoverParams,
    Hover,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
    Diagnostic,
    DiagnosticSeverity,
    CodeAction,
    CodeActionParams,
    CodeLens,
    CodeLensParams,
    Command,
    ExecuteCommandParams,
    InitializeParams
)

from tailchasing.core.loader import parse_files
from tailchasing.core.symbols import SymbolTable
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.analyzers.semantic_hv import SemanticHVAnalyzer
from tailchasing.analyzers.duplicates import DuplicateFunctionAnalyzer
from tailchasing.analyzers.placeholders import PlaceholderAnalyzer


class TailChasingLanguageServer(LanguageServer):
    """Language server for tail-chasing detection."""
    
    def __init__(self):
        super().__init__('tail-chasing-ls', 'v0.1.0')
        self.analysis_cache: Dict[str, Any] = {}
        self.semantic_analyzer = SemanticHVAnalyzer()
        self.duplicate_analyzer = DuplicateFunctionAnalyzer()
        self.placeholder_analyzer = PlaceholderAnalyzer()


server = TailChasingLanguageServer()


@server.feature(INITIALIZE)
async def initialize(params: InitializeParams):
    """Initialize the language server."""
    server.show_message("Tail-Chasing Detector Language Server initialized!")


@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(params: DidOpenTextDocumentParams):
    """Handle file open event."""
    uri = params.text_document.uri
    await analyze_document(uri)


@server.feature(TEXT_DOCUMENT_DID_CHANGE)
async def did_change(params: DidChangeTextDocumentParams):
    """Handle text change event."""
    uri = params.text_document.uri
    # Debounce analysis for performance
    await asyncio.sleep(0.5)
    await analyze_document(uri)


@server.feature(TEXT_DOCUMENT_DID_SAVE)
async def did_save(params: DidSaveTextDocumentParams):
    """Handle file save event."""
    uri = params.text_document.uri
    await analyze_document(uri)


@server.feature(HOVER)
async def hover(params: HoverParams) -> Optional[Hover]:
    """Provide hover information."""
    uri = params.text_document.uri
    position = params.position
    
    # Get cached analysis
    analysis = server.analysis_cache.get(uri, {})
    semantic_info = analysis.get('semantic', {})
    
    # Find function at position
    function_name = get_function_at_position(uri, position)
    if not function_name:
        return None
    
    # Get semantic duplicates for this function
    duplicates = semantic_info.get(function_name, {}).get('duplicates', [])
    
    if duplicates:
        content = f"### Semantic Analysis for `{function_name}`\\n\\n"
        content += "**Semantic Duplicates:**\\n\\n"
        
        for dup in duplicates[:3]:  # Show top 3
            content += f"- `{dup['name']}` ({dup['similarity']:.0%} similar) "
            content += f"in {dup['file']}:{dup['line']}\\n"
        
        content += "\\n**Suggestion:** Consider consolidating these functions"
        
        return Hover(
            contents=MarkupContent(
                kind=MarkupKind.Markdown,
                value=content
            ),
            range=Range(
                start=Position(line=position.line, character=0),
                end=Position(line=position.line, character=100)
            )
        )
    
    return None


@server.feature(CODE_LENS)
async def code_lens(params: CodeLensParams) -> List[CodeLens]:
    """Provide code lens information."""
    uri = params.text_document.uri
    lenses = []
    
    # Get cached analysis
    analysis = server.analysis_cache.get(uri, {})
    functions = analysis.get('functions', {})
    
    for func_name, info in functions.items():
        if info.get('duplicate_count', 0) > 0:
            lens = CodeLens(
                range=Range(
                    start=Position(line=info['line'] - 1, character=0),
                    end=Position(line=info['line'] - 1, character=0)
                ),
                command=Command(
                    title=f"⚠️ {info['duplicate_count']} semantic duplicates",
                    command="tailChasingDetector.showSemanticDuplicates",
                    arguments=[func_name]
                )
            )
            lenses.append(lens)
    
    return lenses


@server.feature(CODE_ACTION)
async def code_action(params: CodeActionParams) -> List[CodeAction]:
    """Provide code actions for diagnostics."""
    actions = []
    
    for diagnostic in params.context.diagnostics:
        if diagnostic.source != 'tail-chasing':
            continue
        
        if diagnostic.code == 'semantic_duplicate_function':
            action = CodeAction(
                title="Merge with semantic duplicate",
                kind="quickfix",
                diagnostics=[diagnostic],
                command=Command(
                    title="Merge Duplicates",
                    command="tailChasingDetector.mergeDuplicates",
                    arguments=[params.text_document.uri, diagnostic.range]
                )
            )
            actions.append(action)
    
    return actions


@server.feature(WORKSPACE_EXECUTE_COMMAND)
async def execute_command(params: ExecuteCommandParams):
    """Execute workspace commands."""
    if params.command == "tailChasingDetector.analyzeWorkspace":
        await analyze_workspace()
    elif params.command == "tailChasingDetector.showSemanticDuplicates":
        # Would show semantic duplicates UI
        pass


async def analyze_document(uri: str):
    """Analyze a single document."""
    try:
        # Get document from workspace
        document = server.workspace.get_document(uri)
        text = document.source
        
        # Parse the document
        path = Path(uri.replace('file://', ''))
        tree = parse_files([path])
        
        if not tree:
            return
        
        # Build symbol table
        symbol_table = SymbolTable()
        for file, ast_tree in tree.items():
            symbol_table.ingest(file, ast_tree, text)
        
        # Create analysis context
        config = {
            'semantic': {
                'enable': True,
                'min_functions': 1,  # Analyze even single files
                'z_threshold': 2.0
            }
        }
        
        ctx = AnalysisContext(
            config=config,
            files=[path],
            ast_index=tree,
            symbol_table=symbol_table,
            cache={}
        )
        
        # Run analyzers
        issues = []
        issues.extend(server.semantic_analyzer.run(ctx))
        issues.extend(server.duplicate_analyzer.run(ctx))
        issues.extend(server.placeholder_analyzer.run(ctx))
        
        # Convert to diagnostics
        diagnostics = []
        for issue in issues:
            severity = DiagnosticSeverity.Warning
            if issue.severity >= 4:
                severity = DiagnosticSeverity.Error
            elif issue.severity <= 1:
                severity = DiagnosticSeverity.Information
            
            diagnostic = Diagnostic(
                range=Range(
                    start=Position(line=issue.line - 1 if issue.line else 0, character=0),
                    end=Position(line=issue.line - 1 if issue.line else 0, character=100)
                ),
                message=issue.message,
                severity=severity,
                code=issue.kind,
                source='tail-chasing'
            )
            
            diagnostics.append(diagnostic)
        
        # Publish diagnostics
        server.publish_diagnostics(uri, diagnostics)
        
        # Cache analysis results
        server.analysis_cache[uri] = {
            'issues': issues,
            'functions': extract_function_info(symbol_table, issues),
            'semantic': extract_semantic_info(issues)
        }
        
    except Exception as e:
        server.show_message_log(f"Error analyzing {uri}: {e}")


async def analyze_workspace():
    """Analyze entire workspace."""
    # Get all Python files in workspace
    for folder in server.workspace.folders:
        folder_path = Path(folder.uri.replace('file://', ''))
        python_files = list(folder_path.rglob('*.py'))
        
        for file in python_files:
            uri = f"file://{file}"
            await analyze_document(uri)


def get_function_at_position(uri: str, position: Position) -> Optional[str]:
    """Get function name at given position."""
    # This would parse the document and find the function
    # For now, return mock data
    return "calculate_average"


def extract_function_info(symbol_table: SymbolTable, issues: List) -> Dict:
    """Extract function information for code lens."""
    info = {}
    
    for func_name, entries in symbol_table.functions.items():
        for entry in entries:
            # Count semantic duplicates
            dup_count = sum(
                1 for issue in issues
                if issue.kind == 'semantic_duplicate_function' and
                issue.symbol == func_name
            )
            
            info[func_name] = {
                'line': entry['lineno'],
                'duplicate_count': dup_count
            }
    
    return info


def extract_semantic_info(issues: List) -> Dict:
    """Extract semantic analysis information."""
    semantic_info = {}
    
    for issue in issues:
        if issue.kind == 'semantic_duplicate_function':
            pair = issue.evidence.get('pair', [])
            if len(pair) == 2:
                func1, func2 = pair
                
                # Store bidirectional relationships
                for f1, f2 in [(func1, func2), (func2, func1)]:
                    if f1['name'] not in semantic_info:
                        semantic_info[f1['name']] = {'duplicates': []}
                    
                    semantic_info[f1['name']]['duplicates'].append({
                        'name': f2['name'],
                        'file': f2['file'],
                        'line': f2['line'],
                        'similarity': 1.0 - issue.evidence.get('distance', 0.5)
                    })
    
    return semantic_info


if __name__ == '__main__':
    server.start_io()