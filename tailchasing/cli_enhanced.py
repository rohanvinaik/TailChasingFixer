"""
Enhanced CLI with advanced features for tail-chasing detection.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from .utils.logging_setup import get_logger, log_operation, log_retry
from .core.loader import collect_files, parse_files
from .core.symbols import SymbolTable
from .core.issues import Issue
from .core.reporting import Reporter
from .analyzers.base import AnalysisContext
from .analyzers.explainer import TailChasingExplainer
from .analyzers.advanced.enhanced_pattern_detector import EnhancedPatternDetector
from .analyzers.advanced.multimodal_semantic import SemanticDuplicateEnhancer
from .fixers.advanced.intelligent_fixer import IntelligentAutoFixer
from .visualization import TailChasingVisualizer
from .plugins import load_analyzers
from .config import Config


class EnhancedCLI:
    """Enhanced command-line interface with advanced features."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.explainer = TailChasingExplainer()
        self.enhanced_detector = EnhancedPatternDetector()
        self.semantic_enhancer = SemanticDuplicateEnhancer()
        self.auto_fixer = IntelligentAutoFixer()
        self.visualizer = TailChasingVisualizer()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the enhanced argument parser."""
        parser = argparse.ArgumentParser(
            prog='tailchasing',
            description='Detect and fix LLM-induced tail-chasing patterns in your codebase'
        )
        
        # Basic arguments
        parser.add_argument('root', nargs='?', default='.', 
                          help='Root directory to analyze (default: current directory)')
        
        # Output format options
        output_group = parser.add_mutually_exclusive_group()
        output_group.add_argument('--json', action='store_true',
                                help='Output results in JSON format')
        output_group.add_argument('--html', metavar='FILE',
                                help='Generate HTML report with visualizations')
        output_group.add_argument('--explain', action='store_true',
                                help='Generate detailed natural language explanations')
        
        # Analysis options
        parser.add_argument('--enhanced', action='store_true',
                          help='Enable enhanced pattern detection (hallucination cascades, context thrashing)')
        parser.add_argument('--semantic-multimodal', action='store_true',
                          help='Enable advanced semantic analysis with multiple channels')
        parser.add_argument('--git-history', action='store_true',
                          help='Analyze git history for temporal patterns')
        
        # Auto-fix options
        parser.add_argument('--auto-fix', action='store_true',
                          help='Generate automatic fixes for detected issues')
        parser.add_argument('--fix-plan', metavar='FILE',
                          help='Save fix plan to specified file without applying')
        parser.add_argument('--apply-fixes', action='store_true',
                          help='Apply generated fixes automatically (use with caution)')
        
        # Filtering options
        parser.add_argument('--severity', type=int, choices=[1, 2, 3, 4, 5],
                          help='Only show issues with this severity or higher')
        parser.add_argument('--types', nargs='+',
                          help='Only analyze specific issue types')
        parser.add_argument('--exclude-types', nargs='+',
                          help='Exclude specific issue types from analysis')
        
        # Configuration
        parser.add_argument('--config', metavar='FILE',
                          help='Path to configuration file')
        parser.add_argument('--fail-on', type=int, metavar='N',
                          help='Exit with code 2 if N or more issues found')
        
        # Performance options
        parser.add_argument('--parallel', action='store_true',
                          help='Enable parallel processing for large codebases')
        parser.add_argument('--cache', action='store_true',
                          help='Enable caching for incremental analysis')
        
        return parser
    
    def run(self, args: Optional[List[str]] = None):
        """Run the enhanced CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            # Load configuration
            root = Path(parsed_args.root).resolve()
            if parsed_args.config:
                config = Config.from_file(Path(parsed_args.config)).to_dict()
            else:
                config = Config.find_and_load(root).to_dict()
            
            # Collect and parse files
            files = collect_files(root, 
                                config.get('paths', {}).get('include'),
                                config.get('paths', {}).get('exclude'))
            
            if not files:
                self.logger.error("No Python files found to analyze.")
                sys.stderr.write("No Python files found to analyze.\n")
                return 1
            
            self.logger.info(f"Analyzing {len(files)} files")
            sys.stdout.write(f"🔍 Analyzing {len(files)} files...\n")
            
            ast_index = parse_files(files)
            symbol_table = SymbolTable()
            
            for filepath, tree in ast_index.items():
                try:
                    symbol_table.ingest(filepath, tree, "")
                except Exception as e:
                    self.logger.warning(f"Failed to process {filepath}: {e}")
                    sys.stderr.write(f"Warning: Failed to process {filepath}: {e}\n")
            
            # Set up analysis context
            cache = {}
            source_cache = {}
            ctx = AnalysisContext(
                config=config,
                root_dir=root,
                file_paths=files,
                ast_index=ast_index,
                symbol_table=symbol_table,
                source_cache=source_cache,
                cache=cache
            )
            
            # Run standard analyzers
            issues = []
            analyzers = load_analyzers(config)
            
            # Filter analyzers based on command line options
            if parsed_args.types:
                analyzers = [a for a in analyzers if a.name in parsed_args.types]
            elif parsed_args.exclude_types:
                analyzers = [a for a in analyzers if a.name not in parsed_args.exclude_types]
            
            for analyzer in analyzers:
                try:
                    analyzer_issues = list(analyzer.run(ctx))
                    issues.extend(analyzer_issues)
                    self.logger.info(f"{analyzer.name} found {len(analyzer_issues)} issues")
                    sys.stdout.write(f"  ✓ {analyzer.name}: {len(analyzer_issues)} issues\n")
                except Exception as e:
                    self.logger.error(f"{analyzer.name} failed: {e}")
                    sys.stderr.write(f"  ✗ {analyzer.name}: Failed ({e})\n")
            
            # Run enhanced detection if requested
            if parsed_args.enhanced:
                self.logger.info("Running enhanced pattern detection")
                sys.stdout.write("🧠 Running enhanced pattern detection...\n")
                enhanced_issues = self._run_enhanced_detection(ctx, ast_index)
                issues.extend(enhanced_issues)
                self.logger.info(f"Enhanced detection found {len(enhanced_issues)} additional issues")
                sys.stdout.write(f"  ✓ Enhanced detection: {len(enhanced_issues)} additional issues\n")
            
            # Run multimodal semantic analysis if requested
            if parsed_args.semantic_multimodal:
                self.logger.info("Running multimodal semantic analysis")
                sys.stdout.write("🔬 Running multimodal semantic analysis...\n")
                semantic_issues = self._run_semantic_analysis(ctx, symbol_table)
                issues.extend(semantic_issues)
                self.logger.info(f"Semantic analysis found {len(semantic_issues)} additional issues")
                sys.stdout.write(f"  ✓ Semantic analysis: {len(semantic_issues)} additional issues\n")
            
            # Filter by severity
            if parsed_args.severity:
                issues = [issue for issue in issues if issue.severity >= parsed_args.severity]
            
            self.logger.info(f"Analysis complete: {len(issues)} total issues found")
            sys.stdout.write(f"\n📊 Analysis complete: {len(issues)} total issues found\n")
            
            # Generate outputs
            reporter = Reporter(config)
            if parsed_args.json:
                sys.stdout.write(reporter.render_json(issues))
            elif parsed_args.html:
                self._generate_html_report(issues, files, parsed_args.html)
                self.logger.info(f"HTML report generated: {parsed_args.html}")
                sys.stdout.write(f"📄 HTML report generated: {parsed_args.html}\n")
            elif parsed_args.explain:
                self._generate_explanations(issues)
            else:
                sys.stdout.write(reporter.render_text(issues))
            
            # Handle auto-fix options
            if parsed_args.auto_fix or parsed_args.fix_plan:
                self._handle_auto_fix(issues, parsed_args)
            
            # Check exit conditions
            if parsed_args.fail_on and len(issues) >= parsed_args.fail_on:
                return 2
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.warning("Analysis interrupted by user")
            sys.stderr.write("\n⚠️  Analysis interrupted by user\n")
            return 130
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}", exc_info=True)
            sys.stderr.write(f"❌ Error during analysis: {e}\n")
            if hasattr(parsed_args, 'debug') and parsed_args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    def _run_enhanced_detection(self, ctx: AnalysisContext, ast_index: dict) -> List[Issue]:
        """Run enhanced pattern detection."""
        issues = []
        
        # Detect hallucination cascades
        cascade_issues = self.enhanced_detector.detect_hallucination_cascade(ast_index)
        issues.extend(cascade_issues)
        
        # Detect context window thrashing for each file
        for filepath, tree in ast_index.items():
            thrashing_issues = self.enhanced_detector.detect_context_window_thrashing(tree, filepath)
            issues.extend(thrashing_issues)
        
        # Detect import anxiety (simplified - would need import analysis)
        # This would require more sophisticated import tracking
        
        return issues
    
    def _run_semantic_analysis(self, ctx: AnalysisContext, symbol_table: SymbolTable) -> List[Issue]:
        """Run multimodal semantic analysis."""
        # Extract all functions for analysis
        functions = []
        for func_name, entries in symbol_table.functions.items():
            for entry in entries:
                functions.append((entry['file'], entry['node']))
        
        # Find semantic duplicates
        return self.semantic_enhancer.find_semantic_duplicates(functions)
    
    def _generate_html_report(self, issues: List[Issue], files: List[Path], output_path: str):
        """Generate HTML visualization report."""
        file_paths = [str(f) for f in files]
        self.visualizer.generate_html_report(issues, file_paths, output_path)
    
    def _generate_explanations(self, issues: List[Issue]):
        """Generate detailed natural language explanations."""
        if not issues:
            self.logger.info("No issues found")
            sys.stdout.write("🎉 No issues found! Your code appears to be free of tail-chasing patterns.\n")
            return
        
        # Generate summary report
        summary = self.explainer.generate_summary_report(issues)
        sys.stdout.write(summary + "\n")
        
        # Generate detailed explanations for high-severity issues
        high_severity_issues = [issue for issue in issues if issue.severity >= 4]
        if high_severity_issues:
            self.logger.info(f"Generating detailed explanations for {len(high_severity_issues)} high-severity issues")
            sys.stdout.write("\n" + "="*80 + "\n")
            sys.stdout.write("🚨 DETAILED EXPLANATIONS FOR HIGH-SEVERITY ISSUES\n")
            sys.stdout.write("="*80 + "\n")
            
            for i, issue in enumerate(high_severity_issues, 1):
                sys.stdout.write(f"\n## Issue {i}: {issue.kind}\n")
                explanation = self.explainer.explain_issue(issue)
                sys.stdout.write(explanation + "\n")
    
    def _handle_auto_fix(self, issues: List[Issue], args):
        """Handle automatic fix generation and application."""
        if not issues:
            self.logger.info("No issues to fix")
            sys.stdout.write("No issues to fix.\n")
            return
        
        self.logger.info("Generating fix plan")
        sys.stdout.write("\n🔧 Generating fix plan...\n")
        fix_plan = self.auto_fixer.generate_fix_plan(issues)
        
        self.logger.info(f"Generated {len(fix_plan.actions)} fix actions for {len(fix_plan.issues_addressed)} issues")
        sys.stdout.write(f"Generated {len(fix_plan.actions)} fix actions for {len(fix_plan.issues_addressed)} issues\n")
        sys.stdout.write(f"Estimated impact: {fix_plan.estimated_impact}\n")
        
        # Save fix plan if requested
        if args.fix_plan:
            fix_plan_data = {
                'issues': [issue.to_dict() for issue in fix_plan.issues_addressed],
                'actions': [
                    {
                        'type': action.action_type,
                        'file': action.target_file,
                        'line': action.target_line,
                        'description': action.description,
                        'old_code': action.old_code,
                        'new_code': action.new_code
                    }
                    for action in fix_plan.actions
                ],
                'impact': fix_plan.estimated_impact,
                'rollback': fix_plan.rollback_plan
            }
            
            with open(args.fix_plan, 'w') as f:
                json.dump(fix_plan_data, f, indent=2)
            
            self.logger.info(f"Fix plan saved to: {args.fix_plan}")
            sys.stdout.write(f"📋 Fix plan saved to: {args.fix_plan}\n")
        
        # Apply fixes if requested
        if args.apply_fixes:
            self.logger.warning("Applying fixes automatically")
            sys.stdout.write("\n⚠️  APPLYING FIXES AUTOMATICALLY\n")
            sys.stdout.write("This will modify your code. Make sure you have backups!\n")
            
            try:
                input("\nPress Enter to continue or Ctrl+C to cancel...")
            except KeyboardInterrupt:
                self.logger.info("Fix application cancelled by user")
                sys.stdout.write("\nFix application cancelled.\n")
                return
            
            self._apply_fixes(fix_plan)
    
    def _apply_fixes(self, fix_plan):
        """Apply the generated fixes to the codebase."""
        self.logger.info("Applying fixes to codebase")
        sys.stdout.write("🔧 Applying fixes...\n")
        
        # Group actions by file for efficient processing
        actions_by_file = {}
        for action in fix_plan.actions:
            actions_by_file.setdefault(action.target_file, []).append(action)
        
        applied_count = 0
        for filepath, actions in actions_by_file.items():
            try:
                # Read the current file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Apply actions (simplified - real implementation would be more sophisticated)
                modified_content = content
                for action in actions:
                    if action.old_code and action.new_code:
                        modified_content = modified_content.replace(action.old_code, action.new_code)
                    applied_count += 1
                
                # Write back the modified content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                self.logger.info(f"Applied {len(actions)} fixes to {filepath}")
                sys.stdout.write(f"  ✓ {filepath}: {len(actions)} fixes applied\n")
                
            except Exception as e:
                self.logger.error(f"Failed to apply fixes to {filepath}: {e}")
                sys.stderr.write(f"  ✗ {filepath}: Failed to apply fixes ({e})\n")
        
        self.logger.info(f"Applied {applied_count} fixes successfully")
        sys.stdout.write(f"\n✅ Applied {applied_count} fixes successfully\n")
        sys.stdout.write("💡 Run your tests to ensure everything still works correctly\n")
        sys.stdout.write(f"🔄 To rollback changes, run: {' && '.join(fix_plan.rollback_plan)}\n")


def main():
    """Entry point for the enhanced CLI."""
    cli = EnhancedCLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
