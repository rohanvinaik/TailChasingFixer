import * as vscode from 'vscode';
import { TailChasingAnalyzer, TailChasingIssue } from './analyzer';

export class DiagnosticProvider {
    private diagnosticCollection?: vscode.DiagnosticCollection;
    
    constructor(private analyzer: TailChasingAnalyzer) {}

    setDiagnosticCollection(collection: vscode.DiagnosticCollection) {
        this.diagnosticCollection = collection;
    }

    updateDiagnostics(document: vscode.TextDocument, issues: TailChasingIssue[]) {
        if (!this.diagnosticCollection) {
            return;
        }

        const diagnostics: vscode.Diagnostic[] = [];
        
        for (const issue of issues) {
            // Only show issues for the current file
            if (issue.file !== document.uri.fsPath && issue.file !== document.fileName) {
                continue;
            }

            const range = this.getRange(document, issue);
            const diagnostic = new vscode.Diagnostic(
                range,
                this.formatMessage(issue),
                this.getSeverity(issue)
            );

            diagnostic.code = issue.kind;
            diagnostic.source = 'tail-chasing';

            // Add code actions if suggestions are available
            if (issue.suggestions && issue.suggestions.length > 0) {
                diagnostic.relatedInformation = issue.suggestions.map(suggestion => 
                    new vscode.DiagnosticRelatedInformation(
                        new vscode.Location(document.uri, range),
                        `💡 ${suggestion}`
                    )
                );
            }

            diagnostics.push(diagnostic);
        }

        this.diagnosticCollection.set(document.uri, diagnostics);
    }

    private getRange(document: vscode.TextDocument, issue: TailChasingIssue): vscode.Range {
        const line = Math.max(0, issue.line - 1); // Convert to 0-based
        const column = issue.column || 0;
        
        // Try to find the symbol in the line
        if (issue.symbol) {
            const lineText = document.lineAt(line).text;
            const symbolIndex = lineText.indexOf(issue.symbol);
            
            if (symbolIndex >= 0) {
                return new vscode.Range(
                    line, symbolIndex,
                    line, symbolIndex + issue.symbol.length
                );
            }
        }
        
        // Default to highlighting the whole line
        const lineLength = document.lineAt(line).text.length;
        return new vscode.Range(line, 0, line, lineLength);
    }

    private formatMessage(issue: TailChasingIssue): string {
        let message = issue.message;
        
        // Add evidence details for semantic duplicates
        if (issue.kind === 'semantic_duplicate_function' && issue.evidence) {
            const pair = issue.evidence.pair;
            if (pair && pair.length === 2) {
                message += ` (${pair[0].name} ≈ ${pair[1].name}, z-score: ${issue.evidence.z_score?.toFixed(2)})`;
            }
        }
        
        return message;
    }

    private getSeverity(issue: TailChasingIssue): vscode.DiagnosticSeverity {
        const configSeverity = vscode.workspace
            .getConfiguration('tailchasing')
            .get<string>('severity', 'Warning');
        
        // Override based on issue severity
        if (issue.severity >= 4) {
            return vscode.DiagnosticSeverity.Error;
        } else if (issue.severity >= 3) {
            return vscode.DiagnosticSeverity.Warning;
        } else if (issue.severity >= 2) {
            return vscode.DiagnosticSeverity.Information;
        } else {
            return vscode.DiagnosticSeverity.Hint;
        }
    }
}