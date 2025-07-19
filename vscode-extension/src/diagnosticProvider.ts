import * as vscode from 'vscode';

export class TailChasingDiagnosticProvider {
    private diagnosticCollection: vscode.DiagnosticCollection;

    constructor(diagnosticCollection: vscode.DiagnosticCollection) {
        this.diagnosticCollection = diagnosticCollection;
    }

    updateDiagnostics(document: vscode.TextDocument, issues: any[]): void {
        const diagnostics: vscode.Diagnostic[] = [];

        for (const issue of issues) {
            const range = new vscode.Range(
                issue.line - 1,
                issue.column || 0,
                issue.line - 1,
                Number.MAX_VALUE
            );

            const diagnostic = new vscode.Diagnostic(
                range,
                issue.message,
                this.getSeverity(issue.severity)
            );

            diagnostic.code = issue.kind;
            diagnostic.source = 'tail-chasing';

            // Add related information if available
            if (issue.evidence && issue.evidence.pair) {
                diagnostic.relatedInformation = [
                    new vscode.DiagnosticRelatedInformation(
                        new vscode.Location(
                            vscode.Uri.file(issue.evidence.pair[1].file),
                            new vscode.Position(issue.evidence.pair[1].line - 1, 0)
                        ),
                        `Semantic duplicate: ${issue.evidence.pair[1].name}`
                    )
                ];
            }

            diagnostics.push(diagnostic);
        }

        this.diagnosticCollection.set(document.uri, diagnostics);
    }

    private getSeverity(severity: number): vscode.DiagnosticSeverity {
        const config = vscode.workspace.getConfiguration('tailChasingDetector');
        const configuredSeverity = config.get<string>('severityLevel', 'warning');

        // Override based on configuration
        switch (configuredSeverity) {
            case 'error':
                return vscode.DiagnosticSeverity.Error;
            case 'warning':
                return vscode.DiagnosticSeverity.Warning;
            case 'information':
                return vscode.DiagnosticSeverity.Information;
            case 'hint':
                return vscode.DiagnosticSeverity.Hint;
        }

        // Default based on issue severity
        if (severity >= 4) {
            return vscode.DiagnosticSeverity.Error;
        } else if (severity >= 3) {
            return vscode.DiagnosticSeverity.Warning;
        } else if (severity >= 2) {
            return vscode.DiagnosticSeverity.Information;
        }
        return vscode.DiagnosticSeverity.Hint;
    }

    clear(): void {
        this.diagnosticCollection.clear();
    }
}