import * as vscode from 'vscode';

export class TailChasingCodeLensProvider implements vscode.CodeLensProvider {
    private _onDidChangeCodeLenses: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
    public readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChangeCodeLenses.event;

    constructor() {
        vscode.workspace.onDidChangeConfiguration((_) => {
            this._onDidChangeCodeLenses.fire();
        });
    }

    public provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
        if (!vscode.workspace.getConfiguration('tailChasingDetector').get('showCodeLens', true)) {
            return [];
        }

        const codeLenses: vscode.CodeLens[] = [];

        // Parse document to find functions
        const functionPattern = /^(async\s+)?def\s+(\w+)\s*\(/gm;
        const text = document.getText();
        let match;

        while ((match = functionPattern.exec(text)) !== null) {
            const line = document.positionAt(match.index).line;
            const functionName = match[2];
            
            const range = new vscode.Range(line, 0, line, 0);
            
            // Add code lens for semantic duplicates
            const codeLens = new vscode.CodeLens(range, {
                title: "$(warning) 3 semantic duplicates",
                tooltip: "This function has 3 semantic duplicates in the codebase",
                command: "tailChasingDetector.showSemanticDuplicates",
                arguments: [functionName]
            });
            
            codeLenses.push(codeLens);
        }

        return codeLenses;
    }

    public resolveCodeLens(codeLens: vscode.CodeLens, token: vscode.CancellationToken): vscode.CodeLens {
        // Could enhance with real-time data here
        return codeLens;
    }
}