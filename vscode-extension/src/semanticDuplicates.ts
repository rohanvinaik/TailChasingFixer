import * as vscode from 'vscode';
import { TailChasingAnalyzer } from './analyzer';

export class SemanticDuplicateProvider implements vscode.CodeLensProvider {
    private _onDidChangeCodeLenses: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
    public readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChangeCodeLenses.event;

    constructor(private analyzer: TailChasingAnalyzer) {
        vscode.workspace.onDidChangeConfiguration((_) => {
            this._onDidChangeCodeLenses.fire();
        });
    }

    async provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): Promise<vscode.CodeLens[]> {
        const codeLenses: vscode.CodeLens[] = [];
        
        // Get semantic duplicates
        const duplicates = await this.analyzer.getSemanticDuplicates();
        
        // Group duplicates by file and function
        const duplicateMap = new Map<string, any[]>();
        
        for (const dup of duplicates) {
            if (dup.file1 === document.uri.fsPath) {
                const key = `${dup.file1}:${dup.line1}`;
                if (!duplicateMap.has(key)) {
                    duplicateMap.set(key, []);
                }
                duplicateMap.get(key)!.push(dup);
            }
            
            if (dup.file2 === document.uri.fsPath) {
                const key = `${dup.file2}:${dup.line2}`;
                if (!duplicateMap.has(key)) {
                    duplicateMap.set(key, []);
                }
                duplicateMap.get(key)!.push({
                    ...dup,
                    // Swap for consistent display
                    function1: dup.function2,
                    function2: dup.function1,
                    file1: dup.file2,
                    file2: dup.file1,
                    line1: dup.line2,
                    line2: dup.line1
                });
            }
        }
        
        // Create code lenses
        for (const [key, dups] of duplicateMap.entries()) {
            const [file, lineStr] = key.split(':');
            const line = parseInt(lineStr) - 1; // Convert to 0-based
            
            if (line >= 0 && line < document.lineCount) {
                const range = new vscode.Range(line, 0, line, 0);
                
                const title = dups.length === 1
                    ? `🔄 Semantic duplicate of ${dups[0].function2}`
                    : `🔄 ${dups.length} semantic duplicates`;
                
                const codeLens = new vscode.CodeLens(range, {
                    title,
                    command: 'tailchasing.showDuplicateDetails',
                    arguments: [dups]
                });
                
                codeLenses.push(codeLens);
            }
        }
        
        return codeLenses;
    }

    resolveCodeLens(codeLens: vscode.CodeLens, token: vscode.CancellationToken): vscode.CodeLens {
        return codeLens;
    }
}