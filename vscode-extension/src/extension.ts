import * as vscode from 'vscode';
import { TailChasingAnalyzer } from './analyzer';
import { SemanticDuplicateProvider } from './semanticDuplicates';
import { DiagnosticProvider } from './diagnostics';

let analyzer: TailChasingAnalyzer;
let diagnosticProvider: DiagnosticProvider;
let semanticProvider: SemanticDuplicateProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('Tail-Chasing Detector is now active!');

    // Initialize components
    analyzer = new TailChasingAnalyzer(context);
    diagnosticProvider = new DiagnosticProvider(analyzer);
    semanticProvider = new SemanticDuplicateProvider(analyzer);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('tailchasing.analyze', () => {
            analyzeCurrentFile();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('tailchasing.showSemanticDuplicates', () => {
            showSemanticDuplicates();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('tailchasing.toggleRealTime', () => {
            toggleRealTimeAnalysis();
        })
    );

    // Register diagnostic provider
    const diagnosticCollection = vscode.languages.createDiagnosticCollection('tailchasing');
    context.subscriptions.push(diagnosticCollection);
    diagnosticProvider.setDiagnosticCollection(diagnosticCollection);

    // Register code lens provider for semantic duplicates
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(
            { scheme: 'file', language: 'python' },
            semanticProvider
        )
    );

    // Watch for file changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((e) => {
            if (e.document.languageId === 'python' && getRealTimeEnabled()) {
                analyzeDocument(e.document);
            }
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument((document) => {
            if (document.languageId === 'python') {
                analyzeDocument(document);
            }
        })
    );

    // Analyze all open Python files
    vscode.workspace.textDocuments.forEach((document) => {
        if (document.languageId === 'python') {
            analyzeDocument(document);
        }
    });

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = "$(eye) Tail-Chasing";
    statusBarItem.tooltip = "Click to analyze current file";
    statusBarItem.command = 'tailchasing.analyze';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
}

async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'python') {
        vscode.window.showWarningMessage('Please open a Python file to analyze');
        return;
    }

    await analyzeDocument(editor.document, true);
}

async function analyzeDocument(document: vscode.TextDocument, showProgress: boolean = false) {
    if (showProgress) {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Analyzing for tail-chasing patterns...",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0 });
            
            try {
                const issues = await analyzer.analyzeFile(document);
                diagnosticProvider.updateDiagnostics(document, issues);
                
                progress.report({ increment: 100 });
                
                if (issues.length === 0) {
                    vscode.window.showInformationMessage('No tail-chasing patterns detected!');
                } else {
                    const severity = issues.reduce((max, issue) => 
                        Math.max(max, issue.severity), 0);
                    
                    if (severity >= 4) {
                        vscode.window.showErrorMessage(
                            `Critical tail-chasing patterns detected: ${issues.length} issues`
                        );
                    } else if (severity >= 3) {
                        vscode.window.showWarningMessage(
                            `Tail-chasing patterns detected: ${issues.length} issues`
                        );
                    } else {
                        vscode.window.showInformationMessage(
                            `Minor issues detected: ${issues.length} issues`
                        );
                    }
                }
            } catch (error) {
                vscode.window.showErrorMessage(
                    `Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`
                );
            }
        });
    } else {
        try {
            const issues = await analyzer.analyzeFile(document);
            diagnosticProvider.updateDiagnostics(document, issues);
        } catch (error) {
            console.error('Analysis error:', error);
        }
    }
}

async function showSemanticDuplicates() {
    const duplicates = await analyzer.getSemanticDuplicates();
    
    if (duplicates.length === 0) {
        vscode.window.showInformationMessage('No semantic duplicates found');
        return;
    }

    // Create a quick pick to show duplicates
    const items = duplicates.map(dup => ({
        label: `$(symbol-function) ${dup.function1} ≈ ${dup.function2}`,
        description: `Similarity: ${(dup.similarity * 100).toFixed(1)}%`,
        detail: `${dup.file1}:${dup.line1} ↔ ${dup.file2}:${dup.line2}`,
        duplicate: dup
    }));

    const selected = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select a semantic duplicate to view',
        canPickMany: false
    });

    if (selected) {
        // Show diff view
        const uri1 = vscode.Uri.file(selected.duplicate.file1);
        const uri2 = vscode.Uri.file(selected.duplicate.file2);
        
        await vscode.commands.executeCommand(
            'vscode.diff',
            uri1,
            uri2,
            `${selected.duplicate.function1} ↔ ${selected.duplicate.function2}`
        );
    }
}

function toggleRealTimeAnalysis() {
    const config = vscode.workspace.getConfiguration('tailchasing');
    const current = config.get('realTime', false);
    config.update('realTime', !current, vscode.ConfigurationTarget.Global);
    
    vscode.window.showInformationMessage(
        `Real-time analysis ${!current ? 'enabled' : 'disabled'}`
    );
}

function getRealTimeEnabled(): boolean {
    return vscode.workspace.getConfiguration('tailchasing').get('realTime', false);
}

export function deactivate() {
    if (analyzer) {
        analyzer.dispose();
    }
}