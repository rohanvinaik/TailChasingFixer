import * as vscode from 'vscode';
import * as path from 'path';
import { spawn } from 'child_process';

export interface TailChasingIssue {
    kind: string;
    message: string;
    severity: number;
    file: string;
    line: number;
    column?: number;
    symbol?: string;
    evidence?: any;
    suggestions?: string[];
}

export interface SemanticDuplicate {
    function1: string;
    function2: string;
    file1: string;
    file2: string;
    line1: number;
    line2: number;
    similarity: number;
    zScore: number;
}

export class TailChasingAnalyzer {
    private pythonPath: string;
    private cache: Map<string, TailChasingIssue[]> = new Map();
    
    constructor(private context: vscode.ExtensionContext) {
        this.pythonPath = this.getPythonPath();
    }

    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('python');
        return config.get<string>('defaultInterpreterPath') || 'python';
    }

    async analyzeFile(document: vscode.TextDocument): Promise<TailChasingIssue[]> {
        const filePath = document.uri.fsPath;
        const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
        
        if (!workspaceFolder) {
            return [];
        }

        // Check if tail-chasing-detector is installed
        const isInstalled = await this.checkInstallation();
        if (!isInstalled) {
            const install = await vscode.window.showWarningMessage(
                'tail-chasing-detector is not installed',
                'Install',
                'Cancel'
            );
            
            if (install === 'Install') {
                await this.installPackage();
            }
            return [];
        }

        return new Promise((resolve, reject) => {
            const args = [
                '-m', 'tailchasing',
                filePath,
                '--json',
                '--quiet'
            ];

            // Add semantic analysis if enabled
            const semanticEnabled = vscode.workspace
                .getConfiguration('tailchasing')
                .get('semantic.enable', true);
            
            if (semanticEnabled) {
                args.push('--semantic');
            }

            const process = spawn(this.pythonPath, args, {
                cwd: workspaceFolder.uri.fsPath
            });

            let stdout = '';
            let stderr = '';

            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            process.on('close', (code) => {
                if (code !== 0 && code !== 2) { // 2 = issues found
                    console.error('Tail-chasing analysis failed:', stderr);
                    reject(new Error(stderr || 'Analysis failed'));
                    return;
                }

                try {
                    const result = JSON.parse(stdout);
                    const issues = result.issues || [];
                    this.cache.set(filePath, issues);
                    resolve(issues);
                } catch (error) {
                    reject(error);
                }
            });
        });
    }

    async getSemanticDuplicates(): Promise<SemanticDuplicate[]> {
        const duplicates: SemanticDuplicate[] = [];
        
        // Collect all semantic duplicate issues from cache
        for (const [file, issues] of this.cache.entries()) {
            for (const issue of issues) {
                if (issue.kind === 'semantic_duplicate_function' && issue.evidence?.pair) {
                    const pair = issue.evidence.pair;
                    duplicates.push({
                        function1: pair[0].name,
                        function2: pair[1].name,
                        file1: pair[0].file,
                        file2: pair[1].file,
                        line1: pair[0].line,
                        line2: pair[1].line,
                        similarity: 1 - issue.evidence.distance,
                        zScore: issue.evidence.z_score
                    });
                }
            }
        }
        
        return duplicates;
    }

    private async checkInstallation(): Promise<boolean> {
        return new Promise((resolve) => {
            const process = spawn(this.pythonPath, ['-c', 'import tailchasing']);
            
            process.on('close', (code) => {
                resolve(code === 0);
            });
        });
    }

    private async installPackage(): Promise<void> {
        return new Promise((resolve, reject) => {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Installing tail-chasing-detector...",
                cancellable: false
            }, async () => {
                const process = spawn(this.pythonPath, [
                    '-m', 'pip', 'install', 'tail-chasing-detector'
                ]);

                process.on('close', (code) => {
                    if (code === 0) {
                        vscode.window.showInformationMessage(
                            'tail-chasing-detector installed successfully'
                        );
                        resolve();
                    } else {
                        reject(new Error('Installation failed'));
                    }
                });
            });
        });
    }

    dispose() {
        this.cache.clear();
    }
}