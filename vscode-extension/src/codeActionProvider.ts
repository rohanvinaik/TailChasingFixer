import * as vscode from 'vscode';

export class TailChasingCodeActionProvider implements vscode.CodeActionProvider {
    public provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<(vscode.CodeAction | vscode.Command)[]> {
        const actions: vscode.CodeAction[] = [];

        // Check if there are tail-chasing diagnostics in the range
        for (const diagnostic of context.diagnostics) {
            if (diagnostic.source !== 'tail-chasing') {
                continue;
            }

            switch (diagnostic.code) {
                case 'semantic_duplicate_function':
                    actions.push(this.createMergeDuplicateAction(document, diagnostic));
                    actions.push(this.createExtractSharedAction(document, diagnostic));
                    break;
                
                case 'phantom_function':
                    actions.push(this.createRemovePhantomAction(document, diagnostic));
                    actions.push(this.createImplementStubAction(document, diagnostic));
                    break;
                
                case 'circular_import':
                    actions.push(this.createResolveCircularAction(document, diagnostic));
                    break;
                
                case 'wrapper_abstraction':
                    actions.push(this.createInlineWrapperAction(document, diagnostic));
                    break;
            }

            // Add generic "Learn more" action
            actions.push(this.createLearnMoreAction(diagnostic));
        }

        return actions;
    }

    private createMergeDuplicateAction(
        document: vscode.TextDocument,
        diagnostic: vscode.Diagnostic
    ): vscode.CodeAction {
        const action = new vscode.CodeAction(
            'Merge with semantic duplicate',
            vscode.CodeActionKind.QuickFix
        );
        
        action.edit = new vscode.WorkspaceEdit();
        // Would implement actual merge logic here
        
        action.diagnostics = [diagnostic];
        action.isPreferred = true;
        
        return action;
    }

    private createExtractSharedAction(
        document: vscode.TextDocument,
        diagnostic: vscode.Diagnostic
    ): vscode.CodeAction {
        const action = new vscode.CodeAction(
            'Extract to shared function',
            vscode.CodeActionKind.Refactor
        );
        
        action.command = {
            command: 'tailChasingDetector.extractShared',
            title: 'Extract Shared Function',
            arguments: [document.uri, diagnostic.range]
        };
        
        action.diagnostics = [diagnostic];
        
        return action;
    }

    private createRemovePhantomAction(
        document: vscode.TextDocument,
        diagnostic: vscode.Diagnostic
    ): vscode.CodeAction {
        const action = new vscode.CodeAction(
            'Remove phantom function',
            vscode.CodeActionKind.QuickFix
        );
        
        action.edit = new vscode.WorkspaceEdit();
        action.edit.delete(document.uri, diagnostic.range);
        
        action.diagnostics = [diagnostic];
        
        return action;
    }

    private createImplementStubAction(
        document: vscode.TextDocument,
        diagnostic: vscode.Diagnostic
    ): vscode.CodeAction {
        const action = new vscode.CodeAction(
            'Implement function',
            vscode.CodeActionKind.QuickFix
        );
        
        action.command = {
            command: 'tailChasingDetector.implementStub',
            title: 'Implement Stub Function',
            arguments: [document.uri, diagnostic.range]
        };
        
        action.diagnostics = [diagnostic];
        
        return action;
    }

    private createResolveCircularAction(
        document: vscode.TextDocument,
        diagnostic: vscode.Diagnostic
    ): vscode.CodeAction {
        const action = new vscode.CodeAction(
            'Resolve circular import',
            vscode.CodeActionKind.QuickFix
        );
        
        action.command = {
            command: 'tailChasingDetector.resolveCircular',
            title: 'Resolve Circular Import',
            arguments: [document.uri, diagnostic]
        };
        
        action.diagnostics = [diagnostic];
        
        return action;
    }

    private createInlineWrapperAction(
        document: vscode.TextDocument,
        diagnostic: vscode.Diagnostic
    ): vscode.CodeAction {
        const action = new vscode.CodeAction(
            'Inline wrapper function',
            vscode.CodeActionKind.QuickFix
        );
        
        action.edit = new vscode.WorkspaceEdit();
        // Would implement inline logic here
        
        action.diagnostics = [diagnostic];
        
        return action;
    }

    private createLearnMoreAction(diagnostic: vscode.Diagnostic): vscode.CodeAction {
        const action = new vscode.CodeAction(
            'Learn more about tail-chasing patterns',
            vscode.CodeActionKind.Empty
        );
        
        action.command = {
            command: 'vscode.open',
            title: 'Open Documentation',
            arguments: [vscode.Uri.parse('https://tail-chasing-detector.readthedocs.io/patterns/' + diagnostic.code)]
        };
        
        return action;
    }
}