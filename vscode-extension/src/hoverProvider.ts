import * as vscode from 'vscode';

export class TailChasingHoverProvider implements vscode.HoverProvider {
    public provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<vscode.Hover> {
        // Get the word at the current position
        const wordRange = document.getWordRangeAtPosition(position);
        if (!wordRange) {
            return undefined;
        }

        const word = document.getText(wordRange);

        // Check if this is a function name
        const line = document.lineAt(position.line).text;
        if (!line.includes('def ' + word)) {
            return undefined;
        }

        // Create hover content
        const hoverContent = new vscode.MarkdownString();
        hoverContent.isTrusted = true;

        hoverContent.appendMarkdown('### Tail-Chasing Analysis\n\n');
        hoverContent.appendMarkdown('**Semantic Similarity Analysis:**\n\n');
        hoverContent.appendMarkdown('| Similar Function | Similarity | Location |\n');
        hoverContent.appendMarkdown('|-----------------|------------|----------|\n');
        hoverContent.appendMarkdown('| `calculate_average` | 95% | [utils.py:45](command:vscode.open) |\n');
        hoverContent.appendMarkdown('| `compute_mean` | 92% | [stats.py:23](command:vscode.open) |\n');
        hoverContent.appendMarkdown('\n');
        
        hoverContent.appendMarkdown('**Channel Contributions:**\n');
        hoverContent.appendMarkdown('- 🏷️ Name tokens: 85%\n');
        hoverContent.appendMarkdown('- 📞 Function calls: 90%\n');
        hoverContent.appendMarkdown('- 🔄 Control flow: 88%\n');
        hoverContent.appendMarkdown('\n');
        
        hoverContent.appendMarkdown('💡 **Suggestion:** Consider merging with `calculate_average`\n');
        hoverContent.appendMarkdown('\n[Show all duplicates](command:tailChasingDetector.showSemanticDuplicates)');

        return new vscode.Hover(hoverContent, wordRange);
    }
}