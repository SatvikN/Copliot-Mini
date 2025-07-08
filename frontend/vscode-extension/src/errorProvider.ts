import * as vscode from 'vscode';
import { CopilotMiniWebSocketClient } from './websocketClient';

export class CopilotMiniErrorProvider implements vscode.CodeActionProvider {
    private readonly webSocketClient: CopilotMiniWebSocketClient;

    constructor(webSocketClient: CopilotMiniWebSocketClient) {
        this.webSocketClient = webSocketClient;
    }

    async provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.CodeAction[]> {
        const actions: vscode.CodeAction[] = [];

        // Only provide actions if there are diagnostics
        if (context.diagnostics.length === 0) {
            return actions;
        }

        for (const diagnostic of context.diagnostics) {
            // Create "Fix with CopilotMini" action
            const fixAction = new vscode.CodeAction(
                `🤖 Fix with CopilotMini: ${diagnostic.message}`,
                vscode.CodeActionKind.QuickFix
            );

            fixAction.command = {
                title: 'Fix with CopilotMini',
                command: 'copilotMini.fixSpecificError',
                arguments: [document, diagnostic]
            };

            fixAction.diagnostics = [diagnostic];
            fixAction.isPreferred = true; // Make it appear at the top

            actions.push(fixAction);
        }

        // Add "Explain Error" action
        if (context.diagnostics.length > 0) {
            const explainAction = new vscode.CodeAction(
                '💡 Explain this error with CopilotMini',
                vscode.CodeActionKind.QuickFix
            );

            explainAction.command = {
                title: 'Explain Error',
                command: 'copilotMini.explainError',
                arguments: [document, context.diagnostics[0]]
            };

            actions.push(explainAction);
        }

        return actions;
    }
}

export class CopilotMiniHoverProvider implements vscode.HoverProvider {
    private readonly webSocketClient: CopilotMiniWebSocketClient;

    constructor(webSocketClient: CopilotMiniWebSocketClient) {
        this.webSocketClient = webSocketClient;
    }

    async provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<vscode.Hover | undefined> {
        // Check if there are diagnostics at this position
        const diagnostics = vscode.languages.getDiagnostics(document.uri);
        const relevantDiagnostics = diagnostics.filter(diagnostic => 
            diagnostic.range.contains(position)
        );

        if (relevantDiagnostics.length === 0) {
            return undefined;
        }

        const diagnostic = relevantDiagnostics[0];
        
        // Create hover content with error explanation and fix suggestion
        const hoverContent = new vscode.MarkdownString();
        hoverContent.isTrusted = true;
        
        hoverContent.appendMarkdown(`**🚨 ${diagnostic.source || 'Error'}**: ${diagnostic.message}\n\n`);
        hoverContent.appendMarkdown(`**Severity**: ${this.getSeverityText(diagnostic.severity)}\n\n`);
        
        // Add quick fix buttons
        hoverContent.appendMarkdown(`[🤖 Fix with CopilotMini](command:copilotMini.fixSpecificError?${encodeURIComponent(JSON.stringify([document.uri.toString(), diagnostic]))}) | `);
        hoverContent.appendMarkdown(`[💡 Explain Error](command:copilotMini.explainError?${encodeURIComponent(JSON.stringify([document.uri.toString(), diagnostic]))})\n\n`);

        // Add common error patterns and suggestions
        const suggestion = this.getCommonErrorSuggestion(diagnostic.message, document.languageId);
        if (suggestion) {
            hoverContent.appendMarkdown(`**💡 Quick Tip**: ${suggestion}`);
        }

        return new vscode.Hover(hoverContent, diagnostic.range);
    }

    private getSeverityText(severity: vscode.DiagnosticSeverity): string {
        switch (severity) {
            case vscode.DiagnosticSeverity.Error:
                return '❌ Error';
            case vscode.DiagnosticSeverity.Warning:
                return '⚠️ Warning';
            case vscode.DiagnosticSeverity.Information:
                return 'ℹ️ Info';
            case vscode.DiagnosticSeverity.Hint:
                return '💡 Hint';
            default:
                return 'Unknown';
        }
    }

    private getCommonErrorSuggestion(message: string, language: string): string | undefined {
        const messageLower = message.toLowerCase();

        // Common error patterns and suggestions
        const errorPatterns = {
            'python': {
                'indentationerror': 'Check your indentation - Python requires consistent spaces or tabs.',
                'syntaxerror': 'Look for missing colons, parentheses, or quotes.',
                'nameerror': 'Make sure the variable is defined before using it.',
                'importerror': 'Check if the module is installed: `pip install module_name`',
                'typeerror': 'Check if you\'re calling the right method on the right object type.',
            },
            'javascript': {
                'referenceerror': 'Variable is not defined. Declare it with let, const, or var.',
                'syntaxerror': 'Check for missing semicolons, brackets, or quotes.',
                'typeerror': 'Check if the object/function exists before calling it.',
                'cannot read property': 'Object might be null or undefined. Use optional chaining (?.)',
            },
            'typescript': {
                'cannot find name': 'Import the module or check the variable name.',
                'type': 'Check if the types match what the function expects.',
                'property does not exist': 'Add the property to the interface or use type assertion.',
            }
        };

        const patterns = errorPatterns[language as keyof typeof errorPatterns];
        if (!patterns) return undefined;

        for (const [pattern, suggestion] of Object.entries(patterns)) {
            if (messageLower.includes(pattern)) {
                return suggestion;
            }
        }

        return undefined;
    }
}

export class CopilotMiniDiagnosticsProvider {
    private readonly webSocketClient: CopilotMiniWebSocketClient;
    private readonly diagnosticCollection: vscode.DiagnosticCollection;

    constructor(webSocketClient: CopilotMiniWebSocketClient) {
        this.webSocketClient = webSocketClient;
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('copilotMini');
    }

    async analyzeFunctions(document: vscode.TextDocument): Promise<void> {
        if (!this.webSocketClient.isConnected()) {
            return;
        }

        const config = vscode.workspace.getConfiguration('copilotMini');
        const enableErrorFix = config.get<boolean>('enableErrorFix', true);
        
        if (!enableErrorFix) {
            return;
        }

        const diagnostics: vscode.Diagnostic[] = [];
        const text = document.getText();
        const lines = text.split('\n');

        // Analyze code for potential issues
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNumber = i;

            // Check for common code smells and issues
            const issues = this.analyzeLineForIssues(line, lineNumber, document.languageId);
            diagnostics.push(...issues);
        }

        this.diagnosticCollection.set(document.uri, diagnostics);
    }

    private analyzeLineForIssues(line: string, lineNumber: number, language: string): vscode.Diagnostic[] {
        const diagnostics: vscode.Diagnostic[] = [];
        const trimmedLine = line.trim();

        // Language-specific analysis
        if (language === 'python') {
            // Check for print without parentheses (Python 2 style)
            if (/print\s+(?!\()/.test(trimmedLine) && !trimmedLine.includes('#')) {
                const range = new vscode.Range(lineNumber, 0, lineNumber, line.length);
                const diagnostic = new vscode.Diagnostic(
                    range,
                    'Consider using print() with parentheses for Python 3 compatibility',
                    vscode.DiagnosticSeverity.Information
                );
                diagnostic.source = 'CopilotMini';
                diagnostic.code = 'python-print-style';
                diagnostics.push(diagnostic);
            }

            // Check for missing return type hints
            if (/def\s+\w+\([^)]*\)\s*:/.test(trimmedLine) && !trimmedLine.includes('->')) {
                const range = new vscode.Range(lineNumber, 0, lineNumber, line.length);
                const diagnostic = new vscode.Diagnostic(
                    range,
                    'Consider adding a return type hint for better code documentation',
                    vscode.DiagnosticSeverity.Hint
                );
                diagnostic.source = 'CopilotMini';
                diagnostic.code = 'python-missing-return-type';
                diagnostics.push(diagnostic);
            }
        }

        if (language === 'javascript' || language === 'typescript') {
            // Check for var usage
            if (/\bvar\s+/.test(trimmedLine)) {
                const range = new vscode.Range(lineNumber, 0, lineNumber, line.length);
                const diagnostic = new vscode.Diagnostic(
                    range,
                    'Consider using "let" or "const" instead of "var"',
                    vscode.DiagnosticSeverity.Information
                );
                diagnostic.source = 'CopilotMini';
                diagnostic.code = 'js-no-var';
                diagnostics.push(diagnostic);
            }

            // Check for == usage
            if (/==(?!=)/.test(trimmedLine) && !trimmedLine.includes('===')) {
                const range = new vscode.Range(lineNumber, 0, lineNumber, line.length);
                const diagnostic = new vscode.Diagnostic(
                    range,
                    'Consider using "===" for strict equality comparison',
                    vscode.DiagnosticSeverity.Information
                );
                diagnostic.source = 'CopilotMini';
                diagnostic.code = 'js-strict-equality';
                diagnostics.push(diagnostic);
            }
        }

        return diagnostics;
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
    }
} 