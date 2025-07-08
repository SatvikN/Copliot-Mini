import * as vscode from 'vscode';
import { CopilotMiniWebSocketClient } from './websocketClient';

export class CopilotMiniCompletionProvider implements vscode.InlineCompletionItemProvider {
    private readonly webSocketClient: CopilotMiniWebSocketClient;
    private readonly debounceTime = 150; // ms
    private debounceTimer: NodeJS.Timeout | null = null;

    constructor(webSocketClient: CopilotMiniWebSocketClient) {
        this.webSocketClient = webSocketClient;
    }

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | vscode.InlineCompletionList | undefined> {
        // Check if extension is enabled
        const config = vscode.workspace.getConfiguration('copilotMini');
        const enabled = config.get<boolean>('enabled', true);
        
        if (!enabled) {
            return [];
        }

        // Check if WebSocket is connected
        if (!this.webSocketClient.isConnected()) {
            return [];
        }

        // Check if we should trigger completion
        if (!this.shouldTriggerCompletion(document, position, context)) {
            return [];
        }

        try {
            // Cancel previous request if exists
            if (this.debounceTimer) {
                clearTimeout(this.debounceTimer);
            }

            // Debounce completion requests
            return new Promise((resolve) => {
                this.debounceTimer = setTimeout(async () => {
                    try {
                        const completions = await this.getCompletions(document, position, token);
                        resolve(completions);
                    } catch (error) {
                        console.error('Error getting completions:', error);
                        resolve([]);
                    }
                }, this.debounceTime);
            });

        } catch (error) {
            console.error('Error in completion provider:', error);
            return [];
        }
    }

    private async getCompletions(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {
        const language = document.languageId;
        const code = this.getCodeContext(document, position);
        const cursorPosition = document.offsetAt(position);
        
        const config = vscode.workspace.getConfiguration('copilotMini');
        const maxSuggestions = config.get<number>('maxSuggestions', 3);

        try {
            const response = await this.webSocketClient.sendCompletionRequest(
                code,
                language,
                cursorPosition,
                maxSuggestions
            );

            if (token.isCancellationRequested) {
                return [];
            }

            return this.formatCompletions(response.suggestions, response.confidence_scores, position);

        } catch (error) {
            console.error('Failed to get completions from server:', error);
            return [];
        }
    }

    private getCodeContext(document: vscode.TextDocument, position: vscode.Position): string {
        const config = vscode.workspace.getConfiguration('copilotMini');
        const contextLines = config.get<number>('contextLines', 50);
        
        // Get context around cursor position
        const startLine = Math.max(0, position.line - contextLines);
        const endLine = Math.min(document.lineCount - 1, position.line + contextLines);
        
        const range = new vscode.Range(startLine, 0, endLine, document.lineAt(endLine).text.length);
        return document.getText(range);
    }

    private formatCompletions(
        suggestions: string[],
        confidenceScores: number[],
        position: vscode.Position
    ): vscode.InlineCompletionItem[] {
        const config = vscode.workspace.getConfiguration('copilotMini');
        const minConfidence = config.get<number>('minConfidence', 0.3);

        return suggestions
            .map((suggestion, index) => {
                const confidence = confidenceScores[index] || 0;
                
                // Filter out low confidence suggestions
                if (confidence < minConfidence) {
                    return null;
                }

                // Clean up the suggestion
                const cleanSuggestion = this.cleanSuggestion(suggestion);
                
                if (!cleanSuggestion) {
                    return null;
                }

                return new vscode.InlineCompletionItem(
                    cleanSuggestion,
                    new vscode.Range(position, position),
                    {
                        title: `CopilotMini (${Math.round(confidence * 100)}%)`,
                        command: 'copilotMini.logAcceptance',
                        arguments: [suggestion, confidence]
                    }
                );
            })
            .filter((item): item is vscode.InlineCompletionItem => item !== null)
            .slice(0, 3); // Limit to top 3 suggestions
    }

    private cleanSuggestion(suggestion: string): string {
        // Remove leading/trailing whitespace
        suggestion = suggestion.trim();
        
        // Remove common prefixes that might be artifacts
        const prefixesToRemove = ['```', '```python', '```javascript', '```typescript'];
        for (const prefix of prefixesToRemove) {
            if (suggestion.startsWith(prefix)) {
                suggestion = suggestion.substring(prefix.length).trim();
            }
        }
        
        // Remove common suffixes
        const suffixesToRemove = ['```'];
        for (const suffix of suffixesToRemove) {
            if (suggestion.endsWith(suffix)) {
                suggestion = suggestion.substring(0, suggestion.length - suffix.length).trim();
            }
        }
        
        return suggestion;
    }

    private shouldTriggerCompletion(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext
    ): boolean {
        const config = vscode.workspace.getConfiguration('copilotMini');
        
        // Check if autocomplete is enabled for this language
        const supportedLanguages = config.get<string[]>('supportedLanguages', [
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'ruby'
        ]);
        
        if (!supportedLanguages.includes(document.languageId)) {
            return false;
        }

        // Don't trigger on very short lines or at beginning of file
        if (position.line === 0 && position.character < 2) {
            return false;
        }

        const currentLine = document.lineAt(position.line);
        const textBeforeCursor = currentLine.text.substring(0, position.character);
        
        // Don't trigger in comments (basic check)
        if (this.isInComment(textBeforeCursor, document.languageId)) {
            return false;
        }
        
        // Don't trigger in strings (basic check)
        if (this.isInString(textBeforeCursor)) {
            return false;
        }
        
        // Trigger if we have enough context
        return textBeforeCursor.trim().length >= 2;
    }

    private isInComment(text: string, language: string): boolean {
        // Basic comment detection
        const commentPatterns: { [key: string]: RegExp[] } = {
            'python': [/#.*$/],
            'javascript': [/\/\/.*$/, /\/\*[\s\S]*?\*\//],
            'typescript': [/\/\/.*$/, /\/\*[\s\S]*?\*\//],
            'java': [/\/\/.*$/, /\/\*[\s\S]*?\*\//],
            'cpp': [/\/\/.*$/, /\/\*[\s\S]*?\*\//],
            'c': [/\/\/.*$/, /\/\*[\s\S]*?\*\//],
            'go': [/\/\/.*$/, /\/\*[\s\S]*?\*\//],
            'rust': [/\/\/.*$/, /\/\*[\s\S]*?\*\//],
            'php': [/\/\/.*$/, /#.*$/, /\/\*[\s\S]*?\*\//],
            'ruby': [/#.*$/]
        };

        const patterns = commentPatterns[language];
        if (!patterns) {
            return false;
        }

        return patterns.some(pattern => pattern.test(text));
    }

    private isInString(text: string): boolean {
        // Basic string detection - count quotes
        const singleQuotes = (text.match(/'/g) || []).length;
        const doubleQuotes = (text.match(/"/g) || []).length;
        const backticks = (text.match(/`/g) || []).length;
        
        return (singleQuotes % 2 === 1) || (doubleQuotes % 2 === 1) || (backticks % 2 === 1);
    }
} 