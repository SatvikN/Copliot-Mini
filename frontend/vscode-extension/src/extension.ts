import * as vscode from 'vscode';
import { CopilotMiniCompletionProvider } from './completionProvider';
import { CopilotMiniWebSocketClient } from './websocketClient';
import { CopilotMiniChatProvider } from './chatProvider';
import { CopilotMiniStatusBar } from './statusBar';
import { CopilotMiniErrorProvider, CopilotMiniHoverProvider, CopilotMiniDiagnosticsProvider } from './errorProvider';

let completionProvider: CopilotMiniCompletionProvider;
let websocketClient: CopilotMiniWebSocketClient;
let chatProvider: CopilotMiniChatProvider;
let statusBar: CopilotMiniStatusBar;
let errorProvider: CopilotMiniErrorProvider;
let hoverProvider: CopilotMiniHoverProvider;
let diagnosticsProvider: CopilotMiniDiagnosticsProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('CopilotMini extension is now active!');

    // Initialize components
    statusBar = new CopilotMiniStatusBar();
    websocketClient = new CopilotMiniWebSocketClient();
    completionProvider = new CopilotMiniCompletionProvider(websocketClient);
    chatProvider = new CopilotMiniChatProvider(websocketClient);
    errorProvider = new CopilotMiniErrorProvider(websocketClient);
    hoverProvider = new CopilotMiniHoverProvider(websocketClient);
    diagnosticsProvider = new CopilotMiniDiagnosticsProvider(websocketClient);

    // Register completion provider for supported languages
    const supportedLanguages = [
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'ruby'
    ];

    const completionDisposable = vscode.languages.registerInlineCompletionItemProvider(
        supportedLanguages,
        completionProvider
    );

    // Register error detection and hover providers
    const errorDisposable = vscode.languages.registerCodeActionsProvider(
        supportedLanguages,
        errorProvider,
        {
            providedCodeActionKinds: [vscode.CodeActionKind.QuickFix]
        }
    );

    const hoverDisposable = vscode.languages.registerHoverProvider(
        supportedLanguages,
        hoverProvider
    );

    // Register commands
    const enableCommand = vscode.commands.registerCommand('copilotMini.enable', () => {
        const config = vscode.workspace.getConfiguration('copilotMini');
        config.update('enabled', true, vscode.ConfigurationTarget.Global);
        statusBar.setEnabled(true);
        vscode.window.showInformationMessage('CopilotMini enabled');
    });

    const disableCommand = vscode.commands.registerCommand('copilotMini.disable', () => {
        const config = vscode.workspace.getConfiguration('copilotMini');
        config.update('enabled', false, vscode.ConfigurationTarget.Global);
        statusBar.setEnabled(false);
        vscode.window.showInformationMessage('CopilotMini disabled');
    });

    const openChatCommand = vscode.commands.registerCommand('copilotMini.openChat', () => {
        chatProvider.openChatPanel();
    });

    const explainCodeCommand = vscode.commands.registerCommand('copilotMini.explainCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (!selectedText) {
            vscode.window.showWarningMessage('No code selected');
            return;
        }

        const language = editor.document.languageId;
        await chatProvider.explainCode(selectedText, language);
    });

    const fixErrorCommand = vscode.commands.registerCommand('copilotMini.fixError', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        // Get diagnostics for current file
        const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
        if (diagnostics.length === 0) {
            vscode.window.showInformationMessage('No errors found in current file');
            return;
        }

        // Get the first error
        const firstError = diagnostics[0];
        const errorRange = firstError.range;
        const errorText = editor.document.getText(errorRange);
        const language = editor.document.languageId;

        await chatProvider.fixError(errorText, firstError.message, language);
    });

    const logAcceptanceCommand = vscode.commands.registerCommand('copilotMini.logAcceptance', (suggestion: string, confidence: number) => {
        // Log completion acceptance for analytics
        console.log(`CopilotMini: Completion accepted - Confidence: ${Math.round(confidence * 100)}%`);
        
        // You could send this to analytics service
        // analytics.track('completion_accepted', { confidence, suggestion_length: suggestion.length });
    });

    const toggleStatusCommand = vscode.commands.registerCommand('copilotMini.toggleStatus', () => {
        const config = vscode.workspace.getConfiguration('copilotMini');
        const enabled = config.get<boolean>('enabled', true);
        
        if (enabled) {
            vscode.commands.executeCommand('copilotMini.disable');
        } else {
            vscode.commands.executeCommand('copilotMini.enable');
        }
    });

    const fixSpecificErrorCommand = vscode.commands.registerCommand('copilotMini.fixSpecificError', async (document: vscode.TextDocument, diagnostic: vscode.Diagnostic) => {
        const errorText = document.getText(diagnostic.range);
        const language = document.languageId;
        
        await chatProvider.fixError(errorText, diagnostic.message, language);
    });

    const explainErrorCommand = vscode.commands.registerCommand('copilotMini.explainError', async (document: vscode.TextDocument, diagnostic: vscode.Diagnostic) => {
        const errorText = document.getText(diagnostic.range);
        const language = document.languageId;
        
        const message = `Can you explain this ${language} error: "${diagnostic.message}"\n\nCode: ${errorText}`;
        await chatProvider.openChatPanel();
        // Send the question directly to chat
        // Note: This would need to be implemented in the chat provider
    });

    // Register configuration change listener
    const configChangeListener = vscode.workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration('copilotMini')) {
            const config = vscode.workspace.getConfiguration('copilotMini');
            const enabled = config.get<boolean>('enabled', true);
            statusBar.setEnabled(enabled);
            
            if (enabled) {
                websocketClient.connect();
            } else {
                websocketClient.disconnect();
            }
        }
    });

    // Register document change listener for error analysis
    const documentChangeListener = vscode.workspace.onDidChangeTextDocument(async (event) => {
        const config = vscode.workspace.getConfiguration('copilotMini');
        const enableErrorFix = config.get<boolean>('enableErrorFix', true);
        
        if (enableErrorFix && diagnosticsProvider) {
            // Debounce analysis to avoid too frequent calls
            setTimeout(() => {
                diagnosticsProvider.analyzeFunctions(event.document);
            }, 1000);
        }
    });

    const documentOpenListener = vscode.workspace.onDidOpenTextDocument(async (document) => {
        const config = vscode.workspace.getConfiguration('copilotMini');
        const enableErrorFix = config.get<boolean>('enableErrorFix', true);
        
        if (enableErrorFix && diagnosticsProvider) {
            await diagnosticsProvider.analyzeFunctions(document);
        }
    });

    // Add to context subscriptions
    context.subscriptions.push(
        completionDisposable,
        errorDisposable,
        hoverDisposable,
        enableCommand,
        disableCommand,
        openChatCommand,
        explainCodeCommand,
        fixErrorCommand,
        logAcceptanceCommand,
        toggleStatusCommand,
        fixSpecificErrorCommand,
        explainErrorCommand,
        configChangeListener,
        documentChangeListener,
        documentOpenListener,
        statusBar,
        websocketClient,
        diagnosticsProvider
    );

    // Initialize connection
    const config = vscode.workspace.getConfiguration('copilotMini');
    const enabled = config.get<boolean>('enabled', true);
    
    if (enabled) {
        websocketClient.connect();
        statusBar.setEnabled(true);
    }

    // Set up status bar
    statusBar.show();
}

export function deactivate() {
    console.log('CopilotMini extension is being deactivated');
    
    if (websocketClient) {
        websocketClient.disconnect();
    }
    
    if (statusBar) {
        statusBar.dispose();
    }
} 