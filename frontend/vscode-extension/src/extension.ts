import * as vscode from 'vscode';
import { CopilotMiniCompletionProvider } from './completionProvider';
import { CopilotMiniWebSocketClient } from './websocketClient';
import { CopilotMiniChatProvider } from './chatProvider';
import { CopilotMiniStatusBar } from './statusBar';

let completionProvider: CopilotMiniCompletionProvider;
let websocketClient: CopilotMiniWebSocketClient;
let chatProvider: CopilotMiniChatProvider;
let statusBar: CopilotMiniStatusBar;

export function activate(context: vscode.ExtensionContext) {
    console.log('CopilotMini extension is now active!');

    // Initialize components
    statusBar = new CopilotMiniStatusBar();
    websocketClient = new CopilotMiniWebSocketClient();
    completionProvider = new CopilotMiniCompletionProvider(websocketClient);
    chatProvider = new CopilotMiniChatProvider(websocketClient);

    // Register completion provider for supported languages
    const supportedLanguages = [
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'ruby'
    ];

    const completionDisposable = vscode.languages.registerInlineCompletionItemProvider(
        supportedLanguages,
        completionProvider
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

    // Add to context subscriptions
    context.subscriptions.push(
        completionDisposable,
        enableCommand,
        disableCommand,
        openChatCommand,
        explainCodeCommand,
        fixErrorCommand,
        configChangeListener,
        statusBar,
        websocketClient
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