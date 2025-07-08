import * as vscode from 'vscode';
import { CopilotMiniWebSocketClient } from './websocketClient';

export class CopilotMiniChatProvider implements vscode.Disposable {
    private readonly webSocketClient: CopilotMiniWebSocketClient;
    private chatPanel: vscode.WebviewPanel | undefined;

    constructor(webSocketClient: CopilotMiniWebSocketClient) {
        this.webSocketClient = webSocketClient;
    }

    async openChatPanel(): Promise<void> {
        if (this.chatPanel) {
            this.chatPanel.reveal(vscode.ViewColumn.Beside);
            return;
        }

        this.chatPanel = vscode.window.createWebviewPanel(
            'copilotMiniChat',
            'CopilotMini Chat',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: []
            }
        );

        this.chatPanel.webview.html = this.getChatHtml();

        // Handle messages from the webview
        this.chatPanel.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.type) {
                    case 'sendMessage':
                        await this.handleChatMessage(message.text, message.codeContext);
                        break;
                    case 'insertCode':
                        await this.insertCodeAtCursor(message.code);
                        break;
                    case 'replaceCode':
                        await this.replaceSelectedCode(message.code);
                        break;
                }
            },
            undefined,
            []
        );

        // Clean up when panel is closed
        this.chatPanel.onDidDispose(() => {
            this.chatPanel = undefined;
        });
    }

    async explainCode(code: string, language: string): Promise<void> {
        await this.openChatPanel();
        
        const message = `Please explain this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\``;
        await this.handleChatMessage(message, code, language);
    }

    async fixError(code: string, errorMessage: string, language: string): Promise<void> {
        await this.openChatPanel();
        
        const message = `I'm getting this error: "${errorMessage}"\n\nIn this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\`\n\nCan you help me fix it?`;
        await this.handleChatMessage(message, code, language);
    }

    private async handleChatMessage(message: string, codeContext?: string, language?: string): Promise<void> {
        if (!this.chatPanel) {
            return;
        }

        // Show user message in chat
        this.chatPanel.webview.postMessage({
            type: 'userMessage',
            message: message
        });

        // Show typing indicator
        this.chatPanel.webview.postMessage({
            type: 'typing',
            isTyping: true
        });

        try {
            const response = await this.webSocketClient.sendChatRequest(
                message,
                codeContext,
                language
            );

            // Hide typing indicator
            this.chatPanel.webview.postMessage({
                type: 'typing',
                isTyping: false
            });

            // Show assistant response
            this.chatPanel.webview.postMessage({
                type: 'assistantMessage',
                message: response.response,
                codeSuggestions: response.code_suggestions || []
            });

        } catch (error) {
            // Hide typing indicator
            this.chatPanel.webview.postMessage({
                type: 'typing',
                isTyping: false
            });

            // Show error message
            this.chatPanel.webview.postMessage({
                type: 'error',
                message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
            });
        }
    }

    private async insertCodeAtCursor(code: string): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const position = editor.selection.active;
        await editor.edit(editBuilder => {
            editBuilder.insert(position, code);
        });

        // Format the inserted code
        await vscode.commands.executeCommand('editor.action.formatDocument');
    }

    private async replaceSelectedCode(code: string): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showWarningMessage('No code selected');
            return;
        }

        await editor.edit(editBuilder => {
            editBuilder.replace(selection, code);
        });

        // Format the replaced code
        await vscode.commands.executeCommand('editor.action.formatDocument');
    }

    private getChatHtml(): string {
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CopilotMini Chat</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            margin: 0;
            padding: 10px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            margin-bottom: 10px;
        }

        .message {
            margin-bottom: 15px;
            padding: 8px 12px;
            border-radius: 8px;
        }

        .user-message {
            background-color: var(--vscode-textCodeBlock-background);
            border-left: 3px solid var(--vscode-textLink-foreground);
        }

        .assistant-message {
            background-color: var(--vscode-editor-background);
            border-left: 3px solid var(--vscode-notificationsInfoIcon-foreground);
        }

        .error-message {
            background-color: var(--vscode-inputValidation-errorBackground);
            border-left: 3px solid var(--vscode-inputValidation-errorBorder);
            color: var(--vscode-inputValidation-errorForeground);
        }

        .code-suggestion {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-family: var(--vscode-editor-font-family);
            white-space: pre-wrap;
            position: relative;
        }

        .code-actions {
            margin-top: 5px;
        }

        .code-action-btn {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 4px 8px;
            margin-right: 5px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }

        .code-action-btn:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        .input-container {
            display: flex;
            gap: 10px;
        }

        .message-input {
            flex: 1;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 8px 12px;
            border-radius: 4px;
            font-family: inherit;
            font-size: inherit;
        }

        .send-btn {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }

        .send-btn:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .typing-indicator {
            font-style: italic;
            color: var(--vscode-descriptionForeground);
            padding: 8px 12px;
        }

        .typing-dots {
            display: inline-block;
            animation: typing 1.5s infinite;
        }

        @keyframes typing {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
    </style>
</head>
<body>
    <div class="chat-container" id="chatContainer">
        <div class="message assistant-message">
            <div>👋 Hi! I'm CopilotMini. I can help you with code explanations, error fixes, and general programming questions. How can I assist you today?</div>
        </div>
    </div>

    <div class="input-container">
        <input type="text" class="message-input" id="messageInput" placeholder="Ask me anything about your code..." />
        <button class="send-btn" id="sendBtn">Send</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');

        function addMessage(content, type = 'assistant') {
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message \${type}-message\`;
            messageDiv.innerHTML = content;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message to chat
            addMessage(message, 'user');
            
            // Send to extension
            vscode.postMessage({
                type: 'sendMessage',
                text: message
            });

            messageInput.value = '';
            sendBtn.disabled = true;
        }

        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            
            switch (message.type) {
                case 'userMessage':
                    addMessage(message.message, 'user');
                    break;
                    
                case 'assistantMessage':
                    let content = message.message;
                    
                    if (message.codeSuggestions && message.codeSuggestions.length > 0) {
                        content += '<br><br><strong>Code suggestions:</strong>';
                        message.codeSuggestions.forEach((code, index) => {
                            content += \`
                                <div class="code-suggestion">
                                    <pre><code>\${code}</code></pre>
                                    <div class="code-actions">
                                        <button class="code-action-btn" onclick="insertCode('\${code}')">Insert at Cursor</button>
                                        <button class="code-action-btn" onclick="replaceCode('\${code}')">Replace Selection</button>
                                    </div>
                                </div>
                            \`;
                        });
                    }
                    
                    addMessage(content, 'assistant');
                    sendBtn.disabled = false;
                    break;
                    
                case 'error':
                    addMessage(message.message, 'error');
                    sendBtn.disabled = false;
                    break;
                    
                case 'typing':
                    if (message.isTyping) {
                        addMessage('<div class="typing-indicator">CopilotMini is typing<span class="typing-dots">...</span></div>', 'assistant');
                    } else {
                        // Remove typing indicator
                        const typingIndicator = chatContainer.querySelector('.typing-indicator');
                        if (typingIndicator) {
                            typingIndicator.parentElement.remove();
                        }
                    }
                    break;
            }
        });

        function insertCode(code) {
            vscode.postMessage({
                type: 'insertCode',
                code: code
            });
        }

        function replaceCode(code) {
            vscode.postMessage({
                type: 'replaceCode',
                code: code
            });
        }
    </script>
</body>
</html>
        `;
    }

    dispose(): void {
        if (this.chatPanel) {
            this.chatPanel.dispose();
        }
    }
} 