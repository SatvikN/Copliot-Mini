import * as vscode from 'vscode';

export interface CompletionRequest {
    type: 'completion_request';
    request_id: string;
    code: string;
    language: string;
    cursor_position?: number;
    max_suggestions?: number;
}

export interface CompletionResponse {
    type: 'completion_response';
    request_id: string;
    suggestions: string[];
    confidence_scores: number[];
    model_used: string;
}

export interface ChatRequest {
    type: 'chat_request';
    request_id: string;
    message: string;
    code_context?: string;
    language?: string;
}

export interface ChatResponse {
    type: 'chat_response';
    request_id: string;
    response: string;
    code_suggestions?: string[];
}

export interface ErrorResponse {
    type: 'error';
    request_id?: string;
    message: string;
}

export type WebSocketMessage = CompletionResponse | ChatResponse | ErrorResponse | { type: 'pong'; timestamp: string };

export class CopilotMiniWebSocketClient implements vscode.Disposable {
    private ws: WebSocket | null = null;
    private isConnecting = false;
    private pendingRequests = new Map<string, (response: any) => void>();
    private reconnectTimeout: NodeJS.Timeout | null = null;
    private heartbeatInterval: NodeJS.Timeout | null = null;
    private readonly maxReconnectAttempts = 5;
    private reconnectAttempts = 0;

    constructor() {
        // Start connection
        this.connect();
    }

    async connect(): Promise<void> {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return;
        }

        if (this.isConnecting) {
            return;
        }

        this.isConnecting = true;

        try {
            const config = vscode.workspace.getConfiguration('copilotMini');
            const serverUrl = config.get<string>('serverUrl', 'ws://localhost:8000/ws');

            console.log(`Connecting to CopilotMini server: ${serverUrl}`);

            this.ws = new WebSocket(serverUrl);

            this.ws.onopen = () => {
                console.log('Connected to CopilotMini server');
                this.isConnecting = false;
                this.reconnectAttempts = 0;
                this.startHeartbeat();
            };

            this.ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket connection closed:', event.code, event.reason);
                this.isConnecting = false;
                this.stopHeartbeat();
                
                if (event.code !== 1000) { // Not a normal closure
                    this.scheduleReconnect();
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnecting = false;
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.isConnecting = false;
            this.scheduleReconnect();
        }
    }

    disconnect(): void {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        this.stopHeartbeat();

        if (this.ws) {
            this.ws.close(1000, 'Extension deactivated');
            this.ws = null;
        }

        // Reject all pending requests
        this.pendingRequests.forEach(reject => {
            reject(new Error('WebSocket disconnected'));
        });
        this.pendingRequests.clear();
    }

    private scheduleReconnect(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            return;
        }

        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000); // Exponential backoff, max 30s
        this.reconnectAttempts++;

        console.log(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);

        this.reconnectTimeout = setTimeout(() => {
            this.connect();
        }, delay);
    }

    private startHeartbeat(): void {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
            }
        }, 30000); // Ping every 30 seconds
    }

    private stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    private handleMessage(message: WebSocketMessage): void {
        if (message.type === 'pong') {
            // Heartbeat response, nothing to do
            return;
        }

        const requestId = 'request_id' in message ? message.request_id : undefined;
        
        if (requestId && this.pendingRequests.has(requestId)) {
            const resolve = this.pendingRequests.get(requestId)!;
            this.pendingRequests.delete(requestId);
            resolve(message);
        } else {
            console.warn('Received message for unknown request:', message);
        }
    }

    async sendCompletionRequest(
        code: string,
        language: string,
        cursorPosition?: number,
        maxSuggestions?: number
    ): Promise<CompletionResponse> {
        return new Promise((resolve, reject) => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket not connected'));
                return;
            }

            const requestId = this.generateRequestId();
            const request: CompletionRequest = {
                type: 'completion_request',
                request_id: requestId,
                code,
                language,
                cursor_position: cursorPosition,
                max_suggestions: maxSuggestions
            };

            this.pendingRequests.set(requestId, (response) => {
                if (response.type === 'error') {
                    reject(new Error(response.message));
                } else if (response.type === 'completion_response') {
                    resolve(response);
                } else {
                    reject(new Error('Unexpected response type'));
                }
            });

            // Set timeout for request
            setTimeout(() => {
                if (this.pendingRequests.has(requestId)) {
                    this.pendingRequests.delete(requestId);
                    reject(new Error('Request timeout'));
                }
            }, 10000); // 10 second timeout

            this.ws.send(JSON.stringify(request));
        });
    }

    async sendChatRequest(
        message: string,
        codeContext?: string,
        language?: string
    ): Promise<ChatResponse> {
        return new Promise((resolve, reject) => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket not connected'));
                return;
            }

            const requestId = this.generateRequestId();
            const request: ChatRequest = {
                type: 'chat_request',
                request_id: requestId,
                message,
                code_context: codeContext,
                language
            };

            this.pendingRequests.set(requestId, (response) => {
                if (response.type === 'error') {
                    reject(new Error(response.message));
                } else if (response.type === 'chat_response') {
                    resolve(response);
                } else {
                    reject(new Error('Unexpected response type'));
                }
            });

            // Set timeout for request
            setTimeout(() => {
                if (this.pendingRequests.has(requestId)) {
                    this.pendingRequests.delete(requestId);
                    reject(new Error('Request timeout'));
                }
            }, 15000); // 15 second timeout for chat

            this.ws.send(JSON.stringify(request));
        });
    }

    private generateRequestId(): string {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }

    dispose(): void {
        this.disconnect();
    }
} 