import * as vscode from 'vscode';

export class CopilotMiniStatusBar implements vscode.Disposable {
    private statusBarItem: vscode.StatusBarItem;
    private isEnabled = true;
    private isConnected = false;

    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        
        this.statusBarItem.command = 'copilotMini.toggleStatus';
        this.updateStatusBar();
    }

    setEnabled(enabled: boolean): void {
        this.isEnabled = enabled;
        this.updateStatusBar();
    }

    setConnected(connected: boolean): void {
        this.isConnected = connected;
        this.updateStatusBar();
    }

    show(): void {
        this.statusBarItem.show();
    }

    hide(): void {
        this.statusBarItem.hide();
    }

    private updateStatusBar(): void {
        if (!this.isEnabled) {
            this.statusBarItem.text = "$(circle-slash) CopilotMini";
            this.statusBarItem.tooltip = "CopilotMini is disabled. Click to enable.";
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        } else if (!this.isConnected) {
            this.statusBarItem.text = "$(circle-outline) CopilotMini";
            this.statusBarItem.tooltip = "CopilotMini is connecting...";
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        } else {
            this.statusBarItem.text = "$(check-all) CopilotMini";
            this.statusBarItem.tooltip = "CopilotMini is active and connected";
            this.statusBarItem.backgroundColor = undefined;
        }
    }

    dispose(): void {
        this.statusBarItem.dispose();
    }
} 