"""
WebSocket connection manager for handling multiple client connections.
"""

from typing import List, Dict, Any
import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store connection metadata
        self.connection_info[websocket] = {
            "connected_at": asyncio.get_event_loop().time(),
            "client_info": websocket.client,
            "messages_sent": 0,
            "messages_received": 0
        }
        
        logger.info(f"New WebSocket connection from {websocket.client}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        if websocket in self.connection_info:
            connection_time = asyncio.get_event_loop().time() - self.connection_info[websocket]["connected_at"]
            logger.info(f"WebSocket disconnected after {connection_time:.2f}s. Messages: "
                       f"sent={self.connection_info[websocket]['messages_sent']}, "
                       f"received={self.connection_info[websocket]['messages_received']}")
            del self.connection_info[websocket]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
            if websocket in self.connection_info:
                self.connection_info[websocket]["messages_sent"] += 1
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def send_personal_json(self, data: Dict[str, Any], websocket: WebSocket):
        """Send JSON data to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(data))
            if websocket in self.connection_info:
                self.connection_info[websocket]["messages_sent"] += 1
        except Exception as e:
            logger.error(f"Error sending JSON to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all active connections."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                if connection in self.connection_info:
                    self.connection_info[connection]["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all active connections."""
        await self.broadcast(json.dumps(data))
    
    async def disconnect_all(self):
        """Disconnect all active connections."""
        for connection in self.active_connections.copy():
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            finally:
                self.disconnect(connection)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections."""
        total_messages_sent = sum(info["messages_sent"] for info in self.connection_info.values())
        total_messages_received = sum(info["messages_received"] for info in self.connection_info.values())
        
        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "connections": [
                {
                    "client": str(ws.client),
                    "connected_at": info["connected_at"],
                    "messages_sent": info["messages_sent"],
                    "messages_received": info["messages_received"]
                }
                for ws, info in self.connection_info.items()
            ]
        }
    
    def track_received_message(self, websocket: WebSocket):
        """Track that a message was received from a WebSocket."""
        if websocket in self.connection_info:
            self.connection_info[websocket]["messages_received"] += 1 