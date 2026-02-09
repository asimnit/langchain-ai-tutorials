"""
FastAPI Backend for Customer Support Chat
WebSocket endpoint for real-time streaming
"""

import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent import stream_agent_response, clear_conversation

app = FastAPI(title="Customer Support API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store active WebSocket connections (history is now managed by LangGraph MemorySaver)
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]


manager = ConnectionManager()


@app.get("/")
async def root():
    return {"message": "Customer Support API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for chat.
    
    Client sends: {"type": "message", "content": "user message"}
    Server sends: 
        {"type": "thinking", "content": "..."}
        {"type": "tool_call", "name": "...", "args": {...}}
        {"type": "tool_result", "name": "...", "result": "..."}
        {"type": "response", "content": "...", "done": false}
        {"type": "done", "messages": [...]}
        {"type": "error", "content": "..."}
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "message":
                user_content = message.get("content", "")
                
                if not user_content.strip():
                    await websocket.send_json({"type": "error", "content": "Empty message"})
                    continue
                
                # Stream agent response - history is managed by MemorySaver via thread_id
                full_response = ""
                async for event in stream_agent_response(user_content, client_id):
                    print(f"[WS] Sending event: {event['type']} - {str(event)[:100]}")
                    await websocket.send_json(event)
                    
                    # Collect full response for history
                    if event["type"] == "response":
                        full_response += event.get("content", "")
                
                print(f"[WS] Done streaming. Full response length: {len(full_response)}")
            
            elif message.get("type") == "clear":
                # Clear conversation history via MemorySaver
                clear_conversation(client_id)
                await websocket.send_json({"type": "cleared"})
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        await websocket.send_json({"type": "error", "content": str(e)})
        manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
