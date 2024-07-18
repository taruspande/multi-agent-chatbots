from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from chainlit.utils import mount_chainlit
import os
import asyncio

app = FastAPI()

@app.get("/app")
def read_main():
    return {"message": "Hello World from main app"}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the custom HTML file
@app.get("/")
async def get_custom_ui():
    print("Serving custom UI")
    return FileResponse(os.path.join(os.getcwd(), "index.html"))

# WebSocket endpoint for active agent updates
connected_websockets = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received data: {data}")  # Debug log
            for ws in connected_websockets:
                await ws.send_text(f"Active agent: {data}")
    except WebSocketDisconnect:
        print("WebSocket connection closed")
        connected_websockets.remove(websocket)
    except Exception as e:
        print(f"WebSocket connection error: {e}")

async def broadcast_active_agent(agent_name: str):
    print(f"Broadcasting active agent: {agent_name}")  # Debug log
    for websocket in connected_websockets:
        await websocket.send_text(f"{agent_name}")

async def notify_active_agent(agent_name: str):
    await broadcast_active_agent(agent_name)

# Mount the Chainlit app
mount_chainlit(app=app, target="app.py", path="/chainlit")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
