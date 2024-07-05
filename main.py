from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from chainlit.utils import mount_chainlit
import os

app = FastAPI()

@app.get("/app")
def read_main():
    return {"message": "Hello World from main app"}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure the root route is defined after mounting Chainlit to avoid overrides
@app.get("/")
async def get_custom_ui():
    print("Serving custom UI")
    return FileResponse(os.path.join(os.getcwd(), "index.html"))

# Mount the Chainlit app
mount_chainlit(app=app, target="app.py", path="/chainlit")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
