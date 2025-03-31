import json
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Import your OllamaMCPClient from the original file
from abstract.server_config import ConfigContainer
from clients.ollama_client import OllamaMCPClient

# Global client instance
client_instance = None
client_lock = asyncio.Lock()
client_default_model = "qwen2.5:14b"


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = client_default_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize the client
    global client_instance
    # You'll need to initialize your config here
    config = ConfigContainer.form_file("examples/server.json")
    client_instance = await OllamaMCPClient.create(config)

    yield

    # Shutdown: cleanup the client
    if client_instance:
        await client_instance.__aexit__(None, None, None)


# Create FastAPI app with lifespan handler
app = FastAPI(title="Ollama MCP API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify the allowed origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


async def get_client():
    global client_instance, client_lock

    if client_instance is not None:
        return client_instance

    # Use a lock to prevent multiple initializations
    async with client_lock:
        if client_instance is None:
            try:
                config = ConfigContainer.form_file("examples/server.json")
                client_instance = await OllamaMCPClient.create(config)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize client: {str(e)}")

    return client_instance


@app.post("/api/chat")
async def stream_chat(request: ChatRequest):
    client = await get_client()

    iter = client.process_message(request.message, request.model)
    first_chunk = None

    async def response_generator():
        if first_chunk:
            yield json.dumps(first_chunk)
        async for part in iter:
            yield json.dumps(part)
            await asyncio.sleep(0.01)

    try:
        first_chunk = await iter.__anext__()
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")


@app.get("/api/tools")
async def get_tools():
    client = await get_client()
    tools = await client.list_tools()

    return Response(json.dumps([tool.model_dump() for tool in tools]), media_type="text/json")


@app.get("/api/models")
async def get_models():
    """Get available models from Ollama server"""
    client = await get_client()
    models = await client.get_available_models()
    return Response(
        json.dumps({"models": models, "default": client_default_model}),
        media_type="text/json",
    )


@app.get("/api/status")
async def get_status():
    """Get status of the server"""
    client = await get_client()
    status = await client.get_status()
    return Response(json.dumps(status), media_type="text/json")
