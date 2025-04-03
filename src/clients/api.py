import json
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

# Import your OllamaMCPClient from the original file
from abstract.config_container import ConfigContainer
from clients.ollama_client import OllamaMCPClient

# Global client instance
client_instance = None
client_lock = asyncio.Lock()


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


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "qwen2.5:14b"


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


@app.delete("/api/chat")
async def delete_chat():
    client = await get_client()
    await client.prepare_prompt()


@app.get("/api/tools")
async def get_tools():
    client = await get_client()
    tools = client.get_tools()

    return Response(json.dumps([tool.model_dump() for tool in tools]), media_type="text/json")


@app.get("/api/servers")
async def get_server():
    client = await get_client()
    return Response(json.dumps(list(client.selected_server.keys())))


@app.put("/api/servers")
async def select_server(request: list[str]):
    client = await get_client()
    client.select_server(request)


@app.get("/api/models")
async def get_models():
    client = await get_client()
    models = await client.client.list()
    return Response(json.dumps([m.model for m in models.models]), media_type="text/json")
