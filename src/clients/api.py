import json
from fastapi import FastAPI, HTTPException, Response, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import time
from pathlib import Path

# Import your clients
from abstract.server_config import ConfigContainer
from clients.ollama_client import OllamaMCPClient
from clients.shapE_client import ShapEClient, TextToModelRequest, ModelResponse, JobStatus, ImageToModelRequest

# Import the SDClient for image generation
from clients.sd_client import SDClient, TextToImageRequest, ImageToImageRequest

# Configure logging
import logging
import colorlog

logger = logging.getLogger("APIServer")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Global client instances
client_instance = None
client_lock = asyncio.Lock()
client_default_model = "qwen2.5:14b"

# Global Shap-E client instance
shape_client = None
shape_client_lock = asyncio.Lock()

# Global Stable Diffusion client instance
sd_client = None
sd_client_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = client_default_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize all clients
    global client_instance, shape_client, sd_client
    
    # Initialize OllamaMCP client
    config = ConfigContainer.form_file("examples/server.json")
    client_instance = await OllamaMCPClient.create(config)
    logger.info("Ollama MCP client initialized")

    # Initialize Shap-E client
    shape_client = await ShapEClient.create()
    logger.info("Shap-E client initialized")
    
    # Initialize Stable Diffusion client
    sd_client = await SDClient.create()
    logger.info("Stable Diffusion client initialized")

    yield

    # Shutdown: cleanup clients
    if client_instance:
        await client_instance.__aexit__(None, None, None)
    if shape_client:
        await shape_client.__aexit__(None, None, None)
    if sd_client:
        await sd_client.__aexit__(None, None, None)
    logger.info("All clients shut down")


# Create FastAPI app with lifespan handler
app = FastAPI(title="Multi-Modal Generation API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


async def get_shape_client():
    global shape_client, shape_client_lock

    if shape_client is not None:
        return shape_client

    # Use a lock to prevent multiple initializations
    async with shape_client_lock:
        if shape_client is None:
            try:
                shape_client = await ShapEClient.create()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize Shap-E client: {str(e)}")

    return shape_client


async def get_sd_client():
    global sd_client, sd_client_lock

    if sd_client is not None:
        return sd_client

    # Use a lock to prevent multiple initializations
    async with sd_client_lock:
        if sd_client is None:
            try:
                sd_client = await SDClient.create()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize Stable Diffusion client: {str(e)}")

    return sd_client


# OLLAMA MCP ENDPOINTS

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


# SHAPE-E ENDPOINTS (3D MODEL GENERATION)

@app.post("/api/shape", response_model=ModelResponse)
async def generate_3d_model(request: TextToModelRequest):
    """Convert text to 3D object shape using Shap-E"""
    client = await get_shape_client()

    if not client or not (client.xm and client.model and client.diffusion):
        raise HTTPException(status_code=503, detail="Shap-E models are not loaded yet. Please try again later.")

    job_id = await client.create_job(request)

    return ModelResponse(job_id=job_id, status="pending", message="3D model generation started. Use /api/shape/status/{job_id} to check progress.")


@app.post("/api/generate-from-image", response_model=ModelResponse)
async def generate_model_from_image(
    file: UploadFile = File(...),
    batch_size: int = Form(1),
    guidance_scale: float = Form(3.0),
    render_mode: str = Form("nerf"),
    render_size: int = Form(64),
    use_karras: bool = Form(True),
    karras_steps: int = Form(64),
):
    client = await get_shape_client()

    # Validate file is an image
    content_type = file.content_type
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Create request object
    request = ImageToModelRequest(
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        render_mode=render_mode,
        render_size=render_size,
        use_karras=use_karras,
        karras_steps=karras_steps,
    )

    # Read the uploaded file
    image_data = await file.read()

    # Create job
    job_id = await client.create_image_job(image_data, request)

    return ModelResponse(job_id=job_id, status="pending", message="Image-to-3D generation started. Use /api/shape/status/{job_id} to check progress.")


@app.get("/api/shape/status/{job_id}", response_model=JobStatus)
async def get_shape_job_status(job_id: str):
    """Get status of a Shap-E generation job"""
    client = await get_shape_client()
    job = client.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    return job


@app.get("/api/shape/jobs", response_model=List[JobStatus])
async def list_shape_jobs():
    """List all Shap-E generation jobs"""
    client = await get_shape_client()
    return client.list_jobs()


@app.get("/api/shape/result/gif/{job_id}")
async def get_shape_gif(job_id: str):
    """Get GIF result for a completed Shap-E job"""
    client = await get_shape_client()
    job = client.get_job_status(job_id)
    if not job or not job.gif_path:
        raise HTTPException(status_code=404, detail=f"GIF for job {job_id} not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet (status: {job.status})")

    if not os.path.isfile(job.gif_path):
        raise HTTPException(status_code=404, detail=f"GIF file not found at {job.gif_path}")

    filename = os.path.basename(job.gif_path)
    if not filename.endswith(".gif"):
        filename += ".gif"

    return FileResponse(job.gif_path, media_type="image/gif", filename=filename)


@app.get("/api/shape/result/obj/{job_id}")
async def get_shape_obj(job_id: str):
    """Get OBJ result for a completed Shap-E job"""
    client = await get_shape_client()
    job = client.get_job_status(job_id)
    if not job or not job.obj_path:
        raise HTTPException(status_code=404, detail=f"OBJ for job {job_id} not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet (status: {job.status})")

    if not os.path.isfile(job.obj_path):
        raise HTTPException(status_code=404, detail=f"OBJ file not found at {job.obj_path}")

    filename = os.path.basename(job.obj_path)
    if not filename.endswith(".obj"):
        filename += ".obj"

    return FileResponse(job.obj_path, media_type="model/obj", filename=filename)


# STABLE DIFFUSION ENDPOINTS (IMAGE GENERATION)

@app.post("/api/text-to-image", response_model=ModelResponse)
async def text_to_image(request: TextToImageRequest):
    """Generate image from text prompt using Stable Diffusion"""
    client = await get_sd_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Stable Diffusion service not ready")
    
    job_id = await client.create_text_job(request)
    return ModelResponse(job_id=job_id, status="pending", message="Image generation started. Use /api/sd/status/{job_id} to check progress.")


@app.post("/api/image-to-image", response_model=ModelResponse)
async def image_to_image(
    file: UploadFile = File(...),
    prompt: str = Form(str),
    num_inference_steps: int = Form(2),
    width: int = Form(128),
    height: int = Form(128),
    negative_prompt: Optional[str] = Form(None),
    model_id: Optional[str] = Form(None),
    strength: float = Form(0.5),
    guidance_scale: float = Form(0.0),
    image_url: Optional[str] = Form(None),):
    """Generate image from input image and prompt using Stable Diffusion"""
    client = await get_sd_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Stable Diffusion service not ready")
    
    # Validate file is an image
    image_data = await file.read()

    request = ImageToImageRequest(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        model_id=model_id,
        strength=strength,
        guidance_scale=guidance_scale,
    )

    job_id = await client.create_image_job(request,image_data)
    return ModelResponse(job_id=job_id, status="pending", message="Image generation started. Use /api/sd/status/{job_id} to check progress.")


@app.get("/api/sd/status/{job_id}", response_model=JobStatus)
async def get_sd_job_status(job_id: str):
    """Get status of a Stable Diffusion generation job"""
    client = await get_sd_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Stable Diffusion service not ready")
    
    job = client.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/sd/result/{job_id}")
async def get_sd_image(job_id: str):
    """Get image result for a completed Stable Diffusion job"""
    client = await get_sd_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Stable Diffusion service not ready")
    
    job = client.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job.status}")
    
    if job.image_path is None:
        raise HTTPException(status_code=400, detail="No image available for this job")
    
    if not Path(job.image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(job.image_path)


@app.get("/api/sd/jobs", response_model=List[JobStatus])
async def list_sd_jobs():
    """List all Stable Diffusion generation jobs"""
    client = await get_sd_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Stable Diffusion service not ready")
    
    return client.list_jobs()


# STATUS ENDPOINTS

@app.get("/api/status")
async def get_status():
    """Get status of all services"""
    status = {}
    
    try:
        client = await get_client()
        status["ollama"] = await client.get_status()
    except Exception as e:
        status["ollama"] = {"error": str(e)}

    try:
        shape_client = await get_shape_client()
        status["shape_e"] = await shape_client.get_status()
    except Exception as e:
        status["shape_e"] = {"error": str(e)}
        
    try:
        sd_client = await get_sd_client()
        status["stable_diffusion"] = await sd_client.get_status()
    except Exception as e:
        status["stable_diffusion"] = {"error": str(e)}

    return Response(json.dumps(status), media_type="application/json")