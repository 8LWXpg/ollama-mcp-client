from contextlib import AbstractAsyncContextManager, AsyncExitStack
import os
import json
import logging
import asyncio
import colorlog
from typing import AsyncIterator, Self, Optional, Dict, Any, List
import torch
import imageio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import time
import io
from PIL import Image

# Import Shap-E modules
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh

# Configure logging
logger = logging.getLogger("ShapEClient")
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

# Initialize FastAPI app
app = FastAPI(title="Shap-E Text-to-3D API", description="Generate 3D models from text prompts using Shap-E")


class TextToModelRequest(BaseModel):
    prompt: str
    batch_size: Optional[int] = 1
    guidance_scale: Optional[float] = 15.0
    render_mode: Optional[str] = "nerf"  # 'stf' or 'nerf'
    render_size: Optional[int] = 64
    use_karras: Optional[bool] = True
    karras_steps: Optional[int] = 64


class ImageToModelRequest(BaseModel):
    batch_size: Optional[int] = 1
    guidance_scale: Optional[float] = 3.0
    render_mode: Optional[str] = "nerf"  # 'stf' or 'nerf'
    render_size: Optional[int] = 64
    use_karras: Optional[bool] = True
    karras_steps: Optional[int] = 64


class ModelResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: float
    completed_at: Optional[float] = None
    progress: float = 0.0
    gif_path: Optional[str] = None
    obj_path: Optional[str] = None
    error: Optional[str] = None


class ShapEClient(AbstractAsyncContextManager):
    def __init__(self, output_dir: str = "./shap-e"):
        self.logger = logger
        self.device = None
        self.xm = None
        self.model = None
        self.diffusion = None
        self.output_dir = output_dir
        self.gif_dir = os.path.join(output_dir, "gif")
        self.obj_dir = os.path.join(output_dir, "obj")
        self.exit_stack = AsyncExitStack()
        self.active_jobs: Dict[str, JobStatus] = {}

        # Create output directories
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.obj_dir, exist_ok=True)

    async def __aenter__(self):
        await self.load_models()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            await self.exit_stack.aclose()
        except ValueError:
            pass

    @classmethod
    async def create(cls, output_dir: str = "./shap-e") -> Self:
        """Factory method to create and initialize a client instance"""
        client = cls(output_dir)
        await client.load_models()
        return client

    async def load_models(self):
        """Load required models for Shap-E"""
        self.logger.info("Loading Shap-E models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        try:
            # Load models asynchronously
            await asyncio.to_thread(self._load_models_sync)
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def _load_models_sync(self):
        """Synchronous model loading (will be called in a thread)"""
        self.xm = load_model("transmitter", device=self.device)
        self.model = load_model("text300M", device=self.device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    async def create_job(self, request: TextToModelRequest) -> str:
        """Create a new generation job"""
        import time
        import uuid

        # Create unique job ID
        job_id = str(uuid.uuid4())
        timestamp = time.time()

        # Create base filename from prompt
        prompt_slug = "".join(c for c in request.prompt[:20] if c.isalnum() or c.isspace()).strip().replace(" ", "_")
        base_filename = f"{int(timestamp)}_{prompt_slug}"

        gif_path = os.path.join(self.gif_dir, f"{base_filename}.gif")
        obj_path = os.path.join(self.obj_dir, f"{base_filename}.obj")

        # Store job information
        self.active_jobs[job_id] = JobStatus(job_id=job_id, status="pending", created_at=timestamp, gif_path=gif_path, obj_path=obj_path)

        # Start job processing
        asyncio.create_task(self.process_job(job_id, request, gif_path, obj_path))

        return job_id

    # Add the new methods for image processing
    async def create_image_job(self, image_data: bytes, request: ImageToModelRequest) -> str:
        """Create a new generation job from an image"""
        import time
        import uuid

        # Create unique job ID
        job_id = str(uuid.uuid4())
        timestamp = time.time()

        # Create base filename
        base_filename = f"{int(timestamp)}_image_upload"

        gif_path = os.path.join(self.gif_dir, f"{base_filename}.gif")
        obj_path = os.path.join(self.obj_dir, f"{base_filename}.obj")

        # Store job information
        self.active_jobs[job_id] = JobStatus(job_id=job_id, status="pending", created_at=timestamp, gif_path=gif_path, obj_path=obj_path)

        # Start job processing
        asyncio.create_task(self.process_image_job(job_id, image_data, request, gif_path, obj_path))

        return job_id

    async def process_image_job(self, job_id: str, image_data: bytes, request: ImageToModelRequest, gif_path: str, obj_path: str):
        """Process image-to-3D generation job asynchronously"""
        if job_id not in self.active_jobs:
            self.logger.error(f"Job {job_id} not found")
            return

        job = self.active_jobs[job_id]
        job.status = "processing"

        try:
            # Load and process the image
            self.logger.info(f"Job {job_id}: Loading image data")
            job.progress = 0.05

            # Convert bytes to PIL Image
            image = await asyncio.to_thread(self._load_image_from_bytes, image_data)

            # Sample latents from the model
            self.logger.info(f"Job {job_id}: Sampling latents for image")
            job.progress = 0.1

            latents = await asyncio.to_thread(
                self._sample_image_latents, image, request.batch_size, request.guidance_scale, request.use_karras, request.karras_steps
            )

            job.progress = 0.5
            self.logger.info(f"Job {job_id}: Latents generated, rendering views")

            # Create camera views
            cameras = await asyncio.to_thread(create_pan_cameras, request.render_size, self.device)

            job.progress = 0.6

            # Process latents
            for i, latent in enumerate(latents):
                # Generate images from different camera angles
                images = await asyncio.to_thread(decode_latent_images, self.xm, latent, cameras, request.render_mode)

                job.progress = 0.8
                self.logger.info(f"Job {job_id}: Images rendered, saving outputs")

                # Save as GIF
                await asyncio.to_thread(imageio.mimsave, gif_path, images, duration=0.1)

                # Save as OBJ
                await asyncio.to_thread(self._save_obj, latent, obj_path)

            job.status = "completed"
            job.progress = 1.0
            job.completed_at = time.time()
            self.logger.info(f"Job {job_id} completed. GIF: {gif_path}, OBJ: {obj_path}")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"Error processing job {job_id}: {e}")

    def _load_image_from_bytes(self, image_bytes):
        """Load image from bytes and prepare it for the model"""
        image = Image.open(io.BytesIO(image_bytes))
        # Convert the image to the format expected by the model
        # You may need to resize, crop, or convert the image based on the model's requirements
        return image

    def _sample_image_latents(self, image, batch_size, guidance_scale, use_karras, karras_steps):
        """Sample latents from image (synchronous operation for thread)"""
        return sample_latents(
            batch_size=batch_size,
            model=self.model,  # Use image model here
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(images=[image] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=use_karras,
            karras_steps=karras_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

    async def process_job(self, job_id: str, request: TextToModelRequest, gif_path: str, obj_path: str):
        """Process generation job asynchronously"""
        if job_id not in self.active_jobs:
            self.logger.error(f"Job {job_id} not found")
            return

        job = self.active_jobs[job_id]
        job.status = "processing"

        try:
            # Sample latents from the model
            self.logger.info(f"Job {job_id}: Sampling latents for prompt: {request.prompt}")
            job.progress = 0.1

            latents = await asyncio.to_thread(
                self._sample_latents, request.prompt, request.batch_size, request.guidance_scale, request.use_karras, request.karras_steps
            )

            job.progress = 0.5
            self.logger.info(f"Job {job_id}: Latents generated, rendering views")

            # Create camera views
            cameras = await asyncio.to_thread(create_pan_cameras, request.render_size, self.device)

            job.progress = 0.6

            # Process latents
            for i, latent in enumerate(latents):
                # Generate images from different camera angles
                images = await asyncio.to_thread(decode_latent_images, self.xm, latent, cameras, request.render_mode)

                job.progress = 0.8
                self.logger.info(f"Job {job_id}: Images rendered, saving outputs")

                # Save as GIF
                await asyncio.to_thread(imageio.mimsave, gif_path, images, duration=0.1)

                # Save as OBJ
                await asyncio.to_thread(self._save_obj, latent, obj_path)

            job.status = "completed"
            job.progress = 1.0
            job.completed_at = time.time()
            self.logger.info(f"Job {job_id} completed. GIF: {gif_path}, OBJ: {obj_path}")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"Error processing job {job_id}: {e}")

    def _sample_latents(self, prompt, batch_size, guidance_scale, use_karras, karras_steps):
        """Sample latents (synchronous operation for thread)"""
        return sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=use_karras,
            karras_steps=karras_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

    def _save_obj(self, latent, obj_path):
        """Save OBJ file (synchronous operation for thread)"""
        t = decode_latent_mesh(self.xm, latent).tri_mesh()
        with open(obj_path, "w") as f:
            t.write_obj(f)

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a specific job"""
        return self.active_jobs.get(job_id)

    def list_jobs(self) -> List[JobStatus]:
        """List all jobs"""
        return list(self.active_jobs.values())

    async def get_status(self) -> Dict[str, Any]:
        """Get client status information"""
        return {
            "device": str(self.device),
            "models_loaded": self.model is not None and self.xm is not None and self.diffusion is not None,
            "active_jobs": len(self.active_jobs),
            "output_dirs": {"gif": self.gif_dir, "obj": self.obj_dir},
        }
