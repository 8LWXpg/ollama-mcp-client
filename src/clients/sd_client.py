from contextlib import AbstractAsyncContextManager, AsyncExitStack
import os
import logging
import asyncio
import colorlog
from typing import Self, Optional, Dict, Any, List
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import time
import io
from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np



# Import Diffusers modules
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image

# Configure logging
logger = logging.getLogger("SDClient")
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
app = FastAPI(title="Stable Diffusion API", description="Generate images from text prompts or images using Stable Diffusion")


class TextToImageRequest(BaseModel):
    prompt: str
    model_id: Optional[str] = "stabilityai/sd-turbo"
    num_inference_steps: Optional[int] = 1
    guidance_scale: Optional[float] = 2.0
    width: Optional[int] = 512
    height: Optional[int] = 512
    negative_prompt: Optional[str] = None


class ImageToImageRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    model_id: Optional[str] = "stabilityai/sd-turbo"
    num_inference_steps: Optional[int] = 2
    strength: Optional[float] = 0.5
    guidance_scale: Optional[float] = 0.0
    width: Optional[int] = 512
    height: Optional[int] = 512
    negative_prompt: Optional[str] = None


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
    image_path: Optional[str] = None
    error: Optional[str] = None


class SDClient(AbstractAsyncContextManager):
    def __init__(self, output_dir: str = "./sd-output"):
        self.logger = logger
        self.device = None
        self.text2img_pipe = None
        self.img2img_pipe = None
        self.output_dir = output_dir
        self.exit_stack = AsyncExitStack()
        self.active_jobs: Dict[str, JobStatus] = {}

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    async def __aenter__(self):
        await self.load_models()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            await self.exit_stack.aclose()
        except ValueError:
            pass

    @classmethod
    async def create(cls, output_dir: str = "./sd-output") -> Self:
        """Factory method to create and initialize a client instance"""
        client = cls(output_dir)
        await client.load_models()
        return client

    async def load_models(self):
        """Load required models for Stable Diffusion"""
        self.logger.info("Loading Stable Diffusion models...")
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
        # Load the default models
        model_id = "stabilityai/sd-turbo"
        self.text2img_pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)
        
        # Reuse the same pipeline for image to image
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)

    async def create_text_job(self, request: TextToImageRequest) -> str:
        """Create a new text-to-image generation job"""
        import time
        import uuid

        # Create unique job ID
        job_id = str(uuid.uuid4())
        timestamp = time.time()

        # Create base filename from prompt
        prompt_slug = "".join(c for c in request.prompt[:20] if c.isalnum() or c.isspace()).strip().replace(" ", "_")
        base_filename = f"{int(timestamp)}_{prompt_slug}.png"
        image_path = os.path.join(self.output_dir, base_filename)

        # Store job information
        self.active_jobs[job_id] = JobStatus(job_id=job_id, status="pending", created_at=timestamp, image_path=image_path)

        # Start job processing
        asyncio.create_task(self.process_text_job(job_id, request, image_path))

        return job_id

    async def create_image_job(self, request: ImageToImageRequest, image_data: Optional[bytes] = None) -> str:
        """Create a new image-to-image generation job"""
        import time
        import uuid

        # Create unique job ID
        job_id = str(uuid.uuid4())
        timestamp = time.time()

        # Create base filename from prompt
        prompt_slug = "".join(c for c in request.prompt[:20] if c.isalnum() or c.isspace()).strip().replace(" ", "_")
        base_filename = f"{int(timestamp)}_img2img_{prompt_slug}.png"
        image_path = os.path.join(self.output_dir, base_filename)

        # Store job information
        self.active_jobs[job_id] = JobStatus(job_id=job_id, status="pending", created_at=timestamp, image_path=image_path)

        # Start job processing
        asyncio.create_task(self.process_image_job(job_id, request, image_path, image_data))

        return job_id

    async def process_text_job(self, job_id: str, request: TextToImageRequest, image_path: str):
        """Process text-to-image generation job asynchronously"""
        if job_id not in self.active_jobs:
            self.logger.error(f"Job {job_id} not found")
            return

        job = self.active_jobs[job_id]
        job.status = "processing"

        try:
            # Generate image from prompt
            self.logger.info(f"Job {job_id}: Generating image for prompt: {request.prompt}")
            job.progress = 0.1

            # Check if we need to load a different model
            if request.model_id != "stabilityai/sd-turbo" and self.text2img_pipe.config.model_type != request.model_id:
                self.logger.info(f"Loading model: {request.model_id}")
                await asyncio.to_thread(self._load_specific_model, request.model_id, "text2img")
                job.progress = 0.3
            
            # Generate the image
            image = await asyncio.to_thread(
                self._generate_text_to_image,
                request.prompt,
                request.num_inference_steps,
                request.guidance_scale,
                request.width,
                request.height,
                request.negative_prompt
            )

            job.progress = 0.8
            self.logger.info(f"Job {job_id}: Image generated, saving output")

            # Save the image
            await asyncio.to_thread(image.save, image_path)

            job.status = "completed"
            job.progress = 1.0
            job.completed_at = time.time()
            self.logger.info(f"Job {job_id} completed. Image saved at: {image_path}")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"Error processing job {job_id}: {e}")

    async def process_image_job(self, job_id: str, request: ImageToImageRequest, image_path: str, image_data: Optional[bytes] = None):
        """Process image-to-image generation job asynchronously"""
        if job_id not in self.active_jobs:
            self.logger.error(f"Job {job_id} not found")
            return

        job = self.active_jobs[job_id]
        job.status = "processing"

        try:
            # Load the input image
            self.logger.info(f"Job {job_id}: Loading input image")
            job.progress = 0.1

            if image_data:
                pil_img = self.bytes_to_pil(image_data)
                # np_img = self.pil_to_numpy(pil_img)
                # tensor_img = self.numpy_to_tensor(np_img)
                # 確認 init_image 是有效圖像
                init_image = load_image(pil_img).resize((request.width, request.height))
            else:
                raise ValueError("No input image provided")
            
            

            job.progress = 0.2

            # Check if we need to load a different model
            if request.model_id != "stabilityai/sd-turbo" and (self.img2img_pipe is None or self.img2img_pipe.config.model_type != request.model_id):
                self.logger.info(f"Loading model: {request.model_id}")
                await asyncio.to_thread(self._load_specific_model, request.model_id, "img2img")
                job.progress = 0.4
            
            # Generate the image
            self.logger.info(f"Job {job_id}: Generating image from input image with prompt: {request.prompt}")
            image = await asyncio.to_thread(
                self._generate_image_to_image,
                request.prompt,
                init_image,
                request.num_inference_steps,
                request.strength,
                request.guidance_scale,
                request.negative_prompt
            )

            job.progress = 0.8
            self.logger.info(f"Job {job_id}: Image generated, saving output")

            # Save the image
            await asyncio.to_thread(image.save, image_path)

            job.status = "completed"
            job.progress = 1.0
            job.completed_at = time.time()
            self.logger.info(f"Job {job_id} completed. Image saved at: {image_path}")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"Error processing job {job_id}: {e}")

    def _load_specific_model(self, model_id: str, pipeline_type: str):
        """Load a specific model for a pipeline type"""
        if pipeline_type == "text2img":
            self.text2img_pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=torch.float32, variant="fp16"
            ).to(self.device)
        elif pipeline_type == "img2img":
            self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id, torch_dtype=torch.float32, variant="fp16"
            ).to(self.device)

    def bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes))

    def pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        return np.array(image)
    
    def numpy_to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        # 如果需要 CHW 格式 (C, H, W)，轉置順序
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        return image_tensor

    def _generate_text_to_image(self, prompt, num_inference_steps, guidance_scale, width, height, negative_prompt):
        """Generate image from text prompt (synchronous operation for thread)"""
        if self.text2img_pipe is None:
            raise ValueError("Image-to-image pipeline is not loaded")
        return self.text2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]

    def _generate_image_to_image(self, prompt, init_image, num_inference_steps, strength, guidance_scale, negative_prompt):
        """Generate image from input image and text prompt (synchronous operation for thread)"""
        if self.img2img_pipe is None:
            raise ValueError("Image-to-image pipeline is not loaded")
        return self.img2img_pipe(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale
        ).images[0]

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
            "models_loaded": self.text2img_pipe is not None and self.img2img_pipe is not None,
            "active_jobs": len(self.active_jobs),
            "output_dir": self.output_dir,
        }