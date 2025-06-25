from diffusers import DiffusionPipeline
import torch

# Load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define steps and split (80/20)
n_steps = 60
high_noise_frac = 0.8

prompt = "A chicken in a business suit giving a PowerPoint presentation on worm economics"

# Generate image
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

# Save image locally
image_path = "generated_image.png"
image.save(image_path)
print(f"Image saved at {image_path}")















from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
import os

# === CONFIG ===
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"

# Absolute path for consistent behavior
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL (Stable Medium with NF4 Quantization) ===
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

# === FASTAPI SETUP ===
app = FastAPI()

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static images at /medium/images
app.mount("/xl/images", StaticFiles(directory=OUTPUT_DIR), name="xl-images")

# === Request Schema ===
class PromptRequest(BaseModel):
    prompt: str

# === /xl endpoint ===
@app.post("/xl")
async def generate_xl(request: Request, body: PromptRequest):
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    image = pipeline(
        prompt=prompt,
        num_inference_steps=60,
        guidance_scale=5.5,
    ).images[0]

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    print("✅ Saved image:", filepath)

    # ✅ Return image URL that matches mounted path
    return {"image_url": f"https://api.wildmindai.com/xl/images/{filename}"}
