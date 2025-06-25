# app_xl.py  –  run with:
#   uvicorn app_xl:app --host 0.0.0.0 --port 7866 --workers 1

import os
from uuid import uuid4

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline

# ───────────────────────── CONFIG ──────────────────────────
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
API_KEY  = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"

BASE_DIR   = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_xl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────── LOAD MODEL ───────────────────────
print("🔄 Loading SDXL …")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_model_cpu_offload()          # keeps VRAM low
print("✅ SDXL ready!")

# ───────────────────────── FASTAPI ─────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.wildmindai.com",
        "https://api.wildmindai.com",
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key", "Accept"],
)

# serve at  https://api.wildmindai.com/xl/images/<file>.png
app.mount("/xl/images", StaticFiles(directory=OUTPUT_DIR), name="xl-images")

class PromptRequest(BaseModel):
    prompt: str
    steps: int = 60
    guidance: float = 5.5
    height: int = 1024      # SDXL defaults
    width:  int = 1024

@app.post("/xl")
async def generate_xl(request: Request, body: PromptRequest):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    img = pipe(
        prompt            = prompt,
        height            = body.height,
        width             = body.width,
        num_inference_steps = body.steps,
        guidance_scale    = body.guidance
    ).images[0]

    fname = f"{uuid4().hex}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    img.save(fpath)
    print("🖼️  saved", fpath)

    return JSONResponse(
        {"image_url": f"https://api.wildmindai.com/xl/images/{fname}"}
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": "SDXL-base-1.0"}
