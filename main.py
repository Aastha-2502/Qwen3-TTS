import logging
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from qwen_tts import Qwen3TTSModel
from services.voice_clone import router as voice_clone_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    tts = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    app.state.tts = tts

    yield
    del app.state.tts


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def check_function():
    return {"message": "hello from the root!"} 

app.include_router(voice_clone_router) 