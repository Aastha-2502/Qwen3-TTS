import io
import logging
import numpy as np
import torch
import soundfile as sf
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import Response
from typing import Optional

# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__, )

router = APIRouter()

SUPPORTED_LANGUAGES = {"Auto", "English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish"}


# ── Helper ────────────────────────────────────────────────────────────────────
def audio_array_to_bytes(wav_array: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav_array, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/clone-single", response_class=Response)
async def clone_single(
    request: Request,
    ref_audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(default=None),
    target_text: str = Form(...),
    language: str = Form(default="Auto"),
    x_vector_only: bool = Form(default=False),
):
    logger.info(
        "clone-single request | filename=%s | language=%s | x_vector_only=%s | target_text_len=%d",
        ref_audio.filename, language, x_vector_only, len(target_text),
    )

    # ── Input validation ──────────────────────────────────────────────────────
    if not target_text.strip():
        logger.warning("Rejected request: target_text is empty")
        raise HTTPException(status_code=422, detail="target_text cannot be empty.")

    if language not in SUPPORTED_LANGUAGES:
        logger.warning("Rejected request: unsupported language=%s", language)
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{language}'. Choose from: {sorted(SUPPORTED_LANGUAGES)}",
        )

    if ref_audio.content_type not in ("audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/flac", "application/octet-stream"):
        logger.warning("Unexpected content_type=%s — proceeding anyway", ref_audio.content_type)

    # ── Read + decode reference audio ─────────────────────────────────────────
    try:
        ref_audio_bytes = await ref_audio.read()
        if len(ref_audio_bytes) == 0:
            raise ValueError("Uploaded ref_audio file is empty.")

        ref_buffer = io.BytesIO(ref_audio_bytes)
        audio_array, sample_rate = sf.read(ref_buffer)
        logger.info("Ref audio loaded | sr=%d | shape=%s | dtype=%s", sample_rate, audio_array.shape, audio_array.dtype)

        # Pre-convert stereo → mono to avoid Qwen3-TTS internal bug
        if audio_array.ndim > 1:
            logger.info("Stereo audio detected — converting to mono")
            audio_array = np.mean(audio_array, axis=-1).astype(np.float32)

    except Exception as e:
        logger.exception("Failed to read reference audio: %s", e)
        raise HTTPException(status_code=400, detail=f"Could not decode reference audio: {str(e)}")

    # ── Model inference ───────────────────────────────────────────────────────
    try:
        tts = request.app.state.tts

        gen_kwargs = dict(
            max_new_tokens=2048, do_sample=True,
            top_k=50, top_p=1.0, temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True, subtalker_top_k=50,
            subtalker_top_p=1.0, subtalker_temperature=0.9,
        )

        logger.info("Starting voice clone inference...")
        wavs, sr = tts.generate_voice_clone(
            text=target_text,
            language=language,
            ref_audio=(audio_array, sample_rate),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            voice_clone_prompt=None,
            non_streaming_mode=True,
            **gen_kwargs,
        )
        logger.info("Inference complete | output_sr=%d | num_wavs=%d", sr, len(wavs))

    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA OOM during inference: %s", e)
        raise HTTPException(status_code=503, detail="GPU out of memory. Try a shorter target_text.")

    except Exception as e:
        logger.exception("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

    # ── Encode output audio → bytes ───────────────────────────────────────────
    try:
        audio_bytes = audio_array_to_bytes(wavs[0], sr)
        logger.info("Audio encoded to bytes | size=%d bytes", len(audio_bytes))
    except Exception as e:
        logger.exception("Failed to encode output audio: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to encode output audio: {str(e)}")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )
