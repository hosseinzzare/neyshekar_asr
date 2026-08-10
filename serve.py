"""
Inference Service
=================

A small HTTP service around the fine-tuned adapter. Accepts an audio file, returns the Persian
transcript.

    GET  /health      is the service up, and did the model load
    POST /transcribe  multipart audio file  ->  {"text": "..."}
    GET  /            a one-page form, so the service can be tried without a terminal

Two decisions shape this file.

The audio goes through the same preprocessing as training. trim_silence and peak_normalize are
imported from src/dataset.py rather than reimplemented, because a service that preprocesses
differently from the training pipeline will quietly produce worse transcripts than the reported
WER suggests, and nothing will announce that it is happening.

Quantisation is chosen from the hardware rather than fixed. 4-bit NF4 needs a CUDA GPU;
bitsandbytes has no CPU path for it. On a machine without a GPU the base model loads in float32
and the adapter is applied on top, which is slow but correct. Most people running this will not
have a GPU, so failing on that machine would defeat the point of shipping a container.

Usage
-----
    uvicorn serve:app --host 0.0.0.0 --port 8000
    curl -F "file=@clip.wav" http://localhost:8000/transcribe
"""

import io
import logging
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("serve")

ADAPTER = os.environ.get("ADAPTER_ID", "hosseinzr/neyshekar-whisper-large-v3-lora")
BASE_MODEL = os.environ.get("BASE_MODEL", "openai/whisper-large-v3")
LANGUAGE = os.environ.get("LANGUAGE", "persian")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))
TARGET_SR = 16000

_state = {"model": None, "processor": None, "device": None, "quantised": None, "error": None}


def _load_model():
    """
    Load the base model and apply the adapter. Called once at startup.

    Kept out of module scope so an import of this file (by a test, or by the app server's
    reloader) does not pull three gigabytes off the network as a side effect.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    has_cuda = torch.cuda.is_available()
    log.info("device: %s", torch.cuda.get_device_name(0) if has_cuda else "CPU")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task="transcribe")

    if has_cuda:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
        base = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, quantization_config=quant, device_map="auto")
        quantised = True
    else:
        # No CUDA means no bitsandbytes. float32 on CPU is slow but produces the same text.
        base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=torch.float32)
        quantised = False

    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    device = "cuda" if has_cuda else "cpu"
    if not has_cuda:
        model = model.to(device)

    _state.update(model=model, processor=processor, device=device, quantised=quantised)
    log.info("adapter %s loaded on %s (4-bit: %s)", ADAPTER, device, quantised)


def _read_audio(raw: bytes) -> np.ndarray:
    """Decode to mono float32 at 16 kHz, then apply the training-time preprocessing."""
    import soundfile as sf
    from dataset import trim_silence, peak_normalize
    import config

    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        # Whisper's feature extractor assumes 16 kHz. Declaring 16 kHz for audio that is not
        # would time-stretch it and quietly wreck the transcript.
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    if getattr(config, "ENABLE_VAD_TRIM", True):
        audio = trim_silence(audio, sr=TARGET_SR)
    if getattr(config, "ENABLE_PEAK_NORM", True):
        audio = peak_normalize(audio)
    return audio.astype(np.float32)


def _transcribe(audio: np.ndarray) -> str:
    import torch
    processor, model = _state["processor"], _state["model"]
    feats = processor.feature_extractor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features
    feats = feats.to(_state["device"])
    if _state["quantised"]:
        feats = feats.half()
    with torch.no_grad():
        ids = model.generate(input_features=feats, max_new_tokens=225)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------------------
try:
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:                                          # pragma: no cover
    sys.exit("FastAPI is not installed. Run:  pip install -r requirements-docker.txt")


@asynccontextmanager
async def lifespan(_: "FastAPI"):
    """
    Load the model when the service starts rather than on the first request.

    Loading lazily would make the first caller wait several minutes and probably time out,
    while /health reported success for a service that could not actually answer. A failure is
    recorded rather than raised, so /health can say what went wrong instead of the container
    dying with a stack trace and restarting into the same failure.
    """
    t0 = time.time()
    try:
        _load_model()
        log.info("ready in %.1fs", time.time() - t0)
    except Exception as e:                                   # noqa: BLE001
        _state["error"] = f"{type(e).__name__}: {e}"
        log.error("model failed to load: %s", _state["error"])
    yield


app = FastAPI(title="Neyshekar Persian ASR",
              description="Whisper large-v3 fine-tuned with QLoRA on the Neyshekar corpus.",
              version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    ready = _state["model"] is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "unavailable",
                 "device": _state["device"], "quantised": _state["quantised"],
                 "base_model": BASE_MODEL, "adapter": ADAPTER,
                 "error": _state["error"]})


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(503, detail=f"model not loaded: {_state['error']}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, detail="empty file")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, detail=f"file exceeds {MAX_UPLOAD_MB:.0f} MB")

    t0 = time.time()
    try:
        audio = _read_audio(raw)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, detail=f"could not decode audio: {type(e).__name__}: {e}")

    duration = len(audio) / TARGET_SR
    if duration < 0.1:
        raise HTTPException(400, detail="audio is shorter than 0.1 s after silence trimming")

    text = _transcribe(audio)
    elapsed = time.time() - t0
    log.info("%s  %.1fs audio  %.1fs compute", file.filename, duration, elapsed)
    return {"text": text,
            "audio_seconds": round(duration, 2),
            "compute_seconds": round(elapsed, 2),
            "device": _state["device"]}


INDEX = """
<!doctype html><meta charset="utf-8"><title>Neyshekar Persian ASR</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem;color:#222}
 h1{font-size:1.35rem;margin-bottom:.2rem} p.sub{color:#666;margin-top:0;font-size:.9rem}
 form{border:1px solid #ddd;border-radius:8px;padding:1.2rem;margin-top:1.5rem}
 button{padding:.5rem 1.1rem;font-size:1rem;cursor:pointer;margin-top:.8rem}
 #out{margin-top:1.2rem;padding:1rem;background:#f6f8fa;border-radius:8px;
      direction:rtl;text-align:right;font-size:1.05rem;min-height:1.5rem}
 small{color:#666}
</style>
<h1>Neyshekar Persian ASR</h1>
<p class="sub">Whisper large-v3, QLoRA fine-tuned. WER 8.05% / CER 2.00% on the validation split.</p>
<form id="f"><input type="file" name="file" accept="audio/*" required>
<button>Transcribe</button></form>
<div id="out"></div>
<p><small>On a CPU-only machine a short clip takes tens of seconds.</small></p>
<script>
f.onsubmit = async e => {
  e.preventDefault(); out.textContent = 'transcribing...';
  const r = await fetch('/transcribe', {method:'POST', body:new FormData(f)});
  const j = await r.json();
  out.textContent = r.ok ? j.text : ('error: ' + (j.detail || r.status));
};
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX
