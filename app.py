# app.py
"""
AI Assistant Proxy (FastAPI)
- Primary provider: Groq (free tier), OpenAI-compatible Chat Completions API
- Optional fallback: Hugging Face Inference API (serverless) if configured
- Safe for static front-ends: keys stay server-side
- Endpoints:
    GET  /                -> basic status
    GET  /api/ping        -> health check
    POST /api/chat        -> chat endpoint (also /api/chat/)
Env variables to set in the Space (Settings → Variables & secrets):
  Secrets:
    GROQ_API_KEY          (required for Groq)
    HF_API_TOKEN          (optional; enables HF fallback)
  Variables:
    GROQ_MODEL            default: llama-3.1-8b-instruct
    HF_MODEL              default: mistralai/Mistral-7B-Instruct
    ALLOWED_ORIGINS       e.g. "http://localhost:8080,https://your-domain,https://yourname.github.io"
    PER_MINUTE            default: 6
    DAILY_CAP             default: 50
"""

import os
import time
import ipaddress
from typing import Dict, Any, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

# === Config ===
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instruct")

HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip()
HF_MODEL     = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct")

ALLOWED_ORIGINS = [o.strip() for o in (os.environ.get("ALLOWED_ORIGINS") or "*").split(",") if o.strip()]
PER_MINUTE = int(os.environ.get("PER_MINUTE", "6"))
DAILY_CAP  = int(os.environ.get("DAILY_CAP", "50"))

app = FastAPI()

# CORS (allow your local dev and public domains via ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["content-type"],
)

# === Simple in-memory rate limiting (per IP) ===
rate_min: Dict[str, Dict[str, Any]] = {}
rate_day: Dict[str, Dict[str, Any]] = {}

def client_ip(req: Request) -> str:
    fwd = (req.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = fwd or (req.client.host if req.client else "") or "0.0.0.0"
    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        return "0.0.0.0"

def hit(bucket: Dict[str, Dict[str, Any]], key: str, limit: int, ttl: int) -> bool:
    now = time.time()
    entry = bucket.get(key)
    if not entry or now > entry["exp"]:
        bucket[key] = {"count": 1, "exp": now + ttl}
        return False
    entry["count"] += 1
    return entry["count"] > limit

# === Utility: convert chat messages -> simple instruct prompt (for HF fallback) ===
def to_instruct_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    parts.append("Assistant:")
    return "\n".join(parts)

@app.get("/")
def root():
    mode = "groq" if GROQ_API_KEY else ("huggingface" if HF_API_TOKEN else "unconfigured")
    return PlainTextResponse(f"AI Assistant Proxy up. Mode: {mode}. POST /api/chat")

# === Providers ===
async def groq_generate_chat(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    """Call Groq's OpenAI-compatible Chat Completions API."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing GROQ_API_KEY")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,          # expects [{"role": "...", "content": "..."}]
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            # Surface Groq errors clearly to logs/client
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise HTTPException(status_code=500, detail="Unexpected Groq response shape")

async def hf_generate(prompt: str, temperature: float, max_new_tokens: int) -> str:
    """Call Hugging Face Serverless Inference (text-generation pipeline)."""
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_API_TOKEN")

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        # HF can return a list or a dict depending on backend
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # If the structure is different (e.g., model loading), surface it
        return ""

# === Main handler ===
async def handle_chat(req: Request):
    # Parse & validate
    try:
        body = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages[] is required")

    # Rate limits
    ip = client_ip(req)
    if hit(rate_min, f"min:{ip}", PER_MINUTE, 60):
        raise HTTPException(status_code=429, detail="Per-minute limit hit.")
    if hit(rate_day, f"day:{ip}", DAILY_CAP, 24 * 60 * 60):
        raise HTTPException(status_code=429, detail="Daily cap reached.")

    # Params
    temperature = float(body.get("temperature", 0.7))
    max_tokens  = int(body.get("max_tokens", 256))

    # Provider order: Groq → HF fallback
    # If Groq call fails with HTTP errors and HF is configured, we try HF.
    provider = None
    model = None
    text = None

    # Try Groq first if key present
    if GROQ_API_KEY:
        try:
            text = await groq_generate_chat(messages, temperature, max_tokens)
            provider = "groq"; model = GROQ_MODEL
        except HTTPException as e:
            # Only fall back if HF is configured; otherwise propagate error
            if HF_API_TOKEN:
                # Build instruct-style prompt for HF
                prompt = to_instruct_prompt(messages)
                text = await hf_generate(prompt, temperature, max_tokens)
                provider = "huggingface"; model = HF_MODEL
            else:
                raise e
    # Else, try HF directly (if configured)
    elif HF_API_TOKEN:
        prompt = to_instruct_prompt(messages)
        text = await hf_generate(prompt, temperature, max_tokens)
        provider = "huggingface"; model = HF_MODEL
    else:
        raise HTTPException(status_code=500, detail="No provider configured. Set GROQ_API_KEY or HF_API_TOKEN.")

    return JSONResponse({"reply": text or "", "usage": {}, "model": model, "provider": provider})

# === Routes ===
@app.post("/api/chat")
async def chat(req: Request):
    return await handle_chat(req)

@app.post("/api/chat/")
async def chat_slash(req: Request):
    return await handle_chat(req)

@app.get("/api/ping")
def ping():
    return {"ok": True}