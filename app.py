# app.py
"""
AI Assistant Proxy (FastAPI)
- Primary provider: Groq (OpenAI-compatible Chat Completions)
- Optional fallback: Hugging Face Serverless Inference (text-generation)
- Wake word: responds only when addressed by name (default: "Marion")
- Real-time context: injects current date/time (USER_TZ) into system prompt

Env (Space → Settings → Variables & secrets)
  Secrets:
    GROQ_API_KEY           (required for Groq)
    HF_API_TOKEN           (optional; enables HF fallback)
  Variables:
    GROQ_MODEL             default: llama-3.1-8b-instant
    HF_MODEL               default: mistralai/Mistral-7B-Instruct
    ALLOWED_ORIGINS        e.g. "http://localhost:8080,https://your-domain,https://you.github.io"
    PER_MINUTE             default: 6
    DAILY_CAP              default: 50
    ASSISTANT_NAME         default: Marion
    WAKE_WORD_MODE         default: require   # require | prefer | off
    USER_TZ                default: America/New_York
"""

import os
import re
import time
import ipaddress
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

# === Config ===
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip()
HF_MODEL     = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct")

ALLOWED_ORIGINS = [o.strip() for o in (os.environ.get("ALLOWED_ORIGINS") or "*").split(",") if o.strip()]
PER_MINUTE = int(os.environ.get("PER_MINUTE", "6"))
DAILY_CAP  = int(os.environ.get("DAILY_CAP", "50"))

ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "Marion")
WAKE_WORD_MODE = (os.environ.get("WAKE_WORD_MODE") or "require").lower()  # require|prefer|off
USER_TZ        = os.environ.get("USER_TZ", "America/New_York")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["content-type"],
)

# === Rate limits (per IP, in-memory) ===
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

# === Time helpers ===
def _now_strings():
    now_utc = datetime.now(timezone.utc)
    try:
        tz = ZoneInfo(USER_TZ) if ZoneInfo else timezone.utc
    except Exception:
        tz = timezone.utc
    now_local = now_utc.astimezone(tz)
    off = now_local.utcoffset() or timedelta(0)
    sign = "+" if off >= timedelta(0) else "-"
    hh = int(abs(off).total_seconds() // 3600)
    mm = int((abs(off).total_seconds() % 3600) // 60)
    offset_str = f"{sign}{hh:02d}:{mm:02d}"
    return {
        "iso_local": now_local.isoformat(timespec="seconds"),
        "iso_utc": now_utc.isoformat(timespec="seconds"),
        "tz_abbr": now_local.tzname() or USER_TZ,
        "offset": offset_str,
        "pretty_date": now_local.strftime("%A, %B %d, %Y"),
        "pretty_time": now_local.strftime("%I:%M:%S %p").lstrip("0"),
    }

def time_system_message():
    n = _now_strings()
    tail = ""
    if WAKE_WORD_MODE == "prefer":
        tail = f" If the user's message doesn't include your name, politely encourage them to say '{ASSISTANT_NAME}' next time."
    return (
        f"You are {ASSISTANT_NAME}, a concise, friendly assistant."
        f" Current datetime (local): {n['iso_local']} ({n['tz_abbr']} {n['offset']})."
        f" Today is {n['pretty_date']}, time is {n['pretty_time']}."
        f" Use this as the source of truth for 'now' and the current date/time."
        f"{tail}"
    )

# === Wake-word helpers ===
def mentions_name(text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(ASSISTANT_NAME)}\b", text or "", re.IGNORECASE))

# === Prompt conversion for HF fallback ===
def to_instruct_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = [f"System: {time_system_message()}"]
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
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise HTTPException(status_code=500, detail="Unexpected Groq response shape")

async def hf_generate(prompt: str, temperature: float, max_new_tokens: int) -> str:
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_API_TOKEN")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature, "return_full_text": False},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return ""

# === Main handler ===
async def handle_chat(req: Request):
    # Parse
    try:
        body = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages[] is required")

    # Rate limit
    ip = client_ip(req)
    if hit(rate_min, f"min:{ip}", PER_MINUTE, 60):
        raise HTTPException(status_code=429, detail="Per-minute limit hit.")
    if hit(rate_day, f"day:{ip}", DAILY_CAP, 24 * 60 * 60):
        raise HTTPException(status_code=429, detail="Daily cap reached.")

    # Params
    temperature = float(body.get("temperature", 0.7))
    max_tokens  = int(body.get("max_tokens", 256))

    # Wake-word gating (look at most recent user message)
    user_last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if WAKE_WORD_MODE == "require" and not mentions_name(user_last):
        return JSONResponse(
            {"reply": f"(…waiting) Say '{ASSISTANT_NAME}' to talk to me.",
             "provider": "guard", "model": None}
        )

    # Inject time + persona system message (always fresh)
    sys = {"role": "system", "content": time_system_message()}
    if not messages or messages[0].get("role") != "system":
        messages = [sys] + messages
    else:
        messages = [sys] + messages  # prepend ours to ensure current timestamp

    # Provider order: Groq → HF
    if GROQ_API_KEY:
        try:
            text = await groq_generate_chat(messages, temperature, max_tokens)
            return JSONResponse({"reply": text or "", "usage": {}, "model": GROQ_MODEL, "provider": "groq"})
        except HTTPException as e:
            if HF_API_TOKEN:
                prompt = to_instruct_prompt(messages)
                text = await hf_generate(prompt, temperature, max_tokens)
                return JSONResponse({"reply": text or "", "usage": {}, "model": HF_MODEL, "provider": "huggingface"})
            raise e
    elif HF_API_TOKEN:
        prompt = to_instruct_prompt(messages)
        text = await hf_generate(prompt, temperature, max_tokens)
        return JSONResponse({"reply": text or "", "usage": {}, "model": HF_MODEL, "provider": "huggingface"})
    else:
        raise HTTPException(status_code=500, detail="No provider configured. Set GROQ_API_KEY or HF_API_TOKEN.")

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
