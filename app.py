import os, time, ipaddress, asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PER_MINUTE = int(os.environ.get("PER_MINUTE", "6"))
DAILY_CAP = int(os.environ.get("DAILY_CAP", "50"))

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["content-type"],
)

# Simple in-memory rate limiter (per-IP). Fine for demos; resets on Space restarts.
rate_window_minute: Dict[str, Dict[str, Any]] = {}
rate_window_day: Dict[str, Dict[str, Any]] = {}

def client_ip(req: Request) -> str:
    # HF sets x-forwarded-for; fall back to client.host
    fwd = (req.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = fwd or (req.client.host if req.client else "") or "0.0.0.0"
    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        return "0.0.0.0"

def hit_limit(bucket: Dict[str, Dict[str, Any]], key: str, limit: int, ttl: int) -> bool:
    now = time.time()
    entry = bucket.get(key)
    if not entry or now > entry["exp"]:
        bucket[key] = {"count": 1, "exp": now + ttl}
        return False
    entry["count"] += 1
    return entry["count"] > limit

def normalize_messages(body: Dict[str, Any]):
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages[] is required")
    # Basic size guard
    if len(messages) > 40:
        raise HTTPException(status_code=400, detail="too many messages")
    return messages

def base_headers():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing OPENAI_API_KEY")
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = normalize_messages(body)

    # Rate limits
    ip = client_ip(req)
    if hit_limit(rate_window_minute, f"min:{ip}", PER_MINUTE, 60):
        raise HTTPException(status_code=429, detail="Per-minute limit hit. Try again shortly.")
    if hit_limit(rate_window_day, f"day:{ip}", DAILY_CAP, 24*60*60):
        raise HTTPException(status_code=429, detail="Daily cap reached.")

    model = body.get("model", OPENAI_MODEL)
    json_payload = {
        "model": model,
        "messages": messages,
        "temperature": min(max(body.get("temperature", 0.7), 0.0), 1.5),
        "max_tokens": min(int(body.get("max_tokens", 512)), 2048),
        "stream": False
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{OPENAI_BASE_URL}/chat/completions", json=json_payload, headers=base_headers())
        if r.status_code >= 400:
            # Pass OpenAI error through for easier debugging
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return JSONResponse({"reply": reply, "usage": usage, "model": data.get("model", model)})

@app.post("/api/chat/stream")
async def chat_stream(req: Request):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Frontend should use EventSource or fetch+reader; content-type: text/event-stream.
    """
    body = await req.json()
    messages = normalize_messages(body)

    # Rate limits (lighter for streaming start)
    ip = client_ip(req)
    if hit_limit(rate_window_minute, f"min:{ip}", PER_MINUTE, 60):
        raise HTTPException(status_code=429, detail="Per-minute limit hit. Try again shortly.")
    if hit_limit(rate_window_day, f"day:{ip}", DAILY_CAP, 24*60*60):
        raise HTTPException(status_code=429, detail="Daily cap reached.")

    model = body.get("model", OPENAI_MODEL)
    json_payload = {
        "model": model,
        "messages": messages,
        "temperature": min(max(body.get("temperature", 0.7), 0.0), 1.5),
        "max_tokens": min(int(body.get("max_tokens", 512)), 2048),
        "stream": True
    }

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OPENAI_BASE_URL}/chat/completions",
                                         json=json_payload, headers=base_headers()) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        # OpenAI streams "data: {json}" lines; passthrough with SSE format
                        yield f"data: {line}\n\n"
        except httpx.HTTPError as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
async def root():
    return PlainTextResponse("AI Assistant Proxy up. POST /api/chat or /api/chat/stream")
