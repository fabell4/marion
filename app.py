# app.py
import os, time, ipaddress
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
PER_MINUTE = int(os.environ.get("PER_MINUTE", "6"))
DAILY_CAP = int(os.environ.get("DAILY_CAP", "50"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["content-type"],
)

# ---- simple in-memory rate limits ----
rate_window_minute: Dict[str, Dict[str, Any]] = {}
rate_window_day: Dict[str, Dict[str, Any]] = {}

def client_ip(req: Request) -> str:
    fwd = (req.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = fwd or (req.client.host if req.client else "") or "0.0.0.0"
    try:
        ipaddress.ip_address(ip); return ip
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

def to_instruct_prompt(messages: List[Dict[str,str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role","user"); content = m.get("content","")
        if role == "user": parts.append(f"User: {content}")
        elif role == "assistant": parts.append(f"Assistant: {content}")
        else: parts.append(content)
    parts.append("Assistant:"); return "\n".join(parts)

@app.get("/")
def root():
    mode = "huggingface" if HF_API_TOKEN else "unconfigured"
    return PlainTextResponse(f"AI Assistant Proxy up. Mode: {mode}. POST /api/chat")

@app.post("/api/chat")
async def chat(req: Request):
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_API_TOKEN")
    body = await req.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages[] is required")

    ip = client_ip(req)
    if hit_limit(rate_window_minute, f"min:{ip}", PER_MINUTE, 60):
        raise HTTPException(status_code=429, detail="Per-minute limit hit.")
    if hit_limit(rate_window_day, f"day:{ip}", DAILY_CAP, 24*60*60):
        raise HTTPException(status_code=429, detail="Daily cap reached.")

    payload = {
        "inputs": to_instruct_prompt(messages),
        "parameters": {
            "max_new_tokens": int(body.get("max_tokens", 256)),
            "temperature": float(body.get("temperature", 0.7)),
            "return_full_text": False
        }
    }
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        text = ""
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"]
        return JSONResponse({"reply": text, "usage": {}, "model": HF_MODEL, "provider": "huggingface"})
