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

# --- simple in-memory quotas ---
rate_min: Dict[str, Dict[str, Any]] = {}
rate_day: Dict[str, Dict[str, Any]] = {}

def client_ip(req: Request) -> str:
    fwd = (req.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = fwd or (req.client.host if req.client else "") or "0.0.0.0"
    try:
        ipaddress.ip_address(ip); return ip
    except ValueError:
        return "0.0.0.0"

def hit(bucket: Dict[str, Dict[str, Any]], key: str, limit: int, ttl: int) -> bool:
    now = time.time()
    entry = bucket.get(key)
    if not entry or now > entry["exp"]:
        bucket[key] = {"count": 1, "exp": now + ttl}; return False
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

async def hf_generate(prompt: str, temperature: float, max_new_tokens: int) -> str:
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_API_TOKEN")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload =
