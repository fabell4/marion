
import os, time, ipaddress
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct")

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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
        bucket[key] = {"count": 1, "exp": now + ttl}; return False
    entry["count"] += 1
    return entry["count"] > limit

def normalize_messages(body: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages[] is required")
    if len(messages) > 40:
        raise HTTPException(status_code=400, detail="too many messages")
    return messages

def openai_headers():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing OPENAI_API_KEY")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

async def call_openai(messages: List[Dict[str, str]], temperature: float, max_tokens: int):
    payload = {"model": OPENAI_MODEL,"messages": messages,"temperature": temperature,"max_tokens": max_tokens,"stream": False}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=openai_headers())
        if r.status_code >= 400: raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        return {"reply": data["choices"][0]["message"]["content"], "usage": data.get("usage", {}), "model": data.get("model", OPENAI_MODEL), "provider": "openai"}

def build_hf_prompt(messages: List[Dict[str,str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role","user"); content = m.get("content","")
        if role == "user": parts.append(f"User: {content}")
        elif role == "assistant": parts.append(f"Assistant: {content}")
        else: parts.append(content)
    parts.append("Assistant:"); return "\n".join(parts)

def hf_headers():
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_API_TOKEN")
    return {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}

async def call_hf(messages: List[Dict[str,str]], temperature: float, max_tokens: int):
    payload = {"inputs": build_hf_prompt(messages), "parameters": {"max_new_tokens": max_tokens, "temperature": temperature, "return_full_text": False}}
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload, headers=hf_headers())
        if r.status_code >= 400: raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"]
        else:
            text = ""
        return {"reply": text, "usage": {}, "model": HF_MODEL, "provider": "huggingface"}

@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = normalize_messages(body)
    ip = client_ip(req)
    if hit_limit(rate_window_minute, f"min:{ip}", PER_MINUTE, 60): raise HTTPException(status_code=429, detail="Per-minute limit hit.")
    if hit_limit(rate_window_day, f"day:{ip}", DAILY_CAP, 24*60*60): raise HTTPException(status_code=429, detail="Daily cap reached.")
    temperature = min(max(float(body.get("temperature", 0.7)), 0.0), 1.5)
    max_tokens = min(int(body.get("max_tokens", 512)), 1024)

    if OPENAI_API_KEY: return JSONResponse(await call_openai(messages, temperature, max_tokens))
    if HF_API_TOKEN: return JSONResponse(await call_hf(messages, temperature, max_tokens))
    raise HTTPException(status_code=500, detail="No provider configured. Set OPENAI_API_KEY or HF_API_TOKEN.")

@app.post("/api/chat/stream")
async def chat_stream(req: Request):
    if not OPENAI_API_KEY: raise HTTPException(status_code=400, detail="Streaming not available in Hugging Face free mode.")
    body = await req.json()
    messages = normalize_messages(body)
    ip = client_ip(req)
    if hit_limit(rate_window_minute, f"min:{ip}", PER_MINUTE, 60): raise HTTPException(status_code=429, detail="Per-minute limit hit.")
    if hit_limit(rate_window_day, f"day:{ip}", DAILY_CAP, 24*60*60): raise HTTPException(status_code=429, detail="Daily cap reached.")
    temperature = min(max(float(body.get("temperature", 0.7)), 0.0), 1.5)
    max_tokens = min(int(body.get("max_tokens", 512)), 1024)

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                payload = {"model": OPENAI_MODEL,"messages": messages,"temperature": temperature,"max_tokens": max_tokens,"stream": True}
                async with client.stream("POST", f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=openai_headers()) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line: continue
                        yield f"data: {line}\n\n"
        except httpx.HTTPError as e:
            yield f"event: error\ndata: {str(e)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
async def root():
    mode = "openai" if OPENAI_API_KEY else ("huggingface" if HF_API_TOKEN else "unconfigured")
    return PlainTextResponse(f"AI Assistant Proxy up. Mode: {mode}. POST /api/chat")
