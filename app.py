import os, time, ipaddress
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

# --- config from environment ---
HF_API_TOKEN = (os.environ.get("HF_API_TOKEN") or "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
PER_MINUTE = int(os.environ.get("PER_MINUTE", "6"))
DAILY_CAP = int(os.environ.get("DAILY_CAP", "50"))

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["content-type"],
)

# --- simple in-memory rate limits (per IP) ---
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
    mode = "huggingface" if HF_API_TOKEN else "unconfigured"
    return PlainTextResponse(f"AI Assistant Proxy up. Mode: {mode}. POST /api/chat")

async def hf_generate(prompt: str, temperature: float, max_new_tokens: int) -> str:
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfigured: missing HF_API_TOKEN")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
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
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return ""

async def handle_chat(req: Request):
    try:
        print(f"=== HANDLE_CHAT START ===")
        
        try:
            body = await req.json()
            print(f"Body parsed: {body}")
        except Exception as e:
            print(f"JSON parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        messages = body.get("messages")
        print(f"Messages: {messages}")
        if not isinstance(messages, list) or not messages:
            print("Messages validation failed")
            raise HTTPException(status_code=400, detail="messages[] is required")

        ip = client_ip(req)
        print(f"Client IP: {ip}")
        
        if hit(rate_min, f"min:{ip}", PER_MINUTE, 60):
            print("Rate limit hit - per minute")
            raise HTTPException(status_code=429, detail="Per-minute limit hit.")
        if hit(rate_day, f"day:{ip}", DAILY_CAP, 24 * 60 * 60):
            print("Rate limit hit - daily")
            raise HTTPException(status_code=429, detail="Daily cap reached.")

        prompt = to_instruct_prompt(messages)
        print(f"Prompt created: {prompt[:100]}...")
        
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens", 256))
        print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
        
        print("Calling HF generate...")
        text = await hf_generate(prompt, temperature, max_tokens)
        print(f"HF response: {text[:100] if text else 'empty'}...")
        
        response = JSONResponse(
            {"reply": text, "usage": {}, "model": HF_MODEL, "provider": "huggingface"}
        )
        print("Response created successfully")
        return response
        
    except HTTPException:
        print("HTTPException re-raised")
        raise
    except Exception as e:
        print(f"Unexpected error in handle_chat: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Multiple route patterns to handle different URL formats - simplified
@app.api_route("/api/chat", methods=["POST"], response_class=JSONResponse)
async def chat_main(req: Request):
    print(f"=== CHAT ENDPOINT HIT ===")
    print(f"Method: {req.method}")
    print(f"Content-Type: {req.headers.get('content-type')}")
    return await handle_chat(req)

@app.api_route("/api/chat/", methods=["POST"], response_class=JSONResponse) 
async def chat_slash(req: Request):
    print(f"=== CHAT SLASH ENDPOINT HIT ===")
    return await handle_chat(req)

# Test with a different endpoint name to rule out caching
@app.api_route("/api/chat-test", methods=["POST"], response_class=JSONResponse)
async def chat_test(req: Request):
    print(f"=== CHAT TEST ENDPOINT HIT ===")
    return await handle_chat(req)

# Simple test endpoint that doesn't call handle_chat
@app.api_route("/api/simple-test", methods=["POST"], response_class=JSONResponse)
async def simple_test(req: Request):
    print(f"=== SIMPLE TEST ENDPOINT HIT ===")
    try:
        body = await req.json()
        return JSONResponse({"status": "success", "received": body})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})
@app.api_route("/api/chat/{path:path}", methods=["POST", "GET"])
async def chat_catchall(req: Request, path: str):
    if req.method == "POST":
        return await handle_chat(req)
    else:
        return {"message": f"POST to /api/chat, received GET to /api/chat/{path}"}

# Add logging middleware to debug requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"=== REQUEST DEBUG ===")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Path: {request.url.path}")
    print(f"Headers: {dict(request.headers)}")
    print(f"==================")
    
    response = await call_next(request)
    
    print(f"=== RESPONSE DEBUG ===")
    print(f"Status: {response.status_code}")
    print(f"==================")
    
    return response

@app.get("/api/ping")
def ping():
    return {"ok": True}

@app.post("/api/echo")
async def echo(req: Request):
    body = await req.json()
    return {"you_sent": body}

# Add a debug route to see all available routes
@app.get("/debug/routes")
def list_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else []
            })
    return {"routes": routes}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "model": HF_MODEL, "token_configured": bool(HF_API_TOKEN)}