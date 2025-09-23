# Marion — AI Assistant Proxy

A tiny FastAPI service that fronts an LLM provider (e.g., Groq) with an optional Hugging Face Inference fallback.  
Designed for portfolio demos: server-side keys, CORS allowlist, wake-word (“Marion”) and current date/time injection.

---

## Features
- Chat Completions (OpenAI-compatible) provider (e.g., Groq)
- Optional fallback to Hugging Face Serverless Inference (text-generation)
- Wake-word modes: `require` | `prefer` | `off` (default: `require`)
- Real-time awareness: server injects current date/time (configurable timezone)
- CORS allowlist + simple per-IP rate limits

---

## Project Layout
```
marion/             # backend service (FastAPI)
  app.py
  requirements.txt
  Dockerfile
README.md
LICENSE
```

---

## Quick Start (locally)

> **Requires**: Python 3.10+ (or Docker)

**Python**
```bash
cd marion
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PORT=7860
uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Docker**
```bash
cd marion
docker build -t marion-proxy .
docker run -p 7860:7860 --env-file .env marion-proxy
```

---

## Configuration

Set via environment variables (or your platform’s “Variables & secrets”).

**Secrets**
- `GROQ_API_KEY` — Chat provider key (if using Groq or similar)
- `HF_API_TOKEN` — (optional) enables Hugging Face fallback

**Variables**
- `GROQ_MODEL` — default: `llama-3.1-8b-instant`
- `HF_MODEL` — default: `mistralai/Mistral-7B-Instruct`
- `ALLOWED_ORIGINS` — comma-separated list (e.g., `http://localhost:8080,https://example.com`)
- `ASSISTANT_NAME` — default: `Marion`
- `WAKE_WORD_MODE` — `require` | `prefer` | `off`
- `USER_TZ` — IANA timezone (e.g., `America/New_York`)
- `PER_MINUTE` — requests per minute per IP (default: `6`)
- `DAILY_CAP` — requests per day per IP (default: `50`)

> For perfect timezone support, add `tzdata` to `requirements.txt` (or install OS tzdata in the image).

---

## API

**POST** `/api/chat`  
Body (example):
```json
{
  "messages": [
    { "role": "user", "content": "Marion, say hi in one short sentence." }
  ],
  "temperature": 0.7,
  "max_tokens": 256
}
```
Response (example):
```json
{ "reply": "Hello.", "model": "llm-name", "provider": "groq" }
```

**GET** `/api/ping` → `{ "ok": true }`

---

## Frontend Usage

Point your frontend to the proxy URL and POST the `messages` array.

Example env for a Vite/React app:
```
VITE_AI_URL=https://<your-backend-host>/api/chat
```

---

## Deploy Notes (optional)

If hosting on a git-based platform, keep secrets in platform settings.  
To mirror this `marion/` folder to another remote as a deploy repo, you can use `git subtree`:

```bash
# pull remote repo into subfolder (one-time)
git remote add deploy <DEPLOY_REPO_URL>
git fetch deploy
git subtree add --prefix=marion deploy main --squash

# push only marion/ to the deploy repo (subsequent deploys)
git subtree push --prefix=marion deploy main
```

---

## Security
- **Do not commit secrets.** Keep API keys in your hosting platform’s secrets store.
- Add to `.gitignore`:
```
.env
.venv
```

---

## License
MIT
