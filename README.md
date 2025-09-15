# AI Assistant Proxy — Hugging Face Space (FastAPI)

This Space exposes a tiny server-side proxy to OpenAI so your GitHub Pages (or any static site) never sees your API key.

## Endpoints
- `POST /api/chat` — calls OpenAI Chat Completions and returns a JSON `{ reply, usage, model }`
- `POST /api/chat/stream` — (optional) Server-Sent Events (SSE) streaming of the reply as events
