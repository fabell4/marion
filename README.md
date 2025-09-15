# Marion

Marion is an AI assistant proxy. The backend is hosted on Hugging Face to minimize cost for the demo.

The bot is configured support two modes:
1) OpenAI mode (set OPENAI_API_KEY)
2) Free Hugging Face Inference API mode (set HF_API_TOKEN; default model mistralai/Mistral-7B-Instruct)

An alternative to Hugging Face hosting could be to use Cloudflare Workers for the back end. 
