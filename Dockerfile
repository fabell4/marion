FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
# HF exposes PORT; default to 7860
ENV PORT=7860
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
