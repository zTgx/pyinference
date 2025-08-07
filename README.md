# pyinference
Python Inference Service

* vLLM
* torch
* fastapi
* transformer

## Usage
```bash
python app.py --port 8888
```

## Inference
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Explain FastAPI in simple terms",
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": false
  }'
```