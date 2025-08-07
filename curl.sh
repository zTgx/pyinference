curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "gpt2",
    "prompt": "Explain FastAPI in simple terms",
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": false
  }'

curl -X GET "http://127.0.0.1:8000/v1/models"