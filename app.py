from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from contextlib import asynccontextmanager

import torch
import uvicorn
from datetime import datetime
from typing import List, Dict, Any
import requests

from log import logger, log_system_info
from hf import initialize_model
from device import available_device
from args import args
from completion import do_completion

device = available_device()

api_key_header = APIKeyHeader(name="X-API-Key")
API_KEYS = {"your-secret-key"}

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    model: str
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application initialization...")

    try:
        initialize_model(args.model)
        logger.info("Application startup complete!")
        yield
    finally:
        logger.info("Shutting down application...")
        try:
            torch.cuda.empty_cache()
            logger.info("Model unloaded and CUDA cache cleared")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        logger.info("Shutdown complete")

app = FastAPI(
    title="Python Inference Service",
    description="OpenAI-compatible API Service for LLM in Python and vLLM",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models", response_model=List[str])
async def list_models():
    """List available models from Hugging Face Hub"""
    API_URL = "https://huggingface.co/api/models"
    params = {
        "sort": "downloads",
        "direction": -1,
        "limit": 100
    }
    response = requests.get(API_URL, params=params)
    models = [model["modelId"] for model in response.json()]
    return models

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def generate(
    request: GenerateRequest,
    # api_key: str = Security(api_key_header)
):
    # if api_key not in API_KEYS:
    #     raise HTTPException(status_code=403, detail="Invalid API Key")

    try:
        generated_text = await do_completion(request)

        return ChatCompletionResponse(
            model=request.model,
            text=generated_text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    log_system_info()

    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)