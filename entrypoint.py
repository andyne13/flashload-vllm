"""
OpenAI-compatible API server with flashtensors for fast model switching.
Supports multimodal (vision) models like Qwen3-VL.
"""

import os

# Set environment variables FIRST, before any imports
os.environ["VLLM_USE_V1"] = "0"  # Required for flashtensors integration

# CRITICAL: Configure and activate flashtensors BEFORE importing torch/vLLM
# The activation patches vLLM's model loader to support the "flash" format
import flashtensors as ft

print("Configuring flashtensors...")
ft.configure(
    storage_path="/models",
    mem_pool_size=1024**3 * 20,  # 20GB (L4 has 24GB)
    chunk_size=1024**2 * 32,      # 32MB
    gpu_memory_utilization=0.80,
)

print("Activating flashtensors vLLM integration...")
ft.activate_vllm_integration()
print("FlashTensors vLLM integration activated.")

# Register "flash" format in vLLM's loader registry (for newer vLLM versions)
try:
    from vllm.model_executor import model_loader
    if hasattr(model_loader, '_LOAD_FORMAT_TO_MODEL_LOADER'):
        from flashtensors.integrations.vllm import FlashLLMLoader
        model_loader._LOAD_FORMAT_TO_MODEL_LOADER["flash"] = FlashLLMLoader
        model_loader._LOAD_FORMAT_TO_MODEL_LOADER["flashtensors"] = FlashLLMLoader
        print("Registered 'flash' and 'flashtensors' formats in vLLM loader registry.")
    else:
        print("Using older vLLM version - flashtensors activation should handle registration.")
except ImportError as e:
    print(f"Note: Could not register loaders directly: {e}")

# Import torch but don't initialize distributed - let vLLM handle it
import torch
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

# Standard library imports
import asyncio
import base64
import json
import re
import time
import uuid
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# NOW we can import vLLM components
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from vllm import SamplingParams


# ============================================================================
# Pydantic Models for OpenAI API Compatibility (with Vision support)
# ============================================================================

class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"  # "low", "high", or "auto"


class ContentPartText(BaseModel):
    type: str = "text"
    text: str


class ContentPartImageUrl(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl


# Content can be a string or list of content parts (text/image)
ContentPart = Union[ContentPartText, ContentPartImageUrl]


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "flashtensors"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Application State
# ============================================================================

class AppState:
    def __init__(self):
        self.current_model: Optional[str] = None
        self.llm = None
        self.lock = asyncio.Lock()


app = FastAPI(title="FlashTensors OpenAI-Compatible API")
state = AppState()

# CORS middleware for broader compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Startup Configuration
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup completion (flashtensors already initialized at module level)."""
    print("FastAPI server started. FlashTensors ready for model loading.")


# ============================================================================
# Helper Functions
# ============================================================================

def generate_id(prefix: str = "chatcmpl") -> str:
    """Generate a unique ID similar to OpenAI's format."""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# ============================================================================
# Image Processing Helpers
# ============================================================================

async def fetch_image_from_url(url: str) -> Image.Image:
    """Fetch an image from a URL and return as PIL Image."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))


def decode_base64_image(data_url: str) -> Image.Image:
    """Decode a base64 data URL to PIL Image."""
    # Format: data:image/jpeg;base64,/9j/4AAQ...
    match = re.match(r"data:image/[^;]+;base64,(.+)", data_url)
    if not match:
        raise ValueError("Invalid base64 image data URL")
    image_data = base64.b64decode(match.group(1))
    return Image.open(BytesIO(image_data))


async def process_image_url(image_url: str) -> Image.Image:
    """Process an image URL (supports both HTTP URLs and base64 data URLs)."""
    if image_url.startswith("data:"):
        return decode_base64_image(image_url)
    else:
        return await fetch_image_from_url(image_url)


def extract_text_from_content(content: Union[str, List[ContentPart]]) -> str:
    """Extract text content from a message's content field."""
    if isinstance(content, str):
        return content

    text_parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        elif hasattr(part, "type") and part.type == "text":
            text_parts.append(part.text)
    return " ".join(text_parts)


async def extract_images_from_messages(messages: List[ChatMessage]) -> List[Image.Image]:
    """Extract all images from messages."""
    images = []
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url:
                        img = await process_image_url(url)
                        images.append(img)
                elif hasattr(part, "type") and part.type == "image_url":
                    url = part.image_url.url
                    if url:
                        img = await process_image_url(url)
                        images.append(img)
    return images


def is_vision_model(model_id: str) -> bool:
    """Check if the model is a vision-language model."""
    model_lower = model_id.lower()
    return any(x in model_lower for x in ["vl", "vision", "llava", "cogvlm", "internvl"])


# ============================================================================
# Prompt Formatting Functions
# ============================================================================

def format_chat_prompt(messages: List[ChatMessage], model_id: str) -> str:
    """
    Format chat messages into a prompt string.
    Adjust template based on model type.
    """
    # Detect model type and use appropriate template
    model_lower = model_id.lower()

    if "llama" in model_lower or "meta" in model_lower:
        return format_llama_prompt(messages)
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return format_mistral_prompt(messages)
    elif "qwen3-vl" in model_lower or "qwen3vl" in model_lower:
        # Qwen3-VL must be checked before generic Qwen
        return format_qwen3_vl_prompt(messages)
    elif "qwen" in model_lower:
        return format_qwen_prompt(messages)
    elif "phi" in model_lower:
        return format_phi_prompt(messages)
    else:
        # Generic ChatML format
        return format_chatml_prompt(messages)


def format_chatml_prompt(messages: List[ChatMessage]) -> str:
    """ChatML format (used by many models)."""
    prompt = ""
    for msg in messages:
        content = extract_text_from_content(msg.content)
        prompt += f"<|im_start|>{msg.role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def format_llama_prompt(messages: List[ChatMessage]) -> str:
    """Llama 2/3 chat format."""
    prompt = ""
    system_msg = None

    for msg in messages:
        content = extract_text_from_content(msg.content)
        if msg.role == "system":
            system_msg = content
        elif msg.role == "user":
            if system_msg:
                prompt += f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{content} [/INST]"
                system_msg = None
            else:
                prompt += f"<s>[INST] {content} [/INST]"
        elif msg.role == "assistant":
            prompt += f" {content} </s>"

    return prompt


def format_mistral_prompt(messages: List[ChatMessage]) -> str:
    """Mistral/Mixtral chat format."""
    prompt = ""
    for msg in messages:
        content = extract_text_from_content(msg.content)
        if msg.role == "user":
            prompt += f"[INST] {content} [/INST]"
        elif msg.role == "assistant":
            prompt += f"{content}</s>"
        elif msg.role == "system":
            prompt += f"[INST] {content}\n"
    return prompt


def format_qwen_prompt(messages: List[ChatMessage]) -> str:
    """Qwen chat format."""
    prompt = ""
    for msg in messages:
        content = extract_text_from_content(msg.content)
        if msg.role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif msg.role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def format_qwen3_vl_prompt(messages: List[ChatMessage], enable_thinking: bool = False) -> str:
    """
    Qwen3-VL (Vision-Language) chat format.

    Qwen3-VL uses the same base format as Qwen3 but with:
    - <|vision_start|><|image_pad|><|vision_end|> tokens for images
    - Optional <think>...</think> blocks for reasoning (disabled by default for API use)
    - No default system prompt
    - Support for /think and /no_think commands in messages

    Recommended parameters for VL tasks:
    - temperature: 0.7, top_p: 0.8, top_k: 20

    For text-only tasks:
    - temperature: 1.0, top_p: 1.0, top_k: 40
    """
    prompt = ""

    for msg in messages:
        content = format_qwen3_vl_content(msg.content)
        if msg.role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif msg.role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif msg.role == "assistant":
            # For historical assistant messages, include content without think blocks
            # (following Qwen3 convention for multi-turn)
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    # Start assistant response
    prompt += "<|im_start|>assistant\n"

    # If thinking is disabled, prepend empty think block to signal no reasoning
    if not enable_thinking:
        prompt += "<think>\n\n</think>\n\n"

    return prompt


def format_qwen3_vl_content(content: Union[str, List[ContentPart]]) -> str:
    """
    Format content for Qwen3-VL, handling both text and images.
    Images are represented with special vision tokens.
    """
    if isinstance(content, str):
        return content

    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                # Add vision tokens placeholder for each image
                # The actual image data is passed separately via multi_modal_data
                parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        elif hasattr(part, "type"):
            if part.type == "text":
                parts.append(part.text)
            elif part.type == "image_url":
                parts.append("<|vision_start|><|image_pad|><|vision_end|>")

    return "".join(parts)


def format_phi_prompt(messages: List[ChatMessage]) -> str:
    """Phi model chat format."""
    prompt = ""
    for msg in messages:
        content = extract_text_from_content(msg.content)
        if msg.role == "system":
            prompt += f"<|system|>\n{content}<|end|>\n"
        elif msg.role == "user":
            prompt += f"<|user|>\n{content}<|end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{content}<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average)."""
    return len(text) // 4


def reinit_distributed_if_needed():
    """Reinitialize PyTorch distributed if it was destroyed or corrupted."""
    import torch.distributed as dist

    # Always try to destroy existing group first to clean up corrupted state
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("Destroyed existing process group.")
    except Exception as e:
        print(f"Note: Could not destroy existing process group: {e}")

    print("Initializing PyTorch distributed process group...")
    try:
        # Try nccl first (preferred for GPU), fall back to gloo
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0
        )
        print("Process group initialized with nccl.")
    except Exception as e:
        print(f"NCCL failed ({e}), trying gloo...")
        try:
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                world_size=1,
                rank=0
            )
            print("Process group initialized with gloo.")
        except Exception as e2:
            print(f"Warning: Could not initialize process group: {e2}")


async def ensure_model_loaded(model_id: str):
    """Ensure the requested model is loaded."""
    import concurrent.futures
    async with state.lock:
        if state.current_model != model_id:
            # Check if model is registered
            registered_models = ft.list_models()
            if model_id not in registered_models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_id}' not found. Register it first via POST /v1/flashtensors/register"
                )

            # Load new model in thread pool to avoid asyncio conflicts
            # Note: Don't call cleanup_gpu() as it destroys vLLM's distributed state
            def load_model_sync():
                # Just clear our reference, let flashtensors handle cleanup
                old_llm = state.llm
                state.llm = None

                # Force garbage collection to free memory
                if old_llm is not None:
                    del old_llm
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", 8192))
                return ft.load_model(
                    model_id=model_id,
                    backend="vllm",
                    gpu_memory_utilization=0.80,
                    tensor_parallel_size=1,  # Single GPU, no parallelism
                    max_model_len=max_model_len,
                )

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state.llm = await loop.run_in_executor(pool, load_model_sync)
            state.current_model = model_id


# ============================================================================
# OpenAI-Compatible Endpoints
# ============================================================================

@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List all available models (OpenAI compatible)."""
    registered = ft.list_models()
    models = []
    for model_id in registered:
        models.append(ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="flashtensors"
        ))

    # Also add current loaded model info
    if state.current_model and state.current_model not in registered:
        models.append(ModelInfo(
            id=state.current_model,
            created=int(time.time()),
            owned_by="flashtensors"
        ))

    return ModelListResponse(data=models)


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str) -> ModelInfo:
    """Get specific model info (OpenAI compatible)."""
    registered = ft.list_models()
    if model_id not in registered:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return ModelInfo(
        id=model_id,
        created=int(time.time()),
        owned_by="flashtensors"
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming and vision (multimodal) inputs.
    """
    await ensure_model_loaded(request.model)

    # Format the prompt
    prompt = format_chat_prompt(request.messages, request.model)

    # Extract images if this is a vision model
    images = []
    if is_vision_model(request.model):
        images = await extract_images_from_messages(request.messages)

    # Build sampling params
    stop = request.stop if request.stop else None
    if isinstance(stop, str):
        stop = [stop]

    sampling_params = SamplingParams(
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        max_tokens=request.max_tokens or 512,
        stop=stop,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0,
        n=request.n or 1,
    )

    if request.stream:
        return StreamingResponse(
            stream_chat_completion(prompt, sampling_params, request.model, images),
            media_type="text/event-stream"
        )

    # Non-streaming response
    # Pass images via multi_modal_data for vision models
    if images:
        outputs = state.llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": images}}],
            sampling_params
        )
    else:
        outputs = state.llm.generate([prompt], sampling_params)
    output = outputs[0]

    choices = []
    for i, out in enumerate(output.outputs):
        choices.append(ChatCompletionChoice(
            index=i,
            message=ChatMessage(role="assistant", content=out.text),
            finish_reason=out.finish_reason or "stop"
        ))

    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = sum(estimate_tokens(c.message.content) for c in choices)

    return ChatCompletionResponse(
        id=generate_id("chatcmpl"),
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


async def stream_chat_completion(
    prompt: str,
    sampling_params: SamplingParams,
    model: str,
    images: List[Image.Image] = None
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses in SSE format."""
    request_id = generate_id("chatcmpl")
    created = int(time.time())

    # vLLM streaming - pass images via multi_modal_data if present
    if images:
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": images}}]
    else:
        inputs = [prompt]

    outputs = state.llm.generate(inputs, sampling_params, use_tqdm=False)

    for output in outputs:
        for i, out in enumerate(output.outputs):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": i,
                    "delta": {"content": out.text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk with finish_reason
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible text completions endpoint.
    Supports streaming.
    """
    await ensure_model_loaded(request.model)

    # Handle single or multiple prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # Build sampling params
    stop = request.stop if request.stop else None
    if isinstance(stop, str):
        stop = [stop]

    sampling_params = SamplingParams(
        temperature=request.temperature or 1.0,
        top_p=request.top_p or 1.0,
        max_tokens=request.max_tokens or 16,
        stop=stop,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0,
        n=request.n or 1,
    )

    if request.stream:
        return StreamingResponse(
            stream_completion(prompts[0], sampling_params, request.model),
            media_type="text/event-stream"
        )

    # Non-streaming response
    outputs = state.llm.generate(prompts, sampling_params)

    choices = []
    total_completion_tokens = 0

    for output in outputs:
        for i, out in enumerate(output.outputs):
            choices.append(CompletionChoice(
                text=out.text,
                index=len(choices),
                finish_reason=out.finish_reason or "stop"
            ))
            total_completion_tokens += estimate_tokens(out.text)

    prompt_tokens = sum(estimate_tokens(p) for p in prompts)

    return CompletionResponse(
        id=generate_id("cmpl"),
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=prompt_tokens + total_completion_tokens
        )
    )


async def stream_completion(
    prompt: str,
    sampling_params: SamplingParams,
    model: str
) -> AsyncGenerator[str, None]:
    """Stream text completion responses in SSE format."""
    request_id = generate_id("cmpl")
    created = int(time.time())

    outputs = state.llm.generate([prompt], sampling_params, use_tqdm=False)

    for output in outputs:
        for i, out in enumerate(output.outputs):
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{
                    "text": out.text,
                    "index": i,
                    "logprobs": None,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": "",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# FlashTensors Management Endpoints
# ============================================================================

@app.post("/v1/flashtensors/register/{model_id:path}")
async def register_model(model_id: str, force: bool = False):
    """
    Register a model with flashtensors.
    Downloads and converts to optimized format.
    """
    import concurrent.futures
    try:
        # Run in thread pool to avoid asyncio.run() conflict
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(
                pool,
                lambda: ft.register_model(model_id, backend="vllm", force=force)
            )
        return {"status": "success", "registered": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/flashtensors/load/{model_id:path}")
async def load_model(model_id: str):
    """
    Explicitly load a specific model.
    Use this to pre-load a model before making inference requests.
    """
    try:
        await ensure_model_loaded(model_id)
        return {"status": "success", "loaded": model_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/flashtensors/unload")
async def unload_model():
    """Unload current model and free GPU memory."""
    import gc
    async with state.lock:
        if state.llm is not None:
            del state.llm
            state.llm = None
            gc.collect()
            torch.cuda.empty_cache()
        state.current_model = None
    return {"status": "success", "message": "Model unloaded"}


@app.get("/v1/flashtensors/models")
async def list_registered_models():
    """List all models registered with flashtensors."""
    models = ft.list_models()
    return {
        "registered_models": models,
        "current_model": state.current_model
    }


@app.get("/v1/flashtensors/info/{model_id:path}")
async def get_model_info(model_id: str):
    """Get detailed info about a registered model."""
    try:
        info = ft.get_model_info(model_id)
        return {"model_id": model_id, "info": info}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/v1/flashtensors/status")
async def get_status():
    """Get current server status."""
    return {
        "status": "running",
        "current_model": state.current_model,
        "registered_models": list(ft.list_models().keys()) if ft.list_models() else []
    }


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
