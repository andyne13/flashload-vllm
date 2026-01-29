# FlashTensors + vLLM: Fast Model Loading API Server

An OpenAI-compatible API server that integrates [flashtensors](https://github.com/leoheuler/flashtensors) with [vLLM](https://github.com/vllm-project/vllm) for ultra-fast LLM model loading and switching. Models are converted once into flashtensors' optimized binary format, then loaded from local storage in seconds instead of minutes.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Docker Container (vllm/vllm-openai:v0.10.2)        │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  FastAPI Server (entrypoint.py, port 8000)    │  │
│  │                                               │  │
│  │  OpenAI-Compatible Endpoints                  │  │
│  │    /v1/chat/completions                       │  │
│  │    /v1/completions                            │  │
│  │    /v1/models                                 │  │
│  │                                               │  │
│  │  FlashTensors Management Endpoints            │  │
│  │    /v1/flashtensors/register/{model_id}       │  │
│  │    /v1/flashtensors/load/{model_id}           │  │
│  │    /v1/flashtensors/unload                    │  │
│  │    /v1/flashtensors/models                    │  │
│  │    /v1/flashtensors/status                    │  │
│  └───────────┬───────────────────────────────────┘  │
│              │                                      │
│  ┌───────────▼───────────────────────────────────┐  │
│  │  flashtensors                                 │  │
│  │  - Converts HF models to optimized format     │  │
│  │  - Patches vLLM model loader                  │  │
│  │  - 20GB memory pool, 32MB chunk I/O           │  │
│  └───────────┬───────────────────────────────────┘  │
│              │                                      │
│  ┌───────────▼───────────────────────────────────┐  │
│  │  vLLM (v0.10.2, V0 engine)                    │  │
│  │  - Inference engine                           │  │
│  │  - SamplingParams, LLM interface              │  │
│  └───────────┬───────────────────────────────────┘  │
│              │                                      │
│  ┌───────────▼───────────────────────────────────┐  │
│  │  GPU (NVIDIA, e.g. L4 24GB)                   │  │
│  │  - 80% VRAM utilization                       │  │
│  │  - Single GPU (tensor_parallel_size=1)        │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

Volumes:
  ./models:/models             ← Persistent flashtensors model storage
  /dev/shm:/dev/shm            ← Shared memory for fast I/O
  ~/.cache/huggingface:/root/… ← HuggingFace download cache
```

## Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support (tested on L4 24GB)
- Sufficient disk space for model storage

## Quick Start

### 1. Build and start the server

```bash
docker compose up -d --build
```

### 2. Register a model (one-time, downloads and converts)

```bash
curl -X POST http://localhost:8000/v1/flashtensors/register/Qwen/Qwen3-4B
```

### 3. Load the model into GPU memory

```bash
curl -X POST http://localhost:8000/v1/flashtensors/load/Qwen/Qwen3-4B
```

### 4. Use the model (OpenAI-compatible API)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

The model is auto-loaded on first request if not already loaded.

## API Reference

### OpenAI-Compatible Endpoints

#### `POST /v1/chat/completions`

Chat completion. Supports streaming via `"stream": true`. Supports multimodal (vision) inputs for VL models.

| Parameter         | Type         | Default  | Description                        |
| ----------------- | ------------ | -------- | ---------------------------------- |
| model             | string       | required | Model ID (must be registered)      |
| messages          | array        | required | Chat messages (`role` + `content`) |
| temperature       | float        | 0.7      | Sampling temperature               |
| top_p             | float        | 1.0      | Nucleus sampling                   |
| max_tokens        | int          | 512      | Max tokens to generate             |
| stream            | bool         | false    | Stream response via SSE            |
| stop              | string/array | null     | Stop sequences                     |
| n                 | int          | 1        | Number of completions              |
| presence_penalty  | float        | 0.0      | Presence penalty                   |
| frequency_penalty | float        | 0.0      | Frequency penalty                  |

**Vision model example (Qwen3-VL):**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]
    }]
  }'
```

Images can also be passed as base64 data URIs: `data:image/jpeg;base64,...`

### OpenAI-Compatible Endpoints

#### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint

#### `GET /v1/models`

List all registered models.

#### `GET /v1/models/{model_id}`

Get info about a specific model.

### FlashTensors Management Endpoints

#### `POST /v1/flashtensors/register/{model_id}`

Register a model: download from HuggingFace and convert to flashtensors format. Query parameter `?force=true` to re-register.

#### `POST /v1/flashtensors/load/{model_id}`

Pre-load a model into GPU memory.

#### `POST /v1/flashtensors/unload`

Unload the current model and free GPU memory.

#### `GET /v1/flashtensors/models`

List registered models and current loaded model.

#### `GET /v1/flashtensors/info/{model_id}`

Get detailed info about a registered model.

#### `GET /v1/flashtensors/status`

Server status including current model and registered models.

### Health Check

`GET /health` or `GET /v1/health` -- returns `{"status": "healthy"}`.

## Configuration

### Environment Variables

Set in `docker-compose.yml`:

| Variable                       | Default | Description                                   |
| ------------------------------ | ------- | --------------------------------------------- |
| `CUDA_VISIBLE_DEVICES`         | `0`     | GPU device index                              |
| `VLLM_USE_V1`                  | `0`     | Must be `0` for flashtensors compatibility    |
| `VLLM_MAX_MODEL_LEN`           | `4096`  | Max context window (reduce for larger models) |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Process spawning method                       |
| `HF_TOKEN`                     | (unset) | HuggingFace token for gated/private models    |

### FlashTensors Configuration (in `entrypoint.py`)

| Parameter                | Value     | Description                       |
| ------------------------ | --------- | --------------------------------- |
| `storage_path`           | `/models` | Where optimized models are stored |
| `mem_pool_size`          | 20 GB     | Memory pool size (L4 has 24GB)    |
| `chunk_size`             | 32 MB     | I/O chunk size                    |
| `gpu_memory_utilization` | 0.80      | Fraction of GPU VRAM to use       |

## Model Workflow

```
Register (one-time)          Load (seconds)              Inference
┌─────────────┐          ┌─────────────────┐         ┌──────────────┐
│ HuggingFace │  ──────► │ flashtensors    │ ──────► │ vLLM Engine  │
│ Download    │  convert │ /models/ (disk) │  load   │ (GPU VRAM)   │
│ (minutes)   │          │                 │         │              │
└─────────────┘          └─────────────────┘         └──────────────┘
                         Persistent storage           Ready for API
```

**Model switching:** To switch to a different (already registered) model, simply send a request with a different `model` parameter. The server unloads the current model and loads the new one automatically.

## Build Patches

The Dockerfile applies several patches to flashtensors for compatibility:

1. **Python 3.12 support** -- flashtensors' CMake config only lists 3.8--3.11; patched to include 3.12.
2. **Process group destruction** -- `dist.destroy_process_group()` in `flashtensors/api.py` is replaced with `pass`. Without this patch, vLLM cannot reload models after the first unload.
3. **Supervisord config** -- `%(ENV_USER)s` changed to `root`, `python` changed to `python3`.

## Known Limitations

- **Streaming is batch-mode:** The streaming endpoints (`stream: true`) use vLLM's synchronous `generate()`, which returns all tokens at once. Responses are sent as SSE but not truly token-by-token. True streaming requires vLLM's `AsyncLLMEngine`.
- **Single model at a time:** Only one model can be loaded in GPU memory. Switching models requires unloading the current one.
- **Single GPU:** Configured for `tensor_parallel_size=1`. Multi-GPU requires adjusting this value and related configuration.
- **No authentication:** The API has no auth. Secure it via network controls or a reverse proxy if exposed beyond localhost.
- **vLLM V0 engine required:** Uses `VLLM_USE_V1=0` to force the legacy engine, which flashtensors requires.

## Project Structure

```
flashtensors/
├── entrypoint.py          # FastAPI server with flashtensors + vLLM integration
├── Dockerfile             # Container build with patches for compatibility
├── docker-compose.yml     # Orchestration with GPU, volumes, health checks
└── models/                # Persistent storage for converted models (created at runtime)
```

## Acknowledgements

This project is built on top of [flashtensors](https://github.com/leoheuler/flashtensors) -- a high-performance tensor serialization library that makes near-instant model loading possible. Great work on making LLM deployment significantly faster.
