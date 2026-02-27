# Vision Transformer Project — Implementation Guide

This guide covers everything remaining after the initial `VisionManager` class. Work through each section in order — each builds on the previous one.

---

## 1. Project Structure Overview

```
VisionTransformerProject/
├── .gitignore
├── .env.example              # Template for environment variables
├── .dockerignore
├── README.md
├── PROJECT_GUIDE.md          # This file
├── requirements.txt
├── Dockerfile
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app + lifespan
│   ├── config.py             # ✅ Created
│   ├── models/
│   │   ├── __init__.py
│   │   └── vision_manager.py # ✅ Created
│   ├── services/
│   │   ├── __init__.py
│   │   └── llm_service.py    # Anthropic Claude integration
│   ├── utils/
│   │   ├── __init__.py
│   │   └── blob_serializer.py # Custom binary format
│   └── routers/
│       ├── __init__.py
│       └── classification.py  # API route handlers
└── tests/
    ├── __init__.py
    ├── test_blob_serializer.py
    ├── test_vision_manager.py
    └── test_api.py
```

---

## 2. Blob Serializer (`app/utils/blob_serializer.py`)

A custom binary serialization format for packaging classification results with optional image data. This is a learning exercise in working with raw bytes.

### Binary Format Spec

```
┌──────────────────────────────────────────────┐
│  Magic Bytes:   0x56 0x49 0x54 ("VIT")       │  3 bytes
│  Version:       0x01                          │  1 byte
│  Timestamp:     int64 (Unix epoch ms)         │  8 bytes
│  JSON Length:   uint32 (big-endian)           │  4 bytes
│  JSON Payload:  UTF-8 encoded JSON            │  variable
│  Image Flag:    0x00 (none) or 0x01 (present) │  1 byte
│  Image Length:  uint32 (big-endian)           │  4 bytes (if flag=0x01)
│  Image Data:    raw bytes (JPEG/PNG)          │  variable (if flag=0x01)
└──────────────────────────────────────────────┘
```

### Functions to Implement

| Function | Description |
|----------|-------------|
| `serialize(data: dict, image_bytes: bytes \| None) -> bytes` | Pack a prediction result dict and optional image into the binary format |
| `deserialize(blob: bytes) -> tuple[dict, bytes \| None]` | Unpack a blob back into (dict, optional image bytes) |

### Key Details
- Use Python's `struct` module for packing/unpacking (`>I` for uint32, `>q` for int64)
- Validate magic bytes on deserialization — raise `ValueError` if they don't match
- Timestamp is `int(time.time() * 1000)` (milliseconds since epoch)
- The JSON payload holds the prediction result from `VisionManager.predict()`

---

## 3. LLM Service (`app/services/llm_service.py`)

Integrates with the Anthropic API (Claude) to generate natural-language descriptions from classification labels.

### Class: `LLMService`

```python
class LLMService:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        # Store anthropic.Anthropic client

    async def describe(self, predictions: list[dict]) -> str:
        # Send predictions to Claude, return a natural-language description
```

### Details
- Use the `anthropic` Python SDK
- The prompt should ask Claude to describe what's in the image based on the top classification labels and their confidence scores
- Return the text content from the response
- The API key comes from the environment variable `ANTHROPIC_API_KEY`

### Config Additions (`app/config.py`)
Add these settings:
```python
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
```

---

## 4. FastAPI Endpoints (`app/routers/classification.py`)

Two endpoints that form the core API.

### `POST /upload`
Accepts an image, classifies it, and returns results serialized as a blob.

```
Request:  multipart/form-data with field "file" (image)
Response: application/octet-stream (blob bytes)
```

**Flow:**
1. Read uploaded file bytes → open with `PIL.Image`
2. Call `VisionManager.predict(image)`
3. Call `blob_serializer.serialize(result, image_bytes)`
4. Return blob as streaming response

### `POST /describe`
Accepts an image, classifies it, then asks Claude to describe the result.

```
Request:  multipart/form-data with field "file" (image)
Response: application/json
{
    "predictions": [...],
    "description": "A natural language description...",
    "model": "google/vit-base-patch16-224"
}
```

**Flow:**
1. Read uploaded file bytes → open with `PIL.Image`
2. Call `VisionManager.predict(image)`
3. Call `LLMService.describe(predictions)`
4. Return combined JSON response

### Shared Concerns
- Both endpoints need access to `VisionManager` and (for `/describe`) `LLMService` — use FastAPI dependency injection via `app.state`
- Validate that the uploaded file is an image (check content type)
- Return proper HTTP error codes (400 for bad input, 500 for model errors)

---

## 5. Application Entry Point (`app/main.py`)

The FastAPI application with lifespan-based model loading.

### Lifespan Pattern

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load models once, attach to app.state
    app.state.vision_manager = VisionManager()
    app.state.llm_service = LLMService(api_key=ANTHROPIC_API_KEY)
    yield
    # SHUTDOWN: Cleanup (if needed)

app = FastAPI(title="Vision Transformer API", lifespan=lifespan)
app.include_router(classification.router)
```

### Why Lifespan?
- The ViT model is large (~350MB) — loading it per-request would be extremely slow
- Lifespan ensures the model loads once at startup and stays in memory
- `app.state` makes it accessible to all route handlers via dependency injection

---

## 6. Project Configuration Files

### `requirements.txt`
```
torch
transformers
pillow
fastapi
uvicorn[standard]
python-multipart
anthropic
python-dotenv
```

### `.env.example`
```
ANTHROPIC_API_KEY=your-api-key-here
```

### `.gitignore`
```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.venv/
```

---

## 7. Docker Setup

### `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `.dockerignore`
```
.env
.git
__pycache__
*.pyc
.venv
```

### Considerations
- The image will be large due to PyTorch (~2GB+). For production, consider `torch` CPU-only builds to reduce size.
- Model weights are downloaded on first run. For faster container starts, you could bake them into the image during build.

---

## 8. Tests

### `tests/test_blob_serializer.py`
- Test round-trip: `deserialize(serialize(data, img)) == (data, img)`
- Test with `None` image data
- Test invalid magic bytes raises `ValueError`
- Test that timestamp is within a reasonable range

### `tests/test_vision_manager.py`
- Test that model loads and properties are correct
- Test `predict()` returns expected shape (requires a sample image)
- Test `top_k` parameter limits results
- Note: These tests require model download (~350MB on first run)

### `tests/test_api.py`
- Use `httpx.AsyncClient` with FastAPI's `TestClient`
- Test `/upload` with a valid image returns bytes
- Test `/describe` with a valid image returns JSON with expected keys
- Test invalid file type returns 400

### Running Tests
```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## 9. Data Flow Diagrams

### `/upload` Endpoint
```
Client                    FastAPI                VisionManager        BlobSerializer
  │                         │                        │                     │
  │── POST /upload ────────▶│                        │                     │
  │   (image file)          │── predict(image) ─────▶│                     │
  │                         │◀── predictions ────────│                     │
  │                         │── serialize(pred, img)─────────────────────▶│
  │                         │◀── blob bytes ─────────────────────────────│
  │◀── octet-stream ───────│                        │                     │
```

### `/describe` Endpoint
```
Client                    FastAPI                VisionManager        LLMService
  │                         │                        │                    │
  │── POST /describe ──────▶│                        │                    │
  │   (image file)          │── predict(image) ─────▶│                    │
  │                         │◀── predictions ────────│                    │
  │                         │── describe(preds) ─────────────────────────▶│
  │                         │◀── description text ───────────────────────│
  │◀── JSON response ──────│                        │                    │
```

---

## Recommended Build Order

1. **Blob Serializer** — pure Python, no dependencies beyond stdlib, easy to test
2. **LLM Service** — requires Anthropic API key, small and self-contained
3. **FastAPI routes + main.py** — wires everything together
4. **Tests** — validate all components
5. **Docker + config files** — packaging for deployment

Each step is independent enough to review before moving on.
