# MuseTalk Orchestrator API (POC)

FastAPI-based REST API for submitting jobs to the Modal GPU inference backend.

## What It Does

- Provides HTTP endpoints to submit MuseTalk inference jobs
- Dispatches jobs to Modal GPU workers asynchronously
- Tracks job status in-memory (POC only; resets on restart)
- Returns job results including output artifact paths

## Components

- `app/main.py`: FastAPI app entrypoint
- `app/models.py`: Pydantic request/response schemas
- `app/routes.py`: API route handlers (`/jobs`, `/jobs/{id}`, `/jobs/{id}/result`)
- `app/workers.py`: Modal client integration and in-memory job state
- `requirements.txt`: FastAPI + Modal client dependencies

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure Modal is authenticated:
   ```bash
   modal setup
   ```

3. Deploy the Modal GPU app first (see `../modal_app/README.md`)

## Running the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## API Endpoints

### `POST /jobs`

Submit a new inference job.

**Request Body (with URLs):**
```json
{
  "video_url": "https://example.com/video.mp4",  // OR "image_url"
  "audio_url": "https://example.com/audio.wav",
  "version": "v15"  // optional, defaults to "v15"
}
```

**Request Body (with local paths - for testing):**
```json
{
  "video_path": "data/video/yongen.mp4",
  "audio_path": "data/audio/yongen.wav",
  "version": "v15"
}
```

**Example curl command:**
```bash
curl -X POST http://localhost:9000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "data/video/yongen.mp4",
    "audio_path": "data/audio/yongen.wav",
    "version": "v15"
  }'
```

**Response:**
```json
{
  "job_id": "abc123...",
  "status": "submitted"
}
```

### `GET /jobs/{job_id}`

Get job status.

**Response:**
```json
{
  "job_id": "abc123...",
  "status": "running",  // or "succeeded", "failed"
  "output_path": "/shared/results/abc123.../v15/output.mp4",  // if succeeded
  "error": "..."  // if failed
}
```

### `GET /jobs/{job_id}/result`

Get the output artifact path once job completes.

**Response:**
```json
{
  "output_path": "/shared/results/abc123.../v15/output.mp4"
}
```

### `GET /healthz`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Architecture

1. **Job Submission**: API accepts input URLs/paths, generates a job ID, spawns a Modal function
2. **Status Tracking**: Stores job metadata in an in-memory dict (`workers.JOB_STORE`)
3. **Background Polling**: A daemon thread joins the Modal invocation and updates job status
4. **Result Retrieval**: Returns the output path from the Modal NetworkFileSystem

## Limitations (POC)

- **In-Memory State**: Job state is not persisted; server restart loses all job history
- **No Authentication**: Public endpoints; add API keys for production
- **No Rate Limiting**: Can be overwhelmed by too many concurrent requests
- **No File Serving**: Returns paths only; external file server or presigned URLs needed for downloads

## Next Steps (Beyond POC)

- Persist job state to SQLite or Redis
- Add authentication (API keys, OAuth)
- Implement rate limiting and request validation
- Serve output files via presigned URLs or direct streaming
- Add webhook/callback support for job completion notifications


