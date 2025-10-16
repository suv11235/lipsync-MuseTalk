# MuseTalk Modal Deployment

Real-time lip-sync inference deployed on Modal GPU with FastAPI orchestrator.

## Features

- 🚀 GPU-accelerated inference on Modal cloud (T4)
- 🌐 FastAPI REST API for job submission
- 📦 Persistent storage via NetworkFileSystem
- ⚡ ~1.2s per frame processing speed
- 💰 Cost-efficient: ~$0.02 per 8-second video
- 📊 Comprehensive evaluation metrics (FID, CSIM, LSE-C)

## Architecture

```
Client → FastAPI (port 9000) → Modal Function (GPU) → MuseTalk → Result
```

- **Orchestrator**: FastAPI REST API for job management
- **Modal Backend**: GPU-accelerated inference (T4)
- **Storage**: NetworkFileSystem for persistent results
- **Processing**: ~4 minutes for 8-second video (200 frames)

---

## Prerequisites

- Python 3.10+
- Modal account (free tier works - sign up at https://modal.com)
- Git

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project
```

### 2. Install Modal CLI

```bash
pip install modal
```

### 3. Authenticate with Modal

```bash
modal token new
```

This will:
- Open your browser to authenticate
- Save credentials to `~/.modal/`
- Persist across sessions (one-time per machine)

### 4. Download MuseTalk Models

```bash
cd MuseTalk
bash download_weights.sh
```

**Note**: Downloads ~2GB of model weights. Models are not included in git due to size.

The script will download models to:
- `models/dwpose/`
- `models/face-parse-bisent/`
- `models/musetalk/`
- `models/sd-vae-ft-mse/`
- `models/whisper/`

### 5. Deploy to Modal

```bash
cd modal_app
modal deploy musetalk_modal.py
```

Expected output:
```
✓ Initialized
✓ Created objects
✓ App deployed! 🎉

View at: https://modal.com/apps/...
```

Verify deployment:
```bash
modal app list | grep musetalk-poc
```

### 6. (Optional) Setup Local Orchestrator

```bash
cd orchestrator
pip install -r requirements.txt
```

---

## Usage

### Option 1: Direct Modal Invocation

```bash
cd modal_app
modal run test_e2e.py
```

### Option 2: Via FastAPI Orchestrator

Start the orchestrator:
```bash
cd orchestrator
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000
```

Submit a job:
```bash
curl -X POST http://localhost:9000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "data/video/yongen.mp4",
    "audio_path": "data/audio/yongen.wav",
    "version": "v15"
  }'
```

Response:
```json
{
  "job_id": "abc123...",
  "status": "submitted"
}
```

Check job status:
```bash
curl http://localhost:9000/jobs/{job_id}
```

### Using Your Own Videos

1. Copy files to MuseTalk data directory:
   ```bash
   cp your_video.mp4 MuseTalk/data/video/
   cp your_audio.wav MuseTalk/data/audio/
   ```

2. Submit job:
   ```bash
   curl -X POST http://localhost:9000/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "video_path": "data/video/your_video.mp4",
       "audio_path": "data/audio/your_audio.wav",
       "version": "v15"
     }'
   ```

### Using URLs

```bash
curl -X POST http://localhost:9000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "audio_url": "https://example.com/audio.wav",
    "version": "v15"
  }'
```

---

## API Endpoints

### Health Check
```bash
GET /healthz
```

### Submit Job
```bash
POST /jobs
Content-Type: application/json

{
  "video_path": "data/video/example.mp4",  # OR video_url
  "audio_path": "data/audio/example.wav",  # OR audio_url
  "version": "v15"  # or "v1"
}
```

### Get Job Status
```bash
GET /jobs/{job_id}
```

Response:
```json
{
  "job_id": "abc123...",
  "status": "succeeded",
  "output_path": "/shared/results/{uuid}/v15/output.mp4",
  "error": null
}
```

### Get Job Result
```bash
GET /jobs/{job_id}/result
```

---

## Project Structure

```
project/
├── README.md                    # This file
├── orchestrator/                # FastAPI orchestrator
│   ├── app/
│   │   ├── main.py             # FastAPI app
│   │   ├── routes.py           # API endpoints
│   │   ├── models.py           # Pydantic models
│   │   └── workers.py          # Modal integration
│   ├── requirements.txt
│   └── README.md               # API documentation
├── modal_app/                   # Modal GPU backend
│   ├── musetalk_modal.py       # Modal app & function
│   ├── test_e2e.py             # E2E test
│   ├── requirements.txt
│   └── README.md               # Modal app docs
├── evals/                       # Evaluation metrics
│   ├── fid_metric.py           # FID (visual fidelity)
│   ├── csim_metric.py          # CSIM (identity preservation)
│   ├── lse_c_metric.py         # LSE-C (lip synchronization)
│   ├── evaluate.py             # Main evaluation script
│   ├── requirements.txt
│   └── README.md               # Evaluation docs
└── MuseTalk/                    # MuseTalk inference code
    ├── scripts/inference.py
    ├── configs/
    ├── data/                    # Test assets
    └── models/                  # Model weights (download via script)
```

---

## Modal Commands

```bash
# List apps
modal app list

# View logs
modal app logs musetalk-poc

# Deploy updates
cd modal_app
modal deploy musetalk_modal.py

# Run E2E test
modal run test_e2e.py
```

---

## Performance

**Benchmark (200 frames, 8-second video)**:
- Processing: ~1.17s per frame
- Total time: ~4 minutes
- GPU: T4
- Cost: ~$0.02 per job

---

## Important Notes

### What's in Git
- ✅ All code (orchestrator, modal_app, MuseTalk scripts)
- ✅ Configuration files
- ✅ Documentation
- ❌ Model weights (~2GB) - download via `download_weights.sh`
- ❌ Virtual environments - recreate with `pip install`
- ❌ Test videos/audio files

### Modal App Persistence
- ✅ **Modal app stays deployed** even when you turn off your machine
- ✅ Available 24/7 from Modal cloud
- ✅ No charges when idle (only pay per job)
- ✅ Access from any machine after authentication

### Cost Structure
- **Deployment**: Free (app stays deployed)
- **Idle**: $0 (no charges when not in use)
- **Usage**: ~$0.02 per 8-second video (T4 GPU @ ~4 minutes)

---

## Troubleshooting

### Modal Authentication Failed
```bash
modal token new
```

### Models Not Found
```bash
cd MuseTalk
bash download_weights.sh
```

### Deployment Failed
```bash
# Update Modal CLI
pip install --upgrade modal

# Redeploy
cd modal_app
modal deploy musetalk_modal.py
```

### Orchestrator Won't Start
```bash
# Kill existing process
pkill -f "uvicorn app.main:app"

# Start fresh
cd orchestrator
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000
```

### Job Stuck in "queued"
- This is normal! Processing takes ~4 minutes
- Watch progress: `modal app logs musetalk-poc`

### App Not Found
```bash
# List all deployed apps
modal app list

# Should see: musetalk-poc (deployed)
```

---

## Evaluation Metrics

The `evals/` folder contains implementations of standard lip-sync quality metrics:

- **FID (Frechet Inception Distance)**: Measures visual fidelity
- **CSIM (Cosine Similarity)**: Measures identity preservation  
- **LSE-C (Lip-Sync Error Confidence)**: Measures lip synchronization quality

### Quick Start

```bash
cd evals
pip install -r requirements.txt

# Run evaluation on generated video
python evaluate.py \
  --source_video ../MuseTalk/data/video/yongen.mp4 \
  --generated_video path/to/your/generated_output.mp4 \
  --audio ../MuseTalk/data/audio/yongen.wav \
  --output_json results.json
```

For detailed documentation, see `evals/README.md`.

---

## Development

For detailed API documentation, see:
- `orchestrator/README.md` - API reference
- `modal_app/README.md` - Modal deployment details
- `evals/README.md` - Evaluation metrics guide

---

## License

See LICENSE file in the MuseTalk directory.

---

