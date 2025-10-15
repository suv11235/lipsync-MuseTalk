# MuseTalk Modal Deployment

Real-time lip-sync inference deployed on Modal GPU with FastAPI orchestrator.

## Quick Start

### 1. Start the Orchestrator
```bash
cd /home/ubuntu/project/orchestrator
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000 &
```

### 2. Submit a Job
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
  "job_id": "47dbb32d-6b1b-4dc3-807a-d4d1a35e5e5c",
  "status": "submitted"
}
```

### 3. Check Status
```bash
curl http://localhost:9000/jobs/{job_id}
```

## Architecture

```
Client → FastAPI (9000) → Modal Function (GPU) → MuseTalk → Result
```

- **Orchestrator**: FastAPI REST API for job management
- **Modal Backend**: GPU-accelerated inference (T4)
- **Storage**: NetworkFileSystem for persistent results
- **Processing**: ~4 minutes for 8-second video (200 frames)

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
  "version": "v15"
}
```

### Get Job Status
```bash
GET /jobs/{job_id}
```

### Get Job Result
```bash
GET /jobs/{job_id}/result
```

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
│   └── README.md
└── MuseTalk/                    # MuseTalk inference code
    ├── scripts/inference.py
    ├── configs/
    └── data/                    # Test assets
```

## Performance

**Yongen Test (200 frames)**:
- Processing: ~1.17s per frame
- Total time: ~4 minutes
- GPU: T4
- Cost: ~$0.02 per job

## Modal Commands

```bash
# List apps
python3 -m modal app list

# View logs
python3 -m modal app logs musetalk-poc

# Deploy updates
cd modal_app
python3 -m modal deploy musetalk_modal.py

# Run E2E test
python3 -m modal run test_e2e.py
```

## Troubleshooting

**Orchestrator won't start**:
```bash
pkill -f "uvicorn app.main:app"
cd /home/ubuntu/project/orchestrator
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000
```

**Job stuck in "queued"**:
- Normal! Processing takes ~4 minutes
- Watch progress: `python3 -m modal app logs musetalk-poc`

**Modal function not found**:
```bash
cd modal_app
python3 -m modal deploy musetalk_modal.py
```

## Development

### Using Your Own Videos

1. Copy files to MuseTalk data directory:
   ```bash
   cp your_video.mp4 MuseTalk/data/video/
   cp your_audio.wav MuseTalk/data/audio/
   ```

2. Submit job with new paths:
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

## Setup on New Machine

See **[SETUP.md](SETUP.md)** for detailed instructions.

**Quick start:**
```bash
git clone <your-repo-url>
cd project
pip install modal
modal token new                           # Authenticate with Modal
cd MuseTalk && bash download_weights.sh   # Download models (~2GB)
cd ../modal_app
modal deploy musetalk_modal.py            # Deploy to Modal cloud
```

## Important Notes

### What's in Git
- ✅ All code (orchestrator, modal_app, MuseTalk/scripts)
- ✅ Configuration and documentation
- ❌ Model weights (~2GB, download via `download_weights.sh`)
- ❌ Virtual environments (recreate with `pip install`)
- ❌ Modal credentials (authenticate per machine)

### Modal App Persistence
- ✅ **Modal app stays deployed** even when you turn off your machine
- ✅ Available 24/7 from Modal cloud
- ✅ No charges when idle (only pay per job: ~$0.02/video)
- ✅ Access from any machine after `modal token new`

## Status

✅ **PRODUCTION READY**

- Orchestrator: Running
- Modal Backend: Deployed
- End-to-end: Tested and verified
- Performance: ~1.2s/frame on T4 GPU

## Links

- **Modal Dashboard**: https://modal.com/apps
- **Setup Guide**: [SETUP.md](SETUP.md)
- **API Docs**: [orchestrator/README.md](orchestrator/README.md)
- **Modal App**: [modal_app/README.md](modal_app/README.md)

