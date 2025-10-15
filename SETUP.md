# Setup Instructions

## Prerequisites

- Python 3.10+
- Modal account (free tier works)
- Git

## Initial Setup (First Time)

### 1. Clone the Repository

```bash
git clone git@github.com:suv11235/lipsync-MuseTalk.git
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
- Persist across sessions (you only need to do this once per machine)

### 4. Setup MuseTalk

```bash
cd MuseTalk

# Download model weights (one-time, ~2GB)
bash download_weights.sh

# The models will be downloaded to:
# - models/dwpose/
# - models/face-parse-bisent/
# - models/musetalk/
# - models/sd-vae-ft-mse/
# - models/whisper/
```

**Note**: Model files are ~2GB and NOT committed to git. You must download them.

### 5. Setup Orchestrator (Optional - for local API)

```bash
cd orchestrator
pip install -r requirements.txt
```

## Deploy to Modal

### Deploy the GPU Function

```bash
cd modal_app
modal deploy musetalk_modal.py
```

**Output:**
```
✓ Initialized
✓ Created objects
✓ App deployed! 🎉

View at: https://modal.com/apps/...
```

### Verify Deployment

```bash
# Check app is deployed
modal app list | grep musetalk-poc

# Run end-to-end test
modal run test_e2e.py
```

## Using the Deployed App

### Option 1: Direct Modal Invocation

```bash
cd modal_app
modal run test_e2e.py
```

### Option 2: Via Orchestrator API

```bash
# Start orchestrator
cd orchestrator
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000 &

# Submit job
curl -X POST http://localhost:9000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "data/video/yongen.mp4",
    "audio_path": "data/audio/yongen.wav",
    "version": "v15"
  }'
```

## Important: What Persists

### ✅ Persists (Safe to Turn Off Workspace)

1. **Modal Deployment**
   - Your app stays deployed on Modal
   - Functions are available 24/7
   - No need to keep workspace running

2. **Modal Credentials**
   - Stored in `~/.modal/` on your machine
   - Persists across sessions
   - Just run `modal token new` once per machine

3. **Code in Git**
   - All your code (orchestrator, modal_app)
   - Configuration files
   - Documentation

### ❌ Does NOT Persist (Need to Recreate)

1. **Model Weights** (~2GB)
   - NOT in git (too large)
   - Must download via `download_weights.sh`
   - One-time setup per machine

2. **Virtual Environments**
   - `.venv` folders are local
   - Recreate with `pip install -r requirements.txt`

3. **Running Orchestrator**
   - Local FastAPI process
   - Restart with `uvicorn app.main:app`

## Reproducing on New Machine

### Quick Setup

```bash
# 1. Clone repo
git clone <your-repo-url>
cd project

# 2. Install Modal and authenticate
pip install modal
modal token new

# 3. Download MuseTalk models
cd MuseTalk
bash download_weights.sh
cd ..

# 4. Deploy to Modal
cd modal_app
modal deploy musetalk_modal.py

# 5. (Optional) Setup orchestrator
cd orchestrator
pip install -r requirements.txt
```

**That's it!** Your Modal app is deployed and ready to use.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Modal (Cloud - Always Running)                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  musetalk-poc (deployed app)                          │  │
│  │  - Function: run_inference                            │  │
│  │  - GPU: T4                                            │  │
│  │  - Storage: NetworkFileSystem (/shared)              │  │
│  │  - Status: ✅ Deployed (persists when you turn off)  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ modal.Function.from_name()
                            │
┌─────────────────────────────────────────────────────────────┐
│  Your Machine (Local - Start when needed)                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Option 1: Direct Modal CLI                          │  │
│  │    $ modal run test_e2e.py                           │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Option 2: FastAPI Orchestrator (port 9000)          │  │
│  │    $ uvicorn app.main:app --host 0.0.0.0 --port 9000 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Modal Costs

- **Deployment**: Free (app stays deployed)
- **Usage**: ~$0.02 per 8-second video (T4 GPU @ ~4 minutes)
- **Idle**: $0 (only pay when function runs)

## Troubleshooting

### Modal Authentication Failed

```bash
# Re-authenticate
modal token new
```

### Models Not Found

```bash
cd MuseTalk
bash download_weights.sh
```

### Deployment Failed

```bash
# Check Modal CLI is up to date
pip install --upgrade modal

# Redeploy
cd modal_app
modal deploy musetalk_modal.py
```

### App Not Found

```bash
# List all apps
modal app list

# Should see: musetalk-poc (deployed)
```

## Files to Commit to Git

✅ **DO commit:**
- `orchestrator/` (all files)
- `modal_app/` (all files)
- `MuseTalk/configs/`
- `MuseTalk/scripts/`
- `MuseTalk/musetalk/`
- `README.md`, `SETUP.md`
- `.gitignore`

❌ **DON'T commit:**
- `MuseTalk/models/` (too large, 2GB)
- `MuseTalk/.venv/`
- `MuseTalk/data/video/*.mp4`
- `MuseTalk/data/audio/*.wav`
- Any `.pyc` or `__pycache__`

## Next Steps After Cloning

1. Run `modal token new` (one-time per machine)
2. Download models with `download_weights.sh` (one-time per machine)
3. Deploy with `modal deploy musetalk_modal.py` (updates your cloud deployment)
4. Use the app!

---

**Key Point**: Once deployed to Modal, your app runs in the cloud. You can turn off your workspace/machine and the Modal app stays running. Just authenticate on your new machine and you're good to go!

