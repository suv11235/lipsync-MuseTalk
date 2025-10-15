# MuseTalk Modal GPU App (POC)

This is the Modal GPU inference component for running MuseTalk lip-sync generation on cloud GPUs.

## What It Does

- Wraps MuseTalk inference in a Modal function with GPU (T4) acceleration
- Accepts video/image + audio inputs and returns lip-synced output videos
- Bakes the entire MuseTalk repo and model weights into a container image
- Uses Modal NetworkFileSystem for persistent storage of results

## Components

- `musetalk_modal.py`: Main Modal app with `run_inference` function
- `requirements.txt`: Modal client SDK dependencies

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Authenticate with Modal:
   ```bash
   modal setup
   ```

3. Ensure MuseTalk weights exist in `../MuseTalk/models/` (see MuseTalk README for download)

## Deployment

Deploy the app to Modal:
```bash
modal deploy musetalk_modal.py
```

## Usage

### Direct Invocation (from Python)

```python
import modal

app = modal.App.lookup("musetalk-poc")

with app.run():
    from musetalk_modal import run_inference
    
    result = run_inference.remote(
        video_path="data/video/yongen.mp4",
        audio_path="data/audio/ted_clip_30sec_audio.wav",
        version="v15"
    )
    print(result)
```

### Function Parameters

- `video_path` or `image_url`: Path to input video/image (relative to MuseTalk repo) or HTTP URL
- `audio_path` or `audio_url`: Path to audio file (relative to MuseTalk repo) or HTTP URL
- `version`: Model version (`"v1"` or `"v15"`, default: `"v15"`)
- `use_float16`: Use FP16 for faster inference (default: `True`)

### Output

Returns a dict with:
```python
{
    "job_id": "unique-uuid",
    "status": "succeeded" | "failed",
    "output_path": "/shared/results/<job_id>/v15/output.mp4",
    "error": "..." (if failed)
}
```

## Architecture

1. **Image Build**: Copies MuseTalk repo + models into a Debian slim + CUDA container
2. **Inference**: Downloads inputs (if URLs), generates YAML config, runs `python -m scripts.inference`
3. **Storage**: Writes outputs to Modal NetworkFileSystem at `/shared/results/<job_id>/`

## Notes

- First run will take ~15 minutes to build the image (includes large model downloads)
- Subsequent runs use the cached image and are much faster
- GPU inference typically takes 2-5 minutes for a 30-second clip





