import os
import subprocess
import tempfile
import uuid
from typing import Dict, Optional

import modal
import requests

# Modal app and persisted storage
app = modal.App("musetalk-poc")
# Use NetworkFileSystem for persisted artifacts
nfs = modal.NetworkFileSystem.from_name("musetalk-shared", create_if_missing=True)

# Paths
import pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
LOCAL_MUSETALK_DIR = str(SCRIPT_DIR.parent / "MuseTalk")
REMOTE_MUSETALK_DIR = "/root/MuseTalk"
REMOTE_RESULTS_DIR = "/shared/results"

# Base image: Modal debian slim + system deps + python deps, and include MuseTalk source
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    # Bake source into the image with copy=True to allow subsequent build steps
    .add_local_dir(LOCAL_MUSETALK_DIR, REMOTE_MUSETALK_DIR, copy=True)
    .pip_install("requests", "PyYAML")
    .run_commands(
        [
            "python3 -m pip install --upgrade pip",
            # Install torch/torchvision/torchaudio with CUDA 11.7 (SUCCESSFUL VERSION)
            "python3 -m pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117",
            # Install MuseTalk's requirements
            f"python3 -m pip install -r {REMOTE_MUSETALK_DIR}/requirements.txt",
            # Install mmcv ecosystem (EXACT working versions from local venv)
            "python3 -m pip install openmim",
            "python3 -m mim install mmengine",
            "python3 -m mim install 'mmcv>=2.0.0,<2.2.0'",  # Must be <2.2.0
            "python3 -m mim install 'mmdet>=3.0.0'",
            "python3 -m mim install 'mmpose>=1.0.0'",
            # Pre-download face detection models to avoid runtime downloads
            "mkdir -p /root/.cache/torch/hub/checkpoints",
            "python3 -c 'import torch; torch.hub.download_url_to_file(\"https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth\", \"/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth\")'",
        ]
    )
)


def _download(url: str, dst_path: str) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(REMOTE_MUSETALK_DIR, p)


@app.function(
    image=image,
    gpu="T4",
    network_file_systems={"/shared": nfs},
    timeout=60 * 60,  # 60 minutes for longer videos and slow landmark extraction
)
def run_inference(
    *,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    video_url: Optional[str] = None,
    image_url: Optional[str] = None,
    audio_url: Optional[str] = None,
    version: str = "v15",
    use_float16: bool = True,
) -> Dict[str, str]:
    os.chdir(REMOTE_MUSETALK_DIR)

    job_id = str(uuid.uuid4())
    workdir = f"/tmp/job_{job_id}"
    os.makedirs(workdir, exist_ok=True)

    if video_path:
        input_video_path = _resolve_path(video_path)
    elif video_url:
        input_video_path = os.path.join(workdir, "input_video.mp4")
        _download(video_url, input_video_path)
    elif image_url:
        input_video_path = os.path.join(workdir, "input_image.png")
        _download(image_url, input_video_path)
    else:
        raise ValueError("Provide video_path or (video_url/image_url)")

    if audio_path:
        input_audio_path = _resolve_path(audio_path)
    elif audio_url:
        input_audio_path = os.path.join(workdir, "input_audio.wav")
        _download(audio_url, input_audio_path)
    else:
        raise ValueError("Provide audio_path or audio_url")

    job_results_dir = os.path.join(REMOTE_RESULTS_DIR, job_id)
    os.makedirs(job_results_dir, exist_ok=True)

    import yaml

    task = {
        "task1": {
            "video_path": input_video_path,
            "audio_path": input_audio_path,
            "result_name": "output.mp4",
        }
    }

    cfg_path = os.path.join(workdir, "inference.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(task, f)

    if version == "v15":
        unet_model_path = f"{REMOTE_MUSETALK_DIR}/models/musetalkV15/unet.pth"
        unet_config = f"{REMOTE_MUSETALK_DIR}/models/musetalkV15/musetalk.json"
        version_arg = "v15"
    else:
        unet_model_path = f"{REMOTE_MUSETALK_DIR}/models/musetalk/pytorch_model.bin"
        unet_config = f"{REMOTE_MUSETALK_DIR}/models/musetalk/musetalk.json"
        version_arg = "v1"

    whisper_dir = f"{REMOTE_MUSETALK_DIR}/models/whisper"

    cmd = [
        "python3",
        "-m",
        "scripts.inference",
        "--inference_config",
        cfg_path,
        "--result_dir",
        job_results_dir,
        "--unet_model_path",
        unet_model_path,
        "--unet_config",
        unet_config,
        "--whisper_dir",
        whisper_dir,
        "--version",
        version_arg,
    ]
    if use_float16:
        cmd.append("--use_float16")

    env = os.environ.copy()

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Inference failed: {e}",
        }

    output_video_path = os.path.join(job_results_dir, version_arg, "output.mp4")
    result = {
        "job_id": job_id,
        "status": "succeeded" if os.path.exists(output_video_path) else "unknown",
        "output_path": output_video_path,
    }
    return result


@app.function(
    image=image,
    network_file_systems={"/shared": nfs},
    timeout=60 * 5,
)
def download_result(job_id: str, version: str = "v15") -> bytes:
    """Download the output video for a given job ID."""
    output_path = f"/shared/results/{job_id}/{version}/output.mp4"
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file not found: {output_path}")
    
    with open(output_path, "rb") as f:
        return f.read()
