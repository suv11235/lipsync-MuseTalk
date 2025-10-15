from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal


class JobRequest(BaseModel):
    # URLs for remote inputs
    video_url: Optional[HttpUrl] = None
    image_url: Optional[HttpUrl] = None
    audio_url: Optional[HttpUrl] = None
    # Paths for local inputs (relative to MuseTalk repo in Modal)
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    version: Literal["v1", "v15"] = "v15"


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "submitted", "dispatching", "running", "succeeded", "failed"]


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "submitted", "dispatching", "running", "succeeded", "failed"]
    output_path: Optional[str] = None
    error: Optional[str] = None

