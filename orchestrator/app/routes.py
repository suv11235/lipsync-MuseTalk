from fastapi import APIRouter, HTTPException
from .models import JobRequest, JobResponse, JobStatusResponse
from .workers import submit_job, get_status

router = APIRouter()


@router.post("/jobs", response_model=JobResponse)
async def create_job(req: JobRequest):
    # Validate that we have either URL or path for video/image
    if not any([req.video_url, req.image_url, req.video_path]):
        raise HTTPException(status_code=400, detail="Provide video_url, image_url, or video_path")
    # Validate that we have audio
    if not req.audio_url and not req.audio_path:
        raise HTTPException(status_code=400, detail="Provide audio_url or audio_path")
    
    job_id = submit_job(
        video_url=str(req.video_url) if req.video_url else None,
        image_url=str(req.image_url) if req.image_url else None,
        audio_url=str(req.audio_url) if req.audio_url else None,
        video_path=req.video_path,
        audio_path=req.audio_path,
        version=req.version,
    )
    return JobResponse(job_id=job_id, status="submitted")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    data = get_status(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job_id, status=data.get("status", "submitted"), output_path=data.get("output_path"), error=data.get("error"))


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    data = get_status(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data.get("status") != "succeeded":
        raise HTTPException(status_code=409, detail="Job not completed")
    return {"output_path": data.get("output_path")}

