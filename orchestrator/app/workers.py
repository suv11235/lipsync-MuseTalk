import threading
import sys
from typing import Dict, Optional
from pathlib import Path
import modal

# Add modal_app to path for imports
MODAL_APP_DIR = Path(__file__).parent.parent.parent / "modal_app"
sys.path.insert(0, str(MODAL_APP_DIR))

# In-memory store for job statuses (POC only)
JOB_STORE: Dict[str, Dict] = {}

# Access the Modal function by name at runtime

def submit_job(
    video_url: Optional[str] = None,
    image_url: Optional[str] = None,
    audio_url: Optional[str] = None,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    version: str = "v15"
) -> str:
    """
    Submit a job to Modal using spawn (detached mode).
    The function call continues even if client disconnects.
    """
    import uuid
    
    # Generate a job_id
    job_id = str(uuid.uuid4())
    
    # Initialize job status as queued
    JOB_STORE[job_id] = {
        "status": "queued",
        "output_path": None,
        "error": None,
        "modal_call_id": None
    }

    def _spawn_and_monitor():
        try:
            print(f"[Job {job_id}] Starting background thread...")
            
            # Get reference to the deployed function
            print(f"[Job {job_id}] Looking up deployed function...")
            run_inference = modal.Function.from_name("musetalk-poc", "run_inference")
            
            print(f"[Job {job_id}] Calling deployed function...")
            # Call the deployed function directly - no app.run() needed!
            result = run_inference.remote(
                video_url=video_url,
                audio_url=audio_url,
                video_path=video_path,
                audio_path=audio_path,
                version=version
            )
            
            print(f"[Job {job_id}] Inference complete, updating status...")
            JOB_STORE[job_id]["status"] = result.get("status", "failed")
            JOB_STORE[job_id]["output_path"] = result.get("output_path")
            JOB_STORE[job_id]["error"] = result.get("error")
            print(f"[Job {job_id}] Final status: {JOB_STORE[job_id]['status']}")
                
        except Exception as e:
            print(f"[Job {job_id}] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            JOB_STORE[job_id]["status"] = "failed"
            JOB_STORE[job_id]["error"] = f"Orchestrator error: {str(e)}"

    # Run in background thread so API returns immediately
    t = threading.Thread(target=_spawn_and_monitor, daemon=True)
    t.start()
    return job_id


def get_status(job_id: str) -> Dict:
    return JOB_STORE.get(job_id, {"status": "submitted"})

