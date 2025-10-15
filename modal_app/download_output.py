"""Download output from a completed MuseTalk inference job."""
import sys
import modal
from musetalk_modal import app, download_result

@app.local_entrypoint()
def main(job_id: str, output_file: str = "output.mp4"):
    """
    Download the result video from a MuseTalk job.
    
    Args:
        job_id: The job ID from the inference result
        output_file: Local path to save the video (default: output.mp4)
    """
    print(f"Downloading result for job {job_id}...")
    
    try:
        video_bytes = download_result.remote(job_id=job_id, version="v15")
        
        with open(output_file, "wb") as f:
            f.write(video_bytes)
        
        print(f"✅ Video saved to: {output_file}")
        print(f"   Size: {len(video_bytes) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        sys.exit(1)

