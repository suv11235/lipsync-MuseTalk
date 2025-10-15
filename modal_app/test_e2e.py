"""End-to-end test of MuseTalk Modal deployment."""
import modal
from musetalk_modal import app, run_inference

@app.local_entrypoint()
def main():
    # Invoke the deployed function with sample data (shortest files for fast iteration)
    print("Starting Modal inference...")
    result = run_inference.remote(
        video_path="data/video/yongen.mp4",
        audio_path="data/audio/yongen.wav",
        version="v15"
    )
    
    print("\n=== RESULT ===")
    print(f"Job ID: {result.get('job_id')}")
    print(f"Status: {result.get('status')}")
    print(f"Output: {result.get('output_path')}")
    if result.get('error'):
        print(f"Error: {result.get('error')}")


