"""Quick test of MuseTalk Modal deployment with a single image (faster than video)."""
import modal
from musetalk_modal import app, run_inference, download_result

@app.local_entrypoint()
def main():
    # Use a single image instead of video for faster testing
    print("Starting Modal inference with image...")
    result = run_inference.remote(
        image_url="https://github.com/TMElyralab/MuseTalk/raw/main/assets/demo/yongen/yongen.jpeg",
        audio_path="data/audio/yongen.wav",
        version="v15",
        use_float16=True
    )
    
    print("\n=== RESULT ===")
    print(f"Job ID: {result.get('job_id')}")
    print(f"Status: {result.get('status')}")
    print(f"Output: {result.get('output_path')}")
    if result.get('error'):
        print(f"Error: {result.get('error')}")
    
    # Download the result if successful
    if result.get('status') == 'succeeded':
        job_id = result.get('job_id')
        output_file = f"output_{job_id}.mp4"
        print(f"\nDownloading result to {output_file}...")
        
        video_bytes = download_result.remote(job_id=job_id, version="v15")
        with open(output_file, "wb") as f:
            f.write(video_bytes)
        
        print(f"✅ Video downloaded: {output_file} ({len(video_bytes) / 1024 / 1024:.2f} MB)")

