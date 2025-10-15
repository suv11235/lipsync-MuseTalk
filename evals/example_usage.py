"""
Example usage of evaluation metrics.

This script demonstrates how to use individual metrics or the full evaluation pipeline.
"""

from pathlib import Path
import sys

# Add parent directory to path to import from MuseTalk
sys.path.append(str(Path(__file__).parent.parent))

from fid_metric import FIDMetric
from csim_metric import CSIMMetric
from lse_c_metric import LSECMetric
from evaluate import evaluate_video


def example_individual_metrics():
    """Example: Use individual metrics separately."""
    
    print("\n" + "="*80)
    print("Example 1: Using Individual Metrics")
    print("="*80)
    
    # Example paths - replace with your actual paths
    source_video = "../MuseTalk/data/video/yongen.mp4"
    generated_video = "../modal_app/yongen_output.mp4"
    audio_file = "../MuseTalk/data/audio/yongen.wav"
    
    # Check if files exist
    if not Path(source_video).exists():
        print(f"⚠ Source video not found: {source_video}")
        print("Please update the paths in this script to your actual video files.")
        return
    
    # FID Metric
    print("\n1. Computing FID (Visual Fidelity)...")
    try:
        fid_metric = FIDMetric()
        fid_score = fid_metric.compute_fid(
            real_images=source_video,
            generated_images=generated_video
        )
        print(f"   FID Score: {fid_score:.2f}")
        print(f"   → Lower is better (typical range: 0-300+)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # CSIM Metric
    print("\n2. Computing CSIM (Identity Preservation)...")
    try:
        csim_metric = CSIMMetric()
        avg_csim, std_csim, csim_scores = csim_metric.compute_csim_videos(
            source_video=source_video,
            generated_video=generated_video,
            sample_rate=10  # Sample every 10th frame
        )
        print(f"   Average CSIM: {avg_csim:.4f} ± {std_csim:.4f}")
        print(f"   Frames evaluated: {len(csim_scores)}")
        print(f"   → Higher is better (range: 0-1)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # LSE-C Metric
    print("\n3. Computing LSE-C (Lip Synchronization)...")
    try:
        lse_c_metric = LSECMetric()
        avg_conf, lse_dist, conf_scores = lse_c_metric.compute_lse_c(
            video_path=generated_video,
            audio_path=audio_file
        )
        print(f"   Sync Confidence: {avg_conf:.4f}")
        print(f"   LSE Distance: {lse_dist:.4f}")
        print(f"   Windows evaluated: {len(conf_scores)}")
        print(f"   → Higher confidence (lower distance) is better")
    except Exception as e:
        print(f"   Error: {e}")


def example_full_evaluation():
    """Example: Use the full evaluation pipeline."""
    
    print("\n" + "="*80)
    print("Example 2: Using Full Evaluation Pipeline")
    print("="*80)
    
    # Example paths - replace with your actual paths
    source_video = "../MuseTalk/data/video/yongen.mp4"
    generated_video = "../modal_app/yongen_output.mp4"
    audio_file = "../MuseTalk/data/audio/yongen.wav"
    
    # Check if files exist
    if not Path(source_video).exists():
        print(f"⚠ Source video not found: {source_video}")
        print("Please update the paths in this script to your actual video files.")
        return
    
    try:
        results = evaluate_video(
            source_video_path=source_video,
            generated_video_path=generated_video,
            audio_path=audio_file,
            output_json="evaluation_results.json",
            skip_fid=False,  # Set to True to skip FID
            skip_csim=False,  # Set to True to skip CSIM
            skip_lse_c=False,  # Set to True to skip LSE-C
            csim_sample_rate=10
        )
        
        print("\n✅ Evaluation complete! Results saved to evaluation_results.json")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")


def example_quick_test():
    """Example: Quick test with minimal computation."""
    
    print("\n" + "="*80)
    print("Example 3: Quick Test (Skip Heavy Metrics)")
    print("="*80)
    
    # Example paths - replace with your actual paths
    source_video = "../MuseTalk/data/video/yongen.mp4"
    generated_video = "../modal_app/yongen_output.mp4"
    audio_file = "../MuseTalk/data/audio/yongen.wav"
    
    # Check if files exist
    if not Path(source_video).exists():
        print(f"⚠ Source video not found: {source_video}")
        print("Please update the paths in this script to your actual video files.")
        return
    
    try:
        # Only compute CSIM for quick testing
        results = evaluate_video(
            source_video_path=source_video,
            generated_video_path=generated_video,
            audio_path=audio_file,
            output_json="quick_test_results.json",
            skip_fid=True,  # Skip FID (slow)
            skip_csim=False,
            skip_lse_c=True,  # Skip LSE-C (slow)
            csim_sample_rate=30  # Sample fewer frames for speed
        )
        
        print("\n✅ Quick test complete!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")


def print_usage_instructions():
    """Print usage instructions."""
    
    print("\n" + "="*80)
    print("MuseTalk Evaluation Metrics - Example Usage")
    print("="*80)
    print("\nThis script demonstrates three ways to use the evaluation metrics:\n")
    print("1. Individual Metrics - Use each metric separately")
    print("2. Full Evaluation - Run all metrics together")
    print("3. Quick Test - Skip heavy metrics for faster testing")
    print("\nTo run these examples:")
    print("  python example_usage.py individual    # Run example 1")
    print("  python example_usage.py full          # Run example 2")
    print("  python example_usage.py quick         # Run example 3")
    print("  python example_usage.py all           # Run all examples")
    print("\nOr use the main evaluation script:")
    print("  python evaluate.py --source_video <path> --generated_video <path> --audio <path>")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print_usage_instructions()
        sys.exit(0)
    
    mode = sys.argv[1].lower()
    
    if mode == "individual":
        example_individual_metrics()
    elif mode == "full":
        example_full_evaluation()
    elif mode == "quick":
        example_quick_test()
    elif mode == "all":
        example_individual_metrics()
        example_full_evaluation()
        example_quick_test()
    else:
        print(f"Unknown mode: {mode}")
        print_usage_instructions()

