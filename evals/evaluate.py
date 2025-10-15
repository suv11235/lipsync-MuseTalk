"""
Main evaluation script for MuseTalk generated videos.

This script computes FID, CSIM, and LSE-C metrics for evaluating:
- Visual fidelity (FID)
- Identity preservation (CSIM)
- Lip synchronization (LSE-C)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import time

from fid_metric import FIDMetric
from csim_metric import CSIMMetric
from lse_c_metric import LSECMetric


def evaluate_video(
    source_video_path: str,
    generated_video_path: str,
    audio_path: str,
    output_json: str = None,
    skip_fid: bool = False,
    skip_csim: bool = False,
    skip_lse_c: bool = False,
    csim_sample_rate: int = 10,
    syncnet_model_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate a generated video using multiple metrics.
    
    Args:
        source_video_path: Path to source/original video
        generated_video_path: Path to generated video
        audio_path: Path to audio file used for generation
        output_json: Path to save results as JSON (optional)
        skip_fid: Skip FID computation
        skip_csim: Skip CSIM computation
        skip_lse_c: Skip LSE-C computation
        csim_sample_rate: Sample rate for CSIM frames
        syncnet_model_path: Path to SyncNet weights for LSE-C
        
    Returns:
        Dictionary containing all metric results
    """
    results = {
        "source_video": str(source_video_path),
        "generated_video": str(generated_video_path),
        "audio": str(audio_path),
        "metrics": {}
    }
    
    print("=" * 80)
    print("MuseTalk Evaluation Metrics")
    print("=" * 80)
    print(f"Source Video: {source_video_path}")
    print(f"Generated Video: {generated_video_path}")
    print(f"Audio: {audio_path}")
    print("=" * 80)
    
    # FID - Visual Fidelity
    if not skip_fid:
        print("\n[1/3] Computing FID (Frechet Inception Distance)...")
        try:
            start_time = time.time()
            fid_metric = FIDMetric()
            fid_score = fid_metric.compute_fid(
                real_images=source_video_path,
                generated_images=generated_video_path
            )
            elapsed = time.time() - start_time
            
            results["metrics"]["FID"] = {
                "score": float(fid_score),
                "interpretation": "Lower is better (measures visual fidelity)",
                "computation_time_seconds": elapsed
            }
            print(f"✓ FID Score: {fid_score:.2f} (computed in {elapsed:.1f}s)")
            print(f"  → Lower FID indicates better visual quality")
            
        except Exception as e:
            print(f"✗ Error computing FID: {e}")
            results["metrics"]["FID"] = {"error": str(e)}
    
    # CSIM - Identity Preservation
    if not skip_csim:
        print("\n[2/3] Computing CSIM (Cosine Similarity for Identity Preservation)...")
        try:
            start_time = time.time()
            csim_metric = CSIMMetric()
            avg_csim, std_csim, csim_scores = csim_metric.compute_csim_videos(
                source_video=source_video_path,
                generated_video=generated_video_path,
                sample_rate=csim_sample_rate
            )
            elapsed = time.time() - start_time
            
            results["metrics"]["CSIM"] = {
                "average": float(avg_csim),
                "std": float(std_csim),
                "min": float(min(csim_scores)),
                "max": float(max(csim_scores)),
                "num_frames_evaluated": len(csim_scores),
                "interpretation": "Higher is better (range: 0-1, measures identity preservation)",
                "computation_time_seconds": elapsed
            }
            print(f"✓ CSIM Score: {avg_csim:.4f} ± {std_csim:.4f} (computed in {elapsed:.1f}s)")
            print(f"  → Evaluated {len(csim_scores)} frames")
            print(f"  → Higher CSIM indicates better identity preservation")
            
        except Exception as e:
            print(f"✗ Error computing CSIM: {e}")
            results["metrics"]["CSIM"] = {"error": str(e)}
    
    # LSE-C - Lip Synchronization
    if not skip_lse_c:
        print("\n[3/3] Computing LSE-C (Lip-Sync Error Confidence)...")
        try:
            start_time = time.time()
            lse_c_metric = LSECMetric(syncnet_model_path=syncnet_model_path)
            avg_conf, lse_dist, conf_scores = lse_c_metric.compute_lse_c(
                video_path=generated_video_path,
                audio_path=audio_path
            )
            elapsed = time.time() - start_time
            
            results["metrics"]["LSE-C"] = {
                "sync_confidence": float(avg_conf),
                "lse_distance": float(lse_dist),
                "min_confidence": float(min(conf_scores)),
                "max_confidence": float(max(conf_scores)),
                "num_windows_evaluated": len(conf_scores),
                "interpretation": "Higher confidence (lower distance) is better",
                "computation_time_seconds": elapsed
            }
            print(f"✓ Sync Confidence: {avg_conf:.4f} (LSE Distance: {lse_dist:.4f}, computed in {elapsed:.1f}s)")
            print(f"  → Evaluated {len(conf_scores)} windows")
            print(f"  → Higher confidence indicates better lip synchronization")
            
        except Exception as e:
            print(f"✗ Error computing LSE-C: {e}")
            results["metrics"]["LSE-C"] = {"error": str(e)}
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    if "FID" in results["metrics"] and "score" in results["metrics"]["FID"]:
        print(f"FID (Visual Fidelity):       {results['metrics']['FID']['score']:.2f}")
    
    if "CSIM" in results["metrics"] and "average" in results["metrics"]["CSIM"]:
        print(f"CSIM (Identity):             {results['metrics']['CSIM']['average']:.4f} ± {results['metrics']['CSIM']['std']:.4f}")
    
    if "LSE-C" in results["metrics"] and "sync_confidence" in results["metrics"]["LSE-C"]:
        print(f"LSE-C (Lip Sync):            {results['metrics']['LSE-C']['sync_confidence']:.4f} (distance: {results['metrics']['LSE-C']['lse_distance']:.4f})")
    
    print("=" * 80)
    
    # Save results to JSON if requested
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_json}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MuseTalk generated videos using FID, CSIM, and LSE-C metrics"
    )
    
    parser.add_argument(
        "--source_video",
        type=str,
        required=True,
        help="Path to source/original video"
    )
    
    parser.add_argument(
        "--generated_video",
        type=str,
        required=True,
        help="Path to generated video"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file used for generation"
    )
    
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON"
    )
    
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID computation"
    )
    
    parser.add_argument(
        "--skip_csim",
        action="store_true",
        help="Skip CSIM computation"
    )
    
    parser.add_argument(
        "--skip_lse_c",
        action="store_true",
        help="Skip LSE-C computation"
    )
    
    parser.add_argument(
        "--csim_sample_rate",
        type=int,
        default=10,
        help="Sample every Nth frame for CSIM evaluation (default: 10)"
    )
    
    parser.add_argument(
        "--syncnet_model",
        type=str,
        default=None,
        help="Path to SyncNet model weights for LSE-C"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.source_video).exists():
        raise FileNotFoundError(f"Source video not found: {args.source_video}")
    
    if not Path(args.generated_video).exists():
        raise FileNotFoundError(f"Generated video not found: {args.generated_video}")
    
    if not Path(args.audio).exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    
    # Run evaluation
    results = evaluate_video(
        source_video_path=args.source_video,
        generated_video_path=args.generated_video,
        audio_path=args.audio,
        output_json=args.output_json,
        skip_fid=args.skip_fid,
        skip_csim=args.skip_csim,
        skip_lse_c=args.skip_lse_c,
        csim_sample_rate=args.csim_sample_rate,
        syncnet_model_path=args.syncnet_model
    )
    
    return results


if __name__ == "__main__":
    main()

