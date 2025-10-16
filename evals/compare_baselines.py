"""
Compare MuseTalk vs Wav2Lip (or any two models) using evaluation metrics.

Usage:
    python compare_baselines.py \
        --source_video path/to/source.mp4 \
        --audio path/to/audio.wav \
        --model1_output path/to/musetalk_output.mp4 \
        --model2_output path/to/wav2lip_output.mp4 \
        --model1_name "MuseTalk" \
        --model2_name "Wav2Lip" \
        --output_json comparison_results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import evaluate_video


def compare_models(
    source_video: str,
    audio: str,
    model1_output: str,
    model2_output: str,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    output_json: str = "comparison_results.json",
    skip_fid: bool = False,
    skip_csim: bool = False,
    skip_lse_c: bool = False
):
    """
    Compare two models using FID, CSIM, LSE-C metrics.
    
    Args:
        source_video: Path to source/original video
        audio: Path to audio file
        model1_output: Path to first model's output
        model2_output: Path to second model's output
        model1_name: Name of first model (default: "Model 1")
        model2_name: Name of second model (default: "Model 2")
        output_json: Path to save comparison results
        skip_fid: Skip FID computation
        skip_csim: Skip CSIM computation
        skip_lse_c: Skip LSE-C computation
    
    Returns:
        Comparison dictionary with results and analysis
    """
    print("=" * 80)
    print(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
    print("=" * 80)
    print(f"\nSource: {source_video}")
    print(f"Audio: {audio}")
    print(f"{model1_name}: {model1_output}")
    print(f"{model2_name}: {model2_output}")
    print()
    
    # Evaluate Model 1
    print(f"\n[1/2] Evaluating {model1_name}...")
    print("-" * 80)
    model1_results = evaluate_video(
        source_video_path=source_video,
        generated_video_path=model1_output,
        audio_path=audio,
        output_json=f"{model1_name.lower().replace(' ', '_')}_metrics.json",
        skip_fid=skip_fid,
        skip_csim=skip_csim,
        skip_lse_c=skip_lse_c
    )
    
    # Evaluate Model 2
    print(f"\n[2/2] Evaluating {model2_name}...")
    print("-" * 80)
    model2_results = evaluate_video(
        source_video_path=source_video,
        generated_video_path=model2_output,
        audio_path=audio,
        output_json=f"{model2_name.lower().replace(' ', '_')}_metrics.json",
        skip_fid=skip_fid,
        skip_csim=skip_csim,
        skip_lse_c=skip_lse_c
    )
    
    # Generate comparison
    comparison = {
        "source_video": str(source_video),
        "audio": str(audio),
        "models": {
            model1_name: {
                "output": str(model1_output),
                "metrics": model1_results.get("metrics", {})
            },
            model2_name: {
                "output": str(model2_output),
                "metrics": model2_results.get("metrics", {})
            }
        },
        "comparison": generate_comparison_summary(
            model1_results.get("metrics", {}),
            model2_results.get("metrics", {}),
            model1_name,
            model2_name
        )
    }
    
    # Save comparison
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print_comparison_table(comparison, model1_name, model2_name)
    
    print(f"\n✓ Comparison saved to: {output_json}")
    
    return comparison


def generate_comparison_summary(model1_metrics, model2_metrics, model1_name, model2_name):
    """Generate detailed comparison summary."""
    summary = {}
    
    # FID comparison (lower is better)
    if "FID" in model1_metrics and "FID" in model2_metrics:
        m1_fid = model1_metrics["FID"].get("score")
        m2_fid = model2_metrics["FID"].get("score")
        if m1_fid is not None and m2_fid is not None:
            winner = model1_name if m1_fid < m2_fid else model2_name
            improvement = ((m2_fid - m1_fid) / m2_fid * 100) if winner == model1_name else ((m1_fid - m2_fid) / m1_fid * 100)
            
            summary["FID"] = {
                model1_name: m1_fid,
                model2_name: m2_fid,
                "winner": winner,
                "difference": abs(m1_fid - m2_fid),
                "improvement_pct": improvement,
                "interpretation": "Lower is better"
            }
    
    # CSIM comparison (higher is better)
    if "CSIM" in model1_metrics and "CSIM" in model2_metrics:
        m1_csim = model1_metrics["CSIM"].get("average")
        m2_csim = model2_metrics["CSIM"].get("average")
        if m1_csim is not None and m2_csim is not None:
            winner = model1_name if m1_csim > m2_csim else model2_name
            improvement = ((m1_csim - m2_csim) / m2_csim * 100) if winner == model1_name else ((m2_csim - m1_csim) / m1_csim * 100)
            
            summary["CSIM"] = {
                model1_name: m1_csim,
                model2_name: m2_csim,
                "winner": winner,
                "difference": abs(m1_csim - m2_csim),
                "improvement_pct": improvement,
                "interpretation": "Higher is better"
            }
    
    # LSE-C comparison (higher confidence is better)
    if "LSE-C" in model1_metrics and "LSE-C" in model2_metrics:
        m1_lsec = model1_metrics["LSE-C"].get("sync_confidence")
        m2_lsec = model2_metrics["LSE-C"].get("sync_confidence")
        if m1_lsec is not None and m2_lsec is not None:
            winner = model1_name if m1_lsec > m2_lsec else model2_name
            improvement = ((m1_lsec - m2_lsec) / m2_lsec * 100) if winner == model1_name else ((m2_lsec - m1_lsec) / m1_lsec * 100)
            
            summary["LSE-C"] = {
                model1_name: m1_lsec,
                model2_name: m2_lsec,
                "winner": winner,
                "difference": abs(m1_lsec - m2_lsec),
                "improvement_pct": improvement,
                "interpretation": "Higher confidence is better"
            }
    
    return summary


def print_comparison_table(comparison, model1_name, model2_name):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    summary = comparison.get("comparison", {})
    
    if not summary:
        print("\n⚠  No metrics available for comparison")
        return
    
    # Header
    col_width = max(len(model1_name), len(model2_name), 15)
    print(f"\n{'Metric':<12} | {model1_name:>{col_width}} | {model2_name:>{col_width}} | {'Winner':<15} | Improvement")
    print("-" * (12 + col_width * 2 + 15 + 30))
    
    # Metrics
    for metric, data in summary.items():
        m1_val = data[model1_name]
        m2_val = data[model2_name]
        winner = data["winner"]
        improvement = data["improvement_pct"]
        
        # Format winner display
        winner_mark = "✓" if winner == model1_name else " "
        winner_mark2 = "✓" if winner == model2_name else " "
        
        print(f"{metric:<12} | {m1_val:>{col_width}.4f} {winner_mark} | {m2_val:>{col_width}.4f} {winner_mark2} | {winner:<15} | {improvement:+.1f}%")
    
    print("=" * 80)
    
    # Overall summary
    winners = [data["winner"] for data in summary.values()]
    model1_wins = winners.count(model1_name)
    model2_wins = winners.count(model2_name)
    total_metrics = len(winners)
    
    print(f"\n📊 Overall Summary:")
    print(f"   {model1_name}: {model1_wins}/{total_metrics} metrics")
    print(f"   {model2_name}: {model2_wins}/{total_metrics} metrics")
    
    if model1_wins > model2_wins:
        print(f"\n✅ {model1_name} outperforms {model2_name} ({model1_wins}/{total_metrics} metrics)")
    elif model2_wins > model1_wins:
        print(f"\n⚠️  {model2_name} outperforms {model1_name} ({model2_wins}/{total_metrics} metrics)")
    else:
        print(f"\n⚖️  Tied performance ({model1_wins}/{total_metrics} each)")
    
    # Detailed insights
    print(f"\n💡 Key Insights:")
    for metric, data in summary.items():
        improvement = data["improvement_pct"]
        winner = data["winner"]
        if abs(improvement) > 10:  # Significant difference
            print(f"   • {metric}: {winner} shows {abs(improvement):.1f}% improvement")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two lip-sync models using evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare MuseTalk vs Wav2Lip
    python compare_baselines.py \\
        --source_video ../MuseTalk/data/video/yongen.mp4 \\
        --audio ../MuseTalk/data/audio/yongen.wav \\
        --model1_output path/to/musetalk_output.mp4 \\
        --model2_output path/to/wav2lip_output.mp4 \\
        --model1_name "MuseTalk" \\
        --model2_name "Wav2Lip"
    
    # Quick comparison (skip FID for speed)
    python compare_baselines.py \\
        --source_video source.mp4 \\
        --audio audio.wav \\
        --model1_output output1.mp4 \\
        --model2_output output2.mp4 \\
        --skip_fid
        """
    )
    
    parser.add_argument(
        "--source_video",
        required=True,
        help="Path to source/original video"
    )
    
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file"
    )
    
    parser.add_argument(
        "--model1_output",
        required=True,
        help="Path to first model's output video"
    )
    
    parser.add_argument(
        "--model2_output",
        required=True,
        help="Path to second model's output video"
    )
    
    parser.add_argument(
        "--model1_name",
        default="Model 1",
        help="Name of first model (default: 'Model 1')"
    )
    
    parser.add_argument(
        "--model2_name",
        default="Model 2",
        help="Name of second model (default: 'Model 2')"
    )
    
    parser.add_argument(
        "--output_json",
        default="comparison_results.json",
        help="Path to save comparison results (default: comparison_results.json)"
    )
    
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID computation (faster)"
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
    
    args = parser.parse_args()
    
    # Validate paths
    for path_name, path_value in [
        ("source_video", args.source_video),
        ("audio", args.audio),
        ("model1_output", args.model1_output),
        ("model2_output", args.model2_output)
    ]:
        if not Path(path_value).exists():
            print(f"❌ Error: {path_name} not found: {path_value}")
            sys.exit(1)
    
    # Run comparison
    try:
        compare_models(
            source_video=args.source_video,
            audio=args.audio,
            model1_output=args.model1_output,
            model2_output=args.model2_output,
            model1_name=args.model1_name,
            model2_name=args.model2_name,
            output_json=args.output_json,
            skip_fid=args.skip_fid,
            skip_csim=args.skip_csim,
            skip_lse_c=args.skip_lse_c
        )
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

