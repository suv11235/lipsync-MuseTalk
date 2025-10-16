# Baseline Comparison Guide

This guide explains how to compare MuseTalk against other lip-sync models (like Wav2Lip) using the evaluation metrics.

---

## Overview

The `compare_baselines.py` script allows you to:
- Compare two models side-by-side using FID, CSIM, and LSE-C metrics
- Generate detailed comparison reports
- Identify which model performs better on each metric

---

## Prerequisites

1. **Evaluation environment set up** (see main README.md)
2. **Generated outputs from both models** to compare
3. **Source video and audio files**

---

## Quick Start

### Step 1: Generate Model Outputs

First, generate outputs from both models you want to compare:

**MuseTalk Output:**
```bash
# Use Modal deployment or local MuseTalk inference
# Results in: musetalk_output.mp4
```

**Wav2Lip Output (or any other model):**
```bash
# Option A: HuggingFace Spaces
# 1. Go to: https://huggingface.co/spaces/hasibzunair/Wav2Lip
# 2. Upload your video and audio
# 3. Download the generated output

# Option B: Local Wav2Lip installation
# Follow Wav2Lip repo instructions: https://github.com/Rudrabha/Wav2Lip
```

### Step 2: Run Comparison

Activate the evaluation environment and run the comparison:

```bash
cd evals
source venv_eval/bin/activate  # or your virtual environment

python compare_baselines.py \
  --source_video path/to/original_video.mp4 \
  --audio path/to/audio.wav \
  --model1_output path/to/musetalk_output.mp4 \
  --model2_output path/to/wav2lip_output.mp4 \
  --model1_name "MuseTalk" \
  --model2_name "Wav2Lip" \
  --output_json comparison_results.json
```

### Step 3: Review Results

The script will:
1. Run all three metrics (FID, CSIM, LSE-C) on both outputs
2. Display a comparison table in the terminal
3. Save detailed results to `comparison_results.json`

**Example Output:**
```
================================================================================
COMPARISON RESULTS
================================================================================

Metric       |        MuseTalk |         Wav2Lip | Winner          | Improvement
---------------------------------------------------------------------------------------
FID          |          1.2468 |          5.6914 | MuseTalk        | +78.1%
CSIM         |          0.8426 |          0.9263 | Wav2Lip         | +9.9%
LSE-C        |         -0.0507 |          0.0387 | Wav2Lip         | +176.3%
================================================================================

📊 Overall Summary:
   MuseTalk: 1/3 metrics
   Wav2Lip: 2/3 metrics
```

---

## Understanding the Metrics

### FID (Frechet Inception Distance)
- **Measures:** Visual quality and realism
- **Lower is better**
- **Interpretation:** How close the generated frames are to natural, realistic video

### CSIM (Cosine Similarity for Identity)
- **Measures:** Identity preservation
- **Higher is better** (range: 0-1)
- **Interpretation:** How well the person's face identity is maintained

### LSE-C (Lip-Sync Error Confidence)
- **Measures:** Lip synchronization quality
- **Higher is better**
- **Interpretation:** How well the lip movements match the audio

---

## Advanced Usage

### Comparing Multiple Models

You can run multiple comparisons to evaluate several models:

```bash
# MuseTalk vs Wav2Lip
python compare_baselines.py \
  --source_video original.mp4 \
  --audio audio.wav \
  --model1_output musetalk.mp4 \
  --model2_output wav2lip.mp4 \
  --model1_name "MuseTalk" \
  --model2_name "Wav2Lip" \
  --output_json musetalk_vs_wav2lip.json

# MuseTalk vs Another Model
python compare_baselines.py \
  --source_video original.mp4 \
  --audio audio.wav \
  --model1_output musetalk.mp4 \
  --model2_output other_model.mp4 \
  --model1_name "MuseTalk" \
  --model2_name "OtherModel" \
  --output_json musetalk_vs_other.json
```

### Using MuseTalk Demo Files

For quick testing with included demo files:

```bash
python compare_baselines.py \
  --source_video ../MuseTalk/data/video/yongen.mp4 \
  --audio ../MuseTalk/data/audio/yongen.wav \
  --model1_output model1_output.mp4 \
  --model2_output model2_output.mp4 \
  --model1_name "Model1" \
  --model2_name "Model2"
```

---

## Output Format

The comparison script generates a JSON file with detailed results:

```json
{
  "source_video": "path/to/source.mp4",
  "audio": "path/to/audio.wav",
  "models": {
    "MuseTalk": {
      "output": "path/to/musetalk.mp4",
      "metrics": {
        "FID": { "score": 1.25, "computation_time_seconds": 25.4 },
        "CSIM": { "average": 0.84, "std": 0.03 },
        "LSE-C": { "sync_confidence": -0.05, "lse_distance": 1.05 }
      }
    },
    "Wav2Lip": { ... }
  },
  "comparison": {
    "FID": { "winner": "MuseTalk", "improvement_pct": 78.1 },
    "CSIM": { "winner": "Wav2Lip", "improvement_pct": 9.9 },
    "LSE-C": { "winner": "Wav2Lip", "improvement_pct": -176.3 }
  }
}
```

---

## Tips for Accurate Comparisons

1. **Use the same source video and audio** for both models
2. **Use default parameters** for fair comparison (unless testing specific settings)
3. **Test on multiple videos** to avoid bias from a single sample
4. **Consider computational cost** along with quality metrics
5. **Run on GPU environment** for accurate LSE-C scores (macOS uses fallback)

---

## Troubleshooting

### Issue: Different video lengths
**Solution:** The metrics handle different lengths, but ensure both models used the same input.

### Issue: LSE-C scores seem unreliable
**Solution:** LSE-C requires pre-trained SyncNet weights. On macOS, it uses a fallback. For accurate LSE-C, run on Linux/GPU environment.

### Issue: Out of memory
**Solution:** Reduce the number of frames evaluated or use a machine with more RAM.

---

## Resources

- **Wav2Lip Repository:** https://github.com/Rudrabha/Wav2Lip
- **Wav2Lip HuggingFace Space:** https://huggingface.co/spaces/hasibzunair/Wav2Lip
- **MuseTalk Repository:** https://github.com/TMElyralab/MuseTalk
- **Evaluation Metrics Details:** See `README.md` in this directory

---

For questions or issues, please open an issue on GitHub.
