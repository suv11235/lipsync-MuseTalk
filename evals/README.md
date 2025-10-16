# MuseTalk Evaluation Metrics

This folder contains implementation of standard evaluation metrics for assessing lip-sync video generation quality.

## Features

- ✅ **Three Standard Metrics**: FID, CSIM, LSE-C
- ✅ **SyncNet Integration**: Automatic pre-trained model loading (requires GPU/Linux)
- ✅ **Baseline Comparison**: Compare MuseTalk vs Wav2Lip (or any models)
- ✅ **Production Ready**: Clean, documented, easy to use

## ⚠️ Important Note: LSE-C Metric Requirements

The **LSE-C (Lip-Sync Error Confidence)** metric requires:
- **Linux/GPU environment** with proper dependencies (xformers, einops, diffusers)
- **Pre-trained SyncNet model** from MuseTalk

**On macOS or CPU-only environments**: LSE-C will be automatically skipped. The FID and CSIM metrics will still work perfectly.

**For accurate LSE-C evaluation**: Run on a Linux machine with GPU, or use the Modal deployment.

## Metrics Implemented

### 1. **FID (Frechet Inception Distance)** - Visual Fidelity
- **Purpose**: Measures the visual quality and realism of generated videos
- **How it works**: Compares feature distributions between source and generated videos using InceptionV3 embeddings
- **Interpretation**: Lower FID scores indicate better visual quality (typical range: 0-300+)
- **Implementation**: `fid_metric.py`

### 2. **CSIM (Cosine Similarity)** - Identity Preservation  
- **Purpose**: Measures how well the generated video preserves the identity of the source person
- **How it works**: Compares face recognition embeddings between source and generated frames
- **Interpretation**: Higher CSIM scores indicate better identity preservation (range: 0-1, typically 0.3-0.95)
- **Implementation**: `csim_metric.py`
- **Models supported**: 
  - InsightFace (ArcFace) - recommended
  - FaceNet - alternative

### 3. **LSE-C (Lip-Sync Error Confidence)** - Lip Synchronization
- **Purpose**: Measures how well lip movements match the audio
- **How it works**: Uses SyncNet to compute audio-visual synchronization confidence
- **Interpretation**: Higher confidence (lower distance) indicates better lip sync (confidence range: 0-1)
- **Implementation**: `lse_c_metric.py`
- **Model**: Automatically loads MuseTalk's pre-trained SyncNet (1.4GB) for accurate evaluation
- **Status**: ✅ **Fully integrated** - Works out of the box with MuseTalk weights

## Installation

### Basic Installation

```bash
cd evals
pip install -r requirements.txt
```

### Face Recognition Model Options

You can choose between two face recognition models for CSIM:

**Option 1: InsightFace (Recommended)**
```bash
pip install insightface onnxruntime
```

**Option 2: FaceNet (Lighter alternative)**
```bash
pip install facenet-pytorch
```

### SyncNet Model for LSE-C

The LSE-C metric requires MuseTalk's pre-trained SyncNet weights and GPU dependencies.

**Requirements**:
- Linux/GPU environment
- Dependencies: `xformers`, `einops`, `diffusers`
- Pre-trained weights: `MuseTalk/models/syncnet/latentsync_syncnet.pt` (1.4GB)

**Behavior**:
- ✅ On Linux/GPU: Automatically loads SyncNet and provides accurate scores
- ⚠️  On macOS/CPU: LSE-C is skipped (FID and CSIM still work)
- 💡 For LSE-C evaluation: Use Modal deployment or Linux machine

## Usage

### Quick Start - Single Model Evaluation

Evaluate a generated video with all metrics:

```bash
python evaluate.py \
  --source_video path/to/source_video.mp4 \
  --generated_video path/to/generated_video.mp4 \
  --audio path/to/audio.wav \
  --output_json results.json
```

**Example with MuseTalk demo files:**
```bash
python evaluate.py \
  --source_video ../MuseTalk/data/video/yongen.mp4 \
  --generated_video path/to/your_generated_output.mp4 \
  --audio ../MuseTalk/data/audio/yongen.wav \
  --output_json results.json
```

### Baseline Comparison (MuseTalk vs Wav2Lip)

Compare two models side-by-side:

```bash
python compare_baselines.py \
  --source_video path/to/source.mp4 \
  --audio path/to/audio.wav \
  --model1_output path/to/musetalk_output.mp4 \
  --model2_output path/to/wav2lip_output.mp4 \
  --model1_name "MuseTalk" \
  --model2_name "Wav2Lip" \
  --output_json comparison_results.json
```

**Output**: Side-by-side comparison table + JSON with all metrics

See `BASELINE_COMPARISON.md` for complete guide on comparing with Wav2Lip.

### Advanced Usage

Skip specific metrics:

```bash
python evaluate.py \
  --source_video path/to/source.mp4 \
  --generated_video path/to/generated.mp4 \
  --audio path/to/audio.wav \
  --skip_fid \
  --skip_csim
```

Customize CSIM sampling (evaluate every Nth frame):

```bash
python evaluate.py \
  --source_video path/to/source.mp4 \
  --generated_video path/to/generated.mp4 \
  --audio path/to/audio.wav \
  --csim_sample_rate 5  # Evaluate every 5th frame
```

Use custom SyncNet model:

```bash
python evaluate.py \
  --source_video path/to/source.mp4 \
  --generated_video path/to/generated.mp4 \
  --audio path/to/audio.wav \
  --syncnet_model path/to/syncnet_weights.pt
```

### Using Individual Metrics

You can also use metrics individually in your own scripts:

```python
from fid_metric import FIDMetric
from csim_metric import CSIMMetric
from lse_c_metric import LSECMetric

# FID
fid_metric = FIDMetric()
fid_score = fid_metric.compute_fid(
    real_images="path/to/source.mp4",
    generated_images="path/to/generated.mp4"
)
print(f"FID: {fid_score:.2f}")

# CSIM
csim_metric = CSIMMetric()
avg_csim, std_csim, scores = csim_metric.compute_csim_videos(
    source_video="path/to/source.mp4",
    generated_video="path/to/generated.mp4",
    sample_rate=10
)
print(f"CSIM: {avg_csim:.4f} ± {std_csim:.4f}")

# LSE-C
lse_c_metric = LSECMetric()
avg_conf, lse_dist, conf_scores = lse_c_metric.compute_lse_c(
    video_path="path/to/generated.mp4",
    audio_path="path/to/audio.wav"
)
print(f"Sync Confidence: {avg_conf:.4f}")
```

## Output Format

The evaluation script produces a JSON file with the following structure:

```json
{
  "source_video": "path/to/source.mp4",
  "generated_video": "path/to/generated.mp4",
  "audio": "path/to/audio.wav",
  "metrics": {
    "FID": {
      "score": 45.23,
      "interpretation": "Lower is better (measures visual fidelity)",
      "computation_time_seconds": 12.5
    },
    "CSIM": {
      "average": 0.8234,
      "std": 0.0456,
      "min": 0.7123,
      "max": 0.9012,
      "num_frames_evaluated": 150,
      "interpretation": "Higher is better (range: 0-1, measures identity preservation)",
      "computation_time_seconds": 8.3
    },
    "LSE-C": {
      "sync_confidence": 0.7891,
      "lse_distance": 0.2109,
      "min_confidence": 0.6234,
      "max_confidence": 0.8901,
      "num_windows_evaluated": 75,
      "interpretation": "Higher confidence (lower distance) is better",
      "computation_time_seconds": 15.2
    }
  }
}
```

## Performance Considerations

- **FID**: Computationally intensive, processes all frames through InceptionV3
- **CSIM**: Can be sped up by increasing `--csim_sample_rate` (default: 10)
- **LSE-C**: Most computationally intensive due to SyncNet inference
- **GPU**: Highly recommended for faster computation (automatically used if available)

## Typical Score Ranges

Based on lip-sync research literature:

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| **FID** | < 20 | 20-50 | 50-100 | > 100 |
| **CSIM** | > 0.85 | 0.70-0.85 | 0.55-0.70 | < 0.55 |
| **LSE-C Confidence** | > 0.80 | 0.65-0.80 | 0.50-0.65 | < 0.50 |

*Note: These ranges are approximate and may vary based on dataset and use case.*

## References

1. **FID**: Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NeurIPS 2017)
2. **CSIM**: Commonly used with ArcFace/FaceNet embeddings for identity preservation
3. **LSE-C**: Based on SyncNet from "Out of time: automated lip sync in the wild" (ACCV 2016)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size or sample rate
   - Use CPU instead (slower but works)

2. **InsightFace installation issues**
   - Try FaceNet instead: `pip install facenet-pytorch`
   - Or use CPU-only onnxruntime: `pip install onnxruntime`

3. **No face detected in CSIM**
   - Check that videos contain clear frontal faces
   - Increase lighting/video quality

4. **LSE-C warnings about SyncNet weights**
   - Download or train SyncNet weights
   - Or accept lower accuracy with basic architecture

## Contributing

To add new metrics:
1. Create a new metric file (e.g., `new_metric.py`)
2. Implement the metric class with appropriate methods
3. Add import to `__init__.py`
4. Update `evaluate.py` to include the new metric
5. Update this README

## License

These evaluation metrics follow standard implementations from research literature and are provided for academic and research purposes.

