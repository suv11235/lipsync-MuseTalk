"""
Lip-Sync-Error Confidence (LSE-C) metric for lip synchronization evaluation.

LSE-C measures how well the lip movements match the audio using a SyncNet model.
Lower LSE-C indicates better lip synchronization.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Union, Tuple
from pathlib import Path
import librosa


class LSECMetric:
    """
    Calculates Lip-Sync-Error Confidence for lip synchronization evaluation.
    
    LSE-C uses a pre-trained SyncNet model to measure audio-visual synchronization.
    Lower LSE-C indicates better lip sync (typically reported as distance/error).
    Higher confidence scores indicate better synchronization.
    """
    
    def __init__(
        self, 
        syncnet_model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize LSE-C metric.
        
        Args:
            syncnet_model_path: Path to pre-trained SyncNet model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.syncnet_model = None
        self.syncnet_model_path = syncnet_model_path
        self._load_syncnet()
        
    def _load_syncnet(self):
        """Load SyncNet model for lip-sync evaluation."""
        try:
            # Add MuseTalk to Python path
            import sys
            musetalk_path = str(Path(__file__).parent.parent / "MuseTalk")
            if musetalk_path not in sys.path:
                sys.path.insert(0, musetalk_path)
            
            # Try to load existing SyncNet from MuseTalk
            from musetalk.loss.syncnet import SyncNet_color
            
            self.syncnet_model = SyncNet_color()
            
            # Load weights if provided
            if self.syncnet_model_path:
                checkpoint = torch.load(self.syncnet_model_path, map_location=self.device)
                self.syncnet_model.load_state_dict(checkpoint)
                print(f"✓ Loaded SyncNet weights from {self.syncnet_model_path}")
            else:
                # Try to find the default syncnet weights
                default_path = Path(__file__).parent.parent / "MuseTalk/models/syncnet/latentsync_syncnet.pt"
                if default_path.exists():
                    print(f"Loading pre-trained SyncNet weights from {default_path}...")
                    checkpoint = torch.load(default_path, map_location=self.device)
                    self.syncnet_model.load_state_dict(checkpoint)
                    print(f"✓ Loaded pre-trained SyncNet weights ({default_path.stat().st_size / 1e9:.2f} GB)")
                else:
                    print("⚠ Warning: No SyncNet weights found. Using random initialization (not reliable).")
                    print(f"   Expected location: {default_path}")
            
            self.syncnet_model.to(self.device)
            self.syncnet_model.eval()
            self.model_type = "musetalk_syncnet"
            print("✓ SyncNet loaded successfully")
            
        except (ImportError, Exception) as e:
            print(f"✗ Could not load MuseTalk SyncNet: {e}")
            print(f"   Falling back to basic SyncNet (results will not be reliable)")
            # Fall back to implementing basic SyncNet architecture
            self._initialize_basic_syncnet()
    
    def _initialize_basic_syncnet(self):
        """Initialize a basic SyncNet architecture."""
        print("Initializing basic SyncNet architecture...")
        
        class BasicSyncNet(nn.Module):
            def __init__(self):
                super(BasicSyncNet, self).__init__()
                
                # Audio encoder
                self.audio_encoder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 256),
                )
                
                # Video encoder (for mouth region)
                self.video_encoder = nn.Sequential(
                    nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),  # 5 frames * 3 channels
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 256),
                )
                
            def forward_audio(self, audio):
                return self.audio_encoder(audio)
            
            def forward_video(self, video):
                return self.video_encoder(video)
            
            def forward(self, audio, video):
                audio_emb = self.forward_audio(audio)
                video_emb = self.forward_video(video)
                return audio_emb, video_emb
        
        self.syncnet_model = BasicSyncNet()
        self.syncnet_model.to(self.device)
        self.syncnet_model.eval()
        self.model_type = "basic_syncnet"
        print("Warning: Using basic SyncNet without pre-trained weights.")
    
    def extract_audio_features(
        self, 
        audio_path: Union[str, Path],
        sr: int = 16000,
        n_mels: int = 80
    ) -> np.ndarray:
        """
        Extract mel-spectrogram features from audio.
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            n_mels: Number of mel bands
            
        Returns:
            Mel-spectrogram features
        """
        # Load audio
        audio, _ = librosa.load(str(audio_path), sr=sr)
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=160,
            win_length=400
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def load_video_frames(
        self, 
        video_path: Union[str, Path],
        extract_mouth: bool = True
    ) -> List[np.ndarray]:
        """
        Load frames from video, optionally extracting mouth region.
        
        Args:
            video_path: Path to video file
            extract_mouth: Whether to extract mouth region (default: True)
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if extract_mouth:
                # Simple approach: extract lower half of face
                # In practice, you'd use facial landmarks to extract precise mouth region
                h, w = frame_rgb.shape[:2]
                mouth_region = frame_rgb[h//2:, :]  # Lower half
                frames.append(mouth_region)
            else:
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def compute_sync_confidence(
        self,
        audio_embedding: torch.Tensor,
        video_embedding: torch.Tensor
    ) -> float:
        """
        Compute synchronization confidence using cosine similarity.
        
        Args:
            audio_embedding: Audio embedding tensor
            video_embedding: Video embedding tensor
            
        Returns:
            Synchronization confidence score
        """
        # Normalize embeddings
        audio_norm = torch.nn.functional.normalize(audio_embedding, p=2, dim=-1)
        video_norm = torch.nn.functional.normalize(video_embedding, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(audio_norm, video_norm, dim=-1)
        
        return similarity.mean().item()
    
    def compute_lse_c(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        fps: int = 25,
        window_size: int = 5
    ) -> Tuple[float, float, List[float]]:
        """
        Compute LSE-C metric for a video-audio pair.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            fps: Video frame rate
            window_size: Number of frames per window for evaluation
            
        Returns:
            Tuple of (average confidence, LSE distance, list of per-window scores)
        """
        print("Loading video frames...")
        frames = self.load_video_frames(video_path, extract_mouth=True)
        
        print("Extracting audio features...")
        audio_features = self.extract_audio_features(audio_path)
        
        # Compute number of windows
        num_frames = len(frames)
        num_windows = max(1, num_frames - window_size + 1)
        
        confidence_scores = []
        
        print(f"Computing sync confidence for {num_windows} windows...")
        
        with torch.no_grad():
            for i in range(0, num_frames - window_size + 1, window_size // 2):
                # Get video window
                video_window = frames[i:i + window_size]
                if len(video_window) < window_size:
                    continue
                
                # Prepare video input (stack frames)
                video_array = np.array(video_window)
                video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()
                video_tensor = video_tensor.reshape(1, -1, video_array.shape[1], video_array.shape[2])
                video_tensor = video_tensor.to(self.device)
                
                # Prepare audio input (corresponding time window)
                audio_start = int((i / fps) * 100)  # Assuming 100 Hz mel features
                audio_end = int(((i + window_size) / fps) * 100)
                audio_window = audio_features[:, audio_start:audio_end]
                
                if audio_window.shape[1] == 0:
                    continue
                
                audio_tensor = torch.from_numpy(audio_window).unsqueeze(0).unsqueeze(0).float()
                audio_tensor = audio_tensor.to(self.device)
                
                # Get embeddings
                try:
                    # Both MuseTalk SyncNet and BasicSyncNet use forward(audio, video)
                    audio_emb, video_emb = self.syncnet_model(audio_tensor, video_tensor)
                    
                    # Compute confidence
                    confidence = self.compute_sync_confidence(audio_emb, video_emb)
                    confidence_scores.append(confidence)
                    
                except Exception as e:
                    print(f"Warning: Error processing window {i}: {e}")
                    continue
        
        if not confidence_scores:
            raise ValueError("No valid sync scores computed")
        
        # Calculate statistics
        avg_confidence = np.mean(confidence_scores)
        lse_distance = 1.0 - avg_confidence  # Convert confidence to error/distance
        
        return avg_confidence, lse_distance, confidence_scores


if __name__ == "__main__":
    # Example usage
    lse_c_metric = LSECMetric()
    
    # Example with video and audio files
    # avg_conf, lse_dist, scores = lse_c_metric.compute_lse_c(
    #     video_path="path/to/generated_video.mp4",
    #     audio_path="path/to/audio.wav"
    # )
    # print(f"Average Sync Confidence: {avg_conf:.4f}")
    # print(f"LSE Distance: {lse_dist:.4f}")

