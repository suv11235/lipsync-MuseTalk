"""
Cosine Similarity (CSIM) metric for identity preservation evaluation.

CSIM measures the similarity between identity embeddings of source and generated images
using a face recognition model.
"""

import torch
import numpy as np
import cv2
from typing import List, Union, Tuple
from pathlib import Path
from PIL import Image


class CSIMMetric:
    """
    Calculates Cosine Similarity for identity preservation.
    
    Higher CSIM indicates better identity preservation (range: -1 to 1, typically 0 to 1).
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize CSIM metric.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.face_recognition_model = None
        self._load_model()
        
    def _load_model(self):
        """Load face recognition model for embedding extraction."""
        try:
            # Try to use InsightFace (ArcFace) for robust face embeddings
            from insightface.app import FaceAnalysis
            
            self.face_recognition_model = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_recognition_model.prepare(ctx_id=0 if self.device == "cuda" else -1)
            self.model_type = "insightface"
            print("Using InsightFace (ArcFace) for identity embeddings")
            
        except ImportError:
            # Fallback to FaceNet via facenet-pytorch
            try:
                from facenet_pytorch import InceptionResnetV1
                
                self.face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval()
                self.face_recognition_model.to(self.device)
                self.model_type = "facenet"
                print("Using FaceNet for identity embeddings")
                
            except ImportError:
                raise ImportError(
                    "Please install either 'insightface' or 'facenet-pytorch' for identity embeddings.\n"
                    "  pip install insightface  OR  pip install facenet-pytorch"
                )
    
    def _extract_embedding_insightface(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding using InsightFace.
        
        Args:
            image: Image as numpy array (H, W, C) in RGB
            
        Returns:
            Embedding vector
        """
        faces = self.face_recognition_model.get(image)
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Use the first detected face
        embedding = faces[0].embedding
        return embedding
    
    def _extract_embedding_facenet(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding using FaceNet.
        
        Args:
            image: Image as numpy array (H, W, C) in RGB
            
        Returns:
            Embedding vector
        """
        # Resize to expected input size
        img_resized = cv2.resize(image, (160, 160))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = (img_tensor - 127.5) / 128.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.face_recognition_model(img_tensor)
        
        return embedding.cpu().numpy().squeeze()
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from image.
        
        Args:
            image: Image as numpy array (H, W, C) in RGB
            
        Returns:
            Embedding vector
        """
        if self.model_type == "insightface":
            return self._extract_embedding_insightface(image)
        else:
            return self._extract_embedding_facenet(image)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)
    
    def load_video_frames(self, video_path: Union[str, Path], sample_rate: int = 1) -> List[np.ndarray]:
        """
        Load frames from a video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every Nth frame (default: 1 = all frames)
            
        Returns:
            List of frames as numpy arrays (H, W, C) in RGB
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def compute_csim(
        self,
        source_images: Union[List[np.ndarray], str, Path, np.ndarray],
        generated_images: Union[List[np.ndarray], str, Path],
        sample_rate: int = 10
    ) -> Tuple[float, List[float]]:
        """
        Compute CSIM score between source and generated images.
        
        Args:
            source_images: Either single image, list of images, or path to video file
            generated_images: Either list of images or path to video file
            sample_rate: Sample every Nth frame from videos (default: 10)
            
        Returns:
            Tuple of (average CSIM, list of per-frame CSIM scores)
        """
        # Load source image(s)
        if isinstance(source_images, (str, Path)):
            source_frames = self.load_video_frames(source_images, sample_rate=1)
            source_image = source_frames[0]  # Use first frame as reference
        elif isinstance(source_images, list):
            source_image = source_images[0]
        else:
            source_image = source_images
        
        # Load generated images
        if isinstance(generated_images, (str, Path)):
            generated_frames = self.load_video_frames(generated_images, sample_rate=sample_rate)
        else:
            generated_frames = generated_images
        
        # Extract source embedding
        print("Extracting source identity embedding...")
        source_embedding = self.extract_embedding(source_image)
        
        # Compute CSIM for each generated frame
        csim_scores = []
        print(f"Computing CSIM for {len(generated_frames)} frames...")
        
        for i, gen_frame in enumerate(generated_frames):
            try:
                gen_embedding = self.extract_embedding(gen_frame)
                csim = self.cosine_similarity(source_embedding, gen_embedding)
                csim_scores.append(csim)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(generated_frames)} frames")
                    
            except ValueError as e:
                print(f"Warning: Skipping frame {i}: {e}")
                continue
        
        if not csim_scores:
            raise ValueError("No valid CSIM scores computed - no faces detected")
        
        avg_csim = np.mean(csim_scores)
        return avg_csim, csim_scores
    
    def compute_csim_videos(
        self,
        source_video: Union[str, Path],
        generated_video: Union[str, Path],
        sample_rate: int = 10
    ) -> Tuple[float, float, List[float]]:
        """
        Compute CSIM between source and generated videos.
        
        Args:
            source_video: Path to source video
            generated_video: Path to generated video
            sample_rate: Sample every Nth frame (default: 10)
            
        Returns:
            Tuple of (average CSIM, std CSIM, list of per-frame CSIM scores)
        """
        avg_csim, csim_scores = self.compute_csim(
            source_images=source_video,
            generated_images=generated_video,
            sample_rate=sample_rate
        )
        
        std_csim = np.std(csim_scores)
        return avg_csim, std_csim, csim_scores


if __name__ == "__main__":
    # Example usage
    csim_metric = CSIMMetric()
    
    # Example with video files
    # avg_csim, std_csim, scores = csim_metric.compute_csim_videos(
    #     source_video="path/to/source_video.mp4",
    #     generated_video="path/to/generated_video.mp4",
    #     sample_rate=10
    # )
    # print(f"Average CSIM: {avg_csim:.4f} ± {std_csim:.4f}")

