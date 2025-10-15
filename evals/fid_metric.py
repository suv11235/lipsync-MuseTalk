"""
Frechet Inception Distance (FID) metric for visual fidelity evaluation.

FID measures the similarity between generated and real image distributions
by comparing their statistics in the feature space of an Inception network.
"""

import torch
import numpy as np
from scipy import linalg
from torch import nn
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import cv2
from typing import List, Union, Tuple
from pathlib import Path


class FIDMetric:
    """
    Calculates Frechet Inception Distance between two sets of images.
    
    Lower FID indicates better visual quality and similarity to real images.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize FID metric.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.inception_model = self._load_inception_model()
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_inception_model(self) -> nn.Module:
        """Load InceptionV3 model and modify for feature extraction."""
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()  # Remove final classification layer
        model.to(self.device)
        model.eval()
        return model
    
    def _extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from images using InceptionV3.
        
        Args:
            images: List of images as numpy arrays (H, W, C) in RGB
            
        Returns:
            Feature vectors as numpy array (N, 2048)
        """
        features = []
        
        with torch.no_grad():
            for img in images:
                # Convert numpy array to PIL Image
                if img.dtype == np.uint8:
                    pil_img = Image.fromarray(img)
                else:
                    pil_img = Image.fromarray((img * 255).astype(np.uint8))
                
                # Transform and add batch dimension
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # Extract features
                feat = self.inception_model(img_tensor)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def _calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and covariance of features.
        
        Args:
            features: Feature array (N, D)
            
        Returns:
            Mean and covariance matrix
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def _calculate_fid(
        self, 
        mu1: np.ndarray, 
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Calculate Frechet Inception Distance.
        
        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Small constant for numerical stability
            
        Returns:
            FID score
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return float(fid)
    
    def load_video_frames(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load frames from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames as numpy arrays (H, W, C) in RGB
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def compute_fid(
        self,
        real_images: Union[List[np.ndarray], str, Path],
        generated_images: Union[List[np.ndarray], str, Path]
    ) -> float:
        """
        Compute FID score between real and generated images.
        
        Args:
            real_images: Either list of numpy arrays or path to video file
            generated_images: Either list of numpy arrays or path to video file
            
        Returns:
            FID score (lower is better)
        """
        # Load images from video if paths are provided
        if isinstance(real_images, (str, Path)):
            real_images = self.load_video_frames(real_images)
        if isinstance(generated_images, (str, Path)):
            generated_images = self.load_video_frames(generated_images)
        
        # Extract features
        print(f"Extracting features from {len(real_images)} real images...")
        real_features = self._extract_features(real_images)
        
        print(f"Extracting features from {len(generated_images)} generated images...")
        gen_features = self._extract_features(generated_images)
        
        # Calculate statistics
        mu_real, sigma_real = self._calculate_statistics(real_features)
        mu_gen, sigma_gen = self._calculate_statistics(gen_features)
        
        # Calculate FID
        fid_score = self._calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        
        return fid_score


if __name__ == "__main__":
    # Example usage
    fid_metric = FIDMetric()
    
    # Example with video files
    # fid_score = fid_metric.compute_fid(
    #     real_images="path/to/original_video.mp4",
    #     generated_images="path/to/generated_video.mp4"
    # )
    # print(f"FID Score: {fid_score:.2f}")

