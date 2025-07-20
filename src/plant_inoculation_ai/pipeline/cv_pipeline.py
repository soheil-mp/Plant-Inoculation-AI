"""
Computer Vision Pipeline for Plant Root Analysis.

This module provides an end-to-end computer vision pipeline that integrates
petri dish extraction, plant segmentation, U-Net inference, and root analysis.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Import our custom modules
from plant_inoculation_ai.computer_vision.petri_dish import PetriDishExtractor
from plant_inoculation_ai.computer_vision.segmentation import PlantSegmenter
from plant_inoculation_ai.computer_vision.root_analysis import RootArchitectureAnalyzer
from plant_inoculation_ai.models.unet import UNetModel


class CVPipeline:
    """End-to-end computer vision pipeline for root analysis."""
    
    def __init__(
        self,
        patch_size: int = 960,
        model_weights_path: Optional[str] = None
    ):
        """
        Initialize the CV pipeline.
        
        Args:
            patch_size: Size of patches for U-Net processing
            model_weights_path: Path to pre-trained model weights
        """
        self.patch_size = patch_size
        self.model_weights_path = model_weights_path
        
        # Initialize components
        self.petri_extractor = PetriDishExtractor()
        self.plant_segmenter = PlantSegmenter()
        self.root_analyzer = RootArchitectureAnalyzer()
        
        # Initialize U-Net model
        self.unet_model = UNetModel(
            img_height=patch_size,
            img_width=patch_size,
            img_channels=1
        )
        self.model = self.unet_model.build_model()
        self.unet_model.compile_model()
        
        # Load weights if provided
        if model_weights_path:
            self.load_model_weights(model_weights_path)
    
    def load_model_weights(self, weights_path: str) -> None:
        """Load pre-trained model weights."""
        try:
            self.model.load_weights(weights_path)
            print(f"Successfully loaded model weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
    
    def process_image(
        self, 
        image_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict], pd.DataFrame, List[Dict]]:
        """
        Process a plant image through the complete CV pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple containing:
            - petri_dish: Extracted petri dish image
            - root_mask: Binary root mask from U-Net
            - root_data: Detailed root analysis data
            - root_df: DataFrame with root measurements
            - tip_coordinates: Root tip coordinates for robotic targeting
        """
        import cv2
        
        # Load the image
        image = cv2.imread(image_path, 0)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Step 1: Extract petri dish
        petri_dish, crop_coords = self.petri_extractor.extract_petri_dish(image)
        
        # Step 2: U-Net inference for root segmentation
        root_mask = self._predict_roots(petri_dish)
        
        # Step 3: Root architecture analysis
        root_data, root_df, tip_coordinates = self.root_analyzer.process_image(
            mask=root_mask, 
            petri_dish=petri_dish
        )
        
        return petri_dish, root_mask, root_data, root_df, tip_coordinates
    
    def _predict_roots(self, petri_dish: np.ndarray) -> np.ndarray:
        """
        Predict root masks using U-Net.
        
        Args:
            petri_dish: Extracted petri dish image
            
        Returns:
            Binary root mask
        """
        # Prepare image for U-Net
        padded_image = self._pad_image(petri_dish, self.patch_size)
        
        # Create patches
        patches = self._create_patches(padded_image, self.patch_size)
        
        # Predict on patches
        predictions = []
        for patch in patches:
            # Normalize patch
            patch_normalized = self._normalize_patch(patch)
            
            # Add batch dimension
            patch_input = patch_normalized[np.newaxis, ..., np.newaxis]
            
            # Predict
            pred = self.model.predict(patch_input, verbose=0)
            predictions.append(pred[0, ..., 0])
        
        # Reconstruct full prediction
        full_prediction = self._reconstruct_from_patches(
            predictions, padded_image.shape, self.patch_size
        )
        
        # Remove padding and resize to original petri dish size
        prediction_cropped = self._remove_padding(full_prediction, petri_dish.shape)
        
        # Apply threshold to get binary mask
        binary_mask = (prediction_cropped > 0.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def _pad_image(self, image: np.ndarray, patch_size: int) -> np.ndarray:
        """Pad image to be divisible by patch size."""
        h, w = image.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        # Pad with reflection
        padded = np.pad(
            image, 
            ((0, pad_h), (0, pad_w)), 
            mode='reflect'
        )
        
        return padded
    
    def _create_patches(
        self, 
        image: np.ndarray, 
        patch_size: int
    ) -> List[np.ndarray]:
        """Create overlapping patches from image."""
        patches = []
        h, w = image.shape
        
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                if patch.shape == (patch_size, patch_size):
                    patches.append(patch)
        
        return patches
    
    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Normalize patch using z-score normalization."""
        mean_val = np.mean(patch)
        std_val = np.std(patch)
        
        if std_val > 0:
            normalized = (patch - mean_val) / std_val
        else:
            normalized = patch - mean_val
        
        return normalized.astype(np.float32)
    
    def _reconstruct_from_patches(
        self, 
        predictions: List[np.ndarray], 
        original_shape: Tuple[int, int], 
        patch_size: int
    ) -> np.ndarray:
        """Reconstruct full image from patch predictions."""
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        patch_idx = 0
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                if patch_idx < len(predictions):
                    patch_pred = predictions[patch_idx]
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    
                    reconstructed[y:y_end, x:x_end] += patch_pred[:y_end-y, :x_end-x]
                    count[y:y_end, x:x_end] += 1
                    patch_idx += 1
        
        # Average overlapping regions
        reconstructed = np.divide(
            reconstructed, 
            count, 
            out=np.zeros_like(reconstructed), 
            where=count!=0
        )
        
        return reconstructed
    
    def _remove_padding(
        self, 
        padded_image: np.ndarray, 
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Remove padding and resize to target shape."""
        import cv2
        
        h, w = target_shape
        resized = cv2.resize(padded_image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def get_root_tips_for_robotics(
        self, 
        tip_coordinates: List[Dict]
    ) -> List[Dict]:
        """
        Extract root tip coordinates formatted for robotic targeting.
        
        Args:
            tip_coordinates: Raw tip coordinates from root analysis
            
        Returns:
            List of formatted tip coordinates
        """
        robot_targets = []
        
        for col_data in tip_coordinates:
            for tip in col_data['tips']:
                robot_targets.append({
                    'id': len(robot_targets),
                    'column': col_data['column'],
                    'pixel_x': tip['tip_x'],
                    'pixel_y': tip['tip_y'],
                    'root_id': tip['root_id'],
                    'priority': tip.get('path_length', 0)  # Longer roots = higher priority
                })
        
        # Sort by priority (longest roots first)
        robot_targets.sort(key=lambda x: x['priority'], reverse=True)
        
        return robot_targets
    
    def visualize_pipeline_results(
        self,
        petri_dish: np.ndarray,
        root_mask: np.ndarray,
        root_data: List[Dict],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of pipeline results.
        
        Args:
            petri_dish: Original petri dish image
            root_mask: Binary root mask
            root_data: Root analysis data
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        import cv2
        
        # Create visualization
        viz = self.root_analyzer.visualize_results(
            root_data, root_mask, petri_dish
        )
        
        if save_path:
            cv2.imwrite(save_path, viz)
        
        return viz
    
    def get_pipeline_summary(
        self, 
        root_data: List[Dict], 
        tip_coordinates: List[Dict]
    ) -> Dict[str, Any]:
        """
        Get summary statistics from pipeline results.
        
        Args:
            root_data: Root analysis data
            tip_coordinates: Root tip coordinates
            
        Returns:
            Summary statistics
        """
        total_tips = sum(len(col['tips']) for col in tip_coordinates)
        
        if root_data:
            root_lengths = [root['length'] for root in root_data]
            primary_root_length = max(root_lengths) if root_lengths else 0
            average_root_length = np.mean(root_lengths) if root_lengths else 0
        else:
            primary_root_length = 0
            average_root_length = 0
        
        return {
            'total_roots_detected': len(root_data),
            'total_root_tips': total_tips,
            'primary_root_length': primary_root_length,
            'average_root_length': average_root_length,
            'tips_per_column': [len(col['tips']) for col in tip_coordinates]
        }


def cv_inference(image_path: str, model_weights_path: Optional[str] = None) -> Tuple:
    """
    Convenience function for quick CV pipeline inference.
    
    Args:
        image_path: Path to input image
        model_weights_path: Optional path to model weights
        
    Returns:
        Pipeline results tuple
    """
    pipeline = CVPipeline(model_weights_path=model_weights_path)
    return pipeline.process_image(image_path) 