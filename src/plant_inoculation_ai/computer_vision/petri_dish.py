"""
Petri dish detection and extraction module.

This module provides functionality to detect and extract petri dishes from
images using traditional computer vision techniques.
"""

from typing import Tuple
import cv2
import numpy as np


class PetriDishExtractor:
    """Extracts petri dish regions from images using computer vision."""
    
    def __init__(self, blur_kernel_size: int = 5, min_area_ratio: float = 0.1):
        """
        Initialize the petri dish extractor.
        
        Args:
            blur_kernel_size: Size of the Gaussian blur kernel
            min_area_ratio: Minimum area ratio for valid contours
        """
        self.blur_kernel_size = blur_kernel_size
        self.min_area_ratio = min_area_ratio
    
    def extract_petri_dish(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract a square region containing the Petri dish from the input image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Tuple of (cropped_image, (x1, y1, x2, y2)) where coordinates 
            define the crop region
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        kernel_size = (self.blur_kernel_size, self.blur_kernel_size)
        blurred = cv2.GaussianBlur(gray, kernel_size, 0)
        
        # Use Otsu's thresholding to separate dish from background
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            raise ValueError("No contours found in image")
        
        # Find the largest contour (should be the Petri dish)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        contour_area = cv2.contourArea(largest_contour)
        image_area = image.shape[0] * image.shape[1]
        
        if contour_area < image_area * self.min_area_ratio:
            min_required = image_area * self.min_area_ratio
            raise ValueError(
                f"Largest contour too small: {contour_area} < {min_required}"
            )
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate the size of the square crop (use larger dimension)
        size = max(w, h)
        
        # Calculate new coordinates to center the dish
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate crop coordinates
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(image.shape[1], x1 + size)
        y2 = min(image.shape[0], y1 + size)
        
        # Crop the image
        cropped = image[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2, y2)
    
    def visualize_detection(
        self, image: np.ndarray, crop_coords: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Visualize the detected petri dish region on the original image.
        
        Args:
            image: Original image
            crop_coords: Crop coordinates (x1, y1, x2, y2)
            
        Returns:
            Image with detection visualization
        """
        viz_image = image.copy()
        if len(viz_image.shape) == 2:
            viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
        
        x1, y1, x2, y2 = crop_coords
        cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        return viz_image


def extract_petri_dish(image: np.ndarray) -> np.ndarray:
    """
    Convenience function for simple petri dish extraction.
    
    Args:
        image: Input image
        
    Returns:
        Extracted petri dish region
    """
    extractor = PetriDishExtractor()
    cropped, _ = extractor.extract_petri_dish(image)
    return cropped 