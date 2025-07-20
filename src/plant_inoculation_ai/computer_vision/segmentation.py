"""
Plant segmentation module.

This module provides functionality for segmenting individual plants in images
using traditional computer vision techniques.
"""

from typing import List, Tuple, Dict, Any
import cv2
import numpy as np


class PlantSegmenter:
    """Segments individual plants using traditional computer vision methods."""
    
    def __init__(
        self,
        threshold_value: int = 140,
        min_area: int = 1500,
        min_height: int = 200,
        height_width_ratio: float = 0.4,
        max_width_ratio: float = 0.5,
        bottom_threshold: float = 0.95
    ):
        """
        Initialize the plant segmenter.
        
        Args:
            threshold_value: Binary threshold value
            min_area: Minimum area for valid plants
            min_height: Minimum height for valid plants
            height_width_ratio: Minimum height/width ratio
            max_width_ratio: Maximum width relative to image width
            bottom_threshold: Maximum y position relative to image height
        """
        self.threshold_value = threshold_value
        self.min_area = min_area
        self.min_height = min_height
        self.height_width_ratio = height_width_ratio
        self.max_width_ratio = max_width_ratio
        self.bottom_threshold = bottom_threshold
        
        # Plant colors (BGR format)
        self.plant_colors = [
            [180, 120, 240],  # Soft purple
            [240, 180, 120],  # Soft orange
            [120, 240, 180],  # Soft mint
            [180, 240, 120],  # Soft lime
            [240, 120, 180],  # Soft pink
        ]
        
        self.background_color = [20, 20, 30]  # BGR format
    
    def extract_petri_dish(self, image: np.ndarray) -> np.ndarray:
        """
        Extract square region containing the Petri dish.
        
        Args:
            image: Input image
            
        Returns:
            Cropped square image containing the Petri dish
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Otsu's thresholding to separate dish from background
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find the largest contour (should be the Petri dish)
        largest_contour = max(contours, key=cv2.contourArea)
        
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
        
        return cropped
    
    def segment_plants(self, image: np.ndarray) -> np.ndarray:
        """
        Segment individual plants in the image using traditional CV methods.
        
        Args:
            image: Input BGR image
            
        Returns:
            colored_mask: Image with each plant colored differently
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize image
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Enhance contrast before thresholding
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Apply thresholding
        _, binary = cv2.threshold(
            enhanced, self.threshold_value, 255, cv2.THRESH_BINARY_INV
        )

        kernel = np.ones((3, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Find connected components
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary
        )
        
        # Create output image with black background
        colored_mask = np.ones(
            (image.shape[0], image.shape[1], 3), dtype=np.uint8
        )
        colored_mask[:] = self.background_color

        # Collect valid plants
        valid_plants = self._filter_valid_plants(retval, stats, image.shape)
        
        # Sort plants by x-coordinate (left to right)
        valid_plants.sort(key=lambda x: x[2])
        
        # Color each plant
        for color_idx, (plant_label, height, x) in enumerate(valid_plants):
            if color_idx < len(self.plant_colors):
                # Create mask for this plant
                plant_mask = (labels == plant_label)
                colored_mask[plant_mask] = self.plant_colors[color_idx]
        
        return colored_mask
    
    def _filter_valid_plants(
        self, retval: int, stats: np.ndarray, image_shape: Tuple[int, ...]
    ) -> List[Tuple[int, int, int]]:
        """
        Filter connected components to find valid plants.
        
        Args:
            retval: Number of connected components
            stats: Statistics for each component
            image_shape: Shape of the input image
            
        Returns:
            List of valid plant tuples (label, height, x_position)
        """
        valid_plants = []
        
        for i in range(1, retval):
            x, y, w, h, area = stats[i]
            
            # Apply filtering criteria
            if (
                area > self.min_area and
                h > self.min_height and
                h / w > self.height_width_ratio and
                y < image_shape[0] * self.bottom_threshold and
                w < image_shape[1] * self.max_width_ratio
            ):
                valid_plants.append((i, h, x))
        
        return valid_plants
    
    def get_plant_info(
        self, image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get detailed information about segmented plants.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing plant information
        """
        # Extract petri dish
        petri_dish = self.extract_petri_dish(image)
        
        # Segment plants
        segmented = self.segment_plants(petri_dish)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(petri_dish, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        _, binary = cv2.threshold(
            enhanced, self.threshold_value, 255, cv2.THRESH_BINARY_INV
        )
        
        kernel = np.ones((3, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary
        )
        
        valid_plants = self._filter_valid_plants(
            retval, stats, petri_dish.shape
        )
        
        plant_data = []
        for idx, (plant_label, height, x) in enumerate(valid_plants):
            if idx < len(self.plant_colors):
                x_pos, y_pos, w, h, area = stats[plant_label]
                centroid_x, centroid_y = centroids[plant_label]
                
                plant_data.append({
                    'id': idx,
                    'label': plant_label,
                    'area': area,
                    'height': h,
                    'width': w,
                    'x_position': x_pos,
                    'y_position': y_pos,
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'color': self.plant_colors[idx]
                })
        
        return {
            'petri_dish': petri_dish,
            'segmented_image': segmented,
            'binary_mask': binary,
            'plant_count': len(plant_data),
            'plants': plant_data
        } 