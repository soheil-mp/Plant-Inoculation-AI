"""
Root System Architecture (RSA) analysis module.

This module provides functionality for analyzing root systems, extracting
branching structures, and detecting root tips for robotic targeting.
"""

from typing import List, Tuple, Dict, Optional, Any
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skimage.graph import route_through_array


class RootArchitectureAnalyzer:
    """Analyzes root system architecture from binary masks."""
    
    def __init__(self, min_area: int = 50, min_length: int = 10):
        """
        Initialize the root architecture analyzer.
        
        Args:
            min_area: Minimum area for valid roots
            min_length: Minimum length for valid roots
        """
        self.min_area = min_area
        self.min_length = min_length
    
    def process_image(
        self, 
        mask: np.ndarray, 
        petri_dish: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict], pd.DataFrame, List[Dict]]:
        """
        Main processing pipeline for root analysis.
        
        Args:
            mask: Binary mask of the roots (already extracted from petri dish)
            petri_dish: Optional extracted petri dish image for visualization
            
        Returns:
            - root_data: List of dictionaries containing root information
            - df: DataFrame with detailed root analysis
            - tip_coordinates: List of dictionaries containing root tip coords
        """
        # Process roots
        root_data = []
        root_masks = self._segment_roots(mask)
        tip_coordinates = []
        
        # Get dimensions directly from the petri dish or mask
        height, width = mask.shape
        
        # Calculate the width of each column
        column_width = width / 5
        
        # Initialize tip coordinates for each column
        for col in range(5):
            tip_coordinates.append({
                'column': col,
                'column_center_x': (col + 0.5) * column_width,
                'tips': []
            })
        
        # Process each root
        for i, root_info in enumerate(root_masks):
            root_mask = root_info['mask']
            path = root_info['path']
            
            # Get root tips (endpoints of the path)
            if len(path) >= 2:
                tip1 = path[0]
                tip2 = path[-1]
                
                # Determine which tip is the actual growing tip (bottommost)
                if tip1[0] > tip2[0]:  # tip1 is lower (larger y coordinate)
                    growing_tip = tip1
                else:
                    growing_tip = tip2
                
                # Determine which column this tip belongs to
                tip_x = growing_tip[1]  # x coordinate
                column_idx = min(4, int(tip_x / column_width))
                
                # Add tip to the appropriate column
                tip_coordinates[column_idx]['tips'].append({
                    'root_id': i,
                    'tip_x': float(tip_x),
                    'tip_y': float(growing_tip[0]),
                    'path_length': len(path)
                })
            
            # Store root data
            root_data.append({
                'root_id': i,
                'area': root_info['area'],
                'length': root_info['length'],
                'path': path,
                'mask': root_mask
            })
        
        # Create DataFrame for analysis
        df_data = []
        for col_data in tip_coordinates:
            for tip in col_data['tips']:
                df_data.append({
                    'Column': col_data['column'],
                    'Root_ID': tip['root_id'],
                    'Tip_X': tip['tip_x'],
                    'Tip_Y': tip['tip_y'],
                    'Path_Length': tip['path_length']
                })
        
        df = pd.DataFrame(df_data)
        
        return root_data, df, tip_coordinates
    
    def _segment_roots(self, mask: np.ndarray) -> List[Dict]:
        """Separate individual roots from the full mask."""
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        # Extract each root mask and calculate metrics
        root_info = []
        for i in range(1, num_labels):
            root_mask = (labels == i).astype(np.uint8) * 255
            area = cv2.countNonZero(root_mask)
            
            # Get skeleton for more accurate length measurement
            skeleton = self._get_skeleton(root_mask)
            if skeleton is None:
                continue
            
            # Find endpoints
            endpoints = self._find_endpoints(skeleton)
            if len(endpoints) < 2:
                continue
            
            # Find the path
            path_data = self._find_primary_path(skeleton, endpoints)
            if path_data is None:
                continue
            
            # Use actual path length
            length = len(path_data['path'])
            
            # Only keep if length is sufficient
            if length >= self.min_length and area >= self.min_area:
                root_info.append({
                    'mask': root_mask,
                    'area': area,
                    'length': length,
                    'path': path_data['path']
                })
        
        # Sort by length (descending) and keep only the 5 longest roots
        root_info.sort(key=lambda x: x['length'], reverse=True)
        root_info = root_info[:5]
        
        return root_info
    
    def _get_skeleton(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Get skeleton of the root mask."""
        try:
            # Convert to binary
            binary = mask > 0
            
            # Apply skeletonization
            skeleton = skeletonize(binary)
            
            # Convert back to uint8
            skeleton = skeleton.astype(np.uint8) * 255
            
            return skeleton
        except Exception:
            return None
    
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in the skeleton."""
        # Convert to binary
        binary = skeleton > 0
        
        # Create kernel for finding endpoints
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
        
        # Convolve to find endpoints (pixels with only one neighbor)
        conv = cv2.filter2D(binary.astype(np.uint8), -1, kernel)
        
        # Endpoints have value 11 (center pixel + one neighbor)
        endpoints_mask = (conv == 11) & binary
        
        # Get coordinates
        y_coords, x_coords = np.where(endpoints_mask)
        endpoints = [(int(y), int(x)) for y, x in zip(y_coords, x_coords)]
        
        return endpoints
    
    def _find_primary_path(
        self, skeleton: np.ndarray, endpoints: List[Tuple[int, int]]
    ) -> Optional[Dict[str, Any]]:
        """Find the primary path between endpoints."""
        if len(endpoints) < 2:
            return None
        
        try:
            # Convert skeleton to cost array (inverse for pathfinding)
            cost_array = np.where(skeleton > 0, 1, 1000)
            
            # Find path between first two endpoints
            start = endpoints[0]
            end = endpoints[1]
            
            # Use route_through_array to find path
            indices, weight = route_through_array(
                cost_array, start, end, fully_connected=True
            )
            
            if indices is None or len(indices) == 0:
                return None
            
            return {
                'path': indices,
                'weight': weight,
                'start': start,
                'end': end
            }
        except Exception:
            return None
    
    def visualize_results(
        self, 
        root_data: List[Dict], 
        original_mask: np.ndarray,
        petri_dish: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize root analysis results.
        
        Args:
            root_data: Root analysis data
            original_mask: Original binary mask
            petri_dish: Optional petri dish image
            
        Returns:
            Visualization image
        """
        # Create visualization image
        if petri_dish is not None:
            if len(petri_dish.shape) == 2:
                viz = cv2.cvtColor(petri_dish, cv2.COLOR_GRAY2BGR)
            else:
                viz = petri_dish.copy()
        else:
            viz = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR)
        
        # Colors for different roots
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]
        
        # Draw each root path
        for i, root_info in enumerate(root_data):
            if i < len(colors):
                color = colors[i]
                path = root_info['path']
                
                # Draw path
                for j in range(len(path) - 1):
                    pt1 = (int(path[j][1]), int(path[j][0]))
                    pt2 = (int(path[j+1][1]), int(path[j+1][0]))
                    cv2.line(viz, pt1, pt2, color, 2)
                
                # Draw endpoints
                if len(path) >= 2:
                    start_pt = (int(path[0][1]), int(path[0][0]))
                    end_pt = (int(path[-1][1]), int(path[-1][0]))
                    cv2.circle(viz, start_pt, 5, color, -1)
                    cv2.circle(viz, end_pt, 5, color, -1)
        
        return viz
    
    def get_primary_root_length(self, root_data: List[Dict]) -> float:
        """
        Get the length of the primary (longest) root.
        
        Args:
            root_data: Root analysis data
            
        Returns:
            Primary root length in pixels
        """
        if not root_data:
            return 0.0
        
        # Primary root is the first one (longest after sorting)
        return float(root_data[0]['length'])
    
    def get_root_tips(self, tip_coordinates: List[Dict]) -> List[Dict]:
        """
        Extract all root tip coordinates.
        
        Args:
            tip_coordinates: Tip coordinates data
            
        Returns:
            List of all root tips with coordinates
        """
        all_tips = []
        for col_data in tip_coordinates:
            for tip in col_data['tips']:
                all_tips.append({
                    'column': col_data['column'],
                    'x': tip['tip_x'],
                    'y': tip['tip_y'],
                    'root_id': tip['root_id']
                })
        
        return all_tips 