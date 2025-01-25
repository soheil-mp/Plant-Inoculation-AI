# Import the libraries
import cv2
import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

class RootArchitectureAnalyzer:
    def __init__(self):
        """Initialize the root architecture analyzer."""
        # Parameters for root detection
        self.min_area = 50  # Reduced minimum area
        self.min_length = 10  # Reduced minimum length
        
    def process_image(self, mask: np.ndarray, petri_dish: Optional[np.ndarray] = None) -> Tuple[List[Dict], pd.DataFrame, List[Dict]]:
        """
        Main processing pipeline for root analysis.
        Args:
            mask: Binary mask of the roots (already extracted from petri dish)
            petri_dish: Optional extracted petri dish image for visualization and analysis
            
        Returns:
            - root_data: List of dictionaries containing root information
            - df: DataFrame with detailed root analysis
            - tip_coordinates: List of dictionaries containing root tip coordinates for each column
        """
        # Process roots
        root_data = []
        root_masks = self._segment_roots(mask)
        tip_coordinates = []  # New list to store tip coordinates
        
        # Get dimensions directly from the petri dish or mask
        height, width = mask.shape
        
        # Calculate the width of each column
        column_width = width / 5
        
        # Create column boundaries
        column_boundaries = [int(i * column_width) for i in range(6)]
        
        # Initialize columns
        columns_roots = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        # Assign roots to columns based on their top point
        for root_info in root_masks:
            # Get the top point of the root
            y_coords, x_coords = np.where(root_info['mask'] > 0)
            if len(y_coords) == 0:
                continue
            
            # Find the topmost point
            top_y = min(y_coords)
            top_x_candidates = x_coords[y_coords == top_y]
            top_x = int(np.median(top_x_candidates))
            
            # Find the bottom-most point (root tip)
            bottom_y = max(y_coords)
            bottom_x_candidates = x_coords[y_coords == bottom_y]
            bottom_x = int(np.median(bottom_x_candidates))
            
            # Determine which column this root belongs to
            for i in range(5):
                if column_boundaries[i] <= top_x < column_boundaries[i + 1]:
                    column_number = i + 1
                    root_info['column'] = column_number
                    root_info['tip_coordinates'] = (bottom_x, bottom_y)  # Store tip coordinates
                    columns_roots[column_number].append(root_info)
                    break
        
        # Process each column
        for column_num in range(1, 6):
            roots_in_column = columns_roots[column_num]
            if roots_in_column:
                # Keep only the longest root in this column
                longest_root = max(roots_in_column, key=lambda x: x['length'])
                root_data.append({
                    'column': column_num,
                    'has_root': True,
                    'length': longest_root['length'],
                    'area': longest_root['area'],
                    'path': longest_root['path'],
                    'mask': longest_root['mask'],
                    'bounds': (column_boundaries[column_num-1], column_boundaries[column_num])
                })
                # Add tip coordinates for this column
                tip_coordinates.append({
                    'column': column_num,
                    'tip_x': longest_root['tip_coordinates'][0],
                    'tip_y': longest_root['tip_coordinates'][1]
                })
            else:
                # Add empty column data
                root_data.append({
                    'column': column_num,
                    'has_root': False,
                    'length': 0,
                    'area': 0,
                    'bounds': (column_boundaries[column_num-1], column_boundaries[column_num])
                })
                # Add null tip coordinates for empty column
                tip_coordinates.append({
                    'column': column_num,
                    'tip_x': None,
                    'tip_y': None
                })
        
        # Sort by column number
        root_data = sorted(root_data, key=lambda x: x['column'])
        tip_coordinates = sorted(tip_coordinates, key=lambda x: x['column'])
        
        # Create DataFrame
        df = self.create_root_dataframe(root_data)
        
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
    
    def _analyze_single_root(self, mask: np.ndarray) -> Optional[Dict]:
        """Analyze a single root mask."""
        # Get skeleton
        skeleton = self._get_skeleton(mask)
        if skeleton is None:
            print("Failed to get skeleton")
            return None
            
        # Find endpoints
        endpoints = self._find_endpoints(skeleton)
        print(f"Found {len(endpoints)} endpoints")
        
        if len(endpoints) < 2:
            print("Not enough endpoints")
            return None
            
        # Find primary root path
        path_data = self._find_primary_path(skeleton, endpoints)
        if path_data is None:
            print("Failed to find primary path")
            return None
            
        return {
            'mask': mask,
            'skeleton': skeleton,
            'endpoints': endpoints,
            'path': path_data['path'],
            'length': path_data['length'],
            'start': path_data['start'],
            'end': path_data['end']
        }
    
    def _get_skeleton(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract skeleton from mask."""
        binary = mask > 0
        skeleton = skeletonize(binary)
        if not np.any(skeleton):
            return None
        return skeleton.astype(np.uint8)
    
    def _find_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """Find endpoints in skeleton."""
        endpoints = []
        y_coords, x_coords = np.where(skeleton > 0)
        
        for y, x in zip(y_coords, x_coords):
            # Get 3x3 neighborhood
            neighborhood = skeleton[
                max(0, y-1):min(y+2, skeleton.shape[0]),
                max(0, x-1):min(x+2, skeleton.shape[1])
            ]
            # Count neighbors (subtract center point)
            num_neighbors = np.sum(neighborhood) - 1
            if num_neighbors == 1:  # Endpoint has exactly one neighbor
                endpoints.append((y, x))
                
        return np.array(endpoints)
    

    def _find_primary_path(self, skeleton: np.ndarray, 
                         endpoints: np.ndarray) -> Optional[Dict]:
        """Find the primary root path."""
        if len(endpoints) < 2:
            print("Not enough endpoints")
            return None
            
        # Get all skeleton points
        y_coords, x_coords = np.where(skeleton > 0)
        
        # Find the absolute topmost point in the skeleton
        top_y = np.min(y_coords)
        top_candidates = [(y, x) for y, x in zip(y_coords, x_coords) if y == top_y]
        
        # Use the topmost point as start, even if it's not in endpoints
        start = top_candidates[0]  # Take the first if multiple points at same height
        
        # Try paths to all endpoints that are below the start point
        max_length = 0
        best_path = None
        best_end = None
        
        for end in endpoints:
            end = tuple(end)
            if end[0] <= start[0]:  # Skip points above or at same level
                continue
                
            path = self._find_path(skeleton, start, end)
            if path is not None and len(path) > max_length:
                max_length = len(path)
                best_path = path
                best_end = end
        
        if best_path is None:
            print("No valid path found")
            return None
            
        return {
            'path': best_path,
            'length': max_length,
            'start': start,
            'end': best_end
        }

    def _find_path(self, skeleton: np.ndarray, start: Tuple[int, int], 
                   end: Tuple[int, int]) -> Optional[np.ndarray]:
        """Find path between two points in skeleton."""
        # Create cost array
        costs = np.full_like(skeleton, fill_value=1000, dtype=float)
        costs[skeleton > 0] = 1
        
        try:
            # Note: route_through_array expects (start, end) not (start_point, end_point)
            indices, _ = route_through_array(
                costs,
                start=start,  # Changed from start_point
                end=end,      # Changed from end_point
                fully_connected=True,
                geometric=True
            )
            path = np.array(indices)
            
            # Verify path follows skeleton
            path_mask = np.zeros_like(skeleton)
            path_mask[path[:, 0], path[:, 1]] = 1
            overlap = np.sum(path_mask * skeleton) / len(path)
            
            # Return path only if it follows skeleton closely
            if overlap >= 0.9:  # At least 90% overlap
                return path
            else:
                print(f"Path rejected: only {overlap:.2%} overlap with skeleton")
                return None
                
        except Exception as e:
            print(f"Path finding error: {e}")
            return None

    def create_root_dataframe(self, root_data: List[Dict]) -> pd.DataFrame:
        """Create a DataFrame containing detailed information about each root branch."""
        rows = []
        
        for root in root_data:
            column_num = root['column']
            has_root = root.get('has_root', False)
            
            if has_root and 'path' in root:  # Check if this is a valid root with path data
                path = root['path']
                # Process each segment in the path
                for i in range(len(path) - 1):
                    src = path[i]
                    dst = path[i + 1]
                    
                    # Calculate Euclidean distance
                    euclidean_distance = np.sqrt(
                        (dst[0] - src[0])**2 + (dst[1] - src[1])**2
                    )
                    
                    # Create row
                    row = {
                        'skeleton-id': column_num,  # Use column number as skeleton-id
                        'node-id-src': i,
                        'node-id-dst': i + 1,
                        'branch-distance': euclidean_distance,
                        'branch-type': 1,  # Primary root = 1
                        'mean-pixel-value': 1.0,
                        'stdev-pixel-value': 0.0,
                        'image-coord-src-0': src[0],
                        'image-coord-src-1': src[1],
                        'image-coord-dst-0': dst[0],
                        'image-coord-dst-1': dst[1],
                        'coord-src-0': src[0],
                        'coord-src-1': src[1],
                        'coord-dst-0': dst[0],
                        'coord-dst-1': dst[1],
                        'euclidean-distance': euclidean_distance,
                        'has_root': True
                    }
                    rows.append(row)
            else:
                # Add a single row for empty columns
                row = {
                    'skeleton-id': column_num,
                    'node-id-src': 0,
                    'node-id-dst': 0,
                    'branch-distance': 0,
                    'branch-type': 0,  # No root = 0
                    'mean-pixel-value': 0.0,
                    'stdev-pixel-value': 0.0,
                    'image-coord-src-0': 0,
                    'image-coord-src-1': 0,
                    'image-coord-dst-0': 0,
                    'image-coord-dst-1': 0,
                    'coord-src-0': 0,
                    'coord-src-1': 0,
                    'coord-dst-0': 0,
                    'coord-dst-1': 0,
                    'euclidean-distance': 0,
                    'has_root': False
                }
                rows.append(row)
        
        # Create DataFrame
        if not rows:  # If no rows were created, create a dummy row
            rows.append({
                'skeleton-id': 0,
                'node-id-src': 0,
                'node-id-dst': 0,
                'branch-distance': 0,
                'branch-type': 0,
                'mean-pixel-value': 0.0,
                'stdev-pixel-value': 0.0,
                'image-coord-src-0': 0,
                'image-coord-src-1': 0,
                'image-coord-dst-0': 0,
                'image-coord-dst-1': 0,
                'coord-src-0': 0,
                'coord-src-1': 0,
                'coord-dst-0': 0,
                'coord-dst-1': 0,
                'euclidean-distance': 0,
                'has_root': False
            })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure correct column order
        columns = [
            'skeleton-id', 'node-id-src', 'node-id-dst', 'branch-distance',
            'branch-type', 'mean-pixel-value', 'stdev-pixel-value',
            'image-coord-src-0', 'image-coord-src-1', 'image-coord-dst-0',
            'image-coord-dst-1', 'coord-src-0', 'coord-src-1', 'coord-dst-0',
            'coord-dst-1', 'euclidean-distance', 'has_root'
        ]
        
        return df[columns]

    def visualize_results(self, root_data: List[Dict], 
                        original_mask: Optional[np.ndarray] = None,
                        petri_dish: Optional[np.ndarray] = None,
                        figsize: Tuple[int, int] = (12, 12)) -> None:
        """
        Visualize the divided sections with different colors.
        Args:
            root_data: List of dictionaries containing root information
            original_mask: Optional binary mask of the roots
            petri_dish: Optional extracted petri dish image for visualization
            figsize: Figure size for the plot
        """
        if not root_data:
            print("No roots to visualize")
            return
        
        # Create figure
        plt.figure(figsize=figsize, facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')
        
        # Show the petri dish image/mask
        if petri_dish is not None:
            plt.imshow(petri_dish, cmap='gray', alpha=0.3)
        if original_mask is not None:
            plt.imshow(original_mask, cmap='gray', alpha=0.1)
        
        # Define colors for each column
        colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#FF3366', '#33FF99']
        
        # Get image dimensions for column boundaries
        if petri_dish is not None:
            height, width = petri_dish.shape
        elif original_mask is not None:
            height, width = original_mask.shape
        else:
            # Try to get dimensions from root data
            max_x = max(data.get('bounds', (0, 0))[1] for data in root_data if data.get('bounds'))
            height, width = max_x, max_x
        
        # Calculate column boundaries
        column_width = width / 5
        column_boundaries = [int(i * column_width) for i in range(6)]
        
        # Draw vertical lines for each column boundary
        for i, x in enumerate(column_boundaries):
            plt.axvline(x=x, color='white', linestyle='--', alpha=0.3)
            
            # Add column number at the top (except for the last boundary)
            if i < 5:
                column_center = x + column_width/2
                plt.text(column_center, 20, f'Column {i+1}', 
                        color='white', ha='center', va='bottom', alpha=0.5)
        
        # Plot roots
        for data in root_data:
            if data.get('has_root', False) and 'path' in data:
                column = data['column']
                color = colors[column - 1]
                path = data['path']
                
                # Plot the main path with glow effect
                for width, alpha in [(8, 0.1), (6, 0.2), (4, 0.4), (2, 1.0)]:
                    plt.plot(path[:, 1], path[:, 0], '-', 
                            color=color, linewidth=width, alpha=alpha)
                
                # Add endpoints
                start_point = path[0]
                end_point = path[-1]
                
                # Start point (green glow)
                plt.scatter(start_point[1], start_point[0], c='white', s=100, alpha=1, zorder=5)
                plt.scatter(start_point[1], start_point[0], c='#00FF00', s=200, alpha=0.5, zorder=4)
                plt.scatter(start_point[1], start_point[0], c='#00FF00', s=300, alpha=0.2, zorder=3)
                
                # End point (red glow)
                plt.scatter(end_point[1], end_point[0], c='white', s=100, alpha=1, zorder=5)
                plt.scatter(end_point[1], end_point[0], c='#FF0000', s=200, alpha=0.5, zorder=4)
                plt.scatter(end_point[1], end_point[0], c='#FF0000', s=300, alpha=0.2, zorder=3)
                
                # Add root tip marker (yellow star with glow effect)
                if 'tip_coordinates' in data:
                    tip_x, tip_y = data['tip_coordinates']
                    plt.scatter(tip_x, tip_y, c='white', marker='*', s=200, alpha=1, zorder=7)
                    plt.scatter(tip_x, tip_y, c='yellow', marker='*', s=300, alpha=0.5, zorder=6)
                    plt.scatter(tip_x, tip_y, c='yellow', marker='*', s=400, alpha=0.2, zorder=5)
                    
                    # Add tip coordinates annotation
                    plt.annotate(
                        f'Tip ({tip_x}, {tip_y})',
                        xy=(tip_x, tip_y),
                        xytext=(tip_x + 30, tip_y + 30),
                        color='yellow',
                        fontsize=10,
                        bbox=dict(facecolor='black', edgecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='yellow', connectionstyle='arc3,rad=0.2')
                    )
                
                # Add length annotation
                mid_point = path[len(path)//2]
                plt.annotate(
                    f'Root {column}: {int(data["length"])}px',
                    xy=(mid_point[1], mid_point[0]),
                    xytext=(mid_point[1] + 50, mid_point[0]),
                    color=color,
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(facecolor='black', edgecolor=color, alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color=color, connectionstyle='arc3,rad=0.2')
                )
        
        plt.title('Root System Architecture Analysis', 
                color='white', fontsize=16, pad=20, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _get_roi_coordinates(self, original_image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get ROI coordinates using extract_petri_dish logic.
        """
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Otsu's thresholding to separate dish from background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (should be the Petri dish)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate the size of the square crop (use larger dimension)
        size = max(w, h)
        
        # Calculate new coordinates to center the dish
        center_x = x + w//2
        center_y = y + h//2
        
        # Calculate crop coordinates
        x1 = max(0, center_x - size//2)
        y1 = max(0, center_y - size//2)
        x2 = min(original_image.shape[1], x1 + size)
        y2 = min(original_image.shape[0], y1 + size)
        
        return x1, x2, y1, y2

    def divide_into_sections(self, mask: np.ndarray, n_sections: int = 5) -> List[np.ndarray]:
        """Divide the root system into n equally spaced sections based on root presence area."""
        # Get ROI bounds
        bounds = self._get_roi_bounds(mask)
        if bounds is None:
            return []
        
        min_x, max_x, min_y, max_y = bounds
        root_width = max_x - min_x
        
        # Calculate equal section widths
        section_width = root_width / n_sections
        
        # Create section boundaries at equal intervals
        section_boundaries = [int(min_x + (i * section_width)) for i in range(n_sections)]
        section_boundaries.append(max_x)  # Add the final boundary
        
        # Create sections
        sections = []
        for i in range(n_sections):
            start_x = section_boundaries[i]
            end_x = section_boundaries[i + 1]
            
            # Create section mask
            section_mask = np.zeros_like(mask)
            section_mask[:, start_x:end_x + 1] = mask[:, start_x:end_x + 1]
            sections.append(section_mask)
        
        return sections

    def visualize_sections(self, mask: np.ndarray, sections: List[np.ndarray], 
                          figsize: Tuple[int, int] = (15, 8)) -> None:
        """Visualize the divided sections with different colors."""
        plt.figure(figsize=figsize, facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')
        
        colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#FF3366', '#33FF99']
        
        # Plot original mask in gray
        plt.imshow(mask, cmap='gray', alpha=0.3)
        
        # Get ROI bounds
        bounds = self._get_roi_bounds(mask)
        if bounds:
            min_x, max_x, min_y, max_y = bounds
            root_width = max_x - min_x
            section_width = root_width / len(sections)
            
            # Plot equally spaced vertical lines
            for i in range(len(sections) + 1):
                x = int(min_x + (i * section_width))
                color = colors[min(i, len(colors)-1)]
                plt.axvline(x=x, color=color, linestyle='--', alpha=0.8)
        
        plt.title('Root System Sections', color='white', fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _get_root_area_bounds(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get the actual root area bounds using contour detection, similar to Petri dish extraction.
        """
        # Apply threshold to ensure binary image
        _, thresh = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the bounding rectangle of all contours combined
        x_coords = []
        y_coords = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
        
        # Get the overall bounds
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return min_x, max_x, min_y, max_y