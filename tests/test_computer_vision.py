"""Tests for computer vision modules."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from plant_inoculation_ai.computer_vision.petri_dish import PetriDishExtractor
from plant_inoculation_ai.computer_vision.segmentation import PlantSegmenter
from plant_inoculation_ai.computer_vision.root_analysis import RootArchitectureAnalyzer


class TestPetriDishExtractor:
    """Test PetriDishExtractor class."""
    
    def test_init(self):
        """Test extractor initialization."""
        extractor = PetriDishExtractor()
        assert extractor.blur_kernel_size == 5
        assert extractor.min_area_ratio == 0.1
    
    def test_init_with_params(self):
        """Test extractor initialization with custom parameters."""
        extractor = PetriDishExtractor(blur_kernel_size=7, min_area_ratio=0.2)
        assert extractor.blur_kernel_size == 7
        assert extractor.min_area_ratio == 0.2
    
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    @patch('cv2.GaussianBlur')
    @patch('cv2.threshold')
    @patch('cv2.findContours')
    @patch('cv2.contourArea')
    @patch('cv2.boundingRect')
    def test_extract_petri_dish(
        self, 
        mock_bounding_rect,
        mock_contour_area,
        mock_find_contours,
        mock_threshold,
        mock_blur,
        mock_cvt_color,
        mock_imread
    ):
        """Test petri dish extraction."""
        # Mock setup
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mock_imread.return_value = test_image
        mock_cvt_color.return_value = np.ones((100, 100), dtype=np.uint8) * 128
        mock_blur.return_value = np.ones((100, 100), dtype=np.uint8) * 128
        mock_threshold.return_value = (127, np.ones((100, 100), dtype=np.uint8))
        mock_find_contours.return_value = ([np.array([[10, 10], [90, 90]])], None)
        mock_contour_area.return_value = 6400  # Large enough area
        mock_bounding_rect.return_value = (10, 10, 80, 80)
        
        extractor = PetriDishExtractor()
        result, coords = extractor.extract_petri_dish(test_image)
        
        # Verify the result
        assert result is not None
        assert len(coords) == 4
        assert all(isinstance(coord, int) for coord in coords)
    
    def test_extract_petri_dish_no_contours(self):
        """Test extraction when no contours are found."""
        with patch('cv2.findContours') as mock_find_contours:
            mock_find_contours.return_value = ([], None)
            
            extractor = PetriDishExtractor()
            test_image = np.ones((100, 100), dtype=np.uint8)
            
            with pytest.raises(ValueError, match="No contours found"):
                extractor.extract_petri_dish(test_image)


class TestPlantSegmenter:
    """Test PlantSegmenter class."""
    
    def test_init(self):
        """Test segmenter initialization."""
        segmenter = PlantSegmenter()
        assert segmenter.threshold_value == 140
        assert segmenter.min_area == 1500
        assert len(segmenter.plant_colors) == 5
    
    def test_init_with_params(self):
        """Test segmenter initialization with custom parameters."""
        segmenter = PlantSegmenter(
            threshold_value=120,
            min_area=1000,
            min_height=150
        )
        assert segmenter.threshold_value == 120
        assert segmenter.min_area == 1000
        assert segmenter.min_height == 150
    
    @patch('cv2.cvtColor')
    @patch('cv2.normalize')
    @patch('cv2.GaussianBlur')
    @patch('cv2.createCLAHE')
    @patch('cv2.threshold')
    @patch('cv2.dilate')
    @patch('cv2.connectedComponentsWithStats')
    def test_segment_plants(
        self,
        mock_connected_components,
        mock_dilate,
        mock_threshold,
        mock_clahe,
        mock_blur,
        mock_normalize,
        mock_cvt_color
    ):
        """Test plant segmentation."""
        # Mock setup
        test_image = np.ones((500, 500, 3), dtype=np.uint8) * 128
        
        # Mock CV operations
        mock_cvt_color.return_value = np.ones((500, 500), dtype=np.uint8) * 128
        mock_normalize.return_value = np.ones((500, 500), dtype=np.uint8) * 128
        mock_blur.return_value = np.ones((500, 500), dtype=np.uint8) * 128
        
        clahe_mock = Mock()
        clahe_mock.apply.return_value = np.ones((500, 500), dtype=np.uint8) * 128
        mock_clahe.return_value = clahe_mock
        
        mock_threshold.return_value = (127, np.ones((500, 500), dtype=np.uint8))
        mock_dilate.return_value = np.ones((500, 500), dtype=np.uint8)
        
        # Mock connected components with valid plants
        mock_stats = np.array([
            [0, 0, 0, 0, 0],  # Background
            [50, 50, 100, 300, 30000],  # Valid plant 1
            [200, 50, 100, 300, 30000],  # Valid plant 2
        ])
        mock_centroids = np.array([[0, 0], [100, 200], [250, 200]])
        mock_connected_components.return_value = (
            3, 
            np.ones((500, 500), dtype=np.int32), 
            mock_stats, 
            mock_centroids
        )
        
        segmenter = PlantSegmenter()
        result = segmenter.segment_plants(test_image)
        
        # Verify result
        assert result.shape == (500, 500, 3)
        assert result.dtype == np.uint8
    
    def test_filter_valid_plants(self):
        """Test plant filtering logic."""
        segmenter = PlantSegmenter()
        
        # Mock stats for different plant scenarios
        mock_stats = np.array([
            [0, 0, 0, 0, 0],  # Background
            [50, 50, 100, 300, 30000],  # Valid plant
            [200, 50, 50, 100, 1000],   # Too small area
            [300, 400, 100, 300, 30000], # Too close to bottom
            [400, 50, 200, 300, 30000],  # Too wide
        ])
        
        valid_plants = segmenter._filter_valid_plants(
            5, mock_stats, (500, 500)
        )
        
        # Only the first plant should be valid
        assert len(valid_plants) == 1
        assert valid_plants[0][0] == 1  # Plant label


class TestRootArchitectureAnalyzer:
    """Test RootArchitectureAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = RootArchitectureAnalyzer()
        assert analyzer.min_area == 50
        assert analyzer.min_length == 10
    
    def test_init_with_params(self):
        """Test analyzer initialization with custom parameters."""
        analyzer = RootArchitectureAnalyzer(min_area=100, min_length=20)
        assert analyzer.min_area == 100
        assert analyzer.min_length == 20
    
    def test_process_image_empty_mask(self):
        """Test processing with empty mask."""
        analyzer = RootArchitectureAnalyzer()
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        
        root_data, root_df, tip_coordinates = analyzer.process_image(
            mask=empty_mask
        )
        
        assert len(root_data) == 0
        assert len(root_df) == 0
        assert len(tip_coordinates) == 5  # 5 columns always created
        for col in tip_coordinates:
            assert len(col['tips']) == 0
    
    @patch('cv2.connectedComponents')
    def test_segment_roots_no_components(self, mock_connected_components):
        """Test root segmentation with no connected components."""
        mock_connected_components.return_value = (1, np.zeros((100, 100)))
        
        analyzer = RootArchitectureAnalyzer()
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        result = analyzer._segment_roots(mask)
        assert len(result) == 0
    
    def test_get_skeleton_empty_mask(self):
        """Test skeleton extraction with empty mask."""
        analyzer = RootArchitectureAnalyzer()
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        
        result = analyzer._get_skeleton(empty_mask)
        assert result is not None
        assert result.shape == (100, 100)
    
    def test_find_endpoints_empty_skeleton(self):
        """Test endpoint detection with empty skeleton."""
        analyzer = RootArchitectureAnalyzer()
        empty_skeleton = np.zeros((100, 100), dtype=np.uint8)
        
        endpoints = analyzer._find_endpoints(empty_skeleton)
        assert len(endpoints) == 0
    
    def test_get_primary_root_length_empty_data(self):
        """Test primary root length calculation with empty data."""
        analyzer = RootArchitectureAnalyzer()
        
        length = analyzer.get_primary_root_length([])
        assert length == 0.0
    
    def test_get_primary_root_length_with_data(self):
        """Test primary root length calculation with data."""
        analyzer = RootArchitectureAnalyzer()
        
        root_data = [
            {'length': 100},
            {'length': 50},
            {'length': 75}
        ]
        
        length = analyzer.get_primary_root_length(root_data)
        assert length == 100.0
    
    def test_get_root_tips_empty_coordinates(self):
        """Test root tip extraction with empty coordinates."""
        analyzer = RootArchitectureAnalyzer()
        
        tip_coordinates = [
            {'column': 0, 'tips': []},
            {'column': 1, 'tips': []},
        ]
        
        tips = analyzer.get_root_tips(tip_coordinates)
        assert len(tips) == 0
    
    def test_get_root_tips_with_data(self):
        """Test root tip extraction with data."""
        analyzer = RootArchitectureAnalyzer()
        
        tip_coordinates = [
            {
                'column': 0, 
                'tips': [
                    {'tip_x': 10.0, 'tip_y': 20.0, 'root_id': 0}
                ]
            },
            {
                'column': 1, 
                'tips': [
                    {'tip_x': 30.0, 'tip_y': 40.0, 'root_id': 1}
                ]
            },
        ]
        
        tips = analyzer.get_root_tips(tip_coordinates)
        assert len(tips) == 2
        assert tips[0]['column'] == 0
        assert tips[0]['x'] == 10.0
        assert tips[1]['column'] == 1
        assert tips[1]['y'] == 40.0


if __name__ == "__main__":
    pytest.main([__file__]) 