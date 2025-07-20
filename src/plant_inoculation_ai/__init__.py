"""
Plant Inoculation AI: AI-powered automated plant root inoculation system.

This package provides a comprehensive solution for automated plant root 
inoculation combining computer vision, deep learning, and robotics.

Modules:
    computer_vision: Image processing, segmentation, and analysis
    models: Deep learning models for root detection and segmentation
    robotics: Robot simulation, control, and automation
    pipeline: End-to-end integration pipelines
    data: Data loading and preprocessing utilities
    utils: Helper functions and utilities
"""

__version__ = "0.1.0"
__author__ = "Plant-AI Team"
__email__ = "team@plant-ai.com"

# Core imports for convenience (with graceful failure handling)
_available_modules = []

try:
    from plant_inoculation_ai.robotics.controllers.pid import PIDController
    _available_modules.append("PIDController")
except ImportError:
    PIDController = None

try:
    from plant_inoculation_ai.computer_vision.segmentation import (
        PlantSegmenter
    )
    _available_modules.append("PlantSegmenter")
except ImportError:
    PlantSegmenter = None

try:
    from plant_inoculation_ai.computer_vision.root_analysis import (
        RootArchitectureAnalyzer
    )
    _available_modules.append("RootArchitectureAnalyzer")
except ImportError:
    RootArchitectureAnalyzer = None

try:
    from plant_inoculation_ai.models.unet import UNetModel
    _available_modules.append("UNetModel")
except ImportError:
    UNetModel = None

try:
    from plant_inoculation_ai.pipeline.cv_pipeline import CVPipeline
    _available_modules.append("CVPipeline")
except ImportError:
    CVPipeline = None

__all__ = [
    "PIDController",
    "PlantSegmenter",
    "RootArchitectureAnalyzer", 
    "UNetModel",
    "CVPipeline",
    "__version__",
    "_available_modules",
] 