# Plant Inoculation Robot

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/package_manager-poetry-blue.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GPU Support](https://img.shields.io/badge/GPU-Supported-green.svg)](https://www.tensorflow.org/install/gpu)

**Automated plant root inoculation system combining computer vision, deep learning, and robotics for precise root tip targeting.**

![PID Controller Demo](assets/PIDcontrollerdemo-ezgif.com-speed.gif)

[Quick Start](#quick-start) • [Documentation](#documentation) • [Features](#features) • [Development](#development) • [Performance](#performance)

</div>

## Overview

Plant Inoculation Robot is a solution to automate the process of inoculating plant root tips with high precision. The system integrates computer vision algorithms, deep learning models, and robotic control systems to achieve sub-millimeter accuracy in root tip targeting.


## System Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[Plant Image] --> B[Petri Dish Detection]
        B --> C[Image Preprocessing]
    end
    
    subgraph "Computer Vision Pipeline"
        C --> D[Plant Segmentation]
        D --> E[Root Analysis]
        E --> F[Tip Detection]
        F --> G[Coordinate Extraction]
    end
    
    subgraph "Control System"
        G --> H[PID Controller]
        G --> I[RL Agent]
        H --> J[Hybrid Controller]
        I --> J
        J --> K[Robot Control]
    end
    
    subgraph "Output Layer"
        K --> L[Precise Positioning]
        L --> M[Inoculation]
    end
    
    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style J fill:#fff3e0
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/plant-inoculation-ai.git
cd plant-inoculation-ai

# Install with Poetry (recommended)
poetry install

# For GPU acceleration
poetry install --extras gpu

# For development environment
poetry install --extras all
```

### Basic Usage

```python
import numpy as np
from plant_inoculation_ai import CVPipeline, PIDController

# Initialize computer vision pipeline
pipeline = CVPipeline(
    patch_size=960,
    model_weights_path="path/to/model.weights.h5"
)

# Process plant image and extract root tips
image_path = "plant_image.jpg"
results = pipeline.process_image(image_path)
root_tips = pipeline.get_root_tips_for_robotics(results)

# Initialize precision controller
controller = PIDController(
    kp=[1.0, 1.0, 1.0],  # Proportional gains [x, y, z]
    ki=[0.1, 0.1, 0.1],  # Integral gains
    kd=[0.05, 0.05, 0.05]  # Derivative gains
)

# Execute precise targeting
for target in root_tips:
    current_pos = np.array([0.0, 0.0, 0.15])
    target_pos = np.array([target['pixel_x'], target['pixel_y'], 0.2])
    
    # Compute control action
    action = controller.compute(current_pos, target_pos, dt=1/240)
    
    # Apply to robot (implement your interface)
    robot.move(action)
```

## Features

### Computer Vision Pipeline

```mermaid
flowchart LR
    subgraph "Image Processing"
        A[Raw Image] --> B[Petri Dish Detection]
        B --> C[Plant Segmentation]
        C --> D[Root Analysis]
    end
    
    subgraph "Deep Learning"
        D --> E[U-Net Model]
        E --> F[Root Mask]
        F --> G[Tip Detection]
    end
    
    subgraph "Output"
        G --> H[Target Coordinates]
        H --> I[Robot Control]
    end
    
    style A fill:#e3f2fd
    style I fill:#e8f5e8
```

| Component | Technology | Performance |
|-----------|------------|-------------|
| **Petri Dish Detection** | OpenCV Contour Analysis | ~50ms per image |
| **Plant Segmentation** | Traditional CV + Morphology | ~100ms per image |
| **Root Analysis** | U-Net + MobileNet | ~200ms per patch |
| **Tip Detection** | Skeletonization + Pathfinding | ~100ms per image |
| **End-to-End** | Optimized Pipeline | ~500ms per plant |

### Robotics & Control

- **PID Controllers**: Anti-windup protection with output saturation
- **Reinforcement Learning**: SAC algorithm for adaptive control
- **Hybrid Control**: Seamless PID + RL blending
- **Precision Targeting**: Sub-millimeter accuracy (±0.1mm)
- **Real-time Response**: <1ms controller latency

### Machine Learning

```mermaid
graph TD
    subgraph "Model Architecture"
        A[Input Image] --> B[Encoder: MobileNet]
        B --> C[Decoder: U-Net]
        C --> D[Output Mask]
    end
    
    subgraph "Training Process"
        E[Training Data] --> F[Data Augmentation]
        F --> G[Model Training]
        G --> H[Validation]
        H --> I[Model Save]
    end
    
    subgraph "Inference"
        J[New Image] --> K[Preprocessing]
        K --> L[Model Inference]
        L --> M[Post-processing]
        M --> N[Root Tips]
    end
    
    style A fill:#e1f5fe
    style N fill:#c8e6c9
```

- **Custom U-Net**: MobileNet backbone with transfer learning
- **Multi-loss Optimization**: Dice + Focal + Tversky + Boundary losses
- **GPU Acceleration**: Mixed precision training support
- **Performance Monitoring**: Real-time metrics and visualization

## Package Architecture

```
src/plant_inoculation_ai/
├── computer_vision/          # CV pipeline and analysis
│   ├── petri_dish.py        # Petri dish detection
│   ├── segmentation.py      # Plant segmentation
│   └── root_analysis.py     # Root architecture analysis
├── models/                  # Deep learning models
│   └── unet.py             # U-Net implementation
├── robotics/               # Robot control systems
│   └── controllers/        # PID and RL controllers
│       └── pid.py         # PID controller implementation
├── pipeline/               # End-to-end workflows
│   └── cv_pipeline.py     # Integrated CV pipeline
├── data/                   # Data loading utilities
├── utils/                  # Helper functions
└── core/                   # Core configurations
```

## Performance Benchmarks

### Computer Vision Performance

| Operation | Time (ms) | Hardware | Notes |
|-----------|-----------|----------|-------|
| Petri Dish Detection | 50 | CPU | OpenCV optimized |
| U-Net Inference | 200 | GPU (RTX 3080) | 960×960 patch |
| Root Analysis | 100 | CPU | Skeletonization |
| End-to-End Pipeline | 500 | Mixed | Full plant processing |

### Robotics Control Performance

```mermaid
graph LR
    subgraph "Control Metrics"
        A[Controller Latency] --> B[<1ms]
        C[RL Inference] --> D[~5ms]
        E[Positioning Accuracy] --> F[±0.1mm]
        G[Response Time] --> H[<10ms]
    end
    
    subgraph "Hardware Requirements"
        I[GPU Acceleration] --> J[Real-time]
        K[Calibrated System] --> L[High Precision]
    end
    
    style B fill:#c8e6c9
    style F fill:#c8e6c9
```

| Metric | Value | Conditions |
|--------|-------|------------|
| Controller Latency | <1ms | Real-time loop |
| RL Inference | ~5ms | GPU acceleration |
| Positioning Accuracy | ±0.1mm | Calibrated system |
| Response Time | <10ms | Target acquisition |

## Deployment

### Docker Containerization

```bash
# Build development environment
docker build -t plant-inoculation-ai:dev .

# Build production image
docker build --target production -t plant-inoculation-ai:prod .

# Run with GPU support
docker run --gpus all -it plant-inoculation-ai:dev

# Run Jupyter environment
docker run --gpus all -p 8888:8888 plant-inoculation-ai:dev jupyter lab
```

### Cloud Deployment

```mermaid
graph TB
    subgraph "Infrastructure"
        A[Load Balancer] --> B[Application Servers]
        B --> C[GPU Clusters]
        C --> D[Database]
    end
    
    subgraph "Security"
        E[Non-root Containers] --> F[Dependency Scanning]
        F --> G[Health Checks]
    end
    
    subgraph "Monitoring"
        H[Performance Metrics] --> I[Alerting]
        I --> J[Logging]
    end
    
    style A fill:#e3f2fd
    style G fill:#fff3e0
```

The package is designed for seamless cloud deployment with:
- **Security**: Non-root containers, dependency scanning
- **Scalability**: Multi-GPU support, load balancing ready
- **Monitoring**: Health checks, performance metrics
- **CI/CD**: Automated testing and deployment pipelines

## Development

### Development Environment Setup

```bash
# Install development dependencies
poetry install --with dev

# Setup pre-commit hooks
poetry run pre-commit install

# Run code quality checks
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run flake8 src/ tests/
poetry run mypy src/
poetry run bandit -r src/
```

### Testing Framework

```bash
# Run comprehensive test suite
poetry run pytest

# Generate coverage report
poetry run pytest --cov=src --cov-report=html

# Run specific test categories
poetry run pytest -m "unit"           # Unit tests
poetry run pytest -m "integration"    # Integration tests
poetry run pytest -m "not slow"       # Skip slow tests
```

### Documentation

```bash
# Build Sphinx documentation
cd docs && poetry run sphinx-build -b html . _build/html

# Serve documentation locally
python -m http.server 8000 --directory _build/html
```

## GPU Support

### GPU Availability Check

```python
from plant_inoculation_ai.utils.gpu_utils import print_gpu_summary

# Comprehensive GPU information
print_gpu_summary()

# Get recommended configuration
config = get_recommended_gpu_config()
print(f"Recommended batch size: {config['batch_size_multiplier']}x")
```

### GPU Configuration

```python
from plant_inoculation_ai.utils.gpu_utils import (
    configure_tensorflow_gpu, 
    set_tensorflow_mixed_precision
)

# Configure TensorFlow GPU
configure_tensorflow_gpu(memory_growth=True)

# Enable mixed precision for faster training
set_tensorflow_mixed_precision(enabled=True)
```

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

