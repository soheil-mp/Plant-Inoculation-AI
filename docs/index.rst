Plant Inoculation AI Documentation
=================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/package_manager-poetry-blue.svg
   :target: https://python-poetry.org/
   :alt: Poetry

Welcome to Plant Inoculation AI, an AI-powered automated plant root inoculation system that combines computer vision, deep learning, and robotics to precisely target and inoculate plant root tips.

Features
--------

🔬 **Computer Vision Pipeline**
  - Petri dish detection and extraction
  - Plant instance segmentation using traditional CV
  - Deep learning segmentation with U-Net + MobileNet
  - Root architecture analysis and skeletonization
  - Precise root tip detection for robotic targeting

🤖 **Robotics & Control**
  - OT-2 robot simulation using PyBullet
  - PID controllers with anti-windup protection
  - Reinforcement learning with SAC algorithm
  - Hybrid control blending PID and RL approaches
  - Sub-millimeter precision targeting

🧠 **Machine Learning**
  - Custom U-Net architecture with multi-loss optimization
  - Mixed precision training for GPU acceleration
  - Transfer learning with pre-trained MobileNet
  - Real-time inference with patch-based processing

Quick Start
-----------

Installation::

    # Install with Poetry (recommended)
    poetry install
    
    # For GPU support
    poetry install --extras gpu
    
    # Install everything
    poetry install --extras all

Basic Usage::

    from plant_inoculation_ai import CVPipeline, PIDController
    
    # Initialize computer vision pipeline
    pipeline = CVPipeline(patch_size=960)
    
    # Process plant image
    results = pipeline.process_image("plant_image.jpg")
    
    # Initialize robot controller
    controller = PIDController(
        kp=[1.0, 1.0, 1.0],
        ki=[0.1, 0.1, 0.1], 
        kd=[0.05, 0.05, 0.05]
    )

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/computer_vision
   api/models
   api/robotics
   api/pipeline
   api/data
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   
   development/setup
   development/testing
   development/contributing
   development/deployment

.. toctree::
   :maxdepth: 1
   :caption: Additional Information
   
   changelog
   license
   bibliography

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 