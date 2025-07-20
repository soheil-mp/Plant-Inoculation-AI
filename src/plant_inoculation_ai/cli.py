"""
Command Line Interface for Plant Inoculation AI.

This module provides a simple CLI for demonstrating package functionality.
"""

import sys
from typing import Dict


def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available."""
    dependencies = {}
    
    # Check core dependencies
    try:
        import numpy  # noqa: F401
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    try:
        import cv2  # noqa: F401
        dependencies['opencv'] = True
    except ImportError:
        dependencies['opencv'] = False
    
    try:
        import tensorflow  # noqa: F401
        dependencies['tensorflow'] = True
    except ImportError:
        dependencies['tensorflow'] = False
    
    try:
        import torch  # noqa: F401
        dependencies['torch'] = True
    except ImportError:
        dependencies['torch'] = False
    
    try:
        import pandas  # noqa: F401
        dependencies['pandas'] = True
    except ImportError:
        dependencies['pandas'] = False
    
    return dependencies


def show_package_info() -> None:
    """Display package information and module structure."""
    print("🌱 Plant Inoculation AI Package")
    print("=" * 50)
    print("Version: 0.1.0")
    print("Author: Plant-AI Team")
    print()
    
    print("📁 Package Structure:")
    print("src/plant_inoculation_ai/")
    print("├── computer_vision/     # CV pipeline and analysis")
    print("│   ├── petri_dish.py   # Petri dish detection")
    print("│   ├── segmentation.py # Plant segmentation")
    print("│   └── root_analysis.py # Root architecture analysis")
    print("├── models/             # Deep learning models")
    print("│   └── unet.py        # U-Net implementation")
    print("├── robotics/          # Robot control systems")
    print("│   └── controllers/   # PID and RL controllers")
    print("│       └── pid.py     # PID controller implementation")
    print("├── pipeline/          # End-to-end workflows")
    print("│   └── cv_pipeline.py # Integrated CV pipeline")
    print("├── data/              # Data loading utilities")
    print("├── utils/             # Helper functions")
    print("└── core/              # Core configurations")
    print()


def check_module_availability() -> None:
    """Check which modules can be imported."""
    print("🔍 Module Availability Check:")
    print("-" * 30)
    
    # Test module imports
    modules_to_test = [
        ("PID Controller", "plant_inoculation_ai.robotics.controllers.pid"),
        ("Computer Vision", "plant_inoculation_ai.computer_vision"),
        ("Models", "plant_inoculation_ai.models"),
        ("Pipeline", "plant_inoculation_ai.pipeline"),
        ("Data", "plant_inoculation_ai.data"),
        ("Utils", "plant_inoculation_ai.utils"),
    ]
    
    for module_name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"✅ {module_name:<20} - Available")
        except ImportError as e:
            print(f"❌ {module_name:<20} - Missing dependencies: {e}")
        except Exception as e:
            print(f"⚠️  {module_name:<20} - Error: {e}")
    
    print()


def check_dependencies_status() -> None:
    """Display dependency status."""
    print("📦 Dependencies Status:")
    print("-" * 30)
    
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✅ Installed" if available else "❌ Not found"
        print(f"{dep:<15} - {status}")
    
    print()
    
    # Calculate installation success rate
    installed = sum(deps.values())
    total = len(deps)
    success_rate = (installed / total) * 100
    
    success_msg = f"Installation Success Rate: {success_rate:.1f}%"
    print(f"{success_msg} ({installed}/{total})")
    print()


def show_features() -> None:
    """Display key features of the package."""
    print("🎯 Key Features:")
    print("-" * 20)
    print("🔬 Computer Vision Pipeline:")
    print("  • Petri dish detection and extraction")
    print("  • Plant segmentation using traditional CV")
    print("  • U-Net deep learning for root detection")
    print("  • Root architecture analysis")
    print("  • Root tip coordinate extraction")
    print()
    print("🤖 Robotics & Control:")
    print("  • OT-2 robot simulation (PyBullet)")
    print("  • PID controllers with anti-windup")
    print("  • Reinforcement learning (SAC)")
    print("  • Hybrid PID + RL control")
    print("  • Precision targeting system")
    print()
    print("🧠 Machine Learning:")
    print("  • Custom U-Net architecture")
    print("  • Multi-loss optimization")
    print("  • Mixed precision training")
    print("  • Real-time inference")
    print()


def show_usage_examples() -> None:
    """Show code usage examples."""
    print("💡 Usage Examples:")
    print("-" * 20)
    print("# Basic PID Controller")
    controller_import = "plant_inoculation_ai.robotics.controllers.pid"
    print(f"from {controller_import} import PIDController")
    print("controller = PIDController(")
    print("    kp=[1.0, 1.0, 1.0],")
    print("    ki=[0.1, 0.1, 0.1],")
    print("    kd=[0.05, 0.05, 0.05]")
    print(")")
    print()
    print("# Computer Vision Pipeline")
    print("from plant_inoculation_ai import CVPipeline")
    print("pipeline = CVPipeline(patch_size=960)")
    print("results = pipeline.process_image('plant.jpg')")
    print()


def main() -> None:
    """Main CLI function."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "info":
            show_package_info()
        elif command == "deps":
            check_dependencies_status()
        elif command == "modules":
            check_module_availability()
        elif command == "features":
            show_features()
        elif command == "examples":
            show_usage_examples()
        elif command == "all":
            show_package_info()
            check_dependencies_status()
            check_module_availability()
            show_features()
            show_usage_examples()
        else:
            print(f"Unknown command: {command}")
            show_help()
    else:
        # Default: show everything
        show_package_info()
        check_dependencies_status()
        check_module_availability()
        show_features()


def show_help() -> None:
    """Show help information."""
    print("🌱 Plant Inoculation AI CLI")
    print("Usage: python -m plant_inoculation_ai.cli [command]")
    print()
    print("Commands:")
    print("  info      - Show package information")
    print("  deps      - Check dependencies status")
    print("  modules   - Check module availability")
    print("  features  - Show key features")
    print("  examples  - Show usage examples")
    print("  all       - Show everything")
    print("  help      - Show this help message")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "help":
        show_help()
    else:
        main() 