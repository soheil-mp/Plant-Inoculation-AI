# OT-2 Robot Simulation Environment

This project contains a simulation environment for the OT-2 liquid handling robot, focusing on exploring and visualizing the robot's working envelope. The simulation allows control of the robot's movements and provides detailed state observations.

## Environment Setup

1. Clone this repository
2. Create a Python virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- Python 3.9+
- NumPy (1.26.3): For numerical computations and array operations
- Matplotlib (3.8.2): For visualization and plotting
- OpenCV (4.9.0.80): For image processing and visualization
- PyBullet (3.2.6): For robot simulation physics
- Jupyter (1.0.0): For running the simulation notebook

## Working Envelope

The working envelope represents the reachable space of the pipette tip. Based on our simulation results, the boundaries of the working envelope are:

- X-axis: -0.1871 to 0.2530 units
- Y-axis: -0.1705 to 0.2195 units
- Z-axis: 0.1195 to 0.2895 units

These measurements define the cubic volume within which the pipette tip can safely and reliably operate.

## Usage

The simulation can be run using the provided Jupyter notebook (`task9.ipynb`). The notebook contains:

1. Simulation Initialization
   - Creates simulation environment with GUI rendering
   - Sets up robot instance with unique ID

2. Robot Control Functions
   - `get_states()`: Retrieves current robot state including:
     - Robot base position
     - Pipette tip position
     - Joint positions and velocities
   - `move_to_target()`: Moves robot to specified coordinates
   - `find_working_envelope()`: Determines reachable workspace boundaries

3. Working Envelope Analysis
   - Systematic testing of robot movement limits
   - Recording and visualization of workspace boundaries

## Project Structure

- `task9.ipynb`: Main simulation and control notebook
- `requirements.txt`: List of Python package dependencies
- `Y2B-2023-OT2_Twin/`: Directory containing simulation assets and classes
  - `sim_class.py`: Core simulation class implementation
  - Robot model and environment assets

## Key Features

- Real-time state monitoring
- Precise movement control with velocity scaling
- Automatic workspace boundary detection
- Timeout handling for movement commands
- Position tolerance checking 