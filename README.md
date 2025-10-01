# LinkMotion

A comprehensive Python library for robotics applications, providing tools for robot modeling, joint control, collision detection, visualization, and URDF import/export functionality.

## Features

### 🤖 Robot Modeling
- **Flexible Robot Construction**: Build complex robot models from primitive shapes (box, sphere, cylinder, cone, capsule, mesh)
- **Joint System**: Support for revolute, prismatic, and fixed joints with configurable constraints
- **Hierarchical Structure**: Tree-based robot structure with parent-child relationships
- **Custom Robots**: Advanced robot construction utilities and custom modeling tools

### 🔄 Transform System
- **Spatial Transformations**: Comprehensive 3D transformation calculations
- **Coordinate Frames**: Hierarchical coordinate frame management
- **Transform Manager**: Advanced transform hierarchy handling and operations

### 🎯 Robot Movement
- **Joint Control**: Precise control of robot joint positions and states
- **State Management**: Real-time management of robot configuration
- **Transform Updates**: Automatic update of link transformations based on joint movements

### 💥 Collision Detection
- **Real-time Collision Checking**: Fast collision detection using FCL (Flexible Collision Library)
- **Safety Verification**: Comprehensive collision checking for robot configurations
- **Collision Manager**: High-level collision detection interface

### 📐 Workspace Analysis
- **Reachability Analysis**: Calculate robot workspace and reachable areas, which is called `range` in this project
- **Range Calculations**: Joint and workspace limit analysis
- **Multi-axis Analysis**: 2D and 3D workspace visualization

### 🎨 3D Visualization
- **Interactive Visualization**: 3D robot visualization using K3D
- **Motion Animation**: Real-time motion visualization and animation
- **Collision Visualization**: Visual collision detection and safety checking

### 📄 URDF Support
- **URDF Import**: Parse and import URDF (Unified Robot Description Format) files
- **URDF Export**: Generate URDF files from robot models
- **Mesh Support**: Handle complex mesh geometries in URDF files

### ⚙️ Advanced Modeling
- **Sweep Operations**: Complex geometry generation through sweep operations
- **Geometry Modification**: Advanced geometry removal and modification tools

## Installation

### Prerequisites

- Python 3.12 or higher

### Basic Installation

```bash
pip install linkmotion
```

### With Jupyter Support

For interactive notebooks and visualization:

```bash
pip install linkmotion[jup]
```

### Dependencies

**Core Dependencies:**
- `joblib` - Parallel computing utilities
- `manifold3d` - 3D geometry processing
- `python-fcl` - Collision detection library
- `scipy` - Scientific computing
- `shapely` - Geometric operations
- `trimesh` - 3D mesh processing

**Optional Dependencies (with `--extra jup`):**
- `jupyter` - Interactive notebooks
- `k3d` - 3D visualization in Jupyter
- `plotly` - Interactive plotting

## Quick Start

### Basic Robot Construction

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion import Robot, Link, Joint, JointType, Transform

humanoid = Robot()

body = Link.from_box("body", extents=np.array((3, 2, 10)))

t = Transform(rotate=R.from_rotvec(90.0 * np.array((0, 1, 0)), degrees=True), translate=np.array((4, 0, 3)))
right_arm = Link.from_cylinder("right_arm", radius=0.5, height=5, default_transform=t)

t = Transform(rotate=R.from_rotvec(90.0 * np.array((0, 1, 0)), degrees=True), translate=np.array((-4, 0, 3)))
left_arm = Link.from_cylinder("left_arm", radius=0.5, height=5, default_transform=t)

t = Transform(rotate=R.from_rotvec(180.0 * np.array((0, 1, 0)), degrees=True), translate=np.array((1, 0, -8)))
right_leg = Link.from_cone("right_leg", radius=0.5, height=6, default_transform=t)

t = Transform(rotate=R.from_rotvec(180.0 * np.array((0, 1, 0)), degrees=True), translate=np.array((-1, 0, -8)))
left_leg = Link.from_cone("left_leg", radius=0.5, height=6, default_transform=t)

t = Transform(translate=np.array((0, 0, -5)))
head = Link.from_sphere("head", 3, center=np.array((0, 0, 8)))

left_leg_joint = Joint(
    "left_leg_joint",
    JointType.REVOLUTE,
    child_link_name="left_leg",
    parent_link_name="body",
    center=np.array((-1, 0, -5)),
    direction=np.array((1, 0, 0)),
    min_=-np.pi / 2.0,
    max_=np.pi / 4.0,
)

t = Transform(translate=np.array((0, -5, 0)))
wall = Link.from_box("wall", extents=np.array((20, 1, 20)), default_transform=t, color=np.array((0.0, 0.8, 0.8, 0.2)))

humanoid.add_link(body)
humanoid.add_link(right_arm)
humanoid.add_link(left_arm)
humanoid.add_link(right_leg)
humanoid.add_link(left_leg)
humanoid.add_link(head)
humanoid.add_joint(left_leg_joint)
humanoid.add_link(wall)
```

#### Robot Appearance
![OutputExample](imgs/readme1.png)

### Transform Operations
```python
from linkmotion import MoveManager

mm = MoveManager(humanoid)
mm.move(joint_name="left_leg_joint", value=-np.pi / 6.0)
```
#### Robot Appearance
![OutputExample](imgs/readme2.png)

### Collision Detection

```python
from linkmotion import CollisionManager

cm = CollisionManager(mm)
cm.distance({"wall"}, {"body", "left_leg", "right_leg"})
```
#### Robot Appearance
![OutputExample](imgs/readme3.png)

### URDF Import/Export

```python
robot = Robot.from_urdf_file("../../models/toy_model/toy_model.urdf")
```
#### Robot Appearance
![OutputExample](notebooks/urdf/img/import_export/import_export1.png)

### Range Calculations
```python
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion import Robot, Link, Joint, JointType, Transform

robot = Robot()

base_link = Link.from_sphere(
    name="base_link", radius=0.1, center=np.array([0, 0, 0])
)
arm_link = Link.from_cylinder(
    name="arm_link",
    radius=0.1,
    height=1.0,
    default_transform=Transform(translate=np.array([0, 0, 0.5])),
)
hand_link = Link.from_box(
    name="hand_link",
    extents=np.array([0.1, 0.1, 0.1]),
    default_transform=Transform(translate=np.array([0, 0, 1.0])),
    color=np.array([0, 1, 0, 1]),
)
finger_link = Link.from_box(
    name="finger_link",
    extents=np.array([0.05, 0.05, 0.1]),
    default_transform=Transform(translate=np.array([0, 0, 1.1])),
    color=np.array([1, 0, 0, 1]),
)
obstacle_link = Link.from_box(
    name="obstacle_link",
    extents=np.array([10, 10, 0.1]),
    default_transform=Transform(translate=np.array([0, 0, 1.5])),
)

revolute_joint = Joint(
    name="revolute_joint",
    type=JointType.REVOLUTE,
    parent_link_name="base_link",
    child_link_name="arm_link",
    direction=np.array([1, 0, 0]),
    center=np.array([0, 0, 0.0]),
    min_=-np.pi / 2,
    max_=np.pi / 2,
)

prismatic_joint = Joint(
    name="prismatic_joint",
    type=JointType.PRISMATIC,
    parent_link_name="arm_link",
    child_link_name="hand_link",
    direction=np.array([0, 1, 0]),
    center=np.array([0, 0, 1.0]),
    min_=-10,
    max_=10,
)

prismatic_joint2 = Joint(
    name="prismatic_joint2",
    type=JointType.PRISMATIC,
    parent_link_name="hand_link",
    child_link_name="finger_link",
    direction=np.array([0, 0, 1]),
    center=np.array([0, 0, 1.1]),
    min_=-10,
    max_=10,
)

robot.add_link(base_link)
robot.add_link(arm_link)
robot.add_link(hand_link)
robot.add_link(finger_link)
robot.add_link(obstacle_link)
robot.add_joint(revolute_joint)
robot.add_joint(prismatic_joint)
robot.add_joint(prismatic_joint2)
```

#### Robot Appearance
![OutputExample](notebooks/visual/img/range_2axes/range_2axes2.png)

```python
from linkmotion.range.range import RangeCalculator

calculator = RangeCalculator(
    robot, {"hand_link", "finger_link"}, {"obstacle_link"}
)
calculator.add_axis("revolute_joint", np.linspace(-np.pi / 2, np.pi / 2, 100))
calculator.add_axis("prismatic_joint", np.linspace(-3, 3, 200))
calculator.execute()
```

#### Range Appearance
- 1 means collided
- 0 means collision-free

![OutputExample](notebooks/visual/img/range_2axes/range_2axes3.png)

## Examples

The `examples/` directory contains comprehensive examples organized by functionality:

## Interactive Notebooks

Explore the library through interactive Jupyter notebooks:

### Visualization Notebooks
- `notebooks/visual/01.base.ipynb` - Basic visualization concepts
- `notebooks/visual/02.mesh.ipynb` - Mesh visualization
- `notebooks/visual/03.robot.ipynb` - Robot visualization
- `notebooks/visual/04.move.ipynb` - Motion visualization
- `notebooks/visual/05.collision.ipynb` - Collision visualization
- `notebooks/visual/06.range_2axes.ipynb` - 2D workspace analysis
- `notebooks/visual/07.range_3axes.ipynb` - 3D workspace analysis

### URDF Notebooks
- `notebooks/urdf/01.import_export.ipynb` - URDF import/export demonstration

## Future Works

- More URDF compatibility such as dae
- Inverse Kinematics
- Path planning using OMPL

## Development

### Code Quality

```bash
# Format code
uv run ruff format

# Check code quality
uv run ruff check

# Auto-fix issues
uv run ruff check --fix
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_robot/test_robot.py

# Run with coverage
uv run pytest --cov

# Verbose output
uv run pytest -v
```

### Development Scripts

```bash
# Format, lint, and test in sequence
./scripts/format_lint_test.sh

# Run all examples
./scripts/run_all_examples.sh
```

## Project Structure

```
linkmotion/
├── src/linkmotion/           # Main source code
│   ├── robot/               # Robot modeling and manipulation
│   ├── transform/           # Spatial transformations
│   ├── move/                # Robot joint control and movement
│   ├── collision/           # Collision detection
│   ├── range/               # Workspace analysis
│   ├── modeling/            # Advanced geometric modeling
│   ├── urdf/                # URDF import/export
│   ├── visual/              # 3D visualization
│   ├── typing/              # Type definitions
│   └── const/               # Constants and configuration
├── tests/                   # Comprehensive test suite
├── examples/                # Usage examples by functionality
├── notebooks/               # Interactive Jupyter notebooks
├── models/                  # Sample robot models and meshes
├── docs/                    # Auto-generated documentation
└── scripts/                 # Development automation scripts
```

## API Reference

Here is [API Reference](https://hshrg-kw.github.io/linkmotion/).

### Core Classes

- **`Robot`** - Main robot class for building and manipulating robot models
- **`Link`** - Robot link with geometric shapes and properties
- **`Joint`** - Robot joint with motion constraints and types
- **`Transform`** - 3D transformation operations and calculations
- **`MoveManager`** - Robot joint control and state management interface
- **`CollisionManager`** - Collision detection and safety checking

### Shape Classes

- **`Box`**, **`Sphere`**, **`Cylinder`**, **`Cone`**, **`Capsule`** - Primitive shapes
- **`Mesh`** - Complex mesh geometry
- **`Convex`** - Convex hull shapes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding guidelines in `CLAUDE.md`
4. Run tests and quality checks
5. Submit a pull request

### Coding Standards

- Follow Google-style docstrings
- Use type hints for all functions
- Implement `__repr__` for custom classes
- Use logging instead of print statements
- Follow PEP 8 style guide (enforced by Ruff)

## License

MIT License
