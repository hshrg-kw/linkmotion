# Robot System Examples

This directory contains comprehensive examples demonstrating how to use the LinkMotion robot system for robotics applications. Each example is self-contained and can be run independently.

## Examples Overview

### 1. Basic Robot Construction (`basic_robot_construction.py`)
**Purpose**: Learn how to build robot models from scratch
**Topics Covered**:
- Creating links with different shapes (box, sphere, cylinder, etc.)
- Defining joints with various types (revolute, prismatic, fixed)
- Building robot structures with multiple links and joints
- Robot traversal and querying methods
- Error handling and validation

**Run with**: `uv run python examples/robot/basic_robot_construction.py`

### 2. Robot Kinematics (`robot_kinematics.py`)
**Purpose**: Explore robot kinematics and structure analysis
**Topics Covered**:
- Root and leaf link identification
- Static vs dynamic link classification
- Parent-child relationship traversal
- Kinematic chain analysis
- Robot copying and transformation

**Run with**: `uv run python examples/robot/robot_kinematics.py`

### 3. Advanced Robot Operations (`advanced_robot_operations.py`)
**Purpose**: Master advanced robot manipulation techniques
**Topics Covered**:
- Link renaming and restructuring
- Link division for detailed modeling
- Robot concatenation for complex systems
- Collision object generation
- Visual mesh creation

**Run with**: `uv run python examples/robot/advanced_robot_operations.py`

## Example Structure

Each example follows a consistent structure:
- **Imports and setup**: Required libraries and logging configuration
- **Main functions**: Focused demonstrations of specific concepts
- **Practical scenarios**: Real-world robotics applications
- **Comprehensive logging**: Detailed output explaining operations
- **Error handling**: Robust examples that handle edge cases

## Running Examples

### Individual Examples
```bash
# Run a specific example
uv run python examples/robot/basic_robot_construction.py

# Run with verbose output
uv run python examples/robot/robot_kinematics.py
```

### All Examples
```bash
# Run all robot examples in sequence
for example in examples/robot/*.py; do
    echo "Running $example..."
    uv run python "$example"
    echo "Completed $example"
    echo "---"
done
```

## Key Concepts Demonstrated

### Robot Structure
- **Links**: Physical components with geometric shapes
- **Joints**: Connections between links with motion constraints
- **Hierarchy**: Parent-child relationships forming kinematic trees
- **Validation**: Ensuring structural integrity and consistency

### Robot Analysis
- **Traversal**: Systematic exploration of robot structure
- **Classification**: Static vs dynamic component identification
- **Queries**: Efficient lookup of components and relationships
- **Transformation**: Spatial calculations and coordinate frames

### Advanced Operations
- **Modification**: Runtime changes to robot structure
- **Composition**: Combining multiple robots into complex systems
- **Visualization**: Creating meshes for display and analysis
- **Collision Detection**: Generating objects for safety checking