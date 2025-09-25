# Robot Movement Examples

This directory contains examples demonstrating the LinkMotion movement system using the `MoveManager` class. These examples show how to control robot joint movements, query robot poses, and work with spatial transformations.

## Overview

The movement system provides:
- **Joint Control**: Move different types of joints (revolute, prismatic, continuous, planar, floating)
- **Pose Queries**: Get world-space transforms of robot links
- **Kinematic Chains**: Handle complex multi-joint robot structures
- **Spatial Transforms**: Work with 6DOF transformations for advanced joints

## Examples

### `basic_robot_movement.py`

A comprehensive example demonstrating:

- **Robot Construction**: Creating a multi-joint robot arm with various joint types
- **Basic Movements**: Moving individual joints and observing results
- **Advanced Movements**: Coordinated multi-joint motions
- **Transform Operations**: Working with planar and floating joints using Transform objects
- **Pose Queries**: Getting link positions, orientations, joint centers, and directions
- **Error Handling**: Proper handling of common movement errors

**Key Features Demonstrated:**
- Revolute joints (rotation around an axis)
- Prismatic joints (linear translation)
- Continuous joints (unlimited rotation)
- Planar joints (2D movement)
- Floating joints (6DOF movement)
- Fixed joints (no movement)

## Running the Examples

```bash
# Run the basic movement example
uv run python examples/move/basic_robot_movement.py
```

## Core Concepts

### Joint Types

1. **REVOLUTE**: Rotation around a specific axis with limits
   ```python
   manager.move("shoulder", np.pi/4)  # Rotate 45 degrees
   ```

2. **PRISMATIC**: Linear translation along a specific axis
   ```python
   manager.move("elbow", 0.1)  # Extend 10cm
   ```

3. **CONTINUOUS**: Unlimited rotation around an axis
   ```python
   manager.move("wrist", 2*np.pi)  # Full rotation
   ```

4. **PLANAR**: 2D movement (translation + rotation in a plane)
   ```python
   transform = Transform(translate=[0.1, 0.1, 0], rotate=R.from_euler('z', 30, degrees=True))
   manager.move("platform", transform)
   ```

5. **FLOATING**: 6DOF movement (full 3D translation + rotation)
   ```python
   transform = Transform(translate=[0.1, 0.2, 0.3], rotate=R.from_euler('xyz', [10, 20, 30], degrees=True))
   manager.move("sensor", transform)
   ```

6. **FIXED**: No movement (rigid connection)
   ```python
   # Fixed joints cannot be moved
   # manager.move("fixed_joint", value)  # Will raise ValueError
   ```

### MoveManager Usage

```python
from linkmotion.move.manager import MoveManager

# Create robot (see robot examples)
robot = create_robot()

# Initialize movement manager
manager = MoveManager(robot)

# Move joints
manager.move("joint_name", value)

# Query poses
transform = manager.get_transform("link_name")
position = transform.translation
rotation = transform.rotation

# Get joint information
center = manager.get_center("joint_name")      # Joint center in world coordinates
direction = manager.get_direction("joint_name") # Joint axis in world coordinates

# Reset all movements
manager.reset_move()
```

### Error Handling

The movement system provides comprehensive error handling:

- **Invalid Joint Names**: Attempting to move non-existent joints
- **Fixed Joint Movement**: Trying to move joints that don't allow movement
- **Type Mismatches**: Providing wrong value types for joint types
- **Missing Attributes**: Joints lacking required parameters (e.g., center for revolute joints)

## Integration with Other Systems

The movement system integrates with:

- **Transform System**: Uses `Transform` objects for spatial calculations
- **Robot System**: Works with `Robot`, `Link`, and `Joint` objects
- **Collision System**: Provides transformed meshes for collision detection
- **Visualization**: Enables real-time robot visualization

## Advanced Usage

### Custom Movement Sequences

```python
def move_to_target_pose(manager, target_pos):
    """Move robot to reach a target position."""
    # Calculate inverse kinematics (simplified example)
    shoulder_angle = np.arctan2(target_pos[1], target_pos[0])
    elbow_extension = np.linalg.norm(target_pos[:2]) - base_reach

    manager.move("shoulder", shoulder_angle)
    manager.move("elbow", elbow_extension)
```

### Trajectory Execution

```python
def execute_trajectory(manager, waypoints, joint_name):
    """Execute a smooth trajectory through waypoints."""
    for waypoint in waypoints:
        manager.move(joint_name, waypoint)
        # Add timing/interpolation as needed
```

## See Also

- `examples/robot/` - Robot construction examples
- `examples/transform/` - Transform system examples
- `src/linkmotion/move/` - Movement system source code
- `tests/test_move/` - Movement system tests