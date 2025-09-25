"""
Simple Robot Movement Demo

A minimal example demonstrating the core features of the MoveManager.
This focuses on the essential functionality without complex error handling.

Run with: uv run python examples/move/simple_movement_demo.py
"""

import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion.robot import Robot, Link, Joint, JointType
from linkmotion.move.manager import MoveManager
from linkmotion.transform import Transform

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_simple_robot() -> Robot:
    """Create a simple 2-joint robot arm."""
    robot = Robot()

    # Base link
    base = Link.from_cylinder(name="base", radius=0.1, height=0.05)
    robot.add_link(base)

    # Upper arm
    upper_arm = Link.from_box(name="upper_arm", extents=np.array([0.05, 0.05, 0.2]))
    robot.add_link(upper_arm)

    # Lower arm
    lower_arm = Link.from_cylinder(name="lower_arm", radius=0.03, height=0.15)
    robot.add_link(lower_arm)

    # Shoulder joint (revolute)
    shoulder = Joint(
        name="shoulder",
        type=JointType.REVOLUTE,
        parent_link_name="base",
        child_link_name="upper_arm",
        center=np.array([0.0, 0.0, 0.025]),  # Top of base
        direction=np.array([0.0, 0.0, 1.0]),  # Z-axis rotation
    )
    robot.add_joint(shoulder)

    # Elbow joint (prismatic)
    elbow = Joint(
        name="elbow",
        type=JointType.PRISMATIC,
        parent_link_name="upper_arm",
        child_link_name="lower_arm",
        center=np.array([0.0, 0.0, 0.1]),  # End of upper arm
        direction=np.array([0.0, 0.0, 1.0]),  # Z-axis extension
    )
    robot.add_joint(elbow)

    print(
        f"Created robot with {len(robot.links())} links and {len(robot._joint_dict)} joints"
    )
    return robot


def demonstrate_basic_movements():
    """Demonstrate basic joint movements."""
    print("\n=== Basic Movement Demo ===")

    # Create robot and manager
    robot = create_simple_robot()
    manager = MoveManager(robot)

    print(f"Link name mapping: {manager.link_name_to_id}")

    # Show initial poses
    print("\n--- Initial Poses ---")
    for link_name in manager.link_name_to_id.keys():
        transform = manager.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Move shoulder joint
    print("\n--- Moving shoulder 45 degrees ---")
    manager.move("shoulder", np.pi / 4)

    for link_name in manager.link_name_to_id.keys():
        transform = manager.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Extend elbow joint
    print("\n--- Extending elbow 5cm ---")
    manager.move("elbow", 0.05)

    for link_name in manager.link_name_to_id.keys():
        transform = manager.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Query joint information
    print("\n--- Joint Information ---")
    for joint_name in ["shoulder", "elbow"]:
        center = manager.get_center(joint_name)
        direction = manager.get_direction(joint_name)

        print(f"Joint '{joint_name}':")
        if center is not None:
            print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        else:
            print("  Center: None")
        print(
            f"  Direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]"
        )

    # Reset
    print("\n--- After Reset ---")
    manager.reset_move()

    for link_name in manager.link_name_to_id.keys():
        transform = manager.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")


def demonstrate_transform_joints():
    """Demonstrate planar and floating joints with Transform objects."""
    print("\n=== Transform Joint Demo ===")

    robot = Robot()

    # Base
    base = Link.from_box("base", np.array([0.2, 0.2, 0.05]))
    robot.add_link(base)

    # Mobile platform (planar movement)
    platform = Link.from_cylinder("platform", radius=0.08, height=0.03)
    robot.add_link(platform)

    # Sensor (6DOF movement)
    sensor = Link.from_sphere("sensor", radius=0.04)
    robot.add_link(sensor)

    # Planar joint
    platform_joint = Joint(
        name="platform_motion",
        type=JointType.PLANAR,
        parent_link_name="base",
        child_link_name="platform",
    )
    robot.add_joint(platform_joint)

    # Floating joint
    sensor_joint = Joint(
        name="sensor_motion",
        type=JointType.FLOATING,
        parent_link_name="platform",
        child_link_name="sensor",
    )
    robot.add_joint(sensor_joint)

    manager = MoveManager(robot)

    print("--- Initial poses ---")
    for link_name in manager.link_name_to_id.keys():
        transform = manager.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Move platform with 2D transform
    print("\n--- Moving platform (planar) ---")
    platform_transform = Transform(
        translate=np.array([0.1, 0.05, 0.0]), rotate=R.from_euler("z", 30, degrees=True)
    )
    manager.move("platform_motion", platform_transform)

    for link_name in manager.link_name_to_id.keys():
        transform = manager.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Move sensor with 6DOF transform
    print("\n--- Moving sensor (floating) ---")
    sensor_transform = Transform(
        translate=np.array([0.0, 0.0, 0.1]), rotate=R.from_euler("x", 45, degrees=True)
    )
    manager.move("sensor_motion", sensor_transform)

    for link_name in manager.link_name_to_id.keys():
        transform = manager.get_transform(link_name)
        pos = transform.position
        print(f"{link_name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")


def main():
    """Main demonstration function."""
    print("LinkMotion MoveManager Simple Demo")
    print("==================================")

    # Basic movements with revolute and prismatic joints
    demonstrate_basic_movements()

    # Transform movements with planar and floating joints
    demonstrate_transform_joints()

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
