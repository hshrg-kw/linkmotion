"""
Basic Robot Movement Example

This example demonstrates how to use the MoveManager to control robot motion.
You'll learn how to move different types of joints, query robot poses, and
visualize robot transformations in real-time.

Topics covered:
- Creating a robot with various joint types
- Initializing the MoveManager
- Moving revolute, prismatic, and continuous joints
- Querying world-space transforms
- Getting joint centers and directions
- Resetting robot poses

Run with: uv run python examples/move/basic_robot_movement.py
"""

import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion.robot import Robot, Link, Joint, JointType
from linkmotion.move.manager import MoveManager
from linkmotion.transform import Transform

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_articulated_robot() -> Robot:
    """Create a multi-joint robot arm for movement demonstration.

    This robot has:
    - A fixed base
    - A revolute shoulder joint
    - A prismatic elbow joint
    - A continuous wrist joint
    - An end effector

    Returns:
        A robot suitable for demonstrating various joint movements.
    """
    logger.info("=== Creating Articulated Robot ===")

    robot = Robot()

    # Base platform (fixed)
    base = Link.from_cylinder(
        name="base",
        radius=0.15,
        height=0.1,
        color=np.array([0.3, 0.3, 0.3, 1.0]),  # Gray
    )
    robot.add_link(base)
    logger.info(f"Added base: {base}")

    # Upper arm (shoulder to elbow)
    upper_arm = Link.from_box(
        name="upper_arm",
        extents=np.array([0.08, 0.08, 0.3]),
        color=np.array([0.8, 0.2, 0.2, 1.0]),  # Red
    )
    robot.add_link(upper_arm)
    logger.info(f"Added upper arm: {upper_arm}")

    # Lower arm (elbow to wrist)
    lower_arm = Link.from_cylinder(
        name="lower_arm",
        radius=0.04,
        height=0.25,
        color=np.array([0.2, 0.8, 0.2, 1.0]),  # Green
    )
    robot.add_link(lower_arm)
    logger.info(f"Added lower arm: {lower_arm}")

    # Wrist connector
    wrist = Link.from_sphere(
        name="wrist",
        radius=0.06,
        color=np.array([0.2, 0.2, 0.8, 1.0]),  # Blue
    )
    robot.add_link(wrist)
    logger.info(f"Added wrist: {wrist}")

    # End effector (gripper)
    end_effector = Link.from_box(
        name="end_effector",
        extents=np.array([0.12, 0.03, 0.03]),
        color=np.array([0.8, 0.8, 0.2, 1.0]),  # Yellow
    )
    robot.add_link(end_effector)
    logger.info(f"Added end effector: {end_effector}")

    # Shoulder joint (revolute) - base to upper arm
    shoulder_joint = Joint(
        name="shoulder",
        type=JointType.REVOLUTE,
        parent_link_name="base",
        child_link_name="upper_arm",
        center=np.array([0.0, 0.0, 0.05]),  # Top of base
        direction=np.array([0.0, 0.0, 1.0]),  # Z-axis rotation
        min_=-np.pi / 2,  # -90 degrees
        max_=np.pi / 2,  # +90 degrees
    )
    robot.add_joint(shoulder_joint)
    logger.info(f"Added shoulder joint: {shoulder_joint}")

    # Elbow joint (prismatic) - upper arm to lower arm
    elbow_joint = Joint(
        name="elbow",
        type=JointType.PRISMATIC,
        parent_link_name="upper_arm",
        child_link_name="lower_arm",
        center=np.array([0.0, 0.0, 0.15]),  # End of upper arm
        direction=np.array([0.0, 0.0, 1.0]),  # Z-axis extension
        min_=0.0,  # No retraction
        max_=0.15,  # 15cm extension
    )
    robot.add_joint(elbow_joint)
    logger.info(f"Added elbow joint: {elbow_joint}")

    # Wrist joint (continuous) - lower arm to wrist
    wrist_joint = Joint(
        name="wrist_rotation",
        type=JointType.CONTINUOUS,
        parent_link_name="lower_arm",
        child_link_name="wrist",
        center=np.array([0.0, 0.0, 0.125]),  # End of lower arm
        direction=np.array([0.0, 0.0, 1.0]),  # Z-axis continuous rotation
    )
    robot.add_joint(wrist_joint)
    logger.info(f"Added wrist joint: {wrist_joint}")

    # End effector joint (fixed) - wrist to end effector
    effector_joint = Joint(
        name="effector_mount",
        type=JointType.FIXED,
        parent_link_name="wrist",
        child_link_name="end_effector",
        center=np.array([0.0, 0.0, 0.03]),  # Offset from wrist center
    )
    robot.add_joint(effector_joint)
    logger.info(f"Added effector joint: {effector_joint}")

    logger.info(f"Created articulated robot: {robot}")
    return robot


def demonstrate_basic_movements(manager: MoveManager):
    """Demonstrate basic joint movements and pose queries.

    Args:
        manager: The MoveManager controlling the robot.
    """
    logger.info("=== Basic Movement Demonstrations ===")

    # Initial pose
    logger.info("Robot in initial pose:")
    print_robot_pose(manager)

    print("\n" + "-" * 50 + "\n")

    # Move shoulder joint (revolute)
    logger.info("Moving shoulder joint 45 degrees...")
    manager.move("shoulder", np.pi / 4)  # 45 degrees
    print_robot_pose(manager)

    print("\n" + "-" * 50 + "\n")

    # Extend elbow joint (prismatic)
    logger.info("Extending elbow joint 10cm...")
    manager.move("elbow", 0.1)  # 10cm extension
    print_robot_pose(manager)

    print("\n" + "-" * 50 + "\n")

    # Rotate wrist (continuous)
    logger.info("Rotating wrist 180 degrees...")
    manager.move("wrist_rotation", np.pi)  # 180 degrees
    print_robot_pose(manager)

    print("\n" + "-" * 50 + "\n")


def demonstrate_advanced_movements(manager: MoveManager):
    """Demonstrate advanced movement patterns and transformations.

    Args:
        manager: The MoveManager controlling the robot.
    """
    logger.info("=== Advanced Movement Demonstrations ===")

    # Reset to initial pose
    logger.info("Resetting robot to initial pose...")
    manager.reset_move()
    print_robot_pose(manager)

    print("\n" + "-" * 30 + "\n")

    # Complex coordinated movement
    logger.info("Performing coordinated multi-joint movement...")
    manager.move("shoulder", -np.pi / 6)  # -30 degrees
    manager.move("elbow", 0.08)  # 8cm extension
    manager.move("wrist_rotation", np.pi / 2)  # 90 degrees

    print_robot_pose(manager)

    # Demonstrate joint center and direction queries
    logger.info("\nJoint center and direction analysis:")
    for joint_name in ["shoulder", "elbow", "wrist_rotation"]:
        center = manager.get_center(joint_name)
        direction = manager.get_direction(joint_name)

        logger.info(f"Joint '{joint_name}':")
        if center is not None:
            logger.info(
                f"  World center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
            )
        else:
            logger.info("  No defined center")
        logger.info(
            f"  World direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]"
        )

    print("\n" + "-" * 30 + "\n")


def demonstrate_transform_operations(manager: MoveManager):
    """Demonstrate working with Transform objects for complex joints.

    Args:
        manager: The MoveManager controlling the robot.
    """
    logger.info("=== Transform Operation Demonstrations ===")

    # For this demo, let's create a robot with planar and floating joints
    robot = Robot()

    # Base link
    base = Link.from_box("base", np.array([0.2, 0.2, 0.1]))
    robot.add_link(base)

    # Mobile platform (planar joint - 2D movement)
    platform = Link.from_cylinder("platform", radius=0.08, height=0.05)
    robot.add_link(platform)

    # Floating sensor (6DOF movement)
    sensor = Link.from_sphere("sensor", radius=0.04)
    robot.add_link(sensor)

    # Planar joint for 2D platform movement
    platform_joint = Joint(
        name="platform_motion",
        type=JointType.PLANAR,
        parent_link_name="base",
        child_link_name="platform",
    )
    robot.add_joint(platform_joint)

    # Floating joint for 6DOF sensor movement
    sensor_joint = Joint(
        name="sensor_motion",
        type=JointType.FLOATING,
        parent_link_name="platform",
        child_link_name="sensor",
    )
    robot.add_joint(sensor_joint)

    # Create new manager for this robot
    transform_manager = MoveManager(robot)

    logger.info("Created robot with planar and floating joints")

    # Move platform with 2D transform
    platform_transform = Transform(
        translate=np.array([0.1, 0.15, 0.0]),  # Move in X-Y plane
        rotate=R.from_euler("z", 30, degrees=True),  # Rotate around Z
    )
    transform_manager.move("platform_motion", platform_transform)
    logger.info("Moved platform with planar transform")

    # Move sensor with 6DOF transform
    sensor_transform = Transform(
        translate=np.array([0.05, 0.05, 0.2]),  # Full 3D translation
        rotate=R.from_euler("xyz", [15, 30, 45], degrees=True),  # Full 3D rotation
    )
    transform_manager.move("sensor_motion", sensor_transform)
    logger.info("Moved sensor with 6DOF transform")

    # Query final poses
    platform_pose = transform_manager.get_transform("platform")
    sensor_pose = transform_manager.get_transform("sensor")

    logger.info(f"Platform final pose: {platform_pose}")
    logger.info(f"Sensor final pose: {sensor_pose}")


def print_robot_pose(manager: MoveManager):
    """Print the current pose of all robot links.

    Args:
        manager: The MoveManager to query.
    """
    link_names = list(manager.link_name_to_id.keys())

    for link_name in link_names:
        try:
            transform = manager.get_transform(link_name)
            pos = transform.position
            rot_euler = R.from_matrix(transform.rotation).as_euler("xyz", degrees=True)

            logger.info(f"Link '{link_name}':")
            logger.info(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            logger.info(
                f"  Rotation: [{rot_euler[0]:.1f}°, {rot_euler[1]:.1f}°, {rot_euler[2]:.1f}°]"
            )
        except ValueError:
            logger.warning(f"Could not get transform for link '{link_name}'")


def demonstrate_error_handling(manager: MoveManager):
    """Demonstrate error handling in movement operations.

    Args:
        manager: The MoveManager to test.
    """
    logger.info("=== Error Handling Demonstrations ===")

    # Try to move non-existent joint
    try:
        manager.move("non_existent_joint", 1.0)
    except ValueError as e:
        logger.warning(f"Expected error for non-existent joint: {e}")

    # Try to move fixed joint
    try:
        manager.move("effector_mount", 1.0)
    except ValueError as e:
        logger.warning(f"Expected error for fixed joint: {e}")

    # Try wrong value type for revolute joint
    try:
        manager.move("shoulder", Transform())  # Should be float
    except ValueError as e:
        logger.warning(f"Expected error for wrong value type: {e}")

    # Try to get transform for non-existent link
    try:
        manager.get_transform("non_existent_link")
    except ValueError as e:
        logger.warning(f"Expected error for non-existent link: {e}")

    logger.info("Error handling demonstrations completed")


def main():
    """Main function demonstrating robot movement concepts."""
    logger.info("Starting Robot Movement Examples")

    # Create robot and movement manager
    robot = create_articulated_robot()
    manager = MoveManager(robot)

    logger.info(f"Created MoveManager for robot with {len(robot.links())} links")

    print("\n" + "=" * 60 + "\n")

    # Basic movements
    demonstrate_basic_movements(manager)

    print("\n" + "=" * 60 + "\n")

    # Advanced movements
    demonstrate_advanced_movements(manager)

    print("\n" + "=" * 60 + "\n")

    # Transform operations
    demonstrate_transform_operations(manager)

    print("\n" + "=" * 60 + "\n")

    # Error handling
    demonstrate_error_handling(manager)

    logger.info("Robot Movement Examples completed successfully!")


if __name__ == "__main__":
    main()
