"""
Robot Kinematics and Structure Analysis Example

This example demonstrates advanced robot analysis techniques including
kinematic structure exploration, link classification, and hierarchical
traversal methods. You'll learn how to analyze robot topology and
understand the relationships between components.

Topics covered:
- Root and leaf link identification
- Static vs dynamic link classification
- Parent-child relationship traversal
- Kinematic chain analysis
- Robot structure copying and transformation
- Hierarchical tree traversal algorithms

Run with: uv run python examples/robot/robot_kinematics.py
"""

import logging
import numpy as np

from linkmotion.robot import Robot, Link, Joint, JointType
from linkmotion.transform import Transform

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_kinematic_chain() -> Robot:
    """Create a serial kinematic chain (robot arm).

    This creates a typical robot arm structure with multiple joints
    in series, demonstrating a kinematic chain.

    Returns:
        A robot representing a 4-DOF serial arm.
    """
    logger.info("=== Creating Serial Kinematic Chain ===")

    robot = Robot()

    # Base (fixed to world)
    base = Link.from_cylinder(
        name="base",
        radius=0.15,
        height=0.1,
        color=np.array([0.5, 0.5, 0.5, 1.0]),  # Gray
    )
    robot.add_link(base)

    # Link 1 (shoulder)
    link1 = Link.from_box(
        name="shoulder",
        extents=np.array([0.1, 0.1, 0.3]),
        color=np.array([0.8, 0.2, 0.2, 1.0]),  # Red
    )
    robot.add_link(link1)

    # Link 2 (upper arm)
    link2 = Link.from_cylinder(
        name="upper_arm",
        radius=0.06,
        height=0.4,
        color=np.array([0.2, 0.8, 0.2, 1.0]),  # Green
    )
    robot.add_link(link2)

    # Link 3 (forearm)
    link3 = Link.from_cylinder(
        name="forearm",
        radius=0.05,
        height=0.35,
        color=np.array([0.2, 0.2, 0.8, 1.0]),  # Blue
    )
    robot.add_link(link3)

    # Link 4 (wrist)
    link4 = Link.from_sphere(
        name="wrist",
        radius=0.08,
        color=np.array([0.8, 0.8, 0.2, 1.0]),  # Yellow
    )
    robot.add_link(link4)

    # Joint 1: Base to shoulder (revolute around Z)
    joint1 = Joint(
        name="base_to_shoulder",
        type=JointType.REVOLUTE,
        parent_link_name="base",
        child_link_name="shoulder",
        center=np.array([0.0, 0.0, 0.05]),
        direction=np.array([0.0, 0.0, 1.0]),
        min_=-np.pi,
        max_=np.pi,
    )
    robot.add_joint(joint1)

    # Joint 2: Shoulder to upper arm (revolute around Y)
    joint2 = Joint(
        name="shoulder_to_upper_arm",
        type=JointType.REVOLUTE,
        parent_link_name="shoulder",
        child_link_name="upper_arm",
        center=np.array([0.0, 0.0, 0.15]),
        direction=np.array([0.0, 1.0, 0.0]),
        min_=-np.pi / 2,
        max_=np.pi / 2,
    )
    robot.add_joint(joint2)

    # Joint 3: Upper arm to forearm (revolute around Y)
    joint3 = Joint(
        name="upper_arm_to_forearm",
        type=JointType.REVOLUTE,
        parent_link_name="upper_arm",
        child_link_name="forearm",
        center=np.array([0.0, 0.0, 0.2]),
        direction=np.array([0.0, 1.0, 0.0]),
        min_=-np.pi,
        max_=0.0,
    )
    robot.add_joint(joint3)

    # Joint 4: Forearm to wrist (revolute around Z)
    joint4 = Joint(
        name="forearm_to_wrist",
        type=JointType.REVOLUTE,
        parent_link_name="forearm",
        child_link_name="wrist",
        center=np.array([0.0, 0.0, 0.175]),
        direction=np.array([0.0, 0.0, 1.0]),
        min_=-np.pi,
        max_=np.pi,
    )
    robot.add_joint(joint4)

    logger.info(f"Created kinematic chain: {robot}")
    return robot


def create_branched_robot() -> Robot:
    """Create a robot with branching structure (tree topology).

    This demonstrates a more complex robot structure where one link
    has multiple children, creating a branched kinematic tree.

    Returns:
        A robot with branching kinematic structure.
    """
    logger.info("=== Creating Branched Robot Structure ===")

    robot = Robot()

    # Central hub
    hub = Link.from_sphere(
        name="central_hub",
        radius=0.12,
        color=np.array([0.6, 0.6, 0.6, 1.0]),  # Gray
    )
    robot.add_link(hub)

    # Branch 1: Left arm
    left_arm = Link.from_cylinder(
        name="left_arm",
        radius=0.04,
        height=0.25,
        color=np.array([0.8, 0.3, 0.3, 1.0]),  # Red
    )
    robot.add_link(left_arm)

    # Branch 2: Right arm
    right_arm = Link.from_cylinder(
        name="right_arm",
        radius=0.04,
        height=0.25,
        color=np.array([0.3, 0.8, 0.3, 1.0]),  # Green
    )
    robot.add_link(right_arm)

    # Branch 3: Front sensor
    front_sensor = Link.from_box(
        name="front_sensor",
        extents=np.array([0.08, 0.15, 0.05]),
        color=np.array([0.3, 0.3, 0.8, 1.0]),  # Blue
    )
    robot.add_link(front_sensor)

    # Left hand
    left_hand = Link.from_sphere(
        name="left_hand", radius=0.06, color=np.array([0.9, 0.5, 0.5, 1.0])
    )
    robot.add_link(left_hand)

    # Right hand
    right_hand = Link.from_sphere(
        name="right_hand", radius=0.06, color=np.array([0.5, 0.9, 0.5, 1.0])
    )
    robot.add_link(right_hand)

    # Joints creating the branched structure
    hub_to_left = Joint(
        name="hub_to_left_arm",
        type=JointType.REVOLUTE,
        parent_link_name="central_hub",
        child_link_name="left_arm",
        center=np.array([-0.12, 0.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
    )
    robot.add_joint(hub_to_left)

    hub_to_right = Joint(
        name="hub_to_right_arm",
        type=JointType.REVOLUTE,
        parent_link_name="central_hub",
        child_link_name="right_arm",
        center=np.array([0.12, 0.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
    )
    robot.add_joint(hub_to_right)

    hub_to_sensor = Joint(
        name="hub_to_front_sensor",
        type=JointType.FIXED,
        parent_link_name="central_hub",
        child_link_name="front_sensor",
        center=np.array([0.0, 0.12, 0.0]),
    )
    robot.add_joint(hub_to_sensor)

    left_to_hand = Joint(
        name="left_arm_to_hand",
        type=JointType.REVOLUTE,
        parent_link_name="left_arm",
        child_link_name="left_hand",
        center=np.array([0.0, 0.0, 0.125]),
        direction=np.array([1.0, 0.0, 0.0]),
    )
    robot.add_joint(left_to_hand)

    right_to_hand = Joint(
        name="right_arm_to_hand",
        type=JointType.REVOLUTE,
        parent_link_name="right_arm",
        child_link_name="right_hand",
        center=np.array([0.0, 0.0, 0.125]),
        direction=np.array([1.0, 0.0, 0.0]),
    )
    robot.add_joint(right_to_hand)

    logger.info(f"Created branched robot: {robot}")
    return robot


def analyze_robot_topology(robot: Robot):
    """Perform comprehensive topology analysis of a robot.

    Args:
        robot: The robot to analyze.
    """
    logger.info("=== Robot Topology Analysis ===")

    # Basic structure information
    all_links = robot.links()
    logger.info(f"Total links: {len(all_links)}")

    # Root analysis
    root_links = robot.root_links()
    logger.info(f"Root links ({len(root_links)}): {[link.name for link in root_links]}")

    # Leaf analysis
    leaf_links = robot.leaf_links()
    logger.info(f"Leaf links ({len(leaf_links)}): {[link.name for link in leaf_links]}")

    # Static vs dynamic classification
    static_links = robot.static_links()
    dynamic_links = robot.dynamic_links()
    logger.info(
        f"Static links ({len(static_links)}): {[link.name for link in static_links]}"
    )
    logger.info(
        f"Dynamic links ({len(dynamic_links)}): {[link.name for link in dynamic_links]}"
    )

    # Detailed link analysis
    logger.info("\n--- Detailed Link Analysis ---")
    for link in all_links:
        parent_joint = robot.parent_joint(link.name)
        child_joints = robot.child_joints(link.name)

        parent_info = (
            f"Parent: {parent_joint.name} ({parent_joint.type.name})"
            if parent_joint
            else "Parent: None (root)"
        )
        child_info = (
            f"Children: {[j.name for j in child_joints]}"
            if child_joints
            else "Children: None (leaf)"
        )

        logger.info(f"  {link.name}: {parent_info}, {child_info}")


def demonstrate_traversal_methods(robot: Robot):
    """Demonstrate different robot traversal algorithms.

    Args:
        robot: The robot to traverse.
    """
    logger.info("=== Robot Traversal Demonstrations ===")

    # Full robot traversal
    logger.info("Full robot traversal (breadth-first):")
    for i, link in enumerate(robot.traverse_links()):
        logger.info(f"  {i + 1}. {link.name}")

    # Choose a root link for demonstrations
    root_links = robot.root_links()
    if not root_links:
        logger.warning("No root links found for traversal demonstration")
        return

    start_link = root_links[0]
    logger.info(f"\nTraversal demonstrations starting from: {start_link.name}")

    # Child traversal (descendants)
    logger.info("Child links traversal (including self):")
    for i, link in enumerate(
        robot.traverse_child_links(start_link.name, include_self=True)
    ):
        logger.info(f"  {i + 1}. {link.name}")

    # Find a leaf link for parent traversal
    leaf_links = robot.leaf_links()
    if leaf_links:
        leaf_link = leaf_links[0]
        logger.info(f"\nParent traversal from leaf '{leaf_link.name}':")
        for i, link in enumerate(
            robot.traverse_parent_links(leaf_link.name, include_self=True)
        ):
            logger.info(f"  {i + 1}. {link.name}")


def demonstrate_robot_copying():
    """Demonstrate robot copying and transformation."""
    logger.info("=== Robot Copying Demonstration ===")

    # Create original robot
    original = Robot()
    base = Link.from_box("original_base", np.array([0.2, 0.2, 0.1]))
    arm = Link.from_cylinder("original_arm", radius=0.05, height=0.3)
    original.add_link(base)
    original.add_link(arm)

    joint = Joint(
        name="original_joint",
        type=JointType.REVOLUTE,
        parent_link_name="original_base",
        child_link_name="original_arm",
        center=np.array([0.0, 0.0, 0.05]),
        direction=np.array([0.0, 0.0, 1.0]),
    )
    original.add_joint(joint)

    logger.info(f"Original robot: {original}")

    # Create a copy with transformation
    transform = Transform(translate=np.array([1.0, 0.0, 0.0]))
    copied_robot = Robot.from_other(original, transform)

    logger.info(f"Copied robot: {copied_robot}")
    logger.info("Successfully demonstrated robot copying with transformation")


def demonstrate_kinematic_analysis():
    """Demonstrate kinematic analysis of different robot structures."""
    logger.info("=== Kinematic Analysis Demonstration ===")

    # Analyze serial chain
    logger.info("\n--- Serial Chain Analysis ---")
    serial_robot = create_kinematic_chain()
    analyze_robot_topology(serial_robot)
    demonstrate_traversal_methods(serial_robot)

    print("\n" + "=" * 50 + "\n")

    # Analyze branched structure
    logger.info("--- Branched Structure Analysis ---")
    branched_robot = create_branched_robot()
    analyze_robot_topology(branched_robot)
    demonstrate_traversal_methods(branched_robot)


def main():
    """Main function demonstrating robot kinematics and analysis."""
    logger.info("Starting Robot Kinematics Examples")

    # Demonstrate kinematic analysis
    demonstrate_kinematic_analysis()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate robot copying
    demonstrate_robot_copying()

    logger.info("Robot Kinematics Examples completed successfully!")


if __name__ == "__main__":
    main()
