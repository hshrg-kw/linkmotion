"""
Basic Robot Construction Example

This example demonstrates the fundamental concepts of building robot models
using the LinkMotion robot system. You'll learn how to create links with
different geometric shapes, define joints with various motion types, and
construct complete robot structures.

Topics covered:
- Creating links with primitive shapes (box, sphere, cylinder, etc.)
- Defining joints with different types and constraints
- Building simple and complex robot structures
- Basic robot queries and validation
- Error handling and debugging techniques

Run with: uv run python examples/robot/basic_robot_construction.py
"""

import logging
import numpy as np

from linkmotion.robot import Robot, Link, Joint, JointType

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_simple_robot() -> Robot:
    """Create a simple 2-link robot arm with basic shapes.

    This demonstrates the fundamental process of robot construction:
    1. Create links with geometric shapes
    2. Define joints with motion constraints
    3. Assemble the robot structure

    Returns:
        A simple robot with base and arm links connected by a revolute joint.
    """
    logger.info("=== Creating Simple Robot ===")

    # Create a new robot instance
    robot = Robot()
    logger.info(f"Created empty robot: {robot}")

    # Create base link (fixed platform)
    base_link = Link.from_box(
        name="base",
        extents=np.array([0.3, 0.3, 0.1]),  # 30cm x 30cm x 10cm
        color=np.array([0.5, 0.5, 0.5, 1.0]),  # Gray color
    )
    robot.add_link(base_link)
    logger.info(f"Added base link: {base_link}")

    # Create arm link (movable part)
    arm_link = Link.from_cylinder(
        name="arm",
        radius=0.05,  # 5cm radius
        height=0.4,  # 40cm height
        color=np.array([0.8, 0.2, 0.2, 1.0]),  # Red color
    )
    robot.add_link(arm_link)
    logger.info(f"Added arm link: {arm_link}")

    # Create revolute joint connecting base to arm
    base_to_arm_joint = Joint(
        name="base_to_arm",
        type=JointType.REVOLUTE,
        parent_link_name="base",
        child_link_name="arm",
        center=np.array([0.0, 0.0, 0.05]),  # Joint center at top of base
        direction=np.array([0.0, 0.0, 1.0]),  # Rotate around Z-axis
        min_=-np.pi,  # -180 degrees
        max_=np.pi,  # +180 degrees
    )
    robot.add_joint(base_to_arm_joint)
    logger.info(f"Added joint: {base_to_arm_joint}")

    logger.info(f"Completed simple robot: {robot}")
    return robot


def create_multi_link_robot() -> Robot:
    """Create a more complex robot with multiple links and joint types.

    This demonstrates:
    - Multiple joint types (fixed, revolute, prismatic)
    - Various link shapes
    - Complex kinematic chains
    - Joint constraints and limits

    Returns:
        A robot with base, column, arm, and end-effector links.
    """
    logger.info("=== Creating Multi-Link Robot ===")

    robot = Robot()

    # Base platform (fixed)
    base = Link.from_box(
        name="base_platform",
        extents=np.array([0.5, 0.5, 0.05]),
        color=np.array([0.3, 0.3, 0.3, 1.0]),
    )
    robot.add_link(base)

    # Vertical column (cylindrical)
    column = Link.from_cylinder(
        name="column",
        radius=0.08,
        height=0.6,
        color=np.array([0.4, 0.4, 0.8, 1.0]),  # Blue
    )
    robot.add_link(column)

    # Horizontal arm (box)
    arm = Link.from_box(
        name="horizontal_arm",
        extents=np.array([0.8, 0.1, 0.1]),
        color=np.array([0.8, 0.4, 0.2, 1.0]),  # Orange
    )
    robot.add_link(arm)

    # End effector (sphere)
    end_effector = Link.from_sphere(
        name="end_effector",
        radius=0.06,
        color=np.array([0.2, 0.8, 0.2, 1.0]),  # Green
    )
    robot.add_link(end_effector)

    # Fixed joint: base to column
    base_to_column = Joint(
        name="base_to_column_fixed",
        type=JointType.FIXED,
        parent_link_name="base_platform",
        child_link_name="column",
        center=np.array([0.0, 0.0, 0.025]),  # On top of base
    )
    robot.add_joint(base_to_column)

    # Revolute joint: column to arm (rotation around Z-axis)
    column_to_arm = Joint(
        name="column_to_arm_revolute",
        type=JointType.REVOLUTE,
        parent_link_name="column",
        child_link_name="horizontal_arm",
        center=np.array([0.0, 0.0, 0.3]),  # Middle of column
        direction=np.array([0.0, 0.0, 1.0]),  # Z-axis rotation
        min_=-np.pi,
        max_=np.pi,
    )
    robot.add_joint(column_to_arm)

    # Prismatic joint: arm to end effector (sliding along X-axis)
    arm_to_effector = Joint(
        name="arm_to_effector_prismatic",
        type=JointType.PRISMATIC,
        parent_link_name="horizontal_arm",
        child_link_name="end_effector",
        center=np.array([0.4, 0.0, 0.0]),  # End of arm
        direction=np.array([1.0, 0.0, 0.0]),  # X-axis translation
        min_=0.0,  # No negative extension
        max_=0.2,  # 20cm extension
    )
    robot.add_joint(arm_to_effector)

    logger.info(f"Created multi-link robot: {robot}")
    return robot


def demonstrate_robot_queries(robot: Robot):
    """Demonstrate various robot query methods.

    Args:
        robot: The robot to query.
    """
    logger.info("=== Robot Query Demonstrations ===")

    # Basic information
    logger.info(f"Robot summary: {robot}")
    logger.info(f"Total links: {len(robot.links())}")

    # Root and leaf analysis
    root_links = robot.root_links()
    leaf_links = robot.leaf_links()
    logger.info(f"Root links: {[link.name for link in root_links]}")
    logger.info(f"Leaf links: {[link.name for link in leaf_links]}")

    # Static vs dynamic classification
    static_links = robot.static_links()
    dynamic_links = robot.dynamic_links()
    logger.info(f"Static links: {[link.name for link in static_links]}")
    logger.info(f"Dynamic links: {[link.name for link in dynamic_links]}")

    # Individual component access
    for link in robot.links():
        logger.info(f"Link '{link.name}': {link}")

        # Find parent joint
        parent_joint = robot.parent_joint(link.name)
        if parent_joint:
            logger.info(
                f"  Parent joint: {parent_joint.name} ({parent_joint.type.name})"
            )
        else:
            logger.info("  No parent joint (root link)")

        # Find child joints
        child_joints = robot.child_joints(link.name)
        if child_joints:
            child_names = [j.name for j in child_joints]
            logger.info(f"  Child joints: {child_names}")
        else:
            logger.info("  No child joints (leaf link)")


def demonstrate_shape_varieties():
    """Demonstrate creating links with different geometric shapes.

    This shows all available primitive shapes and their parameters.
    """
    logger.info("=== Shape Variety Demonstration ===")

    robot = Robot()

    # Box shape
    box_link = Link.from_box(
        name="box_example",
        extents=np.array([0.2, 0.3, 0.1]),
        color=np.array([1.0, 0.0, 0.0, 1.0]),  # Red
    )
    robot.add_link(box_link)
    logger.info(f"Created box link: {box_link}")

    # Sphere shape
    sphere_link = Link.from_sphere(
        name="sphere_example",
        radius=0.15,
        center=np.array([0.0, 0.0, 0.0]),
        color=np.array([0.0, 1.0, 0.0, 1.0]),  # Green
    )
    robot.add_link(sphere_link)
    logger.info(f"Created sphere link: {sphere_link}")

    # Cylinder shape
    cylinder_link = Link.from_cylinder(
        name="cylinder_example",
        radius=0.08,
        height=0.25,
        color=np.array([0.0, 0.0, 1.0, 1.0]),  # Blue
    )
    robot.add_link(cylinder_link)
    logger.info(f"Created cylinder link: {cylinder_link}")

    # Cone shape
    cone_link = Link.from_cone(
        name="cone_example",
        radius=0.1,
        height=0.2,
        color=np.array([1.0, 1.0, 0.0, 1.0]),  # Yellow
    )
    robot.add_link(cone_link)
    logger.info(f"Created cone link: {cone_link}")

    # Capsule shape
    capsule_link = Link.from_capsule(
        name="capsule_example",
        radius=0.06,
        height=0.18,
        color=np.array([1.0, 0.0, 1.0, 1.0]),  # Magenta
    )
    robot.add_link(capsule_link)
    logger.info(f"Created capsule link: {capsule_link}")

    logger.info(f"Shape variety robot: {robot}")


def demonstrate_error_handling():
    """Demonstrate proper error handling and validation.

    This shows common mistakes and how the system handles them.
    """
    logger.info("=== Error Handling Demonstrations ===")

    robot = Robot()

    # Create a valid link first
    valid_link = Link.from_box("valid_link", np.array([0.1, 0.1, 0.1]))
    robot.add_link(valid_link)

    # Attempt duplicate link name
    try:
        duplicate_link = Link.from_box("valid_link", np.array([0.2, 0.2, 0.2]))
        robot.add_link(duplicate_link)
    except ValueError as e:
        logger.warning(f"Expected error for duplicate link: {e}")

    # Attempt joint with non-existent parent
    try:
        invalid_joint = Joint(
            name="invalid_joint",
            type=JointType.FIXED,
            parent_link_name="non_existent_parent",
            child_link_name="valid_link",
        )
        robot.add_joint(invalid_joint)
    except ValueError as e:
        logger.warning(f"Expected error for non-existent parent: {e}")

    # Attempt to access non-existent link
    try:
        robot.link("non_existent_link")
    except ValueError as e:
        logger.warning(f"Expected error for non-existent link: {e}")

    # Invalid joint direction (zero vector)
    try:
        _zero_direction_joint = Joint(
            name="zero_joint",
            type=JointType.REVOLUTE,
            parent_link_name="valid_link",
            child_link_name="valid_link",  # This will fail for other reasons too
            direction=np.array([0.0, 0.0, 0.0]),  # Zero vector
            center=np.array([0.0, 0.0, 0.0]),
        )
    except ValueError as e:
        logger.warning(f"Expected error for zero direction: {e}")

    logger.info("Error handling demonstration completed")


def main():
    """Main function demonstrating robot construction concepts."""
    logger.info("Starting Robot Construction Examples")

    # Create and analyze simple robot
    simple_robot = create_simple_robot()
    demonstrate_robot_queries(simple_robot)

    print("\n" + "=" * 60 + "\n")

    # Create and analyze complex robot
    complex_robot = create_multi_link_robot()
    demonstrate_robot_queries(complex_robot)

    print("\n" + "=" * 60 + "\n")

    # Demonstrate shape varieties
    demonstrate_shape_varieties()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate error handling
    demonstrate_error_handling()

    logger.info("Robot Construction Examples completed successfully!")


if __name__ == "__main__":
    main()
