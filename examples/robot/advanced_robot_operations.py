"""
Advanced Robot Operations Example

This example demonstrates sophisticated robot manipulation techniques
including structural modifications, robot composition, and integration
with collision detection and visualization systems.

Topics covered:
- Link renaming and structural modifications
- Link division for detailed modeling
- Robot concatenation for complex multi-robot systems
- Collision object generation for safety checking
- Visual mesh creation for rendering and display
- Advanced error handling and validation

Run with: uv run python examples/robot/advanced_robot_operations.py
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


def create_base_robot() -> Robot:
    """Create a base robot for modification demonstrations.

    Returns:
        A simple robot with base and arm components.
    """
    robot = Robot()

    # Base platform
    base = Link.from_box(
        name="base_platform",
        extents=np.array([0.3, 0.3, 0.1]),
        color=np.array([0.5, 0.5, 0.5, 1.0]),
    )
    robot.add_link(base)

    # Single arm link
    arm = Link.from_cylinder(
        name="arm_segment",
        radius=0.06,
        height=0.4,
        color=np.array([0.8, 0.3, 0.3, 1.0]),
    )
    robot.add_link(arm)

    # Joint connecting them
    base_to_arm = Joint(
        name="base_to_arm_joint",
        type=JointType.REVOLUTE,
        parent_link_name="base_platform",
        child_link_name="arm_segment",
        center=np.array([0.0, 0.0, 0.05]),
        direction=np.array([0.0, 0.0, 1.0]),
    )
    robot.add_joint(base_to_arm)

    return robot


def demonstrate_link_renaming():
    """Demonstrate link renaming operations."""
    logger.info("=== Link Renaming Demonstration ===")

    robot = create_base_robot()
    logger.info(f"Original robot: {robot}")

    # Show original structure
    logger.info("Original link names:")
    for link in robot.links():
        logger.info(f"  - {link.name}")

    # Rename the arm segment
    logger.info("Renaming 'arm_segment' to 'manipulator_arm'")
    robot.rename_link("arm_segment", "manipulator_arm")

    # Show updated structure
    logger.info("Updated link names:")
    for link in robot.links():
        logger.info(f"  - {link.name}")

    # Verify joint connections were updated
    parent_joint = robot.parent_joint("manipulator_arm")
    if parent_joint:
        logger.info(f"Parent joint of renamed link: {parent_joint.name}")
        logger.info(
            f"Joint connects: {parent_joint.parent_link_name} -> {parent_joint.child_link_name}"
        )
    else:
        logger.info("No parent joint found for manipulator_arm")

    # Attempt invalid rename (should fail gracefully)
    try:
        robot.rename_link("manipulator_arm", "base_platform")  # Name conflict
    except ValueError as e:
        logger.warning(f"Expected error for name conflict: {e}")

    logger.info("Link renaming demonstration completed")


def demonstrate_link_division():
    """Demonstrate dividing a single link into multiple parts."""
    logger.info("=== Link Division Demonstration ===")

    robot = create_base_robot()
    logger.info(f"Robot before division: {robot}")

    # Create new links to replace the arm segment
    upper_arm = Link.from_cylinder(
        name="upper_arm_section",
        radius=0.06,
        height=0.2,  # Half the original height
        color=np.array([0.8, 0.4, 0.2, 1.0]),  # Orange
    )

    lower_arm = Link.from_cylinder(
        name="lower_arm_section",
        radius=0.05,  # Slightly smaller
        height=0.18,
        color=np.array([0.2, 0.4, 0.8, 1.0]),  # Blue
    )

    # Create joint connecting the new parts
    elbow_joint = Joint(
        name="elbow_joint",
        type=JointType.REVOLUTE,
        parent_link_name="upper_arm_section",
        child_link_name="lower_arm_section",
        center=np.array([0.0, 0.0, 0.1]),  # Middle of upper arm
        direction=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
        min_=-np.pi / 2,
        max_=0.0,  # Elbow can only bend inward
    )

    # Perform the division
    logger.info(
        "Dividing 'arm_segment' into 'upper_arm_section' and 'lower_arm_section'"
    )
    robot.divide_link("arm_segment", upper_arm, lower_arm, elbow_joint)

    logger.info(f"Robot after division: {robot}")

    # Verify the new structure
    logger.info("New robot structure:")
    for link in robot.links():
        parent_joint = robot.parent_joint(link.name)
        child_joints = robot.child_joints(link.name)

        parent_info = parent_joint.name if parent_joint else "None (root)"
        child_info = [j.name for j in child_joints] if child_joints else "None (leaf)"

        logger.info(f"  {link.name}: parent={parent_info}, children={child_info}")

    logger.info("Link division demonstration completed")


def create_gripper_robot() -> Robot:
    """Create a small gripper robot for concatenation demonstration.

    Returns:
        A robot representing a simple gripper mechanism.
    """
    gripper = Robot()

    # Gripper base (connection point)
    gripper_base = Link.from_sphere(
        name="gripper_base", radius=0.04, color=np.array([0.6, 0.6, 0.6, 1.0])
    )
    gripper.add_link(gripper_base)

    # Left gripper finger
    left_finger = Link.from_box(
        name="left_finger",
        extents=np.array([0.02, 0.08, 0.06]),
        color=np.array([0.2, 0.8, 0.2, 1.0]),
    )
    gripper.add_link(left_finger)

    # Right gripper finger
    right_finger = Link.from_box(
        name="right_finger",
        extents=np.array([0.02, 0.08, 0.06]),
        color=np.array([0.8, 0.2, 0.2, 1.0]),
    )
    gripper.add_link(right_finger)

    # Joints for gripper actuation
    left_joint = Joint(
        name="left_finger_joint",
        type=JointType.PRISMATIC,
        parent_link_name="gripper_base",
        child_link_name="left_finger",
        center=np.array([-0.02, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),  # Slide along X
        min_=-0.02,
        max_=0.02,
    )
    gripper.add_joint(left_joint)

    right_joint = Joint(
        name="right_finger_joint",
        type=JointType.PRISMATIC,
        parent_link_name="gripper_base",
        child_link_name="right_finger",
        center=np.array([0.02, 0.0, 0.0]),
        direction=np.array([-1.0, 0.0, 0.0]),  # Slide opposite direction
        min_=-0.02,
        max_=0.02,
    )
    gripper.add_joint(right_joint)

    return gripper


def demonstrate_robot_concatenation():
    """Demonstrate combining multiple robots into one system."""
    logger.info("=== Robot Concatenation Demonstration ===")

    # Create base robot (arm)
    arm_robot = create_base_robot()
    logger.info(f"Arm robot: {arm_robot}")

    # Create gripper robot
    gripper_robot = create_gripper_robot()
    logger.info(f"Gripper robot: {gripper_robot}")

    # Create joint to connect them
    arm_to_gripper = Joint(
        name="arm_to_gripper_connection",
        type=JointType.FIXED,
        parent_link_name="arm_segment",  # From arm robot
        child_link_name="gripper_base",  # Root of gripper robot
        center=np.array([0.0, 0.0, 0.2]),  # At end of arm
    )

    # Concatenate the robots
    logger.info("Concatenating gripper onto arm robot")
    arm_robot.concatenate_robot(gripper_robot, arm_to_gripper)

    logger.info(f"Combined robot: {arm_robot}")

    # Analyze the combined structure
    logger.info("Combined robot structure:")
    all_links = arm_robot.links()
    for link in all_links:
        parent_joint = arm_robot.parent_joint(link.name)
        child_joints = arm_robot.child_joints(link.name)

        parent_info = parent_joint.name if parent_joint else "None (root)"
        child_info = [j.name for j in child_joints] if child_joints else "None (leaf)"

        logger.info(f"  {link.name}: parent={parent_info}, children={child_info}")

    # Verify root and leaf analysis
    roots = arm_robot.root_links()
    leaves = arm_robot.leaf_links()
    logger.info(f"Root links: {[link.name for link in roots]}")
    logger.info(f"Leaf links: {[link.name for link in leaves]}")

    logger.info("Robot concatenation demonstration completed")


def demonstrate_collision_objects():
    """Demonstrate collision object generation for safety checking."""
    logger.info("=== Collision Object Generation ===")

    robot = create_base_robot()

    # Generate collision objects for each link
    logger.info("Generating collision objects:")
    for link in robot.links():
        # Create collision object with transformation
        transform = Transform(translate=np.array([0.1, 0.2, 0.3]))
        link.collision_object(transform)
        logger.info(f"    Transformed collision object created for {link.name}")

    logger.info("Collision object generation completed")


def demonstrate_visual_meshes():
    """Demonstrate visual mesh generation for rendering."""
    logger.info("=== Visual Mesh Generation ===")

    robot = create_base_robot()

    # Generate visual meshes for each link
    logger.info("Generating visual meshes:")
    for link in robot.links():
        # Create visual mesh at default pose
        visual_mesh = link.visual_mesh()
        logger.info(
            f"  {link.name}: vertices={len(visual_mesh.vertices)}, faces={len(visual_mesh.faces)}"
        )

        # Create visual mesh with transformation
        from scipy.spatial.transform import Rotation as R

        transform = Transform(
            rotate=R.from_euler("x", np.pi / 4)
        )  # 45-degree rotation around X
        transformed_mesh = link.visual_mesh(transform)
        logger.info(
            f"    Transformed mesh: vertices={len(transformed_mesh.vertices)}, faces={len(transformed_mesh.faces)}"
        )

        # Check if mesh has colors
        if (
            hasattr(transformed_mesh, "visual")
            and transformed_mesh.visual is not None
            and hasattr(transformed_mesh.visual, "vertex_colors")
            and transformed_mesh.visual.vertex_colors is not None
        ):
            logger.info(
                f"    Mesh has vertex colors: {transformed_mesh.visual.vertex_colors.shape}"
            )
        else:
            logger.info("    Mesh has no vertex colors")

    logger.info("Visual mesh generation completed")


def demonstrate_advanced_error_handling():
    """Demonstrate advanced error handling scenarios."""
    logger.info("=== Advanced Error Handling ===")

    robot = create_base_robot()

    # Test invalid division scenarios
    logger.info("Testing invalid link division scenarios:")

    # Attempt to divide with conflicting names
    try:
        conflicting_link = Link.from_box(
            "base_platform", np.array([0.1, 0.1, 0.1])
        )  # Name conflict
        dummy_link = Link.from_box("dummy", np.array([0.1, 0.1, 0.1]))
        dummy_joint = Joint(
            name="dummy_joint",
            type=JointType.FIXED,
            parent_link_name="base_platform",
            child_link_name="dummy",
        )
        robot.divide_link("arm_segment", conflicting_link, dummy_link, dummy_joint)
    except ValueError as e:
        logger.warning(f"Expected error for name conflict: {e}")

    # Test invalid concatenation scenarios
    logger.info("Testing invalid concatenation scenarios:")

    # Create a robot with multiple roots (invalid for concatenation)
    invalid_gripper = Robot()
    root1 = Link.from_sphere("root1", 0.05)
    root2 = Link.from_sphere("root2", 0.05)
    invalid_gripper.add_link(root1)
    invalid_gripper.add_link(root2)

    try:
        connection_joint = Joint(
            name="invalid_connection",
            type=JointType.FIXED,
            parent_link_name="arm_segment",
            child_link_name="root1",
        )
        robot.concatenate_robot(invalid_gripper, connection_joint)
    except ValueError as e:
        logger.warning(f"Expected error for multiple roots: {e}")

    logger.info("Advanced error handling demonstration completed")


def main():
    """Main function demonstrating advanced robot operations."""
    logger.info("Starting Advanced Robot Operations Examples")

    # Demonstrate link renaming
    demonstrate_link_renaming()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate link division
    demonstrate_link_division()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate robot concatenation
    demonstrate_robot_concatenation()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate collision objects
    demonstrate_collision_objects()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate visual meshes
    demonstrate_visual_meshes()

    print("\n" + "=" * 60 + "\n")

    # Demonstrate advanced error handling
    demonstrate_advanced_error_handling()

    logger.info("Advanced Robot Operations Examples completed successfully!")


if __name__ == "__main__":
    main()
