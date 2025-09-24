"""
TransformManager Hierarchy Example

This example demonstrates how to use TransformManager to create and manage
hierarchical transform relationships, such as those found in robot kinematic chains,
scene graphs, or multi-body systems.
"""

import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion.transform import Transform, TransformManager

# Set up logging to see transform operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_robot_arm_hierarchy():
    """Create a simple robot arm hierarchy using TransformManager."""
    logger.info("Creating robot arm hierarchy...")

    manager = TransformManager()

    # Robot arm structure:
    # Base (0) -> Shoulder (1) -> Elbow (2) -> Wrist (3) -> End-effector (4)

    # Add base (world frame)
    manager.add_node(
        node_id=0,
        default_transform=Transform(),  # Identity - base is at origin
    )

    # Add shoulder joint (30cm above base)
    manager.add_node(
        node_id=1,
        parent_id=0,
        default_transform=Transform(translate=np.array([0.0, 0.0, 0.3])),
    )

    # Add elbow joint (40cm forward from shoulder)
    manager.add_node(
        node_id=2,
        parent_id=1,
        default_transform=Transform(translate=np.array([0.4, 0.0, 0.0])),
    )

    # Add wrist joint (30cm forward from elbow)
    manager.add_node(
        node_id=3,
        parent_id=2,
        default_transform=Transform(translate=np.array([0.3, 0.0, 0.0])),
    )

    # Add end-effector (10cm forward from wrist)
    manager.add_node(
        node_id=4,
        parent_id=3,
        default_transform=Transform(translate=np.array([0.1, 0.0, 0.0])),
    )

    return manager


def demonstrate_basic_operations(manager):
    """Demonstrate basic TransformManager operations."""
    logger.info("\n=== Basic Operations ===")

    # Get all node IDs
    all_nodes = manager.get_node_ids()
    logger.info(f"All nodes: {all_nodes}")

    # Get root nodes
    root_nodes = manager.get_root_nodes()
    logger.info(f"Root nodes: {root_nodes}")

    # Get children of a node
    shoulder_children = manager.get_children(1)
    logger.info(f"Shoulder (1) children: {shoulder_children}")

    # Calculate world transforms
    logger.info("\nWorld positions (default pose):")
    for node_id in all_nodes:
        world_transform = manager.get_world_transform(node_id)
        logger.info(f"Node {node_id}: {world_transform.position}")


def demonstrate_joint_movements(manager):
    """Demonstrate how to move joints and see the effects."""
    logger.info("\n=== Joint Movements ===")

    # Move shoulder joint: rotate 45 degrees around Z-axis
    shoulder_rotation = Transform(rotate=R.from_euler("z", 45, degrees=True))
    manager.set_local_transform(1, shoulder_rotation)
    logger.info("Rotated shoulder 45° around Z-axis")

    # Move elbow joint: rotate -90 degrees around Y-axis
    elbow_transform = Transform(
        rotate=R.from_euler("y", -90, degrees=True),
        translate=np.array([0.4, 0.0, 0.0]),  # Keep original translation
    )
    manager.set_local_transform(2, elbow_transform)
    logger.info("Rotated elbow -90° around Y-axis")

    # Calculate new world positions
    logger.info("\nWorld positions after joint movements:")
    for node_id in manager.get_node_ids():
        world_transform = manager.get_world_transform(node_id)
        logger.info(f"Node {node_id}: {world_transform.position}")

    # Show end-effector orientation
    end_effector_world = manager.get_world_transform(4)
    euler_angles = end_effector_world.rotation.as_euler("xyz", degrees=True)
    logger.info(f"End-effector orientation (XYZ Euler): {euler_angles}")


def demonstrate_relative_movements(manager):
    """Demonstrate relative transform applications."""
    logger.info("\n=== Relative Movements ===")

    # Apply a relative rotation to the wrist
    relative_wrist_rotation = Transform(rotate=R.from_euler("x", 30, degrees=True))
    manager.apply_relative_transform(3, relative_wrist_rotation)
    logger.info("Applied 30° relative rotation to wrist around X-axis")

    # Show updated end-effector position
    end_effector_world = manager.get_world_transform(4)
    logger.info(f"End-effector after wrist rotation: {end_effector_world.position}")


def demonstrate_reset_operations(manager):
    """Demonstrate reset operations."""
    logger.info("\n=== Reset Operations ===")

    # Reset a single joint
    manager.reset_node_transform(2)  # Reset elbow
    logger.info("Reset elbow to default pose")

    # Show positions after elbow reset
    logger.info("Positions after elbow reset:")
    for node_id in [2, 3, 4]:  # Show affected nodes
        world_transform = manager.get_world_transform(node_id)
        logger.info(f"Node {node_id}: {world_transform.position}")

    # Reset all joints
    manager.reset_all_transforms()
    logger.info("\nReset all joints to default poses")

    # Verify all back to defaults
    end_effector_world = manager.get_world_transform(4)
    logger.info(f"End-effector back to default: {end_effector_world.position}")


def create_scene_graph_example():
    """Create a more complex scene graph with multiple objects."""
    logger.info("\n=== Scene Graph Example ===")

    manager = TransformManager()

    # Scene structure:
    # World (0)
    # ├── Table (1)
    # │   ├── Cup (2)
    # │   └── Plate (3)
    # └── Robot (4)
    #     └── Gripper (5)

    # World frame
    manager.add_node(0, default_transform=Transform())

    # Table (1m forward, 0.7m high)
    manager.add_node(
        1, parent_id=0, default_transform=Transform(translate=np.array([1.0, 0.0, 0.7]))
    )

    # Cup on table (20cm to the right)
    manager.add_node(
        2,
        parent_id=1,
        default_transform=Transform(translate=np.array([0.0, 0.2, 0.05])),
    )

    # Plate on table (20cm to the left)
    manager.add_node(
        3,
        parent_id=1,
        default_transform=Transform(translate=np.array([0.0, -0.2, 0.02])),
    )

    # Robot base (50cm to the left of world origin)
    manager.add_node(
        4,
        parent_id=0,
        default_transform=Transform(translate=np.array([0.0, -0.5, 0.0])),
    )

    # Robot gripper (extended toward table)
    manager.add_node(
        5, parent_id=4, default_transform=Transform(translate=np.array([0.8, 0.0, 0.8]))
    )

    logger.info("Created scene graph with table objects and robot")

    # Show all object positions
    object_names = {
        0: "World",
        1: "Table",
        2: "Cup",
        3: "Plate",
        4: "Robot",
        5: "Gripper",
    }
    logger.info("\nScene object positions:")
    for node_id in manager.get_node_ids():
        world_transform = manager.get_world_transform(node_id)
        name = object_names[node_id]
        logger.info(f"{name} ({node_id}): {world_transform.position}")

    # Move the table and show how objects move with it
    logger.info("\nMoving table backward by 0.2m...")
    table_new_transform = Transform(
        translate=np.array([0.8, 0.0, 0.7])
    )  # Move back 0.2m
    manager.set_local_transform(1, table_new_transform)

    logger.info("Updated positions:")
    for node_id in [1, 2, 3]:  # Table and its objects
        world_transform = manager.get_world_transform(node_id)
        name = object_names[node_id]
        logger.info(f"{name} ({node_id}): {world_transform.position}")

    return manager


def demonstrate_node_removal(manager):
    """Demonstrate node removal and hierarchy cleanup."""
    logger.info("\n=== Node Removal ===")

    # Show initial structure
    logger.info(f"Initial nodes: {manager.get_node_ids()}")
    logger.info(f"Table children: {manager.get_children(1)}")

    # Remove the cup (node 2)
    manager.remove_node(2)
    logger.info("Removed cup (node 2)")
    logger.info(f"Remaining nodes: {manager.get_node_ids()}")
    logger.info(f"Table children after cup removal: {manager.get_children(1)}")

    # Remove robot (node 4) - this should also remove gripper (node 5)
    manager.remove_node(4)
    logger.info("Removed robot (node 4) - gripper (5) should also be removed")
    logger.info(f"Final nodes: {manager.get_node_ids()}")


def main():
    """Run all TransformManager examples."""
    logger.info("=== TransformManager Hierarchy Example ===")

    # Create and demonstrate robot arm
    robot_manager = create_robot_arm_hierarchy()
    demonstrate_basic_operations(robot_manager)
    demonstrate_joint_movements(robot_manager)
    demonstrate_relative_movements(robot_manager)
    demonstrate_reset_operations(robot_manager)

    # Create and demonstrate scene graph
    scene_manager = create_scene_graph_example()
    demonstrate_node_removal(scene_manager)

    logger.info("\n=== All examples completed successfully! ===")


if __name__ == "__main__":
    main()
