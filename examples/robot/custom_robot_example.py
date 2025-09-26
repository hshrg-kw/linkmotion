"""Simple example demonstrating linkmotion.robot.custom features.

This example shows how to use CollisionMeshCustomizer to modify
robot meshes with operations like clipping, sweeping, and rotation.
"""

import numpy as np
import trimesh

from linkmotion.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.robot.custom import CollisionMeshCustomizer
from linkmotion.transform import Transform

import logging

logging.basicConfig(level=logging.INFO)


def create_simple_robot() -> Robot:
    """Create a simple robot with mesh shapes for demonstration."""
    robot = Robot()

    # Create a simple cube mesh
    cube_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    # Create mesh shapes for different links
    base_shape = MeshShape(
        collision_mesh=cube_mesh,
        visual_mesh=cube_mesh,
        default_transform=Transform(),
        color=np.array([1.0, 0.0, 0.0, 1.0]),  # Red
    )

    arm_shape = MeshShape(
        collision_mesh=cube_mesh,
        visual_mesh=cube_mesh,
        default_transform=Transform(),
        color=np.array([0.0, 1.0, 0.0, 1.0]),  # Green
    )

    # Create links
    base_link = Link("base_link", base_shape)
    arm_link = Link("arm_link", arm_shape)

    # Add links to robot
    robot.add_link(base_link)
    robot.add_link(arm_link)

    return robot


def main() -> None:
    """Demonstrate custom robot mesh operations."""
    robot = create_simple_robot()
    print(f"Created robot with {len(robot.links())} links")

    # Example 1: Remove parts outside a bounding box
    min_corner = np.array([-0.3, -0.3, -0.3])
    max_corner = np.array([0.3, 0.3, 0.3])

    CollisionMeshCustomizer.remove_outside_of_box(
        robot, {"base_link"}, min_corner, max_corner
    )
    print("Applied bounding box clipping to base_link")

    # Example 2: Sweep mesh operation
    initial_translate = np.array([0.0, 0.0, 0.1])
    sweep_translate = np.array([0.0, 0.0, 0.5])

    CollisionMeshCustomizer.sweep_mesh(
        robot, {"arm_link"}, initial_translate, sweep_translate
    )
    print("Applied sweep operation to arm_link")

    # Example 3: Create overlapping rotated copies
    center = np.array([0.0, 0.0, 0.0])
    rotation_axis = np.array([0.0, 0.0, 1.0])  # Z-axis
    delta_angle = np.pi / 4  # 45 degrees
    initial_angle = 0.0
    num_copies = 2

    CollisionMeshCustomizer.rotate_overlap(
        robot,
        {"base_link"},
        center,
        rotation_axis,
        delta_angle,
        initial_angle,
        num_copies,
    )
    print("Applied rotation overlap to base_link")

    # Example 4: Convert mesh to bounding primitive
    CollisionMeshCustomizer.from_mesh_to_bounding_primitive(robot, {"arm_link"})
    print("Converted arm_link mesh to bounding primitive")

    print("Custom robot operations completed successfully!")


if __name__ == "__main__":
    main()
