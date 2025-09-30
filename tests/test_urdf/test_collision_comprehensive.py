"""Comprehensive test for URDF collision support with various scenarios."""

import numpy as np
import trimesh
from typing import List, Tuple, Any
from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.transform.transform import Transform
from linkmotion.urdf.writer import UrdfWriter
from scipy.spatial.transform import Rotation as R
from conftest import assert_urdf_round_trip_preserves_structure


def test_simple_robot_collision_support() -> None:
    """Test collision support for a simple single-link robot."""
    robot = Robot()
    box_link = Link("base", Box(np.array([0.2, 0.2, 0.1])))
    robot.add_link(box_link)

    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "simple_robot")

    # Verify both visual and collision elements are present
    assert "<visual>" in urdf_string, "Simple robot missing visual element"
    assert "<collision>" in urdf_string, "Simple robot missing collision element"


def test_multi_joint_robot_collision_support() -> None:
    """Test collision support for a robot with multiple joints and links."""
    robot = Robot()

    # Create links with transforms
    base_transform = Transform(
        R.from_euler("xyz", [0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.05])
    )
    base_link = Link("base_link", Box(np.array([0.2, 0.2, 0.1]), base_transform))
    robot.add_link(base_link)

    torso_link = Link("torso_link", Cylinder(0.08, 0.3))
    robot.add_link(torso_link)

    # Add joint
    joint = Joint(
        name="base_to_torso",
        type=JointType.REVOLUTE,
        child_link_name="torso_link",
        parent_link_name="base_link",
        direction=np.array([0, 0, 1]),
        center=np.array([0.0, 0.0, 0.06]),
        min_=-np.pi / 2,
        max_=np.pi / 2,
    )
    robot.add_joint(joint)

    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "multi_joint_robot")

    # Count visual and collision elements
    visual_count = urdf_string.count("<visual>")
    collision_count = urdf_string.count("<collision>")

    # Should have equal counts for visual and collision (one per link)
    assert visual_count == collision_count == 2, (
        f"Expected 2 visual and 2 collision elements, got visual={visual_count}, collision={collision_count}"
    )


def test_mesh_robot_collision_support() -> None:
    """Test collision support for robots with mesh geometry."""
    robot = Robot()
    cube_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mesh_shape = MeshShape(collision_mesh=cube_mesh)
    mesh_link = Link("mesh_link", mesh_shape)
    robot.add_link(mesh_link)

    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "mesh_robot")

    # Mesh should appear in both visual and collision sections with fallback filename
    mesh_occurrences = urdf_string.count('<mesh filename="mesh.stl"')
    assert mesh_occurrences == 2, (
        f"Mesh should appear twice (visual + collision), found {mesh_occurrences} occurrences"
    )


def test_collision_round_trip_parsing() -> None:
    """Test that collision information is preserved through round-trip parsing."""
    # Create multi-joint robot for comprehensive round-trip test
    robot = Robot()

    base_transform = Transform(
        R.from_euler("xyz", [0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.05])
    )
    base_link = Link("base_link", Box(np.array([0.2, 0.2, 0.1]), base_transform))
    robot.add_link(base_link)

    torso_link = Link("torso_link", Cylinder(0.08, 0.3))
    robot.add_link(torso_link)

    joint = Joint(
        name="base_to_torso",
        type=JointType.REVOLUTE,
        child_link_name="torso_link",
        parent_link_name="base_link",
        direction=np.array([0, 0, 1]),
        center=np.array([0.0, 0.0, 0.06]),
        min_=-np.pi / 2,
        max_=np.pi / 2,
    )
    robot.add_joint(joint)

    # Test round-trip using helper function
    parsed_robot = assert_urdf_round_trip_preserves_structure(robot, "roundtrip_robot")

    # Generate URDF from parsed robot to verify collision info is preserved
    writer = UrdfWriter()
    roundtrip_urdf = writer.to_string(parsed_robot, "roundtrip_robot")

    # Verify collision information is still present after round-trip
    assert "<collision>" in roundtrip_urdf, (
        "Round-trip parsing lost collision information"
    )


def test_multiple_geometry_types_collision_support() -> None:
    """Test collision support for all supported geometry types in one robot."""
    robot = Robot()

    # Define various shape types and their expected URDF tags
    shapes_and_info: List[Tuple[Any, str, str]] = [
        (Box(np.array([0.1, 0.1, 0.1])), "box_link", "<box "),
        (Sphere(0.05), "sphere_link", "<sphere "),
        (Cylinder(0.03, 0.2), "cylinder_link", "<cylinder "),
    ]

    prev_link_name = None
    for i, (shape, link_name, expected_tag) in enumerate(shapes_and_info):
        link = Link(link_name, shape)
        robot.add_link(link)

        if prev_link_name:
            joint = Joint(
                name=f"joint_{i}",
                type=JointType.FIXED,
                child_link_name=link_name,
                parent_link_name=prev_link_name,
                center=np.array([0.0, 0.0, 0.1]),
            )
            robot.add_joint(joint)

        prev_link_name = link_name

    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "multi_shape_robot")

    # Verify each geometry type appears exactly twice (visual + collision)
    for _, _, expected_tag in shapes_and_info:
        count = urdf_string.count(expected_tag)
        assert count == 2, (
            f"{expected_tag.strip('<')} should appear twice (visual + collision), found {count} occurrences"
        )


def test_comprehensive_collision_support() -> None:
    """Comprehensive test of collision support in various scenarios.

    This test runs multiple sub-tests to ensure collision support works
    correctly across different robot configurations and geometry types.
    """
    # Run all individual collision support tests
    test_simple_robot_collision_support()
    test_multi_joint_robot_collision_support()
    test_mesh_robot_collision_support()
    test_collision_round_trip_parsing()
    test_multiple_geometry_types_collision_support()


if __name__ == "__main__":
    test_comprehensive_collision_support()
