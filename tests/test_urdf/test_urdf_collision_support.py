"""Test URDF writer collision geometry support."""

import numpy as np
import trimesh
from typing import List, Tuple
from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.transform.transform import Transform
from linkmotion.urdf.writer import UrdfWriter
from linkmotion.urdf.parser import UrdfParser
from scipy.spatial.transform import Rotation as R


def _extract_link_section(urdf_content: str, link_name: str) -> str:
    """Extract the complete URDF section for a specific link.

    Args:
        urdf_content: The complete URDF content as a string.
        link_name: The name of the link to extract.

    Returns:
        The link section content including opening and closing tags.

    Raises:
        AssertionError: If the link section cannot be found or parsed.
    """
    link_start = urdf_content.find(f'<link name="{link_name}">')
    assert link_start != -1, f"Link {link_name} not found in URDF"

    link_end = urdf_content.find("</link>", link_start)
    assert link_end != -1, f"Link {link_name} end tag not found"

    return urdf_content[link_start : link_end + 7]  # +7 for </link>


def _verify_link_has_visual_and_collision(
    urdf_content: str, link_names: List[str]
) -> None:
    """Verify that all specified links have both visual and collision elements.

    Args:
        urdf_content: The complete URDF content as a string.
        link_names: List of link names to check.

    Raises:
        AssertionError: If any link is missing visual or collision elements.
    """
    for link_name in link_names:
        link_section = _extract_link_section(urdf_content, link_name)

        has_visual = "<visual>" in link_section
        has_collision = "<collision>" in link_section

        assert has_visual, f"Link {link_name} missing visual element"
        assert has_collision, f"Link {link_name} missing collision element"


def _verify_geometry_duplication(
    urdf_content: str, geometry_checks: List[Tuple[str, str]]
) -> None:
    """Verify that geometry elements appear in both visual and collision sections.

    Args:
        urdf_content: The complete URDF content as a string.
        geometry_checks: List of tuples containing (geometry_tag, geometry_description).

    Raises:
        AssertionError: If geometry elements are not properly duplicated.
    """
    for geom_tag, geom_name in geometry_checks:
        occurrences = urdf_content.count(geom_tag)

        if geom_tag in urdf_content:
            # Should appear exactly twice per geometry type (visual + collision)
            assert occurrences == 2, (
                f"{geom_name} appears {occurrences} times, expected 2 (visual + collision)"
            )


def test_urdf_collision_support() -> None:
    """Test that URDF writer includes collision information for all geometry types.

    This test creates a robot with various geometry types (box, sphere, cylinder, mesh)
    and verifies that the generated URDF includes both visual and collision elements
    for each link, with matching geometry specifications.
    """
    # Create robot with different shape types
    robot = Robot()

    # Box link with transform
    box_transform = Transform(
        R.from_euler("xyz", [0.1, 0.2, 0.3]), np.array([0.01, 0.02, 0.03])
    )
    box_link = Link("box_link", Box(np.array([0.2, 0.15, 0.1]), box_transform))
    robot.add_link(box_link)

    # Sphere link (no transform)
    sphere_link = Link("sphere_link", Sphere(0.05))
    robot.add_link(sphere_link)

    # Cylinder link with transform
    cylinder_transform = Transform(
        R.from_euler("xyz", [0.0, 0.5, 0.0]), np.array([0.1, 0.0, 0.0])
    )
    cylinder_link = Link("cylinder_link", Cylinder(0.03, 0.2, cylinder_transform))
    robot.add_link(cylinder_link)

    # Mesh link
    cube_mesh = trimesh.creation.box(extents=[0.08, 0.08, 0.08])
    mesh_shape = MeshShape(collision_mesh=cube_mesh)
    mesh_link = Link("mesh_link", mesh_shape)
    robot.add_link(mesh_link)

    # Add joints to connect links in a chain
    joints = [
        Joint(
            "base_to_sphere",
            JointType.FIXED,
            "sphere_link",
            "box_link",
            center=np.array([0.0, 0.0, 0.1]),
        ),
        Joint(
            "sphere_to_cylinder",
            JointType.REVOLUTE,
            "cylinder_link",
            "sphere_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0.0, 0.0, 0.05]),
            min_=-np.pi / 2,
            max_=np.pi / 2,
        ),
        Joint(
            "cylinder_to_mesh",
            JointType.PRISMATIC,
            "mesh_link",
            "cylinder_link",
            direction=np.array([1, 0, 0]),
            center=np.array([0.0, 0.0, 0.1]),
            min_=0.0,
            max_=0.05,
        ),
    ]

    for joint in joints:
        robot.add_joint(joint)

    # Verify robot structure
    assert len(robot.links()) == 4, f"Expected 4 links, got {len(robot.links())}"
    assert len(robot._joint_dict) == 3, (
        f"Expected 3 joints, got {len(robot._joint_dict)}"
    )

    # Write URDF
    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "collision_test_robot")

    # Verify URDF was generated
    assert urdf_string is not None
    assert "collision_test_robot" in urdf_string

    # Count visual and collision elements
    visual_count = urdf_string.count("<visual>")
    collision_count = urdf_string.count("<collision>")

    # Should have same number of visual and collision elements
    assert visual_count == collision_count, (
        f"Visual/collision count mismatch: visual={visual_count}, collision={collision_count}"
    )
    assert visual_count > 0, "No visual elements found in URDF"

    # Verify each link has both visual and collision elements
    link_names = ["box_link", "sphere_link", "cylinder_link", "mesh_link"]
    _verify_link_has_visual_and_collision(urdf_string, link_names)

    # Verify that geometry types are preserved in both visual and collision
    geometry_checks = [
        ("<box size=", "Box geometry"),
        ("<sphere radius=", "Sphere geometry"),
        ("<cylinder radius=", "Cylinder geometry"),
        ("<mesh filename=", "Mesh geometry"),
    ]
    _verify_geometry_duplication(urdf_string, geometry_checks)

    # Test round-trip parsing to ensure collision info is handled properly
    parser = UrdfParser()
    parsed_robot = parser.parse_string(urdf_string)

    # Verify structure is preserved
    assert len(parsed_robot.links()) == len(robot.links()), (
        f"Round-trip link count mismatch: {len(robot.links())} vs {len(parsed_robot.links())}"
    )
    assert len(parsed_robot._joint_dict) == len(robot._joint_dict), (
        f"Round-trip joint count mismatch: {len(robot._joint_dict)} vs {len(parsed_robot._joint_dict)}"
    )


if __name__ == "__main__":
    test_urdf_collision_support()
