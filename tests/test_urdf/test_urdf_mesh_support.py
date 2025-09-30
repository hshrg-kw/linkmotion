"""Test URDF writer mesh geometry support."""

import tempfile
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.robot.shape.box import Box
from linkmotion.urdf.writer import UrdfWriter
from linkmotion.urdf.parser import UrdfParser


def test_urdf_mesh_geometry_support() -> None:
    """Test URDF writer with mesh geometry support.

    This test verifies that mesh shapes are correctly exported to URDF format
    with fallback filename when no mesh export is performed.
    """
    # Create a simple cube mesh
    cube_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])

    # Create a robot with a mesh shape
    robot = Robot()

    # Add base link with box shape
    base_link = Link("base_link", Box(np.array([0.2, 0.2, 0.1])))
    robot.add_link(base_link)

    # Add mesh link with MeshShape (filename and scale parameters removed)
    mesh_shape = MeshShape(collision_mesh=cube_mesh)
    mesh_link = Link("mesh_link", mesh_shape)
    robot.add_link(mesh_link)

    # Add joint connecting them
    joint = Joint(
        name="base_to_mesh",
        type=JointType.FIXED,
        child_link_name="mesh_link",
        parent_link_name="base_link",
        center=np.array([0.0, 0.0, 0.1]),
    )
    robot.add_joint(joint)

    # Write URDF
    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "mesh_test_robot")

    # Verify URDF was generated
    assert urdf_string is not None
    assert "mesh_test_robot" in urdf_string

    # Check for fallback mesh filename (since no mesh export and no stored filename)
    assert 'filename="mesh.stl"' in urdf_string, (
        "Fallback mesh filename not found in URDF"
    )

    # Should not have scale attribute since scale parameter was removed
    assert "scale=" not in urdf_string, (
        "Unexpected scale attribute found - scale parameter was removed"
    )

    # Check that mesh geometry tag is present
    assert "<mesh filename=" in urdf_string, "Mesh geometry tag missing"

    # Test round-trip parsing (create a temp file since mesh needs file path)
    temp_urdf_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_string)
            temp_urdf_path = f.name

        # Try to parse it back (may fail to load mesh file, but should parse structure)
        parser = UrdfParser()
        try:
            _parsed_robot = parser.parse_file(temp_urdf_path)
            # If parsing succeeds, the parser handles missing mesh files gracefully
        except Exception as parse_error:
            # Expected behavior: parser should report missing mesh file
            error_msg = str(parse_error)
            assert any(
                expected in error_msg
                for expected in ["Failed to load mesh", "No such file"]
            ), f"Unexpected parsing error: {parse_error}"

    finally:
        # Clean up temp file
        if temp_urdf_path:
            Path(temp_urdf_path).unlink(missing_ok=True)


def test_urdf_mesh_without_filename() -> None:
    """Test URDF writer with mesh geometry fallback behavior.

    This test verifies that mesh shapes use appropriate fallback filenames
    and that scale attributes are not generated (since scale parameter was removed).
    """
    # Create a simple mesh without filename
    sphere_mesh = trimesh.creation.icosphere(radius=0.05)

    robot = Robot()

    # Create mesh shape (will use fallback filename in URDF)
    mesh_shape = MeshShape(collision_mesh=sphere_mesh)
    mesh_link = Link("sphere_link", mesh_shape)
    robot.add_link(mesh_link)

    # Write URDF
    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "fallback_test_robot")

    # Verify URDF was generated
    assert urdf_string is not None
    assert "fallback_test_robot" in urdf_string

    # Check for fallback filename
    assert 'filename="mesh.stl"' in urdf_string, "Fallback mesh filename not found"

    # Should not have scale attribute since scale parameter was removed
    assert "scale=" not in urdf_string, (
        "Unexpected scale attribute found - scale parameter was removed"
    )


if __name__ == "__main__":
    test_urdf_mesh_geometry_support()
    test_urdf_mesh_without_filename()
