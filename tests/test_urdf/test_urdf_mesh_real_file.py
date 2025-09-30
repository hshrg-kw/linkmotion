"""Test URDF writer with actual mesh files."""

import tempfile
import numpy as np
import trimesh
from pathlib import Path

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.robot.shape.box import Box
from linkmotion.urdf.writer import UrdfWriter
from linkmotion.urdf.parser import UrdfParser


def test_urdf_mesh_with_real_file() -> None:
    """Test URDF mesh support with actual mesh files.

    This test creates real mesh files in a temporary directory, creates
    a robot with mesh shapes, writes the URDF with fallback filenames,
    and then parses it back to verify that the mesh structure is preserved
    correctly (without filename and scale attributes which were removed).
    """
    # Create a temporary directory for our mesh file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        meshes_dir = temp_path / "meshes"
        meshes_dir.mkdir()

        # Create and save a real mesh file
        cube_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        mesh_file = meshes_dir / "test_cube.stl"
        cube_mesh.export(str(mesh_file))

        # Verify mesh file was created
        assert mesh_file.exists(), f"Mesh file was not created at {mesh_file}"

        # Create URDF in the temp directory
        urdf_file = temp_path / "robot.urdf"

        # Create robot with mesh shape
        robot = Robot()

        # Base link
        base_link = Link("base_link", Box(np.array([0.2, 0.2, 0.1])))
        robot.add_link(base_link)

        # Load the mesh we just created
        loaded_mesh = trimesh.load(str(mesh_file), force="mesh")

        # Create mesh shape (filename and scale parameters removed)
        mesh_shape = MeshShape(collision_mesh=loaded_mesh)
        mesh_link = Link("mesh_link", mesh_shape)
        robot.add_link(mesh_link)

        # Add joint
        joint = Joint(
            name="base_to_mesh",
            type=JointType.REVOLUTE,
            child_link_name="mesh_link",
            parent_link_name="base_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0.0, 0.0, 0.1]),
            min_=-np.pi / 4,
            max_=np.pi / 4,
        )
        robot.add_joint(joint)

        # Write URDF
        writer = UrdfWriter()
        urdf_string = writer.to_string(robot, "mesh_file_test_robot")

        # Verify URDF was generated
        assert urdf_string is not None
        assert "mesh_file_test_robot" in urdf_string

        # Save URDF to file
        with open(urdf_file, "w") as f:
            f.write(urdf_string)

        assert urdf_file.exists(), f"URDF file was not created at {urdf_file}"

        # Verify fallback mesh filename in URDF (since filename parameter was removed)
        assert 'filename="mesh.stl"' in urdf_string, (
            "Fallback mesh filename not found in URDF"
        )

        # Should not have scale attribute since scale parameter was removed
        assert "scale=" not in urdf_string, (
            "Unexpected scale attribute found - scale parameter was removed"
        )

        # Test round-trip parsing with actual file
        parser = UrdfParser()
        parsed_robot = parser.parse_file(str(urdf_file))

        # Verify basic structure
        assert len(parsed_robot.links()) == 2, (
            f"Expected 2 links, got {len(parsed_robot.links())}"
        )
        assert len(parsed_robot._joint_dict) == 1, (
            f"Expected 1 joint, got {len(parsed_robot._joint_dict)}"
        )

        # Check that mesh link was loaded
        mesh_link_parsed = parsed_robot.link("mesh_link")

        # Since we use fallback filename "mesh.stl" which doesn't exist,
        # the parser should fall back to creating a Box shape (default fallback)
        assert isinstance(mesh_link_parsed.shape, Box), (
            f"Expected Box (fallback), got {type(mesh_link_parsed.shape)}"
        )

        # Box shapes don't have filename and scale attributes either
        assert not hasattr(mesh_link_parsed.shape, "filename"), (
            "Box shape should not have filename attribute"
        )

        assert not hasattr(mesh_link_parsed.shape, "scale"), (
            "Box shape should not have scale attribute"
        )


if __name__ == "__main__":
    test_urdf_mesh_with_real_file()
