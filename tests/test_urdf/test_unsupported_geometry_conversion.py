"""Test URDF writer conversion of unsupported geometry types to mesh geometry.

This module provides comprehensive tests for the automatic conversion of geometry
types that are not natively supported by URDF (like Capsule and Cone) into
mesh-based representations during export.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation as R

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.capsule import Capsule
from linkmotion.robot.shape.cone import Cone
from linkmotion.transform.transform import Transform
from linkmotion.urdf.writer import UrdfWriter
from linkmotion.urdf.parser import UrdfParser

logger = logging.getLogger(__name__)


def create_unsupported_geometry_robot() -> Robot:
    """Create a robot with unsupported geometry types for testing.

    Returns:
        Robot instance containing shapes that require conversion (Capsule, Cone).
    """
    robot = Robot()

    # Base link - supported geometry (control case)
    base_link = Link("base_link", Box(np.array([0.2, 0.2, 0.1])))
    robot.add_link(base_link)

    # Capsule link - unsupported in URDF, should be converted to mesh
    capsule_transform = Transform(
        R.from_euler("xyz", [0.1, 0.2, 0.0]), np.array([0.0, 0.0, 0.05])
    )
    capsule_shape = Capsule(
        radius=0.03,
        height=0.15,
        default_transform=capsule_transform,
        color=np.array([1.0, 0.0, 0.0, 1.0]),
    )
    capsule_link = Link("capsule_link", capsule_shape)
    robot.add_link(capsule_link)

    # Cone link - unsupported in URDF, should be converted to mesh
    cone_transform = Transform(
        R.from_euler("xyz", [0.0, 0.1, 0.3]), np.array([0.02, 0.01, 0.0])
    )
    cone_shape = Cone(
        radius=0.05,
        height=0.12,
        default_transform=cone_transform,
        color=np.array([0.0, 1.0, 0.0, 1.0]),
    )
    cone_link = Link("cone_link", cone_shape)
    robot.add_link(cone_link)

    # Add joints
    joints = [
        Joint(
            "base_to_capsule",
            JointType.REVOLUTE,
            "capsule_link",
            "base_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0.0, 0.0, 0.1]),
            min_=-np.pi / 4,
            max_=np.pi / 4,
        ),
        Joint(
            "capsule_to_cone",
            JointType.FIXED,
            "cone_link",
            "capsule_link",
            center=np.array([0.0, 0.0, 0.075]),
        ),
    ]

    for joint in joints:
        robot.add_joint(joint)

    return robot


def _run_unsupported_geometry_conversion() -> bool:
    """Run conversion of unsupported geometry types to mesh geometry in URDF export.

    Returns:
        True if all conversion tests pass, False otherwise.
    """
    logger.info("Testing unsupported geometry conversion")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        urdf_file = temp_path / "unsupported_geometry_robot.urdf"

        # Create robot with unsupported geometry types
        robot = create_unsupported_geometry_robot()

        logger.info(f"Created robot with {len(robot.links())} links")
        for link in robot.links():
            logger.debug(f"  - {link.name}: {type(link.shape).__name__}")

        # Test 1: Export URDF with mesh export enabled
        logger.info("Test 1: Exporting URDF with mesh export enabled")
        writer = UrdfWriter()

        try:
            writer.write_file(
                robot=robot,
                urdf_path=urdf_file,
                robot_name="unsupported_geometry_robot",
                export_meshes=True,
                meshes_dir="converted_meshes",
            )
            logger.info("URDF with converted geometry exported successfully")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

        # Check that URDF file was created
        if not urdf_file.exists():
            logger.error("URDF file not created")
            return False

        # Check that meshes directory was created
        meshes_dir = temp_path / "converted_meshes"
        if not meshes_dir.exists():
            logger.error("Meshes directory not created")
            return False

        # Check mesh files
        mesh_files = list(meshes_dir.glob("*.stl"))
        logger.info(
            f"Found {len(mesh_files)} mesh files: {[f.name for f in mesh_files]}"
        )

        if len(mesh_files) < 2:  # Should have at least capsule and cone meshes
            logger.error(
                f"Expected at least 2 mesh files for converted geometry, "
                f"got {len(mesh_files)}"
            )
            return False

        # Verify mesh files are valid
        valid_meshes = 0
        for mesh_file in mesh_files:
            try:
                loaded_mesh = trimesh.load(str(mesh_file), force="mesh")
                if (
                    isinstance(loaded_mesh, trimesh.Trimesh)
                    and len(loaded_mesh.vertices) > 0
                ):
                    valid_meshes += 1
                    logger.info(
                        f"Valid converted mesh: {mesh_file.name} "
                        f"({len(loaded_mesh.vertices)} vertices)"
                    )
                else:
                    logger.warning(f"Invalid mesh file: {mesh_file.name}")
            except Exception as e:
                logger.error(f"Failed to load mesh file {mesh_file.name}: {e}")

        if valid_meshes < 2:
            logger.error(f"Only {valid_meshes} valid converted mesh files")
            return False

        # Test 2: Check URDF content
        logger.info("Test 2: Checking URDF content")
        with open(urdf_file, "r") as f:
            urdf_content = f.read()

        # Should contain mesh references for converted geometry
        mesh_refs = urdf_content.count("<mesh filename=")
        collision_mesh_refs = urdf_content.count("<collision>")

        logger.info(
            f"URDF analysis - Mesh references: {mesh_refs}, "
            f"Collision elements: {collision_mesh_refs}"
        )

        # Should have mesh references for converted capsule and cone (visual + collision)
        if mesh_refs < 4:  # 2 shapes × 2 (visual + collision) = 4
            logger.error(f"Expected at least 4 mesh references, got {mesh_refs}")
            return False

        # Check that no unsupported geometry tags remain
        if "<capsule" in urdf_content or "<cone" in urdf_content:
            logger.error("URDF still contains unsupported geometry tags")
            return False

        logger.info("All unsupported geometry converted to mesh references")

        # Test 3: Round-trip parsing
        logger.info("Test 3: Testing round-trip parsing")
        try:
            parser = UrdfParser()
            parsed_robot = parser.parse_file(str(urdf_file))
            logger.info(
                f"Successfully parsed URDF: {len(parsed_robot.links())} links, "
                f"{len(parsed_robot._joint_dict)} joints"
            )

            # Check that converted shapes are now mesh shapes or handled gracefully
            mesh_links = 0
            for link in parsed_robot.links():
                if hasattr(link.shape, "collision_mesh"):  # MeshShape or similar
                    mesh_links += 1

            logger.info(f"Found {mesh_links} links with mesh-like shapes after parsing")

        except Exception as e:
            logger.error(f"Round-trip parsing failed: {e}")
            return False

        # Test 4: Export without mesh files (should still work with placeholders)
        logger.info("Test 4: Testing export without mesh files")
        urdf_no_meshes = temp_path / "no_mesh_export.urdf"

        try:
            writer.write_file(
                robot=robot,
                urdf_path=urdf_no_meshes,
                robot_name="no_mesh_robot",
                export_meshes=False,
            )

            with open(urdf_no_meshes, "r") as f:
                no_mesh_content = f.read()

            if "<mesh filename=" in no_mesh_content:
                logger.info("URDF without mesh export still contains mesh references")
            else:
                logger.error("URDF without mesh export missing mesh references")
                return False

        except Exception as e:
            logger.error(f"Export without meshes failed: {e}")
            return False

        logger.info("Unsupported geometry conversion test successful")
        return True


@pytest.fixture
def unsupported_geometry_robot() -> Robot:
    """Create a robot with unsupported geometry types for testing.

    Returns:
        Robot instance with unsupported geometry shapes.
    """
    return create_unsupported_geometry_robot()


def test_robot_creation_with_unsupported_geometry(
    unsupported_geometry_robot: Robot,
) -> None:
    """Test that robot with unsupported geometry is created correctly."""
    links = list(unsupported_geometry_robot.links())
    assert len(links) == 3, "Expected 3 links in unsupported geometry robot"

    # Verify we have the expected geometry types
    shape_types = [type(link.shape).__name__ for link in links]
    assert "Box" in shape_types, "Expected Box shape"
    assert "Capsule" in shape_types, "Expected Capsule shape"
    assert "Cone" in shape_types, "Expected Cone shape"


def test_unsupported_geometry_export_conversion() -> None:
    """Test that unsupported geometry gets properly converted during export."""
    success = _run_unsupported_geometry_conversion()
    assert success, "Unsupported geometry conversion failed"


def test_mesh_validation() -> None:
    """Test that generated mesh files are valid."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        robot = create_unsupported_geometry_robot()
        writer = UrdfWriter()

        writer.write_file(
            robot=robot,
            urdf_path=temp_path / "test.urdf",
            robot_name="test_robot",
            export_meshes=True,
            meshes_dir="test_meshes",
        )

        mesh_files = list((temp_path / "test_meshes").glob("*.stl"))
        assert len(mesh_files) >= 2, "Expected at least 2 mesh files"

        # Validate each mesh file
        for mesh_file in mesh_files:
            mesh = trimesh.load(str(mesh_file), force="mesh")
            assert isinstance(mesh, trimesh.Trimesh), f"Invalid mesh: {mesh_file.name}"
            assert len(mesh.vertices) > 0, f"Empty mesh: {mesh_file.name}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    success = _run_unsupported_geometry_conversion()
    if not success:
        exit(1)
