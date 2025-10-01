"""Practical example demonstrating unsupported geometry conversion in URDF export.

This module provides a comprehensive test for mixed geometry robot export, including
both supported and unsupported geometry types that require conversion to mesh format.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.capsule import Capsule
from linkmotion.robot.shape.cone import Cone
from linkmotion.transform.transform import Transform
from linkmotion.urdf.writer import UrdfWriter

logger = logging.getLogger(__name__)


def create_mixed_geometry_robot() -> Robot:
    """Create a robot with mixed supported and unsupported geometry types.

    Returns:
        Robot instance with a mix of native URDF geometry and shapes requiring conversion.
    """
    robot = Robot()

    # Base platform - supported box
    base_link = Link(
        "base_platform",
        Box(np.array([0.4, 0.4, 0.08]), color=np.array([0.7, 0.7, 0.7, 1.0])),
    )
    robot.add_link(base_link)

    # Column - supported cylinder
    column_link = Link(
        "support_column",
        Cylinder(
            0.05,
            0.3,
            Transform(translate=np.array([0.0, 0.0, 0.15])),
            color=np.array([0.3, 0.3, 0.8, 1.0]),
        ),
    )
    robot.add_link(column_link)

    # Connecting rod - unsupported capsule (should be converted)
    rod_link = Link(
        "connecting_rod",
        Capsule(
            0.02,
            0.2,
            Transform(R.from_euler("y", np.pi / 2), np.array([0.1, 0.0, 0.0])),
            color=np.array([0.8, 0.2, 0.2, 1.0]),
        ),
    )
    robot.add_link(rod_link)

    # End effector housing - supported sphere
    housing_link = Link(
        "effector_housing",
        Sphere(
            0.04,
            center=np.array([0.0, 0.0, 0.02]),
            color=np.array([0.2, 0.8, 0.2, 1.0]),
        ),
    )
    robot.add_link(housing_link)

    # Tool tip - unsupported cone (should be converted)
    tip_link = Link(
        "tool_tip",
        Cone(
            0.015,
            0.06,
            Transform(translate=np.array([0.0, 0.0, -0.03])),
            color=np.array([1.0, 0.8, 0.0, 1.0]),
        ),
    )
    robot.add_link(tip_link)

    # Add joints to connect the links
    joints = [
        Joint(
            "base_to_column",
            JointType.FIXED,
            "support_column",
            "base_platform",
            center=np.array([0.0, 0.0, 0.04]),
        ),
        Joint(
            "column_to_rod",
            JointType.REVOLUTE,
            "connecting_rod",
            "support_column",
            direction=np.array([0, 0, 1]),
            center=np.array([0.0, 0.0, 0.15]),
            min_=-np.pi,
            max_=np.pi,
        ),
        Joint(
            "rod_to_housing",
            JointType.PRISMATIC,
            "effector_housing",
            "connecting_rod",
            direction=np.array([1, 0, 0]),
            center=np.array([0.1, 0.0, 0.0]),
            min_=0.0,
            max_=0.08,
        ),
        Joint(
            "housing_to_tip",
            JointType.FIXED,
            "tool_tip",
            "effector_housing",
            center=np.array([0.0, 0.0, 0.04]),
        ),
    ]

    for joint in joints:
        robot.add_joint(joint)

    return robot


def analyze_robot_geometry(robot: Robot) -> Dict[str, Any]:
    """Analyze geometry types in a robot.

    Args:
        robot: Robot instance to analyze.

    Returns:
        Dictionary containing geometry analysis results.
    """
    supported_count = 0
    unsupported_count = 0
    shape_types = {}

    for link in robot.links():
        shape_type = type(link.shape).__name__
        shape_types[link.name] = shape_type

        if isinstance(link.shape, (Capsule, Cone)):
            unsupported_count += 1
            logger.info(f"Unsupported geometry: {link.name} ({shape_type})")
        else:
            supported_count += 1
            logger.info(f"Supported geometry: {link.name} ({shape_type})")

    return {
        "supported_count": supported_count,
        "unsupported_count": unsupported_count,
        "shape_types": shape_types,
        "total_links": len(robot.links()),
    }


def _run_practical_conversion_example() -> bool:
    """Run practical example of mixed geometry robot with conversions.

    Returns:
        True if all tests pass, False otherwise.
    """
    logger.info("Starting practical mixed geometry robot conversion test")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        urdf_file = temp_path / "mixed_geometry_robot.urdf"

        # Create the robot
        robot = create_mixed_geometry_robot()

        # Analyze geometry
        geometry_analysis = analyze_robot_geometry(robot)
        logger.info(
            f"Robot geometry summary: {geometry_analysis['supported_count']} "
            f"supported, {geometry_analysis['unsupported_count']} unsupported shapes"
        )

        # Export URDF with mesh conversion
        logger.info("Exporting URDF with automatic geometry conversion")
        writer = UrdfWriter()

        try:
            writer.write_file(
                robot=robot,
                urdf_path=urdf_file,
                robot_name="mixed_geometry_robot",
                export_meshes=True,
                meshes_dir="geometry_meshes",
            )
            logger.info("URDF export successful")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

        # Analyze results
        logger.info("Analyzing exported files")

        # Check mesh files
        meshes_dir = temp_path / "geometry_meshes"
        mesh_files = list(meshes_dir.glob("*.stl"))

        logger.info(f"Created {len(mesh_files)} mesh files")
        for mesh_file in mesh_files:
            size_kb = mesh_file.stat().st_size / 1024
            logger.debug(f"Mesh file: {mesh_file.name} ({size_kb:.1f} KB)")

        # Check URDF content
        with open(urdf_file, "r") as f:
            urdf_content = f.read()

        # Count different geometry types in URDF
        geometry_counts = {
            "box": urdf_content.count("<box "),
            "sphere": urdf_content.count("<sphere "),
            "cylinder": urdf_content.count("<cylinder "),
            "mesh": urdf_content.count("<mesh "),
        }

        logger.info(f"URDF geometry counts: {geometry_counts}")

        # Verify conversions
        success = True
        unsupported_count = geometry_analysis["unsupported_count"]

        # Should have mesh elements for converted shapes (2 conversions × 2 elements each = 4)
        expected_mesh_refs = unsupported_count * 2  # visual + collision
        if geometry_counts["mesh"] >= expected_mesh_refs:
            logger.info(
                f"Expected mesh references found "
                f"({geometry_counts['mesh']} >= {expected_mesh_refs})"
            )
        else:
            logger.error(
                f"Missing mesh references "
                f"({geometry_counts['mesh']} < {expected_mesh_refs})"
            )
            success = False

        # Should have no unsupported geometry tags
        unsupported_tags = ["<capsule", "<cone"]
        found_unsupported = any(tag in urdf_content for tag in unsupported_tags)
        if not found_unsupported:
            logger.info("No unsupported geometry tags in URDF")
        else:
            logger.error("Found unsupported geometry tags in URDF")
            success = False

        # Verify mesh files were created for conversions
        if len(mesh_files) >= unsupported_count:
            logger.info(
                f"Sufficient mesh files created "
                f"({len(mesh_files)} >= {unsupported_count})"
            )
        else:
            logger.error(
                f"Insufficient mesh files ({len(mesh_files)} < {unsupported_count})"
            )
            success = False

        # Test parsing
        logger.info("Testing round-trip compatibility")
        try:
            from linkmotion.urdf.parser import UrdfParser

            parser = UrdfParser()
            parsed_robot = parser.parse_file(str(urdf_file))

            logger.info(
                f"URDF parsed successfully: {len(parsed_robot.links())} links, "
                f"{len(parsed_robot._joint_dict)} joints"
            )

            # Check that we can identify mesh shapes in parsed robot
            mesh_shapes = 0
            for link in parsed_robot.links():
                if hasattr(link.shape, "collision_mesh"):
                    mesh_shapes += 1

            logger.info(f"Found {mesh_shapes} mesh-based shapes after parsing")

        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            success = False

        if success:
            logger.info("Practical conversion example successful")
        else:
            logger.error("Practical conversion example had issues")

        return success


@pytest.fixture
def mixed_geometry_robot() -> Robot:
    """Create a robot with mixed geometry types for testing.

    Returns:
        Robot instance with mixed supported and unsupported geometry.
    """
    return create_mixed_geometry_robot()


def test_mixed_geometry_robot_creation(mixed_geometry_robot: Robot) -> None:
    """Test that mixed geometry robot is created correctly."""
    analysis = analyze_robot_geometry(mixed_geometry_robot)

    assert analysis["total_links"] == 5, "Expected 5 links in mixed geometry robot"
    assert analysis["supported_count"] == 3, "Expected 3 supported geometry types"
    assert analysis["unsupported_count"] == 2, "Expected 2 unsupported geometry types"


def test_mixed_geometry_export_with_meshes() -> None:
    """Test export of mixed geometry robot with mesh conversion."""
    success = _run_practical_conversion_example()
    assert success, "Mixed geometry robot export with mesh conversion failed"


def test_geometry_analysis() -> None:
    """Test geometry analysis functionality."""
    robot = create_mixed_geometry_robot()
    analysis = analyze_robot_geometry(robot)

    assert "supported_count" in analysis
    assert "unsupported_count" in analysis
    assert "shape_types" in analysis
    assert "total_links" in analysis

    # Verify specific shape types
    expected_shapes = {"Box", "Cylinder", "Capsule", "Sphere", "Cone"}
    actual_shapes = set(analysis["shape_types"].values())
    assert actual_shapes == expected_shapes, (
        f"Expected {expected_shapes}, got {actual_shapes}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    success = _run_practical_conversion_example()
    if not success:
        exit(1)
