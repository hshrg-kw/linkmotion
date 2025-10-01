"""Test cone transform coordinate system alignment in URDF export.

This module investigates and tests the coordinate system differences between
linkmotion's Cone class and trimesh's cone creation, ensuring proper alignment
during URDF export conversions.
"""

import logging
from typing import List, Tuple

import numpy as np
import pytest
import trimesh

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.shape.cone import Cone
from linkmotion.robot.shape.box import Box
from linkmotion.transform.transform import Transform
from linkmotion.urdf.writer import UrdfWriter

logger = logging.getLogger(__name__)


def investigate_cone_coordinate_systems() -> float:
    """Investigate coordinate system differences between Cone and trimesh cone.

    Returns:
        Z-offset needed to align trimesh cone with linkmotion Cone coordinate system.
    """
    logger.info("Investigating Cone coordinate systems")

    # Create a cone using linkmotion Cone class
    cone_shape = Cone(radius=0.05, height=0.12)
    logger.info(
        f"Linkmotion Cone: radius={cone_shape.radius}, height={cone_shape.height}"
    )

    # Create a cone using trimesh
    trimesh_cone = trimesh.creation.cone(radius=0.05, height=0.12)
    z_coords = trimesh_cone.vertices[:, 2]

    logger.info(
        f"Trimesh cone Z-coordinates: min={z_coords.min():.6f}, "
        f"max={z_coords.max():.6f}, mean={z_coords.mean():.6f}"
    )

    # Expected vs actual positioning
    expected_base_z = -0.12 / 2  # -0.06
    expected_apex_z = 0.12 / 2  # +0.06

    actual_base_z = z_coords.min()
    actual_apex_z = z_coords.max()

    logger.info(
        f"Expected base Z: {expected_base_z:.6f}, apex Z: {expected_apex_z:.6f}"
    )
    logger.info(f"Actual base Z: {actual_base_z:.6f}, apex Z: {actual_apex_z:.6f}")

    # Calculate the transform needed to align them
    z_offset = expected_base_z - actual_base_z
    logger.info(f"Required Z offset: {z_offset:.6f}")

    return z_offset


def _investigate_cone_transform_in_urdf() -> List[Tuple[str, List[float]]]:
    """Investigate cone transform coordinates in URDF export.

    Returns:
        List of tuples containing (link_name, origin_xyz_coordinates).
    """
    logger.info("Testing Cone transform in URDF export")

    # Create a robot with cone geometry
    robot = Robot()

    # Reference box (for comparison)
    box_link = Link("reference_box", Box(np.array([0.1, 0.1, 0.1])))
    robot.add_link(box_link)

    # Cone with identity transform (should be centered at origin)
    cone_identity = Cone(radius=0.05, height=0.12)
    cone_identity_link = Link("cone_identity", cone_identity)
    robot.add_link(cone_identity_link)

    # Cone with translation
    cone_translated = Cone(
        radius=0.05,
        height=0.12,
        default_transform=Transform(translate=np.array([0.1, 0.0, 0.05])),
    )
    cone_translated_link = Link("cone_translated", cone_translated)
    robot.add_link(cone_translated_link)

    # Export URDF
    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "cone_transform_test")

    logger.debug(f"Generated URDF with cone transforms:\n{urdf_string}")

    # Look for cone mesh origins in the URDF
    lines = urdf_string.split("\n")
    cone_origins = []
    current_link = None

    for line in lines:
        line = line.strip()
        if line.startswith("<link name="):
            current_link = line.split('"')[1]
        elif "<origin xyz=" in line and current_link and "cone" in current_link:
            # Extract origin values
            start = line.find('xyz="') + 5
            end = line.find('"', start)
            xyz_str = line[start:end]
            xyz_values = [float(x) for x in xyz_str.split()]
            cone_origins.append((current_link, xyz_values))

    logger.info(f"Found {len(cone_origins)} cone origins in URDF")
    for link_name, origin in cone_origins:
        logger.info(f"  {link_name}: {origin}")

    return cone_origins


def _investigate_corrected_cone_conversion() -> bool:
    """Investigate corrected cone conversion and coordinate system alignment.

    Returns:
        True if coordinate systems match, False otherwise.
    """
    logger.info("Testing corrected cone conversion")

    # Check the current conversion in the writer
    writer = UrdfWriter()

    # Create a test cone
    test_cone = Cone(radius=0.05, height=0.12)

    # Try the conversion
    converted_mesh_shape = writer._convert_unsupported_to_mesh(test_cone)

    if converted_mesh_shape:
        mesh = converted_mesh_shape.collision_mesh
        z_coords = mesh.vertices[:, 2]

        logger.info(
            f"Converted mesh Z-coordinates: min={z_coords.min():.6f}, "
            f"max={z_coords.max():.6f}"
        )

        # Check if this matches the expected Cone coordinate system
        expected_base_z = -0.12 / 2  # -0.06
        expected_apex_z = 0.12 / 2  # +0.06

        actual_base_z = z_coords.min()
        actual_apex_z = z_coords.max()

        logger.info(
            f"Expected base Z: {expected_base_z:.6f}, actual: {actual_base_z:.6f}"
        )
        logger.info(
            f"Expected apex Z: {expected_apex_z:.6f}, actual: {actual_apex_z:.6f}"
        )

        coordinate_systems_match = np.isclose(
            actual_base_z, expected_base_z, atol=1e-6
        ) and np.isclose(actual_apex_z, expected_apex_z, atol=1e-6)

        if coordinate_systems_match:
            logger.info("Coordinate systems match")
        else:
            logger.warning(
                "Coordinate systems don't match - coordinate system bug detected"
            )

        return coordinate_systems_match
    else:
        logger.error("Conversion failed")
        return False


@pytest.fixture
def test_robot() -> Robot:
    """Create a test robot with cone geometry for testing.

    Returns:
        Robot instance with cone shapes for coordinate system testing.
    """
    robot = Robot()

    # Reference box
    box_link = Link("reference_box", Box(np.array([0.1, 0.1, 0.1])))
    robot.add_link(box_link)

    # Cone with identity transform
    cone_identity = Cone(radius=0.05, height=0.12)
    cone_identity_link = Link("cone_identity", cone_identity)
    robot.add_link(cone_identity_link)

    return robot


def test_cone_urdf_export_origins() -> None:
    """Test that cone origins are correctly exported in URDF."""
    cone_origins = _investigate_cone_transform_in_urdf()

    assert len(cone_origins) > 0, "No cone origins found in URDF"

    # Verify that cone origins are properly defined
    for link_name, origin in cone_origins:
        assert len(origin) == 3, f"Origin for {link_name} should have 3 coordinates"
        assert all(isinstance(coord, (int, float)) for coord in origin), (
            f"All coordinates for {link_name} should be numeric"
        )


def test_cone_conversion_accuracy() -> None:
    """Test that cone conversion maintains coordinate system accuracy."""
    conversion_ok = _investigate_corrected_cone_conversion()

    assert conversion_ok, "Cone coordinate system conversion failed accuracy test"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run investigation and tests
    logger.info("Starting cone transform investigation")

    # Investigate coordinate systems
    z_offset = investigate_cone_coordinate_systems()

    # Test current behavior
    cone_origins = _investigate_cone_transform_in_urdf()

    # Test the conversion directly
    conversion_ok = _investigate_corrected_cone_conversion()

    if not conversion_ok:
        logger.warning(
            "Bug identified: trimesh.creation.cone uses different coordinate system"
        )
        logger.info(f"Solution: Apply Z-offset of {z_offset:.6f} during conversion")
    else:
        logger.info("No coordinate system mismatch detected")
