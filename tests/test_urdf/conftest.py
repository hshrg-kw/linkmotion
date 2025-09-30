"""Shared fixtures and utilities for URDF tests."""

import numpy as np
from typing import Set
from linkmotion.robot.robot import Robot
from linkmotion.urdf.parser import UrdfParser
from linkmotion.urdf.writer import UrdfWriter


def verify_robot_structure_consistency(robot1: Robot, robot2: Robot) -> None:
    """Verify that two robots have the same structural elements.

    Args:
        robot1: The original robot object.
        robot2: The robot object to compare against.

    Raises:
        AssertionError: If the robots don't have matching structure.
    """
    # Check link count
    assert len(robot1.links()) == len(robot2.links()), (
        f"Link count mismatch: {len(robot1.links())} vs {len(robot2.links())}"
    )

    # Check joint count
    assert len(robot1._joint_dict) == len(robot2._joint_dict), (
        f"Joint count mismatch: {len(robot1._joint_dict)} vs {len(robot2._joint_dict)}"
    )

    # Check that all links exist with same names
    original_link_names: Set[str] = {link.name for link in robot1.links()}
    roundtrip_link_names: Set[str] = {link.name for link in robot2.links()}
    assert original_link_names == roundtrip_link_names, (
        f"Link names changed: {original_link_names} vs {roundtrip_link_names}"
    )

    # Verify each link can be found by name
    for link1 in robot1.links():
        # This should not raise ValueError if link exists
        robot2.link(link1.name)

    # Check that all joints exist with same names
    original_joint_names: Set[str] = set(robot1._joint_dict.keys())
    roundtrip_joint_names: Set[str] = set(robot2._joint_dict.keys())
    assert original_joint_names == roundtrip_joint_names, (
        f"Joint names changed: {original_joint_names} vs {roundtrip_joint_names}"
    )


def assert_urdf_round_trip_preserves_structure(robot: Robot, robot_name: str) -> Robot:
    """Test that a robot can be written to URDF and parsed back with preserved structure.

    Args:
        robot: The robot to test round-trip with.
        robot_name: The name to use for the robot in URDF.

    Returns:
        The robot parsed back from URDF for further testing.

    Raises:
        AssertionError: If round-trip doesn't preserve structure.
    """
    # Write to URDF
    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, robot_name)

    # Verify URDF was generated
    assert urdf_string is not None
    assert robot_name in urdf_string

    # Parse back from URDF
    parser = UrdfParser()
    parsed_robot = parser.parse_string(urdf_string)

    # Verify structural consistency
    verify_robot_structure_consistency(robot, parsed_robot)

    return parsed_robot


def assert_axis_vectors_are_unit_vectors(urdf_content: str) -> None:
    """Verify that all axis vectors in URDF content are unit vectors.

    Args:
        urdf_content: The URDF content as a string.

    Raises:
        AssertionError: If any axis vector is not a unit vector.
    """
    import re

    # Extract all axis xyz values using regex
    pattern = r'<axis[^>]*xyz="([^"]*)"'
    matches = re.findall(pattern, urdf_content)

    assert len(matches) > 0, "No axis tags found in URDF"

    for axis_str in matches:
        try:
            values = np.array([float(x) for x in axis_str.split()])
            magnitude = np.linalg.norm(values)
            assert np.isclose(magnitude, 1.0, rtol=1e-3), (
                f"Axis vector {axis_str} is not a unit vector (magnitude: {magnitude:.4f})"
            )
        except ValueError:
            # Skip malformed axis values
            continue


def assert_urdf_contains_expected_elements(
    urdf_content: str,
    expected_links: int,
    expected_joints: int,
    expected_origins: int = None,
    expected_axes: int = None,
) -> None:
    """Verify that URDF contains expected structural elements.

    Args:
        urdf_content: The URDF content as a string.
        expected_links: Expected number of link elements.
        expected_joints: Expected number of joint elements.
        expected_origins: Expected minimum number of origin elements (optional).
        expected_axes: Expected minimum number of axis elements (optional).

    Raises:
        AssertionError: If URDF doesn't contain expected elements.
    """
    actual_links = urdf_content.count("<link")
    actual_joints = urdf_content.count("<joint")

    assert actual_links == expected_links, (
        f"Expected {expected_links} links, found {actual_links}"
    )

    assert actual_joints == expected_joints, (
        f"Expected {expected_joints} joints, found {actual_joints}"
    )

    if expected_origins is not None:
        actual_origins = urdf_content.count("<origin")
        assert actual_origins >= expected_origins, (
            f"Expected at least {expected_origins} origins, found {actual_origins}"
        )

    if expected_axes is not None:
        actual_axes = urdf_content.count("<axis")
        assert actual_axes >= expected_axes, (
            f"Expected at least {expected_axes} axes, found {actual_axes}"
        )
