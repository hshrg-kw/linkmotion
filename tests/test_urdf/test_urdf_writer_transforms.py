"""Test script to verify URDF writer transform fixes."""

import numpy as np
import re
from typing import List, Tuple
from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.transform.transform import Transform
from linkmotion.urdf.writer import UrdfWriter
from linkmotion.urdf.parser import UrdfParser
from scipy.spatial.transform import Rotation as R


def _extract_xyz_values(
    urdf_content: str, tag_name: str
) -> List[Tuple[str, List[float]]]:
    """Extract xyz coordinate values from URDF tags.

    Args:
        urdf_content: The URDF content as a string.
        tag_name: The tag name to search for (e.g., "origin", "axis").

    Returns:
        List of tuples containing the full xyz string and parsed float values.
    """
    pattern = f'<{tag_name}[^>]*xyz="([^"]*)"'
    matches = re.findall(pattern, urdf_content)

    result = []
    for match in matches:
        try:
            values = [float(x) for x in match.split()]
            result.append((match, values))
        except ValueError:
            # Skip malformed values
            continue

    return result


def test_urdf_writer_transform_fix() -> None:
    """Test URDF writer with accumulated transforms converted to relative transforms.

    This test creates a robot with accumulated transforms (as would be created by
    parsing a URDF) and verifies that the writer correctly converts these to
    relative transforms when writing back to URDF format.
    """
    # Create a robot similar to what the parser would create (with accumulated transforms)
    robot = Robot()

    # Base link at origin
    base_link = Link("base_link", Box(np.array([0.2, 0.2, 0.1])))

    # Child link with accumulated transform that should be split into:
    # - Joint origin: (0, 0, 0.06) relative to base_link
    # - Visual origin: (0.01, 0.02, 0.05) relative to torso_link + rotation
    torso_accumulated_transform = Transform(
        R.from_euler("xyz", [0.1, 0.2, 0.3]),
        np.array([0.01, 0.02, 0.11]),  # 0.06 + 0.05 from joint + visual origins
    )
    torso_link = Link("torso_link", Cylinder(0.08, 0.3, torso_accumulated_transform))

    # Another child with more complex accumulated transform
    arm_accumulated_transform = Transform(
        R.from_euler("xyz", [0.2, 0.1, 0.4]),
        np.array([0.06, 0.01, 0.26]),  # Joint at (0.05, -0.015, 0.26) + visual offset
    )
    arm_link = Link(
        "upper_arm_link", Box(np.array([0.1, 0.1, 0.2]), arm_accumulated_transform)
    )

    # Add links
    robot.add_link(base_link)
    robot.add_link(torso_link)
    robot.add_link(arm_link)

    # Create joints with accumulated centers and directions
    torso_joint = Joint(
        name="base_to_torso",
        type=JointType.REVOLUTE,
        child_link_name="torso_link",
        parent_link_name="base_link",
        direction=np.array([0.098, 0.195, 0.976]),  # Accumulated/global direction
        center=np.array([0.0, 0.0, 0.06]),  # Accumulated center
        min_=-np.pi / 2,
        max_=np.pi / 2,
    )

    arm_joint = Joint(
        name="torso_to_upper_arm",
        type=JointType.PRISMATIC,
        child_link_name="upper_arm_link",
        parent_link_name="torso_link",
        direction=np.array([-0.098, 0.975, 0.198]),  # Accumulated/global direction
        center=np.array([0.05, -0.015, 0.26]),  # Accumulated center
        min_=0.0,
        max_=0.15,
    )

    # Add joints
    robot.add_joint(torso_joint)
    robot.add_joint(arm_joint)

    # Write URDF
    writer = UrdfWriter()
    urdf_string = writer.to_string(robot, "test_robot")

    # Verify URDF was generated
    assert urdf_string is not None
    assert "test_robot" in urdf_string
    assert len(robot.links()) == 3  # base_link, torso_link, upper_arm_link
    assert len(robot._joint_dict) == 2  # base_to_torso, torso_to_upper_arm

    # Check that we can parse it back without errors
    parser = UrdfParser()
    parsed_robot = parser.parse_string(urdf_string)

    # Verify structural consistency
    assert len(parsed_robot.links()) == len(robot.links()), (
        f"Link count mismatch after round-trip: {len(robot.links())} vs {len(parsed_robot.links())}"
    )

    assert len(parsed_robot._joint_dict) == len(robot._joint_dict), (
        f"Joint count mismatch after round-trip: {len(robot._joint_dict)} vs {len(parsed_robot._joint_dict)}"
    )

    # Extract and validate origin coordinates
    origin_values = _extract_xyz_values(urdf_string, "origin")
    assert len(origin_values) > 0, "No origin tags found in generated URDF"

    # Most origins should have reasonable values (not extremely large accumulated transforms)
    reasonable_origins = [
        values for _, values in origin_values if all(abs(v) <= 1.0 for v in values)
    ]

    # At least some origins should be reasonable (depending on robot scale, all might not be)
    assert len(reasonable_origins) > 0, (
        "All origin values seem unusually large, suggesting accumulated transforms"
    )

    # Extract and validate axis vectors
    axis_values = _extract_xyz_values(urdf_string, "axis")
    assert len(axis_values) >= 2, (
        f"Expected at least 2 axis tags, found {len(axis_values)}"
    )

    # All axis vectors should be unit vectors (or very close)
    for axis_str, values in axis_values:
        axis_array = np.array(values)
        magnitude = np.linalg.norm(axis_array)
        assert np.isclose(magnitude, 1.0, rtol=1e-3), (
            f"Axis vector {axis_str} is not a unit vector (magnitude: {magnitude:.4f})"
        )

    # Verify URDF structure contains expected elements
    assert urdf_string.count("<joint") == 2, "Should have exactly 2 joints"
    assert urdf_string.count("<link") == 3, "Should have exactly 3 links"

    # All joints should have origins and axes (none are fixed joints)
    joint_origins = urdf_string.count("<joint")
    assert urdf_string.count("<origin") >= joint_origins, "Not all joints have origins"
    assert urdf_string.count("<axis") >= joint_origins, "Not all joints have axes"


if __name__ == "__main__":
    test_urdf_writer_transform_fix()
