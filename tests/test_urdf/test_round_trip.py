"""Test round-trip: parse URDF -> write URDF -> parse again."""

from linkmotion.urdf.parser import UrdfParser
from conftest import (
    assert_urdf_round_trip_preserves_structure,
)


def test_round_trip() -> None:
    """Test that parsing and writing URDF maintains consistency.

    This test verifies that a URDF can be parsed into a Robot object,
    written back to URDF format, and parsed again without losing
    structural information like links and joints.
    """
    # Test with a simple URDF that includes materials, links, and joints
    original_urdf = """<?xml version="1.0"?>
<robot name="robot">
  <material name="material_0_50_0_50_0_50_1_00">
    <color rgba="0.500000 0.500000 0.500000 1.000000" />
  </material>
  <material name="material_0_00_0_00_0_80_1_00">
    <color rgba="0.000000 0.000000 0.800000 1.000000" />
  </material>
  <material name="material_1_00_0_00_0_00_1_00">
    <color rgba="1.000000 0.000000 0.000000 1.000000" />
  </material>

  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.200000 0.200000 0.100000" />
      </geometry>
      <origin xyz="0.010000 0.020000 0.050000" rpy="0.100000 0.200000 0.300000" />
    </visual>
  </link>

  <link name="torso_link">
    <visual>
      <geometry>
        <cylinder radius="0.080000" length="0.300000" />
      </geometry>
    </visual>
  </link>

  <joint name="base_to_torso" type="revolute">
    <parent link="base_link" />
    <child link="torso_link" />
    <origin xyz="0.000000 0.000000 0.060000" />
    <axis xyz="0.0 0.0 1.0" />
    <limit lower="-1.570000" upper="1.570000" effort="10.0" velocity="1.0" />
  </joint>
</robot>"""

    # Parse original URDF
    parser = UrdfParser()
    robot1 = parser.parse_string(original_urdf)

    # Verify initial parsing worked
    assert len(robot1.links()) == 2, f"Expected 2 links, got {len(robot1.links())}"
    assert len(robot1._joint_dict) == 1, (
        f"Expected 1 joint, got {len(robot1._joint_dict)}"
    )

    # Test round-trip using helper function
    parsed_robot = assert_urdf_round_trip_preserves_structure(robot1, "robot")

    # Additional verification that the helper doesn't cover
    assert "base_link" in [link.name for link in parsed_robot.links()]
    assert "torso_link" in [link.name for link in parsed_robot.links()]
    assert "base_to_torso" in parsed_robot._joint_dict


if __name__ == "__main__":
    test_round_trip()
