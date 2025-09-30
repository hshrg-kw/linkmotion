"""Test URDF writer with complex robot from models/simple_modified.urdf"""

import pytest
from pathlib import Path
from linkmotion.urdf.parser import UrdfParser
from linkmotion.urdf.writer import UrdfWriter


def test_complex_urdf_round_trip() -> None:
    """Test round-trip with complex URDF that has multiple joints and links.

    This test verifies that complex URDF files with multiple links and joints
    can be parsed, written back to URDF format, and parsed again while
    preserving all structural elements and relationships.
    """
    # Check if the URDF file exists
    urdf_path = Path("models/toy_model/toy_model.urdf")
    if not urdf_path.exists():
        pytest.skip(f"URDF file not found: {urdf_path}")

    # Load the complex URDF
    parser = UrdfParser()
    robot1 = parser.parse_file(str(urdf_path))

    # Verify we have a meaningful robot structure
    assert len(robot1.links()) > 0, "Robot should have at least one link"
    initial_link_count = len(robot1.links())
    initial_joint_count = len(robot1._joint_dict)

    # Write it back
    writer = UrdfWriter()
    generated_urdf = writer.to_string(robot1, "complex_robot")

    # Verify URDF was generated
    assert generated_urdf is not None
    assert "complex_robot" in generated_urdf
    assert "<link" in generated_urdf  # Should have at least one link

    # Parse the generated URDF
    robot2 = parser.parse_string(generated_urdf)

    # Verify structure consistency after round-trip
    assert len(robot1.links()) == len(robot2.links()), (
        f"Link count mismatch: {initial_link_count} vs {len(robot2.links())}"
    )

    assert len(robot1._joint_dict) == len(robot2._joint_dict), (
        f"Joint count mismatch: {initial_joint_count} vs {len(robot2._joint_dict)}"
    )

    # Check that all links exist with same names
    original_link_names = {link.name for link in robot1.links()}
    roundtrip_link_names = {link.name for link in robot2.links()}
    assert original_link_names == roundtrip_link_names, (
        f"Link names changed: {original_link_names} vs {roundtrip_link_names}"
    )

    # Verify each link can be found by name
    for link1 in robot1.links():
        # This should not raise ValueError if link exists
        robot2.link(link1.name)

    # Check that all joints exist with same names
    original_joint_names = set(robot1._joint_dict.keys())
    roundtrip_joint_names = set(robot2._joint_dict.keys())
    assert original_joint_names == roundtrip_joint_names, (
        f"Joint names changed: {original_joint_names} vs {roundtrip_joint_names}"
    )


if __name__ == "__main__":
    test_complex_urdf_round_trip()
