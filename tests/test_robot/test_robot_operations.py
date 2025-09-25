import pytest

from linkmotion.robot import Robot, Link, Joint, JointType


@pytest.fixture
def simple_robot():
    """Provides a simple robot with a single kinematic chain."""
    robot = Robot()
    links = [Link.from_sphere(f"L{i}", 1) for i in range(4)]
    joints = [
        Joint(
            f"J{i + 1}",
            JointType.FLOATING,
            parent_link_name=f"L{i}",
            child_link_name=f"L{i + 1}",
        )
        for i in range(3)
    ]
    for link in links:
        robot.add_link(link)
    for joint in joints:
        robot.add_joint(joint)
    return robot


def test_rename_link(simple_robot: Robot):
    """Tests renaming a link and verifies all connections are updated."""
    simple_robot.rename_link("L2", "L2_renamed")

    # Check if link was renamed
    assert simple_robot.link("L2_renamed").name == "L2_renamed"
    with pytest.raises(ValueError):
        simple_robot.link("L2")

    # Check connections
    assert simple_robot.joint("J2").child_link_name == "L2_renamed"
    assert simple_robot.joint("J3").parent_link_name == "L2_renamed"
    l2_renamed_parent_joint = simple_robot.parent_joint("L2_renamed")
    assert l2_renamed_parent_joint is not None
    assert l2_renamed_parent_joint.name == "J2"
    assert simple_robot.child_joints("L2_renamed")[0].name == "J3"


def test_divide_link(simple_robot: Robot):
    """Tests dividing a link into two new links and a new joint."""
    simple_robot.divide_link(
        "L2",
        parent_link=Link.from_sphere("L2_parent", 1),
        child_link=Link.from_sphere("L2_child", 1),
        new_joint=Joint(
            "J_new",
            JointType.FLOATING,
            parent_link_name="L2_parent",
            child_link_name="L2_child",
        ),
    )

    # Old link should be gone
    with pytest.raises(ValueError):
        simple_robot.link("L2")

    # New components should exist
    assert simple_robot.link("L2_parent")
    assert simple_robot.link("L2_child")
    assert simple_robot.joint("J_new")

    # Check wiring
    assert simple_robot.joint("J2").child_link_name == "L2_parent"
    assert simple_robot.joint("J3").parent_link_name == "L2_child"
    l3_parent_joint = simple_robot.parent_joint("L3")
    assert l3_parent_joint is not None
    assert l3_parent_joint.parent_link_name == "L2_child"


def test_concatenate_robot(simple_robot: Robot):
    """Tests merging two robots together."""
    robot2 = Robot()
    robot2.add_link(Link.from_sphere("tool_base", 1))
    robot2.add_link(Link.from_sphere("tool_tip", 1))
    robot2.add_joint(
        Joint(
            "tool_joint",
            JointType.FLOATING,
            parent_link_name="tool_base",
            child_link_name="tool_tip",
        )
    )

    # Connect tool to the end of simple_robot's arm
    connecting_joint = Joint(
        "wrist", JointType.FLOATING, parent_link_name="L3", child_link_name="tool_base"
    )
    simple_robot.concatenate_robot(robot2, connecting_joint)

    assert len(simple_robot.links()) == 6
    assert simple_robot.joint("wrist")
    tool_base_parent_joint = simple_robot.parent_joint("tool_base")
    assert tool_base_parent_joint is not None
    assert tool_base_parent_joint.name == "wrist"
    assert simple_robot.child_joints("L3")[0].name == "wrist"
