import pytest
import numpy as np

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


@pytest.fixture
def branched_robot():
    """
    Provides a robot with a branching structure:
    base -> L1 -> J2 -> L2 (branch 1)
             |
             -> J3 -> L3 (branch 2)
    """
    robot = Robot()
    robot.add_link(Link.from_sphere("base", 1))
    robot.add_link(Link.from_sphere("L1", 1))
    robot.add_link(Link.from_sphere("L2", 1))
    robot.add_link(Link.from_sphere("L3", 1))
    robot.add_joint(
        Joint("J1", JointType.FLOATING, parent_link_name="base", child_link_name="L1")
    )
    robot.add_joint(
        Joint("J2", JointType.FLOATING, parent_link_name="L1", child_link_name="L2")
    )
    robot.add_joint(
        Joint("J3", JointType.FLOATING, parent_link_name="L1", child_link_name="L3")
    )
    return robot


def test_robot_initialization():
    """Tests that a new Robot instance is empty."""
    robot = Robot()
    assert len(robot.links()) == 0
    assert not robot.root_links()
    assert not robot.leaf_links()


def test_add_link_and_joint(simple_robot: Robot):
    """Tests adding links and joints and basic properties."""
    assert len(simple_robot.links()) == 4
    assert len(simple_robot._joint_dict) == 3
    assert simple_robot.link("L1").name == "L1"
    assert simple_robot.joint("J2").parent_link_name == "L1"


def test_add_duplicate_link_raises_error():
    """Tests that adding a link with a duplicate name raises ValueError."""
    robot = Robot()
    robot.add_link(Link.from_sphere("L1", 1))
    with pytest.raises(ValueError, match="already exists"):
        robot.add_link(Link.from_sphere("L1", 1))


def test_add_joint_with_nonexistent_link_raises_error():
    """Tests that adding a joint with a missing link raises ValueError."""
    robot = Robot()
    robot.add_link(Link.from_sphere("L1", 1))
    with pytest.raises(ValueError, match="not found"):
        robot.add_joint(
            Joint(
                "J1",
                JointType.FLOATING,
                parent_link_name="L1",
                child_link_name="L2_nonexistent",
            )
        )


def test_get_nonexistent_link_or_joint_raises_error():
    """Tests that retrieving a non-existent component raises ValueError."""
    robot = Robot()
    with pytest.raises(ValueError, match="not found"):
        robot.link("nonexistent")
    with pytest.raises(ValueError, match="not found"):
        robot.joint("nonexistent")


def test_root_and_leaf_links(simple_robot: Robot, branched_robot: Robot):
    """Tests the identification of root and leaf links."""
    assert len(simple_robot.root_links()) == 1
    assert simple_robot.root_links()[0].name == "L0"
    assert len(simple_robot.leaf_links()) == 1
    assert simple_robot.leaf_links()[0].name == "L3"

    assert branched_robot.root_links()[0].name == "base"
    leaf_names = {link.name for link in branched_robot.leaf_links()}
    assert leaf_names == {"L2", "L3"}


def test_parent_and_child_queries(simple_robot: Robot):
    """Tests parent_joint and child_joints methods."""
    assert simple_robot.parent_joint("L0") is None
    l2_parent_joint = simple_robot.parent_joint("L2")
    assert l2_parent_joint is not None
    assert l2_parent_joint.name == "J2"

    child_joints_L1 = simple_robot.child_joints("L1")
    assert len(child_joints_L1) == 1
    assert child_joints_L1[0].name == "J2"

    assert not simple_robot.child_joints("L3")  # L3 is a leaf


def test_traverse_child_links(branched_robot: Robot):
    """Tests the BFS traversal of descendant links."""
    # Note: BFS order depends on hash order of joints in the set, so we check sets.
    children = list(branched_robot.traverse_child_links("L1"))
    child_names = {link.name for link in children}
    assert child_names == {"L2", "L3"}

    descendants = list(branched_robot.traverse_child_links("base", include_self=True))
    descendant_names = {link.name for link in descendants}
    assert descendant_names == {"base", "L1", "L2", "L3"}


def test_traverse_parent_links(simple_robot: Robot):
    """Tests the traversal up the kinematic chain towards the root."""
    ancestors = list(simple_robot.traverse_parent_links("L3"))
    ancestor_names = [link.name for link in ancestors]
    assert ancestor_names == ["L2", "L1", "L0"]

    ancestors_with_self = list(
        simple_robot.traverse_parent_links("L3", include_self=True)
    )
    ancestor_names_with_self = [link.name for link in ancestors_with_self]
    assert ancestor_names_with_self == ["L3", "L2", "L1", "L0"]


def test_traverse_links_for_forest():
    """Tests traversal over a robot with multiple disconnected trees (a forest)."""
    robot = Robot()
    robot.add_link(Link.from_sphere("R1_L1", 1))
    robot.add_link(Link.from_sphere("R1_L2", 1))
    robot.add_joint(
        Joint(
            "R1_J1",
            JointType.FLOATING,
            parent_link_name="R1_L1",
            child_link_name="R1_L2",
        )
    )

    robot.add_link(Link.from_sphere("R2_L1", 1))
    robot.add_link(Link.from_sphere("R2_L2", 1))
    robot.add_joint(
        Joint(
            "R2_J1",
            JointType.FLOATING,
            parent_link_name="R2_L1",
            child_link_name="R2_L2",
        )
    )

    all_links = list(robot.traverse_links())
    assert len(all_links) == 4
    all_link_names = {link.name for link in all_links}
    assert all_link_names == {"R1_L1", "R1_L2", "R2_L1", "R2_L2"}


def test_static_and_dynamic_links():
    """Tests the classification of links into static and dynamic."""
    robot = Robot()
    robot.add_link(Link.from_sphere("base", 1))
    robot.add_link(Link.from_sphere("arm_base", 1))
    robot.add_link(Link.from_sphere("arm_link1", 1))
    robot.add_link(Link.from_sphere("gripper", 1))

    robot.add_joint(
        Joint(
            "J_base",
            JointType.FIXED,
            parent_link_name="base",
            child_link_name="arm_base",
        )
    )
    robot.add_joint(
        Joint(
            "J_arm",
            JointType.REVOLUTE,
            parent_link_name="arm_base",
            child_link_name="arm_link1",
            center=np.array([0.0, 0.0, 0.0]),
        )
    )
    robot.add_joint(
        Joint(
            "J_grip",
            JointType.FIXED,
            parent_link_name="arm_link1",
            child_link_name="gripper",
        )
    )

    static_names = {link.name for link in robot.static_links()}
    dynamic_names = {link.name for link in robot.dynamic_links()}

    assert static_names == {"base", "arm_base"}
    assert dynamic_names == {"arm_link1", "gripper"}
