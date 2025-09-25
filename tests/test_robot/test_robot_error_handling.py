import pytest
import numpy as np

from linkmotion.robot import Robot, Link, Joint, JointType


@pytest.fixture
def link1():
    return Link.from_sphere("link1", 1)


@pytest.fixture
def link2():
    return Link.from_sphere("link2", 1)


@pytest.fixture
def revolute_joint(link1: Link, link2: Link):
    return Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link_name=link1.name,
        child_link_name=link2.name,
        center=np.array([0.0, 0.0, 0.0]),
    )


@pytest.fixture
def robot_chain(link1: Link, link2: Link, revolute_joint: Joint):
    """A simple robot with link1 -> joint1 -> link2."""
    robot = Robot()
    robot.add_link(link1)
    robot.add_link(link2)
    robot.add_joint(revolute_joint)
    return robot


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


class TestRobotErrorHandling:
    """Tests for Robot class error handling and edge cases."""

    def test_str_representation(self):
        robot = Robot()
        str_repr = str(robot)
        assert "Robot model" in str_repr
        assert "0 links" in str_repr
        assert "0 joints" in str_repr

        robot.add_link(Link.from_sphere("test", 1))
        str_repr = str(robot)
        assert "1 links" in str_repr

    def test_add_duplicate_joint_fails(self, robot_chain: Robot):
        duplicate_joint = Joint(
            "joint1",  # Same name as existing joint
            JointType.FIXED,
            "link2",
            "link1",
        )
        with pytest.raises(ValueError, match="already exists"):
            robot_chain.add_joint(duplicate_joint)

    def test_add_joint_child_already_has_parent(self, robot_chain: Robot):
        # link2 already has joint1 as parent
        another_joint = Joint(
            "another_joint",
            JointType.FIXED,
            "link2",  # link2 already has a parent
            "link1",
        )
        with pytest.raises(ValueError, match="already has a parent joint"):
            robot_chain.add_joint(another_joint)

    def test_rename_to_existing_name_fails(self, robot_chain: Robot):
        with pytest.raises(ValueError, match="already exists"):
            robot_chain.rename_link("link1", "link2")

    def test_rename_same_name_is_noop(self, robot_chain: Robot):
        # Should not raise error
        robot_chain.rename_link("link1", "link1")
        assert robot_chain.link("link1").name == "link1"

    def test_divide_link_validation_errors(self, robot_chain: Robot):
        parent_link = Link.from_sphere("new_parent", 1)
        child_link = Link.from_sphere("new_child", 1)

        # Test conflicting parent link name
        with pytest.raises(ValueError, match="conflicts with an existing link"):
            robot_chain.divide_link(
                "link1",
                Link.from_sphere("link2", 1),  # link2 already exists
                child_link,
                Joint("new_joint", JointType.FIXED, "new_child", "link2"),
            )

        # Test joint doesn't connect new links properly
        bad_joint = Joint("bad_joint", JointType.FIXED, "wrong_child", "new_parent")
        with pytest.raises(ValueError, match="must connect the new parent and child"):
            robot_chain.divide_link("link1", parent_link, child_link, bad_joint)

    def test_concatenate_robot_validation_errors(self, simple_robot: Robot):
        # Create robot with conflicting names
        conflicting_robot = Robot()
        conflicting_robot.add_link(
            Link.from_sphere("L0", 1)
        )  # L0 already exists in simple_robot

        with pytest.raises(ValueError, match="Link name conflict"):
            simple_robot.concatenate_robot(
                conflicting_robot, Joint("connect", JointType.FIXED, "L0", "L3")
            )

        # Test robot with multiple roots
        multi_root_robot = Robot()
        multi_root_robot.add_link(Link.from_sphere("root1", 1))
        multi_root_robot.add_link(Link.from_sphere("root2", 1))

        with pytest.raises(ValueError, match="exactly one root link"):
            simple_robot.concatenate_robot(
                multi_root_robot, Joint("connect", JointType.FIXED, "root1", "L3")
            )

    def test_traverse_child_links_edge_cases(self, simple_robot: Robot):
        # Test with leaf node
        children = list(simple_robot.traverse_child_links("L3"))
        assert len(children) == 0

        # Test include_self with leaf
        children_with_self = list(
            simple_robot.traverse_child_links("L3", include_self=True)
        )
        assert len(children_with_self) == 1
        assert children_with_self[0].name == "L3"

    def test_traverse_parent_links_edge_cases(self, simple_robot: Robot):
        # Test with root node
        parents = list(simple_robot.traverse_parent_links("L0"))
        assert len(parents) == 0

        # Test include_self with root
        parents_with_self = list(
            simple_robot.traverse_parent_links("L0", include_self=True)
        )
        assert len(parents_with_self) == 1
        assert parents_with_self[0].name == "L0"
