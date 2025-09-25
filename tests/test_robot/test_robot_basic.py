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
def link3():
    return Link.from_sphere("link3", 1)


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
def fixed_joint(link2: Link, link3: Link):
    return Joint(
        name="joint2_fixed",
        type=JointType.FIXED,
        parent_link_name=link2.name,
        child_link_name=link3.name,
    )


@pytest.fixture
def robot_chain(link1: Link, link2: Link, revolute_joint: Joint):
    """A simple robot with link1 -> joint1 -> link2."""
    robot = Robot()
    robot.add_link(link1)
    robot.add_link(link2)
    robot.add_joint(revolute_joint)
    return robot


class TestRobot:
    def test_add_link_and_joint(self, robot_chain: Robot, link1: Link, link2: Link):
        assert robot_chain.link("link1") == link1
        assert robot_chain.joint("joint1").parent_link_name == "link1"
        assert len(robot_chain.links()) == 2

    def test_add_duplicate_link_fails(self, link1: Link):
        robot = Robot()
        robot.add_link(link1)
        with pytest.raises(ValueError, match="already exists"):
            robot.add_link(Link.from_sphere("link1", 10))

    def test_add_joint_with_missing_link_fails(
        self, link1: Link, revolute_joint: Joint
    ):
        robot = Robot()
        robot.add_link(link1)  # Add parent, but not child
        with pytest.raises(
            ValueError, match="Child link 'link2' for joint 'joint1' not found"
        ):
            robot.add_joint(revolute_joint)

    def test_get_missing_link_fails(self):
        robot = Robot()
        with pytest.raises(ValueError, match="not found"):
            robot.link("non_existent_link")

    def test_root_and_leaf_links(self, robot_chain: Robot, link1: Link, link2: Link):
        assert robot_chain.root_links() == [link1]
        assert robot_chain.leaf_links() == [link2]

    def test_parent_child_joints(
        self, robot_chain: Robot, link1: Link, link2: Link, revolute_joint: Joint
    ):
        assert robot_chain.parent_joint(link2.name) == revolute_joint
        assert robot_chain.parent_joint(link1.name) is None
        assert robot_chain.child_joints(link1.name) == [revolute_joint]
        assert robot_chain.child_joints(link2.name) == []

    def test_static_and_dynamic_links(
        self,
        robot_chain: Robot,
        link1: Link,
        link2: Link,
        link3: Link,
        fixed_joint: Joint,
    ):
        # In the simple chain, link1 is root (static), joint is revolute, so link2 is dynamic
        assert robot_chain.static_links() == [link1]
        assert robot_chain.dynamic_links() == [link2]

        # Add a fixed joint and a third link
        robot_chain.add_link(link3)
        robot_chain.add_joint(fixed_joint)  # link2 -> fixed -> link3

        # link1 (root) is static.
        # link2 is connected by a revolute joint, so it's dynamic.
        # link3 is connected to a DYNAMIC link2, even with a fixed joint, it's still part of the moving chain.
        # Thus, link3 is also dynamic.
        static_links = robot_chain.static_links()
        dynamic_links = robot_chain.dynamic_links()

        assert len(static_links) == 1
        assert static_links[0].name == "link1"

        dynamic_names = {link.name for link in dynamic_links}
        assert dynamic_names == {"link2", "link3"}

    def test_from_other_creates_deep_copy(self, robot_chain: Robot):
        robot_copy = Robot.from_other(robot_chain)

        # Check if they are different objects
        assert robot_copy is not robot_chain
        assert robot_copy._link_dict is not robot_chain._link_dict

        # Modify the original and check if the copy is unaffected
        robot_chain.add_link(Link.from_sphere("new_link", 10))
        assert len(robot_chain.links()) == 3
        assert len(robot_copy.links()) == 2
