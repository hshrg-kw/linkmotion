import pytest
import numpy as np

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.range.range_cal import RangeCalculator
from linkmotion import Transform


@pytest.fixture
def simple_robot():
    """Creates a simple robot with two joints for testing."""
    robot = Robot()

    # Create links positioned to avoid initial collision
    base_link = Link.from_sphere(
        name="base_link", radius=0.1, center=np.array([0, 0, 0])
    )
    arm_link = Link.from_cylinder(
        name="arm_link",
        radius=0.1,
        height=1.0,
        default_transform=Transform(translate=np.array([0, 0, 0.5])),
    )
    hand_link = Link.from_box(
        name="hand_link",
        extents=np.array([0.1, 0.1, 0.1]),
        default_transform=Transform(translate=np.array([0, 0, 1.0])),
    )
    obstacle_link = Link.from_box(
        name="obstacle_link",
        extents=np.array([30, 30, 0.1]),
        default_transform=Transform(translate=np.array([0, 0, 1.5])),
    )

    # Prismatic joint with limits
    joint1 = Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link_name="base_link",
        child_link_name="arm_link",
        direction=np.array([1, 0, 0]),
        center=np.array([0, 0, 0.0]),
        min_=-np.pi / 2,
        max_=np.pi / 2,
    )

    # Revolute joint with limits
    joint2 = Joint(
        name="joint2",
        type=JointType.PRISMATIC,
        parent_link_name="arm_link",
        child_link_name="hand_link",
        direction=np.array([0, 1, 0]),
        center=np.array([0, 0, 1.0]),
        min_=-10,
        max_=10,
    )

    robot.add_link(base_link)
    robot.add_link(arm_link)
    robot.add_link(hand_link)
    robot.add_link(obstacle_link)
    robot.add_joint(joint1)
    robot.add_joint(joint2)

    return robot


def test_range_calculation(simple_robot: Robot):
    cal = RangeCalculator(simple_robot, {"hand_link"}, {"obstacle_link"})
    cal.add_axis("joint1", np.linspace(-np.pi / 2, np.pi / 2, 10))
    cal.add_axis("joint2", np.linspace(-10, 10, 20))
    cal.execute()
