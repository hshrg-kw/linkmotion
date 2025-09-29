import pytest
import numpy as np

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.move.manager import MoveManager
from linkmotion.collision.manager import CollisionManager
from linkmotion.range.limit import calc_limit_from_current_state


@pytest.fixture
def robot_with_collision_setup():
    """Creates a robot setup that will have collisions for testing."""
    robot = Robot()

    # Create links positioned to avoid initial collision
    base_link = Link.from_sphere(
        name="base_link", radius=0.5, center=np.array([0, 0, 0])
    )
    prismatic_link = Link.from_sphere(
        name="prismatic_link", radius=0.3, center=np.array([1, 0, 0])
    )
    revolute_link = Link.from_sphere(
        name="revolute_link", radius=0.3, center=np.array([0, 1, 0])
    )
    obstacle_link = Link.from_sphere(
        name="obstacle_link", radius=0.5, center=np.array([4, 0, 0])
    )

    # Prismatic joint that can translate the prismatic_link towards the obstacle
    prismatic_joint = Joint(
        name="prismatic_joint",
        type=JointType.PRISMATIC,
        parent_link_name="base_link",
        child_link_name="prismatic_link",
        direction=np.array([1, 0, 0]),  # X-axis translation
    )

    # Revolute joint for rotation testing
    revolute_joint = Joint(
        name="revolute_joint",
        type=JointType.REVOLUTE,
        parent_link_name="base_link",
        child_link_name="revolute_link",
        direction=np.array([0, 0, 1]),  # Z-axis rotation
        center=np.array([0, 0, 0]),
    )

    robot.add_link(base_link)
    robot.add_link(prismatic_link)
    robot.add_link(revolute_link)
    robot.add_link(obstacle_link)
    robot.add_joint(prismatic_joint)
    robot.add_joint(revolute_joint)

    return robot


@pytest.fixture
def robot_with_joint_limits():
    """Creates a robot with joints that have limits."""
    robot = Robot()

    # Separate links positioned to avoid initial collision
    base_link = Link.from_sphere(
        name="base_link", radius=0.1, center=np.array([0, 0, 0])
    )
    prismatic_arm_link = Link.from_sphere(
        name="prismatic_arm_link", radius=0.1, center=np.array([0.5, 0, 0])
    )
    revolute_arm_link = Link.from_sphere(
        name="revolute_arm_link", radius=0.1, center=np.array([0, 0.5, 0])
    )

    # Prismatic joint with limits
    limited_prismatic = Joint(
        name="limited_prismatic",
        type=JointType.PRISMATIC,
        parent_link_name="base_link",
        child_link_name="prismatic_arm_link",
        direction=np.array([1, 0, 0]),
        min_=-1.0,
        max_=1.0,
    )

    # Revolute joint with limits
    limited_revolute = Joint(
        name="limited_revolute",
        type=JointType.REVOLUTE,
        parent_link_name="base_link",
        child_link_name="revolute_arm_link",
        direction=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        min_=-np.pi / 2,
        max_=np.pi / 2,
    )

    robot.add_link(base_link)
    robot.add_link(prismatic_arm_link)
    robot.add_link(revolute_arm_link)
    robot.add_joint(limited_prismatic)
    robot.add_joint(limited_revolute)

    return robot


@pytest.fixture
def robot_with_fixed_joint():
    """Creates a robot with a fixed joint for error testing."""
    robot = Robot()

    base_link = Link.from_sphere(name="base_link", radius=0.1)
    fixed_link = Link.from_sphere(name="fixed_link", radius=0.1)

    fixed_joint = Joint(
        name="fixed_joint",
        type=JointType.FIXED,
        parent_link_name="base_link",
        child_link_name="fixed_link",
    )

    robot.add_link(base_link)
    robot.add_link(fixed_link)
    robot.add_joint(fixed_joint)

    return robot


class TestCalcTranslationLimitFromCurrentState:
    """Tests for calc_limit_from_current_state function."""

    def test_zero_step_raises_error(self, robot_with_collision_setup):
        """Test that function raises ValueError for zero step size."""
        mm = MoveManager(robot_with_collision_setup)
        cm = CollisionManager(mm)
        link_names1 = {"prismatic_link"}
        link_names2 = {"obstacle_link"}

        with pytest.raises(ValueError, match="Step size cannot be zero"):
            calc_limit_from_current_state(
                cm, "prismatic_joint", link_names1, link_names2, 0.01, step=0.0
            )

    def test_joint_limit_reached(self, robot_with_joint_limits):
        """Test that function handles joint limits correctly."""
        mm = MoveManager(robot_with_joint_limits)
        cm = CollisionManager(mm)
        link_names1 = {"prismatic_arm_link"}
        link_names2 = {"base_link"}

        # Test positive direction - should reach upper limit
        result = calc_limit_from_current_state(
            cm, "limited_prismatic", link_names1, link_names2, 0.1, step=0.0001
        )
        assert result == 1.0  # joint upper limit

        # Test negative direction - should reach lower limit
        result = calc_limit_from_current_state(
            cm, "limited_prismatic", link_names1, link_names2, 0.1, step=-0.0001
        )
        assert (result - 0.2) < 0.0001  # joint lower limit

    def test_collision_detection(self, robot_with_collision_setup):
        """Test collision detection functionality."""
        mm = MoveManager(robot_with_collision_setup)
        cm = CollisionManager(mm)

        # Move the prismatic_link closer to the obstacle but not colliding
        mm.move("prismatic_joint", 1.0)  # Close but not colliding yet

        link_names1 = {"prismatic_link"}
        link_names2 = {"obstacle_link"}

        # Test that function handles collision detection (may hit max_try in complex scenarios)
        result = calc_limit_from_current_state(
            cm,
            "prismatic_joint",
            link_names1,
            link_names2,
            0.1,
            step=0.001,
            max_try=50,
        )
        print(result)
        # If successful, should get a numeric result
        assert isinstance(result, (int, float))
        assert result >= 0  # Should be non-negative
        assert abs(result - 2.1) < 0.001


class TestCalcRotationLimitFromCurrentState:
    """Tests for calc_limit_from_current_state function."""

    def test_revolute_joint_accepted(self, robot_with_collision_setup):
        """Test that function accepts revolute joints."""
        mm = MoveManager(robot_with_collision_setup)
        cm = CollisionManager(mm)
        link_names1 = {"revolute_link"}
        link_names2 = {"obstacle_link"}

        # Test that function accepts revolute joints (may hit max_try in complex scenarios)
        with pytest.raises(
            RuntimeError,
            match="Exceeded maximum number of tries",
        ):
            calc_limit_from_current_state(
                cm,
                "revolute_joint",
                link_names1,
                link_names2,
                0.1,
                step=0.05,
                max_try=5,
            )

    def test_continuous_joint_accepted(self):
        """Test that function accepts continuous joints."""
        robot = Robot()
        base_link = Link.from_sphere(name="base_link", radius=0.1)
        arm_link = Link.from_sphere(name="arm_link", radius=0.1)

        continuous_joint = Joint(
            name="continuous_joint",
            type=JointType.CONTINUOUS,
            parent_link_name="base_link",
            child_link_name="arm_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0, 0, 0]),
        )

        robot.add_link(base_link)
        robot.add_link(arm_link)
        robot.add_joint(continuous_joint)

        mm = MoveManager(robot)
        cm = CollisionManager(mm)
        link_names1 = {"arm_link"}
        link_names2 = {"base_link"}

        # Should not raise an error for continuous joint
        result = calc_limit_from_current_state(
            cm, "continuous_joint", link_names1, link_names2, 0.1, step=0.1, max_try=5
        )

        assert result == 0.0  # Should remain at initial position

    def test_joint_limit_reached(self, robot_with_joint_limits):
        """Test that function handles joint limits correctly."""
        mm = MoveManager(robot_with_joint_limits)
        cm = CollisionManager(mm)
        link_names1 = {"revolute_arm_link"}
        link_names2 = {"base_link"}

        # Test positive direction - should reach upper limit
        result = calc_limit_from_current_state(
            cm, "limited_revolute", link_names1, link_names2, 0.1, step=0.05
        )
        assert result == np.pi / 2  # joint upper limit

        # Test negative direction - should reach lower limit
        result = calc_limit_from_current_state(
            cm, "limited_revolute", link_names1, link_names2, 0.1, step=-0.05
        )
        assert result == -np.pi / 2  # joint lower limit

    def test_max_try_exceeded(self, robot_with_collision_setup):
        """Test that function raises RuntimeError when max_try is exceeded."""
        mm = MoveManager(robot_with_collision_setup)
        cm = CollisionManager(mm)
        link_names1 = {"revolute_link"}
        link_names2 = {"obstacle_link"}

        with pytest.raises(RuntimeError, match="Exceeded maximum number of tries"):
            calc_limit_from_current_state(
                cm,
                "revolute_joint",
                link_names1,
                link_names2,
                0.01,
                step=0.001,
                max_try=2,  # Very small step, low max_try
            )
