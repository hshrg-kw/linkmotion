import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.move.manager import MoveManager, JointLimitError
from linkmotion.transform import Transform


@pytest.fixture
def concrete_robot_and_manager() -> tuple[MoveManager, Robot]:
    """Provides a MoveManager instance with a real, constructed Robot."""
    # 1. Define links
    base_link = Link.from_sphere(name="base_link", radius=0.1)
    arm_link_1 = Link.from_sphere(name="arm_link_1", radius=0.1)
    arm_link_2 = Link.from_sphere(name="arm_link_2", radius=0.1)
    fixed_link = Link.from_sphere(name="fixed_link", radius=0.1)

    # 2. Define joints
    joint1 = Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link_name="base_link",
        child_link_name="arm_link_1",
        direction=np.array([0, 0, 1]),  # Z-axis rotation
        center=np.array([0, 1, 0]),  # Rotate around a point
    )
    joint2 = Joint(
        name="joint2",
        type=JointType.PRISMATIC,
        parent_link_name="arm_link_1",
        child_link_name="arm_link_2",
        direction=np.array([1, 0, 0]),  # X-axis translation
    )
    fixed_joint = Joint(
        name="fixed_joint",
        type=JointType.FIXED,
        parent_link_name="base_link",
        child_link_name="fixed_link",
    )

    # 3. Build the robot
    robot = Robot()
    robot.add_link(base_link)
    robot.add_link(arm_link_1)
    robot.add_link(arm_link_2)
    robot.add_link(fixed_link)

    robot.add_joint(joint1)
    robot.add_joint(joint2)
    robot.add_joint(fixed_joint)

    # 4. Initialize MoveManager with the real robot
    manager = MoveManager(robot)
    return manager, robot


# ===================================================================
# 3. Test Functions
# ===================================================================


def test_initialization(concrete_robot_and_manager):
    """Tests if the MoveManager initializes correctly with a real robot."""
    manager, robot = concrete_robot_and_manager
    assert manager.robot == robot
    assert len(manager.link_name_to_id) == 4
    assert (
        manager.link_name_to_id["base_link"] == 0
    )  # Order dependent, but usually consistent

    # Check that initially, all transforms are identity
    initial_transform = manager.get_transform("arm_link_1")
    assert initial_transform == Transform()


def test_move_revolute_joint(concrete_robot_and_manager):
    """Tests moving a revolute joint and checks the resulting world transform."""
    manager, _ = concrete_robot_and_manager

    # Move joint1 by 90 degrees (pi/2 radians) around the Z-axis
    manager.move("joint1", np.pi / 2)

    # Get the transform of the child link
    actual_transform = manager.get_transform("arm_link_1")

    # Calculate the expected transform manually: T * R * T^-1
    center = np.array([0, 1, 0])
    rotation = R.from_rotvec([0, 0, np.pi / 2])

    t_inv = Transform(translate=-center)
    rot = Transform(rotate=rotation)
    t = Transform(translate=center)
    expected_transform = t.apply(rot.apply(t_inv))

    assert actual_transform == expected_transform


def test_move_prismatic_joint(concrete_robot_and_manager):
    """Tests moving a prismatic joint."""
    manager, _ = concrete_robot_and_manager

    # Move joint2 by 0.5 units along the X-axis
    manager.move("joint2", 0.5)

    actual_transform = manager.get_transform("arm_link_2")

    # Expected is a simple translation, as the parent is at origin
    expected_transform = Transform(translate=np.array([0.5, 0, 0]))

    assert actual_transform == expected_transform


def test_chained_moves(concrete_robot_and_manager):
    """Tests the effect of one joint's movement on its children."""
    manager, _ = concrete_robot_and_manager

    # 1. Move the first joint
    manager.move("joint1", np.pi / 2)

    # 2. Move the second joint
    manager.move("joint2", 0.5)

    # 3. Get the transform of the final link
    final_transform = manager.get_transform("arm_link_2")

    # 4. Calculate expected result
    # The final transform should be T_base_to_arm1 * T_arm1_to_arm2
    transform1 = manager.get_transform("arm_link_1")
    local_transform2 = Transform(translate=np.array([0.5, 0, 0]))
    expected_transform = transform1.apply(local_transform2)

    assert final_transform == expected_transform


def test_get_center_and_direction_after_move(concrete_robot_and_manager):
    """Tests if joint center and direction are correctly transformed."""
    manager, robot = concrete_robot_and_manager

    # Move the parent link of joint2
    manager.move("joint1", np.pi / 2)

    # Get the world transform of the parent link
    parent_transform = manager.get_transform("arm_link_1")
    joint2 = robot.joint("joint2")

    # Get center and direction and check if they are correctly transformed
    world_center = manager.get_center(
        "joint2"
    )  # joint2 has no center, this should be None
    world_direction = manager.get_direction("joint2")

    parent_rotation_transform = Transform(rotate=parent_transform.rotation)
    expected_direction = parent_rotation_transform.apply(joint2.direction)

    assert world_center is None
    assert np.allclose(world_direction, expected_direction)


def test_reset_move(concrete_robot_and_manager):
    """Tests if reset_move correctly resets all transforms to identity."""
    manager, _ = concrete_robot_and_manager

    # Move a joint and verify the transform is not identity
    manager.move("joint1", np.pi)
    moved_transform = manager.get_transform("arm_link_1")
    assert moved_transform != Transform()

    # Reset and verify it's back to identity
    manager.reset_move()
    reset_transform = manager.get_transform("arm_link_1")
    assert reset_transform == Transform()


def test_error_handling(concrete_robot_and_manager):
    """Tests various error conditions."""
    manager, _ = concrete_robot_and_manager

    # 1. Attempt to move a fixed joint
    with pytest.raises(ValueError, match="Cannot move fixed joint: fixed_joint"):
        manager.move("fixed_joint", 1.0)

    # 2. Attempt to move a non-existent joint
    with pytest.raises(
        ValueError, match="Joint 'nonexistent' not found in the robot model."
    ):
        manager.move("nonexistent", 1.0)

    # 3. Provide wrong value type for a revolute joint
    with pytest.raises(ValueError, match="Value for revolute joint must be a float"):
        manager.move("joint1", Transform())

    # 4. Request transform for a non-existent link
    with pytest.raises(ValueError, match="Link 'nonexistent_link' not found"):
        manager.get_transform("nonexistent_link")


@pytest.fixture
def robot_with_joint_limits() -> tuple[MoveManager, Robot]:
    """Provides a MoveManager with joints that have defined limits."""
    # Define links
    base_link = Link.from_sphere(name="base_link", radius=0.1)
    revolute_arm_link = Link.from_sphere(name="revolute_arm_link", radius=0.1)
    prismatic_arm_link = Link.from_sphere(name="prismatic_arm_link", radius=0.1)

    # Define joints with limits
    revolute_joint = Joint(
        name="revolute_joint",
        type=JointType.REVOLUTE,
        parent_link_name="base_link",
        child_link_name="revolute_arm_link",
        direction=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),  # Center required for revolute joints
        min_=-np.pi,  # -180 degrees
        max_=np.pi,  # +180 degrees
    )

    prismatic_joint = Joint(
        name="prismatic_joint",
        type=JointType.PRISMATIC,
        parent_link_name="base_link",
        child_link_name="prismatic_arm_link",
        direction=np.array([1, 0, 0]),
        min_=-1.0,  # -1 meter
        max_=2.0,  # +2 meters
    )

    # Build robot
    robot = Robot()
    robot.add_link(base_link)
    robot.add_link(revolute_arm_link)
    robot.add_link(prismatic_arm_link)
    robot.add_joint(revolute_joint)
    robot.add_joint(prismatic_joint)

    manager = MoveManager(robot)
    return manager, robot


def test_joint_limit_error_revolute(robot_with_joint_limits):
    """Tests JointLimitError for revolute joint limits."""
    manager, robot = robot_with_joint_limits
    joint = robot.joint("revolute_joint")

    # Test exceeding maximum limit
    with pytest.raises(JointLimitError) as exc_info:
        manager.move("revolute_joint", 4.0)  # > pi

    error = exc_info.value
    assert error.value == 4.0
    assert error.joint == joint
    assert "4.0" in str(error)
    assert "revolute_joint" in str(error)
    assert f"[{joint.min}, {joint.max}]" in str(error)

    # Test exceeding minimum limit
    with pytest.raises(JointLimitError) as exc_info:
        manager.move("revolute_joint", -4.0)  # < -pi

    error = exc_info.value
    assert error.value == -4.0
    assert error.joint == joint


def test_joint_limit_error_prismatic(robot_with_joint_limits):
    """Tests JointLimitError for prismatic joint limits."""
    manager, robot = robot_with_joint_limits
    joint = robot.joint("prismatic_joint")

    # Test exceeding maximum limit
    with pytest.raises(JointLimitError) as exc_info:
        manager.move("prismatic_joint", 3.0)  # > 2.0

    error = exc_info.value
    assert error.value == 3.0
    assert error.joint == joint
    assert "prismatic_joint" in str(error)

    # Test exceeding minimum limit
    with pytest.raises(JointLimitError) as exc_info:
        manager.move("prismatic_joint", -2.0)  # < -1.0

    error = exc_info.value
    assert error.value == -2.0
    assert error.joint == joint


def test_joint_limit_valid_moves(robot_with_joint_limits):
    """Tests that valid moves within limits work correctly."""
    manager, _ = robot_with_joint_limits

    # Test valid revolute joint move
    manager.move("revolute_joint", np.pi / 2)  # Within [-pi, pi]
    assert manager.joint_value("revolute_joint") == np.pi / 2

    # Test valid prismatic joint move
    manager.move("prismatic_joint", 1.5)  # Within [-1.0, 2.0]
    assert manager.joint_value("prismatic_joint") == 1.5


def test_joint_value_method(concrete_robot_and_manager):
    """Tests the joint_value method functionality."""
    manager, _ = concrete_robot_and_manager

    # Test getting value of unset joint
    assert manager.joint_value("joint1") == 0

    # Move joint and test getting its value
    test_value = np.pi / 4
    manager.move("joint1", test_value)
    assert manager.joint_value("joint1") == test_value

    # Move another joint and test both values
    test_value2 = 0.75
    manager.move("joint2", test_value2)
    assert manager.joint_value("joint1") == test_value
    assert manager.joint_value("joint2") == test_value2

    # Test that reset clears joint values
    manager.reset_move()
    assert manager.joint_value("joint1") == 0
    assert manager.joint_value("joint2") == 0


def test_joint_value_nonexistent_joint(concrete_robot_and_manager):
    """Tests joint_value method with nonexistent joint."""
    manager, _ = concrete_robot_and_manager

    with pytest.raises(
        ValueError, match="Joint 'nonexistent' not found in the robot model."
    ):
        manager.joint_value("nonexistent")


def test_copy_method(concrete_robot_and_manager):
    """Tests the copy method functionality."""
    manager, robot = concrete_robot_and_manager

    # Move some joints
    manager.move("joint1", np.pi / 3)
    manager.move("joint2", 1.2)

    # Create copy
    manager_copy = manager.copy()

    # Test that copy has same robot reference
    assert manager_copy.robot is robot

    # Test that transforms are copied correctly
    original_transform = manager.get_transform("arm_link_1")
    copied_transform = manager_copy.get_transform("arm_link_1")
    assert original_transform == copied_transform

    # Test that joint values are preserved
    assert manager_copy.joint_value("joint1") == np.pi / 3
    assert manager_copy.joint_value("joint2") == 1.2

    # Test independence: modify original and check copy is unaffected
    manager.move("joint1", np.pi / 2)
    assert manager.joint_value("joint1") == np.pi / 2
    assert manager_copy.joint_value("joint1") == np.pi / 3  # Should remain unchanged

    # Test independence: modify copy and check original is unaffected
    manager_copy.move("joint2", 2.0)
    assert manager.joint_value("joint2") == 1.2  # Should remain unchanged
    assert manager_copy.joint_value("joint2") == 2.0
