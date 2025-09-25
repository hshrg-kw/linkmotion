import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.move.manager import MoveManager
from linkmotion.transform import Transform


class TestMoveManagerEdgeCases:
    """Edge case tests for MoveManager to achieve maximum coverage."""

    @pytest.fixture
    def simple_robot_and_manager(self) -> tuple[MoveManager, Robot]:
        """Simple robot fixture for edge case testing."""
        base_link = Link.from_sphere(name="base_link", radius=0.1)
        arm_link = Link.from_sphere(name="arm_link", radius=0.1)

        prismatic_joint = Joint(
            name="prismatic_joint",
            type=JointType.PRISMATIC,
            parent_link_name="base_link",
            child_link_name="arm_link",
            direction=np.array([1, 0, 0]),
        )

        robot = Robot()
        robot.add_link(base_link)
        robot.add_link(arm_link)
        robot.add_joint(prismatic_joint)

        manager = MoveManager(robot)
        return manager, robot

    def test_move_joint_with_missing_child_link(self, simple_robot_and_manager):
        """Tests moving a joint when its child link is missing."""
        manager, robot = simple_robot_and_manager

        # Create a joint with a child link that doesn't exist in our link mapping
        ghost_joint = Joint(
            name="ghost_joint",
            type=JointType.REVOLUTE,
            parent_link_name="base_link",
            child_link_name="ghost_link",  # This link doesn't exist
            direction=np.array([0, 0, 1]),
            center=np.array([0, 0, 0]),  # Required for revolute joints
        )

        # Add the joint to the robot but not the link
        robot._joint_dict["ghost_joint"] = ghost_joint

        # This will raise an error when trying to get the child link
        with pytest.raises(
            ValueError, match="Link 'ghost_link' not found in the robot model"
        ):
            manager.move("ghost_joint", 1.0)

    def test_prismatic_joint_with_wrong_value_type(self, simple_robot_and_manager):
        """Tests prismatic joint with wrong value type."""
        manager, _ = simple_robot_and_manager

        with pytest.raises(
            ValueError, match="Value for prismatic joint must be a float"
        ):
            manager.move("prismatic_joint", Transform())

        with pytest.raises(
            ValueError, match="Value for prismatic joint must be a float"
        ):
            manager.move("prismatic_joint", "invalid")

    def test_revolute_joint_without_center_error_path(self):
        """Tests the specific error path for revolute joints without center."""
        base_link = Link.from_sphere(name="base_link", radius=0.1)
        arm_link = Link.from_sphere(name="arm_link", radius=0.1)

        robot = Robot()
        robot.add_link(base_link)
        robot.add_link(arm_link)

        # Create revolute joint with center first, then manually set it to None
        revolute_joint = Joint(
            name="revolute_joint",
            type=JointType.REVOLUTE,
            parent_link_name="base_link",
            child_link_name="arm_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0, 0, 0]),  # Required for creation
        )

        robot.add_joint(revolute_joint)

        # Manually set center to None to test the error path
        revolute_joint.center = None

        manager = MoveManager(robot)

        with pytest.raises(
            ValueError, match="does not have a defined center for rotation"
        ):
            manager.move("revolute_joint", 1.0)

    def test_logging_during_initialization(self, caplog):
        """Tests that proper log messages are generated during initialization."""
        base_link = Link.from_sphere(name="base_link", radius=0.1)
        arm_link = Link.from_sphere(name="arm_link", radius=0.1)

        joint = Joint(
            name="test_joint",
            type=JointType.REVOLUTE,
            parent_link_name="base_link",
            child_link_name="arm_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0, 0, 0]),
        )

        robot = Robot()
        robot.add_link(base_link)
        robot.add_link(arm_link)
        robot.add_joint(joint)

        with caplog.at_level("INFO"):
            MoveManager(robot)

        assert "MoveManager initialized for robot with 2 links" in caplog.text

    def test_logging_during_move_operations(self, simple_robot_and_manager, caplog):
        """Tests logging during move operations."""
        manager, _ = simple_robot_and_manager

        with caplog.at_level("DEBUG"):
            manager.move("prismatic_joint", 1.5)

        assert (
            "Moving joint 'prismatic_joint' of type 'JointType.PRISMATIC' to value: 1.5"
            in caplog.text
        )

    def test_logging_during_reset(self, simple_robot_and_manager, caplog):
        """Tests logging during reset operations."""
        manager, _ = simple_robot_and_manager

        with caplog.at_level("INFO"):
            manager.reset_move()

        assert "Resetting all joint transforms" in caplog.text

    def test_logging_warning_for_fixed_joint_attempt(self, caplog):
        """Tests warning log when attempting to move fixed joint."""
        base_link = Link.from_sphere(name="base_link", radius=0.1)
        fixed_link = Link.from_sphere(name="fixed_link", radius=0.1)

        fixed_joint = Joint(
            name="fixed_joint",
            type=JointType.FIXED,
            parent_link_name="base_link",
            child_link_name="fixed_link",
        )

        robot = Robot()
        robot.add_link(base_link)
        robot.add_link(fixed_link)
        robot.add_joint(fixed_joint)

        manager = MoveManager(robot)

        with caplog.at_level("WARNING"):
            with pytest.raises(ValueError):
                manager.move("fixed_joint", 1.0)

        assert "Attempted to move a fixed joint: 'fixed_joint'" in caplog.text

    def test_error_logging_for_joint_not_found_in_move(
        self, simple_robot_and_manager, caplog
    ):
        """Tests error logging when joint is not found during move."""
        manager, _ = simple_robot_and_manager

        with caplog.at_level("ERROR"):
            with pytest.raises(
                ValueError,
                match="Joint 'nonexistent_joint' not found in the robot model",
            ):
                manager.move("nonexistent_joint", 1.0)

        assert "Joint 'nonexistent_joint' not found in the robot model" in caplog.text

    def test_error_logging_for_link_not_found_in_get_transform(
        self, simple_robot_and_manager, caplog
    ):
        """Tests error logging when link is not found in get_transform."""
        manager, _ = simple_robot_and_manager

        with caplog.at_level("ERROR"):
            with pytest.raises(ValueError):
                manager.get_transform("nonexistent_link")

        assert "Link 'nonexistent_link' not found" in caplog.text

    def test_parent_joint_with_none_parent_id(self):
        """Tests initialization when parent_joint returns None (root link)."""
        # Create a simple robot with just one link (no parent)
        root_link = Link.from_sphere(name="root_link", radius=0.1)

        robot = Robot()
        robot.add_link(root_link)

        # This should work fine - root link has no parent
        manager = MoveManager(robot)

        assert len(manager.link_name_to_id) == 1
        assert "root_link" in manager.link_name_to_id

    def test_transform_application_for_revolute_joint(self):
        """Tests the specific transform application logic for revolute joints."""
        base_link = Link.from_sphere(name="base_link", radius=0.1)
        arm_link = Link.from_sphere(name="arm_link", radius=0.1)

        revolute_joint = Joint(
            name="revolute_joint",
            type=JointType.REVOLUTE,
            parent_link_name="base_link",
            child_link_name="arm_link",
            direction=np.array([0, 0, 1]),  # Z-axis
            center=np.array([1, 0, 0]),  # Rotate around point (1,0,0)
        )

        robot = Robot()
        robot.add_link(base_link)
        robot.add_link(arm_link)
        robot.add_joint(revolute_joint)

        manager = MoveManager(robot)

        # Move by 90 degrees
        manager.move("revolute_joint", np.pi / 2)

        transform = manager.get_transform("arm_link")

        # Verify the transform is not identity
        assert transform != Transform()

        # The rotation should be around the center point [1, 0, 0]
        # This tests the specific logic: T(center) * R * T(-center)
        rotation = R.from_rotvec([0, 0, np.pi / 2])
        center = np.array([1, 0, 0])

        expected_transform = Transform(rotate=rotation, translate=center).apply(
            Transform(translate=-center)
        )

        assert transform == expected_transform

    def test_zero_values_for_joints(self, simple_robot_and_manager):
        """Tests moving joints with zero values."""
        manager, _ = simple_robot_and_manager

        # Move prismatic joint by zero (should result in identity transform)
        manager.move("prismatic_joint", 0.0)

        transform = manager.get_transform("arm_link")
        # Zero movement should result in identity transform
        assert transform == Transform()

    def test_negative_values_for_joints(self, simple_robot_and_manager):
        """Tests moving joints with negative values."""
        manager, _ = simple_robot_and_manager

        # Move prismatic joint by negative value
        manager.move("prismatic_joint", -2.5)

        transform = manager.get_transform("arm_link")

        # Expected transform: movement in negative X direction
        expected_transform = Transform(translate=np.array([-2.5, 0, 0]))
        assert transform == expected_transform

    def test_large_values_for_joints(self, simple_robot_and_manager):
        """Tests moving joints with large values."""
        manager, _ = simple_robot_and_manager

        # Move prismatic joint by large value
        large_value = 1000.0
        manager.move("prismatic_joint", large_value)

        transform = manager.get_transform("arm_link")

        # Expected transform: large movement in X direction
        expected_transform = Transform(translate=np.array([large_value, 0, 0]))
        assert transform == expected_transform

    def test_floating_point_precision_values(self, simple_robot_and_manager):
        """Tests moving joints with high precision floating point values."""
        manager, _ = simple_robot_and_manager

        # Use a value with high precision
        precise_value = 1.23456789012345
        manager.move("prismatic_joint", precise_value)

        transform = manager.get_transform("arm_link")

        expected_transform = Transform(translate=np.array([precise_value, 0, 0]))
        assert transform == expected_transform
