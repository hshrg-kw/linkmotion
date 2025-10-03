import pytest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.move.manager import MoveManager
from linkmotion.transform import Transform


@pytest.fixture
def extended_robot_and_manager() -> tuple[MoveManager, Robot]:
    """Provides a MoveManager instance with various joint types for comprehensive testing."""
    # Define links
    base_link = Link.from_sphere(name="base_link", radius=0.1)
    continuous_link = Link.from_sphere(name="continuous_link", radius=0.1)
    planar_link = Link.from_sphere(name="planar_link", radius=0.1)
    floating_link = Link.from_sphere(name="floating_link", radius=0.1)
    child_link = Link.from_sphere(name="child_link", radius=0.1)

    # Define joints with different types
    continuous_joint = Joint(
        name="continuous_joint",
        type=JointType.CONTINUOUS,
        parent_link_name="base_link",
        child_link_name="continuous_link",
        direction=np.array([0, 1, 0]),  # Y-axis rotation
        center=np.array([0.5, 0, 0]),  # Rotate around a point
    )

    planar_joint = Joint(
        name="planar_joint",
        type=JointType.PLANAR,
        parent_link_name="continuous_link",
        child_link_name="planar_link",
        direction=np.array([0, 0, 1]),  # Z-axis (for planar joints)
    )

    floating_joint = Joint(
        name="floating_joint",
        type=JointType.FLOATING,
        parent_link_name="planar_link",
        child_link_name="floating_link",
    )

    # Joint with center for testing get_center
    revolute_with_center = Joint(
        name="revolute_with_center",
        type=JointType.REVOLUTE,
        parent_link_name="floating_link",
        child_link_name="child_link",
        direction=np.array([1, 0, 0]),  # X-axis rotation
        center=np.array([1, 1, 1]),  # Non-zero center
    )

    # Build the robot
    robot = Robot()
    robot.add_link(base_link)
    robot.add_link(continuous_link)
    robot.add_link(planar_link)
    robot.add_link(floating_link)
    robot.add_link(child_link)

    robot.add_joint(continuous_joint)
    robot.add_joint(planar_joint)
    robot.add_joint(floating_joint)
    robot.add_joint(revolute_with_center)

    manager = MoveManager(robot)
    return manager, robot


@pytest.fixture
def inconsistent_robot_fixture() -> Robot:
    """Creates a basic robot for testing inconsistent scenarios."""
    base_link = Link.from_sphere(name="base_link", radius=0.1)
    child_link = Link.from_sphere(name="child_link", radius=0.1)

    robot = Robot()
    robot.add_link(base_link)
    robot.add_link(child_link)

    return robot


class TestMoveManagerExtended:
    """Extended tests for MoveManager to improve coverage."""

    def test_move_continuous_joint(self, extended_robot_and_manager):
        """Tests moving a continuous joint."""
        manager, _ = extended_robot_and_manager

        # Move continuous joint by π/2 radians (90 degrees)
        manager.move("continuous_joint", np.pi / 2)

        actual_transform = manager.get_transform("continuous_link")

        # Calculate expected transform: rotation around Y-axis at center point
        center = np.array([0.5, 0, 0])
        rotation = R.from_rotvec([0, np.pi / 2, 0])

        t_inv = Transform(translate=-center)
        rot = Transform(rotate=rotation)
        t = Transform(translate=center)
        expected_transform = t.apply(rot.apply(t_inv))

        assert actual_transform == expected_transform

    def test_move_planar_joint(self, extended_robot_and_manager):
        """Tests moving a planar joint with Transform value."""
        manager, _ = extended_robot_and_manager

        # Create a transform for planar movement
        planar_transform = Transform(
            translate=np.array([1.0, 2.0, 0.0]),
            rotate=R.from_euler("z", 45, degrees=True),
        )

        manager.move("planar_joint", planar_transform)

        actual_transform = manager.get_transform("planar_link")

        # For planar joints, the transform should be applied to the parent's transform
        parent_transform = manager.get_transform("continuous_link")
        expected_transform = parent_transform.apply(planar_transform)

        assert actual_transform == expected_transform

    def test_move_floating_joint(self, extended_robot_and_manager):
        """Tests moving a floating joint with Transform value."""
        manager, _ = extended_robot_and_manager

        # Create a 6DOF transform for floating movement
        floating_transform = Transform(
            translate=np.array([3.0, 4.0, 5.0]),
            rotate=R.from_euler("xyz", [30, 45, 60], degrees=True),
        )

        manager.move("floating_joint", floating_transform)

        actual_transform = manager.get_transform("floating_link")

        # For floating joints, the transform should be applied to the parent's transform
        parent_transform = manager.get_transform("planar_link")
        expected_transform = parent_transform.apply(floating_transform)

        assert actual_transform == expected_transform

    def test_move_joint_without_required_attributes(self, extended_robot_and_manager):
        """Tests moving joints that lack required attributes."""
        manager, robot = extended_robot_and_manager

        # Create a revolute joint with center, then manually set it to None
        no_center_joint = Joint(
            name="no_center_joint",
            type=JointType.REVOLUTE,
            parent_link_name="base_link",
            child_link_name="continuous_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0, 0, 0]),  # Required for creation
        )

        # Manually add to robot for testing
        robot._joint_dict["no_center_joint"] = no_center_joint

        # Set center to None to test the error path
        no_center_joint.center = None

        with pytest.raises(
            ValueError, match="does not have a defined center for rotation"
        ):
            manager.move("no_center_joint", 1.0)

    def test_invalid_value_types_for_joints(self, extended_robot_and_manager):
        """Tests providing invalid value types for different joint types."""
        manager, _ = extended_robot_and_manager

        # Wrong type for continuous joint
        with pytest.raises(
            ValueError, match="Value for revolute joint must be a float"
        ):
            manager.move("continuous_joint", Transform())

        # Wrong type for planar joint
        with pytest.raises(
            ValueError, match="Value for planar/floating joint must be a Transform"
        ):
            manager.move("planar_joint", 1.0)

        # Wrong type for floating joint
        with pytest.raises(
            ValueError, match="Value for planar/floating joint must be a Transform"
        ):
            manager.move("floating_joint", "invalid")

    def test_move_nonexistent_joint_detailed(self, extended_robot_and_manager):
        """Tests detailed error handling for nonexistent joints."""
        manager, _ = extended_robot_and_manager

        with pytest.raises(
            ValueError, match="Joint 'ghost_joint' not found in the robot model"
        ):
            manager.move("ghost_joint", 1.0)

    def test_get_center_with_joint_having_center(self, extended_robot_and_manager):
        """Tests get_center for a joint that has a defined center."""
        manager, _ = extended_robot_and_manager

        # Move the floating_link first to create a non-identity transform
        floating_transform = Transform(translate=np.array([1.0, 1.0, 1.0]))
        manager.move("floating_joint", floating_transform)

        # Get the center of revolute_with_center joint
        world_center = manager.get_center("revolute_with_center")

        # Calculate expected center: parent transform applied to joint center
        parent_transform = manager.get_transform("floating_link")
        expected_center = parent_transform.apply(np.array([1, 1, 1]))

        assert world_center is not None
        assert np.allclose(world_center, expected_center)

    def test_get_center_nonexistent_joint(self, extended_robot_and_manager):
        """Tests get_center with nonexistent joint."""
        manager, _ = extended_robot_and_manager

        with pytest.raises(ValueError, match="Joint 'phantom_joint' not found"):
            manager.get_center("phantom_joint")

    def test_get_direction_nonexistent_joint(self, extended_robot_and_manager):
        """Tests get_direction with nonexistent joint."""
        manager, _ = extended_robot_and_manager

        with pytest.raises(ValueError, match="Joint 'phantom_joint' not found"):
            manager.get_direction("phantom_joint")

    def test_get_link_visual_mesh_nonexistent_link(self, extended_robot_and_manager):
        """Tests get_link_visual_mesh with nonexistent link."""
        manager, _ = extended_robot_and_manager

        with pytest.raises(ValueError, match="Link 'phantom_link' not found"):
            manager.get_link_visual_mesh("phantom_link")

    def test_get_link_collision_obj_nonexistent_link(self, extended_robot_and_manager):
        """Tests get_link_collision_obj with nonexistent link."""
        manager, _ = extended_robot_and_manager

        with pytest.raises(ValueError, match="Link 'phantom_link' not found"):
            manager.get_link_collision_obj("phantom_link")

    def test_get_link_visual_mesh_success(self, extended_robot_and_manager):
        """Tests successful retrieval of link visual mesh."""
        manager, robot = extended_robot_and_manager

        # Move a link to create a non-identity transform
        manager.move("continuous_joint", np.pi / 4)

        # Get visual mesh
        visual_mesh = manager.get_link_visual_mesh("continuous_link")

        # Verify it's a trimesh object
        assert isinstance(visual_mesh, trimesh.Trimesh)

    def test_get_link_collision_obj_success(self, extended_robot_and_manager):
        """Tests successful retrieval of link collision object."""
        manager, _ = extended_robot_and_manager

        # Move a link to create a non-identity transform
        manager.move("continuous_joint", np.pi / 4)

        # Get collision object
        collision_obj = manager.get_link_collision_obj("continuous_link")

        # Verify it returns a collision object (it's actually an FCL CollisionObject)
        assert collision_obj is not None

    def test_integer_values_for_numeric_joints(self, extended_robot_and_manager):
        """Tests that integer values work for numeric joint types."""
        manager, _ = extended_robot_and_manager

        # Test integer value for continuous joint
        manager.move("continuous_joint", 1)  # Integer instead of float

        transform = manager.get_transform("continuous_link")
        assert transform != Transform()  # Should have moved

    def test_complex_chained_movements(self, extended_robot_and_manager):
        """Tests complex chain of movements through multiple joint types."""
        manager, _ = extended_robot_and_manager

        # Move each joint in the chain
        manager.move("continuous_joint", np.pi / 6)

        planar_transform = Transform(translate=np.array([0.5, 0.5, 0]))
        manager.move("planar_joint", planar_transform)

        floating_transform = Transform(
            translate=np.array([1, 2, 3]), rotate=R.from_euler("x", 30, degrees=True)
        )
        manager.move("floating_joint", floating_transform)

        manager.move("revolute_with_center", np.pi / 4)

        # Get final transform
        final_transform = manager.get_transform("child_link")

        # Verify it's not identity (something moved)
        assert final_transform != Transform()

        # Verify the transform chain is correct
        base_to_continuous = manager.get_transform("continuous_link")
        continuous_to_planar = manager.get_transform("planar_link")
        planar_to_floating = manager.get_transform("floating_link")
        floating_to_child = manager.get_transform("child_link")

        # Each subsequent transform should be different
        transforms = [
            base_to_continuous,
            continuous_to_planar,
            planar_to_floating,
            floating_to_child,
        ]
        for i in range(len(transforms) - 1):
            assert transforms[i] != transforms[i + 1]

    def test_reset_after_complex_movements(self, extended_robot_and_manager):
        """Tests reset functionality after complex movements."""
        manager, _ = extended_robot_and_manager

        # Perform complex movements
        manager.move("continuous_joint", np.pi)
        manager.move("planar_joint", Transform(translate=np.array([5, 5, 5])))
        manager.move(
            "floating_joint",
            Transform(
                translate=np.array([10, 10, 10]),
                rotate=R.from_euler("xyz", [90, 90, 90], degrees=True),
            ),
        )

        # Verify movements occurred
        assert manager.get_transform("continuous_link") != Transform()
        assert manager.get_transform("planar_link") != Transform()
        assert manager.get_transform("floating_link") != Transform()

        # Reset all movements
        manager.reset_move()

        # Verify all transforms are back to identity
        assert manager.get_transform("continuous_link") == Transform()
        assert manager.get_transform("planar_link") == Transform()
        assert manager.get_transform("floating_link") == Transform()

    def test_unsupported_joint_type(self, extended_robot_and_manager):
        """Tests handling of unsupported joint types."""
        manager, robot = extended_robot_and_manager

        # Create a joint with valid type first
        invalid_joint = Joint(
            name="invalid_type_joint",
            type=JointType.REVOLUTE,
            parent_link_name="base_link",
            child_link_name="continuous_link",
            direction=np.array([0, 0, 1]),
            center=np.array([0, 0, 0]),
        )

        # Manually add to robot
        robot._joint_dict["invalid_type_joint"] = invalid_joint

        # Simulate an unsupported joint type by setting an invalid string value
        original_type = invalid_joint.type
        try:
            invalid_joint.type = (
                "UNSUPPORTED_TYPE"  # This would be caught by match case _
            )

            with pytest.raises(ValueError, match="Unsupported joint type"):
                manager.move("invalid_type_joint", 1.0)
        finally:
            # Restore original type
            invalid_joint.type = original_type
