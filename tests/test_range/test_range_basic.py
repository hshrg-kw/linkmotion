import pytest
import numpy as np

from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.range.range_cal import RangeCalcCondition, RangeCalculator


@pytest.fixture
def simple_robot():
    """Creates a simple robot with two joints for testing."""
    robot = Robot()

    # Create links positioned to avoid initial collision
    base_link = Link.from_sphere(
        name="base_link", radius=0.1, center=np.array([0, 0, 0])
    )
    arm_link_1 = Link.from_sphere(
        name="arm_link_1", radius=0.1, center=np.array([1, 0, 0])
    )
    arm_link_2 = Link.from_sphere(
        name="arm_link_2", radius=0.1, center=np.array([0, 1, 0])
    )

    # Prismatic joint with limits
    joint1 = Joint(
        name="joint1",
        type=JointType.PRISMATIC,
        parent_link_name="base_link",
        child_link_name="arm_link_1",
        direction=np.array([1, 0, 0]),
        min_=-2.0,
        max_=2.0,
    )

    # Revolute joint with limits
    joint2 = Joint(
        name="joint2",
        type=JointType.REVOLUTE,
        parent_link_name="base_link",
        child_link_name="arm_link_2",
        direction=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        min_=-np.pi,
        max_=np.pi,
    )

    robot.add_link(base_link)
    robot.add_link(arm_link_1)
    robot.add_link(arm_link_2)
    robot.add_joint(joint1)
    robot.add_joint(joint2)

    return robot


@pytest.fixture
def collision_robot():
    """Creates a robot setup that will have collisions for testing."""
    robot = Robot()

    # Create links that can collide
    base_link = Link.from_sphere(
        name="base_link", radius=0.5, center=np.array([0, 0, 0])
    )
    moving_link = Link.from_sphere(
        name="moving_link", radius=0.3, center=np.array([2, 0, 0])
    )
    obstacle_link = Link.from_sphere(
        name="obstacle_link", radius=0.5, center=np.array([5, 0, 0])
    )

    # Prismatic joint that can move the moving_link towards the obstacle
    joint = Joint(
        name="collision_joint",
        type=JointType.PRISMATIC,
        parent_link_name="base_link",
        child_link_name="moving_link",
        direction=np.array([1, 0, 0]),
        min_=-1.0,
        max_=10.0,
    )

    robot.add_link(base_link)
    robot.add_link(moving_link)
    robot.add_link(obstacle_link)
    robot.add_joint(joint)

    return robot


@pytest.fixture
def continuous_joint_robot():
    """Creates a robot with a continuous joint for testing."""
    robot = Robot()

    base_link = Link.from_sphere(name="base_link", radius=0.1)
    rotating_link = Link.from_sphere(name="rotating_link", radius=0.1)

    continuous_joint = Joint(
        name="continuous_joint",
        type=JointType.CONTINUOUS,
        parent_link_name="base_link",
        child_link_name="rotating_link",
        direction=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
    )

    robot.add_link(base_link)
    robot.add_link(rotating_link)
    robot.add_joint(continuous_joint)

    return robot


@pytest.fixture
def fixed_joint_robot():
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


class TestRangeCalcCondition:
    """Tests for RangeCalcCondition class."""

    def test_initialization(self, simple_robot):
        """Test RangeCalcCondition initialization."""
        joint = simple_robot.joint("joint1")
        survey_points = np.array([0.0, 0.5, 1.0])

        condition = RangeCalcCondition(joint, survey_points)

        assert condition.joint == joint
        np.testing.assert_array_equal(condition.survey_points, survey_points)

    def test_properties(self, simple_robot):
        """Test RangeCalcCondition properties."""
        joint = simple_robot.joint("joint1")
        survey_points = np.array([0.0, 0.5, 1.0])

        condition = RangeCalcCondition(joint, survey_points)

        assert condition.min == joint.min
        assert condition.max == joint.max
        assert condition.joint_name == joint.name

    def test_repr(self, simple_robot):
        """Test RangeCalcCondition string representation."""
        joint = simple_robot.joint("joint1")
        survey_points = np.array([0.0, 0.5, 1.0])

        condition = RangeCalcCondition(joint, survey_points)
        repr_str = repr(condition)

        assert "RangeCalcCondition" in repr_str
        assert "joint1" in repr_str
        assert "points=3" in repr_str
        assert "range=[-2.000, 2.000]" in repr_str


class TestRangeCalculator:
    """Tests for RangeCalculator class."""

    def test_initialization(self, simple_robot):
        """Test RangeCalculator initialization."""
        link_names1 = {"arm_link_1"}
        link_names2 = {"base_link"}

        calculator = RangeCalculator(simple_robot, link_names1, link_names2)

        assert calculator.link_names1 == link_names1
        assert calculator.link_names2 == link_names2
        assert calculator.results is None
        assert len(calculator.calc_conditions) == 0

    def test_repr(self, simple_robot):
        """Test RangeCalculator string representation."""
        link_names1 = {"arm_link_1"}
        link_names2 = {"base_link"}

        calculator = RangeCalculator(simple_robot, link_names1, link_names2)
        repr_str = repr(calculator)

        assert "RangeCalculator" in repr_str
        assert "axes=0" in repr_str
        assert "links1=1" in repr_str
        assert "links2=1" in repr_str
        assert "computed=False" in repr_str

    def test_add_axis_prismatic(self, simple_robot):
        """Test adding a prismatic joint axis."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})
        survey_points = np.array([-1.0, 0.0, 1.0])

        calculator.add_axis("joint1", survey_points)

        assert "joint1" in calculator.calc_conditions
        condition = calculator.calc_conditions["joint1"]
        np.testing.assert_array_equal(
            condition.survey_points, np.array([-1.0, 0.0, 1.0])
        )

    def test_add_axis_revolute(self, simple_robot):
        """Test adding a revolute joint axis."""
        calculator = RangeCalculator(simple_robot, {"arm_link_2"}, {"base_link"})
        survey_points = np.array([-1.0, 0.0, 1.0])

        calculator.add_axis("joint2", survey_points)

        assert "joint2" in calculator.calc_conditions
        condition = calculator.calc_conditions["joint2"]
        np.testing.assert_array_equal(
            condition.survey_points, np.array([-1.0, 0.0, 1.0])
        )

    def test_add_axis_continuous(self, continuous_joint_robot):
        """Test adding a continuous joint axis."""
        calculator = RangeCalculator(
            continuous_joint_robot, {"rotating_link"}, {"base_link"}
        )
        survey_points = np.array([0.0, np.pi / 2, np.pi])

        calculator.add_axis("continuous_joint", survey_points)

        assert "continuous_joint" in calculator.calc_conditions

    def test_add_axis_invalid_joint_type(self, fixed_joint_robot):
        """Test adding an unsupported joint type raises ValueError."""
        calculator = RangeCalculator(fixed_joint_robot, {"fixed_link"}, {"base_link"})
        survey_points = np.array([0.0, 1.0])

        with pytest.raises(
            ValueError, match="Joint 'fixed_joint' type .* is not supported"
        ):
            calculator.add_axis("fixed_joint", survey_points)

    def test_add_axis_nonexistent_joint(self, simple_robot):
        """Test adding a non-existent joint raises KeyError."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})
        survey_points = np.array([0.0, 1.0])

        with pytest.raises(
            KeyError, match="Joint 'nonexistent' not found in robot model"
        ):
            calculator.add_axis("nonexistent", survey_points)

    def test_add_axis_points_outside_limits(self, simple_robot):
        """Test adding survey points outside joint limits raises ValueError."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})
        # joint1 has limits [-2.0, 2.0], so [-3.0, 3.0] should be outside
        survey_points = np.array([-3.0, 0.0, 3.0])

        with pytest.raises(
            ValueError, match="Survey points .* are outside joint limits"
        ):
            calculator.add_axis("joint1", survey_points)

    def test_add_axis_points_at_limits(self, simple_robot):
        """Test adding survey points exactly at joint limits works."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})
        # joint1 has limits [-2.0, 2.0]
        survey_points = np.array([-2.0, 0.0, 2.0])

        calculator.add_axis("joint1", survey_points)

        assert "joint1" in calculator.calc_conditions

    def test_get_axis_names_empty(self, simple_robot):
        """Test get_axis_names with no axes added."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})

        names = calculator.get_axis_names()

        assert names == []

    def test_get_axis_names_with_axes(self, simple_robot):
        """Test get_axis_names with axes added."""
        calculator = RangeCalculator(
            simple_robot, {"arm_link_1", "arm_link_2"}, {"base_link"}
        )

        calculator.add_axis("joint1", np.array([-1.0, 0.0, 1.0]))
        calculator.add_axis("joint2", np.array([-1.0, 0.0, 1.0]))

        names = calculator.get_axis_names()

        assert "joint1" in names
        assert "joint2" in names
        assert len(names) == 2

    def test_get_axis_points_empty(self, simple_robot):
        """Test get_axis_points with no axes added."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})

        points = calculator.get_axis_points()

        assert points == []

    def test_get_axis_points_with_axes(self, simple_robot):
        """Test get_axis_points with axes added."""
        calculator = RangeCalculator(
            simple_robot, {"arm_link_1", "arm_link_2"}, {"base_link"}
        )

        points1 = np.array([-1.0, 0.0, 1.0])
        points2 = np.array([-0.5, 0.0, 0.5])

        calculator.add_axis("joint1", points1)
        calculator.add_axis("joint2", points2)

        axis_points = calculator.get_axis_points()

        assert len(axis_points) == 2
        # Note: order depends on dictionary iteration order in Python 3.7+

    def test_calculate_single_point_no_collision(self, simple_robot):
        """Test _calculate_single_point with no collision."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})
        calculator.add_axis("joint1", np.array([0.0]))
        calculator.add_axis("joint2", np.array([0.0]))

        result = calculator._calculate_single_point((0.0, 0.0))

        assert result == 0.0  # No collision

    def test_calculate_single_point_with_collision(self, collision_robot):
        """Test _calculate_single_point with collision."""
        calculator = RangeCalculator(
            collision_robot, {"moving_link"}, {"obstacle_link"}
        )
        calculator.add_axis("collision_joint", np.array([0.0, 2.5]))

        # Move to a position that causes collision (smaller distance to ensure collision)
        result = calculator._calculate_single_point((2.5,))

        assert result == 1.0  # Collision detected

    def test_generate_grid_points_no_axes(self, simple_robot):
        """Test _generate_grid_points with no axes raises ValueError."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})

        with pytest.raises(ValueError, match="No axes have been added"):
            calculator._generate_grid_points()

    def test_generate_grid_points_single_axis(self, simple_robot):
        """Test _generate_grid_points with single axis."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})
        calculator.add_axis("joint1", np.array([0.0, 1.0]))

        grid = calculator._generate_grid_points()
        points = list(grid)

        assert len(points) == 2
        assert (0.0,) in points
        assert (1.0,) in points

    def test_generate_grid_points_multiple_axes(self, simple_robot):
        """Test _generate_grid_points with multiple axes."""
        calculator = RangeCalculator(
            simple_robot, {"arm_link_1", "arm_link_2"}, {"base_link"}
        )
        calculator.add_axis("joint1", np.array([0.0, 1.0]))
        calculator.add_axis("joint2", np.array([0.0, 0.5]))

        grid = calculator._generate_grid_points()
        points = list(grid)

        # Should have 2 * 2 = 4 combinations
        assert len(points) == 4

    def test_reshape_results(self, simple_robot):
        """Test _reshape_results functionality."""
        calculator = RangeCalculator(
            simple_robot, {"arm_link_1", "arm_link_2"}, {"base_link"}
        )
        calculator.add_axis("joint1", np.array([0.0, 1.0]))
        calculator.add_axis("joint2", np.array([0.0, 0.5]))

        flat_results = [0.0, 1.0, 0.0, 1.0]  # 4 results for 2x2 grid

        reshaped = calculator._reshape_results(flat_results)

        assert reshaped.shape == (2, 2)
        np.testing.assert_array_equal(reshaped, [[0.0, 1.0], [0.0, 1.0]])

    def test_execute_no_axes(self, simple_robot):
        """Test execute with no axes raises ValueError."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})

        with pytest.raises(ValueError, match="No calculation axes defined"):
            calculator.execute()

    def test_execute_single_axis_no_collision(self, simple_robot):
        """Test execute with single axis and no collisions."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})
        calculator.add_axis("joint1", np.array([0.0, 0.5]))

        calculator.execute()

        assert calculator.results is not None
        assert calculator.results.shape == (2,)
        # Both points should have no collision
        np.testing.assert_array_equal(calculator.results, [0.0, 0.0])

    def test_execute_multiple_axes_no_collision(self, simple_robot):
        """Test execute with multiple axes and no collisions."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})
        calculator.add_axis("joint1", np.array([0.0, 0.5]))
        calculator.add_axis("joint2", np.array([0.0, 0.5]))

        calculator.execute()

        assert calculator.results is not None
        assert calculator.results.shape == (2, 2)
        # All combinations should have no collision
        expected = np.array([[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(calculator.results, expected)

    def test_execute_with_collision(self, collision_robot):
        """Test execute with collisions detected."""
        calculator = RangeCalculator(
            collision_robot, {"moving_link"}, {"obstacle_link"}
        )
        # Add points that include collision positions
        calculator.add_axis(
            "collision_joint", np.array([0.0, 2.5])
        )  # 2.5 should cause collision

        calculator.execute()

        assert calculator.results is not None
        assert calculator.results.shape == (2,)
        # First point (0.0) should be no collision, second (2.5) should be collision
        assert calculator.results[0] == 0.0  # No collision
        assert calculator.results[1] == 1.0  # Collision

    def test_execute_updates_repr(self, simple_robot):
        """Test that execute updates the repr to show computation is complete."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})
        calculator.add_axis("joint1", np.array([0.0]))

        # Before execution
        assert "computed=False" in repr(calculator)

        calculator.execute()

        # After execution
        assert "computed=True" in repr(calculator)

    def test_add_axis_sorts_points(self, simple_robot):
        """Test that add_axis sorts survey points."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"base_link"})
        unsorted_points = np.array([1.0, -1.0, 0.5, -0.5])

        calculator.add_axis("joint1", unsorted_points)

        condition = calculator.calc_conditions["joint1"]
        expected_sorted = np.array([-1.0, -0.5, 0.5, 1.0])
        np.testing.assert_array_equal(condition.survey_points, expected_sorted)


class TestRangeCalculatorIntegration:
    """Integration tests for RangeCalculator."""

    def test_complete_workflow_small_grid(self, simple_robot):
        """Test complete workflow with a small grid."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})

        # Add two axes with small number of points
        calculator.add_axis("joint1", np.array([-0.5, 0.0, 0.5]))
        calculator.add_axis("joint2", np.array([-0.5, 0.0, 0.5]))

        # Execute calculation
        calculator.execute()

        # Verify results
        assert calculator.results is not None
        assert calculator.results.shape == (3, 3)
        assert calculator.results.dtype == np.float64

        # All combinations should result in no collision for this setup
        expected = np.zeros((3, 3))
        np.testing.assert_array_equal(calculator.results, expected)

    def test_mixed_joint_types(self, simple_robot):
        """Test with mixed joint types (prismatic and revolute)."""
        calculator = RangeCalculator(
            simple_robot, {"arm_link_1", "arm_link_2"}, {"base_link"}
        )

        # Add prismatic joint
        calculator.add_axis("joint1", np.array([-1.0, 0.0, 1.0]))
        # Add revolute joint
        calculator.add_axis("joint2", np.array([-1.0, 0.0, 1.0]))

        calculator.execute()

        assert calculator.results is not None
        assert calculator.results.shape == (3, 3)

    def test_large_grid_performance(self, simple_robot):
        """Test performance with a larger grid (but still reasonable for testing)."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})

        # Create larger but manageable grid
        calculator.add_axis("joint1", np.linspace(-1.0, 1.0, 5))
        calculator.add_axis("joint2", np.linspace(-1.0, 1.0, 6))

        calculator.execute()

        assert calculator.results is not None
        assert calculator.results.shape == (5, 6)
        assert calculator.results.size == 30

    def test_edge_case_single_point(self, simple_robot):
        """Test edge case with single survey point."""
        calculator = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})

        calculator.add_axis("joint1", np.array([0.0]))

        calculator.execute()

        assert calculator.results is not None
        assert calculator.results.shape == (1,)
        assert calculator.results[0] == 0.0

    def test_reproducible_results(self, simple_robot):
        """Test that results are reproducible."""
        calculator1 = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})
        calculator2 = RangeCalculator(simple_robot, {"arm_link_1"}, {"arm_link_2"})

        points = np.array([0.0, 0.5, 1.0])

        calculator1.add_axis("joint1", points)
        calculator1.add_axis("joint2", points)

        calculator2.add_axis("joint1", points)
        calculator2.add_axis("joint2", points)

        calculator1.execute()
        calculator2.execute()

        np.testing.assert_array_equal(calculator1.results, calculator2.results)
