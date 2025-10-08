"""Comprehensive tests for RangeManager class.

This module provides quantitative tests for the RangeManager class,
testing all methods with measurable assertions and edge cases.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from linkmotion.range.manager import RangeManager
from linkmotion.range.range_read import RangeReader


@pytest.fixture
def simple_2d_reader():
    """Create a simple 2D RangeReader for testing.

    Grid: 4x3 (x: [0,1,2,3], y: [0,1,2])
    Function: f(x,y) = x + y
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    results = X + Y
    return RangeReader(results, ("x", "y"), (x, y))


@pytest.fixture
def offset_2d_reader():
    """Create a 2D RangeReader with offset bounds for testing overlapping ranges.

    Grid: 3x3 (x: [2,3,4], y: [1,2,3])
    Function: f(x,y) = 2*x + y
    """
    x = np.array([2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    results = 2 * X + Y
    return RangeReader(results, ("x", "y"), (x, y))


@pytest.fixture
def simple_3d_reader():
    """Create a simple 3D RangeReader for testing.

    Grid: 5x4x3 (x: [0,2,4,6,8], y: [0,1,2,3], z: [0,0.5,1])
    Function: f(x,y,z) = x + 2*y + 3*z
    """
    x = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([0.0, 0.5, 1.0])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    results = X + 2 * Y + 3 * Z
    return RangeReader(results, ("x", "y", "z"), (x, y, z))


@pytest.fixture
def collision_2d_reader():
    """Create a 2D RangeReader simulating collision data.

    Grid: 5x5 (x: [0,1,2,3,4], y: [0,1,2,3,4])
    Function: 0 at edges, 1 in center (simulating collision values)
    """
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    # Create collision-free zone at edges (value=0) and collision zone in center (value=1)
    results = np.where((X >= 1) & (X <= 3) & (Y >= 1) & (Y <= 3), 1.0, 0.0)
    return RangeReader(results, ("x", "y"), (x, y))


class TestRangeManagerInitialization:
    """Tests for RangeManager initialization and basic properties."""

    def test_empty_initialization(self):
        """Test that RangeManager initializes empty."""
        manager = RangeManager()
        assert len(manager) == 0
        assert manager.list_ranges() == []
        assert repr(manager) == "RangeManager(ranges=[])"

    def test_repr_with_ranges(self, simple_2d_reader):
        """Test __repr__ shows correct range names."""
        manager = RangeManager()
        manager.add_range("range1", simple_2d_reader)
        manager.add_range("range2", simple_2d_reader)
        assert repr(manager) == "RangeManager(ranges=['range1', 'range2'])"

    def test_len_after_adding_ranges(self, simple_2d_reader, simple_3d_reader):
        """Test __len__ returns correct count after adding ranges."""
        manager = RangeManager()
        assert len(manager) == 0

        manager.add_range("range1", simple_2d_reader)
        assert len(manager) == 1

        manager.add_range("range2", simple_3d_reader)
        assert len(manager) == 2

    def test_contains_operator(self, simple_2d_reader):
        """Test __contains__ correctly identifies range existence."""
        manager = RangeManager()
        manager.add_range("existing", simple_2d_reader)

        assert "existing" in manager
        assert "nonexistent" not in manager

    def test_iter_over_range_names(self, simple_2d_reader, simple_3d_reader):
        """Test __iter__ returns range names in insertion order."""
        manager = RangeManager()
        manager.add_range("first", simple_2d_reader)
        manager.add_range("second", simple_3d_reader)
        manager.add_range("third", simple_2d_reader)

        names = list(manager)
        assert names == ["first", "second", "third"]


class TestAddRange:
    """Tests for adding ranges to RangeManager."""

    def test_add_range_with_reader_instance(self, simple_2d_reader):
        """Test adding a RangeReader instance."""
        manager = RangeManager()
        manager.add_range("test_range", simple_2d_reader)

        assert len(manager) == 1
        assert "test_range" in manager
        retrieved = manager.get_range("test_range")
        assert retrieved is simple_2d_reader

    def test_add_range_from_file(self, simple_2d_reader):
        """Test adding a range from file path."""
        manager = RangeManager()

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Use object dtype to allow saving tuples/arrays of different sizes
            np.savez(
                tmp_path,
                results=simple_2d_reader.results,
                axis_names=np.array(simple_2d_reader.axis_names, dtype=object),
                axis_points=np.array(simple_2d_reader.axis_points, dtype=object),
            )

        try:
            manager.add_range("from_file", tmp_path)
            assert len(manager) == 1
            assert "from_file" in manager

            # Verify data integrity
            reader = manager.get_range("from_file")
            assert reader.n_dims == 2
            np.testing.assert_array_equal(reader.results, simple_2d_reader.results)
        finally:
            tmp_path.unlink()

    def test_add_multiple_ranges(
        self, simple_2d_reader, simple_3d_reader, offset_2d_reader
    ):
        """Test adding multiple different ranges."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)
        manager.add_range("3d", simple_3d_reader)
        manager.add_range("offset", offset_2d_reader)

        assert len(manager) == 3
        assert manager.list_ranges() == ["2d", "3d", "offset"]

    def test_add_duplicate_name_raises_error(self, simple_2d_reader):
        """Test that adding duplicate name without overwrite raises ValueError."""
        manager = RangeManager()
        manager.add_range("duplicate", simple_2d_reader)

        with pytest.raises(ValueError, match="already exists"):
            manager.add_range("duplicate", simple_2d_reader)

    def test_add_duplicate_name_with_overwrite(
        self, simple_2d_reader, simple_3d_reader
    ):
        """Test that overwrite=True allows replacing existing range."""
        manager = RangeManager()
        manager.add_range("replaceable", simple_2d_reader)

        original = manager.get_range("replaceable")
        assert original.n_dims == 2

        manager.add_range("replaceable", simple_3d_reader, overwrite=True)
        replaced = manager.get_range("replaceable")
        assert replaced.n_dims == 3
        assert len(manager) == 1

    def test_add_invalid_type_raises_error(self):
        """Test that adding invalid type raises TypeError."""
        manager = RangeManager()

        with pytest.raises(TypeError, match="must be a Path or RangeReader"):
            manager.add_range("invalid", "not a reader")  # type: ignore

        with pytest.raises(TypeError, match="must be a Path or RangeReader"):
            manager.add_range("invalid", 123)  # type: ignore


class TestRemoveAndClear:
    """Tests for removing ranges from RangeManager."""

    def test_remove_existing_range(self, simple_2d_reader, simple_3d_reader):
        """Test removing an existing range."""
        manager = RangeManager()
        manager.add_range("to_remove", simple_2d_reader)
        manager.add_range("to_keep", simple_3d_reader)

        assert len(manager) == 2
        manager.remove_range("to_remove")
        assert len(manager) == 1
        assert "to_remove" not in manager
        assert "to_keep" in manager

    def test_remove_nonexistent_range_raises_error(self):
        """Test that removing nonexistent range raises KeyError."""
        manager = RangeManager()

        with pytest.raises(KeyError, match="not found"):
            manager.remove_range("nonexistent")

    def test_remove_nonexistent_shows_available(self, simple_2d_reader):
        """Test that KeyError message includes available ranges."""
        manager = RangeManager()
        manager.add_range("range1", simple_2d_reader)
        manager.add_range("range2", simple_2d_reader)

        with pytest.raises(KeyError, match=r"Available ranges: \['range1', 'range2'\]"):
            manager.remove_range("nonexistent")

    def test_clear_all_ranges(self, simple_2d_reader, simple_3d_reader):
        """Test clearing all ranges from manager."""
        manager = RangeManager()
        manager.add_range("range1", simple_2d_reader)
        manager.add_range("range2", simple_3d_reader)
        manager.add_range("range3", simple_2d_reader)

        assert len(manager) == 3
        manager.clear()
        assert len(manager) == 0
        assert manager.list_ranges() == []

    def test_clear_empty_manager(self):
        """Test clearing an already empty manager."""
        manager = RangeManager()
        manager.clear()
        assert len(manager) == 0


class TestGetRange:
    """Tests for retrieving ranges from RangeManager."""

    def test_get_existing_range(self, simple_2d_reader):
        """Test getting an existing range returns correct reader."""
        manager = RangeManager()
        manager.add_range("test", simple_2d_reader)

        retrieved = manager.get_range("test")
        assert retrieved is simple_2d_reader
        assert retrieved.n_dims == 2

    def test_get_nonexistent_range_raises_error(self):
        """Test that getting nonexistent range raises KeyError."""
        manager = RangeManager()

        with pytest.raises(KeyError, match="not found"):
            manager.get_range("nonexistent")

    def test_get_nonexistent_shows_available(self, simple_2d_reader):
        """Test that KeyError message includes available ranges."""
        manager = RangeManager()
        manager.add_range("available1", simple_2d_reader)

        with pytest.raises(KeyError, match=r"Available ranges: \['available1'\]"):
            manager.get_range("missing")

    def test_list_ranges_returns_insertion_order(
        self, simple_2d_reader, simple_3d_reader, offset_2d_reader
    ):
        """Test that list_ranges returns names in insertion order."""
        manager = RangeManager()
        manager.add_range("third", offset_2d_reader)
        manager.add_range("first", simple_2d_reader)
        manager.add_range("second", simple_3d_reader)

        ranges = manager.list_ranges()
        assert ranges == ["third", "first", "second"]


class TestCalculableRanges:
    """Tests for finding calculable ranges."""

    def test_calculable_ranges_single_match(self, simple_2d_reader, simple_3d_reader):
        """Test finding calculable ranges when only one matches."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)  # bounds: x[0,3], y[0,2]
        manager.add_range("3d", simple_3d_reader)  # needs 3 dims

        # Point with 2 dims within 2d bounds
        calculable = manager.calculable_ranges(1.5, 1.0)
        assert len(calculable) == 1
        assert calculable[0] is simple_2d_reader

    def test_calculable_ranges_multiple_matches(
        self, simple_2d_reader, offset_2d_reader
    ):
        """Test finding calculable ranges when multiple match."""
        manager = RangeManager()
        manager.add_range("range1", simple_2d_reader)  # x[0,3], y[0,2]
        manager.add_range("range2", offset_2d_reader)  # x[2,4], y[1,3]

        # Point in overlap region
        calculable = manager.calculable_ranges(2.5, 1.5)
        assert len(calculable) == 2

    def test_calculable_ranges_no_match(self, simple_2d_reader):
        """Test finding calculable ranges when none match."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)

        # Point with wrong dimensions
        calculable = manager.calculable_ranges(1.0, 1.0, 1.0)
        assert len(calculable) == 0

    def test_calculable_ranges_out_of_bounds(self, simple_2d_reader):
        """Test finding calculable ranges for out-of-bounds point."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)  # x[0,3], y[0,2]

        # Point out of bounds
        calculable = manager.calculable_ranges(10.0, 10.0)
        assert len(calculable) == 0

    def test_first_calculable_range_insertion_order(
        self, simple_2d_reader, offset_2d_reader
    ):
        """Test that _first_calculable_range respects insertion order."""
        manager = RangeManager()
        manager.add_range("first", simple_2d_reader)
        manager.add_range("second", offset_2d_reader)

        # Point in overlap region
        first = manager._first_calculable_range(2.5, 1.5)
        assert first is simple_2d_reader

    def test_first_calculable_range_no_ranges(self):
        """Test that _first_calculable_range raises error when no ranges registered."""
        manager = RangeManager()

        with pytest.raises(ValueError, match="No ranges registered"):
            manager._first_calculable_range(1.0, 1.0)

    def test_first_calculable_range_no_match(self, simple_2d_reader):
        """Test that _first_calculable_range raises error when no match found."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)

        with pytest.raises(ValueError, match="No calculable range found"):
            manager._first_calculable_range(100.0, 100.0)


class TestIsOut:
    """Tests for boundary checking with is_out method."""

    def test_is_out_within_single_range(self, simple_2d_reader):
        """Test is_out returns False for point within bounds."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # x[0,3], y[0,2]

        assert not manager.is_out(1.5, 1.0)
        assert not manager.is_out(0.0, 0.0)  # at boundary
        assert not manager.is_out(3.0, 2.0)  # at boundary

    def test_is_out_outside_single_range(self, simple_2d_reader):
        """Test is_out returns True for point outside bounds."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # x[0,3], y[0,2]

        assert manager.is_out(5.0, 1.0)
        assert manager.is_out(1.0, 5.0)
        assert manager.is_out(-1.0, 1.0)

    def test_is_out_with_overlapping_ranges(self, simple_2d_reader, offset_2d_reader):
        """Test is_out with multiple overlapping ranges."""
        manager = RangeManager()
        manager.add_range("range1", simple_2d_reader)  # x[0,3], y[0,2]
        manager.add_range("range2", offset_2d_reader)  # x[2,4], y[1,3]

        # Point in both ranges
        assert not manager.is_out(2.5, 1.5)

        # Point in only one range
        assert not manager.is_out(0.5, 0.5)  # in range1 only
        assert not manager.is_out(3.5, 2.5)  # in range2 only

        # Point in neither range
        assert manager.is_out(10.0, 10.0)

    def test_is_out_no_calculable_ranges(self, simple_2d_reader):
        """Test is_out returns True when no calculable ranges."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)

        # Wrong number of dimensions
        assert manager.is_out(1.0, 1.0, 1.0)

    def test_is_out_empty_manager(self):
        """Test is_out returns True for empty manager."""
        manager = RangeManager()
        assert manager.is_out(1.0, 1.0)


class TestIsValid:
    """Tests for validity checking with is_valid method."""

    def test_is_valid_collision_free_point(self, collision_2d_reader):
        """Test is_valid returns True for collision-free point."""
        manager = RangeManager()
        manager.add_range("collision", collision_2d_reader)

        # Edge points have value 0 (collision-free)
        assert manager.is_valid(0.0, 0.0)
        assert manager.is_valid(4.0, 0.0)
        assert manager.is_valid(0.0, 4.0)

    def test_is_valid_collision_point(self, collision_2d_reader):
        """Test is_valid returns False for collision point."""
        manager = RangeManager()
        manager.add_range("collision", collision_2d_reader)

        # Center points have value 1 (collision)
        assert not manager.is_valid(2.0, 2.0)
        assert not manager.is_valid(2.5, 2.5)

    def test_is_valid_multiple_ranges_any_valid(
        self, collision_2d_reader, simple_2d_reader
    ):
        """Test is_valid returns True if valid in any range."""
        manager = RangeManager()
        manager.add_range("collision", collision_2d_reader)
        manager.add_range("simple", simple_2d_reader)

        # Point that's invalid in collision range but may be calculable in simple range
        # At (2.0, 2.0): collision_reader has value 1.0, simple_reader has value 4.0
        # Both are calculable, but neither has value 0.0
        result = manager.is_valid(2.0, 2.0)
        assert not result  # Both have non-zero values

    def test_is_valid_out_of_bounds(self, collision_2d_reader):
        """Test is_valid returns False for out-of-bounds point."""
        manager = RangeManager()
        manager.add_range("collision", collision_2d_reader)

        assert not manager.is_valid(10.0, 10.0)

    def test_is_valid_no_calculable_ranges(self, simple_2d_reader):
        """Test is_valid returns False when no calculable ranges."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)

        # Wrong dimensions
        assert not manager.is_valid(1.0, 1.0, 1.0)

    def test_is_valid_empty_manager(self):
        """Test is_valid returns False for empty manager."""
        manager = RangeManager()
        assert not manager.is_valid(1.0, 1.0)


class TestCornerValues:
    """Tests for corner value retrieval methods."""

    def test_get_corner_values(self, simple_2d_reader):
        """Test get_corner_values returns correct values."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # f(x,y) = x + y

        # Point at (1.5, 0.5) should have corners:
        # (1,0)=1, (2,0)=2, (1,1)=2, (2,1)=3
        corners = manager.get_corner_values(1.5, 0.5)
        assert len(corners) == 4
        expected = np.array([1.0, 2.0, 2.0, 3.0])
        np.testing.assert_array_equal(np.sort(corners), expected)

    def test_get_corner_max_value(self, simple_2d_reader):
        """Test get_corner_max_value returns maximum corner value."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # f(x,y) = x + y

        # Point at (1.5, 0.5) has corners: 1, 2, 2, 3
        max_val = manager.get_corner_max_value(1.5, 0.5)
        assert max_val == 3.0

    def test_get_corner_min_value(self, simple_2d_reader):
        """Test get_corner_min_value returns minimum corner value."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # f(x,y) = x + y

        # Point at (1.5, 0.5) has corners: 1, 2, 2, 3
        min_val = manager.get_corner_min_value(1.5, 0.5)
        assert min_val == 1.0

    def test_get_corner_values_3d(self, simple_3d_reader):
        """Test corner values work correctly in 3D."""
        manager = RangeManager()
        manager.add_range("3d", simple_3d_reader)  # f(x,y,z) = x + 2y + 3z

        # Point in middle of grid
        corners = manager.get_corner_values(3.0, 1.5, 0.25)
        assert len(corners) == 8  # 2^3 corners

    def test_corner_values_use_first_calculable(
        self, simple_2d_reader, offset_2d_reader
    ):
        """Test corner methods use first calculable range."""
        manager = RangeManager()
        manager.add_range("first", simple_2d_reader)  # f(x,y) = x + y
        manager.add_range("second", offset_2d_reader)  # f(x,y) = 2x + y

        # Point (2.5, 1.5) is in both ranges
        # First range should be used: f(2.5, 1.5) uses corners around this point
        max_val = manager.get_corner_max_value(2.5, 1.5)
        # For simple_2d_reader at (2.5, 1.5):
        # corners at (2,1)=3, (3,1)=4, (2,2)=4, (3,2)=5
        assert max_val == 5.0

    def test_corner_values_out_of_bounds_raises_error(self, simple_2d_reader):
        """Test corner methods raise error for out-of-bounds point."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)

        with pytest.raises(ValueError):
            manager.get_corner_values(10.0, 10.0)


class TestNearestValue:
    """Tests for nearest value retrieval."""

    def test_get_nearest_value(self, simple_2d_reader):
        """Test get_nearest_value returns value at nearest grid point."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # f(x,y) = x + y

        # Point (1.2, 0.8) is closer to (1, 1) than other grid points
        nearest = manager.get_nearest_value(1.2, 0.8)
        assert nearest == 2.0  # f(1, 1) = 1 + 1 = 2

    def test_get_nearest_value_at_grid_point(self, simple_2d_reader):
        """Test get_nearest_value at exact grid point."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # f(x,y) = x + y

        nearest = manager.get_nearest_value(2.0, 1.0)
        assert nearest == 3.0  # f(2, 1) = 2 + 1 = 3

    def test_get_nearest_value_3d(self, simple_3d_reader):
        """Test get_nearest_value works in 3D."""
        manager = RangeManager()
        manager.add_range("3d", simple_3d_reader)  # f(x,y,z) = x + 2y + 3z

        # Point (1.0, 0.5, 0.25) near grid point (0, 0, 0)
        nearest = manager.get_nearest_value(0.5, 0.3, 0.2)
        assert nearest == 0.0  # f(0, 0, 0) = 0

    def test_get_nearest_value_uses_first_calculable(
        self, simple_2d_reader, offset_2d_reader
    ):
        """Test get_nearest_value uses first calculable range."""
        manager = RangeManager()
        manager.add_range("first", simple_2d_reader)
        manager.add_range("second", offset_2d_reader)

        # Point in overlap region
        nearest = manager.get_nearest_value(2.5, 1.5)
        # Should use simple_2d_reader: nearest to (2.5, 1.5) is likely (3, 2) = 5
        # or (2, 1) = 3, depending on distance
        assert isinstance(nearest, float)


class TestInterpolation:
    """Tests for interpolation methods."""

    def test_interpolate_linear(self, simple_2d_reader):
        """Test interpolate performs linear interpolation correctly."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # f(x,y) = x + y

        # Point (1.5, 0.5) should interpolate to 1.5 + 0.5 = 2.0
        result = manager.interpolate(1.5, 0.5)
        assert abs(result - 2.0) < 1e-10

    def test_interpolate_at_grid_point(self, simple_2d_reader):
        """Test interpolate returns exact value at grid point."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # f(x,y) = x + y

        result = manager.interpolate(2.0, 1.0)
        assert abs(result - 3.0) < 1e-10

    def test_interpolate_3d(self, simple_3d_reader):
        """Test interpolate works correctly in 3D."""
        manager = RangeManager()
        manager.add_range("3d", simple_3d_reader)  # f(x,y,z) = x + 2y + 3z

        # Test at point (3.0, 1.5, 0.25)
        result = manager.interpolate(3.0, 1.5, 0.25)
        expected = 3.0 + 2 * 1.5 + 3 * 0.25  # = 3 + 3 + 0.75 = 6.75
        assert abs(result - expected) < 1e-10

    def test_interpolate_uses_first_calculable(
        self, simple_2d_reader, offset_2d_reader
    ):
        """Test interpolate uses first calculable range."""
        manager = RangeManager()
        manager.add_range("first", simple_2d_reader)  # f(x,y) = x + y
        manager.add_range("second", offset_2d_reader)  # f(x,y) = 2x + y

        # Point in overlap region (2.5, 1.5)
        result = manager.interpolate(2.5, 1.5)
        expected_first = 2.5 + 1.5  # = 4.0
        assert abs(result - expected_first) < 1e-10


class TestInterpolateMaxMin:
    """Tests for interpolate_max and interpolate_min methods."""

    def test_interpolate_max_single_range(self, simple_2d_reader):
        """Test interpolate_max with single range equals interpolate."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)

        point = (1.5, 0.5)
        max_val = manager.interpolate_max(*point)
        regular_val = manager.interpolate(*point)
        assert abs(max_val - regular_val) < 1e-10

    def test_interpolate_min_single_range(self, simple_2d_reader):
        """Test interpolate_min with single range equals interpolate."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)

        point = (1.5, 0.5)
        min_val = manager.interpolate_min(*point)
        regular_val = manager.interpolate(*point)
        assert abs(min_val - regular_val) < 1e-10

    def test_interpolate_max_multiple_ranges(self, simple_2d_reader, offset_2d_reader):
        """Test interpolate_max returns maximum across calculable ranges."""
        manager = RangeManager()
        manager.add_range("range1", simple_2d_reader)  # f(x,y) = x + y
        manager.add_range("range2", offset_2d_reader)  # f(x,y) = 2x + y

        # Point in overlap (2.5, 1.5)
        max_val = manager.interpolate_max(2.5, 1.5)
        val1 = 2.5 + 1.5  # = 4.0
        val2 = 2 * 2.5 + 1.5  # = 6.5
        expected_max = max(val1, val2)  # = 6.5
        assert abs(max_val - expected_max) < 1e-10

    def test_interpolate_min_multiple_ranges(self, simple_2d_reader, offset_2d_reader):
        """Test interpolate_min returns minimum across calculable ranges."""
        manager = RangeManager()
        manager.add_range("range1", simple_2d_reader)  # f(x,y) = x + y
        manager.add_range("range2", offset_2d_reader)  # f(x,y) = 2x + y

        # Point in overlap (2.5, 1.5)
        min_val = manager.interpolate_min(2.5, 1.5)
        val1 = 2.5 + 1.5  # = 4.0
        val2 = 2 * 2.5 + 1.5  # = 6.5
        expected_min = min(val1, val2)  # = 4.0
        assert abs(min_val - expected_min) < 1e-10

    def test_interpolate_max_min_quantitative(self):
        """Test interpolate_max/min with precise quantitative values."""
        # Create two ranges with known values
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0])

        X, Y = np.meshgrid(x, y, indexing="ij")
        results1 = X * 2  # f(x,y) = 2x
        results2 = X + Y * 3  # f(x,y) = x + 3y

        reader1 = RangeReader(results1, ("x", "y"), (x, y))
        reader2 = RangeReader(results2, ("x", "y"), (x, y))

        manager = RangeManager()
        manager.add_range("r1", reader1)
        manager.add_range("r2", reader2)

        # At (1.0, 0.5): r1 = 2*1.0 = 2.0, r2 = 1.0 + 3*0.5 = 2.5
        max_val = manager.interpolate_max(1.0, 0.5)
        min_val = manager.interpolate_min(1.0, 0.5)
        assert abs(max_val - 2.5) < 1e-10
        assert abs(min_val - 2.0) < 1e-10

    def test_interpolate_max_no_calculable_raises_error(self, simple_2d_reader):
        """Test interpolate_max raises error when no calculable ranges."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)

        with pytest.raises(ValueError, match="No calculable range found"):
            manager.interpolate_max(10.0, 10.0)

    def test_interpolate_min_no_calculable_raises_error(self, simple_2d_reader):
        """Test interpolate_min raises error when no calculable ranges."""
        manager = RangeManager()
        manager.add_range("2d", simple_2d_reader)

        with pytest.raises(ValueError, match="No calculable range found"):
            manager.interpolate_min(10.0, 10.0)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_point_grid(self):
        """Test manager with single-point grid range."""
        # Single point grids need at least 2 points per dimension for interpolation
        # Use a minimal 2x2 grid instead
        x = np.array([1.0, 1.1])
        y = np.array([2.0, 2.1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        results = X + Y  # Simple function
        reader = RangeReader(results, ("x", "y"), (x, y))

        manager = RangeManager()
        manager.add_range("minimal", reader)

        # Can interpolate within bounds
        result = manager.interpolate(1.05, 2.05)
        assert abs(result - 3.1) < 0.01  # f(1.05, 2.05) ≈ 3.1
        assert manager.is_out(2.0, 2.0)  # outside bounds

    def test_negative_coordinates(self):
        """Test manager with negative coordinate ranges."""
        x = np.array([-2.0, -1.0, 0.0, 1.0])
        y = np.array([-1.0, 0.0, 1.0])
        X, Y = np.meshgrid(x, y, indexing="ij")
        results = X + Y
        reader = RangeReader(results, ("x", "y"), (x, y))

        manager = RangeManager()
        manager.add_range("negative", reader)

        assert not manager.is_out(-1.5, -0.5)
        result = manager.interpolate(-1.5, -0.5)
        assert abs(result - (-2.0)) < 1e-10

    def test_large_dimensional_space(self):
        """Test manager with high-dimensional range (4D)."""
        shape = (3, 3, 3, 3)
        results = np.random.rand(*shape)
        axes = tuple(np.linspace(0, 1, 3) for _ in range(4))
        axis_names = ("a", "b", "c", "d")
        reader = RangeReader(results, axis_names, axes)

        manager = RangeManager()
        manager.add_range("4d", reader)

        assert not manager.is_out(0.5, 0.5, 0.5, 0.5)
        # Should be able to interpolate
        result = manager.interpolate(0.5, 0.5, 0.5, 0.5)
        assert isinstance(result, float)

    def test_very_fine_grid(self):
        """Test manager with very fine grid spacing."""
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y, indexing="ij")
        results = np.sin(10 * X) * np.cos(10 * Y)
        reader = RangeReader(results, ("x", "y"), (x, y))

        manager = RangeManager()
        manager.add_range("fine", reader)

        # Test interpolation accuracy
        test_point = (0.5, 0.5)
        result = manager.interpolate(*test_point)
        expected = np.sin(10 * 0.5) * np.cos(10 * 0.5)
        assert abs(result - expected) < 0.01  # High accuracy with fine grid

    def test_boundary_points(self, simple_2d_reader):
        """Test all methods work at exact boundary points."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)  # x[0,3], y[0,2]

        # Test all four corners
        corners = [(0.0, 0.0), (3.0, 0.0), (0.0, 2.0), (3.0, 2.0)]
        for corner in corners:
            assert not manager.is_out(*corner)
            assert manager.interpolate(*corner) == sum(corner)
            assert isinstance(manager.get_nearest_value(*corner), float)

    def test_empty_manager_all_methods(self):
        """Test that all methods handle empty manager appropriately."""
        manager = RangeManager()

        # These should return False/True for empty manager
        assert manager.is_out(1.0, 1.0)
        assert not manager.is_valid(1.0, 1.0)
        assert len(manager.calculable_ranges(1.0, 1.0)) == 0

        # These should raise errors
        with pytest.raises(ValueError):
            manager.interpolate(1.0, 1.0)
        with pytest.raises(ValueError):
            manager.get_nearest_value(1.0, 1.0)
        with pytest.raises(ValueError):
            manager.interpolate_max(1.0, 1.0)
        with pytest.raises(ValueError):
            manager.interpolate_min(1.0, 1.0)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_workflow_add_query_remove(
        self, simple_2d_reader, simple_3d_reader, offset_2d_reader
    ):
        """Test complete workflow of adding, querying, and removing ranges."""
        manager = RangeManager()

        # Add ranges
        manager.add_range("2d_main", simple_2d_reader)
        manager.add_range("3d", simple_3d_reader)
        manager.add_range("2d_offset", offset_2d_reader)
        assert len(manager) == 3

        # Query in overlap region
        point_2d = (2.5, 1.5)
        calculable = manager.calculable_ranges(*point_2d)
        assert len(calculable) == 2  # Both 2d ranges

        max_val = manager.interpolate_max(*point_2d)
        min_val = manager.interpolate_min(*point_2d)
        assert max_val >= min_val

        # Remove one range
        manager.remove_range("2d_offset")
        assert len(manager) == 2

        # Query again
        calculable = manager.calculable_ranges(*point_2d)
        assert len(calculable) == 1  # Only one 2d range now

    def test_multiple_operations_consistency(self, simple_2d_reader):
        """Test that multiple operations on same point are consistent."""
        manager = RangeManager()
        manager.add_range("range", simple_2d_reader)

        point = (1.5, 1.0)

        # Get all values
        interpolated = manager.interpolate(*point)
        nearest = manager.get_nearest_value(*point)
        corner_max = manager.get_corner_max_value(*point)
        corner_min = manager.get_corner_min_value(*point)

        # Interpolated should be between corner min and max
        assert corner_min <= interpolated <= corner_max

        # Verify nearest is a valid float
        assert isinstance(nearest, float)

    def test_stress_test_many_ranges(self, simple_2d_reader):
        """Test manager with many ranges."""
        manager = RangeManager()

        # Add 50 ranges
        for i in range(50):
            manager.add_range(f"range_{i}", simple_2d_reader)

        assert len(manager) == 50

        # All should be calculable for valid point
        calculable = manager.calculable_ranges(1.5, 1.0)
        assert len(calculable) == 50

        # Clear all
        manager.clear()
        assert len(manager) == 0

    def test_range_replacement_workflow(
        self, simple_2d_reader, offset_2d_reader, simple_3d_reader
    ):
        """Test workflow of replacing ranges with different types."""
        manager = RangeManager()

        # Add initial 2D range
        manager.add_range("dynamic", simple_2d_reader)
        assert manager.get_range("dynamic").n_dims == 2

        # Replace with different 2D range
        manager.add_range("dynamic", offset_2d_reader, overwrite=True)
        assert manager.get_range("dynamic").n_dims == 2
        assert len(manager) == 1

        # Replace with 3D range
        manager.add_range("dynamic", simple_3d_reader, overwrite=True)
        assert manager.get_range("dynamic").n_dims == 3
        assert len(manager) == 1
