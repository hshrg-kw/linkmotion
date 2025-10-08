import pytest
import numpy as np
from pathlib import Path
import tempfile

from linkmotion.range.range_read import RangeReader


@pytest.fixture
def simple_2d_data():
    """Create simple 2D range data for testing."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    results = X + Y  # Simple linear function
    return results, ("x", "y"), (x, y)


@pytest.fixture
def simple_3d_data():
    """Create simple 3D range data for testing."""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 5, 15)
    z = np.linspace(0, 3, 10)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    results = np.sin(X) * np.cos(Y) * Z
    return results, ("x", "y", "z"), (x, y, z)


@pytest.fixture
def range_reader_2d(simple_2d_data):
    """Create a RangeReader instance with 2D data."""
    results, axis_names, axis_points = simple_2d_data
    return RangeReader(results, axis_names, axis_points)


@pytest.fixture
def range_reader_3d(simple_3d_data):
    """Create a RangeReader instance with 3D data."""
    results, axis_names, axis_points = simple_3d_data
    return RangeReader(results, axis_names, axis_points)


class TestRangeReaderInitialization:
    """Tests for RangeReader initialization and validation."""

    def test_valid_initialization_2d(self, simple_2d_data):
        """Test successful initialization with valid 2D data."""
        results, axis_names, axis_points = simple_2d_data
        reader = RangeReader(results, axis_names, axis_points)

        assert reader.n_dims == 2
        assert reader.axis_names == axis_names
        assert len(reader.axis_points) == 2
        assert reader.results.shape == results.shape

    def test_valid_initialization_3d(self, simple_3d_data):
        """Test successful initialization with valid 3D data."""
        results, axis_names, axis_points = simple_3d_data
        reader = RangeReader(results, axis_names, axis_points)

        assert reader.n_dims == 3
        assert reader.axis_names == axis_names
        assert len(reader.axis_points) == 3

    def test_axis_names_mismatch(self, simple_2d_data):
        """Test that ValueError is raised when axis_names count doesn't match axis_points."""
        results, _, axis_points = simple_2d_data
        wrong_axis_names = ("x",)  # Only one name for two axes

        with pytest.raises(ValueError, match="Number of axis names.*must match"):
            RangeReader(results, wrong_axis_names, axis_points)

    def test_results_dimension_mismatch(self, simple_2d_data):
        """Test that ValueError is raised when results dimensions don't match axes."""
        results, axis_names, axis_points = simple_2d_data
        wrong_results = np.array([1, 2, 3])  # 1D array instead of 2D

        with pytest.raises(ValueError, match="Results array dimensions.*must match"):
            RangeReader(wrong_results, axis_names, axis_points)

    def test_axis_points_size_mismatch(self, simple_2d_data):
        """Test that ValueError is raised when axis_points size doesn't match results shape."""
        results, axis_names, _ = simple_2d_data
        wrong_axis_points = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))  # Wrong sizes

        with pytest.raises(ValueError, match="points array length.*must match"):
            RangeReader(results, axis_names, wrong_axis_points)

    def test_unsorted_axis_points(self, simple_2d_data):
        """Test that ValueError is raised when axis_points are not sorted in ascending order."""
        results, axis_names, axis_points = simple_2d_data
        unsorted_x = np.array([0.0, 2.0, 1.0, 3.0])  # Not sorted
        wrong_axis_points = (unsorted_x, axis_points[1])

        with pytest.raises(
            ValueError, match="must be strictly sorted in ascending order"
        ):
            RangeReader(results, axis_names, wrong_axis_points)

    def test_single_point_axis(self):
        """Test initialization with a single point in an axis (edge case)."""
        results = np.array([[1.0], [2.0]])
        axis_names = ("x", "y")
        axis_points = (np.array([0.0, 1.0]), np.array([0.0]))

        reader = RangeReader(results, axis_names, axis_points)
        assert reader.n_dims == 2
        assert reader._grid_sizes[1] == 1

    def test_repr(self, range_reader_2d):
        """Test __repr__ method."""
        repr_str = repr(range_reader_2d)
        assert "RangeReader" in repr_str
        assert "n_dims=2" in repr_str
        assert "axis_names=('x', 'y')" in repr_str
        assert "grid_sizes" in repr_str


class TestIsOut:
    """Tests for boundary checking methods."""

    def test_is_out_within_bounds_2d(self, range_reader_2d):
        """Test is_out returns False for points within bounds."""
        assert not range_reader_2d.is_out(1.5, 1.0)
        assert not range_reader_2d.is_out(0.0, 0.0)  # At minimum
        assert not range_reader_2d.is_out(3.0, 2.0)  # At maximum

    def test_is_out_outside_bounds_2d(self, range_reader_2d):
        """Test is_out returns True for points outside bounds."""
        assert range_reader_2d.is_out(-0.1, 1.0)  # Below x minimum
        assert range_reader_2d.is_out(3.1, 1.0)  # Above x maximum
        assert range_reader_2d.is_out(1.0, -0.1)  # Below y minimum
        assert range_reader_2d.is_out(1.0, 2.1)  # Above y maximum

    def test_is_out_within_bounds_3d(self, range_reader_3d):
        """Test is_out returns False for points within bounds in 3D."""
        assert not range_reader_3d.is_out(5.0, 2.5, 1.5)
        assert not range_reader_3d.is_out(0.0, 0.0, 0.0)
        assert not range_reader_3d.is_out(10.0, 5.0, 3.0)

    def test_is_out_outside_bounds_3d(self, range_reader_3d):
        """Test is_out returns True for points outside bounds in 3D."""
        assert range_reader_3d.is_out(-0.1, 2.5, 1.5)
        assert range_reader_3d.is_out(5.0, 5.1, 1.5)
        assert range_reader_3d.is_out(5.0, 2.5, 3.1)

    def test_is_out_batch_within_bounds(self, range_reader_2d):
        """Test is_out_batch returns False for all points within bounds."""
        points = np.array([[0.5, 0.5], [1.5, 1.0], [2.5, 1.5]])
        result = range_reader_2d.is_out_batch(points)
        assert not np.any(result)
        assert result.shape == (3,)

    def test_is_out_batch_outside_bounds(self, range_reader_2d):
        """Test is_out_batch returns True for points outside bounds."""
        points = np.array(
            [
                [0.5, 0.5],  # Inside
                [-0.1, 1.0],  # Outside
                [1.5, 1.0],  # Inside
                [3.5, 1.0],  # Outside
            ]
        )
        result = range_reader_2d.is_out_batch(points)
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_is_out_batch_3d(self, range_reader_3d):
        """Test is_out_batch with 3D data."""
        points = np.array(
            [
                [5.0, 2.5, 1.5],  # Inside
                [-0.1, 2.5, 1.5],  # Outside
                [5.0, 5.1, 1.5],  # Outside
                [10.0, 5.0, 3.0],  # Inside (at boundary)
            ]
        )
        result = range_reader_3d.is_out_batch(points)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)


class TestFindCellIndices:
    """Tests for _find_cell_indices method."""

    def test_find_cell_indices_2d(self, range_reader_2d):
        """Test finding cell indices for 2D data."""
        lower, upper = range_reader_2d._find_cell_indices(1.5, 0.5)

        # Point at (1.5, 0.5) should be in cell between indices [1,0] and [2,1]
        assert lower[0] == 1  # x: between 1.0 and 2.0
        assert upper[0] == 2
        assert lower[1] == 0  # y: between 0.0 and 1.0
        assert upper[1] == 1

    def test_find_cell_indices_at_boundary(self, range_reader_2d):
        """Test finding cell indices at exact grid points."""
        lower, upper = range_reader_2d._find_cell_indices(1.0, 1.0)

        # At exact grid point, searchsorted finds the next index
        # The function clips to valid cell range
        assert lower[0] >= 0
        assert upper[0] == lower[0] + 1
        assert lower[1] >= 0
        assert upper[1] == lower[1] + 1

    def test_find_cell_indices_3d(self, range_reader_3d):
        """Test finding cell indices for 3D data."""
        lower, upper = range_reader_3d._find_cell_indices(5.5, 2.3, 1.7)

        assert len(lower) == 3
        assert len(upper) == 3
        assert np.all(upper == lower + 1)


class TestGetCornerValues:
    """Tests for corner value retrieval methods."""

    def test_get_corner_values_2d(self, range_reader_2d):
        """Test getting corner values for 2D data with exact value verification.

        For point (1.5, 0.5) in grid where f(x,y) = x + y:
        - Cell corners are at (1,0), (2,0), (1,1), (2,1)
        - Expected values: [1, 2, 2, 3]
        """
        corner_values = range_reader_2d.get_corner_values(1.5, 0.5)

        # Should have 2^2 = 4 corners
        assert len(corner_values) == 4
        assert corner_values.dtype == np.float64

        # Verify exact corner values: corners at (1,0), (2,0), (1,1), (2,1)
        expected_values = np.array([1.0, 2.0, 2.0, 3.0])
        np.testing.assert_array_equal(np.sort(corner_values), np.sort(expected_values))

    def test_get_corner_values_3d(self, range_reader_3d):
        """Test getting corner values for 3D data with mathematical consistency check."""
        test_point = (5.0, 2.5, 1.5)
        corner_values = range_reader_3d.get_corner_values(*test_point)

        # Should have 2^3 = 8 corners
        assert len(corner_values) == 8
        assert corner_values.dtype == np.float64

        # For f(x,y,z) = sin(x)*cos(y)*z, verify values are reasonable
        # All corner values should be finite and within expected range
        assert np.all(np.isfinite(corner_values))
        # Given the function, max absolute value should be bounded by max(|z|)
        assert np.all(np.abs(corner_values) <= 3.0)

    def test_get_corner_values_out_of_bounds(self, range_reader_2d):
        """Test that get_corner_values raises error for out-of-bounds points."""
        with pytest.raises(ValueError, match="Point is out of bounds"):
            range_reader_2d.get_corner_values(-1.0, 1.0)

    def test_get_corner_max_value(self, range_reader_2d):
        """Test getting maximum corner value with exact verification.

        For point (1.5, 0.5) with corners at (1,0), (2,0), (1,1), (2,1):
        Maximum value should be at corner (2,1) = 3.0
        """
        max_value = range_reader_2d.get_corner_max_value(1.5, 0.5)
        corner_values = range_reader_2d.get_corner_values(1.5, 0.5)

        assert max_value == np.max(corner_values)
        assert isinstance(max_value, (float, np.floating))
        # Verify exact maximum value
        assert max_value == 3.0

    def test_get_corner_min_value(self, range_reader_2d):
        """Test getting minimum corner value with exact verification.

        For point (1.5, 0.5) with corners at (1,0), (2,0), (1,1), (2,1):
        Minimum value should be at corner (1,0) = 1.0
        """
        min_value = range_reader_2d.get_corner_min_value(1.5, 0.5)
        corner_values = range_reader_2d.get_corner_values(1.5, 0.5)

        assert min_value == np.min(corner_values)
        assert isinstance(min_value, (float, np.floating))
        # Verify exact minimum value
        assert min_value == 1.0

    def test_corner_min_max_relationship(self, range_reader_2d):
        """Test that min is less than or equal to max with quantitative bounds.

        For point (1.5, 0.5): min=1.0, max=3.0, difference=2.0
        """
        min_val = range_reader_2d.get_corner_min_value(1.5, 0.5)
        max_val = range_reader_2d.get_corner_max_value(1.5, 0.5)

        assert min_val <= max_val
        # Verify exact values
        assert min_val == 1.0
        assert max_val == 3.0
        assert max_val - min_val == 2.0


class TestGetNearestValue:
    """Tests for nearest neighbor interpolation."""

    def test_get_nearest_value_2d(self, range_reader_2d):
        """Test getting nearest value for 2D data with exact verification.

        Point (1.4, 0.4) is closest to grid point (1.0, 0.0).
        Distance to (1,0): sqrt(0.4^2 + 0.4^2) = 0.566
        Distance to (2,0): sqrt(0.6^2 + 0.4^2) = 0.721
        Distance to (1,1): sqrt(0.4^2 + 0.6^2) = 0.721
        Expected value: f(1,0) = 1.0
        """
        value = range_reader_2d.get_nearest_value(1.4, 0.4)

        # Should return value at (1.0, 0.0) which is 1.0 + 0.0 = 1.0
        assert isinstance(value, float)
        assert value == 1.0

    def test_get_nearest_value_exact_point(self, range_reader_2d):
        """Test getting nearest value at exact grid point.

        At exact point (2.0, 1.0): f(2,1) = 2 + 1 = 3.0
        """
        value = range_reader_2d.get_nearest_value(2.0, 1.0)

        # At exact point (2.0, 1.0), value should be 2.0 + 1.0 = 3.0
        assert value == 3.0

    def test_get_nearest_value_3d(self, range_reader_3d):
        """Test getting nearest value for 3D data with consistency check.

        For f(x,y,z) = sin(x)*cos(y)*z at (5.5, 2.3, 1.7):
        Result should be mathematically consistent with the function.
        """
        test_point = (5.5, 2.3, 1.7)
        value = range_reader_3d.get_nearest_value(*test_point)

        assert isinstance(value, float)
        assert np.isfinite(value)
        # For this function, value should be bounded
        assert abs(value) <= 3.0

        # Verify consistency: get the nearest grid point manually
        x_idx = np.argmin(np.abs(range_reader_3d.axis_points[0] - test_point[0]))
        y_idx = np.argmin(np.abs(range_reader_3d.axis_points[1] - test_point[1]))
        z_idx = np.argmin(np.abs(range_reader_3d.axis_points[2] - test_point[2]))
        expected = range_reader_3d.results[x_idx, y_idx, z_idx]
        assert value == expected

    def test_get_nearest_value_out_of_bounds(self, range_reader_2d):
        """Test that get_nearest_value raises error for out-of-bounds points."""
        with pytest.raises(ValueError, match="Point is out of bounds"):
            range_reader_2d.get_nearest_value(-1.0, 1.0)

    def test_get_nearest_value_at_boundaries(self, range_reader_2d):
        """Test getting nearest value at domain boundaries with exact values.

        Grid: x=[0,1,2,3], y=[0,1,2]
        f(x,y) = x + y
        """
        # At minimum boundary: f(0,0) = 0 + 0 = 0
        value_min = range_reader_2d.get_nearest_value(0.0, 0.0)
        assert value_min == 0.0

        # At maximum boundary: f(3,2) = 3 + 2 = 5
        value_max = range_reader_2d.get_nearest_value(3.0, 2.0)
        assert value_max == 5.0

        # At corner boundaries
        assert range_reader_2d.get_nearest_value(0.0, 2.0) == 2.0  # f(0,2)
        assert range_reader_2d.get_nearest_value(3.0, 0.0) == 3.0  # f(3,0)


class TestInterpolate:
    """Tests for linear interpolation."""

    def test_interpolate_2d(self, range_reader_2d):
        """Test linear interpolation for 2D data with exact mathematical verification.

        For linear function f(x,y) = x + y at point (1.5, 0.5):
        Since f is linear, bilinear interpolation should give exact result.
        Expected: 1.5 + 0.5 = 2.0
        """
        value = range_reader_2d.interpolate(1.5, 0.5)

        # For linear function f(x,y) = x + y at (1.5, 0.5)
        expected = 1.5 + 0.5
        assert np.isclose(value, expected, rtol=1e-10)
        # Exact comparison for linear function
        assert abs(value - 2.0) < 1e-10

    def test_interpolate_at_grid_point(self, range_reader_2d):
        """Test interpolation at exact grid point should return exact value.

        At grid point (2.0, 1.0): f(2,1) = 2 + 1 = 3.0 exactly
        """
        value = range_reader_2d.interpolate(2.0, 1.0)

        # At exact point (2.0, 1.0), value should be 2.0 + 1.0 = 3.0
        assert np.isclose(value, 3.0, rtol=1e-10)
        assert value == 3.0

    def test_interpolate_3d(self, range_reader_3d):
        """Test linear interpolation for 3D data with analytical verification.

        For f(x,y,z) = sin(x)*cos(y)*z at (5.0, 2.5, 1.5):
        At exact grid point, interpolation should equal function value.
        """
        # Test at exact grid point
        x_val = range_reader_3d.axis_points[0][10]  # Middle of x range
        y_val = range_reader_3d.axis_points[1][7]
        z_val = range_reader_3d.axis_points[2][5]

        value = range_reader_3d.interpolate(x_val, y_val, z_val)
        expected = np.sin(x_val) * np.cos(y_val) * z_val

        assert isinstance(value, float)
        assert not np.isnan(value)
        assert np.isclose(value, expected, rtol=1e-10)

    def test_interpolate_out_of_bounds(self, range_reader_2d):
        """Test that interpolate raises error for out-of-bounds points."""
        with pytest.raises(ValueError, match="Point is out of bounds"):
            range_reader_2d.interpolate(-1.0, 1.0)

        with pytest.raises(ValueError, match="Point is out of bounds"):
            range_reader_2d.interpolate(3.5, 1.0)

    def test_interpolate_batch_2d(self, range_reader_2d):
        """Test batch interpolation for 2D data with exact values.

        Test points and expected values for f(x,y) = x + y:
        - (1.5, 0.5) → 2.0
        - (2.0, 1.0) → 3.0
        - (0.5, 0.5) → 1.0
        """
        points = np.array([[1.5, 0.5], [2.0, 1.0], [0.5, 0.5]])
        values = range_reader_2d.interpolate_batch(points)

        assert values.shape == (3,)
        expected = np.array([2.0, 3.0, 1.0])
        np.testing.assert_allclose(values, expected, rtol=1e-10)

        # Verify each value individually
        assert abs(values[0] - 2.0) < 1e-10
        assert abs(values[1] - 3.0) < 1e-10
        assert abs(values[2] - 1.0) < 1e-10

    def test_interpolate_batch_3d(self, range_reader_3d):
        """Test batch interpolation for 3D data with consistency verification.

        Verify that batch and single interpolation give same results.
        """
        points = np.array([[5.0, 2.5, 1.5], [7.5, 3.0, 2.0], [2.5, 1.0, 0.5]])
        batch_values = range_reader_3d.interpolate_batch(points)

        assert batch_values.shape == (3,)
        assert not np.any(np.isnan(batch_values))

        # Verify consistency with single interpolation
        for i, point in enumerate(points):
            single_value = range_reader_3d.interpolate(*point)
            assert np.isclose(batch_values[i], single_value, rtol=1e-10)

    def test_interpolate_batch_out_of_bounds(self, range_reader_2d):
        """Test that batch interpolation raises error when any point is out of bounds.

        Test with 3 points where second point is outside domain.
        """
        points = np.array(
            [
                [1.5, 0.5],  # Inside: x∈[0,3], y∈[0,2]
                [-1.0, 1.0],  # Outside: x=-1 < 0
                [0.5, 0.5],  # Inside
            ]
        )

        with pytest.raises(ValueError, match="1 points are out of bounds"):
            range_reader_2d.interpolate_batch(points)

    def test_interpolate_batch_large_array(self, range_reader_3d):
        """Test batch interpolation with large array and statistical verification.

        For f(x,y,z) = sin(x)*cos(y)*z over domain [0,10]×[0,5]×[0,3]:
        - All values should be finite
        - Mean should be near 0 (due to sin oscillation)
        - Max absolute value bounded by 3.0
        """
        n_points = 1000
        np.random.seed(42)  # For reproducibility
        points = np.random.uniform(
            [0.0, 0.0, 0.0], [10.0, 5.0, 3.0], size=(n_points, 3)
        )
        values = range_reader_3d.interpolate_batch(points)

        assert values.shape == (n_points,)
        assert not np.any(np.isnan(values))
        assert np.all(np.isfinite(values))

        # Statistical checks
        assert np.all(np.abs(values) <= 3.0)  # Bounded by max |z|
        assert abs(np.mean(values)) < 0.5  # Mean near 0 due to sin oscillation
        assert np.std(values) > 0.1  # Non-trivial variation


class TestImportExport:
    """Tests for file import/export functionality."""

    def test_import_from_file(self, simple_2d_data):
        """Test importing range data from file."""
        results, axis_names, axis_points = simple_2d_data

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Use allow_pickle=True for axis_points which is a tuple of arrays
            np.savez(
                tmp_path,
                results=results,
                axis_names=axis_names,
                axis_points=np.array(axis_points, dtype=object),
            )

        try:
            reader = RangeReader.import_from_file(tmp_path)

            assert reader.n_dims == 2
            # axis_names might be loaded as tuple or array, compare elements
            assert len(reader.axis_names) == len(axis_names)
            for i, name in enumerate(axis_names):
                assert reader.axis_names[i] == name
            np.testing.assert_array_equal(reader.results, results)
        finally:
            tmp_path.unlink()

    def test_import_from_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        non_existent = Path("/tmp/nonexistent_file_12345.npz")

        with pytest.raises(FileNotFoundError, match="was not found"):
            RangeReader.import_from_file(non_existent)

    def test_import_from_file_missing_keys(self):
        """Test that KeyError is raised when required keys are missing."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Save with missing 'axis_points' key
            np.savez(
                tmp_path, results=np.array([[1, 2], [3, 4]]), axis_names=("x", "y")
            )

        try:
            with pytest.raises(KeyError, match="missing required data"):
                RangeReader.import_from_file(tmp_path)
        finally:
            tmp_path.unlink()

    def test_import_export_roundtrip(self, simple_3d_data):
        """Test that data can be exported and imported without loss."""
        results, axis_names, axis_points = simple_3d_data
        original_reader = RangeReader(results, axis_names, axis_points)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Use allow_pickle=True for axis_points which is a tuple of arrays
            np.savez(
                tmp_path,
                results=original_reader.results,
                axis_names=original_reader.axis_names,
                axis_points=np.array(original_reader.axis_points, dtype=object),
            )

        try:
            imported_reader = RangeReader.import_from_file(tmp_path)

            assert imported_reader.n_dims == original_reader.n_dims
            # axis_names might be loaded as tuple or array, compare elements
            assert len(imported_reader.axis_names) == len(original_reader.axis_names)
            for i in range(len(original_reader.axis_names)):
                assert imported_reader.axis_names[i] == original_reader.axis_names[i]
            np.testing.assert_array_equal(
                imported_reader.results, original_reader.results
            )

            # Test that interpolation works the same
            test_point = (5.5, 2.3, 1.7)
            original_value = original_reader.interpolate(*test_point)
            imported_value = imported_reader.interpolate(*test_point)
            assert np.isclose(original_value, imported_value)
        finally:
            tmp_path.unlink()


class TestPlot:
    """Tests for plotting functionality."""

    def test_plot_2d_not_implemented_check(self, range_reader_2d):
        """Test that plot method exists for 2D data."""
        # We can't easily test plotting output, but we can verify the method doesn't crash
        # This would require mocking matplotlib, so we just check it's callable
        assert callable(range_reader_2d.plot)

    def test_plot_3d_not_implemented_check(self, range_reader_3d):
        """Test that plot method exists for 3D data."""
        assert callable(range_reader_3d.plot)

    def test_plot_4d_raises_error(self):
        """Test that plot raises NotImplementedError for 4D data."""
        # Create 4D data
        results = np.random.rand(5, 4, 3, 2)
        axis_names = ("x", "y", "z", "w")
        axis_points = (
            np.linspace(0, 1, 5),
            np.linspace(0, 1, 4),
            np.linspace(0, 1, 3),
            np.linspace(0, 1, 2),
        )
        reader = RangeReader(results, axis_names, axis_points)

        # The error could be NotImplementedError or ModuleNotFoundError if plotly is not installed
        with pytest.raises((NotImplementedError, ModuleNotFoundError)):
            reader.plot()


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_cell_grid(self):
        """Test with a 2x2 grid (single cell) with bilinear interpolation.

        Grid: [[1, 2], [3, 4]] at points (0,0), (1,0), (0,1), (1,1)
        At center (0.5, 0.5): bilinear interpolation gives (1+2+3+4)/4 = 2.5
        """
        results = np.array([[1.0, 2.0], [3.0, 4.0]])
        axis_names = ("x", "y")
        axis_points = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        reader = RangeReader(results, axis_names, axis_points)

        # Test interpolation in the middle
        value = reader.interpolate(0.5, 0.5)
        assert isinstance(value, float)
        # Bilinear interpolation: (1+2+3+4)/4 = 2.5
        assert np.isclose(value, 2.5, rtol=1e-10)

        # Test corners
        assert reader.interpolate(0.0, 0.0) == 1.0
        assert reader.interpolate(1.0, 0.0) == 3.0
        assert reader.interpolate(0.0, 1.0) == 2.0
        assert reader.interpolate(1.0, 1.0) == 4.0

        # Test edges (midpoints)
        assert np.isclose(reader.interpolate(0.5, 0.0), 2.0, rtol=1e-10)  # (1+3)/2
        assert np.isclose(reader.interpolate(0.5, 1.0), 3.0, rtol=1e-10)  # (2+4)/2

    def test_uniform_results(self):
        """Test with uniform (constant) results across entire domain.

        For constant function f(x,y,z) = 1.0:
        - All interpolations should return 1.0
        - All corner values should be 1.0
        - Min and max should both be 1.0
        """
        results = np.ones((5, 4, 3))
        axis_names = ("x", "y", "z")
        axis_points = (np.linspace(0, 1, 5), np.linspace(0, 1, 4), np.linspace(0, 1, 3))

        reader = RangeReader(results, axis_names, axis_points)

        # Test multiple random points
        test_points = [
            (0.5, 0.5, 0.5),
            (0.1, 0.2, 0.3),
            (0.9, 0.8, 0.7),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ]
        for point in test_points:
            value = reader.interpolate(*point)
            assert np.isclose(value, 1.0, rtol=1e-10)
            assert reader.get_corner_min_value(*point) == 1.0
            assert reader.get_corner_max_value(*point) == 1.0
            assert reader.get_nearest_value(*point) == 1.0

    def test_high_dimensional_data(self):
        """Test with 5D data (valid but not plottable).

        Create 5D test function: f(a,b,c,d,e) = a + 2b + 3c + 4d + 5e
        Verify interpolation at known point.
        """
        axis_names = ("a", "b", "c", "d", "e")
        axis_points = tuple(np.linspace(0, 1, 3) for _ in range(5))

        # Create meshgrid for 5D
        grids = np.meshgrid(*axis_points, indexing="ij")
        results = grids[0] + 2 * grids[1] + 3 * grids[2] + 4 * grids[3] + 5 * grids[4]

        reader = RangeReader(results, axis_names, axis_points)

        assert reader.n_dims == 5
        assert reader._grid_sizes.tolist() == [3, 3, 3, 3, 3]

        # Test at exact grid point: (0.5, 0.5, 0.5, 0.5, 0.5)
        test_point = (0.5, 0.5, 0.5, 0.5, 0.5)
        value = reader.interpolate(*test_point)
        expected = 0.5 + 2 * 0.5 + 3 * 0.5 + 4 * 0.5 + 5 * 0.5  # = 7.5
        assert isinstance(value, float)
        assert not np.isnan(value)
        assert np.isclose(value, expected, rtol=1e-10)

    def test_very_fine_grid(self):
        """Test with a very fine grid for high interpolation accuracy.

        For quadratic function f(x,y) = x² + y²:
        With 100x100 grid, interpolation error should be < 1%
        """
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 5, 100)
        X, Y = np.meshgrid(x, y, indexing="ij")
        results = X**2 + Y**2

        reader = RangeReader(results, ("x", "y"), (x, y))

        # Test multiple points
        test_cases = [
            (5.5, 2.5, 5.5**2 + 2.5**2),
            (3.14, 1.59, 3.14**2 + 1.59**2),
            (7.25, 4.33, 7.25**2 + 4.33**2),
        ]

        for test_x, test_y, expected in test_cases:
            value = reader.interpolate(test_x, test_y)
            relative_error = abs(value - expected) / expected
            # With fine grid, error should be very small
            assert relative_error < 1e-2, (
                f"Error {relative_error} too large at ({test_x}, {test_y})"
            )
            assert np.isclose(value, expected, rtol=1e-2)

    def test_negative_coordinates(self):
        """Test with negative coordinate values.

        For function f(x,y) = x * y over domain [-2,2] × [-1,1]:
        Test symmetry properties and exact values.
        """
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = np.array([-1.0, 0.0, 1.0])
        X, Y = np.meshgrid(x, y, indexing="ij")
        results = X * Y

        reader = RangeReader(results, ("x", "y"), (x, y))

        # Test exact values at grid points
        assert reader.interpolate(0.0, 0.0) == 0.0
        assert reader.interpolate(-1.0, -1.0) == 1.0
        assert reader.interpolate(1.0, 1.0) == 1.0
        assert reader.interpolate(-1.0, 1.0) == -1.0
        assert reader.interpolate(1.0, -1.0) == -1.0
        assert reader.interpolate(-2.0, 1.0) == -2.0
        assert reader.interpolate(2.0, -1.0) == -2.0

        # Test interpolated value: f(-0.5, -0.5) = 0.25
        value = reader.interpolate(-0.5, -0.5)
        assert isinstance(value, float)
        assert np.isclose(value, 0.25, rtol=1e-10)

        # Test symmetry: f(x,y) = f(y,x) and f(-x,-y) = f(x,y)
        assert np.isclose(
            reader.interpolate(-0.5, -0.5), reader.interpolate(0.5, 0.5), rtol=1e-10
        )

    def test_interpolation_accuracy_convergence(self):
        """Test that interpolation accuracy improves with grid refinement.

        For f(x,y) = sin(πx)*cos(πy), test that finer grids produce small errors.
        Error should generally decrease, and final error should be very small.
        """

        def true_func(x, y):
            return np.sin(np.pi * x) * np.cos(np.pi * y)

        test_point = (0.333, 0.667)
        true_value = true_func(*test_point)

        errors = []
        grid_sizes = [5, 10, 20, 40, 80]

        for n in grid_sizes:
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            X, Y = np.meshgrid(x, y, indexing="ij")
            results = true_func(X, Y)

            reader = RangeReader(results, ("x", "y"), (x, y))
            interpolated = reader.interpolate(*test_point)
            error = abs(interpolated - true_value)
            errors.append(error)

        # With finer grids, error should generally be small
        # The last error should be smallest
        assert errors[-1] < 1e-3, f"Final error {errors[-1]} too large"
        assert errors[0] > errors[-1], (
            f"Error did not decrease from coarsest to finest grid: {errors[0]} vs {errors[-1]}"
        )

        # Average of last 3 errors should be better than average of first 2
        assert np.mean(errors[-3:]) < np.mean(errors[:2])
