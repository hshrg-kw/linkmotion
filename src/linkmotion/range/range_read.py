from pathlib import Path
import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator


logger = logging.getLogger(__name__)


class RangeReader:
    """Read and manage range data with optimized interpolation.

    This class provides high-performance interpolation for multi-dimensional range data,
    prioritizing speed through pre-computed bounds and scipy's RegularGridInterpolator.

    Example:
        When axis_names = ('x', 'y', 'z'), results[2, 4, 1] gives the result at
        (x, y, z) = (axis_points[0][2], axis_points[1][4], axis_points[2][1])

    Args:
        results: A numpy array containing the calculation results.
        axis_names: A tuple of strings representing the names of the axes.
        axis_points: A tuple of numpy 1D arrays where each element corresponds to the
            survey grid points for an axis. Each array must be sorted in ascending order.

    Raises:
        ValueError: If dimensions mismatch between results, axis_names, and axis_points,
            or if axis_points arrays are not sorted in ascending order.

    Note:
        All axis_points arrays must be sorted in ascending order for binary search operations.
    """

    def __init__(
        self,
        results: np.ndarray,
        axis_names: tuple[str, ...],
        axis_points: tuple[np.ndarray[tuple[int], np.dtype[np.float64]], ...],
    ):
        # Validate input dimensions
        if len(axis_names) != len(axis_points):
            raise ValueError(
                f"Number of axis names ({len(axis_names)}) must match number of axis_points ({len(axis_points)})"
            )

        if results.ndim != len(axis_names):
            raise ValueError(
                f"Results array dimensions ({results.ndim}) must match number of axes ({len(axis_names)})"
            )

        for i, (points, expected_size) in enumerate(zip(axis_points, results.shape)):
            if len(points) != expected_size:
                raise ValueError(
                    f"Axis {i} ({axis_names[i]}): points array length ({len(points)}) "
                    f"must match results dimension {i} size ({expected_size})"
                )

            # Validate that axis_points are sorted in ascending order
            if len(points) > 1 and not np.all(np.diff(points) > 0):
                raise ValueError(
                    f"Axis {i} ({axis_names[i]}): points array must be strictly sorted in ascending order. "
                    f"Binary search operations require sorted arrays."
                )

        self.results = results
        self.axis_names = axis_names
        self.axis_points = axis_points
        self.n_dims = len(axis_names)

        # Pre-compute bounds
        self._min_bounds = np.array(
            [np.min(points) for points in axis_points], dtype=np.float64
        )
        self._max_bounds = np.array(
            [np.max(points) for points in axis_points], dtype=np.float64
        )

        # Pre-compute grid sizes
        self._grid_sizes = np.array(
            [len(points) for points in axis_points], dtype=np.int32
        )

        # Create scipy interpolator for high-quality interpolation (cached)
        self._interpolator = RegularGridInterpolator(
            axis_points,
            results,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,  # Will return NaN for out-of-bounds
        )

    def __repr__(self) -> str:
        return (
            f"RangeReader(n_dims={self.n_dims}, "
            f"axis_names={self.axis_names}, "
            f"grid_sizes={self._grid_sizes.tolist()})"
        )

    def is_calculable(self, *points: float) -> bool:
        """Check if the points can be used for calculations.

        Points are calculable if the number of coordinates matches the dimensionality
        of the range data and all coordinates are within bounds.

        Args:
            *points: Coordinates to check.

        Returns:
            True if the length matches dimensionality and points are within bounds,
            False otherwise.
        """
        return len(points) == self.n_dims and not self.is_out(*points)

    def is_valid(self, *points: float) -> bool:
        """Check if the given points are valid.

        In the default implementation, points are valid if they are within bounds
        and have an interpolated value of 0.0 (collision-free).

        Note:
            Override this method to customize the definition of "valid" for your use case.

        Args:
            *points: Coordinates to check.

        Returns:
            True if points are within bounds and interpolated value is 0.0, False otherwise.
        """
        return not self.is_out(*points) and self.interpolate(*points) == 0.0

    def is_out(self, *points: float) -> np.bool_:
        """Check if the given points are out of the defined bounds.

        Args:
            *points: Coordinates to check.

        Returns:
            True if any coordinate is out of bounds, False otherwise.
        """
        pts = np.array(points, dtype=np.float64)
        return np.any(pts < self._min_bounds) | np.any(pts > self._max_bounds)

    def is_out_batch(
        self, points_array: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Check if multiple points are out of bounds (batch processing).

        Args:
            points_array: A 2D numpy array of shape (n_points, n_dims) where each row is a point.

        Returns:
            A 1D boolean array of shape (n_points,) where True indicates out of bounds.
        """
        below_min = np.any(points_array < self._min_bounds, axis=1)
        above_max = np.any(points_array > self._max_bounds, axis=1)
        return below_min | above_max

    def _find_cell_indices(
        self, *points: float
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ]:
        """Find the lower and upper cell indices for interpolation.

        Args:
            *points: Coordinates to find cell indices for.

        Returns:
            Tuple of (lower_indices, upper_indices) where each is an array of indices
            for each dimension.
        """
        pts = np.array(points, dtype=np.float64)

        # Find the cell containing the point
        lower_indices = np.zeros(self.n_dims, dtype=np.int32)
        upper_indices = np.zeros(self.n_dims, dtype=np.int32)

        for i in range(self.n_dims):
            # Use searchsorted for fast binary search
            idx = np.searchsorted(self.axis_points[i], pts[i], side="right") - 1
            idx = np.clip(idx, 0, self._grid_sizes[i] - 2)
            lower_indices[i] = idx
            upper_indices[i] = idx + 1

        return lower_indices, upper_indices

    def get_corner_values(
        self, *points: float
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the values of all corner neighbors in the grid cell enclosing the given points.

        Args:
            *points: Coordinates to query.

        Returns:
            Array of shape (2^n_dims,) containing values at all corners of the enclosing cell.

        Raises:
            ValueError: If the point is out of bounds.
        """
        if self.is_out(*points):
            raise ValueError("Point is out of bounds. Extrapolation is not allowed.")

        lower_idx, upper_idx = self._find_cell_indices(*points)

        # Generate all corner indices using binary iteration
        n_corners = 2**self.n_dims
        corner_values = np.zeros(n_corners, dtype=np.float64)

        for corner in range(n_corners):
            # Convert corner number to binary to select lower/upper for each dimension
            indices = []
            for dim in range(self.n_dims):
                if (corner >> dim) & 1:
                    indices.append(upper_idx[dim])
                else:
                    indices.append(lower_idx[dim])
            corner_values[corner] = self.results[tuple(indices)]

        return corner_values

    def get_corner_max_value(self, *points: float) -> float:
        """Get the maximum value among corner neighbors in the grid cell enclosing the given points.

        Args:
            *points: Coordinates to query.

        Returns:
            Maximum value among all corners of the enclosing cell.

        Raises:
            ValueError: If the point is out of bounds.
        """
        return np.max(self.get_corner_values(*points))

    def get_corner_min_value(self, *points: float) -> float:
        """Get the minimum value among corner neighbors in the grid cell enclosing the given points.

        Args:
            *points: Coordinates to query.

        Returns:
            Minimum value among all corners of the enclosing cell.

        Raises:
            ValueError: If the point is out of bounds.
        """
        return np.min(self.get_corner_values(*points))

    def get_nearest_value(self, *points: float) -> float:
        """Get the value at the nearest grid point to the given coordinates.

        Args:
            *points: Coordinates to query.

        Returns:
            Value at the nearest grid point.

        Raises:
            ValueError: If the point is out of bounds.
        """
        if self.is_out(*points):
            raise ValueError("Point is out of bounds. Extrapolation is not allowed.")

        pts = np.array(points, dtype=np.float64)

        # Find nearest index for each dimension
        nearest_indices = np.zeros(self.n_dims, dtype=np.int32)
        for i in range(self.n_dims):
            # Find the closest grid point
            idx = np.searchsorted(self.axis_points[i], pts[i])
            if idx == 0:
                nearest_indices[i] = 0
            elif idx >= self._grid_sizes[i]:
                nearest_indices[i] = self._grid_sizes[i] - 1
            else:
                # Compare distances to idx-1 and idx
                if abs(self.axis_points[i][idx] - pts[i]) < abs(
                    self.axis_points[i][idx - 1] - pts[i]
                ):
                    nearest_indices[i] = idx
                else:
                    nearest_indices[i] = idx - 1

        return float(self.results[tuple(nearest_indices)])

    def interpolate(self, *points: float) -> float:
        """Interpolate the value at the given points using multi-linear interpolation.

        Uses scipy's RegularGridInterpolator for accurate and fast interpolation.

        Args:
            *points: Coordinates to interpolate at.

        Returns:
            Interpolated value.

        Raises:
            ValueError: If the point is out of bounds.
        """
        if self.is_out(*points):
            raise ValueError("Point is out of bounds. Extrapolation is not allowed.")

        # Use pre-built scipy interpolator for best performance
        result = self._interpolator(points)
        return float(result)

    def interpolate_batch(
        self, points_array: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Batch interpolation for multiple points.

        Args:
            points_array: Array of shape (n_points, n_dims) where each row is a point.

        Returns:
            Array of shape (n_points,) with interpolated values.

        Raises:
            ValueError: If any points are out of bounds.
        """
        out_of_bounds = self.is_out_batch(points_array)
        if np.any(out_of_bounds):
            raise ValueError(
                f"{np.sum(out_of_bounds)} points are out of bounds. Extrapolation is not allowed."
            )

        return self._interpolator(points_array)

    @classmethod
    def import_from_file(cls, file_path: Path) -> "RangeReader":
        """Import range data from a saved .npz file.

        Args:
            file_path: Path to the .npz file containing saved range data.

        Returns:
            RangeReader instance with loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If required data keys are missing from the file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"The file '{file_path}' was not found.")

        logger.info(f"Importing calculation results from '{file_path}'...")

        # WARNING: allow_pickle=True can execute arbitrary code from untrusted files.
        # Only use with files from trusted sources. axis_points requires pickle as it is an array of arrays.
        data = np.load(file_path, allow_pickle=True)

        # Check for required keys
        required_keys = {
            "results",
            "axis_names",
            "axis_points",
        }

        if not required_keys.issubset(data.keys()):
            missing_keys = required_keys - set(data.keys())
            raise KeyError(
                f"The file '{file_path}' is missing required data: {missing_keys}"
            )

        results = data["results"]
        axis_names = data["axis_names"]
        axis_points = data["axis_points"]

        instance = cls(results=results, axis_names=axis_names, axis_points=axis_points)

        logger.info("Successfully imported and reconstructed calculation results.")
        return instance

    def plot(self) -> None:
        """Plot the range data using 2D or 3D visualization.

        Raises:
            NotImplementedError: If data has more than 3 dimensions.
        """
        from linkmotion.visual.range import plot_2d, plot_3d

        if len(self.axis_names) == 2:
            axis_points = self.axis_points
            labels = self.axis_names
            plot_2d(
                mesh_grid=self.results,  # pyright: ignore[reportArgumentType]
                x_points=axis_points[0],
                y_points=axis_points[1],
                x_label=labels[0],
                y_label=labels[1],
            )

        elif len(self.axis_names) == 3:
            axis_points = self.axis_points
            labels = self.axis_names
            plot_3d(
                mesh_grid=self.results,  # pyright: ignore[reportArgumentType]
                x_points=axis_points[0],
                y_points=axis_points[1],
                time_points=axis_points[2],
                x_label=labels[0],
                y_label=labels[1],
                time_label=labels[2],
            )

        else:
            raise NotImplementedError(
                "Plotting is only implemented for 2D and 3D data."
            )
