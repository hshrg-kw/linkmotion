# type: ignore
import logging
from typing import Literal

import numpy as np

try:
    import k3d
except ImportError:
    raise ImportError("k3d is not installed. Please install it via 'pip install k3d'")

from linkmotion.typing.numpy import Vector3s, RGBA0to1s

logger = logging.getLogger(__name__)


def rgba_to_hex(
    rgba_array_0to1: RGBA0to1s,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    hex_colors = []
    for rgba in rgba_array_0to1:
        hex_string = "0x{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
        )
        hex_color = int(hex_string, 16)
        hex_colors.append(hex_color)
    return np.array(hex_colors)


def _get_or_create_plot(plot: k3d.Plot | None) -> k3d.Plot:
    """Creates a new k3d.Plot if one is not provided.

    This private helper method centralizes the logic for plot object
    instantiation, reducing code duplication across public methods.

    Args:
        plot: An existing k3d.Plot object or None.

    Returns:
        An existing or newly created k3d.Plot object.
    """
    if plot is None:
        logger.info("No k3d.Plot provided, creating a new one.")
        return k3d.plot()
    return plot


class BasicVisualizer:
    """A collection of static methods for 3D visualization using k3d.

    This class acts as a namespace for convenient plotting functions,
    allowing chaining or adding different visualization objects to the same
    k3d plot.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def lines(
        points_array: np.ndarray[
            tuple[int, Literal[2], Literal[3]], np.dtype[np.float64]
        ],
        color: int = 0xFF0000,
        width: float = 0.01,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        """Draws multiple line segments.

        Args:
            points_array: A numpy array of shape (N, 2, 3), where N is the
                number of lines. Each line is defined by a start and end point.
            color: The color of the lines in hexadecimal format.
            width: The width of the lines.
            plot: An optional existing k3d.Plot object to add the lines to.
                If None, a new plot is created.

        Returns:
            The k3d.Plot object containing the lines.

        Raises:
            ValueError: If the input points_array does not have the expected
                3D shape of (N, 2, 3).
        """
        # --- Input Validation ---
        if not (points_array.ndim == 3 and points_array.shape[1:] == (2, 3)):
            raise ValueError(
                f"points_array must have shape (N, 2, 3), but got {points_array.shape}"
            )

        plot = _get_or_create_plot(plot)
        # Add each line segment to the plot individually.
        for points in points_array:
            plot += k3d.line(
                np.ascontiguousarray(points, np.float32), color=color, width=width
            )
        return plot

    @staticmethod
    def vectors(
        origins: Vector3s,
        directions: Vector3s,
        head_size: float = 1.0,
        color: int = 0xFF0000,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        """Draws multiple vectors.

        Args:
            origins: A numpy array of shape (N, 3) representing the starting
                points of the vectors.
            directions: A numpy array of shape (N, 3) representing the
                direction and magnitude of the vectors from their origins.
            head_size: The size of the vector arrowheads.
            color: The color of the vectors in hexadecimal format.
            plot: An optional existing k3d.Plot object to add the vectors to.
                If None, a new plot is created.

        Returns:
            The k3d.Plot object containing the vectors.

        Raises:
            ValueError: If origins and directions have mismatched shapes or
                are not of shape (N, 3).
        """
        # --- Input Validation ---
        if origins.shape != directions.shape:
            raise ValueError(
                f"Mismatched shapes for origins {origins.shape} and directions {directions.shape}"
            )
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError(
                f"origins and directions must have shape (N, 3), but got {origins.shape}"
            )

        plot = _get_or_create_plot(plot)
        plot += k3d.vectors(
            np.ascontiguousarray(origins, np.float32),
            np.ascontiguousarray(directions, np.float32),
            head_size=head_size,
            color=color,
        )
        return plot

    @staticmethod
    def points(
        points: Vector3s,
        point_size: float = 0.1,
        color: int = 0xFF0000,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        """Draws multiple points.

        Args:
            points: A numpy array of shape (N, 3) representing the coordinates
                of the points.
            point_size: The size of the points.
            color: The color of the points in hexadecimal format.
            plot: An optional existing k3d.Plot object to add the points to.
                If None, a new plot is created.

        Returns:
            The k3d.Plot object containing the points.

        Raises:
            ValueError: If the input points array is not of shape (N, 3).
        """
        # --- Input Validation ---
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), but got {points.shape}")

        plot = _get_or_create_plot(plot)
        plot += k3d.points(
            np.ascontiguousarray(points, np.float32), point_size=point_size, color=color
        )
        return plot

    @staticmethod
    def origin(
        point_size: float = 0.1,
        color: int = 0xFF0000,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        """Draws a point at the origin (0, 0, 0).

        This is a convenience method that wraps the points() method.

        Args:
            point_size: The size of the origin point.
            color: The color of the origin point.
            plot: An optional existing k3d.Plot object. If None, a new plot is created.

        Returns:
            The k3d.Plot object containing the origin point.
        """
        # Define the origin as a 3D point in a numpy array of shape (1, 3).
        origin_point = np.array([[0.0, 0.0, 0.0]])
        return BasicVisualizer.points(origin_point, point_size, color, plot)

    @staticmethod
    def axes(
        length: float = 1.0,
        width: float = 0.01,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        """Draws X, Y, and Z axes centered at the origin.

        This is a convenience method that wraps the lines() method.

        Args:
            length: The total length of each axis line.
            width: The width of the axis lines.
            plot: An optional existing k3d.Plot object. If None, a new plot is created.

        Returns:
            The k3d.Plot object containing the axes.
        """
        half_len = length / 2.0
        # Define the start and end points for X, Y, and Z axes.
        axis_lines = np.array(
            [
                [[-half_len, 0, 0], [half_len, 0, 0]],  # X-axis
                [[0, -half_len, 0], [0, half_len, 0]],  # Y-axis
                [[0, 0, -half_len], [0, 0, half_len]],  # Z-axis
            ]
        )
        # Note: Colors for each axis are not differentiated here.
        # For distinct colors, this method would need to call `lines` multiple times.
        return BasicVisualizer.lines(axis_lines, color=0x808080, width=width, plot=plot)
