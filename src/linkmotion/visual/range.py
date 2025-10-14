"""Range and workspace visualization utilities using Plotly.

This module provides visualization functions for displaying robot workspace
and reachability analysis results. It supports 2D surface plots, 3D animated
plots with time sliders, and N-dimensional plots with conditional slicing.
These visualizations are particularly useful for analyzing robot motion ranges,
workspace boundaries, and time-varying metrics.
"""

import plotly.graph_objects as go
import numpy as np


def plot_2d(
    mesh_grid: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    x_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    title: str = "2D Contour Plot",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    z_label: str = "Z Axis",
    z_min: float | None = None,
    z_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Creates a 2D surface plot for visualizing scalar fields over a 2D domain.

    This function generates an interactive 3D surface plot where the X and Y
    axes represent the domain coordinates and the Z axis represents a scalar
    value (e.g., reachability metric, distance, performance indicator).

    Args:
        mesh_grid: 2D array of Z values with shape (len(x_points), len(y_points)).
        x_points: 1D array of X coordinate values.
        y_points: 1D array of Y coordinate values.
        title: Title for the plot.
        x_label: Label for the X axis.
        y_label: Label for the Y axis.
        z_label: Label for the Z axis (also used for colorbar).
        z_min: Minimum value for Z axis and color scale.
        z_max: Maximum value for Z axis and color scale.
        x_min: Minimum value for X axis range.
        x_max: Maximum value for X axis range.
        y_min: Minimum value for Y axis range.
        y_max: Maximum value for Y axis range.

    Note:
        The function displays the plot immediately using Plotly's show() method,
        which requires an appropriate display environment (browser, Jupyter, etc.).
    """
    # Create meshgrid for surface coordinates
    x, y = np.meshgrid(x_points, y_points)

    # Create surface trace with blue colorscale
    trace = go.Surface(
        x=x,
        y=y,
        z=mesh_grid.transpose(),
        colorscale="Blues",
        colorbar=dict(title=z_label),
        cmin=z_min,
        cmax=z_max,
    )

    # Create figure and configure layout
    fig = go.Figure(data=[trace])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max],
            zaxis_range=[z_min, z_max],
        ),
    )

    fig.show()


def plot_3d(
    mesh_grid: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    x_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    time_points: np.ndarray[tuple[int], np.dtype[np.float64]],
    title: str = "3D Contour Plot",
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    z_label: str = "Z Axis",
    time_label: str = "Time Axis",
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
):
    """Creates an animated 3D plot with a time slider for temporal analysis.

    This function generates an interactive 3D surface plot with a time slider,
    allowing visualization of how a scalar field over a 2D domain changes over
    time. This is particularly useful for analyzing time-varying workspace metrics,
    dynamic reachability, or trajectory analysis.

    Args:
        mesh_grid: 3D array of Z values with shape
            (len(x_points), len(y_points), len(time_points)).
        x_points: 1D array of X coordinate values.
        y_points: 1D array of Y coordinate values.
        time_points: 1D array of time values for the animation.
        title: Title for the plot.
        x_label: Label for the X axis.
        y_label: Label for the Y axis.
        z_label: Label for the Z axis (surface height).
        time_label: Label for the time dimension (slider).
        x_min: Minimum value for X axis range.
        x_max: Maximum value for X axis range.
        y_min: Minimum value for Y axis range.
        y_max: Maximum value for Y axis range.
        z_min: Minimum value for Z axis and color scale.
        z_max: Maximum value for Z axis and color scale.

    Note:
        The function displays the plot immediately with an interactive slider
        for stepping through time. Requires an appropriate display environment
        (browser, Jupyter, etc.).
    """
    # Create meshgrid for surface coordinates
    x, y = np.meshgrid(x_points, y_points)

    # Prepare data for animation - transpose to get time as first dimension
    time_steps = len(time_points)
    z_data = np.transpose(mesh_grid, (2, 1, 0))

    # Create a figure with one surface trace per time step
    fig = go.Figure()
    for t, z in enumerate(z_data):
        fig.add_trace(
            go.Surface(
                z=z,
                x=x,
                y=y,
                visible=(t == 0),  # Only first time step visible initially
                showscale=False,
                cmin=z_min,
                cmax=z_max,
            )
        )

    # Create slider steps for time navigation
    steps = []
    for t in range(time_steps):
        step = dict(
            method="update",
            args=[{"visible": [False] * time_steps}, {"title": f"{time_label}: {t}"}],
            label=f"{time_points[t]:.1f}",
        )
        # Make only the current time step visible
        step["args"][0]["visible"][t] = True  # pyright: ignore[reportArgumentType, reportIndexIssue]
        steps.append(step)

    # Configure the time slider
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": f"{time_label}: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    # Update layout with slider and axis configuration
    fig.update_layout(
        title=title,
        sliders=sliders,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            zaxis=dict(range=(z_min, z_max)),
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max],
            zaxis_range=[z_min, z_max],
        ),
        width=1000,
        scene_aspectmode="cube",
    )

    fig.show()


def plot_nd(
    mesh_grid: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    points_array: tuple[np.ndarray[tuple[int], np.dtype[np.float64]], ...],
    axis_labels: tuple[str, ...],
    axis_ranges: tuple[tuple[float | None, float | None], ...] | None = None,
    title: str = "N-D Contour Plot",
    **conditional_kwargs: int,
):
    """Plots N-dimensional data by conditionally slicing to 2D or 3D.

    This function enables visualization of high-dimensional workspace or
    performance data by allowing the user to fix specific dimensions at
    certain indices, effectively reducing the data to 2D or 3D for plotting.
    This is useful for exploring multi-joint robot configurations or
    parameter spaces.

    The function automatically selects between plot_2d() and plot_3d()
    based on the remaining dimensions after slicing.

    Args:
        mesh_grid: N-dimensional array containing the scalar field values.
        points_array: Tuple of 1D arrays, one for each dimension, containing
            the coordinate values along that dimension.
        axis_labels: Tuple of string labels for each dimension.
        axis_ranges: Optional tuple of (min, max) pairs for each dimension's
            display range.
        title: Title for the plot.
        **conditional_kwargs: Keyword arguments specifying which dimensions
            to fix at specific indices. Keys should match axis labels,
            values are the indices to slice at. For example, joint1=5 would
            fix the "joint1" dimension at index 5.

    Raises:
        ValueError: If dimension mismatches are detected between mesh_grid,
            points_array, axis_labels, or axis_ranges.

    Example:
        >>> # 4D data: x, y, z, time
        >>> plot_nd(data, (x_pts, y_pts, z_pts, t_pts),
        ...         ("x", "y", "z", "time"), z=10)
        >>> # This will fix z at index 10 and create a 3D plot with time slider
    """
    # Validate input dimensions
    if mesh_grid.ndim != len(points_array):
        raise ValueError("Dimension of mesh_grid must match length of points_array")
    for dim, points in zip(mesh_grid.shape, points_array):
        if dim != len(points):
            raise ValueError("Dimension of mesh_grid must match length of points_array")
    if mesh_grid.ndim != len(axis_labels):
        raise ValueError("Dimension of mesh_grid must match length of axis_labels")
    if axis_ranges is not None and mesh_grid.ndim != len(axis_ranges):
        raise ValueError("Dimension of mesh_grid must match length of axis_ranges")

    # Convert axis names to indices for slicing
    axes_and_indices = dict[int, int]()
    for axis_name, index in conditional_kwargs.items():
        axes_and_indices[axis_labels.index(axis_name)] = index

    # Reduce dimensions by slicing at specified indices
    mesh_grid = _reduce_dims(mesh_grid, axes_and_indices)
    points_array = tuple(
        points for i, points in enumerate(points_array) if i not in axes_and_indices
    )
    axis_labels = tuple(
        label for i, label in enumerate(axis_labels) if i not in axes_and_indices
    )
    axis_ranges = (
        tuple(
            ranges for i, ranges in enumerate(axis_ranges) if i not in axes_and_indices
        )
        if axis_ranges
        else None
    )

    # Route to appropriate plotting function based on remaining dimensions
    if len(points_array) == 2:
        # 2D plot: X, Y domain with Z as scalar field
        plot_2d(
            mesh_grid,
            points_array[0],
            points_array[1],
            title=title,
            x_label=axis_labels[0],
            y_label=axis_labels[1],
            z_label=axis_labels[2] if len(axis_labels) > 2 else "Z Axis",
            x_min=axis_ranges[0][0] if axis_ranges else None,
            x_max=axis_ranges[0][1] if axis_ranges else None,
            y_min=axis_ranges[1][0] if axis_ranges else None,
            y_max=axis_ranges[1][1] if axis_ranges else None,
        )

    elif len(points_array) == 3:
        # 3D plot: X, Y domain with Z as scalar field, animated over time
        plot_3d(
            mesh_grid,
            points_array[0],
            points_array[1],
            points_array[2],
            title=title,
            x_label=axis_labels[0],
            y_label=axis_labels[1],
            z_label="Result",
            time_label=axis_labels[2],
            x_min=axis_ranges[0][0] if axis_ranges else None,
            x_max=axis_ranges[0][1] if axis_ranges else None,
            y_min=axis_ranges[1][0] if axis_ranges else None,
            y_max=axis_ranges[1][1] if axis_ranges else None,
        )

    else:
        raise ValueError(
            "After slicing, data must be 2D or 3D for plotting. "
            "Please adjust conditional_kwargs."
        )


def _reduce_dims(array: np.ndarray, axes_and_indices: dict[int, int]) -> np.ndarray:
    """Reduces dimensions of a NumPy array by slicing at specified indices.

    This helper function is used to extract lower-dimensional slices from
    high-dimensional arrays, enabling visualization of specific cross-sections
    through the data.

    Args:
        array: The NumPy array to slice.
        axes_and_indices: Dictionary mapping dimension indices to slice positions.
            For example, {1: 3, 3: 10} means slice dimension 1 at index 3
            and dimension 3 at index 10.

    Returns:
        The reduced array with specified dimensions sliced out.

    Raises:
        IndexError: If any specified index is out of bounds for its dimension.

    Example:
        >>> data = np.random.rand(10, 20, 30, 40)  # 4D array
        >>> sliced = _reduce_dims(data, {1: 5, 3: 15})
        >>> sliced.shape
        (10, 30)  # Dimensions 1 and 3 have been sliced out
    """
    # Create a list of slice objects for each dimension (initially keeping all data)
    slicer: list[slice | int] = [slice(None)] * array.ndim

    # Replace slice objects with integer indices for dimensions to be reduced
    for axis, index in axes_and_indices.items():
        # Validate index bounds
        if not -array.shape[axis] <= index < array.shape[axis]:
            raise IndexError(f"Index {index} is out of bounds for dimension {axis}.")
        slicer[axis] = index

    # Apply the slicing operation and return the reduced array
    return array[tuple(slicer)]
