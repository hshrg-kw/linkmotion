# type: ignore
"""Robot visualization utilities for 3D rendering of robot structures.

This module provides visualization capabilities for robot components including
links, joints, and complete robot assemblies. It supports visualization of
different joint types (revolute, prismatic, fixed) with appropriate visual
helpers to indicate joint axes and ranges of motion.
"""

import logging

import numpy as np

try:
    import k3d
except ImportError:
    raise ImportError("k3d is not installed. Please install it via 'pip install k3d'")

from linkmotion.visual.base import _get_or_create_plot
from linkmotion.robot.robot import Robot
from linkmotion.robot.link import Link
from linkmotion.robot.joint import Joint, JointType
from linkmotion.visual.mesh import MeshVisualizer
from linkmotion.visual.base import BasicVisualizer
from linkmotion.typing.numpy import Vector3

logger = logging.getLogger(__name__)


class RobotVisualizer:
    """Static methods for visualizing robot components and assemblies.

    This class provides a collection of visualization methods for rendering
    robot links, joints, and complete robot structures in 3D using k3d.
    All methods are static and can be called directly without instantiation.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def link(
        link: Link,
        opacity: float | None = None,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        """Visualizes a single robot link using its visual mesh.

        Args:
            link: The Link object to visualize.
            opacity: Optional opacity value for the link mesh (0.0 to 1.0).
                If None, uses the default opacity from the mesh.
            plot: Optional existing k3d.Plot to add the link to.
                If None, creates a new plot.

        Returns:
            The k3d.Plot object containing the visualized link.
        """
        plot = _get_or_create_plot(plot)
        # Extract the visual mesh representation from the link
        mesh = link.visual_mesh()
        plot = MeshVisualizer.trimesh(mesh=mesh, opacity=opacity, plot=plot)
        return plot

    @staticmethod
    def joint(
        joint: Joint,
        plot: k3d.Plot | None = None,
        helper_length: float = 500,
        color: int = 0xFF0000,
        width: float = 1,
        point_size: float = 8,
    ) -> k3d.Plot:
        """Visualizes a robot joint with appropriate visual helpers.

        Displays the joint with visual indicators based on its type:
        - Revolute/Continuous: Shows axis line extending in both directions
        - Prismatic: Shows limited line segment with min/max range indicators
        - Fixed/Floating/Planar: Shows only the joint center point

        Args:
            joint: The Joint object to visualize.
            plot: Optional existing k3d.Plot to add the joint to.
                If None, creates a new plot.
            helper_length: Length of the helper line for revolute joints.
            color: Color of the joint visualization in hexadecimal format.
            width: Width of the joint axis lines.
            point_size: Size of the joint center and range limit points.

        Returns:
            The k3d.Plot object containing the visualized joint.
        """
        plot = _get_or_create_plot(plot)

        direction = joint.direction
        center = joint.center
        # Validate that the joint has a defined center position
        if center is None:
            logger.warning(
                f"Joint {joint.name} has no center defined, skipping visualization."
            )
            return plot

        # Delegate to helper function for actual rendering
        plot = visualize_joint_helper(
            transformed_center=center,
            transformed_direction=direction,
            joint=joint,
            helper_length=helper_length,
            color=color,
            width=width,
            point_size=point_size,
            plot=plot,
        )

        return plot

    @staticmethod
    def robot(
        robot: Robot,
        plot: k3d.Plot | None = None,
        opacity: float | None = None,
    ) -> k3d.Plot:
        """Visualizes a complete robot by rendering all its links.

        This method iterates through all links in the robot and visualizes
        each one using their visual meshes.

        Args:
            robot: The Robot object to visualize.
            plot: Optional existing k3d.Plot to add the robot to.
                If None, creates a new plot.
            opacity: Optional opacity value for all link meshes (0.0 to 1.0).
                If None, uses the default opacity from each mesh.

        Returns:
            The k3d.Plot object containing the complete visualized robot.
        """
        plot = _get_or_create_plot(plot)
        # Visualize each link in the robot assembly
        for link in robot.links():
            plot = RobotVisualizer.link(link, opacity=opacity, plot=plot)
        return plot


def visualize_joint_helper(
    transformed_center: Vector3,
    transformed_direction: Vector3,
    joint: Joint,
    helper_length: float = 5,
    color: int = 0xFF0000,
    width: float = 0.1,
    point_size: float = 0.5,
    plot: k3d.Plot | None = None,
) -> k3d.Plot:
    """Helper function to render joint visualization based on joint type.

    This function handles the type-specific rendering logic for different
    joint types, creating appropriate visual representations:
    - Revolute/Continuous: Infinite axis line extending in both directions
    - Prismatic: Bounded line segment showing min/max translation limits
    - Fixed/Floating/Planar: Only center point (no axis visualization)

    Args:
        transformed_center: The 3D position of the joint center in world coordinates.
        transformed_direction: The 3D unit vector indicating the joint axis direction.
        joint: The Joint object containing type and limit information.
        helper_length: Length of helper lines for visualizing joint axes.
        color: Hexadecimal color value for the visualization.
        width: Line width for rendering joint axes.
        point_size: Size of points for joint centers and limits.
        plot: Optional existing k3d.Plot to add to. If None, creates new plot.

    Returns:
        The k3d.Plot object with the joint visualization added.

    Raises:
        ValueError: If the joint type is not recognized.
    """
    # Always render the joint center point
    plot = BasicVisualizer.points(
        np.array([transformed_center]), point_size=point_size, color=color, plot=plot
    )

    # Render type-specific visualization
    match joint.type:
        case JointType.REVOLUTE | JointType.CONTINUOUS:
            # For revolute joints, show an axis line extending in both directions
            point1 = transformed_center - helper_length * transformed_direction
            point2 = transformed_center + helper_length * transformed_direction
            points = np.ascontiguousarray([point1, point2], dtype=np.float32)
            plot += k3d.line(points, color=color, width=width)
        case JointType.PRISMATIC:
            # For prismatic joints, show the translation range with min/max limits
            min_point = transformed_center + transformed_direction * joint.min
            max_point = transformed_center + transformed_direction * joint.max
            points = np.ascontiguousarray([min_point, max_point], dtype=np.float32)
            plot += k3d.line(points, color=color, width=width)
            # Highlight the min, max, and center positions
            edge_points = np.ascontiguousarray(
                [min_point, max_point, transformed_center], dtype=np.float32
            )
            plot += k3d.points(edge_points, color=color, point_size=point_size)
        case JointType.FIXED | JointType.FLOATING | JointType.PLANAR:
            # For fixed/floating/planar joints, only the center point is shown
            ...
        case _:
            raise ValueError(f"Unknown joint type: {joint.type}")

    return plot
