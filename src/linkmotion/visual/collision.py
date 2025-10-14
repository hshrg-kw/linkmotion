# type: ignore
"""Collision detection visualization utilities.

This module provides visualization capabilities for collision detection
results, including displaying closest points between robot components
and visualizing collision contact points. It works with CollisionManager
to provide visual feedback for collision queries.
"""

import logging

try:
    import k3d
except ImportError:
    raise ImportError("k3d is not installed. Please install it via 'pip install k3d'")
import fcl
import numpy as np

from linkmotion.collision.manager import CollisionManager
from linkmotion.visual.base import BasicVisualizer

logger = logging.getLogger(__name__)


class CollisionVisualizer:
    """Static methods for visualizing collision detection results.

    This class provides visualization capabilities for displaying collision
    and proximity information between robot components. It can show collision
    contact points or the nearest points between non-colliding geometries.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def distance_points(
        cm: CollisionManager,
        link_names1: set[str],
        link_names2: set[str],
        plot: k3d.Plot | None = None,
        point_size: float = 3,
        color: int = 0xFF0000,
        width: float = 1,
    ) -> k3d.Plot:
        """Visualizes closest points or collision contacts between two link sets.

        This method performs collision detection between two sets of links and
        visualizes the result:
        - If collision is detected: Shows the collision contact point(s)
        - If no collision: Shows the two nearest points and a line connecting them

        The visualization provides immediate visual feedback for collision queries,
        useful for debugging collision detection, clearance checking, and
        understanding spatial relationships between robot components.

        Args:
            cm: The CollisionManager containing the robot's collision geometry.
            link_names1: Set of link names for the first group.
            link_names2: Set of link names for the second group.
            plot: Optional existing k3d.Plot to add the visualization to.
                If None, creates a new plot.
            point_size: Size of the points marking nearest/contact locations.
            color: Color for the visualization in hexadecimal format.
            width: Width of the line connecting nearest points (non-collision case).

        Returns:
            The k3d.Plot object containing the visualization.

        Note:
            This method logs collision/distance information at debug level,
            which can be useful for troubleshooting.
        """
        # First check if the link sets are in collision
        collision_result: fcl.CollisionResult = cm.collide(link_names1, link_names2)

        if collision_result.is_collision:
            # Collision detected: visualize the contact point
            points = np.array(collision_result.contacts[0].pos).reshape(-1, 3)
            plot = BasicVisualizer.points(points, plot=plot, point_size=point_size)
            logger.debug(f"Collision detected at {points}.")
        else:
            # No collision: compute and visualize the nearest points
            distance_result = cm.distance(
                link_names1, link_names2, enable_nearest_points=True
            )
            points = np.array(distance_result.nearest_points).reshape(-1, 3)
            # Show both nearest points
            plot = BasicVisualizer.points(points, plot=plot, point_size=point_size)
            # Draw a line connecting the nearest points to show clearance
            plot += k3d.line(
                np.ascontiguousarray(points, np.float32), color=color, width=width
            )
            logger.debug(
                f"minimum distance is {distance_result.min_distance} at {points}."
            )
        return plot
