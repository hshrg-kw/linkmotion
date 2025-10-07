# type: ignore
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
        collision_result: fcl.CollisionResult = cm.collide(link_names1, link_names2)

        if collision_result.is_collision:
            points = np.array(collision_result.contacts[0].pos).reshape(-1, 3)
            plot = BasicVisualizer.points(points, plot=plot, point_size=point_size)
            logger.info(f"Collision detected at {points}.")
        else:
            distance_result = cm.distance(
                link_names1, link_names2, enable_nearest_points=True
            )
            points = np.array(distance_result.nearest_points).reshape(-1, 3)
            plot = BasicVisualizer.points(points, plot=plot, point_size=point_size)
            plot += k3d.line(
                np.ascontiguousarray(points, np.float32), color=color, width=width
            )
            logger.info(
                f"minimum distance is {distance_result.min_distance} at {points}."
            )
        return plot
