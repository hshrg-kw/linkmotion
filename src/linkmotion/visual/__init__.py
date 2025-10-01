import logging

logger = logging.getLogger(__name__)

try:
    from .robot import RobotVisualizer
    from .mesh import MeshVisualizer
    from .base import BasicVisualizer
    from .move import MoveVisualizer
    from .collision import CollisionVisualizer
    from .range import plot_2d, plot_3d

    __all__ = [
        "RobotVisualizer",
        "MeshVisualizer",
        "BasicVisualizer",
        "MoveVisualizer",
        "CollisionVisualizer",
        "plot_2d",
        "plot_3d",
    ]

except ImportError:
    logger.warning(
        "Some visualization modules could not be imported,"
        "because of missing dependencies."
        "Please 'pip install linkmotion[jup]' to install them."
    )
