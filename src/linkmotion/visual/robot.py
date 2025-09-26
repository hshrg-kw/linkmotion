# type: ignore
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
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def link(
        link: Link,
        opacity: float | None = None,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        plot = _get_or_create_plot(plot)
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
        plot = _get_or_create_plot(plot)

        direction = joint.direction
        center = joint.center
        if center is None:
            logger.warning(
                f"Joint {joint.name} has no center defined, skipping visualization."
            )
            return plot

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
        plot = _get_or_create_plot(plot)
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
    plot = BasicVisualizer.points(
        np.array([transformed_center]), point_size=point_size, color=color, plot=plot
    )

    match joint.type:
        case JointType.REVOLUTE | JointType.CONTINUOUS:
            point1 = transformed_center - helper_length * transformed_direction
            point2 = transformed_center + helper_length * transformed_direction
            points = np.ascontiguousarray([point1, point2], dtype=np.float32)
            plot += k3d.line(points, color=color, width=width)
        case JointType.PRISMATIC:
            min_point = transformed_center + transformed_direction * joint.min
            max_point = transformed_center + transformed_direction * joint.max
            points = np.ascontiguousarray([min_point, max_point], dtype=np.float32)
            plot += k3d.line(points, color=color, width=width)
            edge_points = np.ascontiguousarray(
                [min_point, max_point, transformed_center], dtype=np.float32
            )
            plot += k3d.points(edge_points, color=color, point_size=point_size)
        case JointType.FIXED | JointType.FLOATING | JointType.PLANAR:
            ...
        case _:
            raise ValueError(f"Unknown joint type: {joint.type}")

    return plot
