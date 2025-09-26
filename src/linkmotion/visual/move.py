# type: ignore
from time import sleep
import logging
from typing import TypeAlias

try:
    import k3d
except ImportError:
    raise ImportError("k3d is not installed. Please install it via 'pip install k3d'")

from linkmotion.visual.mesh import MeshVisualizer
from linkmotion.move.manager import MoveManager
from linkmotion.visual.base import _get_or_create_plot
from linkmotion.visual.robot import visualize_joint_helper

logger = logging.getLogger(__name__)

JointValues: TypeAlias = dict[str, float]
CommandTimeSeries: TypeAlias = dict[float, JointValues]


class MoveVisualizer:
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def joint(
        mm: MoveManager,
        joint_name: str,
        plot: k3d.Plot | None = None,
        helper_length: float = 5,
        color: int = 0xFF0000,
        width: float = 0.1,
        point_size: float = 0.5,
    ) -> k3d.Plot:
        plot = _get_or_create_plot(plot)

        joint = mm.robot.joint(joint_name)
        direction = mm.get_direction(joint_name)
        center = mm.get_center(joint_name)
        if center is None:
            logger.warning(
                f"Joint {joint_name} has no center defined, skipping visualization."
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
    def link(
        mm: MoveManager,
        link_name: str,
        opacity: float | None = None,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        return MeshVisualizer.trimesh(
            mm.get_link_visual_mesh(link_name), plot, opacity=opacity
        )

    @staticmethod
    def links(
        mm: MoveManager,
        link_names: set[str],
        opacity: float | None = None,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        for n in link_names:
            plot = MoveVisualizer.link(mm, n, opacity, plot)
        return plot

    @staticmethod
    def robot(
        mm: MoveManager,
        opacity: float | None = None,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        return MoveVisualizer.links(
            mm, {link.name for link in mm.robot.links()}, opacity=opacity, plot=plot
        )

    @staticmethod
    def move(
        mm: MoveManager,
        command_series: CommandTimeSeries,
        link_names: set[str],
        plot: k3d.Plot | None = None,
    ):
        joint_set = set()
        for commands in command_series.values():
            joint_set |= set(commands.keys())

        link_names_list = list(link_names)
        for name in link_names_list:
            plot = MoveVisualizer.link(mm, name, plot=plot)
        plot.display()

        sorted_times = sorted(command_series.keys())
        sorted_commands = [command_series[key] for key in sorted_times]
        sorted_times = [0.0] + sorted_times

        for i, command in enumerate(sorted_commands):
            sleep_time = sorted_times[i + 1] - sorted_times[i]
            sleep(sleep_time)

            [mm.move(name, value) for name, value in command.items()]

            for i, link_name in enumerate(reversed(link_names_list)):
                new_vertices = mm.get_link_visual_mesh(link_name).vertices
                plot.objects[-(i + 1)].vertices = new_vertices
