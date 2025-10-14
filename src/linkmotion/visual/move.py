# type: ignore
"""Motion visualization utilities for animating robot movements.

This module provides visualization capabilities for robot motion, including
static visualization of robot configurations and dynamic animation of robot
movements over time. It works with MoveManager to display robot states and
joint transformations.
"""

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

# Type aliases for motion command specifications
JointValues: TypeAlias = dict[str, float]
"""Mapping of joint names to their target values."""

CommandTimeSeries: TypeAlias = dict[float, JointValues]
"""Time-indexed sequence of joint commands, where keys are timestamps in seconds."""


class MoveVisualizer:
    """Static methods for visualizing robot motion and configurations.

    This class provides visualization methods for displaying robot states
    managed by MoveManager, including individual joints, links, and complete
    robot assemblies in their current configuration. It also supports
    animated visualization of robot motion sequences.
    """

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
        """Visualizes a joint in its current transformed state.

        Displays the joint with its current position and orientation as
        managed by the MoveManager, showing the transformed center and
        direction based on the robot's current configuration.

        Args:
            mm: The MoveManager containing the robot and its current state.
            joint_name: Name of the joint to visualize.
            plot: Optional existing k3d.Plot to add the joint to.
                If None, creates a new plot.
            helper_length: Length of the helper line for visualizing the joint axis.
            color: Color of the joint visualization in hexadecimal format.
            width: Width of the joint axis lines.
            point_size: Size of the joint center and range limit points.

        Returns:
            The k3d.Plot object containing the visualized joint.
        """
        plot = _get_or_create_plot(plot)

        # Get joint information from the robot
        joint = mm.robot.joint(joint_name)
        # Get the transformed direction and center from the current robot state
        direction = mm.get_direction(joint_name)
        center = mm.get_center(joint_name)

        # Validate that the joint has a defined center position
        if center is None:
            logger.warning(
                f"Joint {joint_name} has no center defined, skipping visualization."
            )
            return plot

        # Use the joint visualization helper with transformed coordinates
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
        """Visualizes a single link in its current transformed state.

        Displays the link's visual mesh with its current transformation as
        managed by the MoveManager, reflecting the robot's current configuration.

        Args:
            mm: The MoveManager containing the robot and its current state.
            link_name: Name of the link to visualize.
            opacity: Optional opacity value for the link mesh (0.0 to 1.0).
            plot: Optional existing k3d.Plot to add the link to.
                If None, creates a new plot.

        Returns:
            The k3d.Plot object containing the visualized link.
        """
        # Get the transformed visual mesh for the link in its current state
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
        """Visualizes multiple links in their current transformed states.

        Displays all specified links with their current transformations,
        useful for visualizing subsets of a robot.

        Args:
            mm: The MoveManager containing the robot and its current state.
            link_names: Set of link names to visualize.
            opacity: Optional opacity value for all link meshes (0.0 to 1.0).
            plot: Optional existing k3d.Plot to add the links to.
                If None, creates a new plot.

        Returns:
            The k3d.Plot object containing all visualized links.
        """
        # Visualize each link in the set
        for n in link_names:
            plot = MoveVisualizer.link(mm, n, opacity, plot)
        return plot

    @staticmethod
    def robot(
        mm: MoveManager,
        opacity: float | None = None,
        plot: k3d.Plot | None = None,
    ) -> k3d.Plot:
        """Visualizes the complete robot in its current configuration.

        Displays all links of the robot with their current transformations
        as managed by the MoveManager.

        Args:
            mm: The MoveManager containing the robot and its current state.
            opacity: Optional opacity value for all link meshes (0.0 to 1.0).
            plot: Optional existing k3d.Plot to add the robot to.
                If None, creates a new plot.

        Returns:
            The k3d.Plot object containing the complete visualized robot.
        """
        # Visualize all links in the robot
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
        """Animates robot motion by executing a time-series of joint commands.

        This method displays an animated visualization of the robot moving
        through a sequence of configurations. Joint commands are executed
        at specified times, and the visualization is updated in real-time
        to show the robot's motion.

        The animation runs by:
        1. Displaying the initial robot configuration
        2. Waiting for the specified time interval
        3. Executing the joint commands at each timestamp
        4. Updating the visual mesh vertices to reflect the new configuration

        Args:
            mm: The MoveManager containing the robot and its current state.
            command_series: Dictionary mapping timestamps (in seconds) to
                joint commands. Each command is a dict mapping joint names
                to target values.
            link_names: Set of link names to visualize during the animation.
            plot: Optional existing k3d.Plot to use for animation.
                If None, creates a new plot.

        Note:
            This method uses time.sleep() for timing, so accuracy may be
            limited by system scheduler precision. The plot must be displayed
            in an interactive environment (e.g., Jupyter notebook) for the
            animation to be visible.
        """
        # Collect all joints that will be moved during the animation
        joint_set = set()
        for commands in command_series.values():
            joint_set |= set(commands.keys())

        # Initialize visualization with all specified links
        link_names_list = list(link_names)
        for name in link_names_list:
            plot = MoveVisualizer.link(mm, name, plot=plot)
        plot.display()

        # Prepare sorted timeline for sequential execution
        sorted_times = sorted(command_series.keys())
        sorted_commands = [command_series[key] for key in sorted_times]
        sorted_times = [0.0] + sorted_times  # Prepend 0 for initial state timing

        # Execute animation loop
        for i, command in enumerate(sorted_commands):
            # Wait for the time interval before this command
            sleep_time = sorted_times[i + 1] - sorted_times[i]
            sleep(sleep_time)

            # Execute all joint movements for this timestamp
            [mm.move(name, value) for name, value in command.items()]

            # Update the visual mesh vertices for all links
            # Note: Iterates in reverse to match the order objects were added to plot
            for i, link_name in enumerate(reversed(link_names_list)):
                new_vertices = mm.get_link_visual_mesh(link_name).vertices
                plot.objects[-(i + 1)].vertices = new_vertices
