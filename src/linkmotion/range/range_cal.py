import os
import time
import logging
from pathlib import Path
from itertools import product
from typing import Dict, Optional, Iterable
from multiprocessing import Pool

import psutil
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from linkmotion.robot.robot import Robot
from linkmotion.collision.manager import CollisionManager
from linkmotion.move.manager import MoveManager
from linkmotion.robot.joint import JointType, Joint

logger = logging.getLogger(__name__)


class RangeCalcCondition:
    """Represents calculation conditions for a single joint axis.

    This class encapsulates a joint and its survey points for range calculation,
    providing convenient access to joint properties and validation.

    Args:
        joint: The joint object containing limits and properties.
        survey_points: Array of joint values to evaluate during range calculation.
    """

    def __init__(self, joint: Joint, survey_points: npt.NDArray[np.float64]):
        self.joint = joint
        self.survey_points = survey_points

    def __repr__(self) -> str:
        return (
            f"RangeCalcCondition(joint_name='{self.joint.name}', "
            f"points={len(self.survey_points)}, "
            f"range=[{self.min:.3f}, {self.max:.3f}])"
        )

    @property
    def min(self) -> float:
        """Minimum joint limit.

        Returns:
            The minimum allowable joint value.
        """
        return self.joint.min

    @property
    def max(self) -> float:
        """Maximum joint limit.

        Returns:
            The maximum allowable joint value.
        """
        return self.joint.max

    @property
    def joint_name(self) -> str:
        """Name of the joint.

        Returns:
            The joint's name identifier.
        """
        return self.joint.name


_worker_instance: Optional["RangeCalculator"] = None
"""
Global variable to hold the instance for multi-process worker processes
"""


def _worker_init(instance: "RangeCalculator"):
    """
    Store the instance in a global variable for worker processes to access
    """
    global _worker_instance
    _worker_instance = instance


def _calculate_single_point_worker(point):
    """
    Define the parallel processing calculation logic as a top-level function
    This function uses the instance stored in the global variable to perform the calculation for a single point
    """
    # Access necessary methods and data through _worker_instance
    if _worker_instance is None:
        raise RuntimeError("Worker instance is not initialized.")
    return _worker_instance._calculate_single_point(point)


class RangeCalculator:
    """Calculates collision-free ranges across multiple joint axes.

    This class performs parallel computation to determine which combinations
    of joint values result in collisions between specified link groups.

    Args:
        robot: The robot model to analyze.
        link_names1: First set of link names to check for collisions.
        link_names2: Second set of link names to check for collisions.
    """

    def __init__(self, robot: Robot, link_names1: set[str], link_names2: set[str]):
        mm = MoveManager(robot)
        self.cm = CollisionManager(mm)
        self.calc_conditions: Dict[str, RangeCalcCondition] = {}
        self.link_names1 = link_names1
        self.link_names2 = link_names2
        self.results: Optional[npt.NDArray[np.float64]] = None

    def __repr__(self) -> str:
        return (
            f"RangeCalculator(axes={len(self.calc_conditions)}, "
            f"links1={len(self.link_names1)}, "
            f"links2={len(self.link_names2)}, "
            f"computed={self.results is not None})"
        )

    def get_axis_names(self) -> list[str]:
        """Get ordered list of joint names for calculation axes.

        Returns:
            List of joint names in the order they were added.
        """
        return [cond.joint_name for cond in self.calc_conditions.values()]

    def get_axis_points(self) -> list[npt.NDArray[np.float64]]:
        """Get ordered list of survey points for each axis.

        Returns:
            List of survey point arrays corresponding to each joint axis.
        """
        return [cond.survey_points for cond in self.calc_conditions.values()]

    def add_axis(self, joint_name: str, survey_points: npt.NDArray[np.float64]) -> None:
        """Add a joint axis for range calculation.

        Args:
            joint_name: Name of the joint to add as a calculation axis.
            survey_points: Array of joint values to evaluate for this axis.

        Raises:
            ValueError: If survey points are outside joint limits or joint type is unsupported.
            KeyError: If the joint name doesn't exist in the robot model.
        """
        try:
            joint = self.cm.robot.joint(joint_name)
        except Exception as e:
            raise KeyError(f"Joint '{joint_name}' not found in robot model") from e

        # Validate joint type
        supported_types = {
            JointType.PRISMATIC,
            JointType.REVOLUTE,
            JointType.CONTINUOUS,
        }
        if joint.type not in supported_types:
            raise ValueError(
                f"Joint '{joint_name}' type {joint.type} is not supported. "
                f"Supported types: {', '.join(t.name for t in supported_types)}"
            )

        sorted_points = np.sort(survey_points)
        cond = RangeCalcCondition(joint, sorted_points)

        # Check if any survey points are outside joint limits
        points_below_min = np.any(survey_points < cond.min)
        points_above_max = np.any(survey_points > cond.max)

        if points_below_min or points_above_max:
            raise ValueError(
                f"Survey points [{np.min(survey_points):.3f}, {np.max(survey_points):.3f}] "
                f"are outside joint limits [{cond.min:.3f}, {cond.max:.3f}] for joint '{joint_name}'"
            )

        self.calc_conditions[joint_name] = cond
        logger.info(
            f"Added axis '{joint_name}' with {len(survey_points)} survey points"
        )

    def _calculate_single_point(self, point: tuple[float, ...]) -> float:
        """Calculate collision status for a single point in joint space.

        You can overwrite this method for custom collision checks.

        Args:
            point: Tuple of joint values corresponding to each axis.

        Returns:
            1.0 if collision detected, 0.0 otherwise.
        """
        axis_names = self.get_axis_names()
        for name, value in zip(axis_names, point):
            self.cm.mm.move(name, value)

        result = self.cm.collide(self.link_names1, self.link_names2)
        return 1.0 if result.is_collision else 0.0

    def _generate_grid_points(self) -> product:
        """Generate Cartesian product of all axis survey points.

        Returns:
            Iterator over all combinations of joint values.
        """
        axis_points = self.get_axis_points()
        if not axis_points:
            raise ValueError(
                "No axes have been added. Use add_axis() to add joint axes."
            )
        return product(*axis_points)

    def _compute_parallel(
        self, grid_points: product, process_num: int | None = None
    ) -> Iterable[float]:
        """Perform parallel collision detection across all grid points.

        Args:
            grid_points: Iterator of joint value combinations to evaluate.

        Returns:
            Flat list of collision results (1.0 for collision, 0.0 for no collision).
        """
        if process_num is None:
            # get physical CPU core count
            process_num = psutil.cpu_count(logical=False) or 1
            logger.info(f"Found {process_num} physical cpu cores.")

        logger.info(f"Starting parallel execution on {process_num} processes...")

        # set empirical chunk size
        points_list = list(grid_points)
        total_tasks = len(points_list)
        chunksize, remainder = divmod(total_tasks, process_num * 4)
        if remainder:
            chunksize += 1

        result = list[float]()
        with Pool(
            processes=process_num, initializer=_worker_init, initargs=(self,)
        ) as pool:
            imap_result = pool.imap(
                _calculate_single_point_worker, points_list, chunksize
            )
            result = list(
                tqdm(
                    imap_result,
                    total=total_tasks,
                    desc="Calculating Range (Parallel)",
                    unit=" point",
                )
            )

        logger.info("Parallel execution finished.")
        return result

    def _compute_serial(self, grid_points: product) -> Iterable[float]:
        """Perform serial collision detection across all grid points.

        Args:
            grid_points: Iterator of joint value combinations to evaluate.

        Returns:
            Flat list of collision results (1.0 for collision, 0.0 for no collision).
        """
        points_list = list(grid_points)
        return [
            self._calculate_single_point(point)
            for point in tqdm(
                points_list, desc="Calculating Range (Serial)", unit=" point"
            )
        ]

    def _reshape_results(
        self, flat_results: Iterable[float]
    ) -> npt.NDArray[np.float64]:
        """Reshape flat results into multi-dimensional array.

        Args:
            flat_results: Flat list of collision detection results.

        Returns:
            Multi-dimensional array with shape corresponding to axis survey points.
        """
        axis_points = self.get_axis_points()
        result_shape = tuple(len(points) for points in axis_points)

        logger.info(f"Reshaping results into array with shape: {result_shape}")
        return np.array(flat_results, dtype=np.float64).reshape(result_shape)

    def execute(self, process_num: int | None = None) -> None:
        """Execute range calculation across all defined axes.

        Performs collision detection for all combinations of survey points
        across the defined joint axes using parallel processing. Progress
        is logged at regular intervals during computation.

        Raises:
            ValueError: If no axes have been defined.
        """
        if not self.calc_conditions:
            raise ValueError(
                "No calculation axes defined. Use add_axis() to add joint axes."
            )

        # Log initial setup information
        axis_points = self.get_axis_points()
        total_points = int(np.prod([len(points) for points in axis_points]))

        logger.info(f"Starting range calculation with {len(self.calc_conditions)} axes")
        logger.info(f"Total combinations to evaluate: {total_points:,}")
        for i, (name, condition) in enumerate(self.calc_conditions.items()):
            logger.info(
                f"  Axis {i + 1}: '{name}' with {len(condition.survey_points)} points"
            )

        start_time = time.time()

        grid_points = self._generate_grid_points()
        if process_num == 1:
            logger.info("Process number set to 1, running in serial mode.")
            flat_results = self._compute_serial(grid_points)
        else:
            flat_results = self._compute_parallel(grid_points, process_num)

        self.results = self._reshape_results(flat_results)

        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"Range calculation complete [Duration: {elapsed_time:.2f}s]")

    def export(self, file_path: Path) -> None:
        """Exports the calculation results to a compressed NumPy file.

        Saves the collision results, axis names, survey points, and link names
        to a `.npz` file for later use.

        Args:
            file_path: The path to save the file to.

        Raises:
            ValueError: If the calculation has not been executed yet.
        """
        if self.results is None:
            raise ValueError(
                "Calculation results are not available. Run execute() first."
            )

        logger.info(f"Exporting calculation results to '{file_path}'...")
        np.savez_compressed(
            file_path,
            results=self.results,
            axis_names=np.array(self.get_axis_names()),
            axis_points=np.array(self.get_axis_points(), dtype=object),
            link_names1=np.array(list(self.link_names1)),
            link_names2=np.array(list(self.link_names2)),
        )
        logger.info(f"Successfully exported calculation results to '{file_path}'")

    @classmethod
    def import_from_file(cls, file_path: Path, robot: Robot) -> "RangeCalculator":
        """Imports calculation results from a compressed NumPy file.

        Creates a new RangeCalculator instance and populates it with data
        loaded from a `.npz` file.

        Args:
            file_path: The path to the `.npz` file.
            robot: The robot model instance, required to reconstruct
                   the calculation conditions.

        Returns:
            A new RangeCalculator instance with the loaded data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If the file is missing required data keys.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' was not found.")

        logger.info(f"Importing calculation results from '{file_path}'...")

        # allow_pickle=True is needed as axis_points is an array of arrays (object array).
        data = np.load(file_path, allow_pickle=True)

        # Check for required keys
        required_keys = {
            "results",
            "axis_names",
            "axis_points",
            "link_names1",
            "link_names2",
        }
        if not required_keys.issubset(data.keys()):
            missing_keys = required_keys - set(data.keys())
            raise KeyError(
                f"The file '{file_path}' is missing required data: {missing_keys}"
            )

        # Reconstruct the instance state
        link_names1 = set(data["link_names1"])
        link_names2 = set(data["link_names2"])

        instance = cls(robot, link_names1, link_names2)
        instance.results = data["results"]

        axis_names = data["axis_names"]
        axis_points = data["axis_points"]

        # Rebuild calc_conditions dictionary
        for name, points in zip(axis_names, axis_points):
            try:
                joint = robot.joint(name)
                condition = RangeCalcCondition(joint, points)
                instance.calc_conditions[name] = condition
            except Exception as e:
                logger.error(
                    f"Failed to reconstruct axis '{name}'. It may not exist "
                    "in the provided robot model."
                )
                raise e

        logger.info("Successfully imported and reconstructed calculation results.")
        return instance

    def plot(self):
        from linkmotion.visual.range import plot_2d, plot_3d

        if self.results is None:
            raise ValueError(
                "Calculation results are not available. Run execute() first."
            )

        if len(self.get_axis_names()) == 2:
            axis_points = self.get_axis_points()
            labels = self.get_axis_names()
            plot_2d(
                mesh_grid=self.results,
                x_points=axis_points[0],
                y_points=axis_points[1],
                x_label=labels[0],
                y_label=labels[1],
            )

        elif len(self.get_axis_names()) == 3:
            axis_points = self.get_axis_points()
            labels = self.get_axis_names()
            plot_3d(
                mesh_grid=self.results,
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
