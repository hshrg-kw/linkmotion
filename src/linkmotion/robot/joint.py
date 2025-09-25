import logging
from enum import Enum, auto
import numpy as np

from linkmotion.const import SMALL_NUM, LARGE_NUM
from linkmotion.typing.numpy import Vector3
from linkmotion.transform.transform import Transform

logger = logging.getLogger(__name__)


class JointType(Enum):
    """
    Enum for different types of joints, based on URDF specification.
    See: http://wiki.ros.org/urdf/XML/joint#Attributes
    """

    CONTINUOUS = auto()  # Rotates without limits
    PRISMATIC = auto()  # Moves along an axis without limits
    FIXED = auto()  # A non-movable joint
    REVOLUTE = auto()  # Rotates with defined limits
    PLANAR = auto()  # Moves in a 2D plane
    FLOATING = auto()  # Allows 6-DOF motion


class Joint:
    """Represents a joint connecting two links in a robot model."""

    def __init__(
        self,
        name: str,
        type: JointType,
        child_link_name: str,
        parent_link_name: str,
        direction: Vector3 | None = None,
        center: Vector3 | None = None,
        min_: float = -LARGE_NUM,
        max_: float = LARGE_NUM,
    ):
        """Initializes a Joint.

        Args:
            name: The name of the joint.
            type: The type of the joint (e.g., REVOLUTE, PRISMATIC).
            child_link_name: The name of the child link.
            parent_link_name: The name of the parent link.
            direction: The axis of motion for revolute or prismatic joints.
                       Defaults to [0.0, 0.0, 1.0].
            center: The center of rotation for revolute joints.
            min_: The lower limit for revolute or prismatic joints.
            max_: The upper limit for revolute or prismatic joints.

        Raises:
            ValueError: If `direction` is a zero vector, or if a revolute/
                        continuous joint is defined without a `center`.
        """
        self.name = name
        self.type = type
        self.parent_link_name = parent_link_name
        self.child_link_name = child_link_name
        self.center = center
        self.min = min_
        self.max = max_

        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])

        # Normalize the direction vector and ensure it's not a zero vector.
        norm = np.linalg.norm(direction)
        if norm < SMALL_NUM:
            err_msg = f"Joint '{name}' direction vector cannot be a zero vector."
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.direction = direction / norm

        # Revolute and Continuous joints require a center of rotation.
        if (
            self.type in {JointType.REVOLUTE, JointType.CONTINUOUS}
            and self.center is None
        ):
            err_msg = (
                f"Joint '{name}' of type '{self.type.name}' requires a 'center' "
                "for rotation, but it was not provided."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.info(f"Joint '{self.name}' created.")

    def __str__(self) -> str:
        """Returns a string representation of the joint."""
        return (
            f"Joint(name='{self.name}', type={self.type.name}, "
            f"parent='{self.parent_link_name}', child='{self.child_link_name}')"
        )

    @staticmethod
    def from_other(
        other: "Joint", name: str, transform: Transform | None = None
    ) -> "Joint":
        """Creates a new Joint by copying another, but with a new name.

        This performs a deep copy of mutable attributes like 'direction' and
        'center' to ensure the new joint is independent.
        """
        if transform is None:
            transform = Transform()
        return Joint(
            name,
            other.type,
            other.child_link_name,
            other.parent_link_name,
            transform.apply(other.direction),
            None if other.center is None else transform.apply(other.center),
            other.min,
            other.max,
        )
