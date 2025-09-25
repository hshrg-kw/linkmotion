import fcl
import trimesh
import numpy as np
import logging

from linkmotion.transform import Transform
from linkmotion.typing.numpy import Vector3, RGBA0to1
from .base import ShapeBase

logger = logging.getLogger(__name__)


class Box(ShapeBase):
    """A box shape.

    The box is centered at the origin in its local coordinate frame.

    Attributes:
        extents (Vector3): The lengths of the box along the x, y, and z axes.
    """

    def __init__(
        self,
        extents: Vector3,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ):
        """Initializes the Box.

        Args:
            extents: The (x, y, z) dimensions of the box.
            default_transform: The default pose of the box.
            color: The RGBA color of the box, with values in [0, 1].

        Raises:
            ValueError: If extents are not positive values.
        """
        extents_array = np.array(extents, dtype=float)

        if len(extents_array) != 3:
            raise ValueError(
                f"Box extents must have exactly 3 dimensions, got {len(extents_array)}"
            )

        if np.any(extents_array <= 0):
            raise ValueError(f"Box extents must be positive, got {extents_array}")

        self.extents = extents_array

        try:
            # Create a mesh centered at the origin.
            # The default_transform will be handled by the base class.
            visual_mesh = trimesh.creation.box(extents=self.extents)

            collision_primitive = self.create_collision_primitive()
            super().__init__(
                collision_primitive,
                visual_mesh,
                default_transform=default_transform,
                color=color,
            )
        except Exception as e:
            logger.error(f"Failed to create Box with extents {self.extents}: {e}")
            raise

    def create_collision_primitive(self) -> fcl.Box:
        """Creates the FCL collision geometry for the box.

        Returns:
            An fcl.Box object.
        """
        return fcl.Box(*self.extents)

    def copy(self):
        """Creates a copy of the Box instance.

        Returns:
            A new Box instance with the same parameters.
        """
        return Box(
            self.extents.copy(),
            default_transform=self.default_transform.copy(),
            color=self.color,
        )

    def __repr__(self):
        return (
            f"Box(extents={self.extents}, default_transform={self.default_transform})"
        )
