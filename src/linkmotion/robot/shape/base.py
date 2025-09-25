from abc import ABC, abstractmethod
from typing import Type, TypeVar
import logging

import fcl
import trimesh
import numpy as np

from linkmotion.transform import Transform
from linkmotion.typing.numpy import RGBA0to1

logger = logging.getLogger(__name__)

# Generic type for class methods like from_other
T = TypeVar("T", bound="ShapeBase")


class ShapeBase(ABC):
    """Abstract base class for geometric shapes.

    This class provides a common interface for shapes used in robotics,
    encapsulating both a visual representation (a trimesh.Trimesh object)
    and a collision primitive (an fcl.CollisionGeometry object).

    Attributes:
        collision_primitive (fcl.CollisionGeometry): The geometry used for
            collision checking. Its local frame should be centered at the origin.
        visual_mesh (trimesh.Trimesh): The mesh used for visualization.
            Its local frame should be centered at the origin, matching the
            collision primitive.
        default_transform (Transform): The default pose of the shape in its
            parent frame.
    """

    def __init__(
        self,
        collision_primitive: fcl.CollisionGeometry,
        visual_mesh: trimesh.Trimesh,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ):
        """Initializes the ShapeBase.

        Args:
            collision_primitive: The collision geometry.
            visual_mesh: The visual mesh.
            default_transform: The default pose of the shape. If None, an
                identity transform is used.
            color: The RGBA color of the shape, with values in [0, 1].
        """
        self.collision_primitive = collision_primitive
        self.visual_mesh = visual_mesh.copy()

        if color is not None:
            self.visual_mesh = self._apply_color_to_mesh(self.visual_mesh, color)

        self.default_transform = (
            Transform() if default_transform is None else default_transform.copy()
        )
        self.color = None if color is None else color.copy()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "collision_primitive" in state:
            del state["collision_primitive"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self.collision_primitive = self.create_collision_primitive()
        except Exception as e:
            logger.error(
                f"Failed to recreate collision primitive during deserialization: {e}"
            )
            raise

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def copy(self: T) -> T:
        """Creates a deep copy of the shape instance.

        Returns:
            A new instance of the same type with identical parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def create_collision_primitive(self) -> fcl.CollisionGeometry:
        """Creates the FCL collision geometry for the shape.

        This method must be implemented by subclasses to return the appropriate
        `fcl.CollisionGeometry` instance representing the shape.

        Returns:
            An `fcl.CollisionGeometry` object.
        """
        raise NotImplementedError

    @classmethod
    def from_other(cls: Type[T], other: T, transform: Transform | None = None) -> T:
        """Creates a new instance from another instance.

        This method should perform a deep copy of the shape's defining
        parameters but not its internal state.

        Args:
            other: The instance to copy from.

        Returns:
            A new instance of the class with the same parameters.
        """
        new_obj = other.copy()
        if transform is not None:
            new_obj.default_transform = transform.apply(other.default_transform)
        return new_obj

    def transformed_visual_mesh(
        self, transform: Transform | None = None
    ) -> trimesh.Trimesh:
        """Applies a transform to the visual mesh.

        The final transformation is the composition of the provided transform
        and the shape's default_transform.

        Args:
            transform: An additional transform to apply. Defaults to identity.

        Returns:
            A new trimesh.Trimesh object with the transformation applied.
        """
        # Compose the input transform with the default internal transform
        combined_transform = (
            transform.apply(self.default_transform)
            if transform
            else self.default_transform
        )
        # Apply the combined transform to a copy of the visual mesh
        return combined_transform.apply(self.visual_mesh)

    def transformed_collision_object(
        self, transform: Transform | None = None
    ) -> fcl.CollisionObject:
        """Creates a transformed FCL collision object.

        The final transformation is the composition of the provided transform
        and the shape's default_transform.

        Args:
            transform: An additional transform to apply. Defaults to identity.

        Returns:
            An fcl.CollisionObject with the transformation applied.
        """
        # Compose the input transform with the default internal transform
        combined_transform = (
            transform.apply(self.default_transform)
            if transform
            else self.default_transform
        )
        # Create an FCL CollisionObject with the geometry and the combined transform
        return combined_transform.apply(self.collision_primitive)

    @staticmethod
    def _apply_color_to_mesh(mesh: trimesh.Trimesh, color: RGBA0to1) -> trimesh.Trimesh:
        """Applies a uniform color to a trimesh object.

        Args:
            mesh: The mesh to color.
            color: The RGBA color, with each channel in the range [0, 1].

        Returns:
            A new trimesh object with vertex colors.

        Raises:
            ValueError: If color values are not in the valid range [0, 1].
        """
        if not isinstance(color, np.ndarray) or color.shape != (4,):
            raise ValueError(
                f"Color must be a 4-element numpy array, got shape {getattr(color, 'shape', 'unknown')}"
            )

        if not np.all((color >= 0) & (color <= 1)):
            raise ValueError(f"Color values must be in range [0, 1], got {color}")

        try:
            # Convert [0, 1] RGBA to [0, 255] and apply to all vertices
            vertex_num = len(mesh.vertices)
            color = color.copy()
            colors_255 = np.tile(color, (vertex_num, 1)) * 255
            colors_255 = np.clip(colors_255, 0, 255).astype(np.uint8)
            new_mesh = trimesh.Trimesh(
                mesh.vertices, mesh.faces, vertex_colors=colors_255
            )
            return new_mesh
        except Exception as e:
            logger.error(f"Failed to apply color to mesh: {e}")
            raise
