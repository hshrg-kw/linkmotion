import logging
from typing import TypeVar

import trimesh
import fcl

from linkmotion.transform import Transform
from linkmotion.robot.shape.base import ShapeBase
from linkmotion.robot.shape.mesh import MeshShape
from linkmotion.typing.numpy import RGBA0to1, Vector3
from linkmotion.robot.shape.box import Box
from linkmotion.robot.shape.sphere import Sphere
from linkmotion.robot.shape.cylinder import Cylinder
from linkmotion.robot.shape.cone import Cone
from linkmotion.robot.shape.capsule import Capsule

T = TypeVar("T", bound="Link")
logger = logging.getLogger(__name__)


class Link:
    """Represents a physical link in a robot model."""

    def __init__(self, name: str, shape: ShapeBase):
        """Initializes a Link.

        Args:
            name: The name of the link.
            shape: The geometric shape of the link.
        """
        self.name = name
        self.shape = shape
        logger.info(f"Link '{self.name}' created.")

    def __str__(self) -> str:
        """Returns a string representation of the link."""
        return f"Link(name='{self.name}', shape={self.shape})"

    def collision_object(
        self, transform: Transform | None = None
    ) -> fcl.CollisionObject:
        """Creates a transformed FCL collision object for this link.

        Args:
            transform: The transform to apply to the link's shape.
                       Defaults to an identity transform.

        Returns:
            A flexible collision library (FCL) collision object.
        """
        return self.shape.transformed_collision_object(transform)

    def visual_mesh(self, transform: Transform | None = None) -> trimesh.Trimesh:
        """Creates a transformed visual mesh for this link.

        Args:
            transform: The transform to apply to the link's shape.
                       Defaults to an identity transform.

        Returns:
            A transformed trimesh object for visualization.
        """
        return self.shape.transformed_visual_mesh(transform)

    @classmethod
    def from_mesh(cls: type[T], name: str, mesh: trimesh.Trimesh) -> T:
        """Creates a Link from a trimesh object."""
        return cls(name, MeshShape(mesh))

    @classmethod
    def from_box(
        cls: type[T],
        name: str,
        extents: Vector3,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ) -> T:
        """
        Creates a Link with a box shape.
        The box is centered at the origin in its local coordinate frame.
        """
        return cls(name, Box(extents, default_transform, color))

    @classmethod
    def from_sphere(
        cls: type[T],
        name: str,
        radius: float,
        center: Vector3 | None = None,
        color: RGBA0to1 | None = None,
    ) -> T:
        """
        Creates a Link with a sphere shape.
        The sphere is centered at the origin in its local coordinate frame.
        """
        return cls(name, Sphere(radius, center, color))

    @classmethod
    def from_cylinder(
        cls: type[T],
        name: str,
        radius: float,
        height: float,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ) -> T:
        """
        Creates a Link with a cylinder shape.
        The cylinder is centered at the origin in its local coordinate frame,
        with its height aligned with the Z-axis.
        """
        return cls(name, Cylinder(radius, height, default_transform, color))

    @classmethod
    def from_cone(
        cls: type[T],
        name: str,
        radius: float,
        height: float,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ) -> T:
        """
        Creates a Link with a cone shape.
        The cone is centered at the origin in its local coordinate frame,
        with its height aligned with the Z-axis. The base is at z = -height/2
        and the apex is at z = +height/2.
        """
        return cls(name, Cone(radius, height, default_transform, color))

    @classmethod
    def from_capsule(
        cls: type[T],
        name: str,
        radius: float,
        height: float,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ) -> T:
        """
        Creates a Link with a capsule shape.
        A capsule is a cylinder with hemispherical caps at both ends.
        It is centered at the origin in its local coordinate frame, with its
        height aligned with the Z-axis.
        """
        return cls(name, Capsule(radius, height, default_transform, color))

    @classmethod
    def from_other(
        cls: type[T], other: T, name: str, transform: Transform | None = None
    ) -> T:
        """Creates a new Link by copying another, but with a new name.

        This performs a copy of the underlying shape object.
        """
        # Create a new link with a copied shape to avoid shared mutable state.
        copied_shape = other.shape.from_other(other.shape, transform=transform)
        return cls(name, copied_shape)
