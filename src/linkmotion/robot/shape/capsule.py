import fcl
import trimesh

from linkmotion.transform import Transform
from linkmotion.typing.numpy import RGBA0to1
from .base import ShapeBase


class Capsule(ShapeBase):
    """A capsule shape.

    A capsule is a cylinder with hemispherical caps at both ends.
    It is centered at the origin in its local coordinate frame, with its
    height aligned with the Z-axis.

    Attributes:
        radius (float): The radius of the capsule.
        height (float): The height of the cylindrical part of the capsule.
    """

    def __init__(
        self,
        radius: float,
        height: float,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ):
        """Initializes the Capsule.

        Args:
            radius: The radius of the cylinder and hemispheres.
            height: The height of the cylindrical section.
            default_transform: The default pose of the capsule.
            color: The RGBA color of the capsule, with values in [0, 1].
        """
        self.radius = float(radius)
        self.height = float(height)

        # Create a mesh centered at the origin. Both trimesh and FCL capsules
        # are centered at the origin by default.
        visual_mesh = trimesh.creation.capsule(height=self.height, radius=self.radius)

        collision_primitive = self.create_collision_primitive()
        super().__init__(
            collision_primitive,
            visual_mesh,
            default_transform=default_transform,
            color=color,
        )

    def create_collision_primitive(self) -> fcl.Capsule:
        """Creates the FCL collision geometry for the capsule.

        Returns:
            An fcl.Capsule object.
        """
        return fcl.Capsule(self.radius, self.height)

    def copy(self):
        """Creates a copy of the Capsule instance.

        Returns:
            A new Capsule instance with the same parameters.
        """
        return Capsule(
            radius=self.radius,
            height=self.height,
            default_transform=self.default_transform,
            color=self.color,
        )

    def __repr__(self) -> str:
        return (
            f"Capsule(radius={self.radius}, height={self.height}, "
            f"default_transform={self.default_transform}, color={self.color})"
        )
