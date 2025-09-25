import fcl
import trimesh

from linkmotion.transform import Transform
from linkmotion.typing.numpy import RGBA0to1
from .base import ShapeBase


class Cylinder(ShapeBase):
    """A cylinder shape.

    The cylinder is centered at the origin in its local coordinate frame,
    with its height aligned with the Z-axis.

    Attributes:
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
    """

    def __init__(
        self,
        radius: float,
        height: float,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ):
        """Initializes the Cylinder.

        Args:
            radius: The radius of the cylinder.
            height: The height of the cylinder.
            default_transform: The default pose of the cylinder.
            color: The RGBA color of the cylinder, with values in [0, 1].
        """
        self.radius = float(radius)
        self.height = float(height)

        visual_mesh = trimesh.creation.cylinder(radius=self.radius, height=self.height)

        collision_primitive = self.create_collision_primitive()
        super().__init__(
            collision_primitive,
            visual_mesh,
            default_transform=default_transform,
            color=color,
        )

    def create_collision_primitive(self) -> fcl.Cylinder:
        """Creates the FCL collision geometry for the cylinder.

        Returns:
            An fcl.Cylinder object.
        """
        return fcl.Cylinder(self.radius, self.height)

    def __repr__(self) -> str:
        return f"Cylinder(radius={self.radius}, height={self.height})"

    def copy(self):
        return Cylinder(
            radius=self.radius,
            height=self.height,
            default_transform=self.default_transform,
            color=self.color,
        )
