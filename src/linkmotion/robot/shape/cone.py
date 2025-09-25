import fcl
import trimesh

from linkmotion.transform import Transform
from linkmotion.typing.numpy import RGBA0to1
from .base import ShapeBase


class Cone(ShapeBase):
    """A cone shape.

    The cone is centered at the origin in its local coordinate frame,
    with its height aligned with the Z-axis. The base is at z = -height/2
    and the apex is at z = +height/2.

    Attributes:
        radius (float): The radius of the cone's base.
        height (float): The height of the cone.
    """

    def __init__(
        self,
        radius: float,
        height: float,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ):
        """Initializes the Cone.

        Args:
            radius: The radius of the cone's base.
            height: The height of the cone.
            default_transform: The default pose of the cone.
            color: The RGBA color of the cone, with values in [0, 1].
        """
        self.radius = float(radius)
        self.height = float(height)

        visual_mesh = trimesh.creation.cone(radius=self.radius, height=self.height)

        # Re-center the visual mesh to match the FCL primitive's origin.
        # trimesh.creation.cone places the base at z=0, so we shift it
        # down by half the height to center it at the origin.
        visual_mesh.apply_translation([0, 0, -self.height / 2.0])

        collision_primitive = self.create_collision_primitive()
        super().__init__(
            collision_primitive,
            visual_mesh,
            default_transform=default_transform,
            color=color,
        )

    def create_collision_primitive(self) -> fcl.Cone:
        """Creates the FCL collision geometry for the cone.

        Returns:
            An fcl.Cone object.
        """
        return fcl.Cone(self.radius, self.height)

    def __repr__(self) -> str:
        return f"Cone(radius={self.radius}, height={self.height})"

    def copy(self):
        """Creates a copy of the Cone instance.

        Returns:
            A new Cone instance with the same parameters.
        """
        return Cone(
            self.radius,
            self.height,
            default_transform=self.default_transform,
            color=self.color,
        )
