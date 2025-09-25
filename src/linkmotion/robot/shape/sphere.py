import fcl
import trimesh
import logging

from linkmotion.transform import Transform
from linkmotion.typing.numpy import RGBA0to1, Vector3
from .base import ShapeBase

logger = logging.getLogger(__name__)


class Sphere(ShapeBase):
    """A sphere shape.

    The sphere is centered at the origin in its local coordinate frame.
    The 'center' parameter in the constructor defines its default position.

    Attributes:
        radius (float): The radius of the sphere.
        center (Vector3): The center position of the sphere, which defines its
            default_transform.
    """

    def __init__(
        self,
        radius: float,
        center: Vector3 | None = None,
        color: RGBA0to1 | None = None,
    ):
        """Initializes the Sphere.

        Args:
            radius: The radius of the sphere.
            center: The (x, y, z) position of the sphere's center. This sets
                the default translation. If None, it defaults to the origin.
            color: The RGBA color of the sphere, with values in [0, 1].

        Raises:
            ValueError: If radius is not positive.
        """
        if radius <= 0:
            raise ValueError(f"Sphere radius must be positive, got {radius}")

        self.radius = float(radius)

        try:
            # Create a mesh centered at the origin.
            visual_mesh = trimesh.creation.icosphere(radius=self.radius)

            collision_primitive = self.create_collision_primitive()

            default_transform = Transform(translate=center)
            super().__init__(
                collision_primitive,
                visual_mesh,
                default_transform=default_transform,
                color=color,
            )
        except Exception as e:
            logger.error(f"Failed to create Sphere with radius {self.radius}: {e}")
            raise

    @property
    def center(self) -> Vector3:
        """The center position of the sphere."""
        return self.default_transform.position

    def create_collision_primitive(self) -> fcl.Sphere:
        """Creates the FCL collision geometry for the sphere.

        Returns:
            An fcl.Sphere object.
        """
        return fcl.Sphere(self.radius)

    def copy(self):
        return Sphere(
            radius=self.radius,
            center=self.center,
            color=self.color,
        )

    def __repr__(self) -> str:
        return f"Sphere(radius={self.radius}, center={self.center})"
