import trimesh
import fcl
import numpy as np
import logging

from linkmotion.robot.shape.base import ShapeBase
from linkmotion.transform import Transform
from linkmotion.typing.numpy import RGBA0to1

logger = logging.getLogger(__name__)


class ConvexShape(ShapeBase):
    """A convex shape defined by a triangular mesh.

    This class represents a convex hull shape using FCL's Convex geometry
    for collision detection and a trimesh object for visualization.

    Attributes:
        collision_mesh (trimesh.Trimesh): The mesh used for collision detection.
        collision_primitive (fcl.Convex): The FCL convex geometry.
        visual_mesh (trimesh.Trimesh): The mesh used for visualization.
        default_transform (Transform): The default pose of the shape.
    """

    def __init__(
        self,
        collision_mesh: trimesh.Trimesh,
        visual_mesh: trimesh.Trimesh | None = None,
        default_transform: Transform | None = None,
        color: RGBA0to1 | None = None,
    ):
        """Initializes the ConvexShape.

        Args:
            collision_mesh: The mesh to be used for collision checking.
            visual_mesh: The mesh to be used for visualization. If None, a
                copy of collision_mesh is used.
            default_transform: The default pose of the shape. If None, an
                identity transform is used.
            color: The RGBA color of the shape, with values in [0, 1].

        Raises:
            ValueError: If collision_mesh is invalid or empty.
        """
        if not isinstance(collision_mesh, trimesh.Trimesh):
            raise ValueError(
                f"collision_mesh must be a trimesh.Trimesh, got {type(collision_mesh)}"
            )

        if len(collision_mesh.vertices) == 0 or len(collision_mesh.faces) == 0:
            raise ValueError("collision_mesh must have vertices and faces")

        # Set default transform if not provided.
        if default_transform is None:
            default_transform = Transform()

        self.collision_mesh = collision_mesh.copy()

        try:
            convex = self.create_collision_primitive()

            # If no visual mesh is specified, use a copy of the collision mesh for visualization.
            # This ensures that modifications to the visual mesh do not affect the collision geometry.
            final_visual_mesh = (
                collision_mesh.copy() if visual_mesh is None else visual_mesh
            )

            super().__init__(convex, final_visual_mesh, default_transform, color)
            # Note: self.collision_primitive is set in super().__init__
            # Re-assigning it here is redundant. The type hint is for clarity.
            self.collision_primitive: fcl.Convex
        except Exception as e:
            logger.error(f"Failed to create ConvexShape: {e}")
            raise

    def create_collision_primitive(self) -> fcl.Convex:
        """Creates the FCL collision geometry for the convex shape.

        Returns:
            An fcl.Convex object.
        """
        array_of_3 = 3 * np.ones((len(self.collision_mesh.faces), 1), dtype=np.int64)
        fcl_faces = np.concatenate(
            (array_of_3, self.collision_mesh.faces), axis=1
        ).flatten()
        return fcl.Convex(
            self.collision_mesh.vertices, len(self.collision_mesh.faces), fcl_faces
        )

    def __repr__(self) -> str:
        return (
            f"ConvexShape(vertices={len(self.collision_mesh.vertices)}, "
            f"faces={len(self.collision_mesh.faces)})"
        )

    def copy(self) -> "ConvexShape":
        """Creates a copy of the ConvexShape instance.

        Returns:
            A new ConvexShape instance with the same parameters.
        """
        return ConvexShape(
            self.collision_mesh,
            visual_mesh=self.visual_mesh,
            default_transform=self.default_transform,
            color=self.color,
        )
