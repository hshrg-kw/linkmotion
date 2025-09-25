import trimesh
import fcl
import logging

from linkmotion.robot.shape.base import ShapeBase
from linkmotion.transform import Transform
from linkmotion.typing.numpy import RGBA0to1

logger = logging.getLogger(__name__)


class MeshShape(ShapeBase):
    """Represents a shape defined by a 3D mesh.

    This class encapsulates a shape using one or two `trimesh.Trimesh` objects:
    one for collision detection and an optional, separate one for visualization.
    If no visual mesh is provided, the collision mesh is used for both.

    Attributes:
        collision_primitive (FCLBVH): The collision geometry, which is a
            Bounding Volume Hierarchy (BVH) model based on the collision mesh.
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
        """Initializes the MeshShape.

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
            # Create a Bounding Volume Hierarchy model for efficient collision checking.
            bvh = self.create_collision_primitive()

            # If no visual mesh is specified, use a copy of the collision mesh for visualization.
            # This ensures that modifications to the visual mesh do not affect the collision geometry.
            final_visual_mesh = (
                collision_mesh.copy() if visual_mesh is None else visual_mesh
            )

            super().__init__(bvh, final_visual_mesh, default_transform, color)
            # Note: self.collision_primitive is set in super().__init__
            # Re-assigning it here is redundant. The type hint is for clarity.
            self.collision_primitive: fcl.BVHModel
        except Exception as e:
            logger.error(f"Failed to create MeshShape: {e}")
            raise

    def copy(self) -> "MeshShape":
        """Creates a copy of the MeshShape instance.

        Returns:
            A new MeshShape instance with the same parameters.
        """
        return MeshShape(
            collision_mesh=self.collision_mesh,
            visual_mesh=self.visual_mesh,
            default_transform=self.default_transform,
            color=self.color,
        )

    def __repr__(self) -> str:
        return (
            f"MeshShape(vertices={len(self.collision_mesh.vertices)}, "
            f"faces={len(self.collision_mesh.faces)})"
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.collision_primitive = self.create_collision_primitive()

    def create_collision_primitive(self) -> fcl.BVHModel:
        """Creates the FCL collision geometry for the mesh shape.

        Returns:
            An FCLBVH object.
        """
        bvh = fcl.BVHModel()
        bvh.beginModel(
            len(self.collision_mesh.vertices), len(self.collision_mesh.faces)
        )
        bvh.addSubModel(self.collision_mesh.vertices, self.collision_mesh.faces)
        bvh.endModel()
        return bvh
